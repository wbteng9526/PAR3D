import os, math, glob, time, argparse, random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torch.nn.functional as F

import sys
# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dataset.mvdataset import MultiViewCamDataset

# ============ wandb ============
try:
    import wandb
except Exception:
    wandb = None

# ----------------------------
# Utils
# ----------------------------
def is_main():
    return int(os.environ.get("RANK", "0")) == 0

def setup_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def setup_ddp():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        return True
    return False

def cleanup_ddp():
    if dist.is_initialized(): dist.destroy_process_group()

def human_bytes(x):
    for unit in ['B','KB','MB','GB','TB']:
        if abs(x) < 1024.0: return f"{x:3.1f}{unit}"
        x /= 1024.0
    return f"{x:.1f}PB"

# ----------------------------
# Geometry helpers for Plücker
# ----------------------------
def plucker_canonicalize(x):
    # x: [B,6,T,H,W] -> normalize d, orth m, fix sign
    d, m = x[:, :3], x[:, 3:]
    d = d / (d.norm(dim=1, keepdim=True) + 1e-8)
    m = m - (d * (d * m).sum(dim=1, keepdim=True))
    sign = torch.where(d[:,2:3] >= 0, torch.ones_like(d[:,2:3]), -torch.ones_like(d[:,2:3]))
    d = d * sign; m = m * sign
    return torch.cat([d, m], dim=1)

def loss_geometry(pred, gt, w_rec=1.0, w_m=1.0, w_unit=0.1, w_orth=0.1, w_origin=0.0):
    d_p, m_p = pred[:, :3], pred[:, 3:]
    d_g, m_g = gt[:, :3], gt[:, 3:]

    rec1 = (d_p - d_g).pow(2).mean() + w_m*(m_p - m_g).pow(2).mean()
    rec2 = (d_p + d_g).pow(2).mean() + w_m*(m_p + m_g).pow(2).mean()
    L_rec = torch.minimum(rec1, rec2)

    L_unit = (d_p.norm(dim=1) - 1.0).pow(2).mean()
    L_orth = (d_p*m_p).sum(dim=1).pow(2).mean()
    L = w_rec*L_rec + w_unit*L_unit + w_orth*L_orth

    if w_origin > 0:
        B, _, T, H, W = d_p.shape
        I = torch.eye(3, device=d_p.device, dtype=d_p.dtype).view(1,3,3,1,1)
        dxm = torch.stack([
            d_p[:,1]*m_p[:,2] - d_p[:,2]*m_p[:,1],
            d_p[:,2]*m_p[:,0] - d_p[:,0]*m_p[:,2],
            d_p[:,0]*m_p[:,1] - d_p[:,1]*m_p[:,0],
        ], dim=1)  # [B,3,T,H,W]
        ddT = d_p.unsqueeze(1)*d_p.unsqueeze(2)  # [B,3,3,T,H,W]
        A = (I - ddT).reshape(B,3,3,T,H*W).sum(dim=-1)  # [B,3,3,T]
        b = dxm.reshape(B,3,T,H*W).sum(dim=-1)          # [B,3,T]
        o = torch.linalg.solve(A.permute(0,3,1,2), b.permute(0,2,1))  # [B,T,3]
        o_full = o.permute(0,2,1).unsqueeze(-1).unsqueeze(-1)         # [B,3,T,1,1]
        r = (I - ddT) @ o_full - dxm.unsqueeze(1)                     # [B,3,T,H,W]
        L_origin = r.pow(2).mean()
        L = L + w_origin*L_origin
    return {"L_total": L, "L_rec": L_rec, "L_unit": L_unit, "L_orth": L_orth}



# ----------------------------
# Model blocks
# ----------------------------
class SepResBlock3D(nn.Module):
    def __init__(self, c, kt=3, ks=3, gc=32, t_dilation=1):
        super().__init__()
        g = max(1, c//gc)
        self.n1 = nn.GroupNorm(g, c)
        self.t = nn.Conv3d(c, c, kernel_size=(kt,1,1),
                           padding=(t_dilation*(kt//2),0,0), dilation=(t_dilation,1,1))
        self.n2 = nn.GroupNorm(g, c)
        self.s = nn.Conv3d(c, c, kernel_size=(1,ks,ks), padding=(0,ks//2,ks//2))
    def forward(self, x):
        h = self.t(F.silu(self.n1(x)))
        h = self.s(F.silu(self.n2(h)))
        return x + h

class ResBlock3D(nn.Module):
    def __init__(self, c, gc=32):
        super().__init__()
        g = max(1, c//gc)
        self.n1 = nn.GroupNorm(g, c); self.c1 = nn.Conv3d(c, c, 3, padding=1)
        self.n2 = nn.GroupNorm(g, c); self.c2 = nn.Conv3d(c, c, 3, padding=1)
    def forward(self, x):
        h = self.c1(F.silu(self.n1(x))); h = self.c2(F.silu(self.n2(h))); return x + h

class DownBlock3D(nn.Module):
    def __init__(self, cin, cout, stride, res_type="vanilla"):
        super().__init__()
        self.conv = nn.Conv3d(cin, cout, 3, stride=stride, padding=1)
        self.res  = (SepResBlock3D(cout) if res_type=="sep" else ResBlock3D(cout))
    def forward(self, x):
        return self.res(self.conv(x))

class UpBlock3D(nn.Module):
    def __init__(self, cin, cout, scale, res_type="vanilla"):
        super().__init__()
        self.up = nn.Upsample(scale_factor=scale, mode='trilinear', align_corners=False)
        self.conv = nn.Conv3d(cin, cout, 3, padding=1)
        self.res  = (SepResBlock3D(cout) if res_type=="sep" else ResBlock3D(cout))
    def forward(self, x):
        return self.res(self.conv(self.up(x)))

# ----------------------------
# Encoders / Decoders
# ----------------------------
class PluckerEncoder3D(nn.Module):
    """ In: [B,6,16,256,256] -> Out: [B,z,4,32,32] """
    def __init__(self, z=1280, base=128, res_type="vanilla"):
        super().__init__()
        self.pre = nn.Conv3d(6, base, 3, padding=1)
        self.db1 = DownBlock3D(base,   base*2, stride=(2,2,2), res_type=res_type)  # 16->8, 256->128
        self.db2 = DownBlock3D(base*2, base*4, stride=(2,2,2), res_type=res_type)  # 8->4,  128->64
        self.db3 = DownBlock3D(base*4, base*4, stride=(1,2,2), res_type=res_type)  # 4->4,  64->32
        ch = base*4
        g = max(1, ch//32)
        self.post = nn.Sequential(
            (SepResBlock3D(ch) if res_type=="sep" else ResBlock3D(ch)),
            nn.GroupNorm(g, ch), nn.SiLU(),
            nn.Conv3d(ch, z, 1)
        )
    def forward(self, x):
        # x = plucker_canonicalize(x)
        x = self.pre(x); x = self.db1(x); x = self.db2(x); x = self.db3(x); x = self.post(x)
        return x

class PluckerVAEEncoder3D(nn.Module):
    """ outputs mu, logvar at token resolution """
    def __init__(self, z=1280, base=128, res_type="vanilla"):
        super().__init__()
        self.feat = PluckerEncoder3D(z=base*4, base=base, res_type=res_type)
        ch = base*4
        self.head = nn.Conv3d(ch, 2*z, 1)
    def forward(self, x):
        # h = self.feat.pre(plucker_canonicalize(x))
        h = self.feat.pre(x)
        h = self.feat.db1(h); h = self.feat.db2(h); h = self.feat.db3(h)
        h = self.feat.post[:-1](h)
        stats = self.head(h)
        mu, logvar = stats.chunk(2, dim=1)
        return mu, logvar

class PluckerDecoder3D(nn.Module):
    """ In: [B,z,4,32,32] -> Out: [B,6,16,256,256] """
    def __init__(self, z=1280, base=128, res_type="vanilla"):
        super().__init__()
        ch = base*4
        self.pre = nn.Conv3d(z, ch, 1)
        self.up1 = UpBlock3D(ch,      ch,     scale=(1,2,2), res_type=res_type)  # 4->4,   32->64
        self.up2 = UpBlock3D(ch,      base*2, scale=(2,2,2), res_type=res_type)  # 4->8,   64->128
        self.up3 = UpBlock3D(base*2,  base,   scale=(2,2,2), res_type=res_type)  # 8->16, 128->256
        self.head = nn.Conv3d(base, 6, 3, padding=1)

    def forward(self, z):
        x = self.pre(z); x = self.up1(x); x = self.up2(x); x = self.up3(x)
        x = self.head(x)
        d, m = x[:, :3], x[:, 3:]
        d = d / (d.norm(dim=1, keepdim=True) + 1e-8)
        m = m - (d * (d*m).sum(dim=1, keepdim=True))
        sign = torch.where(d[:,2:3] >= 0, torch.ones_like(d[:,2:3]), -torch.ones_like(d[:,2:3]))
        d = d * sign; m = m * sign
        return torch.cat([d, m], dim=1)

# ----------------------------
# Full model wrapper
# ----------------------------
class PluckerTokenModel(nn.Module):
    def __init__(self, z=1280, model_type="ae", base=128, res_type="vanilla", beta=0.1):
        super().__init__()
        self.model_type = model_type
        self.beta = beta
        if model_type == "ae":
            self.encoder = PluckerEncoder3D(z=z, base=base, res_type=res_type)
        else:
            self.encoder = PluckerVAEEncoder3D(z=z, base=base, res_type=res_type)
        self.decoder = PluckerDecoder3D(z=z, base=base, res_type=res_type)

    def forward(self, x, amp_dtype=None):
        if self.model_type == "ae":
            tokens = self.encoder(x)
            recon  = self.decoder(tokens)
            return {"tokens": tokens, "recon": recon, "kl": torch.tensor(0.0, device=x.device, dtype=x.dtype)}
        else:
            mu, logvar = self.encoder(x)
            eps = torch.randn_like(mu, dtype=(amp_dtype if amp_dtype is not None else mu.dtype))
            z = mu + torch.exp(0.5*logvar) * eps
            recon = self.decoder(z)
            kl = 0.5 * (torch.exp(logvar) + mu.pow(2) - 1.0 - logvar)
            kl = kl.mean()
            return {"tokens": z, "recon": recon, "kl": kl}

# ----------------------------
# Scheduler
# ----------------------------
class CosineWithWarmup:
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr_ratio=0.1):
        self.opt = optimizer
        self.warmup = max(1, warmup_steps)
        self.total = max(self.warmup+1, total_steps)
        self.min_ratio = min_lr_ratio
        self.step_num = 0
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]

    def step(self):
        self.step_num += 1
        for i, g in enumerate(self.opt.param_groups):
            if self.step_num <= self.warmup:
                lr = self.base_lrs[i] * self.step_num / self.warmup
            else:
                t = (self.step_num - self.warmup) / (self.total - self.warmup)
                cos = 0.5*(1+math.cos(math.pi*t))
                lr = self.base_lrs[i] * (self.min_ratio + (1-self.min_ratio)*cos)
            g["lr"] = lr

# ----------------------------
# Train / Val
# ----------------------------
def train_one_epoch(args, epoch, model, optimizer, scheduler, scaler, train_loader, device, amp_dtype):
    model.train()
    global_step = epoch * len(train_loader)

    for it, x in enumerate(train_loader):
        x = x.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(dtype=amp_dtype, enabled=(args.precision!="fp32")):
            out = model(x, amp_dtype=amp_dtype)
            x_gt = x #plucker_canonicalize(x)
            loss_dict = loss_geometry(out["recon"], x_gt,
                                      w_rec=1.0, w_m=1.0,
                                      w_unit=args.w_unit, w_orth=args.w_orth, w_origin=args.w_origin)
            loss = loss_dict["L_total"]
            if args.model_type == "vae":
                beta = args.beta * min(1.0, (global_step+1)/max(1, args.kl_warmup))
                loss = loss + beta * out["kl"]

        optimizer.zero_grad(set_to_none=True)
        if args.precision == "fp16":
            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer); scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

        scheduler.step()
        global_step += 1

        # wandb logging
        if is_main() and (global_step % args.log_every == 0):
            lr = scheduler.opt.param_groups[0]["lr"]
            mem = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
            log_data = {
                "train/loss_total": loss_dict["L_total"].item(),
                "train/loss_rec":   loss_dict["L_rec"].item(),
                "train/loss_unit":  loss_dict["L_unit"].item(),
                "train/loss_orth":  loss_dict["L_orth"].item(),
                "train/lr": lr,
                "train/max_mem_bytes": mem
            }
            if args.model_type == "vae":
                log_data["train/kl"] = out["kl"].item()
            wandb.log(log_data, step=global_step)
            print(f"[epoch {epoch} iter {it}] step {global_step} lr {lr:.3e} "
                  f"loss {loss.item():.4f} (mem {human_bytes(mem)})")

        # ckpt save
        if is_main() and (args.ckpt_every>0) and (global_step % args.ckpt_every == 0):
            save_ckpt(args, model, optimizer, scheduler, scaler, global_step, tag=f"step{global_step}")

        # mid-epoch val
        if (args.val_every>0) and (global_step % args.val_every == 0):
            if dist.is_initialized(): dist.barrier()
            val_loss = validate(args, model, args.val_loader, device, amp_dtype, global_step)

def validate(args, model, val_loader, device, amp_dtype, global_step):
    model.eval()
    losses = []
    with torch.no_grad():
        for x in val_loader:
            x = x.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(dtype=amp_dtype, enabled=(args.precision!="fp32")):
                out = model(x, amp_dtype=amp_dtype)
                x_gt = plucker_canonicalize(x)
                loss_dict = loss_geometry(out["recon"], x_gt,
                                          w_rec=1.0, w_m=1.0,
                                          w_unit=args.w_unit, w_orth=args.w_orth, w_origin=args.w_origin)
                loss = loss_dict["L_total"]
                if args.model_type == "vae":
                    loss = loss + args.beta * out["kl"]
            losses.append(loss.item())
    val_loss = float(np.mean(losses))
    if is_main():
        wandb.log({"val/loss_total": val_loss}, step=global_step)
        print(f"[VAL] step {global_step} loss_total={val_loss:.4f}")
        # best model
        best = getattr(args, "_best_val", float("inf"))
        if val_loss < best:
            args._best_val = val_loss
            save_ckpt(args, model, args.optimizer, args.scheduler, args.scaler, global_step, tag="best")
    model.train()
    return val_loss

def save_ckpt(args, model, optimizer, scheduler, scaler, step, tag="latest"):
    ckpt = {
        "step": step,
        "model": (model.module.state_dict() if isinstance(model, DDP) else model.state_dict()),
        "opt": optimizer.state_dict(),
        "scaler": (scaler.state_dict() if scaler is not None else None),
        "sched": {"step_num": scheduler.step_num,
                  "base_lrs": scheduler.base_lrs},
        "args": vars(args)
    }
    out = Path(args.out_dir)/f"ckpt_{tag}.pt"
    torch.save(ckpt, out)
    print(f"[CKPT] saved -> {out}")
    if is_main() and args.wandb_log_artifact:
        art = wandb.Artifact(name=f"{args.run_name}-{tag}", type="checkpoint")
        art.add_file(str(out))
        wandb.log_artifact(art)

def load_ckpt(args, model, optimizer=None, scaler=None):
    if not args.resume: return 0
    mp = torch.load(args.resume, map_location="cpu")
    (model.module if isinstance(model, DDP) else model).load_state_dict(mp["model"], strict=True)
    if optimizer is not None and "opt" in mp:
        optimizer.load_state_dict(mp["opt"])
    if scaler is not None and mp.get("scaler") is not None:
        scaler.load_state_dict(mp["scaler"])
    step = mp.get("step", 0)
    if is_main(): print(f"[CKPT] loaded from {args.resume} at step {step}")
    return step

# ----------------------------
# Main
# ----------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, default="./data")
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--train_len", type=int, default=1000)
    p.add_argument("--val_len", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=4)

    p.add_argument("--z", type=int, default=1280)
    p.add_argument("--base", type=int, default=128)
    p.add_argument("--res_type", type=str, default="vanilla", choices=["vanilla","sep"])
    p.add_argument("--model_type", type=str, default="ae", choices=["ae","vae"])
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--kl_warmup", type=int, default=30000)

    p.add_argument("--precision", type=str, default="bf16", choices=["fp32","fp16","bf16"])
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--betas", type=float, nargs=2, default=(0.9,0.95))
    p.add_argument("--grad_clip", type=float, default=1.0)

    p.add_argument("--max_epochs", type=int, default=50)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--val_every", type=int, default=500)   # steps；0 关闭
    p.add_argument("--ckpt_every", type=int, default=2000)
    p.add_argument("--out_dir", type=str, default="./outputs")
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--seed", type=int, default=42)

    # geometry loss weights
    p.add_argument("--w_unit", type=float, default=0.1)
    p.add_argument("--w_orth", type=float, default=0.1)
    p.add_argument("--w_origin", type=float, default=0.0)

    # wandb
    p.add_argument("--wandb_project", type=str, default="plucker-tokens")
    p.add_argument("--run_name", type=str, default="")
    p.add_argument("--wandb_mode", type=str, default="online", choices=["online","offline","disabled"])
    p.add_argument("--wandb_log_artifact", action="store_true")

    args = p.parse_args()

    setup_seed(args.seed)
    use_ddp = setup_ddp()
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0))) if torch.cuda.is_available() else torch.device("cpu")
    cudnn.benchmark = True
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # wandb init（只在主进程）
    if is_main():
        if wandb is None:
            raise RuntimeError("wandb not installed. pip install wandb")
        if args.run_name == "":
            args.run_name = time.strftime("run-%Y%m%d-%H%M%S")
        if args.wandb_mode == "disabled":
            os.environ["WANDB_MODE"] = "disabled"
        elif args.wandb_mode == "offline":
            os.environ["WANDB_MODE"] = "offline"
        wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))
        print(args)
    else:
        # 禁止子进程写 wandb 日志，避免重复
        os.environ["WANDB_MODE"] = "disabled"

    # Dataset / Dataloader
    train_set = MultiViewCamDataset(args, True)
    val_set = MultiViewCamDataset(args, False)

    train_sampler = DistributedSampler(train_set, shuffle=True, drop_last=True) if use_ddp else None
    val_sampler   = DistributedSampler(val_set, shuffle=False, drop_last=False) if use_ddp else None

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=(train_sampler is None),
                              sampler=train_sampler, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_set, batch_size=1, shuffle=False, sampler=val_sampler,
                              num_workers=args.num_workers, pin_memory=True, drop_last=False)
    args.val_loader = val_loader  # for mid-epoch val

    # Model
    model = PluckerTokenModel(z=args.z, model_type=args.model_type, base=args.base, res_type=args.res_type, beta=args.beta).to(device)
    if use_ddp:
        model = DDP(model, device_ids=[device.index], output_device=device.index, find_unused_parameters=False)

    # Optimizer / Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=tuple(args.betas), weight_decay=args.weight_decay)
    total_steps = args.max_epochs * len(train_loader)
    warmup_steps = max(10, int(0.03 * total_steps))
    scheduler = CosineWithWarmup(optimizer, warmup_steps=warmup_steps, total_steps=total_steps, min_lr_ratio=0.1)

    # AMP precision
    amp_dtype = None
    if args.precision == "fp16": amp_dtype = torch.float16
    elif args.precision == "bf16": amp_dtype = torch.bfloat16
    scaler = torch.cuda.amp.GradScaler(enabled=(args.precision=="fp16"))

    # Resume
    start_step = 0
    if args.resume:
        start_step = load_ckpt(args, model, optimizer, scaler)

    # Attach references used by validate/save for best tracking in main proc
    args.optimizer = optimizer
    args.scheduler = scheduler
    args.scaler = scaler
    args._best_val = float("inf")

    # (可选) watch 模型参数/梯度（仅主进程，量大可关）
    if is_main():
        try:
            wandb.watch(model if not isinstance(model, DDP) else model.module, log="gradients", log_freq=args.log_every)
        except Exception:
            pass

    # Train
    for epoch in range(args.max_epochs):
        if use_ddp and isinstance(train_sampler, DistributedSampler):
            train_sampler.set_epoch(epoch)
        t0 = time.time()
        train_one_epoch(args, epoch, model, optimizer, scheduler, scaler, train_loader, device, amp_dtype)
        if dist.is_initialized(): dist.barrier()
        # epoch-end val
        val_loss = validate(args, model, val_loader, device, amp_dtype, global_step=(epoch+1)*len(train_loader))
        if is_main():
            wandb.log({"epoch/val_loss": val_loss}, step=(epoch+1)*len(train_loader))
            dt = time.time()-t0
            print(f"[epoch {epoch}] val {val_loss:.4f} | {dt/60:.1f} min")
            # 保存一个按 epoch 的 best
            if val_loss < args._best_val:
                args._best_val = val_loss
                save_ckpt(args, model, optimizer, scheduler, scaler, step=(epoch+1)*len(train_loader), tag="best_epoch")

    if is_main():
        save_ckpt(args, model, optimizer, scheduler, scaler, step=args.max_epochs*len(train_loader), tag="final")
        wandb.finish()
    cleanup_ddp()

if __name__ == "__main__":
    main()