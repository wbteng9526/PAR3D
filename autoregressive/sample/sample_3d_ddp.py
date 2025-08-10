import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)     # disable default parameter init for faster speed
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)  # disable default parameter init for faster speed
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from transformers import CLIPVisionModelWithProjection
from torchvision.io import write_video

import os
import numpy as np
import math
import json
import argparse
import pandas as pd
from tqdm import tqdm
from PIL import Image
from einops import rearrange

# metrics
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import lpips                              # pip install lpips
from torchmetrics.image.fid import FrechetInceptionDistance

from tokenizer.vidtok.scripts.inference_evaluate import load_model_from_config
from autoregressive.models.gpt import GPT_models
from autoregressive.models.generate_3d import generate
from dataset.mvdataset import RE10KVideoEvalDataset
os.environ["TOKENIZERS_PARALLELISM"] = "false"




def main(args):
    # Setup PyTorch:
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)

    # Setup DDP:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # create and load model
    tokenizer = load_model_from_config(args.tokenizer_config, args.tokenizer_ckpt)
    tokenizer.to(device)
    tokenizer.eval()
    print(f"video tokenizer is loaded")

    # metrics init
    lpips_loss_fn  = lpips.LPIPS(net='vgg').to(device).eval()
    fid = FrechetInceptionDistance(feature=2048).to(device)

    # load clip model
    clip_model = CLIPVisionModelWithProjection.from_pretrained(args.clip_ckpt).to(device)
    clip_model.eval()

    # create and load gpt model
    precision = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.precision]
    latent_size = args.image_size // args.downsample_size
    gpt_model = GPT_models[args.gpt_model](
        block_size=latent_size ** 2 * args.num_views,
        cls_token_num=args.cls_token_num,
        resid_dropout_p=args.dropout_p if args.drop_path_rate > 0.0 else 0.0,
        ffn_dropout_p=args.dropout_p if args.drop_path_rate > 0.0 else 0.0,
        drop_path_rate=args.drop_path_rate,
        token_dropout_p=args.token_dropout_p,
        spe_token_num=args.spe_token_num,
        ar_token_num=args.ar_token_num,
    ).to(device=device, dtype=precision)

    checkpoint = torch.load(args.gpt_ckpt, map_location="cpu")
 
    if "model" in checkpoint:  # ddp
        model_weight = checkpoint["model"]
    elif "module" in checkpoint: # deepspeed
        model_weight = checkpoint["module"]
    elif "state_dict" in checkpoint:
        model_weight = checkpoint["state_dict"]
    else:
        raise Exception("please check model weight")
    gpt_model.load_state_dict(model_weight, strict=False)
    gpt_model.eval()
    del checkpoint
    print(f"gpt model is loaded")

    if args.compile:
        print(f"compiling the model...")
        gpt_model = torch.compile(
            gpt_model,
            mode="reduce-overhead",
            fullgraph=True
        ) # requires PyTorch 2.0 (optional)
    else:
        print(f"no need to compile model in demo")

    # load dataset
    dataset = RE10KVideoEvalDataset(args)
    dataloader = DataLoader(
        dataset,
        batch_size=args.per_proc_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )    

    # Create folder to save samples:
    model_string_name = args.gpt_model.replace("/", "-")
    ckpt_string_name = os.path.basename(args.gpt_ckpt).replace(".pth", "").replace(".pt", "")
    folder_name = f"{model_string_name}-{ckpt_string_name}-size-{args.image_size}-size-{args.image_size}-{args.vq_model}-" \
                  f"topk-{args.top_k}-topp-{args.top_p}-temperature-{args.temperature}-" \
                  f"cfg-{args.cfg_scale}-seed-{args.global_seed}"
    sample_folder_dir = f"{args.sample_dir}/{folder_name}"
    if rank == 0:
        os.makedirs(f"{sample_folder_dir}/images", exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}/images")
    dist.barrier()

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    n = args.per_proc_batch_size
    global_batch_size = n * dist.get_world_size()
    num_fid_samples = len(dataset)
    # To make things evenly-divisible, we'll sample a bit more than we need and then discard the extra samples:
    total_samples = int(math.ceil(num_fid_samples / global_batch_size) * global_batch_size)
    if rank == 0:
        print(f"Total number of images that will be sampled: {total_samples}")
    assert total_samples % dist.get_world_size() == 0, "total_samples must be divisible by world_size"
    samples_needed_this_gpu = int(total_samples // dist.get_world_size())
    assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    iterations = int(samples_needed_this_gpu // n)
    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar

    for _ in pbar:

        batch = next(iter(dataloader))
        frame_tensors, plucker_coords, clip_input, scene_name = batch
        num_samples = frame_tensors.shape[0]
        cond_input = clip_model(clip_input).image_embeds

        index_sample = generate(
            gpt_model, cond_input, plucker_coords, latent_size ** 2, 
            cfg_scale=args.cfg_scale,
            temperature=args.temperature, top_k=args.top_k,
            top_p=args.top_p, sample_logits=True, 
        )
        
        samples = tokenizer.decode(index_sample, decode_from_indices=True) # output value is between [-1, 1]

        # reconstruction
        encoded_indices = tokenizer.encode(frame_tensors)[1]['indices']
        recons = tokenizer.decode(encoded_indices, decode_from_indices=True)

        for i in range(num_samples):
            gt = torch.clamp(rearrange(frame_tensors[i, :, 1:], "c t h w -> t h w c") * 0.5 + 0.5, 0, 1)
            recon = torch.clamp(rearrange(recons[i, :, 1:], "c t h w -> t h w c") * 0.5 + 0.5, 0, 1)
            sample = torch.clamp(rearrange(samples[i, :, 1:], "c t h w -> t h w c") * 0.5 + 0.5, 0, 1)
            combined = torch.cat([gt, recon, sample], dim=-1)
            combined = (combined * 255.0).cpu().to(torch.uint8)
            write_video(f"{sample_folder_dir}/{scene_name[i]}/result_video.mp4", combined, fps=args.fps)

            if args.compute_metrics:
                psnr_score = np.mean([peak_signal_noise_ratio(a, b) for a, b in zip(sample, gt)])
                ssim_score = np.mean([structural_similarity(a, b) for a, b in zip(sample, gt)])
                lpips_score = lpips_loss_fn(
                    rearrange(sample, "t h w c -> t c h w") * 2 - 1,
                    rearrange(gt, "t h w c -> t c h w") * 2 - 1,
                )
                fid.update((rearrange(gt, "t h w c -> t c h w") * 255).cpu().to(torch.uint8).to(device), real=True)
                fid.update((rearrange(sample, "t h w c -> t c h w") * 255).cpu().to(torch.uint8).to(device), real=False)

                fid_score = fid.compute()
                metrics = {
                    "psnr": psnr_score,
                    "ssim": ssim_score,
                    "lpips": lpips_score,
                    "fid": fid_score,
                }
                with open(f"{sample_folder_dir}/metrics.json", "a") as f:
                    json.dump(metrics, f, indent=4)



    # Make sure all processes have finished saving their samples before attempting to convert to .npz
    dist.barrier()
    dist.destroy_process_group()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer-config", type=str, default="configs/vidtok_fsq_causal_488_262144.yaml")
    parser.add_argument("--tokenizer-ckpt", type=str, default="/wekafs/ict/wenbinte/projects/RandAR3D/tokenizer/VidTok/vidtok_fsq_causal_488_262144.ckpt")
    parser.add_argument("--num-views", type=int, default=4)
    parser.add_argument("--prompt-csv", type=str, default='evaluations/t2i/PartiPrompts.tsv')
    parser.add_argument("--t5-path", type=str, default='pretrained_models/t5-ckpt')
    parser.add_argument("--t5-model-type", type=str, default='flan-t5-xl')
    parser.add_argument("--t5-feature-max-len", type=int, default=120)
    parser.add_argument("--t5-feature-dim", type=int, default=2048)
    parser.add_argument("--no-left-padding", action='store_true', default=False)
    parser.add_argument("--gpt-model", type=str, choices=list(GPT_models.keys()), default="GPT-XL")
    parser.add_argument("--gpt-ckpt", type=str, default=None)
    parser.add_argument("--gpt-type", type=str, choices=['c2i', 't2i'], default="t2i", help="class->image or text->image")  
    parser.add_argument("--cls-token-num", type=int, default=120, help="max token number of condition input")
    parser.add_argument("--spe-token-num", type=int, default=15, help="number of special tokens")
    parser.add_argument("--ar-token-num", type=int, default=16, help="number of autoregressive tokens")
    parser.add_argument("--dropout-p", type=float, default=0.0, help="dropout probability")
    parser.add_argument("--token-dropout-p", type=float, default=0.0, help="token dropout probability")
    parser.add_argument("--drop-path-rate", type=float, default=0.0, help="drop path rate")
    parser.add_argument("--precision", type=str, default='bf16', choices=["none", "fp16", "bf16"]) 
    parser.add_argument("--compile", action='store_true', default=False)
    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-16")
    parser.add_argument("--vq-ckpt", type=str, default=None, help="ckpt path for vq model")
    parser.add_argument("--codebook-size", type=int, default=16384, help="codebook size for vector quantization")
    parser.add_argument("--codebook-embed-dim", type=int, default=8, help="codebook dimension for vector quantization")
    parser.add_argument("--image-size", type=int, choices=[256, 384, 512], default=512)
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=7.5)
    parser.add_argument("--sample-dir", type=str, default="samples_parti", help="samples_coco or samples_parti")
    parser.add_argument("--per-proc-batch-size", type=int, default=32)
    parser.add_argument("--num-fid-samples", type=int, default=30000)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=1000, help="top-k value to sample with")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature value to sample with")
    parser.add_argument("--top-p", type=float, default=1.0, help="top-p value to sample with")
    parser.add_argument("--fps", type=int, default=8, help="fps of the video")
    parser.add_argument("--compute-metrics", action='store_true', default=False)
    args = parser.parse_args()
    main(args)
