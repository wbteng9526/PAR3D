#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
import os
from typing import List, Optional
import sys

import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.mvdataset import MultiViewDataset

# ---------------------------
# 配置
# ---------------------------

K_LIST_DEFAULT = [32, 32, 32]  # 32*32*32 = 32768


# ---------------------------
# 工具函数
# ---------------------------

def prod(xs: List[int]) -> int:
    out = 1
    for x in xs: out *= int(x)
    return out

@torch.no_grad()
def split_to_parts(y: torch.Tensor, K_list: List[int]) -> torch.Tensor:
    """
    把大 vocab id（标量或 [N] 向量）拆成多段（低位在前）。
    返回：
      - y 是标量 → [M]
      - y 是 [N] → [N, M]
    """
    assert y.dtype in (torch.long, torch.int64), "y must be LongTensor"
    scalar = (y.dim() == 0)
    if scalar: y = y.view(1)
    parts, cur = [], y.clone()
    for K in K_list:
        parts.append(cur % K)
        cur = torch.div(cur, K, rounding_mode="floor")
    parts = torch.stack(parts, dim=-1)  # [N, M]
    return parts.squeeze(0) if scalar else parts

def parts_to_id(parts: torch.Tensor, K_list: List[int]) -> torch.Tensor:
    """可选自检：parts:[*,M] → 大 id:[*]"""
    base = 1
    y = torch.zeros(parts.shape[:-1], dtype=parts.dtype, device=parts.device)
    for m, K in enumerate(K_list):
        y = y + parts[..., m] * base
        base *= K
    return y

# ----------------- 你来实现的 DataLoader -----------------

def build_dataloader(batch_size: int = 64, num_workers: int = 4) -> DataLoader:
    """
    TODO: 返回你的 DataLoader。
    要求 __iter__ 每个 batch 是形如 [B, N] 的 LongTensor（元素 ∈ [0, V-1]）。
    """
    raise NotImplementedError("请在 build_dataloader() 中返回你的 DataLoader（batch 为 [B,N] LongTensor）。")

# ----------------- 核心统计：整批 ALL -----------------

@torch.no_grad()
def estimate_init_counts_from_loader(
    dataloader: DataLoader,
    K_list: List[int],
    max_batches: Optional[int] = None,
    laplace_smoothing: float = 1.0,
    verbose_every: int = 100,
    pad_id: Optional[int] = None,   # 可选：若有 padding id（比如 -1 或 32768），这里填上以跳过
) -> List[torch.Tensor]:
    """
    遍历 dataloader，**整批所有值**都计入统计（可选跳过 pad_id）。
    每个 batch 是 [B, N] 的 LongTensor（元素 ∈ [0, V-1] 或等于 pad_id）。
    """
    device = torch.device("cpu")
    counts = [torch.zeros(K, dtype=torch.float64, device=device) for K in K_list]
    V = prod(K_list)

    for bi, batch in enumerate(dataloader):
        if max_batches is not None and bi >= max_batches: break
        idx = batch[0]
        batch_size = idx.shape[0]
        idx = idx.reshape(batch_size, -1)
        if not torch.is_tensor(idx) or idx.dim() != 2:
            raise ValueError(f"期望 Index 是 [B,N] LongTensor，收到 {type(idx)} / {getattr(idx,'shape',None)}")

        y = idx.reshape(-1).to(dtype=torch.long, device=device)  # [B*N] → CPU
        if pad_id is not None:
            y = y[y != pad_id]
        if y.numel() == 0:
            if verbose_every and (bi + 1) % verbose_every == 0:
                print(f"[batch {bi+1}] 全是 pad，跳过。")
            continue

        if (y.min() < 0) or (y.max() >= V):
            raise ValueError(f"[batch {bi}] 存在越界 id（应在 [0,{V-1}]），min={int(y.min())}, max={int(y.max())}")

        # 位分解（低位在前）：V=32768=32*32*32
        cur = y
        for m, K in enumerate(K_list):
            digit = cur % K                      # 第 m 段的位值
            cur = torch.div(cur, K, rounding_mode="floor")
            counts[m] += torch.bincount(digit, minlength=K).to(dtype=torch.float64)

        if verbose_every and (bi + 1) % verbose_every == 0:
            tot = [int(c.sum().item()) for c in counts]
            nz  = [int((c > 0).sum().item()) for c in counts]
            print(f"[batch {bi+1}] 累计样本={sum(tot)}, 各段非零类数={nz}")

    # Laplace 平滑（避免 log(0)）
    if laplace_smoothing and laplace_smoothing > 0:
        for m in range(len(counts)):
            counts[m] += float(laplace_smoothing)

    return [c.to(torch.float32).cpu() for c in counts]

# ----------------- 保存 -----------------

def save_counts(counts: List[torch.Tensor], K_list: List[int], save_prefix: str) -> None:
    """
    保存两份：
      - {save_prefix}.pt   ：{'K_list': K_list, 'counts': List[Tensor]}
      - {save_prefix}.json ：人类可读（四舍五入为 int）
    """
    os.makedirs(os.path.dirname(save_prefix) or ".", exist_ok=True)
    pt_path = f"{save_prefix}.pt"
    torch.save({"K_list": K_list, "counts": counts}, pt_path)

    json_path = f"{save_prefix}.json"
    counts_int = [c.round().to(torch.long).tolist() for c in counts]
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"K_list": K_list, "counts": counts_int}, f, ensure_ascii=False, indent=2)

    print(f"[保存完成] {pt_path} 以及 {json_path}")
    for m, c in enumerate(counts):
        tot = float(c.sum().item())
        nz = int((c > 0).sum().item())
        print(f"  段 {m} (K={K_list[m]}): 总计数={tot:.0f}, 非零类数={nz}, "
              f"min={float(c.min().item()):.1f}, max={float(c.max().item()):.1f}")


# ---------------------------
# 主程序
# ---------------------------

def main():
    parser = argparse.ArgumentParser(description="Offline estimator for init_counts_list (Balanced Softmax)")
    parser.add_argument("--data_path", type=str, required=True, help='path to the mv dataset')
    parser.add_argument("--global_index_file", type=str, required=True, help='path to the global index file')
    parser.add_argument("--clip_ckpt", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--save_prefix", type=str, required=True,
                        help="保存文件前缀（会生成 .pt 和 .json）")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_batches", type=int, default=None,
                        help="可选：只遍历前 N 个 batch 用于快速预估")
    parser.add_argument("--laplace", type=float, default=1.0,
                        help="拉普拉斯平滑系数，避免 0 计数（默认 1.0）")
    parser.add_argument("--k_list", type=int, nargs="+", default=K_LIST_DEFAULT,
                        help="例如: --k_list 32 32 32")
    args = parser.parse_args()

    V = prod(args.k_list)
    assert V == 32768, f"K_list 的乘积应为 32768，你给的是 {args.k_list} (乘积={V})"

    # 1) 构建 dataloader （你需要在 build_dataloader 里实现）
    dataset = MultiViewDataset(args)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)

    # 2) 统计 counts
    counts = estimate_init_counts_from_loader(
        dataloader=dataloader,
        K_list=args.k_list,
        max_batches=args.max_batches,
        laplace_smoothing=args.laplace,
        verbose_every=100
    )

    # 3) 保存
    save_counts(counts, args.k_list, save_prefix=args.save_prefix)


if __name__ == "__main__":
    main()