import torch
# import torch.nn as nn
# from torch.nn import functional as F
# import numpy as np

# def precompute_freqs_cis_3d(num_views: int, grid_size: int, n_elem: int, base: int = 10000, cls_token_num=120, spe_token_num=3, ar_token_num=4):
#     # split the dimension into half, one for x and one for y
#     spatial_dim = n_elem // 8 * 3
#     temporal_dim = n_elem // 4

#     freqs = 1.0 / (
#         base ** (torch.arange(0, spatial_dim, 2)[: (spatial_dim // 2)].float() / spatial_dim)
#     )
#     view_freqs = 1.0 / (base ** (torch.arange(0, temporal_dim, 2)[: (temporal_dim // 2)].float() / temporal_dim))

#     t = torch.arange(grid_size, device=freqs.device)
#     view_t = torch.arange(num_views, device=freqs.device)

#     freqs = torch.outer(t, freqs) # (grid_size, head_dim // 2)
#     view_freqs = torch.outer(view_t, view_freqs) # (num_views, head_dim // 2)
#     # print(freqs.shape, view_freqs.shape)

#     freqs_grid = torch.concat(
#         [
#             view_freqs[:, None, None, :].expand(-1, grid_size, grid_size, -1),
#             freqs[None, :, None, :].expand(num_views, -1, grid_size, -1),
#             freqs[None, None, :, :].expand(num_views, grid_size, -1, -1),
#         ],
#         dim=-1,
#     )  # (num_views, grid_size, grid_size, head_dim // 2)
    
#     cache_grid = torch.stack(
#         [torch.cos(freqs_grid), torch.sin(freqs_grid)], dim=-1
#     )  # (num_views, grid_size, grid_size, head_dim // 2, 2)
#     print(cache_grid.shape)
#     sub_num = int((ar_token_num//num_views)**0.5)
#     cache_grid = cache_grid.reshape(num_views, sub_num, grid_size//sub_num, sub_num, grid_size//sub_num, -1, 2)
#     print(cache_grid.shape)
#     cache_grid = cache_grid.permute(2, 4, 0, 1, 3, 5, 6)
#     print(cache_grid.shape)
#     cache = cache_grid.flatten(0, 4)
#     print(cache.shape)
#     cache_one, cache_two = cache[:ar_token_num], cache[ar_token_num:]
#     print(cache_one.shape, cache_two.shape)
#     sep_cache = torch.zeros(spe_token_num, n_elem // 2, 2)
#     print(sep_cache.shape)
#     cond_cache = torch.cat([torch.zeros(cls_token_num, n_elem // 2, 2), cache_one, sep_cache, cache_two])
#     return cond_cache


# def split_to_parts(y: torch.Tensor, K_list: list[int]) -> torch.Tensor:
#     parts = []
#     cur = y.clone()
#     for K in K_list:
#         parts.append(cur % K)
#         cur = torch.div(cur, K, rounding_mode='floor')
#     return torch.stack(parts, dim=-1)

# def load_init_counts_from_pt(path, device, dtype):
#     ckpt = torch.load(path, map_location="cpu")
#     K_list = ckpt.get("K_list", [32, 32, 32])
#     counts_list = ckpt["counts"]                     # List[Tensor], 每个形状 [32]
#     # 基本校验与转 dtype/device
#     assert len(counts_list) == len(K_list)
#     assert all(c.numel() == K for c, K in zip(counts_list, K_list))
#     # 避免 0 计数（如果你离线时已做过 Laplace，就不会为 0）
#     counts_list = [c.clone().clamp_min(1.0).to(device, dtype=dtype) for c in counts_list]
#     return K_list, counts_list


# class BalancedSoftmaxLoss(nn.Module):
#     """
#     CE on logits + log(counts). counts 以 EMA 方式在线估计或用先验初始化。
#     """
#     def __init__(self, K: int, label_smoothing: float = 0.0, init_counts: torch.Tensor | None = None):
#         super().__init__()
#         if init_counts is None:
#             init_counts = torch.ones(K)  # 均匀先验
#         init_counts = init_counts.float().clamp_min(1.0)
#         self.register_buffer("counts", init_counts)          # [K]
#         self.eps = label_smoothing

#     @torch.no_grad()
#     def update_counts(self, y: torch.Tensor, momentum: float = 0.05):
#         """
#         用当前 batch 标签直方图做 EMA：counts <- (1-m)*counts + m*hist
#         y: [N] (long)
#         """
#         K = self.counts.numel()
#         hist = torch.bincount(y, minlength=K).float().to(self.counts.device)
#         self.counts.mul_(1 - momentum).add_(momentum * hist)

#     def forward(self, logits: torch.Tensor, target: torch.Tensor, reduction: str = "mean"):
#         """
#         logits: [N, K] （未做 softmax 的原始 logit）
#         target: [N]
#         """
#         # logits + log(counts) 实现 Balanced Softmax
#         log_prior = torch.log(self.counts + 1e-12)                  # [K]
#         z = logits + log_prior                                      # [N, K]
#         logp = F.log_softmax(z, dim=-1)                             # [N, K]

#         if self.eps > 0:  # Label Smoothing
#             nll = F.nll_loss(logp, target, reduction=reduction)
#             smooth = -logp.mean(dim=-1)
#             smooth = smooth.mean() if reduction == "mean" else smooth.sum()
#             return (1 - self.eps) * nll + self.eps * smooth
#         else:
#             return F.nll_loss(logp, target, reduction=reduction)

# class RVQHeadsBalanced(nn.Module):
#     """
#     多段 RVQ 头。每段一个 Linear + BalancedSoftmaxLoss（可带 LS）。
#     """
#     def __init__(self, d_model: int, K_list: list[int],
#                  label_smoothing: float = 0.0,
#                  init_counts_list: list[torch.Tensor] | None = None):
#         super().__init__()
#         self.M = len(K_list)
#         self.heads = nn.ModuleList([nn.Linear(d_model, Km) for Km in K_list])
#         if init_counts_list is None:
#             init_counts_list = [None] * self.M
#         self.losses = nn.ModuleList([
#             BalancedSoftmaxLoss(Km, label_smoothing, init_counts_list[m])
#             for m, Km in enumerate(K_list)
#         ])

#     @torch.no_grad()
#     def update_priors(self, y_parts: torch.Tensor, momentum: float = 0.05):
#         # y_parts: [N, M]
#         for m in range(self.M):
#             self.losses[m].update_counts(y_parts[:, m], momentum)

#     def forward(self, h: torch.Tensor, y_parts: torch.Tensor,
#                 update_prior: bool = False, momentum: float = 0.05,
#                 reduction: str = "mean"):
#         """
#         h:       [N, D]       —— 只取“条件位”的隐藏态
#         y_parts: [N, M] (long)—— 每段的目标 id
#         """
#         if update_prior:
#             self.update_priors(y_parts, momentum)

#         per_head_losses = []
#         for m in range(self.M):
#             logits_m = self.heads[m](h)                   # [N, K_m]
#             loss_m = self.losses[m](logits_m, y_parts[:, m], reduction=reduction)
#             per_head_losses.append(loss_m)

#         if reduction == "mean":
#             return sum(per_head_losses) / self.M
#         elif reduction == "sum":
#             return sum(per_head_losses)
#         else:
#             return torch.stack(per_head_losses, dim=-1)    # [N, M]


# init_counts_path = "/wekafs/ict/wenbinte/projects/PAR3D/init_counts/rvq32x32x32.pt"
# K_list, counts_list = load_init_counts_from_pt(init_counts_path, torch.device("cuda"), torch.float32)
# m = RVQHeadsBalanced(2048, K_list, label_smoothing=0.1, init_counts_list=counts_list).to(torch.device("cuda"), dtype=torch.float32)

# idx_path = "/wekafs/ict/wenbinte/data/MVDataset_vidtok_code_fsq_causal_488_32768_video/re10k/train/000000/3a4e08a3f02ad769_stride_2_start_13/indices.npy"
# idx = torch.from_numpy(np.load(idx_path)).to(torch.device("cuda")).long()
# idx = idx[1:,...].reshape(-1)

# print(idx.shape)
# y_parts = split_to_parts(idx, K_list)
# print(y_parts.shape)
# h = torch.randn(4096, 2048).to(torch.device("cuda"), dtype=torch.float32)
# loss = m(h, y_parts, update_prior=True, momentum=0.05, reduction="mean")
# print(loss)

def interleave_tokens_per_token(seq1, seq2, num_tokens):
    # result[0, 1, 2, ..., num_tokens - 1] = seq1[:num_tokens]
    # result[num_tokens, num_tokens + 1, ..., 2*num_tokens - 1] = seq2[:num_tokens]

    result = []
    for i in range(len(seq1) // num_tokens):
        result.extend(seq1[i*num_tokens:(i+1)*num_tokens])
        result.extend(seq2[i*num_tokens:(i+1)*num_tokens])
    return result

l = ["cls", "c1", "x1", "c2", "x2", "c3", "x3", "c4", "x4", 
     "c5", "c6", "c7", "c8", "x5", "x6", "x7", "x8", 
     "c9", "c10", "c11", "c12", "x9", "x10", "x11", "x12", 
     "c13", "c14", "c15", "c16", "x13", "x14", "x15", "x16"]



start_idx = 1
doubled_seq_len = len(l)
valid_indices = [start_idx + i * 2 for i in range(4)]
last_idx = start_idx + 4 * 2
num_groups = (doubled_seq_len-last_idx) // 4
for i in range(num_groups):
    if i % 2 == 0:
        valid_indices += [last_idx + j for j in range(i * 4, (i+1) * 4)]

l1 = [l[i] for i in valid_indices]
l2 = [l[i] for i in range(len(l)) if i not in valid_indices][1:]

print(l1)
print(l2)


t = interleave_tokens_per_token(l1, l2, 4)
print(t)