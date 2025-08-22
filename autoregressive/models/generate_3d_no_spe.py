# Modified from:
#   gpt-fast: https://github.com/pytorch-labs/gpt-fast/blob/main/generate.py
#   DiT:      https://github.com/facebookresearch/DiT/blob/main/models.py
#   LlamaGen: https://github.com/FoundationVision/LlamaGen/blob/main/autoregressive/models/gpt.py

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch._dynamo.config
import torch._inductor.config
import copy
# torch._inductor.config.coordinate_descent_tuning = True
# torch._inductor.config.triton.unique_kernel_names = True
# torch._inductor.config.fx_graph_cache = True # Experimental feature to reduce compilation times, will be on by default in future


### from https://huggingface.co/transformers/v3.2.0/_modules/transformers/generation_utils.html
def top_k_top_p_filtering(
    logits,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


def sample(logits, temperature: float=1.0, top_k: int=0, top_p: float=1.0, sample_logits=True):        
    logits = logits[:, -1, :] / max(temperature, 1e-5)
    if top_k > 0 or top_p < 1.0:
        logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    probs = F.softmax(logits, dim=-1)
    if sample_logits:
        idx = torch.multinomial(probs, num_samples=1)
    else:
        _, idx = torch.topk(probs, k=1, dim=-1)
    return idx, probs

def sample_rvq(logits_list, temperature: float=1.0, top_k: int=0, top_p: float=1.0, sample_logits=True):
    logits = rvq_big_logits_from_heads(logits_list)
    probs = rvq_big_logprobs_from_heads([l[:,-1:,:] for l in logits_list], temperature)
    logits = logits[:, -1, :] / max(temperature, 1e-5)
    probs = probs[:, -1, :]
    if top_k > 0 or top_p < 1.0:
        logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    # probs = F.softmax(logits, dim=-1)
    if sample_logits:
        idx = torch.multinomial(probs, num_samples=1)
    else:
        _, idx = torch.topk(probs, k=1, dim=-1)
    return idx, probs

def sample_multi(logits, temperature: float=1.0, top_k: int=0, top_p: float=1.0, spe_token_num=4, sample_logits=True):
    logits = logits[:, -1*spe_token_num:, :]
    batch_size, num_samples, vocab_size = logits.shape
    logits = logits.reshape(batch_size * num_samples, vocab_size) / max(temperature, 1e-5)
    if top_k > 0 or top_p < 1.0:
        logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    probs = F.softmax(logits, dim=-1)
    if sample_logits:
        idx = torch.multinomial(probs, num_samples=1)
    else:
        _, idx = torch.topk(probs, k=1, dim=-1)
    idx = idx.view(batch_size, num_samples)
    probs = probs.view(batch_size, num_samples, vocab_size)
    return idx, probs

def sample_rvq_multi(logits_list, temperature: float=1.0, top_k: int=0, top_p: float=1.0, sample_logits=True, spe_token_num=4):
    logits = rvq_big_logits_from_heads(logits_list)
    probs = rvq_big_logprobs_from_heads([l[:, -1*spe_token_num:, :] for l in logits_list], temperature)
    logits = logits[:, -1*spe_token_num:, :]
    # probs = probs[:, -1*spe_token_num:, :]
    batch_size, num_samples, vocab_size = logits.shape
    logits = logits.reshape(batch_size * num_samples, vocab_size) / max(temperature, 1e-5)
    probs = probs.reshape(batch_size * num_samples, vocab_size)
    if top_k > 0 or top_p < 1.0:
        logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    # probs = F.softmax(logits, dim=-1)
    if sample_logits:
        idx = torch.multinomial(probs, num_samples=1)
    else:
        _, idx = torch.topk(probs, k=1, dim=-1)
    idx = idx.view(batch_size, num_samples)
    probs = probs.view(batch_size, num_samples, vocab_size)
    return idx, probs

def logits_to_probs(logits, temperature: float = 1.0, top_p: float=1.0, top_k: int = None, **kwargs):
    logits = logits / max(temperature, 1e-5)
    if top_k > 0 or top_p < 1.0:
        logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs

def rvq_big_logits_from_heads(logits_list):
    """
    logits_list = [z0, z1, z2]，每个张量形状 [B, L, 32]
    返回 big_logits: [B, L, 32768]
    """
    z0, z1, z2 = logits_list
    big = (
        z0[:, :, :, None, None] +      # [B,L,32,1,1]
        z1[:, :, None, :, None] +      # [B,L,1,32,1]
        z2[:, :, None, None, :]        # [B,L,1,1,32]
    ).reshape(z0.size(0), z0.size(1), -1)       # [B,L, 32*32*32]
    return big 

def rvq_big_logprobs_from_heads(logits_list, temperature: float = 1.0):
    """
    返回 big_logprobs: [B, 32768]，等价于对 big_logits 做 log_softmax
    但用分段 log_softmax 的和实现，数值更稳。
    """
    z0, z1, z2 = [F.log_softmax(z, dim=-1) for z in logits_list]  # [B,32] 各段 log-prob
    big_logp = (
        z0[:, :, :, None, None] + z1[:, :, None, :, None] + z2[:, :, None, None, :]
    ).reshape(z0.size(0), z0.size(1), -1)  # [B,L, 32768]
    return big_logp


def prefill(model, cond_idx: torch.Tensor, camera_params: torch.Tensor, input_pos: torch.Tensor, cfg_scale: float, **sampling_kwargs):
    # print(f"At prefill stage, input_pos is: {input_pos}")
    if cfg_scale > 1.0:
        logits = model.forward_reference(None, None, camera_params, cond_idx, input_pos)
        logits_combined = logits
        cond_logits, uncond_logits = torch.split(logits_combined, len(logits_combined) // 2, dim=0)
        logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale
    else:
        logits = model.forward_reference(None, None, camera_params, cond_idx, input_pos)

    return sample(logits, **sampling_kwargs)[0]

def prefill_rvq(model, cond_idx: torch.Tensor, camera_params: torch.Tensor, input_pos: torch.Tensor, cfg_scale: float, **sampling_kwargs):
    print(f"At prefill stage, input_pos is: {input_pos}")
    if cfg_scale > 1.0:
        logits_list = model.forward_reference(None, None, camera_params, cond_idx, input_pos)
        cond_logits_list = [torch.split(logit, len(logit) // 2, dim=0)[0] for logit in logits_list]
        uncond_logits_list = [torch.split(logit, len(logit) // 2, dim=0)[1] for logit in logits_list]

        logits_list = [uncond_logits + (cond_logits - uncond_logits) * cfg_scale for cond_logits, uncond_logits in zip(cond_logits_list, uncond_logits_list)]
    else:
        logits_list = model.forward_reference(None, None, camera_params, cond_idx, input_pos)
    return sample_rvq(logits_list, **sampling_kwargs)[0]


def decode_one_token(model, x: torch.Tensor, prev_cam: torch.Tensor, cur_cam: torch.Tensor, input_pos: torch.Tensor, cfg_scale: float, cfg_flag: bool, **sampling_kwargs):
    assert input_pos.shape[-1] == 3
    if cfg_scale > 1.0:
        x_combined = torch.cat([x, x])
        logits = model.forward_reference(x_combined, prev_cam, cur_cam, cond_idx=None, input_pos=input_pos)
        logits_combined = logits
        cond_logits, uncond_logits = torch.split(logits_combined, len(logits_combined) // 2, dim=0) 
        if cfg_flag:
            logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale
        else:
            logits = cond_logits
    else:
        logits = model.forward_reference(x, prev_cam, cur_cam, cond_idx=None, input_pos=input_pos)
    return sample(logits, **sampling_kwargs)


def decode_one_token_rvq(model, x: torch.Tensor, prev_cam: torch.Tensor, cur_cam: torch.Tensor, input_pos: torch.Tensor, cfg_scale: float, cfg_flag: bool, **sampling_kwargs):
    assert input_pos.shape[-1] == 3
    if cfg_scale > 1.0:
        x_combined = torch.cat([x, x])
        logits_list = model.forward_reference(x_combined, prev_cam, cur_cam, cond_idx=None, input_pos=input_pos)
        cond_logits_list = [torch.split(logit, len(logit) // 2, dim=0)[0] for logit in logits_list]
        uncond_logits_list = [torch.split(logit, len(logit) // 2, dim=0)[1] for logit in logits_list]
        logits_list = [uncond_logits + (cond_logits - uncond_logits) * cfg_scale for cond_logits, uncond_logits in zip(cond_logits_list, uncond_logits_list)]
    else:
        logits_list = model.forward_reference(x, prev_cam, cur_cam, cond_idx=None, input_pos=input_pos)
    return sample_rvq(logits_list, **sampling_kwargs)


def decode_one_token_multi(model, x: torch.Tensor, prev_cam: torch.Tensor, cur_cam: torch.Tensor, input_pos: torch.Tensor, cfg_scale: float, cfg_flag: bool, spe_token_num: int, **sampling_kwargs):
    # assert input_pos.shape[-1] == 1
    if cfg_scale > 1.0:
        x_combined = torch.cat([x, x])
        logits = model.forward_reference(x_combined, prev_cam, cur_cam, cond_idx=None, input_pos=input_pos)
        logits_combined = logits
        cond_logits, uncond_logits = torch.split(logits_combined, len(logits_combined) // 2, dim=0) 
        if cfg_flag:
            logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale
        else:
            logits = cond_logits
    else:
        logits = model.forward_reference(x, prev_cam, cur_cam, cond_idx=None, input_pos=input_pos)
    return sample_multi(logits, spe_token_num=spe_token_num, **sampling_kwargs)


def decode_one_token_rvq_multi(model, x: torch.Tensor, prev_cam: torch.Tensor, cur_cam: torch.Tensor, input_pos: torch.Tensor, cfg_scale: float, cfg_flag: bool, spe_token_num: int, **sampling_kwargs):
    # assert input_pos.shape[-1] == 1
    if cfg_scale > 1.0:
        x_combined = torch.cat([x, x])
        logits_list = model.forward_reference(x_combined, prev_cam, cur_cam, cond_idx=None, input_pos=input_pos)
        cond_logits_list = [torch.split(logit, len(logit) // 2, dim=0)[0] for logit in logits_list]
        uncond_logits_list = [torch.split(logit, len(logit) // 2, dim=0)[1] for logit in logits_list]
        logits_list = [uncond_logits + (cond_logits - uncond_logits) * cfg_scale for cond_logits, uncond_logits in zip(cond_logits_list, uncond_logits_list)]
    else:
        logits_list = model.forward_reference(x, prev_cam, cur_cam, cond_idx=None, input_pos=input_pos)
    return sample_rvq_multi(logits_list, spe_token_num=spe_token_num, **sampling_kwargs)

def decode_n_tokens(
    model, cur_token: torch.Tensor, camera_params: torch.Tensor, cam_pointer: int, input_pos: torch.Tensor, num_new_tokens: int, 
    cfg_scale: float, cfg_interval: int,
    **sampling_kwargs):
    new_tokens, new_probs = [], []
    cfg_flag = True
    assert cam_pointer >= 1, "cam_pointer should be at least 1"
    for i in range(num_new_tokens):
        # print(f"At autoregression stage, at step {i}, input_pos is: {input_pos}, cam_pointer is: {cam_pointer}")
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True): # Actually better for Inductor to codegen attention here
            if cfg_interval > -1 and i > cfg_interval:
                cfg_flag = False
            next_token, next_prob = decode_one_token(
                model, cur_token, 
                camera_params[:, cam_pointer-1:cam_pointer], # previous camera token, interleave with cur_token
                camera_params[:,cam_pointer:cam_pointer+1], # cur camera token
                input_pos, 
                cfg_scale, 
                cfg_flag, 
                **sampling_kwargs
            )
            input_pos += 2
            cam_pointer += 1
            new_tokens.append(next_token.clone())
            new_probs.append(next_prob.clone())
            cur_token = next_token.view(-1, 1)
    
    return new_tokens, new_probs

def decode_n_tokens_multi(
    model, cur_token: torch.Tensor, camera_params: torch.Tensor, cam_pointer: int, input_pos: torch.Tensor, num_new_tokens: int, 
    cfg_scale: float, cfg_interval: int, spe_token_num: int, **sampling_kwargs):
    new_tokens, new_probs = [], []
    cfg_flag = True
    for i in range(0, num_new_tokens, spe_token_num):
        # print(f"At autoregression parallel stage, at step {i}, input_pos is: {[input_pos[0], input_pos[-1]]}, cam_pointer is: {cam_pointer}")
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True): # Actually better for Inductor to codegen attention here
            if cfg_interval > -1 and i > cfg_interval:
                cfg_flag = False
            # print(f"cur_token shape: {cur_token.shape}")
            # print(f"camera_params shape: {camera_params.shape}")
            next_token, next_prob = decode_one_token_multi(
                model, cur_token, 
                camera_params[:,cam_pointer-spe_token_num:cam_pointer], # previous camera token, interleave with cur_token
                camera_params[:,cam_pointer:cam_pointer+spe_token_num], 
                input_pos, 
                cfg_scale, 
                cfg_flag, 
                spe_token_num=spe_token_num, 
                **sampling_kwargs
            )
            # print(f"next_token shape: {next_token.shape}")
            cam_pointer += spe_token_num
            input_pos += (spe_token_num*2)
            new_tokens.append(next_token.clone())
            new_probs.append(next_prob.clone())
            cur_token = next_token
    
    return new_tokens, new_probs


@torch.no_grad()
def generate(model, cond, camera_params, max_new_tokens, emb_masks=None, cfg_scale=1.0, cfg_interval=-1, ar_token_num=16, spe_token_num=15, **sampling_kwargs):
    if cfg_scale > 1.0:
        cond_null = torch.zeros_like(cond) + model.cls_embedding.uncond_embedding
        cond_combined = torch.cat([cond, cond_null])
        camera_params_combined = torch.cat([camera_params, camera_params])
    else:
        cond_combined = cond
    T = 1 #cond.shape[1]      

    T_new = T + max_new_tokens
    max_seq_length = T_new
    max_batch_size = cond.shape[0]

    cam_pointer = 0

    device = cond.device
    with torch.device(device):
        max_batch_size_cfg = max_batch_size * 2 if cfg_scale > 1.0 else max_batch_size
        model.setup_caches(max_batch_size=max_batch_size_cfg, max_seq_length=max_seq_length, dtype=model.tok_embeddings.weight.dtype)
    
    if emb_masks is not None:
        assert emb_masks.shape[0] == max_batch_size
        assert emb_masks.shape[-1] == T
        if cfg_scale > 1.0:
            model.causal_mask[:, :, :T] = model.causal_mask[:, :, :T] * torch.cat([emb_masks, emb_masks]).unsqueeze(1)
        else:
            model.causal_mask[:, :, :T] = model.causal_mask[:, :, :T] * emb_masks.unsqueeze(1)

        eye_matrix = torch.eye(model.causal_mask.size(1), model.causal_mask.size(2), device=device)
        model.causal_mask[:] = model.causal_mask * (1 - eye_matrix) + eye_matrix
    
    # create an empty tensor of the expected final shape and fill in the current tokens
    seq = torch.empty((max_batch_size, T_new), dtype=torch.int, device=device)
    input_pos = torch.arange(0, T+1, device=device)
    # prefill: just the first token, no camera params needed
    next_token = prefill(model, cond_combined, camera_params_combined[:,cam_pointer:cam_pointer+1], input_pos, cfg_scale, **sampling_kwargs)
    cam_pointer += 1
    seq[:, T:T+1] = next_token
    prefill_token = next_token.clone()
    
    spe_token_num = spe_token_num+1

    # autoregressive
    input_pos = torch.tensor([T, T+1, T+2], device=device, dtype=torch.int)
    generated_tokens, _ = decode_n_tokens(model, next_token, camera_params_combined, cam_pointer, input_pos, ar_token_num-1, cfg_scale, cfg_interval, **sampling_kwargs)
    cam_pointer += (ar_token_num-1)
    seq[:, T+1:T+1+len(generated_tokens)] = torch.cat(generated_tokens, dim=1)

    # parallel
    input_pos = torch.tensor([T+i for i in range(spe_token_num * 3)], device=device, dtype=torch.int) # FIXME: why *3? Because idx (spe) prev cam (spe) cur cam (spe), but not sure
    next_token = torch.cat(generated_tokens, dim=1)[:,-spe_token_num:]
    next_token = torch.cat([prefill_token, next_token], dim=1)
    # print(f"next_token shape: {next_token.shape}")
    generated_tokens, _ = decode_n_tokens_multi(model, next_token, camera_params_combined, cam_pointer, input_pos, max_new_tokens - ar_token_num, cfg_scale, cfg_interval, spe_token_num, **sampling_kwargs)
    seq[:, T+ar_token_num:] = torch.cat(generated_tokens, dim=1)
    return seq[:, T:]