# Modified from:
#   VQGAN:    https://github.com/CompVis/taming-transformers/blob/master/taming/modules/transformer/mingpt.py
#   DiT:      https://github.com/facebookresearch/DiT/blob/main/models.py  
#   nanoGPT:  https://github.com/karpathy/nanoGPT/blob/master/model.py
#   llama:    https://github.com/facebookresearch/llama/blob/main/llama/model.py
#   gpt-fast: https://github.com/pytorch-labs/gpt-fast/blob/main/model.py
#   PixArt:   https://github.com/PixArt-alpha/PixArt-alpha/blob/master/diffusion/model/nets/PixArt_blocks.py
#   LlamaGen: https://github.com/FoundationVision/LlamaGen/blob/main/autoregressive/models/gpt.py

from dataclasses import dataclass
from typing import Optional, List
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F
from utils.drop_path import DropPath
from tokenizer.vidtok.modules.model_3dcausal import EncoderCausal3DPadding


def find_multiple(n: int, k: int):
    if n % k == 0:
        return n
    return n + k - (n % k)

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layer: int = 32
    n_head: int = 32
    n_kv_head: Optional[int] = None
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    rope_base: float = 10000
    norm_eps: float = 1e-5
    initializer_range: float = 0.02
    
    token_dropout_p: float = 0.1
    attn_dropout_p: float = 0.0
    resid_dropout_p: float = 0.1
    ffn_dropout_p: float = 0.1
    drop_path_rate: float = 0.0
    
    image_dim: int = 512
    class_dropout_prob: float = 0.1

    vocab_size: int = 16384
    cls_token_num: int = 1
    block_size: int = 4096
    max_batch_size: int = 32
    max_seq_len: int = 2048
    spe_token_num: int = 3

    # token permutation
    temporal_size: int = 4
    downsample_size: int = 8
    image_size: int = 256
    ar_token_num: int = 16

    # camera causal encoder
    double_z: bool = False
    in_channels: int = 6
    out_ch: int = 6
    ch: int = 128
    ch_mult: tuple[int, ...] = (1, 2, 4, 4)
    time_downsample_factor: int = 4
    num_res_blocks: int = 2
    dropout: float = 0.0
    use_checkpoint: bool = False
    init_pad_mode: str = "replicate"
    norm_type: str = "layernorm"
    fix_encoder: bool = False
    fix_decoder: bool = False


#################################################################################
#                      Embedding Layers for Class Labels                        #
#################################################################################
class SpecialTokenEmbedding(nn.Module):
    def __init__(self, num_special_tokens, hidden_size):
        super().__init__()
        self.num_special_tokens = num_special_tokens
        self.hidden_size = hidden_size
        self.special_embeddings = nn.Embedding(num_special_tokens, hidden_size)

    def forward(self):
        special_tokens = torch.arange(self.num_special_tokens, device=self.special_embeddings.weight.device)
        special_embeddings = self.special_embeddings(special_tokens)
        return special_embeddings


#################################################################################
#                      Embedding Layers for Image Feature                        #
#################################################################################
class ImageEmbedder(nn.Module):
    """
    Embeds image features into vector representations.
    """
    def __init__(self, in_channels, hidden_size, uncond_prob, token_num=1):
        super().__init__()
        self.img_proj = MLP(in_features=in_channels, hidden_features=hidden_size, out_features=hidden_size)
        self.register_buffer("uncond_embedding", nn.Parameter(torch.randn(token_num, in_channels) / in_channels ** 0.5))
        self.uncond_prob = uncond_prob

    def token_drop(self, image, force_drop_ids=None):
        """
        Drops image features to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(image.shape[0], device=image.device) < self.uncond_prob
        else:
            drop_ids = force_drop_ids == 1
        image = torch.where(drop_ids[:, None], self.uncond_embedding, image)
        return image
    
    def forward(self, image, train, force_drop_ids=None):
        use_dropout = self.uncond_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            image = self.token_drop(image, force_drop_ids)
        embeddings = self.img_proj(image)
        return embeddings


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=False)
        self.act = nn.GELU(approximate='tanh')
        self.fc2 = nn.Linear(hidden_features, out_features, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


#################################################################################
#                                  GPT Model                                    #
#################################################################################
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class FeedForward(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        hidden_dim = 4 * config.dim
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if config.ffn_dim_multiplier is not None:
            hidden_dim = int(config.ffn_dim_multiplier * hidden_dim)
        hidden_dim = find_multiple(hidden_dim, config.multiple_of)

        self.w1 = nn.Linear(config.dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(config.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, config.dim, bias=False)
        self.ffn_dropout = nn.Dropout(config.ffn_dropout_p)

    def forward(self, x):
        return self.ffn_dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class KVCache(nn.Module):
    def __init__(self, max_batch_size, max_seq_length, n_head, head_dim, dtype):
        super().__init__()
        cache_shape = (max_batch_size, n_head, max_seq_length, head_dim)
        self.register_buffer('k_cache', torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer('v_cache', torch.zeros(cache_shape, dtype=dtype))

    def update(self, input_pos, k_val, v_val):
        # input_pos: [S], k_val: [B, H, S, D]
        assert input_pos.shape[0] == k_val.shape[2]
        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val

        return k_out, v_out


class Attention(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        assert config.dim % config.n_head == 0
        self.dim = config.dim
        self.head_dim = config.dim // config.n_head
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head if config.n_kv_head is not None else config.n_head
        total_kv_dim = (self.n_head + 2 * self.n_kv_head) * self.head_dim

        # key, query, value projections for all heads, but in a batch
        self.wqkv = nn.Linear(config.dim, total_kv_dim, bias=False)
        self.wo = nn.Linear(config.dim, config.dim, bias=False)
        self.kv_cache = None

        # regularization
        self.attn_dropout_p = config.attn_dropout_p
        self.resid_dropout = nn.Dropout(config.resid_dropout_p)

    def forward(
        self, x: torch.Tensor, freqs_cis: torch.Tensor = None, 
        input_pos: Optional[torch.Tensor] = None, 
        mask: Optional[torch.Tensor] = None
    ):
        bsz, seqlen, _ = x.shape
        kv_size = self.n_kv_head * self.head_dim
        xq, xk, xv = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)

        xq = xq.view(bsz, seqlen, self.n_head, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_head, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_head, self.head_dim)
        
        xq = apply_rotary_emb(xq, freqs_cis)
        xk = apply_rotary_emb(xk, freqs_cis)

        xq, xk, xv = map(lambda x: x.transpose(1, 2), (xq, xk, xv))

        if self.kv_cache is not None:
            keys, values = self.kv_cache.update(input_pos, xk, xv)
        else:
            keys, values = xk, xv
        keys = keys.repeat_interleave(self.n_head // self.n_kv_head, dim=1)
        values = values.repeat_interleave(self.n_head // self.n_kv_head, dim=1)

        output = F.scaled_dot_product_attention(
            xq, keys, values, 
            attn_mask=mask, 
            is_causal=True if mask is None else False, # is_causal=False is for KV cache
            dropout_p=self.attn_dropout_p if self.training else 0)            
        
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)

        output = self.resid_dropout(self.wo(output))
        return output


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelArgs, drop_path: float):
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(
        self, x: torch.Tensor, freqs_cis: torch.Tensor, start_pos: int, mask: Optional[torch.Tensor] = None):
        h = x + self.drop_path(self.attention(self.attention_norm(x), freqs_cis, start_pos, mask))
        out = h + self.drop_path(self.feed_forward(self.ffn_norm(h)))
        return out


class Transformer(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.n_layer = config.n_layer
        self.block_size = config.block_size
        self.cls_token_num = config.cls_token_num
        self.spe_token_num = config.spe_token_num
        self.ar_token_num = config.ar_token_num
        self.num_views = config.temporal_size

       
        self.cls_embedding = ImageEmbedder(config.image_dim, config.dim, config.class_dropout_prob)

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.tok_dropout = nn.Dropout(config.token_dropout_p)

        self.spe_tok_embeddings = SpecialTokenEmbedding(self.spe_token_num * 2, config.dim)

        # camera causal encoder
        self.camera_encoder = EncoderCausal3DPadding(
            double_z=config.double_z,
            z_channels=config.dim,
            in_channels=config.in_channels,
            out_ch=config.out_ch,
            ch=config.ch,
            ch_mult=config.ch_mult,
            time_downsample_factor=config.time_downsample_factor,
            num_res_blocks=config.num_res_blocks,
            dropout=config.dropout,
            use_checkpoint=config.use_checkpoint,
            init_pad_mode=config.init_pad_mode,
            norm_type=config.norm_type,
            fix_encoder=config.fix_encoder,
            fix_decoder=config.fix_decoder,
        )


        # transformer blocks
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.n_layer)]
        self.layers = torch.nn.ModuleList()
        for layer_id in range(config.n_layer):
            self.layers.append(TransformerBlock(config, dpr[layer_id]))

        # output layer
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        # 2d rotary pos embedding
        grid_size = int((self.block_size // self.num_views) ** 0.5)
        assert grid_size * grid_size == self.block_size // self.num_views
        # Create freqs_cis for interleaved camera and idx tokens
        self.freqs_cis = precompute_freqs_cis_3d(
            self.num_views, 
            grid_size, 
            self.config.dim // self.config.n_head, 
            self.config.rope_base, 
            self.cls_token_num, 
            spe_token_num=self.spe_token_num, 
            ar_token_num=self.ar_token_num
        )
        
        max_len = self.freqs_cis.shape[0]
        # Account for interleaved camera and idx tokens (doubled sequence length)
        doubled_max_len = self.cls_token_num + 2 * (max_len - self.cls_token_num)
        group_mask = torch.tril(torch.ones(doubled_max_len, doubled_max_len, dtype=torch.bool))
        group_mask[:, 0] = True
        group_size = (self.spe_token_num + 1) * 2
        
        # Adjust group mask construction for interleaved tokens
        for i in range(0, (doubled_max_len) // (group_size)):
            start = 2 * self.ar_token_num + i * (group_size)
            end = start + group_size
            group_mask[start:end, :end] = True
        self.group_mask = group_mask

        # KVCache
        self.max_batch_size = -1
        self.max_seq_length = -1

        self.initialize_weights()

    def initialize_weights(self):        
        # Initialize nn.Linear and nn.Embedding
        self.apply(self._init_weights)

        # Zero-out output layers:
        nn.init.constant_(self.output.weight, 0)

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)

    def setup_caches(self, max_batch_size, max_seq_length, dtype):
        # if self.max_seq_length >= max_seq_length and self.max_batch_size >= max_batch_size:
        #     return
        head_dim = self.config.dim // self.config.n_head
        # Account for doubled sequence length due to interleaved camera and idx tokens
        doubled_max_seq_length = max_seq_length * 2
        doubled_max_seq_length = find_multiple(doubled_max_seq_length, 8)
        self.max_seq_length = doubled_max_seq_length
        self.max_batch_size = max_batch_size
        for b in self.layers:
            b.attention.kv_cache = KVCache(max_batch_size, doubled_max_seq_length, self.config.n_head, head_dim, dtype)

        group_mask = torch.tril(torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool))
        group_mask[:, 0] = True
        group_size = (self.spe_token_num + 1) * 2
        # Adjust for interleaved tokens in cache setup
        for i in range(0, (self.max_seq_length) // (group_size)):
            start = 2 * self.ar_token_num + i * (group_size)
            end = start + group_size
            if end <= self.max_seq_length:
                group_mask[start:end, :end] = True
        self.causal_mask = group_mask.unsqueeze(0).repeat(self.max_batch_size, 1, 1)
        
        # causal_mask = torch.tril(torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool))
        # self.causal_mask = causal_mask.unsqueeze(0).repeat(self.max_batch_size, 1, 1)
        grid_size = int((self.config.block_size // self.config.temporal_size) ** 0.5)
        assert grid_size * grid_size == self.block_size // self.num_views
        # Create freqs_cis for doubled sequence length (interleaved camera and idx tokens)
        self.freqs_cis = precompute_freqs_cis_3d(self.config.temporal_size, grid_size, self.config.dim // self.config.n_head, self.config.rope_base, 
                                                  self.cls_token_num, spe_token_num=self.spe_token_num, 
                                                ar_token_num=self.ar_token_num)

    def forward(
        self, 
        idx: Optional[torch.Tensor] = None, 
        cam: Optional[torch.Tensor] = None,
        cond_idx: Optional[torch.Tensor] = None,  # cond_idx_or_embed
        input_pos:  Optional[torch.Tensor] = None, 
        targets: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        valid: Optional[torch.Tensor] = None
    ):
        if idx is not None and cond_idx is not None: # training or naive inference
            # Process camera tensor through camera encoder
            cam_encoded = self.camera_encoder(cam)
            b, dim, v, h, w = cam_encoded.shape
            cam_encoded = cam_encoded.permute(0, 2, 3, 4, 1).contiguous()  # b x v x h x w x dim
            
            # Apply same permutation as idx to camera tokens
            cam_tokens = permute_token(cam_encoded, self.config)    
            
            idx = permute_token(idx, self.config)
            if self.training and targets is None:
                targets = idx
            cond_embeddings = self.cls_embedding(cond_idx, train=self.training).unsqueeze(1)
            token_embeddings = self.tok_embeddings(idx)
            interleaved_spe_tokens = self.spe_tok_embeddings().unsqueeze(0).expand(cond_embeddings.shape[0], -1, -1)

            # interleave camera tokens with idx tokens
            interleaved_tokens = interleave_tokens(cam_tokens, token_embeddings) # b x (2*seq_len) x dim
            
            # Split interleaved tokens for special token insertion
            token_embeddings_first, token_embeddings_last = interleaved_tokens[:,:2*self.ar_token_num], interleaved_tokens[:,2*self.ar_token_num:]
            token_embeddings = torch.cat((cond_embeddings, token_embeddings_first, interleaved_spe_tokens, token_embeddings_last), dim=1)
            h = self.tok_dropout(token_embeddings)

            # Update mask for doubled sequence length
            doubled_seq_len = token_embeddings.shape[1]
            mask = self.group_mask[:doubled_seq_len, :doubled_seq_len]
            batch_size = cond_embeddings.shape[0]
            mask = mask.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1)
            mask = mask.to(h.device)

            token_freqs_cis = self.freqs_cis[self.cls_token_num:].clone().to(h.device)
            freqs_cis = torch.cat(
                (self.freqs_cis[:self.cls_token_num].to(h.device),
                 interleave_tokens_1d(token_freqs_cis, token_freqs_cis))
            )

        else:
            raise NotImplementedError("Only training or naive inference is supported")
        
        if self.training:
            freqs_cis = freqs_cis[:token_embeddings.shape[1]]
        else:
            freqs_cis = freqs_cis[input_pos]

        # transformer blocks
        for layer in self.layers:
            h = layer(h, freqs_cis, input_pos, mask)
        
        # output layers
        h = self.norm(h)
        logits = self.output(h).float()
        
        if self.training:
            # Extract logits only for idx token positions (skip camera tokens)
            # Pattern: [cond_tokens] [c0, idx0, c1, idx1, ...] -> extract idx0, idx1, ...
            start_idx = self.cls_token_num
            # Select only the idx token positions from interleaved sequence
            logits = logits[:, start_idx:(start_idx + self.block_size * 2):2].contiguous()

        # if we are given some desired targets also calculate the loss
        loss = None
        if valid is not None:
            loss_all = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction='none')
            valid_all = valid[:,None].repeat(1, targets.shape[1]).view(-1)
            loss = (loss_all * valid_all).sum() / max(valid_all.sum(), 1)
        elif targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss
    
    def forward_reference(
        self,
        prev_idx: Optional[torch.Tensor] = None,
        prev_cam_token: Optional[torch.Tensor] = None,
        cur_cam_token: Optional[torch.Tensor] = None,
        cond_idx: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
    ):
        if cond_idx is not None:
            self.start = False
            # cond -> first token
            # input_pos should be [0, 1] assuming cond_idx has only one token
            token_embeddings = self.cls_embedding(cond_idx, train=self.training).unsqueeze(1)
            assert prev_cam_token is None, "Predicting the first token given condition token, no camera token should be given"
            token_embeddings = torch.cat([token_embeddings, cur_cam_token], dim=1)
        
        elif prev_idx.shape[1] > 1 and not self.start:
            # parallel autoregression stage, predict multiple tokens at once
            token_embeddings = self.tok_embeddings(prev_idx)
            # need to interleave with previous camera tokens
            token_embeddings = interleave_tokens(prev_cam_token, token_embeddings)
            interleaved_spe_embeddings = self.spe_tok_embeddings().unsqueeze(0).expand(token_embeddings.shape[0], -1, -1)
            token_embeddings = torch.cat([token_embeddings[:,-2:], interleaved_spe_embeddings], dim=1)
            token_embeddings = torch.cat([token_embeddings, cur_cam_token], dim=1)
            self.start = True
        else:
            # autoregression stage, predict all the ar tokens
            token_embeddings = self.tok_embeddings(prev_idx)
            # need to interleave with previous camera tokens
            token_embeddings = interleave_tokens(prev_cam_token, token_embeddings)
            token_embeddings = torch.cat([token_embeddings, cur_cam_token], dim=1)

        bs = token_embeddings.shape[0]
        mask = self.causal_mask[:bs, None, input_pos]
        h = self.tok_dropout(token_embeddings)

        token_freqs_cis = self.freqs_cis[self.cls_token_num:].clone().to(h.device)
        freqs_cis = torch.cat(
            (self.freqs_cis[:self.cls_token_num].to(h.device),
                interleave_tokens_1d(token_freqs_cis, token_freqs_cis))
        )

        freqs_cis = freqs_cis[input_pos]
        # transformer blocks
        for layer in self.layers:
            h = layer(h, freqs_cis, input_pos, mask)
        
        # output layers
        h = self.norm(h)
        logits = self.output(h).float()

        return logits


    def get_fsdp_wrap_module_list(self) -> List[nn.Module]:
        return list(self.layers)



#################################################################################
#                      Rotary Positional Embedding Functions                    #
#################################################################################
# https://github.com/pytorch-labs/gpt-fast/blob/main/model.py 
def precompute_freqs_cis(seq_len: int, n_elem: int, base: int = 10000, cls_token_num=120):
    freqs = 1.0 / (base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem))
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs) # (seq_len, head_dim // 2)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1) # (cls_token_num+seq_len, head_dim // 2, 2)
    cond_cache = torch.cat([torch.zeros(cls_token_num, n_elem // 2, 2), cache]) # (cls_token_num+seq_len, head_dim // 2, 2)
    return cond_cache 

def precompute_freqs_cis_2d(grid_size: int, n_elem: int, base: int = 10000, cls_token_num=120, spe_token_num=3, ar_token_num=4):
    # split the dimension into half, one for x and one for y
    half_dim = n_elem // 2
    freqs = 1.0 / (base ** (torch.arange(0, half_dim, 2)[: (half_dim // 2)].float() / half_dim))
    t = torch.arange(grid_size, device=freqs.device)
    freqs = torch.outer(t, freqs) # (grid_size, head_dim // 2)
    freqs_grid = torch.concat([
        freqs[:, None, :].expand(-1, grid_size, -1),
        freqs[None, :, :].expand(grid_size, -1, -1),
    ], dim=-1)  # (grid_size, grid_size, head_dim // 2)
    cache_grid = torch.stack([torch.cos(freqs_grid), torch.sin(freqs_grid)], dim=-1) # (grid_size, grid_size, head_dim // 2, 2)
    sub_num = int(ar_token_num**0.5)

    cache_grid = cache_grid.reshape(sub_num, grid_size//sub_num, sub_num, grid_size//sub_num, half_dim, 2)
    cache_grid = cache_grid.permute(1, 3, 0, 2, 4, 5)
    cache = cache_grid.flatten(0, 3)
    cache_one, cache_two = cache[:ar_token_num], cache[ar_token_num:]
    sep_cache = torch.zeros(spe_token_num, n_elem // 2, 2)
    cond_cache = torch.cat([torch.zeros(cls_token_num, n_elem // 2, 2), cache_one, sep_cache, cache_two])
    # cond_cache = torch.cat([torch.zeros(cls_token_num, n_elem // 2, 2), cache]) # (cls_token_num+grid_size**2, head_dim // 2, 2)
    return cond_cache

def precompute_freqs_cis_3d(num_views: int, grid_size: int, n_elem: int, base: int = 10000, cls_token_num=120, spe_token_num=3, ar_token_num=4):
    # split the dimension into half, one for x and one for y
    spatial_dim = n_elem // 8 * 3
    temporal_dim = n_elem // 4

    freqs = 1.0 / (
        base ** (torch.arange(0, spatial_dim, 2)[: (spatial_dim // 2)].float() / spatial_dim)
    )
    view_freqs = 1.0 / (base ** (torch.arange(0, temporal_dim, 2)[: (temporal_dim // 2)].float() / temporal_dim))

    t = torch.arange(grid_size, device=freqs.device)
    view_t = torch.arange(num_views, device=freqs.device)

    freqs = torch.outer(t, freqs) # (grid_size, head_dim // 2)
    view_freqs = torch.outer(view_t, view_freqs) # (num_views, head_dim // 2)
    # print(freqs.shape, view_freqs.shape)

    freqs_grid = torch.concat(
        [
            view_freqs[:, None, None, :].expand(-1, grid_size, grid_size, -1),
            freqs[None, :, None, :].expand(num_views, -1, grid_size, -1),
            freqs[None, None, :, :].expand(num_views, grid_size, -1, -1),
        ],
        dim=-1,
    )  # (num_views, grid_size, grid_size, head_dim // 2)
    
    cache_grid = torch.stack(
        [torch.cos(freqs_grid), torch.sin(freqs_grid)], dim=-1
    )  # (num_views, grid_size, grid_size, head_dim // 2, 2)
    sub_num = int((ar_token_num//num_views)**0.5)
    cache_grid = cache_grid.reshape(num_views, sub_num, grid_size//sub_num, sub_num, grid_size//sub_num, -1, 2)
    cache_grid = cache_grid.permute(2, 4, 0, 1, 3, 5, 6)
    cache = cache_grid.flatten(0, 4)
    cache_one, cache_two = cache[:ar_token_num], cache[ar_token_num:]
    sep_cache = torch.zeros(spe_token_num, n_elem // 2, 2)
    cond_cache = torch.cat([torch.zeros(cls_token_num, n_elem // 2, 2), cache_one, sep_cache, cache_two])
    return cond_cache

def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor):
    # x: (bs, seq_len, n_head, head_dim)
    # freqs_cis (seq_len, head_dim // 2, 2)
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2) # (bs, seq_len, n_head, head_dim//2, 2)
    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2) # (1, seq_len, 1, head_dim//2, 2)
    x_out2 = torch.stack([
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
    ], dim=-1)
    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)

def permute_token(x, args):
    """
    Dynamically permute tokens for 3D autoregressive modeling.
    Supports variable input dimensions beyond [b, t, h, w].
    
    Args:
        x: Input tensor with shape [b, t, h, w, ...] where ... represents optional extra dimensions
        args: Configuration object containing permutation parameters
    
    Returns:
        Permuted tensor with shape [b, -1, ...] where ... preserves any extra dimensions
    """
    if len(x.shape) < 4:
        raise ValueError(f"Input tensor must have at least 4 dimensions, got {len(x.shape)}")
    
    latent_size = args.image_size // args.downsample_size
    sub_num = int((args.ar_token_num//args.temporal_size)**0.5)
    
    # Extract core dimensions and extra dimensions
    batch_size = x.shape[0]
    extra_dims = x.shape[4:] if len(x.shape) > 4 else ()
    
    # Flatten to [b, t*h*w, ...]
    z = x.reshape(batch_size, -1, *extra_dims)
    
    # Reshape to [b, t, h, w, ...]
    z = z.reshape(batch_size, args.temporal_size, latent_size, latent_size, *extra_dims)
    
    # Reshape to [b, t, s, h//s, s, w//s, ...]
    z = z.reshape(batch_size, args.temporal_size, sub_num, latent_size//sub_num, sub_num, latent_size//sub_num, *extra_dims)
    
    # Create dynamic permutation indices
    # Core permutation: [b, t, s, h//s, s, w//s] -> [b, h//s, w//s, t, s, s]
    core_perm = [0, 3, 5, 1, 2, 4]

    # Add indices for extra dimensions (they stay in place after the core permutation)
    extra_perm = list(range(6, 6 + len(extra_dims)))
    perm_indices = core_perm + extra_perm
    
    # Apply permutation
    z = z.permute(*perm_indices) # [b, h//s, w//s, t, s, s]
    
    # Reshape to [b, -1, ...]
    z = z.reshape(batch_size, -1, *extra_dims)
    
    return z

def interleave_tokens(seq1, seq2):
    """ Interleave two sequences """
    result = torch.zeros_like(torch.cat((seq1, seq2), dim=1))
    result[:, ::2] = seq1
    result[:, 1::2] = seq2
    return result

def interleave_tokens_1d(seq1, seq2):
    """ Interleave two sequences """
    result = torch.zeros_like(torch.cat((seq1, seq2), dim=0))
    result[::2] = seq1
    result[1::2] = seq2
    return result


#################################################################################
#                                GPT Configs                                    #
#################################################################################
### text-conditional
def GPT_7B(**kwargs):
    return Transformer(ModelArgs(n_layer=32, n_head=32, dim=4096, **kwargs)) # 6.6B

def GPT_3B(**kwargs):
    return Transformer(ModelArgs(n_layer=24, n_head=32, dim=3200, **kwargs)) # 3.1B

def GPT_1B(**kwargs):
    return Transformer(ModelArgs(n_layer=22, n_head=40, dim=2560, **kwargs)) # 1.2B

### class-conditional
def GPT_XXXL(**kwargs):
    return Transformer(ModelArgs(n_layer=48, n_head=40, dim=2560, **kwargs)) # 3.9B

def GPT_XXL(**kwargs):
    return Transformer(ModelArgs(n_layer=48, n_head=24, dim=1536, **kwargs)) # 1.4B

def GPT_XL(**kwargs):
    return Transformer(ModelArgs(n_layer=36, n_head=20, dim=1280, **kwargs)) # 775M

def GPT_L(**kwargs):
    return Transformer(ModelArgs(n_layer=24, n_head=16, dim=1024, **kwargs)) # 343M

def GPT_B(**kwargs):
    return Transformer(ModelArgs(n_layer=12, n_head=12, dim=768, **kwargs)) # 111M
        

GPT_models = {
    'GPT-B': GPT_B, 'GPT-L': GPT_L, 'GPT-XL': GPT_XL, 'GPT-XXL': GPT_XXL, 'GPT-XXXL': GPT_XXXL,
    'GPT-1B': GPT_1B, 'GPT-3B': GPT_3B, 'GPT-7B': GPT_7B, 
}