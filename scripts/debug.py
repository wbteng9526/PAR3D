import torch

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
    print(cache_grid.shape)
    sub_num = int((ar_token_num//num_views)**0.5)
    cache_grid = cache_grid.reshape(num_views, sub_num, grid_size//sub_num, sub_num, grid_size//sub_num, -1, 2)
    print(cache_grid.shape)
    cache_grid = cache_grid.permute(2, 4, 0, 1, 3, 5, 6)
    print(cache_grid.shape)
    cache = cache_grid.flatten(0, 4)
    print(cache.shape)
    cache_one, cache_two = cache[:ar_token_num], cache[ar_token_num:]
    print(cache_one.shape, cache_two.shape)
    sep_cache = torch.zeros(spe_token_num, n_elem // 2, 2)
    print(sep_cache.shape)
    cond_cache = torch.cat([torch.zeros(cls_token_num, n_elem // 2, 2), cache_one, sep_cache, cache_two])
    return cond_cache

cond_cache = precompute_freqs_cis_3d(4, 32, 64, 10000, 1, 63, 64)
print(cond_cache.shape)



