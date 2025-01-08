import numpy as np
try:
    import torch
    import torch.nn.functional as F
except:
    pass


def coords_grid(batch, ht, wd):
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


def bilinear_interpolate_numpy(img, grid, default_val=0.0):
    H, W, _ = img.shape

    x, y = grid[..., 0], grid[..., 1]
    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    # Clip the coordinates that are out-of-bounds
    x0 = np.clip(x0, 0, W-1)
    x1 = np.clip(x1, 0, W-1)
    y0 = np.clip(y0, 0, H-1)
    y1 = np.clip(y1, 0, H-1)

    # Retrieve pixel values
    Ia = img[y0, x0]
    Ib = img[y1, x0]
    Ic = img[y0, x1]
    Id = img[y1, x1]

    # Bilinear interpolation weights
    new_img = (x1-x)[...,None] * (y1-y)[...,None] * Ia \
            + (x1-x)[...,None] * (y-y0)[...,None] * Ib \
            + (x-x0)[...,None] * (y1-y)[...,None] * Ic \
            + (x-x0)[...,None] * (y-y0)[...,None] * Id

    out_of_bound_x = np.logical_or(x < 0, x > W-1)
    out_of_bound_y = np.logical_or(y < 0, y > H-1)
    new_img[out_of_bound_x] = default_val
    new_img[out_of_bound_y] = default_val

    return new_img


def flow_consistency_numpy(forward, backward):
    # Input:
    #     forward: HxWx2 numpy
    #     backward: HxWx2 numpy
    H, W, C = forward.shape
    x, y = np.meshgrid(np.arange(W), np.arange(H))
    coords = np.stack((x, y), axis=-1)

    warped_coords = coords + forward
    interpolated_backward_flow = bilinear_interpolate_numpy(backward, warped_coords)
    consistency = forward + interpolated_backward_flow
    cycle_consistency_error = np.linalg.norm(consistency, axis=2)
    non_occlusion = cycle_consistency_error < 1

    return non_occlusion


def flow_consistency_torch(forward, backward):
    # Input:
    #     forward: HxWx2 numpy
    #     backward: HxWx2 numpy
    H, W, _ = forward.shape
    coords = coords_grid(1, H, W).float().contiguous()
    forward = torch.from_numpy(forward)[None].permute([0, 3, 1, 2]).float()
    backward = torch.from_numpy(backward)[None].permute([0, 3, 1, 2]).float()
    grid = (forward + coords).permute([0, 2, 3, 1]).contiguous()
    grid[:, :, :, 0] = (grid[:, :, :, 0] * 2 - W + 1) / (W - 1)
    grid[:, :, :, 1] = (grid[:, :, :, 1] * 2 - H + 1) / (H - 1)
    backward = F.grid_sample(backward, grid, padding_mode='zeros', align_corners=True)

    consistency = forward + backward
    consistency = consistency[0].permute([1,2,0])
    valid = torch.norm(consistency, dim=2) < 1
    valid = torch.unsqueeze(valid, dim=2).to(dtype=torch.float32, device='cpu').numpy()
    return valid



