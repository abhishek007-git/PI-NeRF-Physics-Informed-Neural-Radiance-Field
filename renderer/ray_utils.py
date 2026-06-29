"""
Ray Generation Utilities
========================
Generates camera rays from poses and intrinsics.

Each pixel in the image corresponds to a ray defined by:
  r(t) = o + t·d
where:
  o = camera origin (world space)
  d = normalized ray direction (world space)
  t = depth parameter (t ∈ [near, far])

Camera model: pinhole camera with intrinsic matrix K:
  K = [[f, 0, cx],
       [0, f, cy],
       [0, 0,  1]]

Ray direction for pixel (u, v):
  d_cam = [(u - cx)/f, -(v - cy)/f, -1]  (OpenGL convention, z points into scene)
  d_world = R · d_cam / ||R · d_cam||
"""

import torch
import numpy as np
from typing import Tuple, Optional


def get_rays(
    H: int,
    W: int,
    focal: float,
    c2w: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate rays for all pixels in an image.

    Args:
        H:      Image height (pixels)
        W:      Image width (pixels)
        focal:  Focal length (pixels)
        c2w:    [4, 4] camera-to-world transformation matrix

    Returns:
        rays_o: [H, W, 3]  ray origins (all = camera position)
        rays_d: [H, W, 3]  ray directions (unit vectors, world space)
    """
    device = c2w.device

    # Pixel grid (image plane)
    i, j = torch.meshgrid(
        torch.arange(W, dtype=torch.float32, device=device),
        torch.arange(H, dtype=torch.float32, device=device),
        indexing="xy",
    )
    # i: [H, W] column index, j: [H, W] row index

    # Camera-space directions (OpenGL: x right, y up, z toward viewer)
    dirs = torch.stack([
        (i - W * 0.5) / focal,       # x: horizontal
        -(j - H * 0.5) / focal,      # y: vertical (flipped)
        -torch.ones_like(i),          # z: into scene
    ], dim=-1)  # [H, W, 3]

    # Rotate from camera to world space using rotation part of c2w
    # dirs_world = (c2w[:3,:3] @ dirs[..., None])[..., 0]
    rays_d = (dirs[..., None, :] * c2w[:3, :3]).sum(dim=-1)  # [H, W, 3]
    rays_d = rays_d / torch.linalg.norm(rays_d, dim=-1, keepdim=True)

    # Ray origins: camera position (translation part of c2w)
    rays_o = c2w[:3, 3].expand_as(rays_d)  # [H, W, 3]

    return rays_o, rays_d


def get_rays_batch(
    H: int,
    W: int,
    focal: float,
    c2w_batch: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate rays for a batch of camera poses.

    Args:
        H, W, focal:   Image params
        c2w_batch:     [B, 4, 4] batch of camera-to-world matrices

    Returns:
        rays_o: [B, H, W, 3]
        rays_d: [B, H, W, 3]
    """
    B = c2w_batch.shape[0]
    all_o, all_d = [], []
    for b in range(B):
        o, d = get_rays(H, W, focal, c2w_batch[b])
        all_o.append(o)
        all_d.append(d)
    return torch.stack(all_o), torch.stack(all_d)


def sample_rays_from_image(
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    target_pixels: torch.Tensor,
    n_rays: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Randomly sample N rays from the full image ray grid.

    Args:
        rays_o:        [H, W, 3] all ray origins
        rays_d:        [H, W, 3] all ray directions
        target_pixels: [H, W, 3] ground truth RGB
        n_rays:        Number of rays to sample

    Returns:
        batch_o:    [N, 3]
        batch_d:    [N, 3]
        batch_rgb:  [N, 3]
    """
    H, W = rays_o.shape[:2]
    n_pixels = H * W

    indices = torch.randperm(n_pixels, device=rays_o.device)[:n_rays]

    batch_o = rays_o.reshape(-1, 3)[indices]
    batch_d = rays_d.reshape(-1, 3)[indices]
    batch_rgb = target_pixels.reshape(-1, 3)[indices]

    return batch_o, batch_d, batch_rgb


def ndc_rays(
    H: int,
    W: int,
    focal: float,
    near: float,
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert rays to Normalized Device Coordinates (NDC).
    Used for forward-facing scenes (LLFF dataset).

    From NeRF supplementary: maps the frustum [near, far] × [left, right] × [top, bottom]
    to the unit cube [-1, 1]³, making uniform sampling more effective.
    """
    # Shift ray origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Project to NDC
    o0 = -1.0 / (W / (2.0 * focal)) * rays_o[..., 0] / rays_o[..., 2]
    o1 = -1.0 / (H / (2.0 * focal)) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1.0 + 2.0 * near / rays_o[..., 2]

    d0 = (-1.0 / (W / (2.0 * focal)) *
          (rays_d[..., 0] / rays_d[..., 2] - rays_o[..., 0] / rays_o[..., 2]))
    d1 = (-1.0 / (H / (2.0 * focal)) *
          (rays_d[..., 1] / rays_d[..., 2] - rays_o[..., 1] / rays_o[..., 2]))
    d2 = -2.0 * near / rays_o[..., 2]

    rays_o = torch.stack([o0, o1, o2], dim=-1)
    rays_d = torch.stack([d0, d1, d2], dim=-1)

    return rays_o, rays_d
