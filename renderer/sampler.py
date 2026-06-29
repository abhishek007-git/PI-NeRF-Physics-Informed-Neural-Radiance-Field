"""
Ray Sampling Strategies
========================
Implements two sampling strategies used in hierarchical NeRF:

1. Stratified Sampling (coarse pass)
   - Divide [near, far] into N equal bins
   - Sample uniformly within each bin (with optional jitter)
   - Ensures coverage of the full ray

2. Importance Sampling (fine pass)
   - Use coarse network density as a PDF over t
   - Draw additional samples proportional to density
   - Focuses fine network compute on surfaces

Mathematical basis:
  The rendering integral C(r) = ∫T(t)σ(t)c(t)dt is approximated via
  quadrature. Importance sampling reduces variance:
      E[f(t)/p(t)] ≈ (1/N) Σ f(tᵢ)/p(tᵢ)   where tᵢ ~ p
"""

import torch
import torch.nn.functional as F
from typing import Tuple


def stratified_sample(
    near: float,
    far: float,
    n_samples: int,
    n_rays: int,
    perturb: bool = True,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Stratified uniform sampling along rays.

    Divides [near, far] into n_samples equal bins, then samples one
    point uniformly within each bin. With perturb=True this becomes
    a continuous approximation of the rendering integral.

    Args:
        near:      Near plane distance
        far:       Far plane distance
        n_samples: Number of samples per ray
        n_rays:    Number of rays
        perturb:   Add random jitter within each bin (True during training)
        device:    Compute device

    Returns:
        t_vals: [n_rays, n_samples]  depth values along rays
    """
    # Evenly spaced bins in [0, 1], then mapped to [near, far]
    t_vals = torch.linspace(0.0, 1.0, n_samples, device=device)  # [N]
    t_vals = near + t_vals * (far - near)                          # [N]
    t_vals = t_vals.expand(n_rays, n_samples)                      # [B, N]

    if perturb:
        # Jitter: sample uniformly within each bin
        mids = 0.5 * (t_vals[..., 1:] + t_vals[..., :-1])  # midpoints [B, N-1]
        upper = torch.cat([mids, t_vals[..., -1:]], dim=-1)
        lower = torch.cat([t_vals[..., :1], mids], dim=-1)
        t_rand = torch.rand_like(t_vals)
        t_vals = lower + (upper - lower) * t_rand

    return t_vals  # [B, N]


def importance_sample(
    t_coarse: torch.Tensor,
    weights_coarse: torch.Tensor,
    n_fine: int,
    perturb: bool = True,
) -> torch.Tensor:
    """
    Importance sampling using coarse density weights as a PDF.

    The coarse weights wᵢ = Tᵢ · (1 - exp(-σᵢδᵢ)) define a piecewise-constant
    PDF over t. We invert this CDF to draw samples where density is high.

    Args:
        t_coarse:      [B, Nc]   coarse sample depths
        weights_coarse:[B, Nc]   coarse rendering weights (sum ≈ 1)
        n_fine:        Number of fine samples to draw per ray
        perturb:       Add uniform noise to avoid discretization artifacts

    Returns:
        t_fine:        [B, n_fine]  importance-sampled depths
    """
    # Normalize weights to form a valid PDF (add eps for numerical stability)
    weights = weights_coarse + 1e-5  # [B, Nc]
    pdf = weights / weights.sum(dim=-1, keepdim=True)  # [B, Nc]
    cdf = torch.cumsum(pdf, dim=-1)  # [B, Nc]
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)  # [B, Nc+1]

    # Draw uniform samples from [0, 1]
    if perturb:
        u = torch.rand((*cdf.shape[:-1], n_fine), device=cdf.device)
    else:
        # Deterministic: evenly spaced
        u = torch.linspace(0.0, 1.0, n_fine, device=cdf.device)
        u = u.expand(*cdf.shape[:-1], n_fine)

    u = u.contiguous()

    # Invert CDF via binary search
    inds = torch.searchsorted(cdf.detach(), u, right=True)  # [B, n_fine]
    below = torch.clamp(inds - 1, min=0)
    above = torch.clamp(inds, max=cdf.shape[-1] - 1)
    inds_g = torch.stack([below, above], dim=-1)  # [B, n_fine, 2]

    # Gather CDF and t values at bin boundaries
    matched_shape = (*inds_g.shape[:-1], cdf.shape[-1])
    cdf_g = torch.gather(
        cdf.unsqueeze(-2).expand(matched_shape),
        dim=-1,
        index=inds_g,
    )  # [B, n_fine, 2]

    # t_coarse has Nc bins; CDF has Nc+1 entries → clamp bin indices to Nc-1
    bins = t_coarse  # [B, Nc]
    bins_inds = torch.clamp(inds_g, max=bins.shape[-1] - 1)
    bins_g = torch.gather(
        bins.unsqueeze(-2).expand(*bins_inds.shape[:-1], bins.shape[-1]),
        dim=-1,
        index=bins_inds,
    )  # [B, n_fine, 2]

    # Linear interpolation within each bin
    denom = cdf_g[..., 1] - cdf_g[..., 0]  # [B, n_fine]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)

    t_fine = bins_g[..., 0] + (u - cdf_g[..., 0]) / denom * (
        bins_g[..., 1] - bins_g[..., 0]
    )  # [B, n_fine]

    return t_fine


def combine_and_sort_samples(
    t_coarse: torch.Tensor,
    t_fine: torch.Tensor,
) -> torch.Tensor:
    """
    Merge coarse and fine samples along each ray and sort by depth.

    Args:
        t_coarse: [B, Nc]
        t_fine:   [B, Nf]

    Returns:
        t_combined: [B, Nc + Nf] sorted
    """
    t_combined = torch.cat([t_coarse, t_fine.detach()], dim=-1)
    t_combined, _ = torch.sort(t_combined, dim=-1)
    return t_combined


def get_sample_points(
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    t_vals: torch.Tensor,
) -> torch.Tensor:
    """
    Compute 3D sample positions along each ray.

    r(t) = o + t·d

    Args:
        rays_o: [B, 3]    ray origins
        rays_d: [B, 3]    ray directions
        t_vals: [B, N]    depth values

    Returns:
        pts:    [B, N, 3] 3D positions
    """
    pts = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * t_vals.unsqueeze(-1)
    return pts  # [B, N, 3]
