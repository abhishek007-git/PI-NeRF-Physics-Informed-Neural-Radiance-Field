"""
Differentiable Volumetric Renderer
====================================
Implements the rendering equation from NeRF / classical volume rendering:

  C(r) = ∫[t_n to t_f] T(t) · σ(r(t)) · c(r(t), d) dt

where:
  T(t) = exp(-∫[t_n to t] σ(r(s)) ds)   — Beer-Lambert transmittance
  σ(x) — volume density (opacity per unit length)
  c(x, d) — emitted radiance at position x in direction d

Discretized form (numerical quadrature):
  C(r) ≈ Σᵢ Tᵢ · (1 - exp(-σᵢ · δᵢ)) · cᵢ

where:
  δᵢ = tᵢ₊₁ - tᵢ         — distance between adjacent samples
  Tᵢ = exp(-Σⱼ<ᵢ σⱼ · δⱼ) — accumulated transmittance up to sample i
  αᵢ = 1 - exp(-σᵢ · δᵢ) — alpha compositing weight

This is the standard alpha compositing formula from computer graphics,
derived from the Beer-Lambert law of light attenuation.

References:
  - Mildenhall et al. (2020) "NeRF: Representing Scenes as Neural Radiance Fields"
  - Max (1995) "Optical models for direct volume rendering"
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict, Optional


def compute_transmittance(
    sigmas: torch.Tensor,
    deltas: torch.Tensor,
) -> torch.Tensor:
    """
    Compute Beer-Lambert transmittance T(t).

    T(t) = exp(-∫₀ᵗ σ(s) ds) ≈ exp(-Σⱼ<ᵢ σⱼ · δⱼ)

    Args:
        sigmas: [B, N]  volume density at each sample
        deltas: [B, N]  distances between samples (δᵢ = tᵢ₊₁ - tᵢ)

    Returns:
        transmittance: [B, N]  accumulated transmittance at each sample
    """
    # Accumulated optical depth up to (but not including) each sample
    # T_i = exp(-sum_{j<i} sigma_j * delta_j)
    optical_depth = sigmas * deltas  # [B, N]

    # Exclusive cumsum: sum of all previous samples
    # Using roll trick: cumsum then shift right
    cumsum = torch.cumsum(optical_depth, dim=-1)  # [B, N]
    # Shift right by 1 (T_0 = 1, T_i = exp(-sum_{j<i}))
    exclusive_cumsum = torch.cat([
        torch.zeros_like(cumsum[..., :1]),  # T_0 = exp(0) = 1
        cumsum[..., :-1],
    ], dim=-1)  # [B, N]

    transmittance = torch.exp(-exclusive_cumsum)  # [B, N]
    return transmittance


def volume_render(
    colors: torch.Tensor,
    densities: torch.Tensor,
    t_vals: torch.Tensor,
    rays_d: torch.Tensor,
    white_background: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Differentiable volumetric renderer.

    Discretized rendering equation:
        αᵢ = 1 - exp(-σᵢ · δᵢ)           (alpha)
        Tᵢ = Π_{j<i} (1 - αⱼ)            (transmittance)
        wᵢ = Tᵢ · αᵢ                     (rendering weight)
        C  = Σᵢ wᵢ · cᵢ                  (rendered color)

    Args:
        colors:           [B, N, 3]  RGB at each sample point
        densities:        [B, N, 1]  volume density σ at each sample
        t_vals:           [B, N]     depth values
        rays_d:           [B, 3]     ray directions (for depth computation)
        white_background: Composite over white background

    Returns:
        dict with:
          'rgb':          [B, 3]  rendered color
          'depth':        [B]     expected depth
          'acc':          [B]     accumulated opacity (0 = background, 1 = solid)
          'weights':      [B, N]  per-sample rendering weights (for importance sampling)
          'transmittance':[B, N]  accumulated transmittance
    """
    densities = densities.squeeze(-1)  # [B, N]

    # Step sizes: δᵢ = tᵢ₊₁ - tᵢ
    deltas = t_vals[..., 1:] - t_vals[..., :-1]  # [B, N-1]
    # Last delta is "infinity" (ray continues to background)
    deltas = torch.cat([
        deltas,
        torch.full_like(deltas[..., :1], fill_value=1e10),
    ], dim=-1)  # [B, N]

    # Alpha compositing weights
    # αᵢ = 1 - exp(-σᵢ · δᵢ)
    alphas = 1.0 - torch.exp(-densities * deltas)  # [B, N]

    # Transmittance: Tᵢ = exp(-Σⱼ<ᵢ σⱼδⱼ) = Πⱼ<ᵢ (1 - αⱼ)
    # We use the log-space version for numerical stability
    transmittance = compute_transmittance(densities, deltas)  # [B, N]

    # Rendering weights: wᵢ = Tᵢ · αᵢ
    weights = transmittance * alphas  # [B, N]

    # Rendered color: C = Σᵢ wᵢ · cᵢ
    rgb = (weights.unsqueeze(-1) * colors).sum(dim=-2)  # [B, 3]

    # Accumulated opacity: Σᵢ wᵢ (should → 1 for opaque surfaces)
    acc = weights.sum(dim=-1)  # [B]

    # Expected depth: Σᵢ wᵢ · tᵢ
    depth = (weights * t_vals).sum(dim=-1)  # [B]

    # White background compositing
    if white_background:
        rgb = rgb + (1.0 - acc.unsqueeze(-1))  # Add white where transparent

    return {
        "rgb": rgb,              # [B, 3]
        "depth": depth,          # [B]
        "acc": acc,              # [B]
        "weights": weights,      # [B, N]  ← used for importance sampling
        "transmittance": transmittance,  # [B, N]
        "alphas": alphas,        # [B, N]
    }


class VolumeRenderer(nn.Module):
    """
    Full hierarchical volume renderer combining:
    1. Coarse pass: stratified sampling → coarse render
    2. Fine pass:   importance sampling → fine render

    Both coarse and fine predictions contribute to the total loss.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.near = cfg.get("near", 2.0)
        self.far = cfg.get("far", 6.0)
        self.n_coarse = cfg.get("n_coarse", 32)
        self.n_fine = cfg.get("n_fine", 64)
        self.perturb = cfg.get("perturb", True)
        self.white_bg = cfg.get("white_background", True)

    def render_rays(
        self,
        nerf_model,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        training: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Full hierarchical rendering pass for a batch of rays.

        Args:
            nerf_model: HierarchicalNeRF instance
            rays_o:     [B, 3] ray origins
            rays_d:     [B, 3] ray directions
            training:   Whether in training mode (controls perturb)

        Returns:
            dict with coarse and fine rendering outputs
        """
        from renderer.sampler import (
            stratified_sample, importance_sample,
            combine_and_sort_samples, get_sample_points
        )

        B = rays_o.shape[0]
        device = rays_o.device
        perturb = self.perturb and training

        # ── Coarse pass ───────────────────────────────────────────────────
        t_coarse = stratified_sample(
            self.near, self.far, self.n_coarse,
            n_rays=B, perturb=perturb, device=device,
        )  # [B, Nc]

        pts_coarse = get_sample_points(rays_o, rays_d, t_coarse)  # [B, Nc, 3]

        # Expand directions for each sample
        dirs_coarse = rays_d.unsqueeze(-2).expand_as(pts_coarse)  # [B, Nc, 3]

        # Flatten for network forward pass
        density_c, color_c = nerf_model.forward_coarse(
            pts_coarse.reshape(-1, 3),
            dirs_coarse.reshape(-1, 3),
        )
        density_c = density_c.reshape(B, self.n_coarse, 1)
        color_c = color_c.reshape(B, self.n_coarse, 3)

        render_coarse = volume_render(
            color_c, density_c, t_coarse, rays_d,
            white_background=self.white_bg,
        )

        # ── Fine pass (importance sampling) ──────────────────────────────
        t_fine = importance_sample(
            t_coarse,
            render_coarse["weights"].detach(),  # Stop gradient through sampling
            n_fine=self.n_fine,
            perturb=perturb,
        )  # [B, Nf]

        t_fine_combined = combine_and_sort_samples(t_coarse, t_fine)  # [B, Nc+Nf]
        N_fine_total = t_fine_combined.shape[-1]

        pts_fine = get_sample_points(rays_o, rays_d, t_fine_combined)  # [B, Nc+Nf, 3]
        dirs_fine = rays_d.unsqueeze(-2).expand_as(pts_fine)

        density_f, color_f = nerf_model.forward_fine(
            pts_fine.reshape(-1, 3),
            dirs_fine.reshape(-1, 3),
        )
        density_f = density_f.reshape(B, N_fine_total, 1)
        color_f = color_f.reshape(B, N_fine_total, 3)

        render_fine = volume_render(
            color_f, density_f, t_fine_combined, rays_d,
            white_background=self.white_bg,
        )

        return {
            # Fine (primary) outputs
            "rgb": render_fine["rgb"],
            "depth": render_fine["depth"],
            "acc": render_fine["acc"],
            "weights_fine": render_fine["weights"],

            # Coarse outputs (for auxiliary loss)
            "rgb_coarse": render_coarse["rgb"],
            "depth_coarse": render_coarse["depth"],
            "weights_coarse": render_coarse["weights"],

            # Sample points (needed for physics loss gradients)
            "pts_coarse": pts_coarse,
            "pts_fine": pts_fine,
            "t_coarse": t_coarse,
            "t_fine": t_fine_combined,
        }

    def render_image(
        self,
        nerf_model,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        chunk_size: int = 512,
    ) -> Dict[str, torch.Tensor]:
        """
        Render a full image by processing rays in chunks (memory-efficient).

        Args:
            nerf_model:  HierarchicalNeRF
            rays_o:      [H*W, 3]
            rays_d:      [H*W, 3]
            chunk_size:  Rays processed at once (reduce if OOM)

        Returns:
            Full-image render outputs
        """
        all_outputs = {}
        N = rays_o.shape[0]

        for i in range(0, N, chunk_size):
            chunk_o = rays_o[i: i + chunk_size]
            chunk_d = rays_d[i: i + chunk_size]

            with torch.no_grad():
                chunk_out = self.render_rays(nerf_model, chunk_o, chunk_d, training=False)

            for k, v in chunk_out.items():
                if k not in all_outputs:
                    all_outputs[k] = []
                all_outputs[k].append(v)

        # Concatenate all chunks
        return {
            k: torch.cat(v, dim=0)
            for k, v in all_outputs.items()
            if v[0].dim() >= 1 and k in ["rgb", "depth", "acc", "rgb_coarse"]
        }
