"""
Physics-Informed Neural Network (PINN) Loss
============================================
Combines rendering loss with physics-based constraints.

Total loss:
  L_total = L_render + λ_e · L_eikonal + λ_s · L_smooth + λ_c · L_coarse

where:
  L_render   = MSE(C_fine, C_gt)           — photometric reconstruction
  L_coarse   = MSE(C_coarse, C_gt)         — auxiliary coarse supervision
  L_eikonal  = E[(|∇σ| - 1)²]             — SDF validity constraint
  L_smooth   = E[(∇²σ)²]                  — density smoothness
  λ_e, λ_s   = loss weights (from config)

The physics losses act as inductive biases that:
  1. Prevent the network from learning degenerate density fields
  2. Encode geometric priors (surfaces should have SDF-like structure)
  3. Improve generalization to novel views

Loss weighting strategy:
  - Physics losses are warmed up gradually (not applied at step 0)
  - This lets the rendering loss first establish a rough geometry
  - Physics constraints then refine the geometry
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple

from physics.eikonal import eikonal_loss
from physics.smoothness import laplacian_loss_finite_diff


class PINNLoss(nn.Module):
    """
    Combined rendering + physics loss for PI-NeRF training.
    """

    def __init__(self, cfg: dict):
        super().__init__()

        self.use_eikonal = cfg.get("use_eikonal", True)
        self.use_laplacian = cfg.get("use_laplacian", True)
        self.lambda_eikonal = cfg.get("lambda_eikonal", 0.1)
        self.lambda_laplacian = cfg.get("lambda_laplacian", 0.01)
        self.n_physics_samples = cfg.get("n_physics_samples", 128)
        self.physics_warmup_steps = cfg.get("physics_warmup_steps", 200)

        self.render_loss = nn.MSELoss()

    def physics_weight(self, step: int) -> float:
        """
        Linear warmup for physics losses.
        Returns 0 before warmup, then linearly ramps to 1.
        """
        if step < self.physics_warmup_steps:
            return float(step) / float(self.physics_warmup_steps)
        return 1.0

    def forward(
        self,
        render_output: Dict[str, torch.Tensor],
        target_rgb: torch.Tensor,
        model,
        step: int = 0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined PI-NeRF loss.

        Args:
            render_output: Output dict from VolumeRenderer.render_rays()
            target_rgb:    [B, 3] ground truth pixel colors
            model:         HierarchicalNeRF (for physics gradient queries)
            step:          Training step (for warmup)

        Returns:
            total_loss: Scalar loss tensor
            loss_dict:  Component losses for logging
        """
        device = target_rgb.device
        loss_dict = {}

        # ── Rendering losses ──────────────────────────────────────────────
        # Fine network (primary)
        l_fine = self.render_loss(render_output["rgb"], target_rgb)
        loss_dict["loss_render_fine"] = l_fine.item()

        # Coarse network (auxiliary supervision)
        l_coarse = self.render_loss(render_output["rgb_coarse"], target_rgb)
        loss_dict["loss_render_coarse"] = l_coarse.item()

        total_loss = l_fine + 0.5 * l_coarse

        # ── Physics losses (warmed up) ────────────────────────────────────
        phys_w = self.physics_weight(step)
        loss_dict["physics_weight"] = phys_w

        if phys_w > 0:
            if self.use_eikonal:
                l_eik, eik_info = eikonal_loss(
                    model,
                    n_samples=self.n_physics_samples,
                    device=device,
                )
                weighted_eik = self.lambda_eikonal * phys_w * l_eik
                total_loss = total_loss + weighted_eik
                loss_dict["loss_eikonal"] = l_eik.item()
                loss_dict["loss_eikonal_weighted"] = weighted_eik.item()
                loss_dict.update(eik_info)

            if self.use_laplacian:
                l_lap, lap_info = laplacian_loss_finite_diff(
                    model,
                    n_samples=max(64, self.n_physics_samples // 2),
                    device=device,
                )
                weighted_lap = self.lambda_laplacian * phys_w * l_lap
                total_loss = total_loss + weighted_lap
                loss_dict["loss_laplacian"] = l_lap.item()
                loss_dict["loss_laplacian_weighted"] = weighted_lap.item()
                loss_dict.update(lap_info)

        loss_dict["loss_total"] = total_loss.item()

        return total_loss, loss_dict
