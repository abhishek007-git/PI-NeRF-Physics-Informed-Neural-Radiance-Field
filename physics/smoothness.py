"""
Laplacian Smoothness Regularization
=====================================
Penalizes large second-order variations in the density field.

Loss: L_smooth = E_x [ clamp(∇²σ(x), -C, C)² ]
Clamping ensures numerical stability during early training.
"""

import torch
from typing import Tuple


def laplacian_loss_finite_diff(
    model,
    n_samples: int = 256,
    epsilon: float = 0.05,
    xyz_min: float = -1.5,
    xyz_max: float = 1.5,
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, dict]:
    """
    Laplacian smoothness via finite differences (CPU-efficient).

    For each axis i ∈ {x, y, z}:
        ∂²σ/∂xᵢ² ≈ [σ(x + εeᵢ) + σ(x - εeᵢ) - 2σ(x)] / ε²

    Total Laplacian: ∇²σ ≈ Σᵢ ∂²σ/∂xᵢ²
    Loss: L = mean(clamp(∇²σ, -C, C)²)
    """
    pts = torch.rand(n_samples, 3, device=device) * (xyz_max - xyz_min) + xyz_min
    e = torch.eye(3, device=device) * epsilon

    if hasattr(model, "fine"):
        sigma_fn = model.fine.density_only
    else:
        sigma_fn = model.density_only

    with torch.no_grad():
        sigma_0 = sigma_fn(pts).squeeze(-1)   # [N]

    laplacian = torch.zeros(n_samples, device=device)

    for i in range(3):
        with torch.no_grad():
            sp = sigma_fn(pts + e[i].unsqueeze(0)).squeeze(-1)
            sm = sigma_fn(pts - e[i].unsqueeze(0)).squeeze(-1)
        d2 = (sp + sm - 2 * sigma_0) / (epsilon ** 2)
        laplacian = laplacian + d2

    # Clamp before squaring to prevent blow-up
    laplacian = laplacian.clamp(-100.0, 100.0)
    loss = (laplacian ** 2).mean()

    info = {
        "laplacian_loss": loss.item(),
        "laplacian_mean": laplacian.abs().mean().item(),
        "laplacian_max":  laplacian.abs().max().item(),
    }
    return loss, info


def laplacian_loss_autograd(
    model,
    n_samples: int = 128,
    xyz_min: float = -1.5,
    xyz_max: float = 1.5,
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, dict]:
    """
    True second-order Laplacian via double autograd (expensive, GPU-only recommended).
    """
    pts = torch.rand(n_samples, 3, device=device, requires_grad=True)
    pts_scaled = pts * (xyz_max - xyz_min) + xyz_min

    if hasattr(model, "fine"):
        sigma = model.fine.density_only(pts_scaled)
    else:
        sigma = model.density_only(pts_scaled)

    grad = torch.autograd.grad(
        outputs=sigma, inputs=pts,
        grad_outputs=torch.ones_like(sigma),
        create_graph=True, retain_graph=True,
    )[0]   # [N, 3]

    laplacian = torch.zeros(n_samples, device=device)
    for i in range(3):
        g_i = torch.autograd.grad(
            outputs=grad[:, i], inputs=pts,
            grad_outputs=torch.ones(n_samples, device=device),
            create_graph=True, retain_graph=True,
        )[0][:, i]
        laplacian = laplacian + g_i

    loss = (laplacian.clamp(-100, 100) ** 2).mean()
    return loss, {"laplacian_loss": loss.item()}
