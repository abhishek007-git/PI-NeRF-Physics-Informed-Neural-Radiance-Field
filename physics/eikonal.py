"""
Eikonal Constraint
==================
The Eikonal equation is a PDE that characterizes valid Signed Distance Functions (SDFs):

  |∇σ(x)| = 1   ∀ x ∈ Ω

where σ(x) is the density field (interpreted as an SDF here).

Physical meaning:
  A function f satisfying |∇f| = 1 everywhere represents the distance to the nearest
  surface. Enforcing this on the NeRF density field ensures that σ behaves like a
  proper distance function, leading to:
    - Sharper, cleaner object surfaces
    - No spurious floaters (random opaque regions in empty space)
    - Geometrically consistent representations

Loss formulation:
  L_eikonal = E_x [ (|∇σ(x)| - 1)² ]

Gradient computation:
  We use torch.autograd.grad to compute ∇σ(x) = ∂σ/∂x.
  This requires σ to be differentiable w.r.t. x (which it is, as a neural network).
  We must set create_graph=True to allow second-order optimization if needed.

References:
  - Gropp et al. (2020) "Implicit Geometric Regularization for Learning Shapes"
  - Yariv et al. (2021) "Volume Rendering of Neural Implicit Surfaces" (NeuS)
"""

import torch
import torch.nn as nn
from typing import Tuple


def eikonal_loss(
    model,
    n_samples: int = 512,
    xyz_min: float = -1.5,
    xyz_max: float = 1.5,
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, dict]:
    """
    Compute the Eikonal regularization loss.

    Samples random 3D points in the scene volume and penalizes
    deviations of |∇σ| from 1.

    L_eikonal = (1/N) Σᵢ (|∇σ(xᵢ)| - 1)²

    Args:
        model:     NeRFMLP or HierarchicalNeRF (will use fine network)
        n_samples: Number of random points to sample
        xyz_min:   Min coordinate of sampling volume
        xyz_max:   Max coordinate of sampling volume
        device:    Compute device

    Returns:
        loss:   Scalar Eikonal loss
        info:   Dict with diagnostics (mean gradient norm, etc.)
    """
    # Sample random 3D points with gradient tracking
    pts = torch.empty(n_samples, 3, device=device).uniform_(xyz_min, xyz_max)
    pts.requires_grad_(True)

    # Forward pass — compute density
    if hasattr(model, "fine"):
        # HierarchicalNeRF: use fine network for physics
        density = model.fine.density_only(pts)  # [N, 1]
    else:
        density = model.density_only(pts)  # [N, 1]

    # Compute spatial gradient ∂σ/∂x via autograd
    # create_graph=True allows gradients through the gradient (for 2nd-order methods)
    grad_outputs = torch.ones_like(density)
    gradients = torch.autograd.grad(
        outputs=density,
        inputs=pts,
        grad_outputs=grad_outputs,
        create_graph=True,      # Needed for gradient to flow through this loss
        retain_graph=True,
        only_inputs=True,
    )[0]  # [N, 3]

    # Gradient norm: |∇σ(x)|
    grad_norm = gradients.norm(dim=-1)  # [N]

    # Eikonal loss: (|∇σ| - 1)²
    loss = ((grad_norm - 1.0) ** 2).mean()

    # Diagnostics
    info = {
        "eikonal_loss": loss.item(),
        "grad_norm_mean": grad_norm.mean().item(),
        "grad_norm_std": grad_norm.std().item(),
        "grad_norm_max": grad_norm.max().item(),
    }

    return loss, info


def eikonal_loss_on_surface(
    model,
    surface_pts: torch.Tensor,
) -> Tuple[torch.Tensor, dict]:
    """
    Compute Eikonal loss specifically on sampled surface points.
    (More targeted than random volume sampling — focuses on where it matters most.)

    Args:
        model:       NeRFMLP
        surface_pts: [N, 3] points on or near the surface (e.g., high-density regions)

    Returns:
        loss, info dict
    """
    pts = surface_pts.clone().requires_grad_(True)

    if hasattr(model, "fine"):
        density = model.fine.density_only(pts)
    else:
        density = model.density_only(pts)

    gradients = torch.autograd.grad(
        outputs=density,
        inputs=pts,
        grad_outputs=torch.ones_like(density),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]  # [N, 3]

    grad_norm = gradients.norm(dim=-1)
    loss = ((grad_norm - 1.0) ** 2).mean()

    return loss, {
        "eikonal_surface_loss": loss.item(),
        "surface_grad_norm_mean": grad_norm.mean().item(),
    }
