"""
Neural Radiance Field (NeRF) MLP
=================================
Implements the core NeRF network from Mildenhall et al. (2020):
  "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis"

Architecture:
  - Input: 3D position x ∈ R³, viewing direction d ∈ R³ (unit vector)
  - Positional encoding applied to both x and d
  - 8-layer MLP with skip connection at layer 4
  - Output: volume density σ ∈ R⁺ and RGB color c ∈ [0,1]³

Key design choices:
  - σ depends ONLY on position (not view direction) → multiview consistency
  - c depends on BOTH position and direction → view-dependent effects (specular)
  - Skip connection prevents gradient vanishing in deep network
  - Softplus activation for σ ensures non-negative density
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

from .encoding import FourierEncoding, HashEncoding, get_encoding


class NeRFMLP(nn.Module):
    """
    Core NeRF MLP.

    Forward pass:
      x → γ(x) → [Linear → ReLU]×depth → σ, feature
      [feature, γ(d)] → [Linear → ReLU]×1 → c (RGB)

    Args:
        pos_enc_levels:  Fourier levels for position (L in paper = 10)
        dir_enc_levels:  Fourier levels for direction (L = 4)
        net_depth:       Number of MLP layers (default 8)
        net_width:       Hidden layer width (default 256)
        skip_connect:    Layers that receive a skip from encoded input
        use_viewdirs:    Whether color depends on view direction
        use_hash_enc:    Use hash encoding instead of Fourier
    """

    def __init__(
        self,
        pos_enc_levels: int = 10,
        dir_enc_levels: int = 4,
        net_depth: int = 8,
        net_width: int = 256,
        skip_connect: List[int] = None,
        use_viewdirs: bool = True,
        use_hash_enc: bool = False,
    ):
        super().__init__()

        self.use_viewdirs = use_viewdirs
        self.skip_connect = skip_connect or [4]
        self.net_depth = net_depth

        # ── Positional encodings ──────────────────────────────────────────
        if use_hash_enc:
            self.pos_encoding = HashEncoding(input_dim=3)
            pos_dim = self.pos_encoding.output_dim
        else:
            self.pos_encoding = FourierEncoding(input_dim=3, n_levels=pos_enc_levels)
            pos_dim = self.pos_encoding.output_dim

        self.dir_encoding = FourierEncoding(input_dim=3, n_levels=dir_enc_levels)
        dir_dim = self.dir_encoding.output_dim

        # ── Position MLP (density + feature) ─────────────────────────────
        layers = []
        in_dim = pos_dim
        for i in range(net_depth):
            if i in self.skip_connect and i > 0:
                in_dim = net_width + pos_dim  # skip connection
            layers.append(nn.Linear(in_dim, net_width))
            in_dim = net_width
        self.pos_layers = nn.ModuleList(layers)

        # Density head: one linear layer → scalar σ
        self.density_head = nn.Linear(net_width, 1)

        # Feature head (passed to color network)
        self.feature_head = nn.Linear(net_width, net_width)

        # ── Color MLP (view-dependent) ────────────────────────────────────
        if use_viewdirs:
            self.color_layer1 = nn.Linear(net_width + dir_dim, net_width // 2)
            self.color_out = nn.Linear(net_width // 2, 3)
        else:
            # View-independent: no direction input
            self.color_out = nn.Linear(net_width, 3)

        # ── Weight initialisation ─────────────────────────────────────────
        self._init_weights()

        # Store dims for reference
        self.pos_dim = pos_dim
        self.dir_dim = dir_dim

    def _init_weights(self):
        """Xavier uniform init for stable training."""
        for layer in self.pos_layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.density_head.weight)
        nn.init.zeros_(self.density_head.bias)

    def forward(
        self,
        positions: torch.Tensor,
        directions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            positions:   [N, 3]  3D positions in world space
            directions:  [N, 3]  unit viewing directions (optional)

        Returns:
            density: [N, 1]  volume density σ (non-negative, via softplus)
            color:   [N, 3]  RGB radiance c ∈ [0, 1]
        """
        # Encode position
        pos_enc = self.pos_encoding(positions)  # [N, pos_dim]
        h = pos_enc

        # Forward through position MLP with skip connections
        for i, layer in enumerate(self.pos_layers):
            if i in self.skip_connect and i > 0:
                h = torch.cat([h, pos_enc], dim=-1)
            h = F.relu(layer(h))

        # Density prediction (softplus for non-negativity + differentiability)
        # Softplus is smoother than ReLU → better gradients for Eikonal loss
        density = F.softplus(self.density_head(h) - 1.0)  # [N, 1], shifted for sparsity

        # Color prediction
        feature = self.feature_head(h)  # [N, net_width]

        if self.use_viewdirs and directions is not None:
            dir_enc = self.dir_encoding(directions)  # [N, dir_dim]
            color_input = torch.cat([feature, dir_enc], dim=-1)
            color = torch.sigmoid(self.color_out(F.relu(self.color_layer1(color_input))))
        else:
            color = torch.sigmoid(self.color_out(feature))

        return density, color  # [N, 1], [N, 3]

    def density_only(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Compute only density (used for importance sampling & physics losses).
        More efficient — skips the color network entirely.
        """
        pos_enc = self.pos_encoding(positions)
        h = pos_enc
        for i, layer in enumerate(self.pos_layers):
            if i in self.skip_connect and i > 0:
                h = torch.cat([h, pos_enc], dim=-1)
            h = F.relu(layer(h))
        return F.softplus(self.density_head(h) - 1.0)  # [N, 1]


class HierarchicalNeRF(nn.Module):
    """
    Two-network hierarchical NeRF (coarse + fine).

    The coarse network gives a rough density estimate along each ray.
    These densities are used as a probability distribution to draw
    additional samples (importance sampling) for the fine network.

    This concentrates compute on regions that actually matter
    (high density / object surfaces), giving ~8–15× better
    sample efficiency vs uniform sampling.
    """

    def __init__(self, cfg: dict):
        super().__init__()

        # Coarse network
        self.coarse = NeRFMLP(
            pos_enc_levels=cfg.get("pos_enc_levels", 6),
            dir_enc_levels=cfg.get("dir_enc_levels", 4),
            net_depth=cfg.get("net_depth", 4),
            net_width=cfg.get("net_width", 128),
            skip_connect=cfg.get("skip_connect", [2]),
            use_viewdirs=cfg.get("use_viewdirs", True),
            use_hash_enc=cfg.get("use_hash_encoding", False),
        )

        # Fine network (separate weights, same architecture)
        self.fine = NeRFMLP(
            pos_enc_levels=cfg.get("pos_enc_levels", 6),
            dir_enc_levels=cfg.get("dir_enc_levels", 4),
            net_depth=cfg.get("fine_net_depth", cfg.get("net_depth", 4)),
            net_width=cfg.get("fine_net_width", cfg.get("net_width", 128)),
            skip_connect=cfg.get("skip_connect", [2]),
            use_viewdirs=cfg.get("use_viewdirs", True),
            use_hash_enc=cfg.get("use_hash_encoding", False),
        )

    def forward_coarse(
        self,
        positions: torch.Tensor,
        directions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.coarse(positions, directions)

    def forward_fine(
        self,
        positions: torch.Tensor,
        directions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.fine(positions, directions)

    def density_coarse(self, positions: torch.Tensor) -> torch.Tensor:
        return self.coarse.density_only(positions)

    def get_param_count(self) -> dict:
        coarse_params = sum(p.numel() for p in self.coarse.parameters())
        fine_params = sum(p.numel() for p in self.fine.parameters())
        return {
            "coarse": coarse_params,
            "fine": fine_params,
            "total": coarse_params + fine_params,
        }
