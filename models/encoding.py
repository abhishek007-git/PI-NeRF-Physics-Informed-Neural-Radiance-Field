"""
Positional Encoding Module
==========================
Implements two encoding strategies:
  1. Fourier Feature Encoding (original NeRF, Mildenhall et al. 2020)
     γ(p) = [sin(2⁰πp), cos(2⁰πp), ..., sin(2^(L-1)πp), cos(2^(L-1)πp)]

  2. Multi-Resolution Hash Encoding (Müller et al. 2022 / Instant-NGP style)
     - Lightweight, fast to evaluate on CPU
     - Learnable hash table at multiple resolutions

Mathematical motivation:
  MLPs have a "spectral bias" — they learn low-frequency functions first
  (Rahaman et al. 2019). Fourier encoding lifts inputs into high-frequency
  space, allowing the MLP to fit sharp edges and fine details.
"""

import torch
import torch.nn as nn
import numpy as np


class FourierEncoding(nn.Module):
    """
    Sinusoidal positional encoding from NeRF (Mildenhall et al. 2020).

    For input x ∈ R^d and L frequency levels:
        γ(x) = [x, sin(2⁰πx), cos(2⁰πx), ..., sin(2^(L-1)πx), cos(2^(L-1)πx)]

    Output dimension: d + 2*d*L
    """

    def __init__(self, input_dim: int, n_levels: int = 10, include_input: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.n_levels = n_levels
        self.include_input = include_input

        # Frequency bands: 2^0, 2^1, ..., 2^(L-1)
        freq_bands = 2.0 ** torch.linspace(0, n_levels - 1, n_levels)
        self.register_buffer("freq_bands", freq_bands)  # [L]

        # Compute output dimension
        self.output_dim = 2 * input_dim * n_levels
        if include_input:
            self.output_dim += input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., input_dim]
        Returns:
            encoded: [..., output_dim]
        """
        # x: [..., D], freq_bands: [L]
        # Outer product: [..., D, L]
        x_freq = x.unsqueeze(-1) * self.freq_bands * np.pi  # [..., D, L]

        # sin and cos for each frequency: [..., D, L]
        encoded = torch.cat([torch.sin(x_freq), torch.cos(x_freq)], dim=-1)  # [..., D, 2L]
        encoded = encoded.flatten(start_dim=-2)  # [..., D*2L]

        if self.include_input:
            encoded = torch.cat([x, encoded], dim=-1)  # [..., D + D*2L]

        return encoded

    def __repr__(self):
        return (f"FourierEncoding(input_dim={self.input_dim}, "
                f"n_levels={self.n_levels}, output_dim={self.output_dim})")


class HashEncoding(nn.Module):
    """
    Multi-resolution hash encoding (simplified Instant-NGP style).
    (Müller et al. 2022 — "Instant Neural Graphics Primitives")

    For each resolution level l:
      - Map 3D point to voxel corners
      - Hash corner indices to lookup table entries
      - Trilinearly interpolate feature vectors

    This is much faster than Fourier encoding on CPU because:
      - Feature lookups replace trigonometric ops
      - The hash table is small and cache-friendly
    """

    def __init__(
        self,
        input_dim: int = 3,
        n_levels: int = 16,
        n_features_per_level: int = 2,
        log2_hashmap_size: int = 19,
        base_resolution: int = 16,
        finest_resolution: int = 512,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.n_levels = n_levels
        self.n_features = n_features_per_level
        self.hashmap_size = 2 ** log2_hashmap_size
        self.output_dim = n_levels * n_features_per_level

        # Growth factor between resolution levels
        b = np.exp((np.log(finest_resolution) - np.log(base_resolution)) / (n_levels - 1))
        self.resolutions = [int(base_resolution * (b ** l)) for l in range(n_levels)]

        # Learnable hash tables for each level
        self.embeddings = nn.ModuleList([
            nn.Embedding(self.hashmap_size, n_features_per_level)
            for _ in range(n_levels)
        ])
        # Initialize small
        for emb in self.embeddings:
            nn.init.uniform_(emb.weight, -1e-4, 1e-4)

        # Primes for spatial hashing (from Instant-NGP paper)
        self.register_buffer("primes", torch.tensor(
            [1, 2654435761, 805459861], dtype=torch.long
        ))

    def _hash(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Spatial hash function: h(x,y,z) = (x·π₁ XOR y·π₂ XOR z·π₃) mod T
        coords: [..., 3] integer grid coords
        Returns: [...] hash indices in [0, hashmap_size)
        """
        result = torch.zeros(coords.shape[:-1], dtype=torch.long, device=coords.device)
        for i in range(coords.shape[-1]):
            result = result ^ (coords[..., i] * self.primes[i])
        return result % self.hashmap_size

    def _trilinear_interp(self, x_local: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """
        Trilinear interpolation of 8 corner features.
        x_local: [M, 3] in [0, 1] — fractional position within voxel
        features: [M, 8, F] — corner features
        Returns: [M, F]
        """
        wx = x_local[:, 0]  # [M]
        wy = x_local[:, 1]
        wz = x_local[:, 2]

        # 8 corners in order (000, 001, 010, 011, 100, 101, 110, 111)
        weights = torch.stack([
            (1 - wx) * (1 - wy) * (1 - wz),
            (1 - wx) * (1 - wy) * wz,
            (1 - wx) * wy * (1 - wz),
            (1 - wx) * wy * wz,
            wx * (1 - wy) * (1 - wz),
            wx * (1 - wy) * wz,
            wx * wy * (1 - wz),
            wx * wy * wz,
        ], dim=-1)  # [M, 8]

        return (weights.unsqueeze(-1) * features).sum(dim=1)  # [M, F]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., 3] positions normalized to [0, 1]
        Returns:
            features: [..., output_dim]
        """
        orig_shape = x.shape[:-1]  # e.g. (N,) or (B, N)
        x_flat = x.reshape(-1, 3)  # [M, 3]
        M = x_flat.shape[0]

        all_features = []

        for l, resolution in enumerate(self.resolutions):
            x_scaled = x_flat * resolution           # [M, 3]
            x_floor = x_scaled.long().clamp(0, resolution - 1)  # [M, 3]
            x_local = x_scaled - x_floor.float()    # [M, 3] fractional

            # 8 corner offsets
            offsets = torch.tensor([
                [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
                [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1],
            ], device=x.device, dtype=torch.long)   # [8, 3]

            # corners: [M, 8, 3]
            corners = (x_floor.unsqueeze(1) + offsets.unsqueeze(0)).clamp(0, resolution)
            # hash_indices: [M, 8]
            hash_indices = self._hash(corners)

            # corner_features: [M, 8, F]
            corner_features = self.embeddings[l](hash_indices)

            # interpolated: [M, F]
            interpolated = self._trilinear_interp(x_local, corner_features)
            all_features.append(interpolated)

        out = torch.cat(all_features, dim=-1)        # [M, output_dim]
        return out.reshape(*orig_shape, self.output_dim)

    def __repr__(self):
        return (f"HashEncoding(n_levels={self.n_levels}, "
                f"output_dim={self.output_dim}, "
                f"resolutions={self.resolutions[0]}..{self.resolutions[-1]})")


def get_encoding(encoding_type: str, input_dim: int, **kwargs) -> nn.Module:
    """Factory for positional encodings."""
    if encoding_type == "fourier":
        return FourierEncoding(input_dim, **kwargs)
    elif encoding_type == "hash":
        return HashEncoding(input_dim, **kwargs)
    else:
        raise ValueError(f"Unknown encoding type: {encoding_type}. Use 'fourier' or 'hash'.")
