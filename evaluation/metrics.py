"""
Image Quality Metrics
======================
Standard metrics for evaluating novel view synthesis quality.

PSNR (Peak Signal-to-Noise Ratio):
  PSNR = 10 · log₁₀(MAX²/MSE)
  Higher is better. Typical NeRF: 25–35 dB.

SSIM (Structural Similarity Index):
  SSIM(x,y) = (2μxμy + c1)(2σxy + c2) / (μx² + μy² + c1)(σx² + σy² + c2)
  Range [0, 1], higher is better. Perceptually motivated.

LPIPS (Learned Perceptual Image Patch Similarity):
  Uses VGG features. Requires optional lpips package.
"""

import torch
import torch.nn.functional as F
import numpy as np
import math
from typing import Union


def compute_psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    max_val: float = 1.0,
) -> float:
    """
    PSNR = 10 · log₁₀(MAX² / MSE)

    Args:
        pred:   [H, W, 3] or [N, 3] predicted image
        target: same shape, ground truth
        max_val: maximum pixel value (1.0 for float images)

    Returns:
        psnr: float (dB)
    """
    mse = F.mse_loss(pred, target).item()
    if mse == 0:
        return float("inf")
    return 10.0 * math.log10(max_val ** 2 / mse)


def compute_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    C1: float = 0.01 ** 2,
    C2: float = 0.03 ** 2,
) -> float:
    """
    SSIM via Gaussian-weighted local statistics.

    Args:
        pred:   [H, W, 3] predicted (float in [0,1])
        target: [H, W, 3] ground truth
        window_size: Gaussian kernel size
        sigma:  Gaussian standard deviation

    Returns:
        ssim: float in [-1, 1] (higher = more similar)
    """
    # Convert to [1, C, H, W]
    pred_t = pred.permute(2, 0, 1).unsqueeze(0)    # [1, 3, H, W]
    target_t = target.permute(2, 0, 1).unsqueeze(0)

    # Create Gaussian kernel
    kernel = _gaussian_kernel(window_size, sigma, channels=3).to(pred.device)

    pad = window_size // 2

    mu1 = F.conv2d(pred_t, kernel, padding=pad, groups=3)
    mu2 = F.conv2d(target_t, kernel, padding=pad, groups=3)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu12 = mu1 * mu2

    sigma1_sq = F.conv2d(pred_t * pred_t, kernel, padding=pad, groups=3) - mu1_sq
    sigma2_sq = F.conv2d(target_t * target_t, kernel, padding=pad, groups=3) - mu2_sq
    sigma12 = F.conv2d(pred_t * target_t, kernel, padding=pad, groups=3) - mu12

    numerator = (2 * mu12 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

    ssim_map = numerator / (denominator + 1e-10)
    return ssim_map.mean().item()


def _gaussian_kernel(size: int, sigma: float, channels: int) -> torch.Tensor:
    """Create a per-channel Gaussian convolutional kernel."""
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel_2d = g[:, None] * g[None, :]  # [size, size]
    kernel = kernel_2d.unsqueeze(0).unsqueeze(0)  # [1, 1, size, size]
    kernel = kernel.repeat(channels, 1, 1, 1)     # [C, 1, size, size]
    return kernel


def compute_all_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> dict:
    """Compute all available metrics."""
    results = {
        "psnr": compute_psnr(pred, target),
        "ssim": compute_ssim(pred, target),
    }
    return results
