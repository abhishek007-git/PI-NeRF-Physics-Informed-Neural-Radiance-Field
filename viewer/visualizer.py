"""
Interactive Novel View Synthesis Visualizer
============================================
Renders novel views from a trained PI-NeRF model and displays:
  - RGB novel view
  - Depth map
  - Accumulated opacity map
  - Loss curves
  - PSNR over training
Uses matplotlib — no web server needed, runs on CPU.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from typing import Optional


def visualize_render(
    rgb: torch.Tensor,
    depth: torch.Tensor,
    acc: torch.Tensor,
    gt_rgb: Optional[torch.Tensor] = None,
    title: str = "PI-NeRF Novel View",
    save_path: Optional[str] = None,
):
    """
    Show rendered outputs side by side.

    Args:
        rgb:      [H, W, 3] rendered color
        depth:    [H, W]    rendered depth
        acc:      [H, W]    accumulated opacity
        gt_rgb:   [H, W, 3] ground truth (optional)
        title:    Figure title
        save_path: If given, saves to file
    """
    n_cols = 4 if gt_rgb is not None else 3
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # RGB
    axes[0].imshow(rgb.cpu().numpy().clip(0, 1))
    axes[0].set_title("Rendered RGB")
    axes[0].axis("off")

    # Depth
    d = depth.cpu().numpy()
    axes[1].imshow(d, cmap="plasma")
    axes[1].set_title("Depth Map")
    axes[1].axis("off")

    # Opacity
    axes[2].imshow(acc.cpu().numpy(), cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Accumulated Opacity")
    axes[2].axis("off")

    # Ground truth comparison
    if gt_rgb is not None:
        axes[3].imshow(gt_rgb.cpu().numpy().clip(0, 1))
        axes[3].set_title("Ground Truth")
        axes[3].axis("off")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"Saved render to: {save_path}")

    plt.show()


def visualize_training_curves(
    csv_path: str,
    save_path: Optional[str] = None,
):
    """
    Plot training loss curves and PSNR from logged CSV.
    """
    import csv

    steps, losses, psnrs, eik_losses, lap_losses = [], [], [], [], []

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("loss_total"):
                steps.append(int(row["step"]))
                losses.append(float(row["loss_total"]))
                psnrs.append(float(row.get("psnr_train", 0) or 0))
                eik_losses.append(float(row.get("loss_eikonal", 0) or 0))
                lap_losses.append(float(row.get("loss_laplacian", 0) or 0))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("PI-NeRF Training Curves", fontsize=14, fontweight="bold")

    # Total loss
    axes[0].semilogy(steps, losses, color="#378ADD", linewidth=1.5)
    axes[0].set_title("Total Loss (log scale)")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)

    # PSNR
    axes[1].plot(steps, psnrs, color="#1D9E75", linewidth=1.5)
    axes[1].set_title("Training PSNR (dB)")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("PSNR (dB)")
    axes[1].grid(True, alpha=0.3)

    # Physics losses
    axes[2].semilogy(steps, [max(e, 1e-10) for e in eik_losses],
                     label="Eikonal", color="#D85A30", linewidth=1.5)
    axes[2].semilogy(steps, [max(l, 1e-10) for l in lap_losses],
                     label="Laplacian", color="#BA7517", linewidth=1.5)
    axes[2].set_title("Physics Losses (log scale)")
    axes[2].set_xlabel("Iteration")
    axes[2].set_ylabel("Loss")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"Saved training curves to: {save_path}")

    plt.show()


def render_360_video(
    model,
    renderer,
    H: int,
    W: int,
    focal: float,
    n_frames: int = 40,
    radius: float = 3.0,
    elevation: float = 0.3,
    chunk_size: int = 512,
    save_dir: str = "experiments/video_frames",
):
    """
    Render a 360° orbit around the scene and save frames.
    Frames can be assembled into a video with imageio.
    """
    from data.synthetic import spherical_pose
    from renderer.ray_utils import get_rays

    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    print(f"Rendering {n_frames} frames at {W}×{H}...")

    frames = []
    for i in range(n_frames):
        theta = 2 * np.pi * i / n_frames
        c2w = torch.FloatTensor(spherical_pose(theta, elevation, radius))

        rays_o, rays_d = get_rays(H, W, focal, c2w)
        rays_o = rays_o.reshape(-1, 3)
        rays_d = rays_d.reshape(-1, 3)

        with torch.no_grad():
            out = renderer.render_image(model, rays_o, rays_d, chunk_size=chunk_size)

        rgb = out["rgb"].reshape(H, W, 3).cpu().numpy()
        rgb = (rgb.clip(0, 1) * 255).astype(np.uint8)
        frames.append(rgb)

        frame_path = os.path.join(save_dir, f"frame_{i:04d}.png")
        plt.imsave(frame_path, rgb)

    print(f"Frames saved to: {save_dir}/")

    # Try to save as GIF
    try:
        import imageio
        gif_path = os.path.join(save_dir, "orbit.gif")
        imageio.mimsave(gif_path, frames, fps=15)
        print(f"360° orbit GIF saved: {gif_path}")
    except ImportError:
        print("Install imageio to save GIF: pip install imageio")

    return frames
