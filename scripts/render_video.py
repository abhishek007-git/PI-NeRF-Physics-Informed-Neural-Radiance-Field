"""
Novel View Synthesis — 360° Orbit Renderer
===========================================
Renders a smooth 360° orbit around the scene using a trained checkpoint.
Saves individual frames as PNGs and optionally assembles a GIF.

Usage:
    python scripts/render_video.py --checkpoint experiments/pi_nerf_cpu_dev/best_checkpoint.pth
    python scripts/render_video.py --checkpoint ... --n_frames 60 --radius 3.0
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import torch
import numpy as np
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Render PI-NeRF 360° orbit")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--n_frames", type=int, default=40,
                        help="Number of frames in the orbit")
    parser.add_argument("--radius", type=float, default=3.0,
                        help="Camera orbit radius")
    parser.add_argument("--elevation", type=float, default=0.3,
                        help="Camera elevation angle (radians)")
    parser.add_argument("--width", type=int, default=None,
                        help="Output width (defaults to training resolution)")
    parser.add_argument("--height", type=int, default=None,
                        help="Output height")
    args = parser.parse_args()

    ckpt_dir = os.path.dirname(args.checkpoint)
    config_path = os.path.join(ckpt_dir, "config.json")
    with open(config_path) as f:
        cfg = json.load(f)

    device = torch.device("cpu")
    output_dir = os.path.join(ckpt_dir, "video_frames")
    os.makedirs(output_dir, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────
    from models.nerf import HierarchicalNeRF
    from renderer.volume_renderer import VolumeRenderer
    from renderer.ray_utils import get_rays
    from data.synthetic import spherical_pose
    from utils.checkpoint import load_checkpoint

    model = HierarchicalNeRF(cfg["model"]).to(device)
    step, _ = load_checkpoint(args.checkpoint, model)
    model.eval()

    renderer = VolumeRenderer(cfg["renderer"])

    data_cfg = cfg["data"]
    H = args.height or data_cfg.get("image_height", 64)
    W = args.width or data_cfg.get("image_width", 64)
    focal = W * 1.0   # Simple focal = width

    print(f"\nRendering {args.n_frames} frames at {W}×{H}...")
    print(f"Orbit radius: {args.radius}  |  Elevation: {args.elevation:.2f} rad")

    frames_rgb = []
    frames_depth = []

    for i in range(args.n_frames):
        theta = 2 * np.pi * i / args.n_frames
        c2w = torch.FloatTensor(spherical_pose(theta, args.elevation, args.radius))

        rays_o, rays_d = get_rays(H, W, focal, c2w)
        rays_o = rays_o.reshape(-1, 3)
        rays_d = rays_d.reshape(-1, 3)

        with torch.no_grad():
            out = renderer.render_image(
                model, rays_o, rays_d,
                chunk_size=cfg["training"]["chunk_size"],
            )

        rgb = out["rgb"].reshape(H, W, 3).cpu().numpy().clip(0, 1)
        depth = out["depth"].reshape(H, W).cpu().numpy()

        frames_rgb.append((rgb * 255).astype(np.uint8))
        frames_depth.append(depth)

        # Save frame
        frame_path = os.path.join(output_dir, f"frame_{i:04d}.png")
        plt.imsave(frame_path, rgb)

        if (i + 1) % 10 == 0 or i == args.n_frames - 1:
            print(f"  [{i+1}/{args.n_frames}] frames rendered")

    # ── Save GIF ──────────────────────────────────────────────────────────
    try:
        import imageio
        gif_path = os.path.join(output_dir, "orbit_rgb.gif")
        imageio.mimsave(gif_path, frames_rgb, fps=15, loop=0)
        print(f"\nOrbit GIF saved: {gif_path}")
    except ImportError:
        print("\nTip: Install imageio to save GIF:  pip install imageio")

    # ── Summary mosaic ────────────────────────────────────────────────────
    n_show = min(8, args.n_frames)
    step_show = args.n_frames // n_show
    fig, axes = plt.subplots(2, n_show, figsize=(2 * n_show, 5))
    fig.suptitle(f"PI-NeRF 360° Orbit — {args.n_frames} frames (step {step})",
                 fontsize=12, fontweight="bold")

    for col, frame_idx in enumerate(range(0, n_show * step_show, step_show)):
        axes[0, col].imshow(frames_rgb[frame_idx])
        axes[0, col].set_title(f"θ={360*frame_idx//args.n_frames}°", fontsize=8)
        axes[0, col].axis("off")

        axes[1, col].imshow(frames_depth[frame_idx], cmap="plasma")
        axes[1, col].axis("off")

    axes[0, 0].set_ylabel("RGB", fontsize=9)
    axes[1, 0].set_ylabel("Depth", fontsize=9)

    plt.tight_layout()
    mosaic_path = os.path.join(output_dir, "orbit_mosaic.png")
    plt.savefig(mosaic_path, dpi=120, bbox_inches="tight")
    plt.show()
    print(f"Mosaic saved: {mosaic_path}")
    print(f"All frames:   {output_dir}/")


if __name__ == "__main__":
    main()
