"""
PI-NeRF Evaluation Script
==========================
Evaluates a trained checkpoint on test views.
Computes PSNR, SSIM and saves rendered images.

Usage:
    python scripts/evaluate.py --checkpoint experiments/pi_nerf_cpu_dev/latest_checkpoint.pth
    python scripts/evaluate.py --checkpoint experiments/pi_nerf_cpu_dev/best_checkpoint.pth --save_images
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import yaml
import torch
import json
import numpy as np
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Evaluate PI-NeRF")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default=None,
                        help="Config path (auto-detected from checkpoint dir if not given)")
    parser.add_argument("--save_images", action="store_true",
                        help="Save rendered test images")
    parser.add_argument("--n_views", type=int, default=10)
    args = parser.parse_args()

    # Auto-detect config
    ckpt_dir = os.path.dirname(args.checkpoint)
    config_path = args.config or os.path.join(ckpt_dir, "config.json")

    if config_path.endswith(".json"):
        with open(config_path) as f:
            cfg = json.load(f)
    else:
        with open(config_path) as f:
            import yaml
            cfg = yaml.safe_load(f)

    device = torch.device("cpu")
    output_dir = os.path.join(ckpt_dir, "eval_outputs")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nEvaluating checkpoint: {args.checkpoint}")

    # ── Load model ────────────────────────────────────────────────────────
    from models.nerf import HierarchicalNeRF
    from renderer.volume_renderer import VolumeRenderer
    from renderer.ray_utils import get_rays
    from evaluation.metrics import compute_all_metrics
    from utils.checkpoint import load_checkpoint

    model = HierarchicalNeRF(cfg["model"]).to(device)
    step, _ = load_checkpoint(args.checkpoint, model)
    model.eval()
    print(f"Loaded model from step {step}")

    renderer = VolumeRenderer(cfg["renderer"])

    # ── Load test dataset ─────────────────────────────────────────────────
    data_cfg = cfg["data"]
    from data.synthetic import SyntheticSphereDataset
    test_dataset = SyntheticSphereDataset(
        n_views=args.n_views,
        H=data_cfg.get("image_height", 64),
        W=data_cfg.get("image_width", 64),
        split="test",
        seed=cfg["experiment"]["seed"],
    )

    # ── Evaluate ──────────────────────────────────────────────────────────
    all_psnr, all_ssim = [], []

    for i in range(len(test_dataset)):
        item = test_dataset[i]
        image_gt = item["image"].to(device)
        pose = item["pose"].to(device)
        focal = item["focal"].item()
        H, W = image_gt.shape[:2]

        rays_o, rays_d = get_rays(H, W, focal, pose)
        rays_o = rays_o.reshape(-1, 3)
        rays_d = rays_d.reshape(-1, 3)

        with torch.no_grad():
            out = renderer.render_image(
                model, rays_o, rays_d,
                chunk_size=cfg["training"]["chunk_size"],
            )

        rgb_pred = out["rgb"].reshape(H, W, 3)
        depth = out["depth"].reshape(H, W)
        acc = out["acc"].reshape(H, W)

        metrics = compute_all_metrics(rgb_pred, image_gt)
        all_psnr.append(metrics["psnr"])
        all_ssim.append(metrics["ssim"])

        print(f"  View {i:3d}: PSNR={metrics['psnr']:.2f} dB | SSIM={metrics['ssim']:.4f}")

        if args.save_images:
            # Save side-by-side comparison
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            fig.suptitle(f"View {i} | PSNR={metrics['psnr']:.2f} dB | SSIM={metrics['ssim']:.4f}")

            axes[0].imshow(rgb_pred.cpu().numpy().clip(0, 1))
            axes[0].set_title("PI-NeRF Render")
            axes[0].axis("off")

            axes[1].imshow(image_gt.cpu().numpy().clip(0, 1))
            axes[1].set_title("Ground Truth")
            axes[1].axis("off")

            # Error map
            err = (rgb_pred - image_gt).abs().mean(-1).cpu().numpy()
            im = axes[2].imshow(err, cmap="hot", vmin=0, vmax=0.1)
            axes[2].set_title("Absolute Error")
            axes[2].axis("off")
            plt.colorbar(im, ax=axes[2])

            axes[3].imshow(depth.cpu().numpy(), cmap="plasma")
            axes[3].set_title("Depth Map")
            axes[3].axis("off")

            plt.tight_layout()
            save_path = os.path.join(output_dir, f"view_{i:04d}.png")
            plt.savefig(save_path, dpi=120, bbox_inches="tight")
            plt.close()

    # ── Summary ───────────────────────────────────────────────────────────
    mean_psnr = sum(all_psnr) / len(all_psnr)
    mean_ssim = sum(all_ssim) / len(all_ssim)

    print(f"\n{'='*50}")
    print(f"EVALUATION SUMMARY (step {step})")
    print(f"{'='*50}")
    print(f"  Views evaluated:  {len(test_dataset)}")
    print(f"  Mean PSNR:        {mean_psnr:.4f} dB")
    print(f"  Mean SSIM:        {mean_ssim:.6f}")
    print(f"  PSNR std:         {np.std(all_psnr):.4f}")
    print(f"{'='*50}")

    results = {
        "step": step,
        "checkpoint": args.checkpoint,
        "n_views": len(test_dataset),
        "mean_psnr": mean_psnr,
        "mean_ssim": mean_ssim,
        "psnr_per_view": all_psnr,
        "ssim_per_view": all_ssim,
    }
    results_path = os.path.join(output_dir, "eval_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_path}")

    if args.save_images:
        print(f"Images saved to:  {output_dir}/")


if __name__ == "__main__":
    main()
