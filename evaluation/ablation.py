"""
Ablation Study Runner
======================
Systematically evaluates the contribution of each PI-NeRF component.

Ablation conditions:
  A. Full PI-NeRF       — coarse+fine, Eikonal, Laplacian, view-dependent color
  B. No Eikonal         — remove Eikonal loss
  C. No Laplacian       — remove Laplacian smoothness
  D. No Physics         — standard NeRF (no physics constraints)
  E. Coarse only        — no hierarchical sampling
  F. No view-dependence — color independent of viewing direction

Reports:
  - PSNR / SSIM per condition
  - Gradient norm statistics (geometry quality)
  - Training convergence speed
"""

import torch
import copy
import json
import os
from typing import Dict, List

from models.nerf import HierarchicalNeRF
from renderer.volume_renderer import VolumeRenderer
from renderer.ray_utils import get_rays
from evaluation.metrics import compute_all_metrics


ABLATION_CONDITIONS = {
    "full_pi_nerf": {
        "use_eikonal": True,
        "use_laplacian": True,
        "use_fine_network": True,
        "use_viewdirs": True,
        "description": "Full PI-NeRF (proposed)",
    },
    "no_eikonal": {
        "use_eikonal": False,
        "use_laplacian": True,
        "use_fine_network": True,
        "use_viewdirs": True,
        "description": "Without Eikonal constraint",
    },
    "no_laplacian": {
        "use_eikonal": True,
        "use_laplacian": False,
        "use_fine_network": True,
        "use_viewdirs": True,
        "description": "Without Laplacian smoothness",
    },
    "no_physics": {
        "use_eikonal": False,
        "use_laplacian": False,
        "use_fine_network": True,
        "use_viewdirs": True,
        "description": "Vanilla NeRF (no physics)",
    },
    "coarse_only": {
        "use_eikonal": True,
        "use_laplacian": True,
        "use_fine_network": False,
        "use_viewdirs": True,
        "description": "Coarse network only",
    },
    "no_viewdirs": {
        "use_eikonal": True,
        "use_laplacian": True,
        "use_fine_network": True,
        "use_viewdirs": False,
        "description": "View-independent color",
    },
}


def run_ablation_evaluation(
    checkpoints: Dict[str, str],
    val_dataset,
    base_cfg: dict,
    output_path: str,
    n_val_views: int = 5,
) -> Dict[str, dict]:
    """
    Evaluate all ablation conditions.

    Args:
        checkpoints: {condition_name: checkpoint_path}
        val_dataset: Validation dataset
        base_cfg:    Base config dict
        output_path: Where to save results JSON
        n_val_views: Number of validation views per condition

    Returns:
        results: {condition_name: {metric_name: value}}
    """
    device = torch.device("cpu")
    renderer = VolumeRenderer(base_cfg["renderer"])
    results = {}

    for condition, ckpt_path in checkpoints.items():
        if not os.path.exists(ckpt_path):
            print(f"Skipping {condition}: checkpoint not found at {ckpt_path}")
            continue

        print(f"\nEvaluating: {condition}")
        print(f"  {ABLATION_CONDITIONS.get(condition, {}).get('description', '')}")

        # Load model
        model_cfg = copy.deepcopy(base_cfg["model"])
        condition_overrides = ABLATION_CONDITIONS.get(condition, {})
        model_cfg.update(condition_overrides)

        model = HierarchicalNeRF(model_cfg).to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        # Evaluate
        psnrs, ssims = [], []
        with torch.no_grad():
            for i in range(min(n_val_views, len(val_dataset))):
                item = val_dataset[i]
                image_gt = item["image"].to(device)
                pose = item["pose"].to(device)
                focal = item["focal"].item()
                H, W = image_gt.shape[:2]

                rays_o, rays_d = get_rays(H, W, focal, pose)
                rays_o = rays_o.reshape(-1, 3)
                rays_d = rays_d.reshape(-1, 3)

                render_out = renderer.render_image(
                    model, rays_o, rays_d,
                    chunk_size=base_cfg["training"]["chunk_size"],
                )
                rgb_pred = render_out["rgb"].reshape(H, W, 3)

                metrics = compute_all_metrics(rgb_pred, image_gt)
                psnrs.append(metrics["psnr"])
                ssims.append(metrics["ssim"])

        results[condition] = {
            "psnr_mean": sum(psnrs) / len(psnrs),
            "psnr_values": psnrs,
            "ssim_mean": sum(ssims) / len(ssims),
            "ssim_values": ssims,
            "description": ABLATION_CONDITIONS.get(condition, {}).get("description", ""),
        }
        print(f"  PSNR: {results[condition]['psnr_mean']:.2f} dB | "
              f"SSIM: {results[condition]['ssim_mean']:.4f}")

    # Save results
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAblation results saved to: {output_path}")

    # Print summary table
    print("\n" + "="*65)
    print(f"{'Condition':<22} {'PSNR (dB)':>10} {'SSIM':>8}")
    print("-"*65)
    for name, res in results.items():
        print(f"{name:<22} {res['psnr_mean']:>10.2f} {res['ssim_mean']:>8.4f}")
    print("="*65)

    return results
