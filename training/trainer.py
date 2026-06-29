"""
PI-NeRF Trainer
================
Main training loop implementing:
  1. Ray sampling from random training images
  2. Hierarchical rendering (coarse + fine)
  3. Physics-Informed loss (rendering + Eikonal + Laplacian)
  4. Exponential LR decay
  5. Checkpoint saving
  6. Periodic validation + PSNR logging
"""

import torch
import os
import time
import json
from typing import Dict, Optional

from models.nerf import HierarchicalNeRF
from renderer.volume_renderer import VolumeRenderer
from renderer.ray_utils import get_rays, sample_rays_from_image
from physics.pinn_loss import PINNLoss
from training.scheduler import NeRFScheduler
from evaluation.metrics import compute_psnr
from utils.logger import Logger
from utils.checkpoint import save_checkpoint, load_checkpoint


class Trainer:
    """
    Full PI-NeRF training orchestrator.
    """

    def __init__(self, cfg: dict, output_dir: str):
        self.cfg = cfg
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Save config
        with open(os.path.join(output_dir, "config.json"), "w") as f:
            json.dump(cfg, f, indent=2)

        self.device = torch.device("cpu")
        print(f"Training on: {self.device}")

        # ── Model ─────────────────────────────────────────────────────────
        self.model = HierarchicalNeRF(cfg["model"]).to(self.device)
        param_info = self.model.get_param_count()
        print(f"Model parameters: {param_info}")

        # ── Renderer ──────────────────────────────────────────────────────
        self.renderer = VolumeRenderer(cfg["renderer"])

        # ── Loss ──────────────────────────────────────────────────────────
        self.loss_fn = PINNLoss(cfg["physics"])

        # ── Optimizer + Scheduler ─────────────────────────────────────────
        train_cfg = cfg["training"]
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=train_cfg["lr_init"],
            betas=(0.9, 0.999),
            eps=1e-7,
        )
        self.scheduler = NeRFScheduler(
            self.optimizer,
            lr_init=train_cfg["lr_init"],
            lr_final=train_cfg["lr_final"],
            decay_steps=train_cfg["lr_decay_steps"],
            warmup_steps=train_cfg["warmup_steps"],
        )

        # ── Logger ────────────────────────────────────────────────────────
        self.logger = Logger(output_dir)

        # Training state
        self.global_step = 0
        self.best_psnr = 0.0

    def train_step(
        self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        target_rgb: torch.Tensor,
    ) -> Dict[str, float]:
        """Single training step on a batch of rays."""
        self.model.train()
        self.optimizer.zero_grad()

        # Render
        render_out = self.renderer.render_rays(
            self.model, rays_o, rays_d, training=True,
        )

        # Physics-informed loss
        loss, loss_dict = self.loss_fn(
            render_out, target_rgb, self.model, step=self.global_step,
        )

        # Backprop
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()
        lr = self.scheduler.step(self.global_step)
        loss_dict["lr"] = lr

        return loss_dict

    def validate(
        self,
        dataset,
        n_val_views: int = 3,
    ) -> Dict[str, float]:
        """Render a few validation views and compute PSNR."""
        self.model.eval()
        total_psnr = 0.0
        n_views = min(n_val_views, len(dataset))

        with torch.no_grad():
            for i in range(n_views):
                item = dataset[i]
                image_gt = item["image"].to(self.device)   # [H, W, 3]
                pose = item["pose"].to(self.device)         # [4, 4]
                focal = item["focal"].item()
                H, W = image_gt.shape[:2]

                # Generate rays
                rays_o, rays_d = get_rays(H, W, focal, pose)
                rays_o = rays_o.reshape(-1, 3)
                rays_d = rays_d.reshape(-1, 3)

                # Render in chunks
                render_out = self.renderer.render_image(
                    self.model, rays_o, rays_d,
                    chunk_size=self.cfg["training"]["chunk_size"],
                )
                rgb_pred = render_out["rgb"].reshape(H, W, 3)

                psnr = compute_psnr(rgb_pred, image_gt)
                total_psnr += psnr

        return {"val_psnr": total_psnr / n_views}

    def train(self, train_dataset, val_dataset=None):
        """
        Full training loop.
        """
        train_cfg = self.cfg["training"]
        n_iters = train_cfg["n_iterations"]
        batch_size = train_cfg["batch_size"]
        log_every = train_cfg["log_every"]
        save_every = train_cfg["save_every"]
        val_every = train_cfg["val_every"]

        print(f"\n{'='*60}")
        print(f"Starting PI-NeRF training for {n_iters} iterations")
        print(f"Batch size: {batch_size} rays")
        print(f"{'='*60}\n")

        # Pre-generate all training rays
        print("Pre-computing training rays...")
        all_rays_o, all_rays_d, all_rgb = [], [], []
        for i in range(len(train_dataset)):
            item = train_dataset[i]
            image = item["image"].to(self.device)
            pose = item["pose"].to(self.device)
            focal = item["focal"].item()
            H, W = image.shape[:2]

            o, d = get_rays(H, W, focal, pose)
            all_rays_o.append(o.reshape(-1, 3))
            all_rays_d.append(d.reshape(-1, 3))
            all_rgb.append(image.reshape(-1, 3))

        all_rays_o = torch.cat(all_rays_o)  # [Total_rays, 3]
        all_rays_d = torch.cat(all_rays_d)
        all_rgb = torch.cat(all_rgb)
        N_total = all_rays_o.shape[0]
        print(f"Total training rays: {N_total:,}\n")

        t_start = time.time()

        for step in range(self.global_step, n_iters):
            self.global_step = step

            # Random ray batch
            indices = torch.randperm(N_total)[:batch_size]
            rays_o = all_rays_o[indices]
            rays_d = all_rays_d[indices]
            target = all_rgb[indices]

            # Training step
            loss_dict = self.train_step(rays_o, rays_d, target)

            # Logging
            if step % log_every == 0:
                elapsed = time.time() - t_start
                psnr_train = compute_psnr_from_mse(loss_dict["loss_render_fine"])
                loss_dict["psnr_train"] = psnr_train
                self.logger.log(step, loss_dict)

                print(
                    f"Step {step:6d}/{n_iters} | "
                    f"Loss: {loss_dict['loss_total']:.4f} | "
                    f"PSNR: {psnr_train:.2f} dB | "
                    f"LR: {loss_dict['lr']:.2e} | "
                    f"Eik: {loss_dict.get('loss_eikonal', 0):.4f} | "
                    f"Lap: {loss_dict.get('loss_laplacian', 0):.4f} | "
                    f"t: {elapsed:.1f}s"
                )

            # Validation
            if val_dataset and step % val_every == 0 and step > 0:
                val_metrics = self.validate(val_dataset, n_val_views=2)
                self.logger.log(step, val_metrics)
                print(f"  → Val PSNR: {val_metrics['val_psnr']:.2f} dB")

                if val_metrics["val_psnr"] > self.best_psnr:
                    self.best_psnr = val_metrics["val_psnr"]
                    save_checkpoint(
                        self.model, self.optimizer, step, val_metrics,
                        os.path.join(self.output_dir, "best_checkpoint.pth"),
                    )

            # Save checkpoint
            if step % save_every == 0 and step > 0:
                save_checkpoint(
                    self.model, self.optimizer, step, loss_dict,
                    os.path.join(self.output_dir, f"checkpoint_{step:06d}.pth"),
                )
                save_checkpoint(
                    self.model, self.optimizer, step, loss_dict,
                    os.path.join(self.output_dir, "latest_checkpoint.pth"),
                )

        # Final save
        save_checkpoint(
            self.model, self.optimizer, n_iters, {},
            os.path.join(self.output_dir, "final_checkpoint.pth"),
        )
        print(f"\nTraining complete. Best val PSNR: {self.best_psnr:.2f} dB")
        self.logger.close()


def compute_psnr_from_mse(mse: float) -> float:
    """PSNR = -10 * log10(MSE)"""
    import math
    return -10.0 * math.log10(max(mse, 1e-10))
