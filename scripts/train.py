"""
PI-NeRF Training Script
========================
Entry point for training the Physics-Informed Neural Radiance Field.

Usage:
    python scripts/train.py --config configs/fast_cpu.yaml
    python scripts/train.py --config configs/base.yaml --resume experiments/run1/latest_checkpoint.pth
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import yaml
import torch
import random
import numpy as np


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    parser = argparse.ArgumentParser(description="Train PI-NeRF")
    parser.add_argument("--config", type=str, default="configs/fast_cpu.yaml",
                        help="Path to config YAML")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override output directory")
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)
    set_seed(cfg["experiment"]["seed"])

    # Output dir
    output_dir = args.output_dir or os.path.join(
        cfg["experiment"]["output_dir"],
        cfg["experiment"]["name"],
    )
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"PI-NeRF — Physics-Informed Neural Radiance Field")
    print(f"{'='*60}")
    print(f"Config:     {args.config}")
    print(f"Output dir: {output_dir}")
    print(f"Device:     CPU")

    # ── Dataset ───────────────────────────────────────────────────────────
    data_cfg = cfg["data"]

    if data_cfg["dataset_type"] == "synthetic":
        from data.synthetic import SyntheticSphereDataset
        train_dataset = SyntheticSphereDataset(
            n_views=data_cfg.get("n_train_views", 50),
            H=data_cfg.get("image_height", 64),
            W=data_cfg.get("image_width", 64),
            split="train",
            seed=cfg["experiment"]["seed"],
        )
        val_dataset = SyntheticSphereDataset(
            n_views=data_cfg.get("n_test_views", 10),
            H=data_cfg.get("image_height", 64),
            W=data_cfg.get("image_width", 64),
            split="test",
            seed=cfg["experiment"]["seed"],
        )
    else:
        raise ValueError(
            f"Dataset type '{data_cfg['dataset_type']}' not implemented yet. "
            f"Use 'synthetic' for CPU demo. For Blender dataset, download from "
            f"https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1"
        )

    # ── Trainer ───────────────────────────────────────────────────────────
    from training.trainer import Trainer
    trainer = Trainer(cfg, output_dir)

    # Resume if requested
    if args.resume:
        from utils.checkpoint import load_checkpoint
        step, _ = load_checkpoint(args.resume, trainer.model, trainer.optimizer)
        trainer.global_step = step
        print(f"Resumed from step {step}")

    # ── Train ─────────────────────────────────────────────────────────────
    trainer.train(train_dataset, val_dataset)


if __name__ == "__main__":
    main()
