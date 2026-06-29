"""Checkpoint utilities for saving and resuming training."""

import torch
import os
from typing import Optional


def save_checkpoint(model, optimizer, step: int, metrics: dict, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }, path)


def load_checkpoint(path: str, model, optimizer=None, device=None):
    device = device or torch.device("cpu")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt.get("step", 0), ckpt.get("metrics", {})
