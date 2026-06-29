"""
Learning Rate Schedulers
=========================
Implements the exponential decay schedule used in NeRF:
  lr(t) = lr_init · (lr_final/lr_init)^(t/T)

with optional linear warmup.
"""

import torch
import numpy as np


class NeRFScheduler:
    """
    Exponential LR decay with linear warmup.

    Schedule:
      t < warmup_steps:  lr = lr_init * (t / warmup_steps)
      t >= warmup_steps: lr = lr_init * decay_factor^(t / decay_steps)
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        lr_init: float = 5e-4,
        lr_final: float = 5e-6,
        decay_steps: int = 250000,
        warmup_steps: int = 500,
    ):
        self.optimizer = optimizer
        self.lr_init = lr_init
        self.lr_final = lr_final
        self.decay_steps = decay_steps
        self.warmup_steps = warmup_steps
        self.log_ratio = np.log(lr_final / lr_init)

    def step(self, global_step: int):
        if global_step < self.warmup_steps:
            lr = self.lr_init * (global_step + 1) / self.warmup_steps
        else:
            t = global_step - self.warmup_steps
            lr = self.lr_init * np.exp(self.log_ratio * t / self.decay_steps)

        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

        return lr

    def get_lr(self, global_step: int) -> float:
        if global_step < self.warmup_steps:
            return self.lr_init * (global_step + 1) / self.warmup_steps
        t = global_step - self.warmup_steps
        return self.lr_init * np.exp(self.log_ratio * t / self.decay_steps)
