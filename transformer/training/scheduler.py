"""Noam (warmup) learning rate scheduler."""
from __future__ import annotations

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def noam_lambda(step: int, d_model: int, warmup_steps: int) -> float:
    """Noam schedule: lrate = d_model^-0.5 * min(step^-0.5, step * warmup^-1.5)"""
    step = max(step, 1)
    return (d_model ** -0.5) * min(step ** -0.5, step * warmup_steps ** -1.5)


class WarmupScheduler(LambdaLR):
    """LambdaLR wrapper implementing the Noam (Attention Is All You Need) schedule.

    The optimizer's initial lr should be set to 1.0; this scheduler scales it.
    """

    def __init__(self, optimizer: Optimizer, d_model: int, warmup_steps: int):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        super().__init__(
            optimizer,
            lr_lambda=lambda step: noam_lambda(step, d_model, warmup_steps),
        )
