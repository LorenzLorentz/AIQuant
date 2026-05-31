"""Annealed late-step guidance schedule for P3 energy guidance."""

from __future__ import annotations

import torch


class AnnealedGuidanceSchedule:
    """Return zero guidance at high-noise timesteps and ramp near t=0."""

    def __init__(self, start_step: int = 10, max_scale: float = 0.01):
        if start_step < 0:
            raise ValueError(f"start_step must be non-negative, got {start_step}")
        if max_scale < 0:
            raise ValueError(f"max_scale must be non-negative, got {max_scale}")
        self.start_step = int(start_step)
        self.max_scale = float(max_scale)

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        if self.max_scale == 0.0:
            return torch.zeros_like(t, dtype=torch.float32)

        t_float = t.float()
        if self.start_step == 0:
            return torch.where(
                t_float <= 0,
                torch.full_like(t_float, self.max_scale),
                torch.zeros_like(t_float),
            )

        progress = (float(self.start_step) - t_float) / float(self.start_step)
        progress = progress.clamp(min=0.0, max=1.0)
        return progress * self.max_scale
