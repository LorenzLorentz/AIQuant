"""Annealed guidance scale for P3 arbitrage energy."""

from __future__ import annotations

import torch


def guidance_lambda(t, alpha_bar, lambda_max: float = 1.0, p: float = 2.0):
    """Compute ``lambda_max * (1 - alpha_bar_t) ** p``.

    ``alpha_bar`` may be either a full diffusion schedule indexed by ``t`` or
    already-selected alpha-bar values.
    """
    if torch.is_tensor(alpha_bar):
        alpha_bar_values = alpha_bar
        if t is not None and alpha_bar.dim() == 1:
            if torch.is_tensor(t):
                alpha_bar_values = alpha_bar.to(device=t.device)[t]
            else:
                alpha_bar_values = alpha_bar[int(t)]
        alpha_bar_values = alpha_bar_values.to(dtype=torch.float32)
        return float(lambda_max) * (1.0 - alpha_bar_values).clamp_min(0.0) ** float(p)

    alpha_bar_value = float(alpha_bar)
    return float(lambda_max) * max(0.0, 1.0 - alpha_bar_value) ** float(p)


__all__ = ["guidance_lambda"]
