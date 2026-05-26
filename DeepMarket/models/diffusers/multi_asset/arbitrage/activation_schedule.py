"""Activation schedule for P2 spread conditioning."""

from __future__ import annotations

import torch


def default_k_spread_steps(num_diffusionsteps: int) -> int:
    """Default number of low-noise reverse steps that receive spread cond."""
    return max(1, int(num_diffusionsteps) // 4)


def spread_inject_active(t, num_diffusionsteps: int, k_spread_steps: int | None = None):
    """Return whether spread conditioning should be active at diffusion step ``t``.

    ``t`` may be a Python int or a tensor. Tensor input returns a boolean
    tensor of the same leading shape; scalar input returns a Python bool.
    """
    threshold = (
        default_k_spread_steps(num_diffusionsteps)
        if k_spread_steps is None
        else int(k_spread_steps)
    )
    threshold = max(0, threshold)
    if torch.is_tensor(t):
        return t < threshold
    return int(t) < threshold
