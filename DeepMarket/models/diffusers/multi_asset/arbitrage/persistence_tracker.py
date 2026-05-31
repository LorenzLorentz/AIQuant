"""Optional persistence multiplier for repeated ETF-basket NAV dislocations."""

from __future__ import annotations

import torch


class PersistenceTracker:
    """Track sign persistence of NAV gaps across reverse steps.

    P3 v1 defaults ``weight`` to zero, so this class exposes the future stateful
    interface without changing guidance strength.
    """

    def __init__(self, weight: float = 0.0):
        if weight < 0:
            raise ValueError(f"weight must be non-negative, got {weight}")
        self.weight = float(weight)
        self._previous_gap: torch.Tensor | None = None

    def reset_state(self) -> None:
        self._previous_gap = None

    def multiplier(self, gaps: torch.Tensor) -> torch.Tensor:
        """Return a per-batch multiplier with shape ``(B,)``."""
        if gaps.dim() != 2:
            raise ValueError(f"gaps must be (B, M), got {tuple(gaps.shape)}")

        if self.weight == 0.0 or self._previous_gap is None:
            self._previous_gap = gaps.detach()
            return torch.ones(gaps.shape[0], device=gaps.device, dtype=gaps.dtype)

        previous = self._previous_gap.to(device=gaps.device, dtype=gaps.dtype)
        sign_same = (previous * gaps.detach() > 0).float().mean(dim=1)
        self._previous_gap = gaps.detach()
        return 1.0 + self.weight * sign_same.to(gaps.dtype)
