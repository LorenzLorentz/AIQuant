"""Mutable state carried across one multi-asset reverse diffusion loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch


@dataclass
class ReverseLoopState:
    """State shared by P2 spread conditioning and later P3 guidance.

    ``last_eps_fused`` is set after each denoising step and consumed by the
    next step's spread decoder. ``last_spread`` stores the most recently
    computed group spread. ``tau`` carries P3 per-group persistence counters.
    """

    last_eps_fused: Optional[torch.Tensor] = None
    last_spread: Optional[torch.Tensor] = None
    tau: Dict[int, torch.Tensor] = field(default_factory=dict)

    def update_eps(self, eps_fused: torch.Tensor, detach: bool = True) -> None:
        self.last_eps_fused = eps_fused.detach() if detach else eps_fused

    def update_spread(self, spread: Optional[torch.Tensor], detach: bool = True) -> None:
        if spread is None:
            self.last_spread = None
        else:
            self.last_spread = spread.detach() if detach else spread

    def reset_tau(self) -> None:
        self.tau.clear()
