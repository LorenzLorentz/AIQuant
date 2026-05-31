"""Energy terms for P3 annealed arbitrage guidance."""

from __future__ import annotations

import torch
from torch import nn

from models.diffusers.multi_asset.arbitrage.spread_computer import spread_groups


def rho(abs_spread: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """Hinge-quadratic penalty ``relu(|spread| - delta) ** 2``."""
    return torch.relu(abs_spread - delta) ** 2


def phi(tau: torch.Tensor) -> torch.Tensor:
    """Default linear persistence weight."""
    return tau.to(dtype=torch.float32)


class StressHead(nn.Module):
    """Small non-negative linear head over P1 rolling stats.

    The zero initialization makes P3 default to ``delta_dyn == delta_base`` when
    loading a P2 checkpoint or running without training this head.
    """

    def __init__(self, stats_dim: int = 4):
        super().__init__()
        self.linear = nn.Linear(stats_dim, 1)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, rolling_stats: torch.Tensor) -> torch.Tensor:
        if rolling_stats.dim() != 3:
            raise ValueError(
                "rolling_stats must be (B, G, stats_dim), "
                f"got {tuple(rolling_stats.shape)}"
            )
        return torch.relu(self.linear(rolling_stats).squeeze(-1))


def group_rolling_stats(rolling_stats: torch.Tensor, asset_universe) -> torch.Tensor:
    """Aggregate per-asset rolling stats into one vector per ETF basket group."""
    if rolling_stats.dim() != 3:
        raise ValueError(
            "rolling_stats must be (B, N, stats_dim), "
            f"got {tuple(rolling_stats.shape)}"
        )
    B, N, D = rolling_stats.shape
    groups = spread_groups(asset_universe)
    if not groups:
        return rolling_stats.new_zeros(B, 0, D)

    out = []
    for etf_idx, basket in groups:
        if etf_idx < 0 or etf_idx >= N:
            raise ValueError(f"ETF asset index {etf_idx} outside N={N}")
        etf_stats = rolling_stats[:, etf_idx, :]
        basket_stats = rolling_stats.new_zeros(B, D)
        weight_total = 0.0
        for asset_idx, weight in basket:
            if asset_idx < 0 or asset_idx >= N:
                raise ValueError(f"basket asset index {asset_idx} outside N={N}")
            abs_weight = abs(float(weight))
            basket_stats = basket_stats + abs_weight * rolling_stats[:, asset_idx, :]
            weight_total += abs_weight
        if weight_total > 0.0:
            basket_stats = basket_stats / weight_total
            out.append(0.5 * (etf_stats + basket_stats))
        else:
            out.append(etf_stats)
    return torch.stack(out, dim=1)


def dynamic_delta(
    delta_base: torch.Tensor,
    group_stats: torch.Tensor,
    stress_head: nn.Module | None = None,
    kappa: float = 1.0,
) -> torch.Tensor:
    """State-dependent tolerance ``delta_base * (1 + kappa * stress)``."""
    if group_stats.dim() != 3:
        raise ValueError(f"group_stats must be (B, G, D), got {tuple(group_stats.shape)}")
    B, G, _ = group_stats.shape
    base = delta_base.to(device=group_stats.device, dtype=group_stats.dtype).flatten()
    if base.numel() == 1:
        base = base.expand(G)
    elif base.numel() != G:
        raise ValueError(f"delta_base must be scalar or length {G}, got {base.numel()}")
    base = base.view(1, G).expand(B, G)

    if stress_head is None:
        stress = group_stats.new_zeros(B, G)
    else:
        stress = stress_head(group_stats.to(dtype=group_stats.dtype)).to(dtype=group_stats.dtype)
    return base * (1.0 + float(kappa) * stress)


def energy(spread: torch.Tensor, delta_dyn: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    """Return scalar ``sum_g rho(|spread_g|, delta_dyn_g) * phi(tau_g)``."""
    if spread.shape != delta_dyn.shape:
        raise ValueError(
            "spread and delta_dyn must have the same shape, "
            f"got {tuple(spread.shape)} and {tuple(delta_dyn.shape)}"
        )
    if tau.shape != spread.shape:
        raise ValueError(f"tau must match spread shape {tuple(spread.shape)}, got {tuple(tau.shape)}")
    return (rho(spread.abs(), delta_dyn) * phi(tau).to(device=spread.device, dtype=spread.dtype)).sum()


def calibrate_delta_base(spread: torch.Tensor) -> torch.Tensor:
    """Median ``|spread|`` per group, suitable for storing as ``delta_base``."""
    if spread.dim() != 2:
        raise ValueError(f"spread must be (B, G), got {tuple(spread.shape)}")
    return spread.detach().abs().median(dim=0).values


__all__ = [
    "StressHead",
    "calibrate_delta_base",
    "dynamic_delta",
    "energy",
    "group_rolling_stats",
    "phi",
    "rho",
]
