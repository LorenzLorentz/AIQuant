"""Conditional spillover metrics for cross-asset returns."""

from __future__ import annotations

from typing import Iterable, Dict

import numpy as np


def conditional_spillover(
    source_returns,
    target_returns,
    quantiles: Iterable[float] = (0.90, 0.95, 0.99),
    horizon: int = 1,
) -> Dict[str, float]:
    """Compute ``E[|target_{t+h}| | |source_t| > q]`` for high quantiles."""

    source = np.asarray(source_returns, dtype=float).reshape(-1)
    target = np.asarray(target_returns, dtype=float).reshape(-1)
    if source.shape != target.shape:
        raise ValueError(f"return vectors must share shape, got {source.shape} and {target.shape}")
    horizon = int(horizon)
    if horizon < 0:
        raise ValueError("horizon must be non-negative")
    if horizon > 0:
        if source.size <= horizon:
            return {f"q{int(q * 100)}": np.nan for q in quantiles}
        source = source[:-horizon]
        target = target[horizon:]

    source_abs = np.abs(source)
    target_abs = np.abs(target)
    mask = np.isfinite(source_abs) & np.isfinite(target_abs)
    if not mask.any():
        result = {f"q{int(q * 100)}": np.nan for q in quantiles}
        result["unconditional_mean_abs_target"] = np.nan
        return result

    source_abs = source_abs[mask]
    target_abs = target_abs[mask]
    result = {"unconditional_mean_abs_target": float(target_abs.mean())}
    for q in quantiles:
        threshold = np.quantile(source_abs, float(q))
        selected = target_abs[source_abs > threshold]
        result[f"q{int(q * 100)}"] = float(selected.mean()) if selected.size else np.nan
    return result


def bidirectional_spillover(
    returns_1,
    returns_2,
    quantiles: Iterable[float] = (0.90, 0.95, 0.99),
    horizon: int = 1,
) -> Dict[str, Dict[str, float]]:
    """Return spillover in both directions for a two-asset pair."""

    return {
        "1_to_2": conditional_spillover(returns_1, returns_2, quantiles=quantiles, horizon=horizon),
        "2_to_1": conditional_spillover(returns_2, returns_1, quantiles=quantiles, horizon=horizon),
    }


__all__ = ["bidirectional_spillover", "conditional_spillover"]
