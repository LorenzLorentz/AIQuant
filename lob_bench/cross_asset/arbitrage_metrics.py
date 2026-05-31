"""Cross-asset arbitrage/spread distribution metrics for P4."""

from __future__ import annotations

from typing import Dict, Iterable

import numpy as np


def _finite(values) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    return arr[np.isfinite(arr)]


def ks_distance(reference, candidate) -> float:
    """Two-sample Kolmogorov-Smirnov distance without a scipy dependency."""

    ref = _finite(reference)
    cand = _finite(candidate)
    if ref.size == 0 or cand.size == 0:
        raise ValueError("reference and candidate must contain finite values")
    values = np.sort(np.unique(np.concatenate([ref, cand])))
    ref_sorted = np.sort(ref)
    cand_sorted = np.sort(cand)
    ref_cdf = np.searchsorted(ref_sorted, values, side="right") / ref_sorted.size
    cand_cdf = np.searchsorted(cand_sorted, values, side="right") / cand_sorted.size
    return float(np.max(np.abs(ref_cdf - cand_cdf)))


def excursion_durations(spread, delta_base: float, time_axis: int = 1) -> np.ndarray:
    """Return run lengths for ``|spread| > delta_base``."""

    values = np.abs(np.asarray(spread, dtype=float))
    values = np.moveaxis(values, time_axis, 0)
    series = values.reshape(values.shape[0], -1)
    durations = []
    for col in range(series.shape[1]):
        run = 0
        for outside in np.isfinite(series[:, col]) & (series[:, col] > delta_base):
            if outside:
                run += 1
            elif run:
                durations.append(run)
                run = 0
        if run:
            durations.append(run)
    return np.asarray(durations, dtype=float)


def summarize_abs_spread(spread) -> Dict[str, float]:
    values = np.abs(_finite(spread))
    if values.size == 0:
        raise ValueError("spread must contain finite values")
    return {
        "mean_abs": float(values.mean()),
        "std_abs": float(values.std()),
        "p50_abs": float(np.quantile(values, 0.50)),
        "p95_abs": float(np.quantile(values, 0.95)),
        "p99_abs": float(np.quantile(values, 0.99)),
        "p99_9_abs": float(np.quantile(values, 0.999)),
    }


def summarize_excursions(spread, delta_base: float, time_axis: int = 1) -> Dict[str, float]:
    durations = excursion_durations(spread, delta_base=delta_base, time_axis=time_axis)
    if durations.size == 0:
        return {
            "count": 0,
            "mean_duration": 0.0,
            "p95_duration": 0.0,
            "p99_duration": 0.0,
            "max_duration": 0.0,
        }
    return {
        "count": int(durations.size),
        "mean_duration": float(durations.mean()),
        "p95_duration": float(np.quantile(durations, 0.95)),
        "p99_duration": float(np.quantile(durations, 0.99)),
        "max_duration": float(durations.max()),
    }


def conditional_stress_spread_mean(spread, stress, quantile: float = 0.95) -> float:
    """Mean ``|spread|`` when the supplied stress proxy exceeds a quantile."""

    spread_abs = np.abs(np.asarray(spread, dtype=float)).reshape(-1)
    stress_values = np.asarray(stress, dtype=float).reshape(-1)
    if spread_abs.shape != stress_values.shape:
        raise ValueError(
            "spread and stress must flatten to the same shape, "
            f"got {spread_abs.shape} and {stress_values.shape}"
        )
    mask = np.isfinite(spread_abs) & np.isfinite(stress_values)
    if not mask.any():
        return np.nan
    spread_abs = spread_abs[mask]
    stress_values = stress_values[mask]
    threshold = np.quantile(stress_values, float(quantile))
    selected = spread_abs[stress_values >= threshold]
    if selected.size == 0:
        return np.nan
    return float(selected.mean())


def compare_arbitrage_metrics(
    real_spread,
    generated_spread,
    delta_base: float | None = None,
    real_stress=None,
    generated_stress=None,
    stress_quantiles: Iterable[float] = (0.90, 0.95, 0.99),
    time_axis: int = 1,
) -> Dict[str, object]:
    """Aggregate P4 spread-distribution metrics for one generated corner."""

    real_abs = np.abs(_finite(real_spread))
    if real_abs.size == 0:
        raise ValueError("real_spread must contain finite values")
    if delta_base is None:
        delta_base = float(np.median(real_abs))

    result: Dict[str, object] = {
        "ks_distance": ks_distance(np.abs(real_spread), np.abs(generated_spread)),
        "delta_base": float(delta_base),
        "real_spread": summarize_abs_spread(real_spread),
        "generated_spread": summarize_abs_spread(generated_spread),
        "real_excursions": summarize_excursions(real_spread, delta_base, time_axis=time_axis),
        "generated_excursions": summarize_excursions(generated_spread, delta_base, time_axis=time_axis),
    }
    if real_stress is not None and generated_stress is not None:
        result["conditional_stress_spread_mean"] = {
            f"q{int(q * 100)}": {
                "real": conditional_stress_spread_mean(real_spread, real_stress, quantile=q),
                "generated": conditional_stress_spread_mean(generated_spread, generated_stress, quantile=q),
            }
            for q in stress_quantiles
        }
    return result


__all__ = [
    "compare_arbitrage_metrics",
    "conditional_stress_spread_mean",
    "excursion_durations",
    "ks_distance",
    "summarize_abs_spread",
    "summarize_excursions",
]
