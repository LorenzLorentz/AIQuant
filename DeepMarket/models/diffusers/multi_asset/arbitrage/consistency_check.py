"""Layer-3 posterior consistency diagnostics for generated trajectories.

The checks in this file are intentionally post-hoc: they summarize completed
wall-clock trajectories and compare them with reference statistics computed
from historical data. They do not feed back into generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, Tuple

import numpy as np


DEFAULT_TOLERANCES = {
    "mean_abs_spread": 0.50,
    "std_abs_spread": 0.50,
    "p95_excursion_duration": 1.00,
    "p99_excursion_duration": 1.00,
    "p99_abs_spread": 0.75,
    "p99_9_abs_spread": 1.00,
    "abs_spread_vol_corr": 0.30,
}


@dataclass(frozen=True)
class ConsistencyReference:
    """Reference statistics and tolerances for Layer-3 checks."""

    stats: Dict[str, float]
    tolerances: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_TOLERANCES))


def _as_float_array(values) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError("input array is empty")
    return arr


def _finite_flat(values) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    return arr[np.isfinite(arr)]


def _infer_num_assets(asset_universe) -> int:
    if hasattr(asset_universe, "num_assets"):
        return int(asset_universe.num_assets)
    if hasattr(asset_universe, "assets"):
        return len(asset_universe.assets)
    raise ValueError("asset_universe must expose num_assets or assets")


def normalize_lob(lob, asset_universe=None, num_assets: int | None = None) -> np.ndarray:
    """Normalize LOB arrays to ``(B, T, N, F)``.

    Accepted input shapes are ``(N,T,F)``, ``(T,N,F)``, ``(B,N,T,F)``, and
    ``(B,T,N,F)``. Ambiguous axes are resolved using ``num_assets`` when
    provided, otherwise the universe size.
    """

    arr = _as_float_array(lob)
    n = num_assets if num_assets is not None else None
    if n is None and asset_universe is not None:
        n = _infer_num_assets(asset_universe)
    if n is None:
        n = 2

    if arr.ndim == 3:
        if arr.shape[0] == n:
            return np.transpose(arr, (1, 0, 2))[None, ...]
        if arr.shape[1] == n:
            return arr[None, ...]
    elif arr.ndim == 4:
        if arr.shape[1] == n:
            return np.transpose(arr, (0, 2, 1, 3))
        if arr.shape[2] == n:
            return arr
    raise ValueError(
        "LOB must have shape (N,T,F), (T,N,F), (B,N,T,F), or (B,T,N,F); "
        f"got {arr.shape} with num_assets={n}"
    )


def normalize_mid(mid, asset_universe=None, num_assets: int | None = None) -> np.ndarray:
    """Normalize mid-price arrays to ``(B, T, N)``."""

    arr = _as_float_array(mid)
    n = num_assets if num_assets is not None else None
    if n is None and asset_universe is not None:
        n = _infer_num_assets(asset_universe)
    if n is None:
        n = 2

    if arr.ndim == 2:
        if arr.shape[0] == n:
            return arr.T[None, ...]
        if arr.shape[1] == n:
            return arr[None, ...]
    elif arr.ndim == 3:
        if arr.shape[1] == n:
            return np.transpose(arr, (0, 2, 1))
        if arr.shape[2] == n:
            return arr
    raise ValueError(
        "mid must have shape (N,T), (T,N), (B,N,T), or (B,T,N); "
        f"got {arr.shape} with num_assets={n}"
    )


def normalize_spread(spread) -> np.ndarray:
    """Normalize spread arrays to ``(B, T, G)``."""

    arr = _as_float_array(spread)
    if arr.ndim == 1:
        return arr[None, :, None]
    if arr.ndim == 2:
        return arr[None, :, :]
    if arr.ndim == 3:
        return arr
    raise ValueError(f"spread must be 1D, 2D, or 3D, got {arr.shape}")


def mid_prices_from_lob(
    lob,
    asset_universe=None,
    num_assets: int | None = None,
    ask_price_col: int = 0,
    bid_price_col: int = 2,
    eps: float = 1e-12,
) -> np.ndarray:
    """Return mid prices with shape ``(B, T, N)`` from LOB trajectories."""

    arr = normalize_lob(lob, asset_universe=asset_universe, num_assets=num_assets)
    if arr.shape[-1] <= max(ask_price_col, bid_price_col):
        raise ValueError(
            "LOB feature axis is too small for requested price columns: "
            f"shape={arr.shape}, ask_col={ask_price_col}, bid_col={bid_price_col}"
        )
    ask = arr[..., ask_price_col]
    bid = arr[..., bid_price_col]
    mid = 0.5 * (ask + bid)
    return np.maximum(np.abs(mid), eps)


def _spread_groups(asset_universe) -> Iterable[Tuple[int, list[Tuple[int, float]]]]:
    groups = []
    for etf_idx, basket in sorted(getattr(asset_universe, "etf_basket_weights", {}).items()):
        groups.append((int(etf_idx), [(int(i), float(w)) for i, w in sorted(basket.items())]))
    return groups


def spread_from_mid(mid, asset_universe) -> np.ndarray:
    """Compute ETF-minus-basket spread with shape ``(B, T, G)``."""

    mid_arr = normalize_mid(mid, asset_universe=asset_universe)
    _, _, n = mid_arr.shape
    groups = list(_spread_groups(asset_universe))
    if not groups:
        raise ValueError("asset_universe.etf_basket_weights is required for spread checks")

    spreads = []
    for etf_idx, basket in groups:
        if etf_idx < 0 or etf_idx >= n:
            raise ValueError(f"ETF asset index {etf_idx} outside N={n}")
        basket_mid = np.zeros(mid_arr.shape[:2], dtype=float)
        for asset_idx, weight in basket:
            if asset_idx < 0 or asset_idx >= n:
                raise ValueError(f"basket asset index {asset_idx} outside N={n}")
            basket_mid = basket_mid + float(weight) * mid_arr[:, :, asset_idx]
        spreads.append(mid_arr[:, :, etf_idx] - basket_mid)
    return np.stack(spreads, axis=-1)


def spread_from_lob(lob, asset_universe, **mid_kwargs) -> np.ndarray:
    """Compute spread from LOB trajectories, returning ``(B, T, G)``."""

    mid = mid_prices_from_lob(lob, asset_universe=asset_universe, **mid_kwargs)
    return spread_from_mid(mid, asset_universe)


def excursion_durations(spread, delta_base, time_axis: int = 1) -> np.ndarray:
    """Return consecutive durations where ``|spread| > delta_base``."""

    values = np.abs(_as_float_array(spread))
    values = np.moveaxis(values, time_axis, 0)
    series = values.reshape(values.shape[0], -1)
    delta = np.asarray(delta_base, dtype=float)
    if delta.size == 1:
        thresholds = np.full(series.shape[1], float(delta.reshape(-1)[0]))
    else:
        group_count = values.shape[-1]
        if delta.size != group_count:
            raise ValueError(f"delta_base must be scalar or length {group_count}, got {delta.size}")
        thresholds = np.resize(delta.reshape(1, -1), (series.shape[1],))

    durations = []
    for col in range(series.shape[1]):
        run = 0
        threshold = thresholds[col]
        finite_outside = np.isfinite(series[:, col]) & (series[:, col] > threshold)
        for is_outside in finite_outside:
            if is_outside:
                run += 1
            elif run:
                durations.append(run)
                run = 0
        if run:
            durations.append(run)
    return np.asarray(durations, dtype=float)


def _safe_quantile(values: np.ndarray, q: float, default: float = 0.0) -> float:
    finite = _finite_flat(values)
    if finite.size == 0:
        return default
    return float(np.quantile(finite, q))


def _realized_volatility(mid, window: int = 10) -> np.ndarray:
    mid_arr = normalize_mid(mid)
    log_mid = np.log(np.maximum(np.abs(mid_arr), 1e-12))
    returns = np.diff(log_mid, axis=1)
    if returns.shape[1] == 0:
        return np.zeros(mid_arr.shape[:2] + (mid_arr.shape[2],), dtype=float)
    window = max(1, int(window))
    out = np.full_like(returns, np.nan, dtype=float)
    for t in range(returns.shape[1]):
        start = max(0, t - window + 1)
        out[:, t, :] = np.nanstd(returns[:, start : t + 1, :], axis=1)
    first = out[:, :1, :]
    return np.concatenate([first, out], axis=1)


def summarize_spread_consistency(
    spread,
    mid_prices=None,
    delta_base=None,
    realized_vol_window: int = 10,
) -> Dict[str, float]:
    """Compute the P4 Layer-3 summary statistic set."""

    spread_arr = normalize_spread(spread)
    abs_spread = np.abs(spread_arr)
    finite_abs = _finite_flat(abs_spread)
    if finite_abs.size == 0:
        raise ValueError("spread contains no finite values")
    if delta_base is None:
        delta_base = float(np.median(finite_abs))

    durations = excursion_durations(spread_arr, delta_base=delta_base, time_axis=1)
    duration_mean = float(durations.mean()) if durations.size else 0.0
    duration_p95 = float(np.quantile(durations, 0.95)) if durations.size else 0.0
    duration_p99 = float(np.quantile(durations, 0.99)) if durations.size else 0.0

    corr = np.nan
    if mid_prices is not None:
        mid_arr = normalize_mid(mid_prices)
        vol = _realized_volatility(mid_arr, window=realized_vol_window).mean(axis=-1)
        if vol.shape[1] != abs_spread.shape[1]:
            min_len = min(vol.shape[1], abs_spread.shape[1])
            vol = vol[:, :min_len]
            abs_for_corr = abs_spread[:, :min_len, :].mean(axis=-1)
        else:
            abs_for_corr = abs_spread.mean(axis=-1)
        x = np.asarray(abs_for_corr, dtype=float).reshape(-1)
        y = np.asarray(vol, dtype=float).reshape(-1)
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() >= 2:
            x = x[mask]
            y = y[mask]
            if not np.isclose(x.std(), 0.0) and not np.isclose(y.std(), 0.0):
                corr = float(np.corrcoef(x, y)[0, 1])

    return {
        "mean_abs_spread": float(finite_abs.mean()),
        "std_abs_spread": float(finite_abs.std()),
        "mean_excursion_duration": duration_mean,
        "p95_excursion_duration": duration_p95,
        "p99_excursion_duration": duration_p99,
        "p99_abs_spread": _safe_quantile(finite_abs, 0.99),
        "p99_9_abs_spread": _safe_quantile(finite_abs, 0.999),
        "abs_spread_vol_corr": corr,
        "delta_base": float(np.asarray(delta_base, dtype=float).reshape(-1)[0]),
        "num_excursions": int(durations.size),
        "num_observations": int(finite_abs.size),
    }


def reference_from_lob(
    lob,
    asset_universe,
    delta_base=None,
    tolerances: Mapping[str, float] | None = None,
    **mid_kwargs,
) -> ConsistencyReference:
    """Build a reusable reference statistic set from historical LOB data."""

    mid = mid_prices_from_lob(lob, asset_universe=asset_universe, **mid_kwargs)
    spread = spread_from_mid(mid, asset_universe)
    stats = summarize_spread_consistency(spread, mid_prices=mid, delta_base=delta_base)
    merged_tolerances = dict(DEFAULT_TOLERANCES)
    if tolerances is not None:
        merged_tolerances.update({str(k): float(v) for k, v in tolerances.items()})
    return ConsistencyReference(stats=stats, tolerances=merged_tolerances)


def reference_from_spread(
    spread,
    mid_prices=None,
    delta_base=None,
    tolerances: Mapping[str, float] | None = None,
) -> ConsistencyReference:
    """Build a reference statistic set from precomputed spread data."""

    stats = summarize_spread_consistency(spread, mid_prices=mid_prices, delta_base=delta_base)
    merged_tolerances = dict(DEFAULT_TOLERANCES)
    if tolerances is not None:
        merged_tolerances.update({str(k): float(v) for k, v in tolerances.items()})
    return ConsistencyReference(stats=stats, tolerances=merged_tolerances)


def _within_tolerance(candidate: float, reference: float, tolerance: float) -> bool:
    if not np.isfinite(reference) and not np.isfinite(candidate):
        return True
    if not np.isfinite(reference) or not np.isfinite(candidate):
        return False
    scale = max(abs(reference), 1e-12)
    return abs(candidate - reference) <= float(tolerance) * scale


def check_stats(
    candidate_stats: Mapping[str, float],
    reference_stats: Mapping[str, float],
    tolerances: Mapping[str, float] | None = None,
) -> Dict[str, bool]:
    """Return pass/fail booleans for every configured statistic."""

    tol = dict(DEFAULT_TOLERANCES)
    if tolerances is not None:
        tol.update({str(k): float(v) for k, v in tolerances.items()})
    checks = {}
    for key, tolerance in tol.items():
        if key not in candidate_stats or key not in reference_stats:
            continue
        checks[key] = _within_tolerance(
            float(candidate_stats[key]),
            float(reference_stats[key]),
            float(tolerance),
        )
    return checks


def consistency_check(
    generated,
    asset_universe,
    reference: ConsistencyReference | Mapping[str, float],
    delta_base=None,
    generated_is_spread: bool = False,
    mid_prices=None,
    **mid_kwargs,
) -> Dict[str, object]:
    """Run Layer-3 consistency checks for one generated trajectory set.

    ``generated`` can be a LOB trajectory or a precomputed spread tensor when
    ``generated_is_spread=True``. The returned dict is JSON-serializable.
    """

    if isinstance(reference, ConsistencyReference):
        reference_stats = reference.stats
        tolerances = reference.tolerances
    else:
        reference_stats = dict(reference)
        tolerances = dict(DEFAULT_TOLERANCES)

    if delta_base is None:
        delta_base = reference_stats.get("delta_base", None)

    if generated_is_spread:
        spread = _as_float_array(generated)
        mid = None if mid_prices is None else normalize_mid(mid_prices, asset_universe=asset_universe)
    else:
        mid = mid_prices_from_lob(generated, asset_universe=asset_universe, **mid_kwargs)
        spread = spread_from_mid(mid, asset_universe)

    stats = summarize_spread_consistency(spread, mid_prices=mid, delta_base=delta_base)
    checks = check_stats(stats, reference_stats, tolerances)
    return {
        "stats": stats,
        "checks": checks,
        "passed": bool(checks and all(checks.values())),
        "reference": reference_stats,
        "tolerances": dict(tolerances),
    }


__all__ = [
    "ConsistencyReference",
    "DEFAULT_TOLERANCES",
    "check_stats",
    "consistency_check",
    "excursion_durations",
    "mid_prices_from_lob",
    "normalize_lob",
    "normalize_mid",
    "normalize_spread",
    "reference_from_lob",
    "reference_from_spread",
    "spread_from_lob",
    "spread_from_mid",
    "summarize_spread_consistency",
]
