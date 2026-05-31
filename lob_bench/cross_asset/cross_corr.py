"""Cross-asset mid-price return correlation sanity metric.

This module is intentionally small: P1 only needs a coarse check that generated
two-asset trajectories carry non-trivial correlation and move closer to the
real-data correlation than the P0 independent baseline.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def aggregate_returns(returns, dt: int = 1):
    """Aggregate log returns into fixed-size bars by summation."""
    arr = np.asarray(returns, dtype=float).reshape(-1)
    dt = max(1, int(dt))
    if dt == 1:
        return arr
    usable = (arr.size // dt) * dt
    if usable == 0:
        return np.empty(0, dtype=float)
    return arr[:usable].reshape(-1, dt).sum(axis=1)


def mid_prices_from_lob(
    lob,
    bid_price_col: int = 0,
    ask_price_col: int = 2,
    eps: float = 1e-12,
):
    """Return mid prices from LOB tensors/arrays with feature axis last.

    Expected input shape is ``(..., T, F_lob)``. The default column convention
    matches the P1 DeepMarket assumption: best bid price at column 0 and best
    ask price at column 2.
    """
    arr = np.asarray(lob, dtype=float)
    if arr.shape[-1] <= max(bid_price_col, ask_price_col):
        raise ValueError(
            "LOB feature axis is too small for requested price columns: "
            f"shape={arr.shape}, bid_col={bid_price_col}, ask_col={ask_price_col}"
        )
    bid = arr[..., bid_price_col]
    ask = arr[..., ask_price_col]
    mid = 0.5 * (bid + ask)
    return np.maximum(np.abs(mid), eps)


def log_returns_from_lob(lob, **mid_kwargs):
    """Compute log mid-price returns over the time axis."""
    mid = mid_prices_from_lob(lob, **mid_kwargs)
    if mid.shape[-1] < 2:
        return np.empty(mid.shape[:-1] + (0,), dtype=float)
    return np.diff(np.log(mid), axis=-1)


def realized_corr(returns_1, returns_2, dt: int = 1, min_obs: int = 2):
    """Pearson correlation between two return vectors, ignoring non-finites."""
    a = aggregate_returns(returns_1, dt=dt)
    b = aggregate_returns(returns_2, dt=dt)
    if a.shape != b.shape:
        raise ValueError(f"return vectors must share shape, got {a.shape} and {b.shape}")
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < min_obs:
        return np.nan
    a = a[mask]
    b = b[mask]
    if np.isclose(a.std(), 0.0) or np.isclose(b.std(), 0.0):
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def realized_correlation(returns_a, returns_b, min_obs: int = 2):
    """Backward-compatible P1 alias for ``realized_corr(..., dt=1)``."""
    return realized_corr(returns_a, returns_b, dt=1, min_obs=min_obs)


def ccf(returns_1, returns_2, max_lag: int, dt: int = 1, min_obs: int = 2):
    """Cross-correlation function ``corr(r1_t, r2_{t+lag})``.

    Returns an array ordered by lags ``[-max_lag, ..., max_lag]``. Positive
    lags mean asset 1 is compared with future asset-2 returns.
    """

    a = aggregate_returns(returns_1, dt=dt)
    b = aggregate_returns(returns_2, dt=dt)
    if a.shape != b.shape:
        raise ValueError(f"return vectors must share shape, got {a.shape} and {b.shape}")
    max_lag = int(max_lag)
    if max_lag < 0:
        raise ValueError("max_lag must be non-negative")

    values = []
    n = a.size
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            x = a[-lag:]
            y = b[: n + lag]
        elif lag > 0:
            x = a[: n - lag]
            y = b[lag:]
        else:
            x = a
            y = b
        if x.size < min_obs:
            values.append(np.nan)
        else:
            values.append(realized_corr(x, y, dt=1, min_obs=min_obs))
    return np.asarray(values, dtype=float)


def ccf_lags(max_lag: int):
    """Return the lag vector matching ``ccf`` output."""
    return np.arange(-int(max_lag), int(max_lag) + 1, dtype=int)


def realized_corr_matrix(returns, dt: int = 1, min_obs: int = 2):
    """Pearson correlation matrix for returns shaped ``(N,T)`` or ``(T,N)``."""
    arr = np.asarray(returns, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"returns must be 2D, got {arr.shape}")
    if arr.shape[0] > arr.shape[1]:
        arr = arr.T
    n_assets = arr.shape[0]
    out = np.eye(n_assets, dtype=float)
    for i in range(n_assets):
        for j in range(i + 1, n_assets):
            value = realized_corr(arr[i], arr[j], dt=dt, min_obs=min_obs)
            out[i, j] = value
            out[j, i] = value
    return out


def cross_corr_from_lob_pair(lob_pair, **mid_kwargs):
    """Return realized correlation for a two-asset LOB path or batch.

    Accepted shapes:
    - ``(2, T, F)`` for one trajectory.
    - ``(B, 2, T, F)`` for a batch; returns the nan-mean across trajectories.
    """
    returns = log_returns_from_lob(lob_pair, **mid_kwargs)
    if returns.ndim == 2 and returns.shape[0] == 2:
        return realized_correlation(returns[0], returns[1])
    if returns.ndim == 3 and returns.shape[1] == 2:
        values = [realized_correlation(path[0], path[1]) for path in returns]
        return float(np.nanmean(values)) if np.isfinite(values).any() else np.nan
    raise ValueError(
        "lob_pair must have shape (2, T, F) or (B, 2, T, F), "
        f"got {np.asarray(lob_pair).shape}"
    )


def compare_cross_corr(real_lob, p0_lob, p1_lob, **mid_kwargs):
    """Compare P0/P1 generated correlations against real-data correlation."""
    real = cross_corr_from_lob_pair(real_lob, **mid_kwargs)
    p0 = cross_corr_from_lob_pair(p0_lob, **mid_kwargs)
    p1 = cross_corr_from_lob_pair(p1_lob, **mid_kwargs)
    p0_abs_error = abs(p0 - real) if np.isfinite(p0) and np.isfinite(real) else np.nan
    p1_abs_error = abs(p1 - real) if np.isfinite(p1) and np.isfinite(real) else np.nan
    return {
        "real_corr": real,
        "p0_corr": p0,
        "p1_corr": p1,
        "p0_abs_error": p0_abs_error,
        "p1_abs_error": p1_abs_error,
        "p1_closer_to_real": bool(p1_abs_error < p0_abs_error)
        if np.isfinite(p0_abs_error) and np.isfinite(p1_abs_error)
        else False,
    }


def _load_npy(path):
    return np.load(Path(path))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-lob", required=True, help="Path to real LOB .npy")
    parser.add_argument("--p0-lob", required=True, help="Path to P0 generated LOB .npy")
    parser.add_argument("--p1-lob", required=True, help="Path to P1 generated LOB .npy")
    parser.add_argument("--bid-price-col", type=int, default=0)
    parser.add_argument("--ask-price-col", type=int, default=2)
    args = parser.parse_args(argv)

    result = compare_cross_corr(
        _load_npy(args.real_lob),
        _load_npy(args.p0_lob),
        _load_npy(args.p1_lob),
        bid_price_col=args.bid_price_col,
        ask_price_col=args.ask_price_col,
    )
    for key, value in result.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
