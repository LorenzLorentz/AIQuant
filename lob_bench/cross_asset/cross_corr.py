"""Cross-asset mid-price return correlation sanity metric.

This module is intentionally small: P1 only needs a coarse check that generated
two-asset trajectories carry non-trivial correlation and move closer to the
real-data correlation than the P0 independent baseline.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


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


def realized_correlation(returns_a, returns_b, min_obs: int = 2):
    """Pearson correlation between two return vectors, ignoring non-finites."""
    a = np.asarray(returns_a, dtype=float).reshape(-1)
    b = np.asarray(returns_b, dtype=float).reshape(-1)
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
