"""Lead-lag diagnostics from cross-correlation functions."""

from __future__ import annotations

import numpy as np


def lag_vector(ccf_values, max_lag: int | None = None):
    values = np.asarray(ccf_values, dtype=float)
    if max_lag is None:
        if values.size % 2 != 1:
            raise ValueError("max_lag is required when ccf length is even")
        max_lag = values.size // 2
    return np.arange(-int(max_lag), int(max_lag) + 1, dtype=int)


def peak_lag(ccf_values, max_lag: int | None = None, use_abs: bool = True) -> int:
    """Return the lag at the largest finite CCF magnitude/value."""

    values = np.asarray(ccf_values, dtype=float).reshape(-1)
    lags = lag_vector(values, max_lag=max_lag)
    if lags.size != values.size:
        raise ValueError(f"lag vector length {lags.size} does not match CCF length {values.size}")
    score = np.abs(values) if use_abs else values
    finite = np.isfinite(score)
    if not finite.any():
        return 0
    masked = np.where(finite, score, -np.inf)
    return int(lags[int(np.argmax(masked))])


def lead_lag_error(
    real_ccf,
    gen_ccf,
    max_lag: int | None = None,
    mode: str = "peak_l1",
    use_abs: bool = True,
) -> float:
    """Distance between real and generated lead-lag structure.

    ``mode="peak_l1"`` returns the absolute difference between peak-lag
    locations. ``mode="l2"`` returns the finite-value Euclidean distance
    between the two full CCF vectors.
    """

    real = np.asarray(real_ccf, dtype=float).reshape(-1)
    gen = np.asarray(gen_ccf, dtype=float).reshape(-1)
    if real.shape != gen.shape:
        raise ValueError(f"CCF vectors must share shape, got {real.shape} and {gen.shape}")
    if mode == "peak_l1":
        return float(abs(peak_lag(real, max_lag=max_lag, use_abs=use_abs) - peak_lag(gen, max_lag=max_lag, use_abs=use_abs)))
    if mode == "l2":
        mask = np.isfinite(real) & np.isfinite(gen)
        if not mask.any():
            return np.nan
        return float(np.linalg.norm(real[mask] - gen[mask]))
    raise ValueError(f"unsupported lead_lag_error mode {mode!r}")


__all__ = ["lag_vector", "lead_lag_error", "peak_lag"]
