"""Lead-lag verification for structured asset pairs (task bucket 4 pre-tool).

Before training a coupled/no-arb model on a pair, *verify* the pair actually has
the structure the mechanism needs -- don't assume it (the BTC+ETH mistake). This
module quantifies lead-lag two ways:

  * **Lagged cross-correlation** of mid-price log-returns on a regular grid:
    interpretable peak lag + sign (which asset leads) + peak correlation.
  * **Hayashi-Yoshida** cross-correlation, designed for *asynchronous*,
    non-uniformly-sampled tick data (no resampling bias), with a coarse
    lead-lag scan via the Hoffmann-Rosenbaum-Yoshida shifted contrast.

Run only pairs with a significant, sizeable lead-lag into the coupled training;
record the rejected ones (negative results are useful too).

CLI::

    python -m preprocessing.lead_lag A.npy B.npy --grid-ms 100 --max-lag-ms 2000

The ``.npy`` are *un-normalized* adapter arrays (price in real units); absolute
time is reconstructed as the cumulative sum of the inter-arrival column. Pass an
``<asset>.t.npy`` sidecar of absolute seconds (written by the build driver) for
exact cross-asset clock alignment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

import constants as cst

_ASK_PX_0 = cst.LEN_ORDER + 0   # col 6
_BID_PX_0 = cst.LEN_ORDER + 2   # col 8


def reconstruct_mid(arr: np.ndarray, abs_t: np.ndarray | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(t_seconds, mid)`` from an un-normalized adapter array.

    ``abs_t`` overrides the reconstructed clock (cumsum of inter-arrivals).
    Rows with a non-positive best ask/bid are dropped.
    """
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < cst.LEN_ORDER + cst.LEN_LEVEL:
        raise ValueError(f"expected adapter array (T,>={cst.LEN_ORDER + cst.LEN_LEVEL}), got {arr.shape}")
    t = np.cumsum(arr[:, 0]) if abs_t is None else np.asarray(abs_t, dtype=np.float64)
    ask = arr[:, _ASK_PX_0]
    bid = arr[:, _BID_PX_0]
    mid = 0.5 * (ask + bid)
    valid = (ask > 0) & (bid > 0) & np.isfinite(mid) & np.isfinite(t)
    return t[valid], mid[valid]


# --- regular-grid lagged cross-correlation -----------------------------------

def _grid_log_returns(t: np.ndarray, mid: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Last-observation-carried-forward sample onto ``grid``, then log-returns."""
    idx = np.searchsorted(t, grid, side="right") - 1
    idx = np.clip(idx, 0, len(mid) - 1)
    p = mid[idx]
    p = np.where(p > 0, p, np.nan)
    return np.diff(np.log(p))


def lagged_xcorr(ra: np.ndarray, rb: np.ndarray, max_lag: int) -> Tuple[np.ndarray, np.ndarray]:
    """Pearson corr of ``ra[t]`` vs ``rb[t+lag]`` for lag in ``[-max_lag, max_lag]``.

    A positive peak lag means **A leads B** (A's return today predicts B's return
    ``lag`` steps later). Returns ``(lags, corr)``.
    """
    ra = np.asarray(ra, dtype=np.float64)
    rb = np.asarray(rb, dtype=np.float64)
    n = min(len(ra), len(rb))
    ra, rb = ra[:n], rb[:n]
    lags = np.arange(-max_lag, max_lag + 1)
    corr = np.full(lags.shape, np.nan)
    for k, lag in enumerate(lags):
        if lag >= 0:
            x, y = ra[: n - lag], rb[lag:]
        else:
            x, y = ra[-lag:], rb[: n + lag]
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 16:
            continue
        x, y = x[m], y[m]
        sx, sy = x.std(), y.std()
        if sx < 1e-12 or sy < 1e-12:
            continue
        corr[k] = np.mean((x - x.mean()) * (y - y.mean())) / (sx * sy)
    return lags, corr


# --- Hayashi-Yoshida (asynchronous) ------------------------------------------

def _hy_cov(a_lo, a_hi, dxa, b_lo, b_hi, dyb) -> float:
    """HY cross-covariance via a two-pointer sweep over sorted increments."""
    cov = 0.0
    j_start = 0
    nB = len(dyb)
    for i in range(len(dxa)):
        alo, ahi = a_lo[i], a_hi[i]
        while j_start < nB and b_hi[j_start] <= alo:
            j_start += 1
        j = j_start
        while j < nB and b_lo[j] < ahi:
            cov += dxa[i] * dyb[j]
            j += 1
    return cov


def hayashi_yoshida(
    ta: np.ndarray,
    pa: np.ndarray,
    tb: np.ndarray,
    pb: np.ndarray,
    *,
    lag_grid_s: np.ndarray | None = None,
    max_points: int = 100_000,
) -> Dict[str, float]:
    """HY correlation (lag 0) plus a coarse lead-lag scan over ``lag_grid_s``.

    Each asset's mid is converted to log-increments over its native intervals;
    overlapping intervals contribute to the cross-covariance. The scan shifts B's
    clock by theta; the peak |cov| theta is the HY lead-lag (theta>0 => A leads).
    """
    def _prep(t, p):
        t = np.asarray(t, dtype=np.float64)
        p = np.asarray(p, dtype=np.float64)
        if len(t) > max_points:  # uniform subsample to keep the sweep cheap
            sel = np.linspace(0, len(t) - 1, max_points).astype(int)
            t, p = t[sel], p[sel]
        lp = np.log(np.where(p > 0, p, np.nan))
        d = np.diff(lp)
        lo, hi = t[:-1], t[1:]
        good = np.isfinite(d) & (hi > lo)
        return lo[good], hi[good], d[good]

    a_lo, a_hi, dxa = _prep(ta, pa)
    b_lo, b_hi, dyb = _prep(tb, pb)
    rv_a = float(np.sum(dxa ** 2))
    rv_b = float(np.sum(dyb ** 2))
    denom = np.sqrt(rv_a * rv_b) if rv_a > 0 and rv_b > 0 else np.nan

    cov0 = _hy_cov(a_lo, a_hi, dxa, b_lo, b_hi, dyb)
    out = {"hy_corr": float(cov0 / denom) if np.isfinite(denom) else float("nan")}

    if lag_grid_s is not None and np.isfinite(denom):
        best_theta, best_abs, best_corr = 0.0, -1.0, float("nan")
        for theta in lag_grid_s:
            cov = _hy_cov(a_lo, a_hi, dxa, b_lo + theta, b_hi + theta, dyb)
            c = cov / denom
            if abs(c) > best_abs:
                best_abs, best_theta, best_corr = abs(c), float(theta), float(c)
        out["hy_leadlag_s"] = best_theta
        out["hy_leadlag_corr"] = best_corr
    return out


# --- top-level verdict --------------------------------------------------------

def analyze(
    ta: np.ndarray,
    mid_a: np.ndarray,
    tb: np.ndarray,
    mid_b: np.ndarray,
    *,
    name_a: str = "A",
    name_b: str = "B",
    grid_ms: float = 100.0,
    max_lag_ms: float = 2000.0,
    min_abs_corr: float = 0.03,
    run_hy: bool = True,
) -> Dict[str, object]:
    """Return a lead-lag verdict dict for a pair given abs times + mids."""
    grid_s = grid_ms / 1000.0
    t0 = max(ta.min(), tb.min())
    t1 = min(ta.max(), tb.max())
    if not (t1 > t0):
        raise ValueError("assets do not overlap in time")
    grid = np.arange(t0, t1, grid_s)
    ra = _grid_log_returns(ta, mid_a, grid)
    rb = _grid_log_returns(tb, mid_b, grid)
    max_lag = max(1, int(round(max_lag_ms / grid_ms)))
    lags, corr = lagged_xcorr(ra, rb, max_lag)

    finite = np.isfinite(corr)
    if finite.any():
        kbest = int(np.nanargmax(np.abs(corr)))
        peak_lag_ms = float(lags[kbest] * grid_ms)
        peak_corr = float(corr[kbest])
        contemp = float(corr[lags == 0][0]) if (lags == 0).any() else float("nan")
    else:
        peak_lag_ms, peak_corr, contemp = float("nan"), float("nan"), float("nan")

    leader = name_a if peak_lag_ms > 0 else (name_b if peak_lag_ms < 0 else "synchronous")
    verdict: Dict[str, object] = {
        "pair": [name_a, name_b],
        "grid_ms": grid_ms,
        "max_lag_ms": max_lag_ms,
        "overlap_seconds": float(t1 - t0),
        "n_grid": int(len(grid)),
        "xcorr_peak_lag_ms": peak_lag_ms,
        "xcorr_peak_corr": peak_corr,
        "xcorr_contemporaneous": contemp,
        "leader": leader,
        "significant": bool(
            np.isfinite(peak_corr)
            and abs(peak_corr) >= min_abs_corr
            and abs(peak_lag_ms) > 0
        ),
    }
    if run_hy:
        lag_grid = np.arange(-max_lag, max_lag + 1) * grid_s
        verdict.update(hayashi_yoshida(ta, mid_a, tb, mid_b, lag_grid_s=lag_grid))
    return verdict


def from_arrays(arr_a, arr_b, *, t_a=None, t_b=None, **kwargs) -> Dict[str, object]:
    """Convenience: verdict from two adapter arrays (+ optional abs-time sidecars)."""
    ta, ma = reconstruct_mid(arr_a, abs_t=t_a)
    tb, mb = reconstruct_mid(arr_b, abs_t=t_b)
    return analyze(ta, ma, tb, mb, **kwargs)


def _load_with_sidecar(path: Path) -> Tuple[np.ndarray, np.ndarray | None]:
    arr = np.load(path)
    side = path.with_suffix(".t.npy")
    return arr, (np.load(side) if side.exists() else None)


def main() -> None:
    ap = argparse.ArgumentParser(description="Lead-lag verification for an asset pair.")
    ap.add_argument("npy_a")
    ap.add_argument("npy_b")
    ap.add_argument("--grid-ms", type=float, default=100.0)
    ap.add_argument("--max-lag-ms", type=float, default=2000.0)
    ap.add_argument("--min-abs-corr", type=float, default=0.03)
    ap.add_argument("--no-hy", action="store_true", help="skip Hayashi-Yoshida")
    ap.add_argument("--out", default=None, help="write verdict JSON here")
    args = ap.parse_args()

    pa, ta = _load_with_sidecar(Path(args.npy_a))
    pb, tb = _load_with_sidecar(Path(args.npy_b))
    verdict = from_arrays(
        pa, pb, t_a=ta, t_b=tb,
        name_a=Path(args.npy_a).stem, name_b=Path(args.npy_b).stem,
        grid_ms=args.grid_ms, max_lag_ms=args.max_lag_ms,
        min_abs_corr=args.min_abs_corr, run_hy=not args.no_hy,
    )
    text = json.dumps(verdict, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
