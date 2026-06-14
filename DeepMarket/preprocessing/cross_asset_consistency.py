"""Cross-asset consistency metric (bucket 4.3, metric (3)).

The headline bucket-4 claim is not that coupling/no-arb lifts the aggregate
lob_bench mean (it won't -- §3.4k/§4.3: the signal hides under spread/timing
noise), but that they make the *cross-asset price relation* of generated rollouts
match reality. This module quantifies that relation directly.

For a structured universe it computes the **basis** of each spread group
(``mid[etf] - sum_i w_i mid[i]``; for a spot/perp pair that is ``perp - spot``)
from per-asset mid series, then compares a generated rollout's basis
distribution to the real one via 1-D Wasserstein + L1-of-histogram + moment
gaps. A coupled / no-arb model should sit closer to real than an independent
baseline. Lower = closer to real.

Works on the LOCF-aligned per-asset arrays (row i = same instant across assets),
so it plugs into both the real data and an aligned generated rollout.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

import constants as cst

_ASK0 = cst.LEN_ORDER + 0
_BID0 = cst.LEN_ORDER + 2


def mids_from_arrays(arrays: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Best-mid series per asset from row-aligned adapter arrays (truncated to min len)."""
    n = min(len(a) for a in arrays.values())
    out = {}
    for name, a in arrays.items():
        a = np.asarray(a, dtype=np.float64)[:n]
        out[name] = 0.5 * (a[:, _ASK0] + a[:, _BID0])
    return out


def spread_groups(universe):
    """Deterministic ``[(etf_idx, [(asset_idx, weight), ...]), ...]`` from a universe.

    Inlined (mirrors arbitrage/spread_computer.spread_groups) so the metric has no
    dependency on the P2/P3 modules, which aren't deployed on the training cluster.
    """
    return [
        (int(etf_idx), [(int(i), float(w)) for i, w in sorted(basket.items())])
        for etf_idx, basket in sorted(universe.etf_basket_weights.items())
    ]


def basis_series(mids: Dict[str, np.ndarray], universe) -> Dict[str, np.ndarray]:
    """Per spread-group basis ``mid[etf] - sum_i w_i mid[i]`` (group key = etf name)."""
    names = list(universe.assets)
    out = {}
    for etf_idx, basket in spread_groups(universe):
        etf = names[etf_idx]
        b = mids[etf].copy()
        for i, w in basket:
            b = b - float(w) * mids[names[i]]
        out[etf] = b
    return out


def _wasserstein1(a: np.ndarray, b: np.ndarray, q: int = 512) -> float:
    """1-D Wasserstein-1 between two empirical samples via quantile differences."""
    a = np.sort(np.asarray(a, dtype=np.float64))
    b = np.sort(np.asarray(b, dtype=np.float64))
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    p = (np.arange(q) + 0.5) / q
    qa = np.quantile(a, p)
    qb = np.quantile(b, p)
    return float(np.mean(np.abs(qa - qb)))


def _hist_l1(a: np.ndarray, b: np.ndarray, bins: int = 100) -> float:
    """L1 distance of normalized histograms on a shared range."""
    lo = min(a.min(), b.min())
    hi = max(a.max(), b.max())
    if not np.isfinite([lo, hi]).all() or hi <= lo:
        return float("nan")
    edges = np.linspace(lo, hi, bins + 1)
    ha, _ = np.histogram(a, edges, density=True)
    hb, _ = np.histogram(b, edges, density=True)
    width = (hi - lo) / bins
    return float(np.sum(np.abs(ha - hb)) * width)


def consistency(basis_real: np.ndarray, basis_gen: np.ndarray) -> Dict[str, float]:
    """Distribution distance + moment gaps between real and generated basis."""
    br = np.asarray(basis_real, dtype=np.float64)
    bg = np.asarray(basis_gen, dtype=np.float64)
    br = br[np.isfinite(br)]
    bg = bg[np.isfinite(bg)]
    return {
        "wasserstein": _wasserstein1(br, bg),
        "hist_l1": _hist_l1(br, bg),
        "mean_real": float(br.mean()), "mean_gen": float(bg.mean()),
        "std_real": float(br.std()), "std_gen": float(bg.std()),
        "absmean_real": float(np.abs(br).mean()), "absmean_gen": float(np.abs(bg).mean()),
    }


def evaluate(real_arrays: Dict[str, np.ndarray], gen_arrays: Dict[str, np.ndarray],
             universe) -> Dict[str, Dict[str, float]]:
    """Per-group consistency between real and generated row-aligned arrays."""
    br = basis_series(mids_from_arrays(real_arrays), universe)
    bg = basis_series(mids_from_arrays(gen_arrays), universe)
    return {g: consistency(br[g], bg[g]) for g in br}


__all__ = ["mids_from_arrays", "basis_series", "consistency", "evaluate"]
