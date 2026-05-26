"""Quick P2 sanity check for generated spread distributions.

This intentionally stays lighter than the P4 metrics. It compares absolute
spread distributions from real data, P1 samples, and P2 samples using summary
statistics plus a shared-bin histogram L1 distance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np


def summarize_abs_spread(spread: np.ndarray) -> Dict[str, float]:
    values = np.abs(np.asarray(spread, dtype=float)).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("spread array contains no finite values")
    return {
        "mean_abs": float(values.mean()),
        "std_abs": float(values.std()),
        "p50_abs": float(np.quantile(values, 0.50)),
        "p95_abs": float(np.quantile(values, 0.95)),
        "p99_abs": float(np.quantile(values, 0.99)),
    }


def histogram_l1(reference: np.ndarray, candidate: np.ndarray, bins: int = 100) -> float:
    ref = np.abs(np.asarray(reference, dtype=float)).reshape(-1)
    cand = np.abs(np.asarray(candidate, dtype=float)).reshape(-1)
    ref = ref[np.isfinite(ref)]
    cand = cand[np.isfinite(cand)]
    if ref.size == 0 or cand.size == 0:
        raise ValueError("reference and candidate spreads must contain finite values")
    lo = min(ref.min(), cand.min())
    hi = max(ref.max(), cand.max())
    if lo == hi:
        return 0.0
    edges = np.linspace(lo, hi, bins + 1)
    ref_hist, _ = np.histogram(ref, bins=edges, density=True)
    cand_hist, _ = np.histogram(cand, bins=edges, density=True)
    widths = np.diff(edges)
    return float(np.sum(np.abs(ref_hist - cand_hist) * widths))


def compare_spread_distributions(
    real_spread: np.ndarray,
    p1_spread: np.ndarray,
    p2_spread: np.ndarray,
    bins: int = 100,
) -> Dict[str, object]:
    p1_distance = histogram_l1(real_spread, p1_spread, bins=bins)
    p2_distance = histogram_l1(real_spread, p2_spread, bins=bins)
    return {
        "real": summarize_abs_spread(real_spread),
        "p1": summarize_abs_spread(p1_spread),
        "p2": summarize_abs_spread(p2_spread),
        "histogram_l1": {
            "p1_vs_real": p1_distance,
            "p2_vs_real": p2_distance,
        },
        "p2_closer_to_real": bool(p2_distance < p1_distance),
    }


def _load(path: str) -> np.ndarray:
    return np.load(Path(path))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", required=True, help="Path to real spread .npy")
    parser.add_argument("--p1", required=True, help="Path to P1 generated spread .npy")
    parser.add_argument("--p2", required=True, help="Path to P2 generated spread .npy")
    parser.add_argument("--bins", type=int, default=100)
    parser.add_argument("--out", default=None, help="Optional JSON output path")
    args = parser.parse_args()

    result = compare_spread_distributions(
        _load(args.real),
        _load(args.p1),
        _load(args.p2),
        bins=args.bins,
    )
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.out is not None:
        Path(args.out).write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
