"""Quick P2/P3 sanity check for generated spread distributions.

This intentionally stays lighter than the P4 metrics. It compares absolute
spread distributions from real data and generated samples using summary
statistics plus a shared-bin histogram L1 distance. When P3 spreads are
provided, it also reports excursion-duration tails above ``delta_base``.
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


def excursion_durations(spread: np.ndarray, delta_base: float, time_axis: int = 0) -> np.ndarray:
    """Return consecutive durations where ``|spread| > delta_base``.

    The selected time axis is preserved and all other axes are treated as
    independent series.
    """
    values = np.abs(np.asarray(spread, dtype=float))
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("spread array contains no finite values")

    values = np.abs(np.asarray(spread, dtype=float))
    values = np.moveaxis(values, time_axis, 0)
    series = values.reshape(values.shape[0], -1)
    durations = []
    for col in range(series.shape[1]):
        run = 0
        for is_outside in np.isfinite(series[:, col]) & (series[:, col] > delta_base):
            if is_outside:
                run += 1
            elif run:
                durations.append(run)
                run = 0
        if run:
            durations.append(run)
    return np.asarray(durations, dtype=float)


def summarize_excursions(spread: np.ndarray, delta_base: float, time_axis: int = 0) -> Dict[str, float]:
    durations = excursion_durations(spread, delta_base=delta_base, time_axis=time_axis)
    if durations.size == 0:
        return {
            "count": 0,
            "mean_duration": 0.0,
            "p50_duration": 0.0,
            "p95_duration": 0.0,
            "p99_duration": 0.0,
            "max_duration": 0.0,
        }
    return {
        "count": int(durations.size),
        "mean_duration": float(durations.mean()),
        "p50_duration": float(np.quantile(durations, 0.50)),
        "p95_duration": float(np.quantile(durations, 0.95)),
        "p99_duration": float(np.quantile(durations, 0.99)),
        "max_duration": float(durations.max()),
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
    p3_spread: np.ndarray | None = None,
    bins: int = 100,
    delta_base: float | None = None,
    time_axis: int = 0,
) -> Dict[str, object]:
    p1_distance = histogram_l1(real_spread, p1_spread, bins=bins)
    p2_distance = histogram_l1(real_spread, p2_spread, bins=bins)
    result = {
        "real": summarize_abs_spread(real_spread),
        "p1": summarize_abs_spread(p1_spread),
        "p2": summarize_abs_spread(p2_spread),
        "histogram_l1": {
            "p1_vs_real": p1_distance,
            "p2_vs_real": p2_distance,
        },
        "p2_closer_to_real": bool(p2_distance < p1_distance),
    }
    if p3_spread is not None:
        p3_distance = histogram_l1(real_spread, p3_spread, bins=bins)
        result["p3"] = summarize_abs_spread(p3_spread)
        result["histogram_l1"]["p3_vs_real"] = p3_distance
        if delta_base is None:
            delta_base = float(np.median(np.abs(np.asarray(real_spread, dtype=float))))
        excursions = {
            "delta_base": float(delta_base),
            "real": summarize_excursions(real_spread, delta_base, time_axis=time_axis),
            "p1": summarize_excursions(p1_spread, delta_base, time_axis=time_axis),
            "p2": summarize_excursions(p2_spread, delta_base, time_axis=time_axis),
            "p3": summarize_excursions(p3_spread, delta_base, time_axis=time_axis),
        }
        result["excursions"] = excursions
        result["p3_reduces_p95_duration_vs_p2"] = bool(
            excursions["p3"]["p95_duration"] < excursions["p2"]["p95_duration"]
        )
    return result


def _load(path: str) -> np.ndarray:
    return np.load(Path(path))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", required=True, help="Path to real spread .npy")
    parser.add_argument("--p1", required=True, help="Path to P1 generated spread .npy")
    parser.add_argument("--p2", required=True, help="Path to P2 generated spread .npy")
    parser.add_argument("--p3", default=None, help="Optional path to P3 generated spread .npy")
    parser.add_argument("--bins", type=int, default=100)
    parser.add_argument("--delta-base", type=float, default=None)
    parser.add_argument("--time-axis", type=int, default=0)
    parser.add_argument("--out", default=None, help="Optional JSON output path")
    args = parser.parse_args()

    result = compare_spread_distributions(
        _load(args.real),
        _load(args.p1),
        _load(args.p2),
        p3_spread=None if args.p3 is None else _load(args.p3),
        bins=args.bins,
        delta_base=args.delta_base,
        time_axis=args.time_axis,
    )
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.out is not None:
        Path(args.out).write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
