"""P4 smoke test for posterior checks and cross-asset metrics.

Invoke from ``DeepMarket/``:

    python -m tests.p4_smoke
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from evaluation.quantitative_eval.posterior_report import build_posterior_report
from lob_bench.cross_asset import (
    ccf,
    conditional_spillover,
    ks_distance,
    lead_lag_error,
    peak_lag,
    realized_corr,
)
from models.diffusers.multi_asset.arbitrage.consistency_check import (
    consistency_check,
    reference_from_lob,
)
from preprocessing.AssetUniverse import AssetUniverse


def _make_lob(batch: int = 2, steps: int = 96, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    lob = np.zeros((batch, 2, steps, 40), dtype=float)
    time = np.arange(steps, dtype=float)
    for b in range(batch):
        constituent = 100.0 + 0.02 * np.cumsum(rng.normal(size=steps))
        spread = 0.20 * np.sin(time / 7.0) + 0.04 * rng.normal(size=steps)
        spread[30:38] += 0.45
        spread[70:78] -= 0.35
        etf = constituent + spread
        for asset_idx, mid in enumerate((etf, constituent)):
            lob[b, asset_idx, :, 0] = mid + 0.01
            lob[b, asset_idx, :, 2] = mid - 0.01
            lob[b, asset_idx, :, 1] = 1000.0
            lob[b, asset_idx, :, 3] = 1000.0
    return lob


def _zero_spread_lob(real_lob: np.ndarray) -> np.ndarray:
    out = real_lob.copy()
    out[:, 0, :, 0] = out[:, 1, :, 0]
    out[:, 0, :, 2] = out[:, 1, :, 2]
    return out


def check_consistency() -> None:
    universe = AssetUniverse.etf_pair("ETF", "CONST")
    real = _make_lob()
    reference = reference_from_lob(real, universe)
    real_check = consistency_check(real, universe, reference)
    assert real_check["passed"], "real-vs-real consistency should pass"

    degenerate = consistency_check(_zero_spread_lob(real), universe, reference)
    assert not degenerate["passed"], "zero-spread trajectory should fail posterior checks"
    assert degenerate["checks"]["std_abs_spread"] is False
    print("  [ok] consistency check")


def check_cross_asset_metrics() -> None:
    rng = np.random.default_rng(11)
    source = rng.normal(size=256)
    target = np.empty_like(source)
    target[:2] = rng.normal(size=2)
    target[2:] = source[:-2]
    values = ccf(source, target, max_lag=5)
    assert peak_lag(values, max_lag=5) == 2
    assert lead_lag_error(values, values, max_lag=5) == 0.0
    assert realized_corr(source, target) < realized_corr(source[:-2], target[2:])
    assert ks_distance(source, source.copy()) == 0.0
    spill = conditional_spillover(source, target, quantiles=(0.90,))
    assert np.isfinite(spill["q90"])
    print("  [ok] cross-asset metrics")


def check_report_driver() -> None:
    universe = AssetUniverse.etf_pair("ETF", "CONST")
    real = _make_lob(seed=13)
    corners = {
        "P0": _make_lob(seed=31),
        "P1": _make_lob(seed=13) + 0.002,
        "P2": _make_lob(seed=13) + 0.001,
        "P3": _make_lob(seed=13),
    }
    report = build_posterior_report(
        real,
        corners,
        universe,
        reference_lob=real,
        max_lag=4,
        bar_size=1,
    )
    assert report["trend_checks"]["all_four_corners_present"] is True
    assert set(report["corners"]) == {"P0", "P1", "P2", "P3"}
    assert "posterior_consistency" in report["corners"]["P3"]
    assert "lead_lag_error" in report["corners"]["P3"]["cross_asset"]
    json.dumps(report, allow_nan=False)
    print("  [ok] posterior report")


def main() -> int:
    print("[1] consistency")
    check_consistency()
    print("[2] cross-asset metrics")
    check_cross_asset_metrics()
    print("[3] report driver")
    check_report_driver()
    print("\nP4 smoke test PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
