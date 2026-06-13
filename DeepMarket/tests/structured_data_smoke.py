"""Smoke tests for the bucket-4 structured-data tooling (no network).

Invoke from ``DeepMarket/``::

    python -m tests.structured_data_smoke
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import constants as cst
from preprocessing.AssetUniverse import AssetUniverse
from preprocessing.adapters.iex_or_databento_adapter import equity_frame_to_array
from preprocessing.adapters.common import EXPECTED_TOTAL_COLS
from preprocessing import lead_lag, structured_universes
from models.diffusers.multi_asset.arbitrage.spread_computer import spread_groups


def _tardis_frame(rows: int = 64, base: float = 42000.0) -> pd.DataFrame:
    cols = {"exchange": "binance", "symbol": "BTCUSDT",
            "timestamp": 1_704_067_200_000_000 + np.arange(rows) * 100_000}  # us, 100ms
    for lvl in range(25):
        cols[f"asks[{lvl}].price"] = base + 1.0 + lvl
        cols[f"asks[{lvl}].amount"] = 1.0 + 0.1 * lvl
        cols[f"bids[{lvl}].price"] = base - 1.0 - lvl
        cols[f"bids[{lvl}].amount"] = 1.0 + 0.1 * lvl
    return pd.DataFrame(cols)


def test_tardis_adapter():
    arr = equity_frame_to_array(_tardis_frame())
    assert arr.shape[1] == EXPECTED_TOTAL_COLS, arr.shape
    # best ask/bid land in the right columns and are non-zero (parsed, not skipped)
    assert arr[:, cst.LEN_ORDER + 0].min() > 0 and arr[:, cst.LEN_ORDER + 2].min() > 0
    # all 10 levels populated -> 40/40 (no degenerate L1)
    assert (arr[:, cst.LEN_ORDER:] != 0).all()
    print("  ok: Tardis adapter -> (T,46), 40/40 levels")


def test_universes_fill_spread_groups():
    a = structured_universes.get("btc_spot_perp").universe
    assert len(spread_groups(a)) == 1 and a.num_assets == 2
    b = structured_universes.get("btc_cross_exchange").universe
    assert b.num_assets == 3 and len(spread_groups(b)) == 2  # 2 non-ref venues
    c = structured_universes.get("qqq_basket").universe
    groups = spread_groups(c)
    assert c.num_assets == 5 and len(groups) == 1 and len(groups[0][1]) == 4
    # every registered universe yields at least one non-empty spread group
    for name in structured_universes.names():
        assert spread_groups(structured_universes.get(name).universe), name
    print("  ok: A/B/C universes all fill spread_groups")


def test_lead_lag_detects_known_lag():
    rng = np.random.default_rng(0)
    n, lag = 4000, 5  # A leads B by 5 grid steps
    rets = rng.normal(0, 1e-3, n)
    mid_a = 42000.0 * np.exp(np.cumsum(rets))
    mid_b = np.empty_like(mid_a)
    mid_b[lag:] = mid_a[:-lag]
    mid_b[:lag] = mid_a[0]
    mid_b *= np.exp(rng.normal(0, 1e-4, n))  # idiosyncratic noise
    t = np.arange(n) * 0.1  # 100ms regular grid
    v = lead_lag.analyze(t, mid_a, t, mid_b, name_a="A", name_b="B",
                         grid_ms=100.0, max_lag_ms=2000.0, run_hy=True)
    assert v["leader"] == "A", v
    assert abs(v["xcorr_peak_lag_ms"] - lag * 100.0) <= 100.0, v
    assert v["significant"], v
    assert np.isfinite(v["hy_corr"]), v
    print(f"  ok: lead-lag recovered leader=A lag={v['xcorr_peak_lag_ms']}ms "
          f"corr={v['xcorr_peak_corr']:.3f} hy={v['hy_corr']:.3f}")


def main() -> int:
    for fn in (test_tardis_adapter, test_universes_fill_spread_groups,
               test_lead_lag_detects_known_lag):
        fn()
    print("STRUCTURED_SMOKE_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
