"""Smoke tests for low-cost data adapters.

Invoke from ``DeepMarket/``:

    python -m tests.data_adapters_smoke
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import constants as cst
from preprocessing.MultiAssetLOBDataset import MultiAssetLOBDataset
from preprocessing.adapters import (
    EXPECTED_TOTAL_COLS,
    equity_frame_to_array,
    fi2010_features_to_array,
    lobster_frames_to_array,
    save_lobster_like_npy,
    save_split_npys,
)


def _lobster_frames(rows: int = 24, price_shift: float = 0.0):
    messages = pd.DataFrame(
        {
            "time": 34200.0 + np.arange(rows, dtype=float) * 0.1,
            "event_type": np.resize(np.array([1, 3, 4], dtype=float), rows),
            "order_id": np.arange(1000, 1000 + rows),
            "size": 100.0 + np.arange(rows) % 5,
            "price": 10000.0 + price_shift + np.arange(rows),
            "direction": np.where(np.arange(rows) % 2 == 0, 1.0, -1.0),
        }
    )
    orderbook = pd.DataFrame(index=np.arange(rows))
    for level in range(1, cst.N_LOB_LEVELS + 1):
        orderbook[f"sell{level}"] = 10100.0 + price_shift + level
        orderbook[f"vsell{level}"] = 1000.0 + level
        orderbook[f"buy{level}"] = 9900.0 + price_shift - level
        orderbook[f"vbuy{level}"] = 900.0 + level
    return messages, orderbook


def _equity_frame(rows: int = 24, price_shift: float = 0.0):
    frame = pd.DataFrame({"ts_event": 1_700_000_000_000_000_000 + np.arange(rows) * 100_000_000})
    for level in range(cst.N_LOB_LEVELS):
        suffix = f"{level:02d}"
        frame[f"ask_px_{suffix}"] = 100.10 + price_shift + 0.01 * level + 0.001 * np.arange(rows)
        frame[f"ask_sz_{suffix}"] = 1000.0 + level + np.arange(rows) % 3
        frame[f"bid_px_{suffix}"] = 100.00 + price_shift - 0.01 * level + 0.001 * np.arange(rows)
        frame[f"bid_sz_{suffix}"] = 900.0 + level + np.arange(rows) % 2
    return frame


def check_lobster_adapter():
    messages, orderbook = _lobster_frames()
    arr = lobster_frames_to_array(messages, orderbook)
    assert arr.shape == (23, EXPECTED_TOTAL_COLS)
    assert np.isfinite(arr).all()
    assert set(np.unique(arr[:, 1])).issubset({0.0, 1.0, 2.0})
    print("  [ok] LOBSTER adapter")


def check_equity_adapter():
    arr = equity_frame_to_array(_equity_frame())
    assert arr.shape == (24, EXPECTED_TOTAL_COLS)
    assert np.isfinite(arr).all()
    assert set(np.unique(arr[:, 1])).issubset({0.0, 1.0, 2.0})
    assert arr[1, 0] > 0.0
    print("  [ok] IEX/Databento snapshot adapter")


def check_fi2010_adapter():
    rows = 24
    features = np.zeros((rows, 45), dtype=np.float32)
    for level in range(cst.N_LOB_LEVELS):
        base = level * cst.LEN_LEVEL
        features[:, base + 0] = 100.10 + 0.01 * level
        features[:, base + 1] = 1000.0 + level
        features[:, base + 2] = 100.00 - 0.01 * level
        features[:, base + 3] = 900.0 + level
    arr = fi2010_features_to_array(features)
    assert arr.shape == (rows, EXPECTED_TOTAL_COLS)
    assert np.isfinite(arr).all()
    print("  [ok] FI-2010 adapter")


def check_save_and_dataset_load():
    arr_a = equity_frame_to_array(_equity_frame(price_shift=0.0))
    arr_b = equity_frame_to_array(_equity_frame(price_shift=1.0))
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        a_path = save_lobster_like_npy(
            arr_a[:, :cst.LEN_ORDER],
            arr_a[:, cst.LEN_ORDER:],
            tmp_path / "A.npy",
            manifest={"source": "synthetic_A"},
        )
        b_path = save_lobster_like_npy(
            arr_b[:, :cst.LEN_ORDER],
            arr_b[:, cst.LEN_ORDER:],
            tmp_path / "B.npy",
            manifest={"source": "synthetic_B"},
        )
        manifest = json.loads(a_path.with_suffix(".npy.manifest.json").read_text())
        assert manifest["rows"] == arr_a.shape[0]

        dataset = MultiAssetLOBDataset(
            [a_path, b_path],
            seq_size=8,
            gen_seq_size=2,
        )
        cond_orders, x_0, cond_lob = dataset[0]
        assert cond_orders.shape == (2, 6, cst.LEN_ORDER)
        assert x_0.shape == (2, 2, cst.LEN_ORDER)
        assert cond_lob.shape == (2, 7, cst.N_LOB_LEVELS * cst.LEN_LEVEL)

        paths = save_split_npys(arr_a, tmp_path / "splits", split_rates=(0.5, 0.25, 0.25))
        assert set(paths) == {"train.npy", "val.npy", "test.npy"}
        assert np.load(paths["train.npy"]).shape[1] == EXPECTED_TOTAL_COLS
    print("  [ok] save helpers and MultiAssetLOBDataset compatibility")


def main() -> int:
    print("[1] raw LOBSTER")
    check_lobster_adapter()
    print("[2] IEX/Databento")
    check_equity_adapter()
    print("[3] FI-2010")
    check_fi2010_adapter()
    print("[4] save + dataset")
    check_save_and_dataset_load()
    print("\nData adapters smoke test PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
