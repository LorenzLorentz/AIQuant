"""Common helpers for converting low-cost LOB data to DeepMarket arrays.

The multi-asset training path consumes one ``.npy`` per asset. Each row must be

``[time, event_type, size, price, direction, depth | ask_px, ask_sz, bid_px, bid_sz] * 10``.

Adapters in this package keep that contract explicit so non-LOBSTER sources
can be used without changing the model code.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import constants as cst


ORDER_COLUMNS = ["time", "event_type", "size", "price", "direction", "depth"]
EXPECTED_ORDER_COLS = cst.LEN_ORDER
EXPECTED_LOB_COLS = cst.N_LOB_LEVELS * cst.LEN_LEVEL
EXPECTED_TOTAL_COLS = EXPECTED_ORDER_COLS + EXPECTED_LOB_COLS


def read_table(path: str | Path) -> pd.DataFrame:
    """Read CSV or parquet input by extension."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported table format {suffix!r} for {path}")


def validate_lob_array(lob: np.ndarray) -> np.ndarray:
    lob = np.asarray(lob, dtype=np.float32)
    if lob.ndim != 2 or lob.shape[1] != EXPECTED_LOB_COLS:
        raise ValueError(
            f"lob must be (T, {EXPECTED_LOB_COLS}), got {tuple(lob.shape)}"
        )
    return np.nan_to_num(lob, nan=0.0, posinf=0.0, neginf=0.0)


def validate_order_array(orders: np.ndarray) -> np.ndarray:
    orders = np.asarray(orders, dtype=np.float32)
    if orders.ndim != 2 or orders.shape[1] != EXPECTED_ORDER_COLS:
        raise ValueError(
            f"orders must be (T, {EXPECTED_ORDER_COLS}), got {tuple(orders.shape)}"
        )
    orders = np.nan_to_num(orders, nan=0.0, posinf=0.0, neginf=0.0)
    orders[:, 1] = np.clip(np.rint(orders[:, 1]), 0, 2)
    orders[:, 4] = np.where(orders[:, 4] >= 0, 1.0, -1.0)
    orders[:, 5] = np.maximum(orders[:, 5], 0.0)
    return orders.astype(np.float32, copy=False)


def lobster_like_array(orders: np.ndarray, lob: np.ndarray) -> np.ndarray:
    orders = validate_order_array(orders)
    lob = validate_lob_array(lob)
    if len(orders) != len(lob):
        raise ValueError(f"orders/lob length mismatch: {len(orders)} vs {len(lob)}")
    out = np.concatenate([orders, lob], axis=1).astype(np.float32, copy=False)
    if out.shape[1] != EXPECTED_TOTAL_COLS:
        raise ValueError(f"expected {EXPECTED_TOTAL_COLS} columns, got {out.shape[1]}")
    return out


def save_lobster_like_npy(
    orders: np.ndarray,
    lob: np.ndarray,
    output_path: str | Path,
    *,
    manifest: dict[str, Any] | None = None,
) -> Path:
    """Save a DeepMarket-compatible single-asset array and sidecar manifest."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arr = lobster_like_array(orders, lob)
    np.save(output_path, arr)

    meta = {
        "format": "deepmarket_lobster_like_npy",
        "rows": int(arr.shape[0]),
        "columns": int(arr.shape[1]),
        "order_columns": ORDER_COLUMNS,
        "lob_layout": "ask_price,ask_size,bid_price,bid_size repeated for 10 levels",
        # Adapters emit the LOBSTERDataBuilder *column layout* but do NOT z-score.
        # Callers must z-score (matching LOBSTERDataBuilder) before training, or
        # override this flag if they already did.
        "normalized": False,
    }
    if manifest:
        meta.update(manifest)
    output_path.with_suffix(output_path.suffix + ".manifest.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output_path


def save_split_npys(
    arr: np.ndarray,
    output_dir: str | Path,
    *,
    split_rates: tuple[float, float, float] = cst.SPLIT_RATES,
    names: tuple[str, str, str] = ("train.npy", "val.npy", "test.npy"),
    manifest: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Save train/val/test arrays using DeepMarket's expected filenames."""
    if len(split_rates) != 3:
        raise ValueError(f"split_rates must have length 3, got {split_rates}")
    if any(rate < 0 for rate in split_rates):
        raise ValueError(f"split_rates must be non-negative, got {split_rates}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    n = arr.shape[0]
    train_end = int(n * split_rates[0])
    val_end = train_end + int(n * split_rates[1])
    splits = {
        names[0]: arr[:train_end],
        names[1]: arr[train_end:val_end],
        names[2]: arr[val_end:],
    }
    paths = {}
    for name, split in splits.items():
        path = output_dir / name
        np.save(path, split.astype(np.float32, copy=False))
        paths[name] = path

    meta = {
        "format": "deepmarket_lobster_like_splits",
        "rows_total": int(n),
        "rows": {name: int(split.shape[0]) for name, split in splits.items()},
        "split_rates": list(split_rates),
        # See note in save_lobster_like_npy: output is un-normalized by default.
        "normalized": False,
    }
    if manifest:
        meta.update(manifest)
    (output_dir / "adapter_manifest.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return paths


def synthesize_orders_from_lob(
    lob: np.ndarray,
    *,
    timestamps: np.ndarray | None = None,
) -> np.ndarray:
    """Create pseudo-order tokens from consecutive LOB snapshots.

    This is intentionally conservative and is meant for snapshot-only sources
    such as MBP data or FI-2010 sanity checks. The generated order token is a
    compact proxy, not a replayable Level-3 message.
    """
    lob = validate_lob_array(lob)
    n = lob.shape[0]
    if n == 0:
        return np.zeros((0, EXPECTED_ORDER_COLS), dtype=np.float32)

    ask = lob[:, 0]
    bid = lob[:, 2]
    ask_size = lob[:, 1]
    bid_size = lob[:, 3]
    mid = 0.5 * (ask + bid)
    top_depth = ask_size + bid_size
    mid_delta = np.diff(mid, prepend=mid[0])
    depth_delta = np.diff(top_depth, prepend=top_depth[0])

    if timestamps is None:
        dt = np.ones(n, dtype=np.float32)
    else:
        ts = np.asarray(timestamps, dtype=np.float64)
        if ts.shape[0] != n:
            raise ValueError(f"timestamps length {ts.shape[0]} does not match lob length {n}")
        dt = np.diff(ts, prepend=ts[0]).astype(np.float32)
        if n > 1 and dt[0] == 0:
            dt[0] = np.median(dt[1:]) if np.any(dt[1:] > 0) else 0.0
        dt = np.maximum(dt, 0.0)

    event_type = np.zeros(n, dtype=np.float32)
    event_type[depth_delta < 0] = 1.0
    event_type[np.abs(mid_delta) > 0] = 2.0
    size = np.maximum(np.abs(depth_delta), 1.0).astype(np.float32)
    direction_signal = np.where(mid_delta != 0, mid_delta, bid_size - ask_size)
    direction = np.where(direction_signal >= 0, 1.0, -1.0).astype(np.float32)

    orders = np.stack(
        [
            dt.astype(np.float32),
            event_type,
            size,
            mid.astype(np.float32),
            direction,
            np.zeros(n, dtype=np.float32),
        ],
        axis=1,
    )
    return validate_order_array(orders)


def timestamps_to_seconds(values: pd.Series) -> np.ndarray:
    """Normalize common timestamp columns to seconds from the first row."""
    if pd.api.types.is_numeric_dtype(values):
        raw = values.to_numpy(dtype=np.float64)
        span = np.nanmax(raw) - np.nanmin(raw) if len(raw) else 0.0
        magnitude = np.nanmedian(np.abs(raw)) if len(raw) else 0.0
        # Infer the epoch unit. 2020s epochs by magnitude: seconds ~1e9,
        # milliseconds ~1e12, microseconds ~1e15, nanoseconds ~1e18. ``span``
        # is the fallback for already-relative timestamps (small magnitude,
        # large range over a trading day).
        if magnitude > 1e17 or span > 1e13:
            raw = raw / 1e9   # nanoseconds -> seconds
        elif magnitude > 1e14 or span > 1e10:
            raw = raw / 1e6   # microseconds -> seconds
        elif magnitude > 1e11 or span > 1e7:
            raw = raw / 1e3   # milliseconds -> seconds
        return raw - raw[0] if len(raw) else raw

    parsed = pd.to_datetime(values, utc=True)
    # tz-aware datetime64 -> UTC-naive ns -> seconds (robust across pandas versions;
    # a direct astype("int64") on tz-aware dtypes raises in pandas 2.x).
    ns = parsed.to_numpy("datetime64[ns]").astype("int64")
    seconds = ns.astype(np.float64) / 1e9
    return seconds - seconds[0] if len(seconds) else seconds
