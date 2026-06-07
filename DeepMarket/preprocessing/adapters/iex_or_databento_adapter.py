"""Adapter for low-cost equity L2 snapshots such as IEX HIST or Databento MBP."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import constants as cst
from preprocessing.adapters.common import (
    EXPECTED_LOB_COLS,
    read_table,
    save_lobster_like_npy,
    save_split_npys,
    synthesize_orders_from_lob,
    timestamps_to_seconds,
    validate_lob_array,
)


TIMESTAMP_CANDIDATES = ("ts_event", "ts_recv", "timestamp", "time", "datetime")


def equity_frame_to_lob(
    frame: pd.DataFrame,
    *,
    levels: int = cst.N_LOB_LEVELS,
) -> np.ndarray:
    """Extract top-10 LOB columns from common MBP/snapshot schemas."""
    if levels > cst.N_LOB_LEVELS:
        raise ValueError(f"levels cannot exceed {cst.N_LOB_LEVELS}, got {levels}")

    lob = np.zeros((len(frame), EXPECTED_LOB_COLS), dtype=np.float32)
    for level in range(levels):
        ask_px = _find_column(frame, _ask_price_candidates(level))
        ask_sz = _find_column(frame, _ask_size_candidates(level))
        bid_px = _find_column(frame, _bid_price_candidates(level))
        bid_sz = _find_column(frame, _bid_size_candidates(level))
        if ask_px is None or ask_sz is None or bid_px is None or bid_sz is None:
            if level == 0:
                raise ValueError(
                    "Could not infer level-1 book columns. Expected names like "
                    "ask_px_00/ask_sz_00/bid_px_00/bid_sz_00 or "
                    "sell1/vsell1/buy1/vbuy1."
                )
            break
        base = level * cst.LEN_LEVEL
        lob[:, base + 0] = frame[ask_px].to_numpy(dtype=np.float32)
        lob[:, base + 1] = frame[ask_sz].to_numpy(dtype=np.float32)
        lob[:, base + 2] = frame[bid_px].to_numpy(dtype=np.float32)
        lob[:, base + 3] = frame[bid_sz].to_numpy(dtype=np.float32)

    return validate_lob_array(lob)


def equity_frame_to_array(
    frame: pd.DataFrame,
    *,
    timestamp_col: str | None = None,
    levels: int = cst.N_LOB_LEVELS,
) -> np.ndarray:
    lob = equity_frame_to_lob(frame, levels=levels)
    timestamp_col = timestamp_col or _find_column(frame, TIMESTAMP_CANDIDATES)
    timestamps = timestamps_to_seconds(frame[timestamp_col]) if timestamp_col else None
    orders = synthesize_orders_from_lob(lob, timestamps=timestamps)
    return np.concatenate([orders, lob], axis=1).astype(np.float32, copy=False)


def equity_file_to_npy(
    input_path: str | Path,
    output_path: str | Path,
    *,
    source_name: str = "iex_or_databento",
    timestamp_col: str | None = None,
    levels: int = cst.N_LOB_LEVELS,
) -> Path:
    frame = read_table(input_path)
    arr = equity_frame_to_array(frame, timestamp_col=timestamp_col, levels=levels)
    return save_lobster_like_npy(
        arr[:, :cst.LEN_ORDER],
        arr[:, cst.LEN_ORDER:],
        output_path,
        manifest={
            "adapter": "iex_or_databento_adapter",
            "source": source_name,
            "input_path": str(input_path),
            "timestamp_col": timestamp_col,
            "levels": levels,
            "warning": "Snapshot-derived order tokens are proxies, not replayable Level-3 messages.",
        },
    )


def equity_file_to_splits(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    split_rates: tuple[float, float, float] = cst.SPLIT_RATES,
    source_name: str = "iex_or_databento",
    timestamp_col: str | None = None,
    levels: int = cst.N_LOB_LEVELS,
) -> dict[str, Path]:
    frame = read_table(input_path)
    arr = equity_frame_to_array(frame, timestamp_col=timestamp_col, levels=levels)
    return save_split_npys(
        arr,
        output_dir,
        split_rates=split_rates,
        manifest={
            "adapter": "iex_or_databento_adapter",
            "source": source_name,
            "input_path": str(input_path),
            "timestamp_col": timestamp_col,
            "levels": levels,
            "warning": "Snapshot-derived order tokens are proxies, not replayable Level-3 messages.",
        },
    )


def _find_column(frame: pd.DataFrame, candidates) -> str | None:
    columns = {str(col).lower(): col for col in frame.columns}
    for candidate in candidates:
        key = candidate.lower()
        if key in columns:
            return columns[key]
    return None


def _ask_price_candidates(level: int) -> tuple[str, ...]:
    one = level + 1
    two = f"{level:02d}"
    return (
        f"ask_px_{two}", f"ask_price_{two}", f"ask_px_{one}", f"ask_price_{one}",
        f"ask{one}_price", f"ask{one}", f"sell{one}",
    )


def _ask_size_candidates(level: int) -> tuple[str, ...]:
    one = level + 1
    two = f"{level:02d}"
    return (
        f"ask_sz_{two}", f"ask_size_{two}", f"ask_qty_{two}",
        f"ask_sz_{one}", f"ask_size_{one}", f"ask_qty_{one}",
        f"ask{one}_size", f"ask{one}_qty", f"vsell{one}",
    )


def _bid_price_candidates(level: int) -> tuple[str, ...]:
    one = level + 1
    two = f"{level:02d}"
    return (
        f"bid_px_{two}", f"bid_price_{two}", f"bid_px_{one}", f"bid_price_{one}",
        f"bid{one}_price", f"bid{one}", f"buy{one}",
    )


def _bid_size_candidates(level: int) -> tuple[str, ...]:
    one = level + 1
    two = f"{level:02d}"
    return (
        f"bid_sz_{two}", f"bid_size_{two}", f"bid_qty_{two}",
        f"bid_sz_{one}", f"bid_size_{one}", f"bid_qty_{one}",
        f"bid{one}_size", f"bid{one}_qty", f"vbuy{one}",
    )
