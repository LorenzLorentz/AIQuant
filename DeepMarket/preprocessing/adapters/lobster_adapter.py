"""Adapter for raw LOBSTER message/orderbook CSV files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import constants as cst
from preprocessing.adapters.common import (
    EXPECTED_LOB_COLS,
    lobster_like_array,
    save_lobster_like_npy,
    save_split_npys,
)
from utils.utils_data import preprocess_data


MESSAGE_COLUMNS = ["time", "event_type", "order_id", "size", "price", "direction"]
ORDERBOOK_COLUMNS = [
    value
    for level in range(1, cst.N_LOB_LEVELS + 1)
    for value in (f"sell{level}", f"vsell{level}", f"buy{level}", f"vbuy{level}")
]


def read_lobster_frames(
    message_path: str | Path,
    orderbook_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    messages = pd.read_csv(message_path, names=MESSAGE_COLUMNS)
    orderbook = pd.read_csv(orderbook_path, names=ORDERBOOK_COLUMNS)
    return messages, orderbook


def lobster_frames_to_array(
    messages: pd.DataFrame,
    orderbook: pd.DataFrame,
    *,
    chosen_model=cst.Models.TRADES,
    price_divisor: float = 100.0,
) -> np.ndarray:
    """Convert raw LOBSTER frames to DeepMarket's single-asset array format.

    ``preprocess_data`` removes unsupported event types, computes inter-arrival
    times/depth, and returns orderbook/message frames. We then apply the same
    event-type encoding used by TRADES preprocessing, but leave z-score
    normalization to the downstream experiment owner because low-cost sources
    may already be normalized or price-scaled differently.
    """
    if price_divisor <= 0:
        raise ValueError(f"price_divisor must be positive, got {price_divisor}")

    orderbook_proc, messages_proc = preprocess_data(
        [messages.copy(), orderbook.copy()],
        cst.N_LOB_LEVELS,
        chosen_model,
    )
    messages_proc = messages_proc[["time", "event_type", "size", "price", "direction", "depth"]].copy()
    messages_proc["event_type"] = messages_proc["event_type"] - 1.0
    messages_proc["event_type"] = messages_proc["event_type"].replace({2.0: 1.0, 3.0: 2.0})
    messages_proc["price"] = messages_proc["price"] / price_divisor
    orderbook_proc = orderbook_proc.iloc[:, :EXPECTED_LOB_COLS].copy()
    orderbook_proc.iloc[:, ::2] = orderbook_proc.iloc[:, ::2] / price_divisor

    return lobster_like_array(
        messages_proc.to_numpy(dtype=np.float32),
        orderbook_proc.to_numpy(dtype=np.float32),
    )


def lobster_csv_to_npy(
    message_path: str | Path,
    orderbook_path: str | Path,
    output_path: str | Path,
    *,
    source_name: str = "lobster",
    price_divisor: float = 100.0,
) -> Path:
    messages, orderbook = read_lobster_frames(message_path, orderbook_path)
    arr = lobster_frames_to_array(messages, orderbook, price_divisor=price_divisor)
    return save_lobster_like_npy(
        arr[:, :cst.LEN_ORDER],
        arr[:, cst.LEN_ORDER:],
        output_path,
        manifest={
            "adapter": "lobster_adapter",
            "source": source_name,
            "message_path": str(message_path),
            "orderbook_path": str(orderbook_path),
            "price_divisor": price_divisor,
        },
    )


def lobster_csv_to_splits(
    message_path: str | Path,
    orderbook_path: str | Path,
    output_dir: str | Path,
    *,
    split_rates: tuple[float, float, float] = cst.SPLIT_RATES,
    source_name: str = "lobster",
    price_divisor: float = 100.0,
) -> dict[str, Path]:
    messages, orderbook = read_lobster_frames(message_path, orderbook_path)
    arr = lobster_frames_to_array(messages, orderbook, price_divisor=price_divisor)
    return save_split_npys(
        arr,
        output_dir,
        split_rates=split_rates,
        manifest={
            "adapter": "lobster_adapter",
            "source": source_name,
            "message_path": str(message_path),
            "orderbook_path": str(orderbook_path),
            "price_divisor": price_divisor,
        },
    )
