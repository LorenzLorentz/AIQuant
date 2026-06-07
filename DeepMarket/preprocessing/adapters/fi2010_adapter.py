"""Adapter for FI-2010-style feature matrices.

FI-2010 is a public sanity-check dataset, not a Level-3 message dataset. This
adapter maps the first 40 LOB features to the DeepMarket LOB layout and
synthesizes proxy order tokens from snapshot changes.
"""

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
    validate_lob_array,
)


def fi2010_features_to_array(features: np.ndarray | pd.DataFrame) -> np.ndarray:
    values = features.to_numpy(dtype=np.float32) if isinstance(features, pd.DataFrame) else np.asarray(features, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] < EXPECTED_LOB_COLS:
        raise ValueError(
            f"FI-2010 features must be (T, >= {EXPECTED_LOB_COLS}), got {tuple(values.shape)}"
        )
    lob = validate_lob_array(values[:, :EXPECTED_LOB_COLS])
    orders = synthesize_orders_from_lob(lob)
    return np.concatenate([orders, lob], axis=1).astype(np.float32, copy=False)


def fi2010_file_to_npy(
    input_path: str | Path,
    output_path: str | Path,
    *,
    source_name: str = "fi2010",
) -> Path:
    frame = read_table(input_path)
    arr = fi2010_features_to_array(frame)
    return save_lobster_like_npy(
        arr[:, :cst.LEN_ORDER],
        arr[:, cst.LEN_ORDER:],
        output_path,
        manifest={
            "adapter": "fi2010_adapter",
            "source": source_name,
            "input_path": str(input_path),
            "warning": "FI-2010 is feature-level sanity data; do not use for ETF-NAV claims.",
        },
    )


def fi2010_file_to_splits(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    split_rates: tuple[float, float, float] = cst.SPLIT_RATES,
    source_name: str = "fi2010",
) -> dict[str, Path]:
    frame = read_table(input_path)
    arr = fi2010_features_to_array(frame)
    return save_split_npys(
        arr,
        output_dir,
        split_rates=split_rates,
        manifest={
            "adapter": "fi2010_adapter",
            "source": source_name,
            "input_path": str(input_path),
            "warning": "FI-2010 is feature-level sanity data; do not use for ETF-NAV claims.",
        },
    )
