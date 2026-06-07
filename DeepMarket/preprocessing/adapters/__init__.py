"""Low-cost data adapters that emit DeepMarket-compatible ``.npy`` files."""

from preprocessing.adapters.common import (
    EXPECTED_TOTAL_COLS,
    ORDER_COLUMNS,
    lobster_like_array,
    save_lobster_like_npy,
    save_split_npys,
    synthesize_orders_from_lob,
)
from preprocessing.adapters.fi2010_adapter import (
    fi2010_features_to_array,
    fi2010_file_to_npy,
    fi2010_file_to_splits,
)
from preprocessing.adapters.iex_or_databento_adapter import (
    equity_file_to_npy,
    equity_file_to_splits,
    equity_frame_to_array,
    equity_frame_to_lob,
)
from preprocessing.adapters.lobster_adapter import (
    lobster_csv_to_npy,
    lobster_csv_to_splits,
    lobster_frames_to_array,
)

__all__ = [
    "EXPECTED_TOTAL_COLS",
    "ORDER_COLUMNS",
    "equity_file_to_npy",
    "equity_file_to_splits",
    "equity_frame_to_array",
    "equity_frame_to_lob",
    "fi2010_features_to_array",
    "fi2010_file_to_npy",
    "fi2010_file_to_splits",
    "lobster_csv_to_npy",
    "lobster_csv_to_splits",
    "lobster_frames_to_array",
    "lobster_like_array",
    "save_lobster_like_npy",
    "save_split_npys",
    "synthesize_orders_from_lob",
]
