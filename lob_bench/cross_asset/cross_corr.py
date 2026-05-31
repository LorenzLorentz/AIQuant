"""Cross-asset sanity metrics for jointly generated LOB trajectories."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def mid_price_returns(
    book: pd.DataFrame | np.ndarray,
    messages: pd.DataFrame | None = None,
    interval: str | None = None,
) -> pd.Series:
    """Compute mid-price returns from LOBSTER-style top-of-book columns.

    DeepMarket/LOBSTER orderbook columns start with ask price, ask size, bid
    price, bid size. If prices are positive we use log returns; otherwise
    first differences are used so normalized arrays remain supported.
    """
    book_df = _as_frame(book)
    if book_df.shape[1] < 3:
        raise ValueError("book must contain at least ask and bid price columns")

    mid = (book_df.iloc[:, 0].astype(float) + book_df.iloc[:, 2].astype(float)) * 0.5
    if messages is not None and "time" in messages:
        mid.index = pd.to_datetime(messages["time"])
    if interval is not None:
        mid = mid.resample(interval, label="right", closed="left").last().dropna()

    values = np.log(mid.clip(lower=1e-12)) if bool((mid > 0).all()) else mid
    returns = values.diff().dropna()
    returns.name = "mid_price_returns"
    return returns


def realized_cross_corr(
    book_a: pd.DataFrame | np.ndarray,
    book_b: pd.DataFrame | np.ndarray,
    messages_a: pd.DataFrame | None = None,
    messages_b: pd.DataFrame | None = None,
    interval: str | None = None,
) -> float:
    """Correlation of realized mid-price returns for two assets."""
    ret_a = mid_price_returns(book_a, messages_a, interval)
    ret_b = mid_price_returns(book_b, messages_b, interval)
    aligned_a, aligned_b = _align_returns(ret_a, ret_b)
    if len(aligned_a) < 2:
        return float("nan")
    return float(aligned_a.corr(aligned_b))


def compare_realized_cross_corr(
    *,
    real_books: tuple[pd.DataFrame | np.ndarray, pd.DataFrame | np.ndarray],
    gen_books: tuple[pd.DataFrame | np.ndarray, pd.DataFrame | np.ndarray],
    p0_books: tuple[pd.DataFrame | np.ndarray, pd.DataFrame | np.ndarray] | None = None,
    interval: str | None = None,
    **message_pairs: Any,
) -> dict[str, float]:
    """Compare real, P1/generated, and optionally P0 cross correlations."""
    real_messages = message_pairs.get("real_messages", (None, None))
    gen_messages = message_pairs.get("gen_messages", (None, None))
    p0_messages = message_pairs.get("p0_messages", (None, None))

    results = {
        "real": realized_cross_corr(
            real_books[0], real_books[1], real_messages[0], real_messages[1], interval
        ),
        "generated": realized_cross_corr(
            gen_books[0], gen_books[1], gen_messages[0], gen_messages[1], interval
        ),
    }
    results["generated_abs_error"] = abs(results["generated"] - results["real"])

    if p0_books is not None:
        results["p0"] = realized_cross_corr(
            p0_books[0], p0_books[1], p0_messages[0], p0_messages[1], interval
        )
        results["p0_abs_error"] = abs(results["p0"] - results["real"])
        results["generated_improvement_vs_p0"] = (
            results["p0_abs_error"] - results["generated_abs_error"]
        )

    return results


def _as_frame(data: pd.DataFrame | np.ndarray) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data
    return pd.DataFrame(data)


def _align_returns(left: pd.Series, right: pd.Series) -> tuple[pd.Series, pd.Series]:
    if isinstance(left.index, pd.DatetimeIndex) and isinstance(right.index, pd.DatetimeIndex):
        aligned = pd.concat([left, right], axis=1, join="inner").dropna()
        return aligned.iloc[:, 0], aligned.iloc[:, 1]

    length = min(len(left), len(right))
    return (
        pd.Series(left.to_numpy()[:length]),
        pd.Series(right.to_numpy()[:length]),
    )
