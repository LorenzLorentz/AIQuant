"""Rolling conditioning statistics for graph edge weights."""

from __future__ import annotations

import torch


def compute_rolling_stats(
    cond_orders: torch.Tensor,
    cond_lob: torch.Tensor | None,
    window: int | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Return the P1 four-stat vector for every asset.

    Parameters
    ----------
    cond_orders:
        Raw DeepMarket event tensor with shape ``(B, N, K_cond, 6)``. The
        expected normalized column order is ``time, event_type, size, price,
        direction, depth``.
    cond_lob:
        Raw LOB tensor with shape ``(B, N, K_cond + 1, 40)``. The first and
        third columns are best ask/bid prices in DeepMarket's LOBSTER layout.
    window:
        Optional trailing window size. ``None`` uses the full conditioning
        history.

    Returns
    -------
    stats:
        Tensor of shape ``(B, N, 4)`` containing:
        mid-price change volatility, signed order-flow imbalance,
        cancellation ratio, and mean inter-arrival time.
    """
    if cond_orders.dim() != 4:
        raise ValueError(
            "cond_orders must be (B, N, K_cond, F), "
            f"got {tuple(cond_orders.shape)}"
        )
    if cond_orders.shape[-1] < 6:
        raise ValueError(
            "rolling stats require raw order features with at least 6 columns, "
            f"got {cond_orders.shape[-1]}"
        )

    orders = _tail(cond_orders, dim=2, window=window)
    dtype = orders.dtype
    device = orders.device

    interarrival = orders[..., 0]
    event_type = orders[..., 1]
    size = orders[..., 2]
    direction = orders[..., 4]

    signed_size = direction.sign() * size.abs()
    ofi = signed_size.sum(dim=-1) / size.abs().sum(dim=-1).clamp_min(eps)

    # normalize_messages maps cancellation to event_type == 1.
    cancel_ratio = (torch.round(event_type).clamp(0, 2) == 1).to(dtype).mean(dim=-1)
    mean_dt = interarrival.mean(dim=-1)

    if cond_lob is None:
        volatility = torch.zeros_like(ofi)
    else:
        if cond_lob.dim() != 4:
            raise ValueError(
                "cond_lob must be (B, N, K_lob, F), "
                f"got {tuple(cond_lob.shape)}"
            )
        if cond_lob.shape[-1] < 3:
            raise ValueError(
                "rolling stats require best ask/bid columns in cond_lob, "
                f"got {cond_lob.shape[-1]} columns"
            )
        # Use one more LOB row than order rows when available so first
        # differences over the trailing window have matching support.
        lob_window = None if window is None else window + 1
        lob = _tail(cond_lob, dim=2, window=lob_window)
        best_ask = lob[..., 0]
        best_bid = lob[..., 2]
        mid = (best_ask + best_bid) * 0.5
        mid_diff = _safe_mid_changes(mid)
        volatility = mid_diff.std(dim=-1, unbiased=False)

    stats = torch.stack([volatility, ofi, cancel_ratio, mean_dt], dim=-1)
    return torch.nan_to_num(stats, nan=0.0, posinf=0.0, neginf=0.0).to(
        device=device, dtype=dtype
    )


def _tail(tensor: torch.Tensor, dim: int, window: int | None) -> torch.Tensor:
    if window is None or window <= 0 or tensor.shape[dim] <= window:
        return tensor
    start = tensor.shape[dim] - window
    return tensor.narrow(dim, start, window)


def _safe_mid_changes(mid: torch.Tensor) -> torch.Tensor:
    if mid.shape[-1] < 2:
        return torch.zeros_like(mid)
    if bool((mid > 0).all()):
        values = torch.log(mid.clamp_min(1e-6))
    else:
        # DeepMarket trains on z-scored LOB prices, so log returns are not
        # always defined. First differences preserve the volatility signal.
        values = mid
    return values.diff(dim=-1)
