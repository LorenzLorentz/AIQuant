"""Rolling conditioning statistics for graph edge weights.

The P1 graph uses a cheap four-dimensional summary per asset:

1. mid-price log-return volatility from the conditioning LOB,
2. signed order-flow imbalance,
3. cancellation ratio,
4. mean inter-arrival time.

The function accepts either a single asset ``(B, K, F)`` or a multi-asset
batch ``(B, N, K, F)``. The multi-asset path returns ``(B, N, 4)``.
"""

from __future__ import annotations

import torch

import constants as cst


def compute_rolling_stats(cond_orders, cond_lob=None, eps: float = 1e-6):
    """Compute P1 rolling stats from raw conditioning tensors.

    Parameters
    ----------
    cond_orders:
        Raw order conditioning. Preferred shape is ``(B, N, K, 6)``. If a
        type-embedded tensor with feature size >= 8 is passed, side and size
        are read from their shifted positions and cancellation ratio falls
        back to zero because the event type was replaced by an embedding.
    cond_lob:
        Raw LOB conditioning of shape ``(B, N, K+1, 40)`` or ``None``.
    eps:
        Denominator clamp used for ratios and logs.
    """
    squeeze_asset = False
    if cond_orders.dim() == 3:
        cond_orders = cond_orders.unsqueeze(1)
        if cond_lob is not None and cond_lob.dim() == 3:
            cond_lob = cond_lob.unsqueeze(1)
        squeeze_asset = True
    if cond_orders.dim() != 4:
        raise ValueError(
            "cond_orders must be (B, K, F) or (B, N, K, F), "
            f"got {tuple(cond_orders.shape)}"
        )

    orders = cond_orders.float()
    B, N, _, F = orders.shape

    dt = orders[..., 0].abs()
    mean_dt = dt.mean(dim=-1)

    if F >= 8:
        # Type embedding changes [dt, type, side, price, size, extra] into
        # [dt, type_emb0, type_emb1, type_emb2, side, price, size, extra].
        side = orders[..., 4]
        size = orders[..., 6].abs()
        cancel_ratio = orders.new_zeros(B, N)
    else:
        event_type = orders[..., 1]
        side = orders[..., 2]
        size = orders[..., 4].abs()
        cancel = torch.isclose(
            event_type,
            torch.full((), float(cst.OrderEvent.CANCELLATION.value), device=orders.device),
            atol=1e-4,
        )
        cancel_ratio = cancel.float().mean(dim=-1)

    signed_side = torch.where(side > 0, torch.ones_like(side), -torch.ones_like(side))
    total_size = size.sum(dim=-1).clamp_min(eps)
    order_flow_imbalance = (signed_side * size).sum(dim=-1) / total_size

    if cond_lob is None:
        volatility = orders.new_zeros(B, N)
    else:
        lob = cond_lob.float()
        if lob.dim() == 3:
            lob = lob.unsqueeze(1)
        if lob.dim() != 4:
            raise ValueError(
                "cond_lob must be None, (B, K, F), or (B, N, K, F), "
                f"got {tuple(lob.shape)}"
            )
        if lob.shape[0] != B or lob.shape[1] != N:
            raise ValueError(
                "cond_lob batch/asset axes must match cond_orders: "
                f"{tuple(lob.shape[:2])} vs {(B, N)}"
            )
        volatility = _mid_log_return_std(lob, eps=eps)

    stats = torch.stack(
        [volatility, order_flow_imbalance, cancel_ratio, mean_dt],
        dim=-1,
    )
    stats = torch.nan_to_num(stats, nan=0.0, posinf=0.0, neginf=0.0)
    if squeeze_asset:
        return stats[:, 0]
    return stats


def _mid_log_return_std(cond_lob, eps: float):
    """Return per-asset std of conditioning-window mid-price log returns."""
    if cond_lob.shape[-1] < cst.LEN_LEVEL:
        return cond_lob.new_zeros(cond_lob.shape[0], cond_lob.shape[1])

    # DeepMarket stores each LOB level as 4 columns. P1 assumes the first
    # level carries bid price at col 0 and ask price at col 2.
    bid = cond_lob[..., 0]
    ask = cond_lob[..., 2]
    mid = (bid + ask) * 0.5
    mid = mid.abs().clamp_min(eps)
    if mid.shape[-1] < 2:
        return cond_lob.new_zeros(cond_lob.shape[0], cond_lob.shape[1])

    log_mid = torch.log(mid)
    returns = log_mid[..., 1:] - log_mid[..., :-1]
    returns = torch.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
    return returns.std(dim=-1, unbiased=False)


__all__ = ["compute_rolling_stats"]
