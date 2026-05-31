"""Spread/NAV-gap context for P2 spread-aware conditioning."""

from __future__ import annotations

import torch


def compute_top_of_book_mid_spread(cond_lob: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-asset mid price and spread from the last raw LOB snapshot.

    ``cond_lob`` follows DeepMarket's LOBSTER orderbook layout:
    ask price, ask size, bid price, bid size, repeated by level.
    """
    if cond_lob.dim() != 4:
        raise ValueError(f"cond_lob must be (B, N, K_lob, F), got {tuple(cond_lob.shape)}")
    if cond_lob.shape[-1] < 3:
        raise ValueError(f"cond_lob must include best ask/bid price columns, got {cond_lob.shape[-1]}")

    last_book = cond_lob[:, :, -1, :]
    best_ask = last_book[..., 0]
    best_bid = last_book[..., 2]
    mid = (best_ask + best_bid) * 0.5
    spread = best_ask - best_bid
    return mid, spread


def compute_signed_nav_gap(asset_universe, mid: torch.Tensor) -> torch.Tensor | None:
    """Compute signed ETF-basket NAV gap per asset.

    The ETF asset receives ``+gap``. Each constituent receives
    ``-weight * gap``. Returns ``None`` if no basket metadata is configured.
    """
    if not asset_universe.etf_basket_weights:
        return None
    if mid.dim() != 2:
        raise ValueError(f"mid must be (B, N), got {tuple(mid.shape)}")

    B, N = mid.shape
    gap_by_asset = torch.zeros(B, N, device=mid.device, dtype=mid.dtype)
    for etf_idx, weights in asset_universe.etf_basket_weights.items():
        if etf_idx >= N:
            raise ValueError(f"ETF index {etf_idx} outside asset axis N={N}")
        basket_mid = torch.zeros(B, device=mid.device, dtype=mid.dtype)
        for constituent_idx, weight in weights.items():
            if constituent_idx >= N:
                raise ValueError(
                    f"Constituent index {constituent_idx} outside asset axis N={N}"
                )
            basket_mid = basket_mid + float(weight) * mid[:, constituent_idx]
        gap = mid[:, etf_idx] - basket_mid
        gap_by_asset[:, etf_idx] = gap_by_asset[:, etf_idx] + gap
        for constituent_idx, weight in weights.items():
            gap_by_asset[:, constituent_idx] = (
                gap_by_asset[:, constituent_idx] - float(weight) * gap
            )

    return gap_by_asset


def compute_spread_context(
    asset_universe,
    cond_lob: torch.Tensor | None,
    context_dim: int = 4,
) -> torch.Tensor | None:
    """Return P2 per-asset context or ``None`` when P2 should no-op."""
    if cond_lob is None or not asset_universe.etf_basket_weights:
        return None
    if context_dim <= 0:
        raise ValueError(f"context_dim must be positive, got {context_dim}")

    mid, spread = compute_top_of_book_mid_spread(cond_lob)
    signed_gap = compute_signed_nav_gap(asset_universe, mid)
    if signed_gap is None:
        return None

    base = torch.stack([mid, spread, signed_gap, signed_gap.abs()], dim=-1)
    base = torch.nan_to_num(base, nan=0.0, posinf=0.0, neginf=0.0)
    if context_dim == base.shape[-1]:
        return base
    if context_dim < base.shape[-1]:
        return base[..., :context_dim]

    pad = torch.zeros(
        *base.shape[:-1],
        context_dim - base.shape[-1],
        device=base.device,
        dtype=base.dtype,
    )
    return torch.cat([base, pad], dim=-1)
