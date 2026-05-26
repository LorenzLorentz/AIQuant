"""Spread decoding utilities for P2 spread-aware conditioning.

The repo's saved LOB arrays are normalized, and the normalization statistics
are not carried by ``MultiAssetLOBDataset``. These helpers therefore operate
in the same coordinate system as the tensors passed to the score net by
default. If callers provide de-normalized tensors upstream, the same formulas
work in price units.
"""

from __future__ import annotations

from typing import List, Tuple

import torch

import constants as cst


def infer_price_feature_index(feature_dim: int) -> int:
    """Infer the order-price column before or after type embedding.

    Raw DeepMarket order tokens are ``[dt, type, size, price, direction,
    depth]``. After type embedding they become ``[dt, type_emb0, type_emb1,
    type_emb2, size, price, direction, depth]``.
    """
    return 5 if feature_dim >= cst.LEN_ORDER + 2 else 3


def decode_mid_price(
    x0_hat: torch.Tensor,
    cond_lob: torch.Tensor,
    price_feature_index: int | None = None,
    price_is_delta: bool = True,
) -> torch.Tensor:
    """Estimate per-asset mid-price from predicted clean orders and LOB cond.

    Parameters
    ----------
    x0_hat:
        Predicted clean generated orders, shape ``(B, N, K_gen, F)``.
    cond_lob:
        Conditioning LOB snapshots, shape ``(B, N, K_cond+1, 40)``.
    price_feature_index:
        Optional explicit index for the generated order price feature.
    price_is_delta:
        If True, add the generated price feature to the last conditioning
        mid-price. This matches the P2 design note's relative-price view. If
        False, the generated price feature is treated as an absolute mid proxy.

    Returns
    -------
    torch.Tensor
        Estimated mid-price, shape ``(B, N)``.
    """
    if x0_hat.dim() != 4:
        raise ValueError(f"x0_hat must be (B, N, K, F), got {tuple(x0_hat.shape)}")
    if cond_lob is None:
        raise ValueError("cond_lob is required to decode mid-price")
    if cond_lob.dim() != 4:
        raise ValueError(f"cond_lob must be (B, N, K, F), got {tuple(cond_lob.shape)}")
    if cond_lob.shape[:2] != x0_hat.shape[:2]:
        raise ValueError(
            "cond_lob batch/asset axes must match x0_hat: "
            f"{tuple(cond_lob.shape[:2])} vs {tuple(x0_hat.shape[:2])}"
        )
    if cond_lob.shape[-1] < cst.LEN_LEVEL:
        raise ValueError(
            "cond_lob must contain at least one LOB level "
            f"({cst.LEN_LEVEL} columns), got {cond_lob.shape[-1]}"
        )

    idx = (
        infer_price_feature_index(x0_hat.shape[-1])
        if price_feature_index is None
        else price_feature_index
    )
    if idx < 0 or idx >= x0_hat.shape[-1]:
        raise ValueError(
            f"price_feature_index={idx} outside generated feature dim {x0_hat.shape[-1]}"
        )

    last_lob = cond_lob[:, :, -1, :]
    best_ask = last_lob[..., 0]
    best_bid = last_lob[..., 2]
    last_mid = 0.5 * (best_ask + best_bid)
    predicted_price = x0_hat[..., idx].mean(dim=-1)
    mid = last_mid + predicted_price if price_is_delta else predicted_price
    return torch.nan_to_num(mid, nan=0.0, posinf=0.0, neginf=0.0)


def spread_groups(asset_universe) -> List[Tuple[int, List[Tuple[int, float]]]]:
    """Return deterministic ``[(etf_idx, [(asset_idx, weight), ...]), ...]``."""
    groups = []
    for etf_idx, basket in sorted(asset_universe.etf_basket_weights.items()):
        groups.append((int(etf_idx), [(int(i), float(w)) for i, w in sorted(basket.items())]))
    return groups


def compute_spread(mid: torch.Tensor, asset_universe) -> torch.Tensor:
    """Compute ETF-minus-basket spread for each configured ETF group.

    Returns an empty ``(B, 0)`` tensor when no ETF basket weights are
    configured; callers can then skip spread conditioning cleanly.
    """
    if mid.dim() != 2:
        raise ValueError(f"mid must be (B, N), got {tuple(mid.shape)}")
    B, N = mid.shape
    spreads = []
    for etf_idx, basket in spread_groups(asset_universe):
        if etf_idx < 0 or etf_idx >= N:
            raise ValueError(f"ETF asset index {etf_idx} outside N={N}")
        basket_mid = mid.new_zeros(B)
        for asset_idx, weight in basket:
            if asset_idx < 0 or asset_idx >= N:
                raise ValueError(f"basket asset index {asset_idx} outside N={N}")
            basket_mid = basket_mid + float(weight) * mid[:, asset_idx]
        spreads.append(mid[:, etf_idx] - basket_mid)
    if not spreads:
        return mid.new_zeros(B, 0)
    return torch.stack(spreads, dim=1)


def broadcast_spread_to_assets(spread: torch.Tensor, asset_universe, num_assets: int) -> torch.Tensor:
    """Broadcast group spreads to every asset participating in each group."""
    if spread.dim() != 2:
        raise ValueError(f"spread must be (B, G), got {tuple(spread.shape)}")
    out = spread.new_zeros(spread.shape[0], num_assets)
    groups = spread_groups(asset_universe)
    if spread.shape[1] != len(groups):
        raise ValueError(
            f"spread group axis {spread.shape[1]} does not match configured groups {len(groups)}"
        )
    for group_idx, (etf_idx, basket) in enumerate(groups):
        out[:, etf_idx] = spread[:, group_idx]
        for asset_idx, _weight in basket:
            out[:, asset_idx] = spread[:, group_idx]
    return out


__all__ = [
    "broadcast_spread_to_assets",
    "compute_spread",
    "decode_mid_price",
    "infer_price_feature_index",
    "spread_groups",
]
