"""Validation helpers for P3 arbitrage energy guidance."""

from __future__ import annotations

import torch


def has_valid_basket_weights(asset_universe, num_assets: int) -> bool:
    """Return True when ETF-basket metadata can be used safely."""
    if not getattr(asset_universe, "etf_basket_weights", None):
        return False
    for etf_idx, weights in asset_universe.etf_basket_weights.items():
        if etf_idx < 0 or etf_idx >= num_assets:
            return False
        if not weights:
            return False
        for constituent_idx in weights:
            if constituent_idx < 0 or constituent_idx >= num_assets:
                return False
    return True


def validate_guidance_inputs(
    eps_fused: torch.Tensor,
    x_t: torch.Tensor | None,
    t: torch.Tensor | None,
    cond_lob: torch.Tensor | None,
    price_proxy_index: int,
) -> bool:
    """Validate hook inputs; False means the guidance hook should no-op."""
    if x_t is None or t is None or cond_lob is None:
        return False
    if eps_fused.dim() != 4 or x_t.shape != eps_fused.shape:
        return False
    if cond_lob.dim() != 4 or cond_lob.shape[:2] != eps_fused.shape[:2]:
        return False
    if t.dim() != 1 or t.shape[0] != eps_fused.shape[0]:
        return False
    if price_proxy_index < 0 or price_proxy_index >= eps_fused.shape[-1]:
        return False
    return True
