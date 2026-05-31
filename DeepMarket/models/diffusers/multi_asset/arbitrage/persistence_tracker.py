"""Persistence counters for P3 arbitrage-guidance excursions."""

from __future__ import annotations

from typing import Dict

import torch


def _as_broadcast_delta(delta_dyn: torch.Tensor, spread: torch.Tensor) -> torch.Tensor:
    if delta_dyn.dim() == 0:
        return delta_dyn.view(1, 1).expand_as(spread)
    if delta_dyn.dim() == 1:
        if delta_dyn.numel() == 1:
            return delta_dyn.view(1, 1).expand_as(spread)
        if delta_dyn.numel() != spread.shape[1]:
            raise ValueError(
                "delta_dyn group axis must match spread groups: "
                f"{delta_dyn.numel()} vs {spread.shape[1]}"
            )
        return delta_dyn.view(1, -1).expand_as(spread)
    if delta_dyn.shape != spread.shape:
        raise ValueError(
            f"delta_dyn must be scalar, (G,), or spread-shaped; got {tuple(delta_dyn.shape)}"
        )
    return delta_dyn


def tau_to_tensor(tau: Dict[int, torch.Tensor], spread: torch.Tensor) -> torch.Tensor:
    """Materialize ``tau`` dict values as a ``(B, G)`` tensor."""
    if spread.dim() != 2:
        raise ValueError(f"spread must be (B, G), got {tuple(spread.shape)}")
    B, G = spread.shape
    out = torch.zeros(B, G, device=spread.device, dtype=spread.dtype)
    for group_idx in range(G):
        value = tau.get(group_idx)
        if value is None:
            continue
        value = value.to(device=spread.device)
        if value.dim() == 0:
            value = value.expand(B)
        if value.shape != (B,):
            raise ValueError(
                f"tau[{group_idx}] must be scalar or shape {(B,)}, got {tuple(value.shape)}"
            )
        out[:, group_idx] = value.to(dtype=spread.dtype)
    return out


def update_tau(
    tau: Dict[int, torch.Tensor],
    spread: torch.Tensor,
    delta_dyn: torch.Tensor,
    active=True,
) -> torch.Tensor:
    """Update consecutive out-of-band counters for each spread group.

    ``tau`` is stored as ``{group_idx: LongTensor[B]}`` so each sample in the
    batch can have a different excursion duration.
    """
    if spread.dim() != 2:
        raise ValueError(f"spread must be (B, G), got {tuple(spread.shape)}")
    B, G = spread.shape
    delta = _as_broadcast_delta(delta_dyn.to(device=spread.device), spread)

    if torch.is_tensor(active):
        active_mask = active.to(device=spread.device, dtype=torch.bool).view(B, 1)
        if not bool(active_mask.any().item()):
            tau.clear()
            return spread.new_zeros(B, G)
    else:
        if not bool(active):
            tau.clear()
            return spread.new_zeros(B, G)
        active_mask = torch.ones(B, 1, device=spread.device, dtype=torch.bool)

    outside = (spread.detach().abs() > delta.detach()) & active_mask
    counters = torch.zeros(B, G, device=spread.device, dtype=torch.long)
    for group_idx in range(G):
        prev = tau.get(group_idx)
        if prev is None:
            prev = torch.zeros(B, device=spread.device, dtype=torch.long)
        else:
            prev = prev.to(device=spread.device, dtype=torch.long)
            if prev.dim() == 0:
                prev = prev.expand(B)
            if prev.shape != (B,):
                prev = torch.zeros(B, device=spread.device, dtype=torch.long)
        current = torch.where(
            outside[:, group_idx],
            prev + 1,
            torch.zeros_like(prev),
        )
        tau[group_idx] = current.detach()
        counters[:, group_idx] = current
    return counters.to(dtype=spread.dtype)


__all__ = ["tau_to_tensor", "update_tau"]
