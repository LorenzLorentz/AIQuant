"""Attention aggregation over incoming graph messages."""

from __future__ import annotations

import math

import torch
from torch import nn


class AttentionAggregator(nn.Module):
    def __init__(self, feature_dim: int, attention_dim: int | None = None):
        super().__init__()
        attention_dim = attention_dim or feature_dim
        self.query = nn.Linear(feature_dim, attention_dim, bias=False)
        self.key = nn.Linear(feature_dim, attention_dim, bias=False)
        self.scale = math.sqrt(attention_dim)

    def forward(
        self,
        messages: torch.Tensor,
        src_states: torch.Tensor,
        dst_state: torch.Tensor,
    ) -> torch.Tensor:
        """Aggregate incoming messages for one destination asset.

        ``messages`` and ``src_states`` have shape ``(B, E, K, F)`` where
        ``E`` is the number of incoming edges. ``dst_state`` is ``(B, K, F)``.
        """
        if messages.dim() != 4:
            raise ValueError(f"messages must be (B, E, K, F), got {tuple(messages.shape)}")
        if messages.shape[1] == 0:
            return torch.zeros_like(dst_state)
        if messages.shape[1] == 1:
            return messages[:, 0]

        dst_summary = dst_state.mean(dim=1)
        src_summary = src_states.mean(dim=2)
        query = self.query(dst_summary).unsqueeze(1)
        key = self.key(src_summary)
        scores = (query * key).sum(dim=-1) / self.scale
        alpha = torch.softmax(scores, dim=1).view(messages.shape[0], messages.shape[1], 1, 1)
        return (alpha * messages).sum(dim=1)
