"""Attention aggregation for incoming graph messages."""

from __future__ import annotations

import torch
from torch import nn


class AttentionAggregator(nn.Module):
    """Attention-weighted sum over incoming directed edge messages."""

    def __init__(self, feature_dim: int, hidden_dim: int | None = None):
        super().__init__()
        hidden_dim = hidden_dim or max(32, 2 * feature_dim)
        self.score = nn.Sequential(
            nn.Linear(2 * feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, messages, node_states, src_index, dst_index, num_nodes=None):
        """Aggregate ``messages`` into one tensor per destination node.

        Parameters
        ----------
        messages:
            Edge messages with shape ``(B, E, K, F)``.
        node_states:
            Current node latents with shape ``(B, N, K, F)``.
        src_index / dst_index:
            Long tensors of shape ``(E,)`` describing each directed edge.
        num_nodes:
            Optional node count. Defaults to ``node_states.shape[1]``.
        """
        B, _, K, F = node_states.shape
        num_nodes = num_nodes or node_states.shape[1]
        if messages.numel() == 0:
            return node_states.new_zeros(B, num_nodes, K, F)

        node_summary = node_states.mean(dim=2)
        src_summary = node_summary[:, src_index, :]
        dst_summary = node_summary[:, dst_index, :]
        score_in = torch.cat([dst_summary, src_summary], dim=-1)
        scores = self.score(score_in).squeeze(-1)

        aggregated = node_states.new_zeros(B, num_nodes, K, F)
        for dst in range(num_nodes):
            mask = dst_index == dst
            if not torch.any(mask):
                continue
            weights = torch.softmax(scores[:, mask], dim=1).view(B, -1, 1, 1)
            aggregated[:, dst] = (messages[:, mask] * weights).sum(dim=1)
        return aggregated


__all__ = ["AttentionAggregator"]
