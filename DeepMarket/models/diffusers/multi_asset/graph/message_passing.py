"""Token-wise graph messages for P1 noise coupling."""

from __future__ import annotations

import torch
from torch import nn


class MessageFunction(nn.Module):
    """Compute m_ji from source latent state, source local noise, and w_ji."""

    def __init__(self, feature_dim: int, hidden_dim: int | None = None):
        super().__init__()
        hidden_dim = hidden_dim or max(32, 2 * feature_dim)
        self.net = nn.Sequential(
            nn.Linear(2 * feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(self, x_src, eps_src, edge_weight):
        """Return messages with the same token shape as ``eps_src``.

        x_src / eps_src:
            ``(B, E, K, F)`` tensors.
        edge_weight:
            ``(B, E)`` or broadcast-compatible edge weights.
        """
        h = torch.cat([x_src, eps_src], dim=-1)
        message = self.net(h)
        while edge_weight.dim() < message.dim():
            edge_weight = edge_weight.unsqueeze(-1)
        return edge_weight * message


__all__ = ["MessageFunction"]
