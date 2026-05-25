"""Directed relation embeddings for graph-coupled diffusion."""

from __future__ import annotations

import torch
from torch import nn


class RelationEmbedding(nn.Module):
    """Lookup table for directed edge relation types."""

    def __init__(self, num_relation_types: int, relation_dim: int = 8):
        super().__init__()
        if num_relation_types < 1:
            raise ValueError("num_relation_types must be >= 1")
        self.num_relation_types = num_relation_types
        self.relation_dim = relation_dim
        self.embedding = nn.Embedding(num_relation_types, relation_dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def forward(self, relation_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(relation_ids.long())


__all__ = ["RelationEmbedding"]
