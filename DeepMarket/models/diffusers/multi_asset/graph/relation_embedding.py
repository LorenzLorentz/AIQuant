"""Relation-type embeddings for directed asset edges."""

from __future__ import annotations

import torch
from torch import nn


class RelationEmbedding(nn.Module):
    def __init__(self, num_relation_types: int, embedding_dim: int):
        super().__init__()
        if num_relation_types <= 0:
            raise ValueError("num_relation_types must be positive")
        self.embedding = nn.Embedding(num_relation_types, embedding_dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def forward(self, relation_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(relation_ids.long())
