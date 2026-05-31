"""Graph-coupled noise fusion for multi-asset diffusion."""

from __future__ import annotations

import torch
from torch import nn

from models.diffusers.multi_asset.ablation_flags import GraphAblationFlags
from models.diffusers.multi_asset.graph.aggregator import AttentionAggregator
from models.diffusers.multi_asset.graph.edge_weight_net import EdgeWeightNet
from models.diffusers.multi_asset.graph.message_passing import MessageFunction
from models.diffusers.multi_asset.graph.noise_fusion import NoiseFusion
from models.diffusers.multi_asset.graph.relation_embedding import RelationEmbedding
from models.diffusers.multi_asset.graph.rolling_stats import compute_rolling_stats


class GraphCoupler(nn.Module):
    """P1 graph stack from rolling stats to residual noise fusion."""

    def __init__(
        self,
        asset_universe,
        feature_dim: int,
        relation_dim: int = 8,
        stats_dim: int = 4,
        hidden_dim: int | None = None,
        stats_window: int | None = None,
        flags: GraphAblationFlags | None = None,
    ):
        super().__init__()
        self.asset_universe = asset_universe
        self.num_assets = asset_universe.num_assets
        self.feature_dim = feature_dim
        self.stats_window = stats_window
        self.flags = flags or GraphAblationFlags()

        directed_edges = asset_universe.directed_edges()
        if self.num_assets > 1 and not directed_edges:
            raise ValueError("GraphCoupler requires directed relation_types for N > 1")

        edge_index = torch.tensor(directed_edges, dtype=torch.long)
        relation_ids = torch.tensor(
            [asset_universe.relation_id(j, i) for j, i in directed_edges],
            dtype=torch.long,
        )
        self.register_buffer("edge_index", edge_index)
        self.register_buffer("relation_ids", relation_ids)

        self.relation_embedding = RelationEmbedding(
            asset_universe.num_relation_types, relation_dim
        )
        hidden_dim = hidden_dim or feature_dim * 2
        self.edge_weight_net = EdgeWeightNet(stats_dim, relation_dim, hidden_dim)
        self.message_fn = MessageFunction(feature_dim, hidden_dim)
        self.aggregator = AttentionAggregator(feature_dim, hidden_dim)
        self.noise_fusion = NoiseFusion(feature_dim, hidden_dim)

    @property
    def gamma(self) -> torch.Tensor:
        return self.noise_fusion.gamma

    def forward(
        self,
        eps_local: torch.Tensor,
        *,
        x_t: torch.Tensor,
        cond_orders: torch.Tensor,
        cond_lob: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.flags.disable_graph or self.edge_index.numel() == 0:
            return eps_local

        if eps_local.shape != x_t.shape:
            raise ValueError(
                "eps_local and x_t must have identical shapes after deaugmentation, "
                f"got {tuple(eps_local.shape)} and {tuple(x_t.shape)}"
            )

        B, N, K, F = eps_local.shape
        if N != self.num_assets or F != self.feature_dim:
            raise ValueError(
                f"expected (B, {self.num_assets}, K, {self.feature_dim}), "
                f"got {tuple(eps_local.shape)}"
            )

        stats = compute_rolling_stats(cond_orders, cond_lob, window=self.stats_window)
        relation_vecs = self.relation_embedding(self.relation_ids.to(eps_local.device))

        incoming_messages: list[list[torch.Tensor]] = [[] for _ in range(N)]
        incoming_states: list[list[torch.Tensor]] = [[] for _ in range(N)]

        for edge_pos, (j, i) in enumerate(self.edge_index.tolist()):
            rel = relation_vecs[edge_pos]
            weight = self.edge_weight_net(stats[:, j], stats[:, i], rel)
            if self.flags.freeze_edge_weights:
                weight = weight.detach()
            message = self.message_fn(x_t[:, j], eps_local[:, j], weight)
            incoming_messages[i].append(message)
            incoming_states[i].append(x_t[:, j])

        aggregated = torch.zeros_like(eps_local)
        for i in range(N):
            if not incoming_messages[i]:
                continue
            messages_i = torch.stack(incoming_messages[i], dim=1)
            states_i = torch.stack(incoming_states[i], dim=1)
            aggregated[:, i] = self.aggregator(messages_i, states_i, x_t[:, i])

        return self.noise_fusion(eps_local, aggregated)
