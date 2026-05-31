"""Graph-coupled noise fusion for multi-asset diffusion."""

from __future__ import annotations

import torch
from torch import nn

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
        hidden_dim: int | None = 32,
        flags=None,
    ):
        super().__init__()
        self.asset_universe = asset_universe
        self.num_assets = asset_universe.num_assets
        self.feature_dim = int(feature_dim)
        self.flags = flags

        edge_src, edge_dst, edge_relation_ids = self._build_graph_edges()
        self.register_buffer("edge_src", edge_src)
        self.register_buffer("edge_dst", edge_dst)
        self.register_buffer("edge_relation_ids", edge_relation_ids)

        hidden_dim = hidden_dim or 32
        self.relation_embedding = RelationEmbedding(
            num_relation_types=max(1, self.asset_universe.num_relation_types),
            relation_dim=relation_dim,
        )
        self.edge_weight_net = EdgeWeightNet(
            stats_dim=4,
            relation_dim=relation_dim,
            hidden_dim=hidden_dim,
        )
        self.message_fn = MessageFunction(
            feature_dim=self.feature_dim,
            hidden_dim=hidden_dim,
        )
        self.aggregator = AttentionAggregator(
            feature_dim=self.feature_dim,
            hidden_dim=hidden_dim,
        )
        self.noise_fusion = NoiseFusion(
            feature_dim=self.feature_dim,
            hidden_dim=hidden_dim,
            init_gamma=0.0,
        )

    @property
    def gamma(self):
        return self.noise_fusion.gamma

    def _build_graph_edges(self):
        edges = list(self.asset_universe.directed_edges())
        if not edges and self.num_assets > 1:
            edges = [
                (j, i)
                for j in range(self.num_assets)
                for i in range(self.num_assets)
                if i != j
            ]

        src = []
        dst = []
        rel_ids = []
        for j, i in edges:
            src.append(j)
            dst.append(i)
            if (j, i) in self.asset_universe.relation_types:
                rel_ids.append(self.asset_universe.relation_id(j, i))
            else:
                rel_ids.append(0)

        return (
            torch.tensor(src, dtype=torch.long),
            torch.tensor(dst, dtype=torch.long),
            torch.tensor(rel_ids, dtype=torch.long),
        )

    def _graph_disabled(self) -> bool:
        return bool(getattr(self.flags, "disable_graph", False))

    def _freeze_edge_weights(self) -> bool:
        return bool(getattr(self.flags, "freeze_edge_weights", False))

    def forward(
        self,
        eps_local: torch.Tensor,
        *,
        x_t: torch.Tensor,
        cond_orders: torch.Tensor | None,
        cond_lob: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self._graph_disabled() or self.edge_src.numel() == 0:
            return eps_local
        if cond_orders is None:
            return eps_local
        if x_t.shape[-1] != eps_local.shape[-1]:
            raise ValueError(
                "graph fusion expects x_t and eps_local to share the feature "
                f"dimension, got {x_t.shape[-1]} and {eps_local.shape[-1]}"
            )

        stats = compute_rolling_stats(cond_orders, cond_lob).to(
            device=eps_local.device,
            dtype=eps_local.dtype,
        )
        src = self.edge_src.to(eps_local.device)
        dst = self.edge_dst.to(eps_local.device)
        rel_ids = self.edge_relation_ids.to(eps_local.device)

        stats_src = stats[:, src, :]
        stats_dst = stats[:, dst, :]
        relation_emb = self.relation_embedding(rel_ids).to(dtype=eps_local.dtype)
        relation_emb = relation_emb.unsqueeze(0).expand(eps_local.shape[0], -1, -1)

        edge_weights = self.edge_weight_net(stats_src, stats_dst, relation_emb)
        if self._freeze_edge_weights():
            edge_weights = edge_weights.detach()

        messages = self.message_fn(x_t[:, src, :, :], eps_local[:, src, :, :], edge_weights)
        aggregated = self.aggregator(messages, x_t, src, dst, num_nodes=self.num_assets)
        return self.noise_fusion(eps_local, aggregated)


__all__ = ["GraphCoupler"]
