"""P2 spread-aware residual pre-fusion conditioning."""

from __future__ import annotations

import torch
from torch import nn

from models.diffusers.multi_asset.ablation_flags import GraphAblationFlags
from models.diffusers.multi_asset.spread.spread_context import compute_spread_context


class SpreadConditioner(nn.Module):
    def __init__(
        self,
        asset_universe,
        feature_dim: int,
        context_dim: int = 4,
        hidden_dim: int | None = None,
        enabled: bool = False,
        flags: GraphAblationFlags | None = None,
    ):
        super().__init__()
        self.asset_universe = asset_universe
        self.feature_dim = feature_dim
        self.context_dim = context_dim
        self.enabled = enabled
        self.flags = flags or GraphAblationFlags()

        hidden_dim = hidden_dim or feature_dim * 2
        self.gamma_spread = nn.Parameter(torch.tensor(0.0))
        self.net = nn.Sequential(
            nn.Linear(feature_dim + context_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, feature_dim),
        )

    @property
    def is_disabled(self) -> bool:
        flag = self.flags.disable_spread_conditioning
        if flag is None:
            return not self.enabled
        return bool(flag) or not self.enabled

    def forward(
        self,
        eps_local: torch.Tensor,
        *,
        cond_lob: torch.Tensor | None,
        **_ctx,
    ) -> torch.Tensor:
        if self.is_disabled:
            return eps_local

        spread_context = compute_spread_context(
            self.asset_universe,
            cond_lob,
            context_dim=self.context_dim,
        )
        if spread_context is None:
            return eps_local
        if eps_local.dim() != 4:
            raise ValueError(
                f"eps_local must be (B, N, K, F), got {tuple(eps_local.shape)}"
            )
        if eps_local.shape[:2] != spread_context.shape[:2]:
            raise ValueError(
                "spread context and eps_local batch/asset axes differ: "
                f"{tuple(spread_context.shape[:2])} vs {tuple(eps_local.shape[:2])}"
            )

        B, N, K, _ = eps_local.shape
        context = spread_context.unsqueeze(2).expand(B, N, K, self.context_dim)
        residual = self.net(torch.cat([eps_local, context.to(eps_local.dtype)], dim=-1))
        return eps_local + self.gamma_spread * residual
