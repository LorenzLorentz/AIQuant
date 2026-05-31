"""P2 spread-aware score conditioning."""

from __future__ import annotations

import torch
from torch import nn

from models.diffusers.multi_asset.arbitrage.activation_schedule import spread_inject_active
from models.diffusers.multi_asset.arbitrage.spread_computer import (
    broadcast_spread_to_assets,
    compute_spread,
    decode_mid_price,
)


class SpreadConditioner(nn.Module):
    """Compute the spread signal injected into ``SharedScoreNet``.

    This keeps the local P2 semantics: during sampling the signal is decoded
    from the previous fused epsilon estimate carried by ``ReverseLoopState``.
    For optional training-time conditioning, it can use the ground-truth
    ``x_0`` directly.
    """

    def __init__(
        self,
        asset_universe,
        alphas_cumprod: torch.Tensor,
        num_diffusionsteps: int,
        num_assets: int,
        *,
        k_spread_steps: int | None = None,
        price_feature_index: int | None = None,
        price_is_delta: bool = True,
        flags=None,
    ):
        super().__init__()
        self.asset_universe = asset_universe
        self.alphas_cumprod = alphas_cumprod
        self.num_diffusionsteps = int(num_diffusionsteps)
        self.num_assets = int(num_assets)
        self.k_spread_steps = k_spread_steps
        self.price_feature_index = price_feature_index
        self.price_is_delta = bool(price_is_delta)
        self.flags = flags

    def _disabled(self) -> bool:
        if bool(getattr(self.flags, "disable_spread_cond", False)):
            return True
        value = getattr(self.flags, "disable_spread_conditioning", None)
        return bool(value) if value is not None else False

    def _predict_x0_from_eps(self, x_t, eps_fused, t):
        if eps_fused.shape != x_t.shape:
            raise ValueError(
                "eps_fused must have same shape as x_t for spread decoding: "
                f"{tuple(eps_fused.shape)} vs {tuple(x_t.shape)}"
            )
        alpha_bar = self.alphas_cumprod.to(device=x_t.device, dtype=x_t.dtype)[t]
        if alpha_bar.dim() == 0:
            alpha_bar = alpha_bar.expand(x_t.shape[0])
        view_shape = (x_t.shape[0],) + (1,) * (x_t.dim() - 1)
        alpha_bar = alpha_bar.view(view_shape)
        sqrt_alpha_bar = torch.sqrt(alpha_bar).clamp_min(1e-12)
        sqrt_one_minus = torch.sqrt((1.0 - alpha_bar).clamp_min(0.0))
        return (x_t - sqrt_one_minus * eps_fused) / sqrt_alpha_bar

    def forward(self, x_t, state=None, t=None, cond_lob=None):
        if (
            state is None
            or self._disabled()
            or cond_lob is None
            or not self.asset_universe.etf_basket_weights
        ):
            if state is not None:
                state.update_spread(None)
            return None

        if t is None:
            raise ValueError("t is required for spread conditioning")

        active = spread_inject_active(
            t,
            num_diffusionsteps=self.num_diffusionsteps,
            k_spread_steps=self.k_spread_steps,
        )
        if torch.is_tensor(active):
            active = active.to(device=x_t.device)
            if not bool(active.any().item()):
                state.update_spread(None)
                return None
        elif not active:
            state.update_spread(None)
            return None

        if state.last_eps_fused is None:
            num_groups = len(self.asset_universe.etf_basket_weights)
            spread = x_t.new_zeros(x_t.shape[0], num_groups)
        else:
            x0_hat = self._predict_x0_from_eps(x_t, state.last_eps_fused, t)
            mid = decode_mid_price(
                x0_hat,
                cond_lob,
                price_feature_index=self.price_feature_index,
                price_is_delta=self.price_is_delta,
            )
            spread = compute_spread(mid, self.asset_universe)

        if spread.shape[1] == 0:
            state.update_spread(None)
            return None
        if torch.is_tensor(active):
            spread = spread * active.to(dtype=spread.dtype).view(-1, 1)

        state.update_spread(spread)
        return broadcast_spread_to_assets(spread, self.asset_universe, self.num_assets)

    def training_condition(self, x_0, cond_lob, dropout_prob: float = 0.0):
        if (
            self._disabled()
            or cond_lob is None
            or not self.asset_universe.etf_basket_weights
        ):
            return None
        mid = decode_mid_price(
            x_0,
            cond_lob,
            price_feature_index=self.price_feature_index,
            price_is_delta=self.price_is_delta,
        )
        spread = compute_spread(mid, self.asset_universe)
        if spread.shape[1] == 0:
            return None
        spread_cond = broadcast_spread_to_assets(spread, self.asset_universe, self.num_assets)
        if dropout_prob > 0.0:
            keep = torch.rand(spread_cond.shape[0], 1, device=spread_cond.device) >= dropout_prob
            spread_cond = spread_cond * keep.to(dtype=spread_cond.dtype)
        return spread_cond


__all__ = ["SpreadConditioner"]
