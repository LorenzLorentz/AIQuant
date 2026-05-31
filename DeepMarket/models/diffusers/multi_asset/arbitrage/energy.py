"""P3 annealed quasi-no-arbitrage energy guidance."""

from __future__ import annotations

import torch
from torch import nn

from models.diffusers.multi_asset.ablation_flags import GraphAblationFlags
from models.diffusers.multi_asset.arbitrage.consistency_check import (
    has_valid_basket_weights,
    validate_guidance_inputs,
)
from models.diffusers.multi_asset.arbitrage.guidance_schedule import AnnealedGuidanceSchedule
from models.diffusers.multi_asset.arbitrage.persistence_tracker import PersistenceTracker
from models.diffusers.multi_asset.spread.spread_context import compute_top_of_book_mid_spread


def reconstruct_x0_from_eps(
    x_t: torch.Tensor,
    eps: torch.Tensor,
    t: torch.Tensor,
    alphas_cumprod: torch.Tensor,
) -> torch.Tensor:
    """DDPM clean-sample reconstruction used as the P3 price proxy source."""
    t_index = t.to(device=alphas_cumprod.device, dtype=torch.long)
    t_index = t_index.clamp(min=0, max=alphas_cumprod.shape[0] - 1)
    alpha_bar = alphas_cumprod[t_index].to(device=x_t.device, dtype=x_t.dtype)
    alpha_bar = alpha_bar.view(-1, 1, 1, 1).clamp(min=1e-8, max=1.0)
    return (x_t - torch.sqrt(1.0 - alpha_bar) * eps) / torch.sqrt(alpha_bar)


def nav_gap_energy(
    asset_universe,
    x_hat_0: torch.Tensor,
    cond_lob: torch.Tensor,
    *,
    price_proxy_index: int = 5,
    delta: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Return per-batch Huber NAV-gap energy and raw gaps.

    ``x_hat_0[..., price_proxy_index]`` is treated as the generated relative
    price proxy, averaged over the generation horizon and added to the last
    conditioning mid price.
    """
    if not has_valid_basket_weights(asset_universe, x_hat_0.shape[1]):
        return None
    if price_proxy_index < 0 or price_proxy_index >= x_hat_0.shape[-1]:
        return None
    if delta <= 0:
        raise ValueError(f"delta must be positive, got {delta}")

    last_mid, _ = compute_top_of_book_mid_spread(cond_lob)
    price_proxy = x_hat_0[..., price_proxy_index].mean(dim=2)
    mid_hat = last_mid.to(x_hat_0.dtype) + price_proxy

    gaps = []
    for etf_idx, weights in asset_universe.etf_basket_weights.items():
        basket_mid = torch.zeros(
            mid_hat.shape[0],
            device=mid_hat.device,
            dtype=mid_hat.dtype,
        )
        for constituent_idx, weight in weights.items():
            basket_mid = basket_mid + float(weight) * mid_hat[:, constituent_idx]
        gaps.append(mid_hat[:, etf_idx] - basket_mid)

    if not gaps:
        return None

    gap_tensor = torch.stack(gaps, dim=1)
    abs_gap = gap_tensor.abs()
    quadratic = torch.minimum(abs_gap, torch.as_tensor(delta, device=gap_tensor.device, dtype=gap_tensor.dtype))
    linear = abs_gap - quadratic
    energy = 0.5 * quadratic.pow(2) + float(delta) * linear
    return energy.mean(dim=1), gap_tensor


class ArbitrageEnergyGuidance(nn.Module):
    """Post-fusion P3 guidance hook.

    The module has no trainable parameters. It computes a detached gradient
    correction in eps-space and applies it with a late-step annealed scale.
    """

    def __init__(
        self,
        asset_universe,
        alphas_cumprod: torch.Tensor,
        *,
        enabled: bool = False,
        flags: GraphAblationFlags | None = None,
        start_step: int = 10,
        max_scale: float = 0.01,
        delta: float = 1.0,
        max_grad_norm: float = 1.0,
        price_proxy_index: int = 5,
        persistence_weight: float = 0.0,
    ):
        super().__init__()
        if max_grad_norm <= 0:
            raise ValueError(f"max_grad_norm must be positive, got {max_grad_norm}")
        self.asset_universe = asset_universe
        self.enabled = bool(enabled)
        self.flags = flags or GraphAblationFlags()
        self.delta = float(delta)
        self.max_grad_norm = float(max_grad_norm)
        self.price_proxy_index = int(price_proxy_index)
        self.schedule = AnnealedGuidanceSchedule(start_step=start_step, max_scale=max_scale)
        self.persistence_tracker = PersistenceTracker(weight=persistence_weight)
        self.register_buffer("alphas_cumprod", alphas_cumprod.detach().clone().float())

    @property
    def is_disabled(self) -> bool:
        return (not self.enabled) or bool(self.flags.disable_arb_guidance)

    def reset_state(self) -> None:
        self.persistence_tracker.reset_state()

    def forward(
        self,
        eps_fused: torch.Tensor,
        *,
        x_t: torch.Tensor | None,
        t: torch.Tensor | None,
        cond_lob: torch.Tensor | None,
        **_ctx,
    ) -> torch.Tensor:
        if self.is_disabled:
            return eps_fused
        if not has_valid_basket_weights(self.asset_universe, eps_fused.shape[1]):
            return eps_fused
        if not validate_guidance_inputs(
            eps_fused, x_t, t, cond_lob, self.price_proxy_index
        ):
            return eps_fused

        scale = self.schedule(t).to(device=eps_fused.device, dtype=eps_fused.dtype)
        if not torch.any(scale > 0):
            return eps_fused

        with torch.enable_grad():
            eps_leaf = eps_fused.detach().requires_grad_(True)
            x_hat_0 = reconstruct_x0_from_eps(
                x_t.detach(),
                eps_leaf,
                t.detach(),
                self.alphas_cumprod,
            )
            energy_and_gaps = nav_gap_energy(
                self.asset_universe,
                x_hat_0,
                cond_lob.detach(),
                price_proxy_index=self.price_proxy_index,
                delta=self.delta,
            )
            if energy_and_gaps is None:
                return eps_fused
            energy, gaps = energy_and_gaps
            if not torch.isfinite(energy).all():
                return eps_fused
            persistence = self.persistence_tracker.multiplier(gaps)
            objective = (energy * persistence).sum()
            grad = torch.autograd.grad(
                objective,
                eps_leaf,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )[0]

        if grad is None or not torch.isfinite(grad).all():
            return eps_fused

        grad = self._clip_grad(grad)
        return eps_fused - scale.view(-1, 1, 1, 1) * grad.detach()

    def _clip_grad(self, grad: torch.Tensor) -> torch.Tensor:
        norm = grad.flatten(start_dim=1).norm(dim=1).clamp(min=1e-12)
        factor = (self.max_grad_norm / norm).clamp(max=1.0)
        return grad * factor.view(-1, 1, 1, 1)
