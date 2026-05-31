"""Energy terms for P3 annealed arbitrage guidance."""

from __future__ import annotations

import torch
from torch import nn

from models.diffusers.multi_asset.arbitrage.activation_schedule import spread_inject_active
from models.diffusers.multi_asset.arbitrage.guidance_schedule import guidance_lambda
from models.diffusers.multi_asset.arbitrage.persistence_tracker import update_tau
from models.diffusers.multi_asset.arbitrage.spread_computer import (
    compute_spread,
    decode_mid_price,
    spread_groups,
)
from models.diffusers.multi_asset.graph import compute_rolling_stats


def rho(abs_spread: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """Hinge-quadratic penalty ``relu(|spread| - delta) ** 2``."""
    return torch.relu(abs_spread - delta) ** 2


def phi(tau: torch.Tensor) -> torch.Tensor:
    """Default linear persistence weight."""
    return tau.to(dtype=torch.float32)


class StressHead(nn.Module):
    """Small non-negative linear head over P1 rolling stats.

    The zero initialization makes P3 default to ``delta_dyn == delta_base`` when
    loading a P2 checkpoint or running without training this head.
    """

    def __init__(self, stats_dim: int = 4):
        super().__init__()
        self.linear = nn.Linear(stats_dim, 1)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, rolling_stats: torch.Tensor) -> torch.Tensor:
        if rolling_stats.dim() != 3:
            raise ValueError(
                "rolling_stats must be (B, G, stats_dim), "
                f"got {tuple(rolling_stats.shape)}"
            )
        return torch.relu(self.linear(rolling_stats).squeeze(-1))


def group_rolling_stats(rolling_stats: torch.Tensor, asset_universe) -> torch.Tensor:
    """Aggregate per-asset rolling stats into one vector per ETF basket group."""
    if rolling_stats.dim() != 3:
        raise ValueError(
            "rolling_stats must be (B, N, stats_dim), "
            f"got {tuple(rolling_stats.shape)}"
        )
    B, N, D = rolling_stats.shape
    groups = spread_groups(asset_universe)
    if not groups:
        return rolling_stats.new_zeros(B, 0, D)

    out = []
    for etf_idx, basket in groups:
        if etf_idx < 0 or etf_idx >= N:
            raise ValueError(f"ETF asset index {etf_idx} outside N={N}")
        etf_stats = rolling_stats[:, etf_idx, :]
        basket_stats = rolling_stats.new_zeros(B, D)
        weight_total = 0.0
        for asset_idx, weight in basket:
            if asset_idx < 0 or asset_idx >= N:
                raise ValueError(f"basket asset index {asset_idx} outside N={N}")
            abs_weight = abs(float(weight))
            basket_stats = basket_stats + abs_weight * rolling_stats[:, asset_idx, :]
            weight_total += abs_weight
        if weight_total > 0.0:
            basket_stats = basket_stats / weight_total
            out.append(0.5 * (etf_stats + basket_stats))
        else:
            out.append(etf_stats)
    return torch.stack(out, dim=1)


def dynamic_delta(
    delta_base: torch.Tensor,
    group_stats: torch.Tensor,
    stress_head: nn.Module | None = None,
    kappa: float = 1.0,
) -> torch.Tensor:
    """State-dependent tolerance ``delta_base * (1 + kappa * stress)``."""
    if group_stats.dim() != 3:
        raise ValueError(f"group_stats must be (B, G, D), got {tuple(group_stats.shape)}")
    B, G, _ = group_stats.shape
    base = delta_base.to(device=group_stats.device, dtype=group_stats.dtype).flatten()
    if base.numel() == 1:
        base = base.expand(G)
    elif base.numel() != G:
        raise ValueError(f"delta_base must be scalar or length {G}, got {base.numel()}")
    base = base.view(1, G).expand(B, G)

    if stress_head is None:
        stress = group_stats.new_zeros(B, G)
    else:
        stress = stress_head(group_stats.to(dtype=group_stats.dtype)).to(dtype=group_stats.dtype)
    return base * (1.0 + float(kappa) * stress)


def energy(spread: torch.Tensor, delta_dyn: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    """Return scalar ``sum_g rho(|spread_g|, delta_dyn_g) * phi(tau_g)``."""
    if spread.shape != delta_dyn.shape:
        raise ValueError(
            "spread and delta_dyn must have the same shape, "
            f"got {tuple(spread.shape)} and {tuple(delta_dyn.shape)}"
        )
    if tau.shape != spread.shape:
        raise ValueError(f"tau must match spread shape {tuple(spread.shape)}, got {tuple(tau.shape)}")
    return (rho(spread.abs(), delta_dyn) * phi(tau).to(device=spread.device, dtype=spread.dtype)).sum()


def calibrate_delta_base(spread: torch.Tensor) -> torch.Tensor:
    """Median ``|spread|`` per group, suitable for storing as ``delta_base``."""
    if spread.dim() != 2:
        raise ValueError(f"spread must be (B, G), got {tuple(spread.shape)}")
    return spread.detach().abs().median(dim=0).values


class ArbitrageEnergyGuidance(nn.Module):
    """P3 annealed energy-gradient guidance hook.

    The module keeps the local implementation's semantics. It computes the
    ETF-basket spread from ``x_t -> x0_hat -> mid``, applies dynamic tolerance
    and persistence counters, then returns the guided epsilon estimate.
    """

    def __init__(
        self,
        asset_universe,
        alphas_cumprod: torch.Tensor,
        num_diffusionsteps: int,
        *,
        k_spread_steps: int | None = None,
        price_feature_index: int | None = None,
        price_is_delta: bool = True,
        kappa: float = 1.0,
        lambda_max: float = 1.0,
        lambda_power: float = 2.0,
        flags=None,
    ):
        super().__init__()
        self.asset_universe = asset_universe
        self.alphas_cumprod = alphas_cumprod
        self.num_diffusionsteps = int(num_diffusionsteps)
        self.k_spread_steps = k_spread_steps
        self.price_feature_index = price_feature_index
        self.price_is_delta = bool(price_is_delta)
        self.kappa = float(kappa)
        self.lambda_max = float(lambda_max)
        self.lambda_power = float(lambda_power)
        self.flags = flags

    def _disabled(self) -> bool:
        return bool(getattr(self.flags, "disable_arb_guidance", False))

    def _predict_x0_from_eps(self, x_t, eps_fused, t):
        if eps_fused.shape != x_t.shape:
            raise ValueError(
                "eps_fused must have same shape as x_t for energy guidance: "
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

    def forward(
        self,
        eps_fused,
        *,
        x_t,
        t=None,
        state=None,
        cond_orders=None,
        cond_lob=None,
        raw_cond_orders=None,
        raw_cond_lob=None,
        delta_base=None,
        stress_head=None,
    ):
        if (
            state is None
            or self._disabled()
            or t is None
            or not self.asset_universe.etf_basket_weights
        ):
            return eps_fused

        active = spread_inject_active(
            t,
            num_diffusionsteps=self.num_diffusionsteps,
            k_spread_steps=self.k_spread_steps,
        )
        if torch.is_tensor(active):
            active = active.to(device=x_t.device)
            if not bool(active.any().item()):
                state.update_spread(None)
                state.reset_tau()
                return eps_fused
        elif not active:
            state.update_spread(None)
            state.reset_tau()
            return eps_fused

        spread_cond_lob = raw_cond_lob if raw_cond_lob is not None else cond_lob
        if spread_cond_lob is None:
            return eps_fused
        if delta_base is None:
            delta_base = eps_fused.new_tensor(1.0)

        with torch.enable_grad():
            x_for_grad = x_t.detach().requires_grad_(True)
            eps_const = eps_fused.detach()
            x0_hat = self._predict_x0_from_eps(x_for_grad, eps_const, t)
            mid = decode_mid_price(
                x0_hat,
                spread_cond_lob.detach(),
                price_feature_index=self.price_feature_index,
                price_is_delta=self.price_is_delta,
            )
            spread = compute_spread(mid, self.asset_universe)
            if spread.shape[1] == 0:
                state.update_spread(None)
                state.reset_tau()
                return eps_fused

            stats_orders = raw_cond_orders if raw_cond_orders is not None else cond_orders
            if stats_orders is None:
                stats_group = spread.new_zeros(spread.shape[0], spread.shape[1], 4)
            else:
                stats_lob = raw_cond_lob if raw_cond_lob is not None else cond_lob
                rolling_stats = compute_rolling_stats(
                    stats_orders.detach(),
                    None if stats_lob is None else stats_lob.detach(),
                ).to(device=spread.device, dtype=spread.dtype)
                stats_group = group_rolling_stats(rolling_stats, self.asset_universe)

            delta_dyn = dynamic_delta(
                delta_base,
                stats_group,
                stress_head=stress_head,
                kappa=self.kappa,
            ).to(device=spread.device, dtype=spread.dtype)
            tau = update_tau(state.tau, spread.detach(), delta_dyn.detach(), active=active)
            tau = tau.to(device=spread.device, dtype=spread.dtype)
            arb_energy = energy(spread, delta_dyn, tau)
            grad = torch.autograd.grad(
                arb_energy,
                x_for_grad,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )[0]

        state.update_spread(spread.detach())
        if grad is None:
            return eps_fused
        grad = torch.nan_to_num(grad.detach(), nan=0.0, posinf=0.0, neginf=0.0)
        if not bool((grad != 0).any().item()):
            return eps_fused

        alpha_bar = self.alphas_cumprod.to(device=x_t.device, dtype=x_t.dtype)[t]
        lambda_t = guidance_lambda(
            t,
            self.alphas_cumprod.to(device=x_t.device),
            lambda_max=self.lambda_max,
            p=self.lambda_power,
        ).to(device=x_t.device, dtype=x_t.dtype)
        if alpha_bar.dim() == 0:
            alpha_bar = alpha_bar.expand(x_t.shape[0])
        if lambda_t.dim() == 0:
            lambda_t = lambda_t.expand(x_t.shape[0])
        scale = lambda_t * torch.sqrt((1.0 - alpha_bar).clamp_min(0.0))
        view_shape = (x_t.shape[0],) + (1,) * (x_t.dim() - 1)
        guided = eps_fused + scale.view(view_shape) * grad.to(
            device=eps_fused.device,
            dtype=eps_fused.dtype,
        )
        return guided.detach()


__all__ = [
    "ArbitrageEnergyGuidance",
    "StressHead",
    "calibrate_delta_base",
    "dynamic_delta",
    "energy",
    "group_rolling_stats",
    "phi",
    "rho",
]
