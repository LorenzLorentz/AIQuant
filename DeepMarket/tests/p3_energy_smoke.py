"""P3 annealed energy-guidance smoke tests.

Invoke:  python -m tests.p3_energy_smoke   (from DeepMarket/)
"""

from __future__ import annotations

import sys

import torch

import constants as cst
from configuration import Configuration
from constants import LearningHyperParameter as LHP
from models.diffusers.multi_asset.ablation_flags import GraphAblationFlags
from models.diffusers.multi_asset.arbitrage import (
    AnnealedGuidanceSchedule,
    ArbitrageEnergyGuidance,
    nav_gap_energy,
    reconstruct_x0_from_eps,
)
from models.diffusers.multi_asset.ma_gaussian_diffusion import MultiAssetGaussianDiffusion
from preprocessing.AssetUniverse import AssetUniverse


def _config(enabled=True, flags=None):
    cfg = Configuration()
    cfg.CHOSEN_MODEL = cst.Models.MA_TRADES
    cfg.MULTI_ASSET = True
    cfg.IS_AUGMENTATION = False
    cfg.ENERGY_GUIDANCE_ENABLED = enabled
    cfg.ENERGY_GUIDANCE_START_STEP = 3
    cfg.ENERGY_GUIDANCE_MAX_SCALE = 0.05
    cfg.ENERGY_GUIDANCE_DELTA = 1.0
    cfg.ENERGY_GUIDANCE_MAX_GRAD_NORM = 0.25
    cfg.ENERGY_PRICE_PROXY_INDEX = 5
    cfg.GRAPH_ABLATION_FLAGS = flags or GraphAblationFlags(disable_arb_guidance=False)
    cfg.HYPER_PARAMETERS[LHP.SEQ_SIZE] = 8
    cfg.HYPER_PARAMETERS[LHP.MASKED_SEQ_SIZE] = 2
    cfg.HYPER_PARAMETERS[LHP.NUM_DIFFUSIONSTEPS] = 4
    cfg.HYPER_PARAMETERS[LHP.CDT_DEPTH] = 1
    cfg.HYPER_PARAMETERS[LHP.CDT_NUM_HEADS] = 1
    from utils.utils import noise_scheduler
    cfg.BETAS = noise_scheduler(num_diffusion_timesteps=cfg.HYPER_PARAMETERS[LHP.NUM_DIFFUSIONSTEPS])
    return cfg


def _universe_with_basket():
    return AssetUniverse.etf_pair("ETF", "STOCK")


def _universe_without_basket():
    return AssetUniverse(
        assets=["ETF", "STOCK"],
        relation_types={(0, 1): 0, (1, 0): 1},
    )


def _make_raw_lob(batch_size=2, num_assets=2, k_lob=7):
    cond_lob = torch.zeros(
        batch_size, num_assets, k_lob, cst.N_LOB_LEVELS * cst.LEN_LEVEL,
        device=cst.DEVICE,
    )
    for b in range(batch_size):
        etf_mid = 110.0 + b
        stock_mid = 100.0 + b
        mids = [etf_mid, stock_mid]
        for asset in range(num_assets):
            spread = 0.5 + 0.1 * asset
            cond_lob[b, asset, :, 0] = mids[asset] + spread * 0.5
            cond_lob[b, asset, :, 2] = mids[asset] - spread * 0.5
    return cond_lob


def _make_raw_orders(batch_size=2, num_assets=2, k_cond=6):
    cond_orders = torch.randn(batch_size, num_assets, k_cond, cst.LEN_ORDER, device=cst.DEVICE)
    cond_orders[..., 1] = torch.randint(
        0, 3, cond_orders[..., 1].shape, device=cst.DEVICE
    ).float()
    cond_orders[..., 2] = cond_orders[..., 2].abs() + 0.1
    cond_orders[..., 4] = torch.where(cond_orders[..., 4] >= 0, 1.0, -1.0)
    return cond_orders


def check_schedule():
    schedule = AnnealedGuidanceSchedule(start_step=3, max_scale=0.05)
    t = torch.tensor([4, 3, 2, 0], device=cst.DEVICE)
    scale = schedule(t)
    assert scale.shape == (4,)
    assert scale[0].item() == 0.0
    assert scale[1].item() == 0.0
    assert scale[2].item() > 0.0
    assert torch.isclose(scale[3], torch.tensor(0.05, device=cst.DEVICE))
    print("  [ok] annealed guidance schedule")


def check_reconstruction_and_energy():
    B, N, K, F = 2, 2, 2, 8
    cfg = _config()
    x_t = torch.randn(B, N, K, F, device=cst.DEVICE)
    eps = torch.randn(B, N, K, F, device=cst.DEVICE)
    t = torch.tensor([0, 2], device=cst.DEVICE)
    alphas_cumprod = torch.cumprod(1 - cfg.BETAS, dim=0)
    x_hat = reconstruct_x0_from_eps(x_t, eps, t, alphas_cumprod)
    assert x_hat.shape == eps.shape
    assert torch.isfinite(x_hat).all()

    energy_and_gaps = nav_gap_energy(
        _universe_with_basket(),
        x_hat,
        _make_raw_lob(B, N),
        price_proxy_index=5,
        delta=1.0,
    )
    assert energy_and_gaps is not None
    energy, gaps = energy_and_gaps
    assert energy.shape == (B,)
    assert gaps.shape == (B, 1)
    assert torch.isfinite(energy).all()
    assert torch.isfinite(gaps).all()

    eps_leaf = eps.detach().clone().requires_grad_(True)
    x_hat_leaf = reconstruct_x0_from_eps(
        x_t,
        eps_leaf,
        t,
        alphas_cumprod,
    )
    energy_leaf, _ = nav_gap_energy(
        _universe_with_basket(),
        x_hat_leaf,
        _make_raw_lob(B, N),
        price_proxy_index=5,
        delta=1.0,
    )
    energy_leaf.sum().backward()
    assert eps_leaf.grad is not None
    assert torch.isfinite(eps_leaf.grad).all()
    print("  [ok] x_hat_0 reconstruction and NAV-gap energy")


def check_guidance_noops_and_late_step_change():
    B, N, K, F = 2, 2, 2, 8
    cfg = _config()
    alphas_cumprod = torch.cumprod(1 - cfg.BETAS, dim=0)
    eps = torch.zeros(B, N, K, F, device=cst.DEVICE)
    x_t = torch.zeros_like(eps)
    cond_lob = _make_raw_lob(B, N)
    late_t = torch.zeros(B, dtype=torch.long, device=cst.DEVICE)
    high_t = torch.full((B,), 3, dtype=torch.long, device=cst.DEVICE)

    no_basket = ArbitrageEnergyGuidance(
        _universe_without_basket(),
        alphas_cumprod,
        enabled=True,
        flags=GraphAblationFlags(disable_arb_guidance=False),
        start_step=3,
        max_scale=0.05,
    ).to(cst.DEVICE)
    assert torch.equal(no_basket(eps, x_t=x_t, t=late_t, cond_lob=cond_lob), eps)

    disabled = ArbitrageEnergyGuidance(
        _universe_with_basket(),
        alphas_cumprod,
        enabled=False,
        flags=GraphAblationFlags(disable_arb_guidance=False),
        start_step=3,
        max_scale=0.05,
    ).to(cst.DEVICE)
    assert torch.equal(disabled(eps, x_t=x_t, t=late_t, cond_lob=cond_lob), eps)

    high_step = ArbitrageEnergyGuidance(
        _universe_with_basket(),
        alphas_cumprod,
        enabled=True,
        flags=GraphAblationFlags(disable_arb_guidance=False),
        start_step=3,
        max_scale=0.05,
        max_grad_norm=0.25,
    ).to(cst.DEVICE)
    assert torch.equal(high_step(eps, x_t=x_t, t=high_t, cond_lob=cond_lob), eps)

    late_step = ArbitrageEnergyGuidance(
        _universe_with_basket(),
        alphas_cumprod,
        enabled=True,
        flags=GraphAblationFlags(disable_arb_guidance=False),
        start_step=3,
        max_scale=0.05,
        max_grad_norm=0.25,
    ).to(cst.DEVICE)
    out = late_step(eps, x_t=x_t, t=late_t, cond_lob=cond_lob)
    assert out.shape == eps.shape
    assert torch.isfinite(out).all()
    assert not torch.equal(out, eps)
    applied_grad = (eps - out) / 0.05
    grad_norm = applied_grad.flatten(start_dim=1).norm(dim=1)
    assert torch.all(grad_norm <= 0.2501)
    print("  [ok] guidance no-ops, late-step change, and grad clipping")


def check_diffuser_single_step():
    B, N, K_gen, F = 2, 2, 2, 8
    cfg = _config(enabled=True)
    universe = _universe_with_basket()
    diffuser = MultiAssetGaussianDiffusion(cfg, universe, feature_augmenter=None).to(cst.DEVICE)
    raw_orders = _make_raw_orders(B, N, k_cond=6)
    raw_lob = _make_raw_lob(B, N, k_lob=7)
    cond_orders = torch.randn(B, N, 6, F, device=cst.DEVICE)
    x_0 = torch.randn(B, N, K_gen, F, device=cst.DEVICE)
    t = torch.zeros(B, dtype=torch.long, device=cst.DEVICE)
    x_t, noise = diffuser.forward_reparametrized(x_0, t)
    weights = torch.ones(cfg.HYPER_PARAMETERS[LHP.NUM_DIFFUSIONSTEPS]).numpy()
    x_recon = diffuser.ddpm_single_step(
        x_0, x_t, x_t, t, cond_orders, noise, weights, raw_lob,
        raw_cond_orders=raw_orders, raw_cond_lob=raw_lob,
    )
    assert x_recon.shape == x_0.shape
    batch_loss, simple_loss, vlb_loss = diffuser.loss()
    assert torch.isfinite(batch_loss).all()
    assert torch.isfinite(simple_loss).all()
    assert torch.isfinite(vlb_loss).all()
    batch_loss.mean().backward()
    diffuser.init_losses()
    print("  [ok] MultiAssetGaussianDiffusion energy ddpm_single_step")


def main() -> int:
    print(f"device: {cst.DEVICE}")
    check_schedule()
    check_reconstruction_and_energy()
    check_guidance_noops_and_late_step_change()
    check_diffuser_single_step()
    print("\nP3 energy smoke test PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
