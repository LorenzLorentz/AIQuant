"""P2 spread-conditioning smoke tests.

Invoke:  python -m tests.p2_spread_smoke   (from DeepMarket/)
"""

from __future__ import annotations

import sys

import torch

import constants as cst
from configuration import Configuration
from constants import LearningHyperParameter as LHP
from models.diffusers.multi_asset.ablation_flags import GraphAblationFlags
from models.diffusers.multi_asset.ma_gaussian_diffusion import MultiAssetGaussianDiffusion
from models.diffusers.multi_asset.spread import (
    SpreadConditioner,
    compute_spread_context,
    compute_top_of_book_mid_spread,
)
from preprocessing.AssetUniverse import AssetUniverse


def _config(enabled=True, flags=None):
    cfg = Configuration()
    cfg.CHOSEN_MODEL = cst.Models.MA_TRADES
    cfg.MULTI_ASSET = True
    cfg.IS_AUGMENTATION = False
    cfg.SPREAD_CONDITIONING_ENABLED = enabled
    cfg.GRAPH_ABLATION_FLAGS = flags or GraphAblationFlags()
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
        for asset in range(num_assets):
            ask = 100.0 + 10.0 * asset + b
            bid = ask - 0.5 - 0.1 * asset
            cond_lob[b, asset, :, 0] = ask
            cond_lob[b, asset, :, 2] = bid
    return cond_lob


def _make_raw_orders(batch_size=2, num_assets=2, k_cond=6):
    cond_orders = torch.randn(batch_size, num_assets, k_cond, cst.LEN_ORDER, device=cst.DEVICE)
    cond_orders[..., 1] = torch.randint(
        0, 3, cond_orders[..., 1].shape, device=cst.DEVICE
    ).float()
    cond_orders[..., 2] = cond_orders[..., 2].abs() + 0.1
    cond_orders[..., 4] = torch.where(cond_orders[..., 4] >= 0, 1.0, -1.0)
    return cond_orders


def check_spread_context():
    cond_lob = _make_raw_lob()
    mid, spread = compute_top_of_book_mid_spread(cond_lob)
    assert mid.shape == (2, 2)
    assert spread.shape == (2, 2)
    context = compute_spread_context(_universe_with_basket(), cond_lob, context_dim=4)
    assert context is not None
    assert context.shape == (2, 2, 4)
    assert torch.isfinite(context).all()
    assert compute_spread_context(_universe_without_basket(), cond_lob) is None
    print("  [ok] spread context")


def check_conditioner_noop_and_gamma():
    B, N, K, F = 2, 2, 2, 8
    eps = torch.randn(B, N, K, F, device=cst.DEVICE)
    cond_lob = _make_raw_lob(B, N, k_lob=7)

    no_basket = SpreadConditioner(
        _universe_without_basket(),
        feature_dim=F,
        enabled=True,
    ).to(cst.DEVICE)
    no_basket.gamma_spread.data.fill_(1.0)
    assert torch.equal(no_basket(eps, cond_lob=cond_lob), eps)

    conditioner = SpreadConditioner(
        _universe_with_basket(),
        feature_dim=F,
        enabled=True,
    ).to(cst.DEVICE)
    out = conditioner(eps, cond_lob=cond_lob)
    assert torch.allclose(out, eps)
    out.sum().backward()
    assert conditioner.gamma_spread.grad is not None

    disabled = SpreadConditioner(
        _universe_with_basket(),
        feature_dim=F,
        enabled=True,
        flags=GraphAblationFlags(disable_spread_conditioning=True),
    ).to(cst.DEVICE)
    disabled.gamma_spread.data.fill_(1.0)
    assert torch.equal(disabled(eps, cond_lob=cond_lob), eps)
    print("  [ok] SpreadConditioner no-op, gamma, and flag")


def check_diffuser_single_step():
    B, N, K_gen, F = 2, 2, 2, 8
    cfg = _config(enabled=True)
    universe = _universe_with_basket()
    diffuser = MultiAssetGaussianDiffusion(cfg, universe, feature_augmenter=None).to(cst.DEVICE)
    raw_orders = _make_raw_orders(B, N, k_cond=6)
    raw_lob = _make_raw_lob(B, N, k_lob=7)
    cond_orders = torch.randn(B, N, 6, F, device=cst.DEVICE)
    x_0 = torch.randn(B, N, K_gen, F, device=cst.DEVICE)
    t = torch.randint(0, cfg.HYPER_PARAMETERS[LHP.NUM_DIFFUSIONSTEPS], (B,), device=cst.DEVICE)
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
    assert diffuser.spread_conditioner.gamma_spread.grad is not None
    diffuser.init_losses()
    print("  [ok] MultiAssetGaussianDiffusion spread ddpm_single_step")


def main() -> int:
    print(f"device: {cst.DEVICE}")
    check_spread_context()
    check_conditioner_noop_and_gamma()
    check_diffuser_single_step()
    print("\nP2 spread smoke test PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
