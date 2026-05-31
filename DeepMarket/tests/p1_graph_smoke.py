"""P1 graph smoke tests without Lightning.

Invoke:  python -m tests.p1_graph_smoke   (from DeepMarket/)
"""

from __future__ import annotations

import sys

import torch

import constants as cst
from configuration import Configuration
from constants import LearningHyperParameter as LHP
from models.diffusers.multi_asset.ablation_flags import GraphAblationFlags
from models.diffusers.multi_asset.graph import (
    AttentionAggregator,
    EdgeWeightNet,
    GraphCoupler,
    MessageFunction,
    NoiseFusion,
    RelationEmbedding,
    compute_rolling_stats,
)
from models.diffusers.multi_asset.ma_gaussian_diffusion import MultiAssetGaussianDiffusion
from preprocessing.AssetUniverse import AssetUniverse


def _make_raw_conditioning(batch_size=2, num_assets=2, k_cond=7):
    cond_orders = torch.randn(batch_size, num_assets, k_cond, cst.LEN_ORDER, device=cst.DEVICE)
    cond_orders[..., 1] = torch.randint(
        0, 3, cond_orders[..., 1].shape, device=cst.DEVICE
    ).float()
    cond_orders[..., 2] = cond_orders[..., 2].abs() + 0.1
    cond_orders[..., 4] = torch.where(cond_orders[..., 4] >= 0, 1.0, -1.0)
    cond_lob = torch.randn(
        batch_size, num_assets, k_cond + 1, cst.N_LOB_LEVELS * cst.LEN_LEVEL,
        device=cst.DEVICE,
    )
    return cond_orders, cond_lob


def _universe(num_assets=2):
    pairs = [(j, i) for j in range(num_assets) for i in range(num_assets) if j != i]
    return AssetUniverse(
        assets=[f"ASSET_{i}" for i in range(num_assets)],
        relation_types={(j, i): k for k, (j, i) in enumerate(pairs)},
    )


def _config(flags=None):
    cfg = Configuration()
    cfg.CHOSEN_MODEL = cst.Models.MA_TRADES
    cfg.MULTI_ASSET = True
    cfg.IS_AUGMENTATION = False
    cfg.GRAPH_ABLATION_FLAGS = flags or GraphAblationFlags()
    cfg.HYPER_PARAMETERS[LHP.SEQ_SIZE] = 8
    cfg.HYPER_PARAMETERS[LHP.MASKED_SEQ_SIZE] = 2
    cfg.HYPER_PARAMETERS[LHP.NUM_DIFFUSIONSTEPS] = 4
    cfg.HYPER_PARAMETERS[LHP.CDT_DEPTH] = 1
    cfg.HYPER_PARAMETERS[LHP.CDT_NUM_HEADS] = 1
    from utils.utils import noise_scheduler
    cfg.BETAS = noise_scheduler(num_diffusion_timesteps=cfg.HYPER_PARAMETERS[LHP.NUM_DIFFUSIONSTEPS])
    return cfg


def check_graph_modules():
    B, N, K, F = 2, 2, 3, 8
    raw_orders, raw_lob = _make_raw_conditioning(B, N, k_cond=7)
    stats = compute_rolling_stats(raw_orders, raw_lob)
    assert stats.shape == (B, N, 4)
    assert torch.isfinite(stats).all()

    relations = RelationEmbedding(num_relation_types=2, embedding_dim=8).to(cst.DEVICE)
    rel = relations(torch.tensor([0, 1], device=cst.DEVICE))
    assert rel.shape == (2, 8)

    edge_net = EdgeWeightNet(stats_dim=4, relation_dim=8, hidden_dim=16).to(cst.DEVICE)
    weight = edge_net(stats[:, 0], stats[:, 1], rel[0])
    assert weight.shape == (B, 1)
    assert torch.all((weight >= 0.0) & (weight <= 1.0))

    msg_fn = MessageFunction(feature_dim=F, hidden_dim=16).to(cst.DEVICE)
    x_src = torch.randn(B, K, F, device=cst.DEVICE)
    eps_src = torch.randn(B, K, F, device=cst.DEVICE)
    weight_for_grad = weight.detach().clone().requires_grad_(True)
    msg = msg_fn(x_src, eps_src, weight_for_grad)
    assert msg.shape == (B, K, F)
    msg.sum().backward()
    assert weight_for_grad.grad is not None

    agg = AttentionAggregator(feature_dim=F, attention_dim=16).to(cst.DEVICE)
    one_incoming = msg.detach().unsqueeze(1)
    agg_out = agg(one_incoming, x_src.unsqueeze(1), eps_src)
    assert torch.allclose(agg_out, one_incoming[:, 0])

    fusion = NoiseFusion(feature_dim=F, hidden_dim=16).to(cst.DEVICE)
    fused = fusion(eps_src, msg.detach())
    assert torch.allclose(fused, eps_src)
    print("  [ok] graph modules")


def check_graph_coupler():
    B, N, K, F = 2, 2, 2, 8
    raw_orders, raw_lob = _make_raw_conditioning(B, N, k_cond=6)
    eps = torch.randn(B, N, K, F, device=cst.DEVICE)
    x_t = torch.randn(B, N, K, F, device=cst.DEVICE)

    coupler = GraphCoupler(_universe(N), feature_dim=F, hidden_dim=16).to(cst.DEVICE)
    out = coupler(eps, x_t=x_t, cond_orders=raw_orders, cond_lob=raw_lob)
    assert torch.allclose(out, eps)
    out.sum().backward()
    assert coupler.gamma.grad is not None

    disabled = GraphCoupler(
        _universe(N), feature_dim=F, hidden_dim=16,
        flags=GraphAblationFlags(disable_graph=True),
    ).to(cst.DEVICE)
    disabled.noise_fusion.gamma.data.fill_(1.0)
    disabled_out = disabled(eps, x_t=x_t, cond_orders=raw_orders, cond_lob=raw_lob)
    assert torch.equal(disabled_out, eps)
    print("  [ok] GraphCoupler gamma and disable_graph")


def check_diffuser_fuse():
    B, N, K, F = 2, 2, 2, 8
    cfg = _config()
    universe = _universe(N)
    diffuser = MultiAssetGaussianDiffusion(cfg, universe, feature_augmenter=None).to(cst.DEVICE)
    raw_orders, raw_lob = _make_raw_conditioning(B, N, k_cond=6)
    eps = torch.randn(B, N, K, F, device=cst.DEVICE)
    x_t = torch.randn(B, N, K, F, device=cst.DEVICE)
    out = diffuser.fuse(eps, x_t=x_t, raw_cond_orders=raw_orders, raw_cond_lob=raw_lob)
    assert torch.allclose(out, eps)

    disabled_cfg = _config(GraphAblationFlags(disable_graph=True))
    disabled = MultiAssetGaussianDiffusion(disabled_cfg, universe, feature_augmenter=None).to(cst.DEVICE)
    disabled.graph_coupler.noise_fusion.gamma.data.fill_(1.0)
    disabled_out = disabled.fuse(eps, x_t=x_t, raw_cond_orders=raw_orders, raw_cond_lob=raw_lob)
    assert torch.equal(disabled_out, eps)
    print("  [ok] MultiAssetGaussianDiffusion.fuse")


def check_diffuser_single_step():
    B, N, K_gen, F = 2, 2, 2, 8
    cfg = _config()
    universe = _universe(N)
    diffuser = MultiAssetGaussianDiffusion(cfg, universe, feature_augmenter=None).to(cst.DEVICE)
    raw_orders, raw_lob = _make_raw_conditioning(B, N, k_cond=6)
    cond_orders = torch.randn(B, N, 6, F, device=cst.DEVICE)
    cond_lob = torch.randn(B, N, 7, cst.N_LOB_LEVELS * cst.LEN_LEVEL, device=cst.DEVICE)
    x_0 = torch.randn(B, N, K_gen, F, device=cst.DEVICE)
    t = torch.randint(0, cfg.HYPER_PARAMETERS[LHP.NUM_DIFFUSIONSTEPS], (B,), device=cst.DEVICE)
    x_t, noise = diffuser.forward_reparametrized(x_0, t)
    weights = torch.ones(cfg.HYPER_PARAMETERS[LHP.NUM_DIFFUSIONSTEPS]).numpy()
    x_recon = diffuser.ddpm_single_step(
        x_0, x_t, x_t, t, cond_orders, noise, weights, cond_lob,
        raw_cond_orders=raw_orders, raw_cond_lob=raw_lob,
    )
    assert x_recon.shape == x_0.shape
    batch_loss, simple_loss, vlb_loss = diffuser.loss()
    assert torch.isfinite(batch_loss).all()
    assert torch.isfinite(simple_loss).all()
    assert torch.isfinite(vlb_loss).all()
    batch_loss.mean().backward()
    assert diffuser.graph_coupler.gamma.grad is not None
    diffuser.init_losses()
    print("  [ok] MultiAssetGaussianDiffusion.ddpm_single_step")


def main() -> int:
    print(f"device: {cst.DEVICE}")
    check_graph_modules()
    check_graph_coupler()
    check_diffuser_fuse()
    check_diffuser_single_step()
    print("\nP1 graph smoke test PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
