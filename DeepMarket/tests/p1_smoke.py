"""P1 smoke test for graph-coupled multi-asset diffusion.

Invoke from ``DeepMarket/``:

    python -m tests.p1_smoke
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

# utils.utils imports torchvision for helper functions that this smoke test
# does not use. Some lightweight local envs have torch but not torchvision.
try:
    import torchvision  # noqa: F401
    from torchvision import transforms  # noqa: F401
except (ModuleNotFoundError, ImportError):
    torchvision_stub = types.ModuleType("torchvision")
    torchvision_models_stub = types.ModuleType("torchvision.models")
    torchvision_transforms_stub = types.ModuleType("torchvision.transforms")
    torchvision_stub.models = torchvision_models_stub
    torchvision_stub.transforms = torchvision_transforms_stub
    sys.modules["torchvision"] = torchvision_stub
    sys.modules["torchvision.models"] = torchvision_models_stub
    sys.modules["torchvision.transforms"] = torchvision_transforms_stub

import constants as cst
from constants import LearningHyperParameter as LHP
from models.diffusers.multi_asset.graph import (
    AttentionAggregator,
    EdgeWeightNet,
    MessageFunction,
    NoiseFusion,
    RelationEmbedding,
    compute_rolling_stats,
)
from models.diffusers.multi_asset.ma_gaussian_diffusion import MultiAssetGaussianDiffusion
from preprocessing.AssetUniverse import AssetUniverse


def _build_config(disable_graph: bool = False) -> SimpleNamespace:
    hp = {lp: None for lp in LHP}
    hp[LHP.SEQ_SIZE] = 16
    hp[LHP.MASKED_SEQ_SIZE] = 1
    hp[LHP.NUM_DIFFUSIONSTEPS] = 8
    hp[LHP.CDT_DEPTH] = 1
    hp[LHP.AUGMENT_DIM] = 64
    hp[LHP.CDT_NUM_HEADS] = 1
    hp[LHP.CDT_MLP_RATIO] = 4
    hp[LHP.BATCH_SIZE] = 2
    hp[LHP.TEST_BATCH_SIZE] = 2
    hp[LHP.DROPOUT] = 0.1
    hp[LHP.CONDITIONAL_DROPOUT] = 0.0
    hp[LHP.SIZE_TYPE_EMB] = 3
    hp[LHP.SIZE_ORDER_EMB] = cst.LEN_ORDER + hp[LHP.SIZE_TYPE_EMB] - 1
    hp[LHP.LAMBDA] = 0.01
    hp[LHP.DDIM_ETA] = 0.0
    hp[LHP.DDIM_NSTEPS] = 4
    return SimpleNamespace(
        HYPER_PARAMETERS=hp,
        SAMPLING_TYPE="DDPM",
        IS_AUGMENTATION=False,
        COND_METHOD="concatenation",
        COND_TYPE="full",
        BETAS=torch.linspace(1e-4, 2e-2, hp[LHP.NUM_DIFFUSIONSTEPS], device=cst.DEVICE),
        DISABLE_GRAPH=disable_graph,
        FREEZE_EDGE_WEIGHTS=False,
        DISABLE_ARB_GUIDANCE=False,
        GRAPH_RELATION_DIM=8,
        GRAPH_HIDDEN_DIM=32,
    )


def _build_diffuser(disable_graph: bool = False) -> tuple[MultiAssetGaussianDiffusion, SimpleNamespace]:
    cfg = _build_config(disable_graph=disable_graph)
    universe = AssetUniverse(
        assets=["ETF", "CONST"],
        relation_types={(0, 1): 0, (1, 0): 1},
    )
    cfg.ASSET_UNIVERSE = universe
    diffuser = MultiAssetGaussianDiffusion(cfg, universe, None).to(cst.DEVICE)
    return diffuser, cfg


def _make_batch(cfg: SimpleNamespace, batch_size: int = 2):
    K_total = cfg.HYPER_PARAMETERS[LHP.SEQ_SIZE]
    K_gen = cfg.HYPER_PARAMETERS[LHP.MASKED_SEQ_SIZE]
    K_cond = K_total - K_gen
    N = 2
    cond_orders = torch.randn(batch_size, N, K_cond, cst.LEN_ORDER, device=cst.DEVICE)
    cond_orders[..., 0] = cond_orders[..., 0].abs()
    cond_orders[..., 1] = torch.randint(0, 3, cond_orders[..., 1].shape, device=cst.DEVICE).float()
    cond_orders[..., 2] = torch.randint(0, 2, cond_orders[..., 2].shape, device=cst.DEVICE).float()
    cond_orders[..., 4] = cond_orders[..., 4].abs() + 0.1
    x_0 = torch.randn(batch_size, N, K_gen, cst.LEN_ORDER, device=cst.DEVICE)
    x_0[..., 1] = torch.randint(0, 3, x_0[..., 1].shape, device=cst.DEVICE).float()
    cond_lob = torch.randn(
        batch_size,
        N,
        K_cond + 1,
        cst.N_LOB_LEVELS * cst.LEN_LEVEL,
        device=cst.DEVICE,
    )
    base = torch.linspace(100.0, 101.0, K_cond + 1, device=cst.DEVICE)
    cond_lob[..., 0] = base
    cond_lob[..., 2] = base + 0.02
    return cond_orders, x_0, cond_lob


def _type_embedding(x_0, cond):
    type_embedder = torch.tensor(
        [[0.4438, -0.2984, 0.2888],
         [0.8249, 0.5847, 0.1448],
         [1.5600, -1.2847, 1.0294]],
        device=cst.DEVICE,
        dtype=torch.float32,
    )
    x_type = type_embedder[x_0[..., 1].long()]
    cond_type = type_embedder[cond[..., 1].long()]
    x_0 = torch.cat((x_0[..., :1], x_type, x_0[..., 2:]), dim=-1)
    cond = torch.cat((cond[..., :1], cond_type, cond[..., 2:]), dim=-1)
    return x_0, cond


def check_graph_modules():
    B, N, K, F_graph = 2, 3, 6, 8
    orders = torch.randn(B, N, K, cst.LEN_ORDER, device=cst.DEVICE)
    orders[..., 0] = orders[..., 0].abs()
    orders[..., 1] = torch.randint(0, 3, orders[..., 1].shape, device=cst.DEVICE).float()
    orders[..., 2] = torch.randint(0, 2, orders[..., 2].shape, device=cst.DEVICE).float()
    orders[..., 4] = orders[..., 4].abs() + 0.1
    lob = torch.randn(B, N, K + 1, cst.N_LOB_LEVELS * cst.LEN_LEVEL, device=cst.DEVICE)
    lob[..., 0] = torch.linspace(10.0, 11.0, K + 1, device=cst.DEVICE)
    lob[..., 2] = lob[..., 0] + 0.01

    stats = compute_rolling_stats(orders, lob)
    assert stats.shape == (B, N, 4)
    assert torch.isfinite(stats).all()

    src = torch.tensor([0, 1, 0], device=cst.DEVICE)
    dst = torch.tensor([2, 2, 1], device=cst.DEVICE)
    relation_ids = torch.tensor([0, 1, 2], device=cst.DEVICE)
    rel = RelationEmbedding(num_relation_types=3, relation_dim=8).to(cst.DEVICE)
    edge_net = EdgeWeightNet(stats_dim=4, relation_dim=8, hidden_dim=16).to(cst.DEVICE)
    relation_emb = rel(relation_ids).unsqueeze(0).expand(B, -1, -1)
    weights = edge_net(stats[:, src, :], stats[:, dst, :], relation_emb)
    assert weights.shape == (B, 3)
    assert torch.all(weights > 0) and torch.all(weights < 1)
    weights.sum().backward()
    assert rel.embedding.weight.grad is not None

    x_src = torch.randn(B, 3, 2, F_graph, device=cst.DEVICE)
    eps_src = torch.randn(B, 3, 2, F_graph, device=cst.DEVICE)
    edge_weight = torch.rand(B, 3, device=cst.DEVICE, requires_grad=True)
    msg_fn = MessageFunction(feature_dim=F_graph, hidden_dim=16).to(cst.DEVICE)
    messages = msg_fn(x_src, eps_src, edge_weight)
    assert messages.shape == x_src.shape
    messages.sum().backward()
    assert edge_weight.grad is not None and edge_weight.grad.abs().sum() > 0

    aggregator = AttentionAggregator(feature_dim=F_graph, hidden_dim=16).to(cst.DEVICE)
    ones = torch.ones(B, 3, 2, F_graph, device=cst.DEVICE)
    ones[:, 2] = 3.0
    node_states = torch.randn(B, N, 2, F_graph, device=cst.DEVICE)
    aggregated = aggregator(ones, node_states, src, dst, num_nodes=N)
    assert torch.allclose(aggregated[:, 1], ones[:, 2])
    assert torch.allclose(aggregated[:, 2], torch.ones_like(aggregated[:, 2]))

    fusion = NoiseFusion(feature_dim=F_graph, hidden_dim=16).to(cst.DEVICE)
    eps = torch.randn(B, N, 2, F_graph, device=cst.DEVICE)
    msg = torch.randn_like(eps)
    out = fusion(eps, msg)
    assert torch.equal(out, eps), "NoiseFusion must be exact identity at init"
    fusion.gamma.data.fill_(0.25)
    assert not torch.equal(fusion(eps, msg), eps)
    print("  [ok] graph modules")


def check_diffuser_fusion_and_ablation():
    diffuser, cfg = _build_diffuser(disable_graph=False)
    cond_orders, x_0_raw, cond_lob = _make_batch(cfg)
    x_0, _ = _type_embedding(x_0_raw, cond_orders)
    t = torch.randint(0, cfg.HYPER_PARAMETERS[LHP.NUM_DIFFUSIONSTEPS], (x_0.shape[0],), device=cst.DEVICE)
    x_t, _ = diffuser.forward_reparametrized(x_0, t)
    eps = torch.randn_like(x_t)

    out_init = diffuser.fuse(
        eps,
        x_t=x_t,
        t=t,
        raw_cond_orders=cond_orders,
        raw_cond_lob=cond_lob,
    )
    assert torch.equal(out_init, eps), "gamma=0 must recover P0 exactly"

    diffuser.noise_fusion.gamma.data.fill_(0.25)
    out_graph = diffuser.fuse(
        eps,
        x_t=x_t,
        t=t,
        raw_cond_orders=cond_orders,
        raw_cond_lob=cond_lob,
    )
    assert out_graph.shape == eps.shape
    assert not torch.equal(out_graph, eps), "nonzero gamma should change fused noise"

    diffuser.ablation_flags.disable_graph = True
    out_disabled = diffuser.fuse(
        eps,
        x_t=x_t,
        t=t,
        raw_cond_orders=cond_orders,
        raw_cond_lob=cond_lob,
    )
    assert torch.equal(out_disabled, eps), "disable_graph must skip graph fusion exactly"
    print("  [ok] graph fusion + ablation")


def check_training_step_gamma_grad():
    diffuser, cfg = _build_diffuser(disable_graph=False)
    cond_orders_raw, x_0_raw, cond_lob = _make_batch(cfg)
    x_0, cond_orders = _type_embedding(x_0_raw, cond_orders_raw)
    t = torch.randint(0, cfg.HYPER_PARAMETERS[LHP.NUM_DIFFUSIONSTEPS], (x_0.shape[0],), device=cst.DEVICE)
    x_t, noise = diffuser.forward_reparametrized(x_0, t)
    weights = np.ones(cfg.HYPER_PARAMETERS[LHP.NUM_DIFFUSIONSTEPS], dtype=float)
    recon = diffuser.ddpm_single_step(
        x_0,
        x_t,
        x_t,
        t,
        cond_orders,
        noise,
        weights,
        cond_lob,
        raw_cond_orders=cond_orders_raw,
        raw_cond_lob=cond_lob,
    )
    assert torch.isfinite(recon).all()
    loss = torch.mean(diffuser.loss()[0])
    assert torch.isfinite(loss)
    loss.backward()
    gamma_grad = diffuser.graph_gamma.grad
    assert gamma_grad is not None and torch.isfinite(gamma_grad)
    print(f"  [ok] ddpm_single_step loss={loss.item():.4f}, gamma_grad={gamma_grad.item():.6f}")


def check_cross_corr_metric():
    from lob_bench.cross_asset.cross_corr import compare_cross_corr

    rng = np.random.default_rng(123)
    T, F = 64, cst.N_LOB_LEVELS * cst.LEN_LEVEL
    base = 100 + np.cumsum(rng.normal(0, 0.05, size=T))
    real = np.zeros((2, T, F), dtype=float)
    p1 = np.zeros_like(real)
    p0 = np.zeros_like(real)

    real[0, :, 0] = base
    real[0, :, 2] = base + 0.01
    real[1, :, 0] = base + rng.normal(0, 0.01, size=T)
    real[1, :, 2] = real[1, :, 0] + 0.01

    p1[0, :, 0] = base + rng.normal(0, 0.02, size=T)
    p1[0, :, 2] = p1[0, :, 0] + 0.01
    p1[1, :, 0] = base + rng.normal(0, 0.03, size=T)
    p1[1, :, 2] = p1[1, :, 0] + 0.01

    independent = 100 + np.cumsum(rng.normal(0, 0.05, size=T))
    p0[0, :, 0] = base
    p0[0, :, 2] = base + 0.01
    p0[1, :, 0] = independent
    p0[1, :, 2] = independent + 0.01

    result = compare_cross_corr(real, p0, p1)
    assert result["p1_closer_to_real"], result
    print("  [ok] cross_corr metric")


def main() -> int:
    print(f"device: {cst.DEVICE}")
    print("[1] graph modules")
    check_graph_modules()
    print("[2] diffuser fusion/ablation")
    check_diffuser_fusion_and_ablation()
    print("[3] training step gamma gradient")
    check_training_step_gamma_grad()
    print("[4] cross-asset correlation metric")
    check_cross_corr_metric()
    print("\nP1 smoke test PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
