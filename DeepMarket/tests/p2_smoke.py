"""P2 smoke test for spread-aware multi-asset diffusion.

Invoke from ``DeepMarket/``:

    python -m tests.p2_smoke
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

# Some lightweight local envs have torch but not einops. The production code
# depends on einops; this smoke test only needs the narrow patterns below.
try:
    import einops  # noqa: F401
except ModuleNotFoundError:
    import types

    einops_stub = types.ModuleType("einops")

    def _repeat(tensor, pattern, **sizes):
        if pattern == "b -> b n l d":
            return tensor.view(-1, 1, 1, 1).expand(-1, sizes["n"], sizes["l"], sizes["d"])
        if pattern == "b -> b n 1 d":
            return tensor.view(-1, 1, 1, 1).expand(-1, sizes["n"], 1, sizes["d"])
        raise NotImplementedError(f"test einops.repeat stub does not support {pattern!r}")

    def _rearrange(tensor, pattern, **sizes):
        if pattern == "n l f -> n (l f)":
            return tensor.reshape(tensor.shape[0], tensor.shape[1] * tensor.shape[2])
        if pattern == "n (l d) -> n l d":
            return tensor.reshape(tensor.shape[0], sizes["l"], sizes["d"])
        if pattern == "b l (h j) -> b h l j":
            h = sizes["h"]
            b, l, d = tensor.shape
            return tensor.reshape(b, l, h, d // h).permute(0, 2, 1, 3)
        if pattern == "b h l j -> b l (h j)":
            b, h, l, j = tensor.shape
            return tensor.permute(0, 2, 1, 3).reshape(b, l, h * j)
        raise NotImplementedError(f"test einops.rearrange stub does not support {pattern!r}")

    einops_stub.repeat = _repeat
    einops_stub.rearrange = _rearrange
    sys.modules["einops"] = einops_stub

# utils.utils imports torchvision for helpers this smoke test does not use.
try:
    import torchvision  # noqa: F401
except ModuleNotFoundError:
    import types

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
from models.diffusers.multi_asset.arbitrage import (
    ReverseLoopState,
    broadcast_spread_to_assets,
    compute_spread,
    decode_mid_price,
    spread_inject_active,
)
from models.diffusers.multi_asset.ma_gaussian_diffusion import MultiAssetGaussianDiffusion
from models.diffusers.multi_asset.shared_score_net import SharedScoreNet
from preprocessing.AssetUniverse import AssetUniverse


def _seed(value: int = 123) -> None:
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def _build_config(disable_spread_cond: bool = False) -> SimpleNamespace:
    hp = {lp: None for lp in LHP}
    hp[LHP.SEQ_SIZE] = 8
    hp[LHP.MASKED_SEQ_SIZE] = 1
    hp[LHP.NUM_DIFFUSIONSTEPS] = 4
    hp[LHP.CDT_DEPTH] = 1
    hp[LHP.AUGMENT_DIM] = 64
    hp[LHP.CDT_NUM_HEADS] = 1
    hp[LHP.CDT_MLP_RATIO] = 4
    hp[LHP.BATCH_SIZE] = 2
    hp[LHP.TEST_BATCH_SIZE] = 2
    hp[LHP.DROPOUT] = 0.0
    hp[LHP.CONDITIONAL_DROPOUT] = 0.0
    hp[LHP.SIZE_TYPE_EMB] = 3
    hp[LHP.SIZE_ORDER_EMB] = cst.LEN_ORDER + hp[LHP.SIZE_TYPE_EMB] - 1
    hp[LHP.LAMBDA] = 0.01
    hp[LHP.DDIM_ETA] = 0.0
    hp[LHP.DDIM_NSTEPS] = 2
    return SimpleNamespace(
        HYPER_PARAMETERS=hp,
        SAMPLING_TYPE="DDPM",
        IS_AUGMENTATION=False,
        COND_METHOD="concatenation",
        COND_TYPE="full",
        BETAS=torch.linspace(1e-4, 2e-2, hp[LHP.NUM_DIFFUSIONSTEPS], device=cst.DEVICE),
        DISABLE_GRAPH=True,
        FREEZE_EDGE_WEIGHTS=False,
        DISABLE_SPREAD_COND=disable_spread_cond,
        DISABLE_ARB_GUIDANCE=True,
        GRAPH_RELATION_DIM=8,
        GRAPH_HIDDEN_DIM=16,
        K_SPREAD_STEPS=2,
        SPREAD_PRICE_FEATURE_INDEX=None,
        SPREAD_PRICE_IS_DELTA=True,
    )


def _build_universe() -> AssetUniverse:
    return AssetUniverse.etf_pair("ETF", "CONST")


def _make_batch(cfg: SimpleNamespace, batch_size: int = 2):
    K_gen = cfg.HYPER_PARAMETERS[LHP.MASKED_SEQ_SIZE]
    K_cond = cfg.HYPER_PARAMETERS[LHP.SEQ_SIZE] - K_gen
    F = cfg.HYPER_PARAMETERS[LHP.SIZE_ORDER_EMB]
    N = 2
    x_0 = torch.zeros(batch_size, N, K_gen, F, device=cst.DEVICE)
    cond_orders = torch.randn(batch_size, N, K_cond, F, device=cst.DEVICE)
    cond_lob = torch.zeros(
        batch_size,
        N,
        K_cond + 1,
        cst.N_LOB_LEVELS * cst.LEN_LEVEL,
        device=cst.DEVICE,
    )
    cond_lob[:, 0, :, 0] = 100.02
    cond_lob[:, 0, :, 2] = 100.00
    cond_lob[:, 1, :, 0] = 99.52
    cond_lob[:, 1, :, 2] = 99.50
    return x_0, cond_orders, cond_lob


def check_spread_utils():
    universe = _build_universe()
    B, N, K, F = 2, 2, 3, cst.LEN_ORDER + 2
    x0_hat = torch.zeros(B, N, K, F, device=cst.DEVICE)
    x0_hat[:, 0, :, 5] = 0.10
    x0_hat[:, 1, :, 5] = -0.20
    cond_lob = torch.zeros(B, N, 4, cst.N_LOB_LEVELS * cst.LEN_LEVEL, device=cst.DEVICE)
    cond_lob[:, 0, -1, 0] = 10.2
    cond_lob[:, 0, -1, 2] = 10.0
    cond_lob[:, 1, -1, 0] = 8.2
    cond_lob[:, 1, -1, 2] = 8.0

    mid = decode_mid_price(x0_hat, cond_lob)
    expected_mid = torch.tensor([[10.2, 7.9], [10.2, 7.9]], device=cst.DEVICE)
    assert torch.allclose(mid, expected_mid)
    spread = compute_spread(mid, universe)
    assert spread.shape == (B, 1)
    assert torch.allclose(spread[:, 0], torch.full((B,), 2.3, device=cst.DEVICE))
    spread_assets = broadcast_spread_to_assets(spread, universe, num_assets=N)
    assert spread_assets.shape == (B, N)
    assert torch.allclose(spread_assets[:, 0], spread[:, 0])
    assert torch.allclose(spread_assets[:, 1], spread[:, 0])

    assert not bool(spread_inject_active(3, num_diffusionsteps=4, k_spread_steps=2))
    assert bool(spread_inject_active(1, num_diffusionsteps=4, k_spread_steps=2))
    print("  [ok] spread utilities")


def check_shared_score_net_spread_cond():
    cfg = _build_config()
    F = cfg.HYPER_PARAMETERS[LHP.SIZE_ORDER_EMB]
    K_gen = cfg.HYPER_PARAMETERS[LHP.MASKED_SEQ_SIZE]
    K_cond = cfg.HYPER_PARAMETERS[LHP.SEQ_SIZE] - K_gen
    net = SharedScoreNet(
        num_assets=2,
        input_size=F,
        cond_seq_len=K_cond,
        num_diffusionsteps=cfg.HYPER_PARAMETERS[LHP.NUM_DIFFUSIONSTEPS],
        depth=1,
        num_heads=1,
        gen_sequence_size=K_gen,
        cond_dropout_prob=0.0,
        is_augmented=False,
        dropout=0.0,
        cond_type="full",
        cond_method="concatenation",
    ).to(cst.DEVICE)
    net.eval()
    x_t = torch.randn(2, 2, K_gen, F, device=cst.DEVICE)
    cond = torch.randn(2, 2, K_cond, F, device=cst.DEVICE)
    cond_lob = torch.randn(2, 2, K_cond + 1, cst.N_LOB_LEVELS * cst.LEN_LEVEL, device=cst.DEVICE)
    t = torch.randint(0, cfg.HYPER_PARAMETERS[LHP.NUM_DIFFUSIONSTEPS], (2,), device=cst.DEVICE)
    asset_ids = torch.arange(2, device=cst.DEVICE)

    with torch.no_grad():
        base, _ = net(x_t, cond, t, cond_lob, asset_ids, spread_cond=None)
        zero, _ = net(x_t, cond, t, cond_lob, asset_ids, spread_cond=torch.zeros(2, 2, device=cst.DEVICE))
        shifted, _ = net(x_t, cond, t, cond_lob, asset_ids, spread_cond=torch.ones(2, 2, device=cst.DEVICE))
    assert torch.equal(base, zero), "zero spread_cond must be exact no-op"
    assert (base - shifted).abs().max().item() > 0.0, "nonzero spread_cond should change score output"
    print("  [ok] SharedScoreNet spread_cond")


def check_diffuser_spread_hook_and_ablation():
    cfg = _build_config()
    diffuser = MultiAssetGaussianDiffusion(cfg, _build_universe(), None).to(cst.DEVICE)
    x_0, cond_orders, cond_lob = _make_batch(cfg)
    weights = np.ones(cfg.HYPER_PARAMETERS[LHP.NUM_DIFFUSIONSTEPS], dtype=float)

    state = ReverseLoopState()
    t_high = torch.full((x_0.shape[0],), 3, device=cst.DEVICE, dtype=torch.long)
    assert diffuser.pre_fusion_hook(x_t=x_0, state=state, t=t_high, cond_lob=cond_lob) is None
    state.update_eps(torch.ones_like(x_0))
    t_low = torch.full((x_0.shape[0],), 1, device=cst.DEVICE, dtype=torch.long)
    spread_cond = diffuser.pre_fusion_hook(x_t=x_0, state=state, t=t_low, cond_lob=cond_lob)
    assert spread_cond is not None and spread_cond.shape == x_0.shape[:2]
    assert state.last_spread is not None and state.last_spread.shape == (x_0.shape[0], 1)

    diffuser.ablation_flags.disable_spread_cond = True
    _seed(7)
    out_disabled = diffuser.ddpm_sample(x_0, cond_orders, cond_lob, weights)
    diffuser.init_losses()

    diffuser.ablation_flags.disable_spread_cond = False
    diffuser.k_spread_steps = 0
    _seed(7)
    out_inactive = diffuser.ddpm_sample(x_0, cond_orders, cond_lob, weights)
    diffuser.init_losses()
    assert torch.equal(out_disabled, out_inactive), "disabled spread path must recover P1 exactly"

    diffuser.k_spread_steps = cfg.K_SPREAD_STEPS
    _seed(7)
    out_enabled = diffuser.ddpm_sample(x_0, cond_orders, cond_lob, weights)
    assert not torch.equal(out_disabled, out_enabled), "active spread conditioning should alter sampling"
    print("  [ok] diffuser spread hook + ablation")


def main() -> int:
    print(f"device: {cst.DEVICE}")
    print("[1] spread utilities")
    check_spread_utils()
    print("[2] SharedScoreNet spread_cond")
    check_shared_score_net_spread_cond()
    print("[3] diffuser hook and ablation")
    check_diffuser_spread_hook_and_ablation()
    print("\nP2 smoke test PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
