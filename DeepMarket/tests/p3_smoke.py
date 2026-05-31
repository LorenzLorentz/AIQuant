"""P3 smoke test for annealed arbitrage energy guidance.

Invoke from ``DeepMarket/``:

    python -m tests.p3_smoke
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

import constants as cst
from constants import LearningHyperParameter as LHP
from tests.p2_smoke import _build_config, _build_universe, _make_batch, _seed
from models.diffusers.multi_asset.arbitrage import (
    ReverseLoopState,
    energy,
    guidance_lambda,
    rho,
    update_tau,
)
from models.diffusers.multi_asset.ma_gaussian_diffusion import MultiAssetGaussianDiffusion


def check_persistence_tracker():
    tau = {}
    delta = torch.tensor([[1.0]], device=cst.DEVICE)
    expected = [0, 1, 2, 0, 1, 0, 1]
    spreads = [0.2, 1.2, 1.4, 0.8, -1.3, 1.6, 1.7]
    active = [True, True, True, True, True, False, True]
    observed = []
    for value, is_active in zip(spreads, active):
        spread = torch.tensor([[value]], device=cst.DEVICE)
        counters = update_tau(tau, spread, delta, active=is_active)
        observed.append(int(counters[0, 0].item()))
    assert observed == expected, f"unexpected tau path {observed}"
    print("  [ok] persistence tracker")


def check_energy_gradient_and_schedule():
    spread = torch.tensor([[0.5, 1.5, -1.5]], device=cst.DEVICE, requires_grad=True)
    delta = torch.ones_like(spread)
    tau = torch.ones_like(spread)
    E = energy(spread, delta, tau)
    E.backward()
    assert torch.equal(rho(torch.tensor([0.5], device=cst.DEVICE), torch.tensor([1.0], device=cst.DEVICE)),
                       torch.zeros(1, device=cst.DEVICE))
    assert spread.grad[0, 0].item() == 0.0
    assert spread.grad[0, 1].item() > 0.0
    assert spread.grad[0, 2].item() < 0.0

    alpha_bar = torch.tensor([0.1, 0.5, 0.9], device=cst.DEVICE)
    lam = guidance_lambda(None, alpha_bar, lambda_max=1.0, p=2.0)
    assert lam[0] > lam[1] > lam[2]
    print("  [ok] energy gradient + guidance schedule")


def check_post_fusion_guidance_and_ablation():
    cfg = _build_config(disable_spread_cond=False)
    cfg.DISABLE_ARB_GUIDANCE = False
    cfg.ARB_DELTA_BASE = 0.0
    cfg.ARB_GUIDANCE_LAMBDA_MAX = 1000.0
    cfg.ARB_GUIDANCE_POWER = 2.0
    diffuser = MultiAssetGaussianDiffusion(cfg, _build_universe(), None).to(cst.DEVICE)
    x_0, cond_orders, cond_lob = _make_batch(cfg)
    eps_fused = torch.zeros_like(x_0)
    t_low = torch.full((x_0.shape[0],), 1, device=cst.DEVICE, dtype=torch.long)

    state = ReverseLoopState()
    guided = diffuser.post_fusion_hook(
        eps_fused,
        x_t=x_0,
        t=t_low,
        state=state,
        raw_cond_orders=cond_orders,
        raw_cond_lob=cond_lob,
    )
    assert guided.shape == eps_fused.shape
    assert (guided - eps_fused).abs().max().item() > 0.0
    assert state.last_spread is not None
    assert 0 in state.tau and int(state.tau[0].max().item()) == 1

    diffuser.ablation_flags.disable_arb_guidance = True
    state_disabled = ReverseLoopState()
    disabled = diffuser.post_fusion_hook(
        eps_fused,
        x_t=x_0,
        t=t_low,
        state=state_disabled,
        raw_cond_orders=cond_orders,
        raw_cond_lob=cond_lob,
    )
    assert torch.equal(disabled, eps_fused), "disable_arb_guidance must recover P2 exactly"
    assert state_disabled.last_spread is None and state_disabled.tau == {}

    old_state_dict = diffuser.state_dict()
    for key in [
        "arb_delta_base",
        "arb_stress_head.linear.weight",
        "arb_stress_head.linear.bias",
    ]:
        old_state_dict.pop(key)
    fresh = MultiAssetGaussianDiffusion(cfg, _build_universe(), None).to(cst.DEVICE)
    fresh.load_state_dict(old_state_dict, strict=True)
    print("  [ok] post_fusion_hook guidance + ablation")


def check_ablation_grid_runs():
    corners = [
        ("P0", True, True, True),
        ("P1", False, True, True),
        ("P2", False, False, True),
        ("P3", False, False, False),
    ]
    for name, disable_graph, disable_spread_cond, disable_arb_guidance in corners:
        cfg = _build_config(disable_spread_cond=disable_spread_cond)
        cfg.DISABLE_GRAPH = disable_graph
        cfg.DISABLE_ARB_GUIDANCE = disable_arb_guidance
        cfg.ARB_DELTA_BASE = 0.0
        cfg.ARB_GUIDANCE_LAMBDA_MAX = 1000.0
        diffuser = MultiAssetGaussianDiffusion(cfg, _build_universe(), None).to(cst.DEVICE)
        x_0, cond_orders, cond_lob = _make_batch(cfg)
        weights = np.ones(cfg.HYPER_PARAMETERS[LHP.NUM_DIFFUSIONSTEPS], dtype=float)
        _seed(31)
        out = diffuser.ddpm_sample(x_0, cond_orders, cond_lob, weights)
        assert out.shape == x_0.shape
        assert torch.isfinite(out).all(), f"{name} ablation corner produced non-finite samples"
        diffuser.init_losses()
    print("  [ok] four-corner ablation grid")


def main() -> int:
    print(f"device: {cst.DEVICE}")
    print("[1] persistence tracker")
    check_persistence_tracker()
    print("[2] energy and schedule")
    check_energy_gradient_and_schedule()
    print("[3] post-fusion guidance")
    check_post_fusion_guidance_and_ablation()
    print("[4] ablation grid")
    check_ablation_grid_runs()
    print("\nP3 smoke test PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
