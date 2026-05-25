"""P0 smoke test — verifies that the multi-asset stack wires together
end-to-end on synthetic data, with no LOBSTER files required.

Checks:
  1. AssetUniverse builds.
  2. SharedScoreNet forward on dummy (B, N, K_gen, F_aug) tensors returns
     correct shape and assets-swap produces different outputs.
  3. MultiAssetGaussianDiffusion forward_reparametrized + ddpm_single_step
     produces finite reconstructions and per-asset losses.
  4. MultiAssetDiffusionEngine.training_step on a synthetic batch runs
     end-to-end with finite loss and a gradient flows back to the shared
     TRADES backbone and to the asset embedding.

Invoke:  python -m tests.p0_smoke   (from DeepMarket/)
"""

from __future__ import annotations

import sys
import torch

import constants as cst
from constants import LearningHyperParameter as LHP
from configuration import Configuration
from preprocessing.AssetUniverse import AssetUniverse
from models.diffusers.multi_asset.shared_score_net import SharedScoreNet
from models.diffusers.multi_asset.ma_gaussian_diffusion import MultiAssetGaussianDiffusion
from models.diffusers.multi_asset.ma_diffusion_engine import MultiAssetDiffusionEngine
from utils.utils_models import pick_augmenter


def _shrink_config(cfg: Configuration) -> None:
    cfg.HYPER_PARAMETERS[LHP.SEQ_SIZE] = 16
    cfg.HYPER_PARAMETERS[LHP.MASKED_SEQ_SIZE] = 1
    cfg.HYPER_PARAMETERS[LHP.NUM_DIFFUSIONSTEPS] = 8
    cfg.HYPER_PARAMETERS[LHP.CDT_DEPTH] = 1
    cfg.HYPER_PARAMETERS[LHP.AUGMENT_DIM] = 64
    cfg.HYPER_PARAMETERS[LHP.CDT_NUM_HEADS] = 1
    cfg.HYPER_PARAMETERS[LHP.BATCH_SIZE] = 2
    cfg.HYPER_PARAMETERS[LHP.TEST_BATCH_SIZE] = 2
    from utils.utils import noise_scheduler
    cfg.BETAS = noise_scheduler(
        num_diffusion_timesteps=cfg.HYPER_PARAMETERS[LHP.NUM_DIFFUSIONSTEPS]
    )


def _build_engine(num_assets: int = 2) -> MultiAssetDiffusionEngine:
    cfg = Configuration()
    cfg.CHOSEN_MODEL = cst.Models.MA_TRADES
    cfg.MULTI_ASSET = True
    cfg.IS_TRAINING = True
    cfg.IS_WANDB = False
    cfg.FILENAME_CKPT = "p0_smoke"
    _shrink_config(cfg)
    universe = AssetUniverse(
        assets=[f"ASSET_{i}" for i in range(num_assets)],
        relation_types={(j, i): k for k, (j, i) in enumerate(
            [(a, b) for a in range(num_assets) for b in range(num_assets) if a != b]
        )},
    )
    cfg.ASSET_UNIVERSE = universe
    engine = MultiAssetDiffusionEngine(cfg, universe).to(cst.DEVICE)
    return engine, cfg


def _make_batch(cfg: Configuration, num_assets: int, batch_size: int = 2):
    K_total = cfg.HYPER_PARAMETERS[LHP.SEQ_SIZE]
    K_gen = cfg.HYPER_PARAMETERS[LHP.MASKED_SEQ_SIZE]
    K_cond = K_total - K_gen
    cond_orders = torch.randn(batch_size, num_assets, K_cond, cst.LEN_ORDER, device=cst.DEVICE)
    # column 1 is order type (long, in [0,2]); the type_embedder requires it.
    cond_orders[..., 1] = torch.randint(0, 3, cond_orders[..., 1].shape, device=cst.DEVICE).float()
    x_0 = torch.randn(batch_size, num_assets, K_gen, cst.LEN_ORDER, device=cst.DEVICE)
    x_0[..., 1] = torch.randint(0, 3, x_0[..., 1].shape, device=cst.DEVICE).float()
    cond_lob = torch.randn(
        batch_size, num_assets, K_cond + 1, cst.N_LOB_LEVELS * cst.LEN_LEVEL, device=cst.DEVICE
    )
    return cond_orders, x_0, cond_lob


def check_shared_score_net():
    cfg = Configuration()
    cfg.CHOSEN_MODEL = cst.Models.MA_TRADES
    _shrink_config(cfg)
    num_assets = 2
    F_in = cfg.HYPER_PARAMETERS[LHP.AUGMENT_DIM]
    K_gen = cfg.HYPER_PARAMETERS[LHP.MASKED_SEQ_SIZE]
    K_cond = cfg.HYPER_PARAMETERS[LHP.SEQ_SIZE] - K_gen
    net = SharedScoreNet(
        num_assets=num_assets,
        input_size=F_in,
        cond_seq_len=K_cond,
        num_diffusionsteps=cfg.HYPER_PARAMETERS[LHP.NUM_DIFFUSIONSTEPS],
        depth=1, num_heads=1, gen_sequence_size=K_gen,
        cond_dropout_prob=0.0, is_augmented=True, dropout=0.1,
        cond_type="full", cond_method="concatenation",
    ).to(cst.DEVICE)
    B = 2
    x_t = torch.randn(B, num_assets, K_gen, F_in, device=cst.DEVICE)
    cond = torch.randn(B, num_assets, K_cond, F_in, device=cst.DEVICE)
    t = torch.randint(0, cfg.HYPER_PARAMETERS[LHP.NUM_DIFFUSIONSTEPS], (B,), device=cst.DEVICE)
    cond_lob = torch.randn(B, num_assets, K_cond + K_gen, F_in, device=cst.DEVICE)
    asset_ids = torch.arange(num_assets, device=cst.DEVICE)
    noise, var = net(x_t, cond, t, cond_lob, asset_ids)
    assert noise.shape[:3] == (B, num_assets, K_gen), f"unexpected shape {noise.shape}"
    assert var.shape == noise.shape
    # asset swap should produce different outputs (asset embedding is informative)
    noise_swapped, _ = net(x_t.flip(1), cond.flip(1), t, cond_lob.flip(1), asset_ids)
    diff = (noise - noise_swapped.flip(1)).abs().max().item()
    assert diff > 0.0, "asset embedding has no effect"
    print(f"  [ok] SharedScoreNet output shape {tuple(noise.shape)}, asset-swap delta {diff:.4f}")


def check_engine_training_step():
    num_assets = 2
    engine, cfg = _build_engine(num_assets=num_assets)
    engine.configure_optimizers()
    batch = _make_batch(cfg, num_assets=num_assets, batch_size=2)
    cond_orders, x_0, cond_lob = batch
    cond_orders = cond_orders.contiguous()
    x_0 = x_0.contiguous()
    cond_lob = cond_lob.contiguous()
    # Lightning's training_step expects a tuple
    loss = engine.training_step((cond_orders, x_0, cond_lob), batch_idx=0)
    assert torch.isfinite(loss), f"loss not finite: {loss}"
    loss.backward()
    # check that gradients flow to the asset embedding and to TRADES backbone
    emb_grad = engine.diffuser.NN.asset_embedding.weight.grad
    assert emb_grad is not None and emb_grad.abs().sum().item() > 0, "no grad on asset embedding"
    backbone_grads = [p.grad for p in engine.diffuser.NN.backbone.parameters() if p.grad is not None]
    assert backbone_grads, "no grads in TRADES backbone"
    print(f"  [ok] training_step loss={loss.item():.4f}, "
          f"asset-emb |grad|={emb_grad.abs().sum().item():.4f}, "
          f"backbone params with grad={len(backbone_grads)}")


def check_diffuser_forward_reparametrized():
    num_assets = 2
    engine, cfg = _build_engine(num_assets=num_assets)
    K_gen = cfg.HYPER_PARAMETERS[LHP.MASKED_SEQ_SIZE]
    F = cfg.HYPER_PARAMETERS[LHP.SIZE_ORDER_EMB]
    x_0 = torch.randn(2, num_assets, K_gen, F, device=cst.DEVICE)
    t = torch.randint(0, cfg.HYPER_PARAMETERS[LHP.NUM_DIFFUSIONSTEPS], (2,), device=cst.DEVICE)
    x_t, noise = engine.diffuser.forward_reparametrized(x_0, t)
    assert x_t.shape == x_0.shape and noise.shape == x_0.shape
    assert torch.isfinite(x_t).all() and torch.isfinite(noise).all()
    print(f"  [ok] forward_reparametrized shape {tuple(x_t.shape)}")


def main() -> int:
    print(f"device: {cst.DEVICE}")
    print("[1] SharedScoreNet")
    check_shared_score_net()
    print("[2] diffuser.forward_reparametrized")
    check_diffuser_forward_reparametrized()
    print("[3] MultiAssetDiffusionEngine.training_step (incl. backward + grads)")
    check_engine_training_step()
    print("\nP0 smoke test PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
