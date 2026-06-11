"""One-batch diagnostic: is the trunk corrupted, or is the fusion forward the
culprit? Loads a diverged MA_TRADES checkpoint and runs the SAME seeded batch
through engine.forward twice -- once with disable_graph=False (actual, uses
fuse) and once with disable_graph=True (fuse returns eps_local exactly) --
then compares the resulting loss. Also reports gamma and ||eps_fused-eps_local||.

  loss(disable=True)  = ||eps_local - noise||  -> pure trunk quality
  loss(disable=False) = ||eps_fused - noise||  -> actual

If both ~equal+high  -> trunk is corrupted (gate/fusion irrelevant).
If True low, False high -> fusion forward injects the error (delta large).
"""
from __future__ import annotations
import os
import numpy as np
import torch

import constants as cst
from constants import LearningHyperParameter as LHP
from configuration import Configuration
from run import HP_DICT_MODEL
from models.diffusers.multi_asset.ma_diffusion_engine import MultiAssetDiffusionEngine
from preprocessing.AssetUniverse import AssetUniverse

CKPT = os.environ["CKPT"]
DATAZ = "/DATA/DATANAS1/wpj24/AIQuant/DeepMarket/data"
ASSETS = ["BTCUSDT", "ETHUSDT"]
B = 64
SEED = 0


def build_engine(disable_graph):
    cfg = Configuration()
    cfg.CHOSEN_MODEL = cst.Models.MA_TRADES
    cfg.MULTI_ASSET = True
    cfg.IS_WANDB = False
    cfg.IS_TRAINING = True
    cfg.IS_AUGMENTATION = True
    cfg.DISABLE_GRAPH = disable_graph
    cfg.DISABLE_ARB_GUIDANCE = True
    cfg.FILENAME_CKPT = "diag"
    fixed = HP_DICT_MODEL[cfg.CHOSEN_MODEL].fixed
    for p in cst.LearningHyperParameter:
        if p.value in fixed:
            cfg.HYPER_PARAMETERS[p] = fixed[p.value]
    universe = AssetUniverse(assets=ASSETS, relation_types={(0, 1): 0, (1, 0): 1})
    eng = MultiAssetDiffusionEngine(cfg, universe).to(cst.DEVICE)
    sd = torch.load(CKPT, map_location=cst.DEVICE)
    eng.load_state_dict(sd.get("state_dict", sd), strict=False)
    eng.eval()
    return eng, cfg


def make_batch(cfg):
    seq = cfg.HYPER_PARAMETERS[LHP.SEQ_SIZE]
    kgen = cfg.HYPER_PARAMETERS[LHP.MASKED_SEQ_SIZE]
    kcond = seq - kgen
    z = {a: np.load(f"{DATAZ}/{a}/train.npy").astype(np.float32) for a in ASSETS}
    L = min(z[a].shape[0] for a in ASSETS)
    rng = np.random.default_rng(SEED)
    idxs = rng.integers(kcond, L - kgen - 1, size=B)
    lob = {}
    for a in ASSETS:
        lo = z[a][:, cst.LEN_ORDER:].copy()
        lo = np.roll(lo, 1, axis=0); lo[0] = 0
        lob[a] = lo
    cond_o, x0, cond_l = [], [], []
    for i in idxs:
        co = np.stack([z[a][i - kcond:i, :cst.LEN_ORDER] for a in ASSETS])      # (N,kcond,6)
        xx = np.stack([z[a][i:i + kgen, :cst.LEN_ORDER] for a in ASSETS])        # (N,kgen,6)
        cl = np.stack([lob[a][i - kcond:i + kgen] for a in ASSETS])              # (N,seq,40)
        cond_o.append(co); x0.append(xx); cond_l.append(cl)
    to = lambda arr: torch.from_numpy(np.stack(arr)).float().to(cst.DEVICE)
    return to(cond_o), to(x0), to(cond_l), (kcond, kgen)


def run_variant(disable_graph, batch):
    cond_o, x0, cond_l, _ = batch
    eng, cfg = build_engine(disable_graph)
    g = eng.diffuser.graph_gamma.item()

    # capture ||eps_local|| and ||eps_fused|| inside fuse (graph path only)
    norms = {}
    orig_fuse = eng.diffuser.fuse
    def fuse_spy(eps_local, **ctx):
        out = orig_fuse(eps_local, **ctx)
        norms["eps_local"] = float(eps_local.norm().item())
        norms["eps_fused"] = float(out.norm().item())
        norms["delta"] = float((out - eps_local).norm().item())
        return out
    eng.diffuser.fuse = fuse_spy

    torch.manual_seed(SEED); np.random.seed(SEED)
    eng.diffuser.init_losses()
    with torch.no_grad():
        _ = eng.forward(cond_o.clone(), x0.clone(),
                        cond_l.clone() if cfg.COND_TYPE == "full" else None,
                        is_train=False)
        L_hybrid, L_simple, L_vlb = eng.loss()
    return g, float(L_hybrid.mean()), float(L_simple.mean()), float(L_vlb.mean()), norms


def main():
    print(f"[ckpt] {CKPT}")
    base = build_engine(False)[1]
    batch = make_batch(base)
    print(f"[batch] B={B} kcond={batch[3][0]} kgen={batch[3][1]} "
          f"cond_o={tuple(batch[0].shape)} x0={tuple(batch[1].shape)} cond_l={tuple(batch[2].shape)}")

    g, lh_on, ls_on, lv_on, norms = run_variant(False, batch)
    print(f"\n[graph ON ]  gamma={g:.6f}  L_hybrid={lh_on:.4f}  L_simple={ls_on:.4f}  L_vlb={lv_on:.4f}")
    print(f"             ||eps_local||={norms.get('eps_local'):.3f}  "
          f"||eps_fused||={norms.get('eps_fused'):.3f}  ||delta=eps_fused-eps_local||={norms.get('delta'):.4f}")

    _, lh_off, ls_off, lv_off, _ = run_variant(True, batch)
    print(f"[graph OFF]  (disable_graph=True)  L_hybrid={lh_off:.4f}  L_simple={ls_off:.4f}  L_vlb={lv_off:.4f}")

    print("\n==== VERDICT ====")
    print(f"  L_simple  graph_ON={ls_on:.4f}   graph_OFF(trunk only)={ls_off:.4f}")
    if ls_off > 4.0:
        print("  -> TRUNK IS CORRUPTED: pure eps_local loss is high even with fuse disabled.")
        print("     The gate/fusion is NOT the cause; the shared backbone weights are bad.")
    elif ls_on - ls_off > 1.0:
        print("  -> FUSION FORWARD is the culprit: trunk eps_local is fine, fuse() inflates the loss.")
    else:
        print("  -> trunk fine AND fusion adds little: divergence likely optimization/variance, not this batch.")


if __name__ == "__main__":
    main()
