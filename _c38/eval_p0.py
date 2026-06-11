"""P0 head-to-head eval: teacher-forced next-order predictive error on the
REAL ETH test set, computed identically for a single-asset TRADES engine and a
shared MA_TRADES (no-graph) engine.

Goal (NEXT_WORK_ZH.md task 2.1): does a SHARED backbone help a data-poor asset?
We compare, on the same real ETH test windows, the next-order prediction quality
of independent TRADES-ETH vs shared MA(BTC+ETH), across data levels (full/p20/p10).

Metric (all teacher-forced, real history -> sample 1 order -> compare to truth):
  * etype_acc : event-type classification accuracy (argmin over type_embedder)
  * dir_acc   : trade-direction sign accuracy
  * L1 on z-scored continuous fields time / size / price (lower = better)
  * cont_L1   : mean of the three above (headline scalar)
The order vector is z-scored in training, so the sampled fields are already in
z-units and directly comparable to the true z-fields -- no denorm needed.

Env: CKPT, MODEL(TRADES|MA_TRADES), ASSET(=ETHUSDT eval target), PEER(=BTCUSDT,
     only for MA), N_POS, SAMPLING(DDIM), DDIM_NSTEPS, SEED, OUT(json path).
"""
from __future__ import annotations
import os, json, math
import numpy as np
import torch

import constants as cst
from constants import LearningHyperParameter as LHP
from configuration import Configuration
from run import HP_DICT_MODEL
from models.diffusers.diffusion_engine import DiffusionEngine
from models.diffusers.multi_asset.ma_diffusion_engine import MultiAssetDiffusionEngine
from preprocessing.AssetUniverse import AssetUniverse

ROOT = "/DATA/DATANAS1/wpj24/AIQuant/DeepMarket"
DATAZ = f"{ROOT}/data"
MODEL = os.environ.get("MODEL", "TRADES")
ASSET = os.environ.get("ASSET", "ETHUSDT")          # eval target (real test data)
PEER = os.environ.get("PEER", "BTCUSDT")            # MA peer asset
CKPT = os.environ["CKPT"]
N_POS = int(os.environ.get("N_POS", "3000"))
BATCH = int(os.environ.get("BATCH", "512"))
SAMPLING = os.environ.get("SAMPLING", "DDIM")
SEED = int(os.environ.get("SEED", "0"))
OUT = os.environ.get("OUT", "")

SEQ = 256
KGEN = 1
KCOND = SEQ - KGEN
STE = 3
LEN_ORDER = cst.LEN_ORDER  # 6


def build_engine():
    cfg = Configuration()
    cfg.IS_WANDB = False
    cfg.IS_TRAINING = False
    cfg.SAMPLING_TYPE = SAMPLING
    if os.environ.get("DDIM_NSTEPS"):
        cfg.HYPER_PARAMETERS[LHP.DDIM_NSTEPS] = int(os.environ["DDIM_NSTEPS"])
    if MODEL == "MA_TRADES":
        cfg.CHOSEN_MODEL = cst.Models.MA_TRADES
        cfg.MULTI_ASSET = True
        fixed = HP_DICT_MODEL[cfg.CHOSEN_MODEL].fixed
        for p in cst.LearningHyperParameter:
            if p.value in fixed:
                cfg.HYPER_PARAMETERS[p] = fixed[p.value]
        cfg.FILENAME_CKPT = "eval_p0"
        assets = [PEER, ASSET]
        universe = AssetUniverse(assets=assets, relation_types={(0, 1): 0, (1, 0): 1})
        eng = MultiAssetDiffusionEngine(cfg, universe).to(cst.DEVICE)
    else:
        cfg.CHOSEN_MODEL = cst.Models.TRADES
        cfg.MULTI_ASSET = False
        fixed = HP_DICT_MODEL[cfg.CHOSEN_MODEL].fixed
        for p in cst.LearningHyperParameter:
            if p.value in fixed:
                cfg.HYPER_PARAMETERS[p] = fixed[p.value]
        cfg.FILENAME_CKPT = "eval_p0"
        eng = DiffusionEngine(cfg).to(cst.DEVICE)
    sd = torch.load(CKPT, map_location=cst.DEVICE)
    state = sd.get("state_dict", sd)
    missing, unexpected = eng.load_state_dict(state, strict=False)
    print(f"[ckpt] {os.path.basename(CKPT)} missing={len(missing)} unexpected={len(unexpected)}")
    eng.eval()
    return eng


def main():
    if SEED:
        torch.manual_seed(SEED); np.random.seed(SEED)
    eng = build_engine()
    temb = eng.type_embedder.weight.data.float().cpu().numpy()   # (n_types, STE)

    zt = {ASSET: np.load(f"{DATAZ}/{ASSET}/test.npy").astype(np.float32)}
    if MODEL == "MA_TRADES":
        zt[PEER] = np.load(f"{DATAZ}/{PEER}/test.npy").astype(np.float32)
    L = min(v.shape[0] for v in zt.values())
    first = KCOND + 1
    last = L - 2
    rng = np.random.default_rng(SEED or 0)
    positions = np.sort(rng.choice(np.arange(first, last), size=min(N_POS, last - first), replace=False))

    et_correct = dir_correct = 0
    l1t = l1s = l1p = 0.0
    n = 0
    with torch.no_grad():
        for b0 in range(0, len(positions), BATCH):
            pos = positions[b0:b0 + BATCH]
            B = len(pos)
            if MODEL == "MA_TRADES":
                assets = [PEER, ASSET]
                co = np.empty((B, 2, KCOND, LEN_ORDER), np.float32)
                cl = np.empty((B, 2, KCOND + 1, 40), np.float32)
                for ai, a in enumerate(assets):
                    for j, p in enumerate(pos):
                        co[j, ai] = zt[a][p - KCOND:p, :LEN_ORDER]
                        cl[j, ai] = zt[a][p - KCOND - 1:p, LEN_ORDER:]
                x0 = torch.zeros(B, 2, KGEN, LEN_ORDER)
                tgt = 1  # ASSET index
                gen = eng.sample(cond_orders=torch.from_numpy(co).to(cst.DEVICE),
                                 x=x0.to(cst.DEVICE),
                                 cond_lob=torch.from_numpy(cl).to(cst.DEVICE))
                g = gen[:, tgt, 0, :].float().cpu().numpy()           # (B,8)
            else:
                co = np.empty((B, KCOND, LEN_ORDER), np.float32)
                cl = np.empty((B, KCOND + 1, 40), np.float32)
                for j, p in enumerate(pos):
                    co[j] = zt[ASSET][p - KCOND:p, :LEN_ORDER]
                    cl[j] = zt[ASSET][p - KCOND - 1:p, LEN_ORDER:]
                x0 = torch.zeros(B, KGEN, LEN_ORDER)
                gen = eng.sample(cond_orders=torch.from_numpy(co).to(cst.DEVICE),
                                 x=x0.to(cst.DEVICE),
                                 cond_lob=torch.from_numpy(cl).to(cst.DEVICE))
                g = gen[:, 0, :].float().cpu().numpy()                # (B,8)

            true = zt[ASSET][pos, :LEN_ORDER]                        # (B,6) [t,etype,size,price,dir,depth]
            # predicted fields
            pred_t = g[:, 0]; pred_size = g[:, STE + 1]; pred_price = g[:, STE + 2]; pred_dir = g[:, STE + 3]
            pred_et = np.argmin(np.sum(np.abs(temb[None, :, :] - g[:, 1:1 + STE][:, None, :]), axis=2), axis=1)
            et_correct += int(np.sum(pred_et == np.round(true[:, 1]).astype(int)))
            dir_correct += int(np.sum((pred_dir >= 0) == (true[:, 4] >= 0)))
            l1t += float(np.sum(np.abs(pred_t - true[:, 0])))
            l1s += float(np.sum(np.abs(pred_size - true[:, 2])))
            l1p += float(np.sum(np.abs(pred_price - true[:, 3])))
            n += B
            print(f"  {n}/{len(positions)}")

    res = {
        "ckpt": os.path.basename(CKPT), "model": MODEL, "asset": ASSET, "n_pos": n,
        "etype_acc": et_correct / n, "dir_acc": dir_correct / n,
        "l1_time_z": l1t / n, "l1_size_z": l1s / n, "l1_price_z": l1p / n,
        "cont_l1_z": (l1t + l1s + l1p) / (3 * n),
    }
    print("RESULT " + json.dumps(res))
    if OUT:
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        json.dump(res, open(OUT, "w"), indent=2)
        print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
