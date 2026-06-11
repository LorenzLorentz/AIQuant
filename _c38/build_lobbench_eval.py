"""Phase-2 lob_bench evaluation data builder (run in deep_market env, on GPU).

Generates lob_bench-format CSVs for the no-graph BTC+ETH MA_TRADES checkpoint:
  data_real/ : real BTC test continuation  (LOBSTER message + orderbook)
  data_cond/ : real BTC prefix (conditioning)
  data_gen/  : model-generated continuation (teacher-forced, real book context)

Generation = batched teacher-forced one-step: at each test position the model
denoises the next order conditioned on the real K_cond window (orders+book) of
BOTH assets; we keep the BTC asset's generated order. This matches how the model
denoises (full DDPM reverse) without a stateful ABIDES rollout. Generated orders
are paired with the real book at that step (book metrics share context; the
order-placement metrics -- type/size/depth/levels/inter-arrival -- carry the
signal). Decode mirrors ABIDES WorldAgent._postprocess_generated_TRADES but uses
this dataset's own per-column train statistics (not the INTC/TSLA constants).
"""
from __future__ import annotations
import os, sys
import numpy as np
import torch
from types import SimpleNamespace

import constants as cst
from constants import LearningHyperParameter as LHP
from configuration import Configuration
from run import HP_DICT_MODEL
from models.diffusers.multi_asset.ma_diffusion_engine import MultiAssetDiffusionEngine
from preprocessing.AssetUniverse import AssetUniverse

CKPT = os.environ.get("CKPT", "/DATA/DATANAS1/wpj24/AIQuant/DeepMarket/data/checkpoints/MA_TRADES/val_ema=0.918_epoch=4_MA_BTCUSDT_ETHUSDT_c38_nograph.ckpt")
RAW = "/DATA/DATANAS1/wpj24/AIQuant/DeepMarket/data/_adapter_raw"   # un-normalized adapter npy
DATAZ = "/DATA/DATANAS1/wpj24/AIQuant/DeepMarket/data"             # z-scored {asset}/test.npy
OUT = os.environ.get("OUT", "/DATA/DATANAS1/wpj24/AIQuant/DeepMarket/data/lobbench/DeepMarketMA/BTCUSDT/2024-01-01")
DATE = "2024-01-01"
ASSETS = ["BTCUSDT", "ETHUSDT"]
SPLIT = (.85, .05, .10)
CONT = [0, 2, 3, 5] + list(range(6, 46))
N_EVENTS = int(os.environ.get("N_EVENTS", "800"))   # generated/real continuation length
GEN_BATCH = 256

SEQ = 256
KGEN = 1
KCOND = SEQ - KGEN


def col_stats(asset):
    """Recompute the exact per-column train mean/std used by build_real_datasets."""
    arr = np.load(f"{RAW}/{asset}.npy").astype(np.float64)
    n = len(arr); a = int(n * SPLIT[0])
    tr = arr[:a]
    mean = np.zeros(46); std = np.ones(46)
    m = tr[:, CONT].mean(0); s = tr[:, CONT].std(0); s = np.where(s < 1e-8, 1.0, s)
    mean[CONT] = m; std[CONT] = s
    return mean, std


def decode_order(time_z, evtype_class, size_z, price_z, dir_raw, depth_z, mean, std):
    """z-scored order fields -> raw LOBSTER-ish [time, type{1,3,4}, size, price, dir]."""
    size = round(size_z * std[2] + mean[2])
    price = round(price_z * std[3] + mean[3])
    time = time_z * std[0] + mean[0]
    if time <= 0:
        time = 1e-7
    etype = {0: 1, 1: 3, 2: 4}[int(evtype_class)]
    direction = 1 if dir_raw >= 0 else -1
    return time, etype, max(size, 1), price, direction


def main():
    os.makedirs(f"{OUT}/data_real", exist_ok=True)
    os.makedirs(f"{OUT}/data_gen", exist_ok=True)
    os.makedirs(f"{OUT}/data_cond", exist_ok=True)

    stats = {a: col_stats(a) for a in ASSETS}
    # save BTC stats for reference / reuse
    np.savez(f"{OUT}/btc_stats.npz", mean=stats["BTCUSDT"][0], std=stats["BTCUSDT"][1])

    # ---- build engine + load checkpoint --------------------------------
    cfg = Configuration()
    cfg.CHOSEN_MODEL = cst.Models.MA_TRADES
    cfg.MULTI_ASSET = True
    cfg.IS_WANDB = False
    cfg.IS_TRAINING = False
    cfg.FILENAME_CKPT = "lobbench_eval"
    fixed = HP_DICT_MODEL[cfg.CHOSEN_MODEL].fixed
    for p in cst.LearningHyperParameter:
        if p.value in fixed:
            cfg.HYPER_PARAMETERS[p] = fixed[p.value]
    universe = AssetUniverse(assets=ASSETS, relation_types={(0, 1): 0, (1, 0): 1})
    engine = MultiAssetDiffusionEngine(cfg, universe).to(cst.DEVICE)
    sd = torch.load(CKPT, map_location=cst.DEVICE)
    state = sd.get("state_dict", sd)
    missing, unexpected = engine.load_state_dict(state, strict=False)
    print(f"[ckpt] loaded; missing={len(missing)} unexpected={len(unexpected)}")
    engine.eval()

    # ---- load z-scored test data (both assets), align ------------------
    z = {a: np.load(f"{DATAZ}/{a}/test.npy").astype(np.float32) for a in ASSETS}
    L = min(z[a].shape[0] for a in ASSETS)
    start = KCOND + 1
    n_ev = min(N_EVENTS, L - start - 1)
    print(f"[data] test len={L}, generating {n_ev} events")

    orders = {a: torch.from_numpy(z[a][:, :cst.LEN_ORDER]) for a in ASSETS}
    lob = {}
    for a in ASSETS:
        lo = z[a][:, cst.LEN_ORDER:].copy()
        lo = np.roll(lo, 1, axis=0); lo[0] = 0
        lob[a] = torch.from_numpy(lo)

    # ---- batched teacher-forced generation -----------------------------
    gen_orders_btc = []
    idxs = list(range(start, start + n_ev))
    with torch.no_grad():
        for c in range(0, len(idxs), GEN_BATCH):
            chunk = idxs[c:c + GEN_BATCH]
            cond_o, x0, cond_l = [], [], []
            for i in chunk:
                co = torch.stack([orders[a][i - KCOND:i] for a in ASSETS], 0)        # (N,Kc,6)
                xl = torch.stack([lob[a][i - KCOND:i + 1] for a in ASSETS], 0)         # (N,Kc+1,40)
                cond_o.append(co); cond_l.append(xl)
                x0.append(torch.zeros(len(ASSETS), KGEN, cst.LEN_ORDER))
            cond_o = torch.stack(cond_o).to(cst.DEVICE)
            cond_l = torch.stack(cond_l).to(cst.DEVICE)
            x0 = torch.stack(x0).to(cst.DEVICE)
            gen = engine.sample(cond_orders=cond_o, x=x0, cond_lob=cond_l)            # (B,N,KGEN,8)
            gen_btc = gen[:, 0, 0, :].float().cpu()                                   # BTC asset, first gen step
            for r in range(gen_btc.shape[0]):
                gen_orders_btc.append(gen_btc[r])
            print(f"  generated {min(c+GEN_BATCH, len(idxs))}/{len(idxs)}")

    # type embedder weights for decoding the 3-dim type embedding
    temb = engine.type_embedder.weight.data.float().cpu()
    ste = cfg.HYPER_PARAMETERS[LHP.SIZE_TYPE_EMB]
    meanB, stdB = stats["BTCUSDT"]

    # ---- write generated continuation (BTC) ----------------------------
    import csv
    BASE = f"BTCUSDT_{DATE}_34200000_57600000"

    def write_msgbook(subdir, idsuffix, rows_orders, rows_books):
        t_abs = 0.0
        msg = []
        for (time, etype, size, price, direction) in rows_orders:
            t_abs += max(time, 1e-7)
            msg.append([f"{t_abs:.9f}", etype, 0, size, price, direction])
        with open(f"{OUT}/{subdir}/{BASE}_message_{idsuffix}.csv", "w", newline="") as f:
            w = csv.writer(f)
            for m in msg:
                w.writerow(m)
        np.savetxt(f"{OUT}/{subdir}/{BASE}_orderbook_{idsuffix}.csv",
                   np.asarray(rows_books), delimiter=",", fmt="%d")

    # decode generated BTC orders; pair each with the real book at that step
    gen_rows, gen_books = [], []
    for k, i in enumerate(idxs):
        g = gen_orders_btc[k]
        etype_class = int(torch.argmin(torch.sum(torch.abs(temb - g[1:1 + ste]), dim=1)).item())
        time, etype, size, price, direction = decode_order(
            g[0].item(), etype_class, g[ste + 1].item(), g[ste + 2].item(),
            g[ste + 3].item(), g[-1].item(), meanB, stdB)
        gen_rows.append((time, etype, size, price, direction))
        gen_books.append(np.round(z["BTCUSDT"][i, cst.LEN_ORDER:] * stdB[6:] + meanB[6:]).astype(int))

    # decode REAL BTC continuation + book
    real_rows, real_books = [], []
    for i in idxs:
        o = z["BTCUSDT"][i, :cst.LEN_ORDER]
        time, etype, size, price, direction = decode_order(
            o[0], int(round(o[1])), o[2], o[3], o[4], o[5], meanB, stdB)
        real_rows.append((time, etype, size, price, direction))
        real_books.append(np.round(z["BTCUSDT"][i, cst.LEN_ORDER:] * stdB[6:] + meanB[6:]).astype(int))

    # cond prefix (real, the K_cond window just before the continuation)
    cond_rows, cond_books = [], []
    for i in range(start - KCOND, start):
        o = z["BTCUSDT"][i, :cst.LEN_ORDER]
        time, etype, size, price, direction = decode_order(
            o[0], int(round(o[1])), o[2], o[3], o[4], o[5], meanB, stdB)
        cond_rows.append((time, etype, size, price, direction))
        cond_books.append(np.round(z["BTCUSDT"][i, cst.LEN_ORDER:] * stdB[6:] + meanB[6:]).astype(int))

    write_msgbook("data_real", "real_id_0", real_rows, real_books)
    write_msgbook("data_gen", "real_id_0_gen_id_0", gen_rows, gen_books)
    write_msgbook("data_cond", "real_id_0", cond_rows, cond_books)
    print(f"GEN_DONE real={len(real_rows)} gen={len(gen_rows)} cond={len(cond_rows)} -> {OUT}")


if __name__ == "__main__":
    main()
