"""Build lob_bench comparison set under a unified protocol (deep_market env, GPU).

One shared real continuation (real orders + real book) and prefix (cond). Four
"generated" variants, each = a set of continuation orders paired with the SAME
real book, so book-only metrics are identical for all (excluded) and the
order-placement / timing metrics are directly comparable:

  floor   : a DISJOINT real segment's orders   (sampling lower bound)
  naive   : per-column shuffle of real orders   (marginals kept, joint broken -> ceiling)
  nograph : MA_TRADES no-graph checkpoint        (model)
  graph   : MA_TRADES graph(noarb) checkpoint     (model)

Output: data/lobbench/cmp/{floor,naive,nograph,graph}/{data_real,data_gen,data_cond}/
"""
from __future__ import annotations
import os, csv, shutil
import numpy as np
import torch

import constants as cst
from constants import LearningHyperParameter as LHP
from configuration import Configuration
from run import HP_DICT_MODEL
from models.diffusers.multi_asset.ma_diffusion_engine import MultiAssetDiffusionEngine
from preprocessing.AssetUniverse import AssetUniverse

ROOT = "/DATA/DATANAS1/wpj24/AIQuant/DeepMarket"
RAW = ROOT + "/data/_adapter_raw"
DATAZ = ROOT + "/data"
CMP = ROOT + "/data/lobbench/cmp"
CKPTS = {
    "nograph": (ROOT + "/data/checkpoints/MA_TRADES/val_ema=0.918_epoch=4_MA_BTCUSDT_ETHUSDT_c38_nograph.ckpt", True),
    "graph":   (ROOT + "/data/checkpoints/MA_TRADES/val_ema=5.976_epoch=0_MA_BTCUSDT_ETHUSDT_c38_graph_noarb.ckpt", False),
}
DATE = "2024-01-01"
ASSETS = ["BTCUSDT", "ETHUSDT"]
SPLIT = (.85, .05, .10)
CONT = [0, 2, 3, 5] + list(range(6, 46))
N = int(os.environ.get("N_EVENTS", "800"))
GEN_BATCH = 256
SEQ, KGEN = 256, 1
KCOND = SEQ - KGEN
BASE = f"BTCUSDT_{DATE}_34200000_57600000"


def col_stats(asset):
    arr = np.load(f"{RAW}/{asset}.npy").astype(np.float64)
    a = int(len(arr) * SPLIT[0]); tr = arr[:a]
    mean = np.zeros(46); std = np.ones(46)
    mean[CONT] = tr[:, CONT].mean(0)
    s = tr[:, CONT].std(0); std[CONT] = np.where(s < 1e-8, 1.0, s)
    return mean, std


def decode(time_z, evclass, size_z, price_z, dir_raw, mean, std):
    size = max(int(round(size_z * std[2] + mean[2])), 1)
    price = int(round(price_z * std[3] + mean[3]))
    t = time_z * std[0] + mean[0]
    if t <= 0: t = 1e-7
    etype = {0: 1, 1: 3, 2: 4}[int(evclass)]
    return t, etype, size, price, (1 if dir_raw >= 0 else -1)


def real_rows_books(zbtc, idxs, mean, std):
    rows, books = [], []
    for i in idxs:
        o = zbtc[i, :cst.LEN_ORDER]
        rows.append(decode(o[0], int(round(o[1])), o[2], o[3], o[4], mean, std))
        books.append(np.round(zbtc[i, cst.LEN_ORDER:] * std[6:] + mean[6:]).astype(int))
    return rows, books


def gen_model(ckpt_path, disable_graph, zdata, idxs, mean, std):
    cfg = Configuration()
    cfg.CHOSEN_MODEL = cst.Models.MA_TRADES
    cfg.MULTI_ASSET = True; cfg.IS_WANDB = False; cfg.IS_TRAINING = False
    cfg.FILENAME_CKPT = "cmp"; cfg.DISABLE_GRAPH = disable_graph
    fixed = HP_DICT_MODEL[cfg.CHOSEN_MODEL].fixed
    for p in cst.LearningHyperParameter:
        if p.value in fixed: cfg.HYPER_PARAMETERS[p] = fixed[p.value]
    universe = AssetUniverse(assets=ASSETS, relation_types={(0, 1): 0, (1, 0): 1})
    engine = MultiAssetDiffusionEngine(cfg, universe).to(cst.DEVICE)
    sd = torch.load(ckpt_path, map_location=cst.DEVICE)
    engine.load_state_dict(sd.get("state_dict", sd), strict=False)
    engine.eval()
    ste = cfg.HYPER_PARAMETERS[LHP.SIZE_TYPE_EMB]
    temb = engine.type_embedder.weight.data.float().cpu()
    orders = {a: torch.from_numpy(zdata[a][:, :cst.LEN_ORDER]) for a in ASSETS}
    lob = {}
    for a in ASSETS:
        lo = zdata[a][:, cst.LEN_ORDER:].copy(); lo = np.roll(lo, 1, 0); lo[0] = 0
        lob[a] = torch.from_numpy(lo)
    out = []
    with torch.no_grad():
        for c in range(0, len(idxs), GEN_BATCH):
            ch = idxs[c:c + GEN_BATCH]
            co = torch.stack([torch.stack([orders[a][i - KCOND:i] for a in ASSETS]) for i in ch]).to(cst.DEVICE)
            cl = torch.stack([torch.stack([lob[a][i - KCOND:i + 1] for a in ASSETS]) for i in ch]).to(cst.DEVICE)
            x0 = torch.zeros(len(ch), len(ASSETS), KGEN, cst.LEN_ORDER, device=cst.DEVICE)
            g = engine.sample(cond_orders=co, x=x0, cond_lob=cl)[:, 0, 0, :].float().cpu()
            for r in range(g.shape[0]):
                v = g[r]
                ev = int(torch.argmin(torch.sum(torch.abs(temb - v[1:1 + ste]), 1)).item())
                out.append(decode(v[0].item(), ev, v[ste + 1].item(), v[ste + 2].item(), v[ste + 3].item(), mean, std))
            print(f"    {min(c+GEN_BATCH,len(idxs))}/{len(idxs)}")
    del engine; torch.cuda.empty_cache()
    return out


def write_dir(tag, real_rows, real_books, cond_rows, cond_books, gen_rows, gen_books):
    d = f"{CMP}/{tag}"
    for sub in ["data_real", "data_gen", "data_cond"]:
        os.makedirs(f"{d}/{sub}", exist_ok=True)

    def wr(path, rows, books):
        t = 0.0
        with open(path + "_message_" + ("real_id_0" if "gen" not in path else "real_id_0_gen_id_0") + ".csv", "w", newline="") as f:
            w = csv.writer(f)
            for (tt, et, sz, pr, di) in rows:
                t += max(tt, 1e-6); w.writerow([f"{t:.6f}", et, 0, sz, pr, di])
        np.savetxt(path + "_orderbook_" + ("real_id_0" if "gen" not in path else "real_id_0_gen_id_0") + ".csv",
                   np.asarray(books), delimiter=",", fmt="%d")
    wr(f"{d}/data_real/{BASE}", real_rows, real_books)
    wr(f"{d}/data_cond/{BASE}", cond_rows, cond_books)
    wr(f"{d}/data_gen/{BASE}", gen_rows, gen_books)


def main():
    stats = {a: col_stats(a) for a in ASSETS}
    meanB, stdB = stats["BTCUSDT"]
    z = {a: np.load(f"{DATAZ}/{a}/test.npy").astype(np.float32) for a in ASSETS}
    L = min(z[a].shape[0] for a in ASSETS)
    start = KCOND + 1
    n = min(N, (L - start - 1) // 2)
    cont_idxs = list(range(start, start + n))
    floor_idxs = list(range(start + n, start + 2 * n))
    print(f"[data] n={n} cont=[{cont_idxs[0]},{cont_idxs[-1]}] floor=[{floor_idxs[0]},{floor_idxs[-1]}]")

    real_rows, real_books = real_rows_books(z["BTCUSDT"], cont_idxs, meanB, stdB)
    cond_rows, cond_books = real_rows_books(z["BTCUSDT"], list(range(start - KCOND, start)), meanB, stdB)

    # floor: disjoint real segment orders WITH ITS OWN book (true real-vs-real floor)
    floor_rows, floor_books = real_rows_books(z["BTCUSDT"], floor_idxs, meanB, stdB)

    # naive: per-column shuffle of the real continuation orders
    rr = np.array([list(r) for r in real_rows], dtype=object)
    rng = np.random.default_rng(0)
    naive = rr.copy()
    for c in range(rr.shape[1]):
        naive[:, c] = rng.permutation(rr[:, c])
    naive_rows = [tuple(row) for row in naive]

    # each variant = (orders, book-for-those-orders). floor uses its own book;
    # naive/model orders are in the continuation's book context.
    variants = {"floor": (floor_rows, floor_books), "naive": (naive_rows, real_books)}
    for tag, (ckpt, dis) in CKPTS.items():
        print(f"[gen] {tag} (disable_graph={dis})")
        variants[tag] = (gen_model(ckpt, dis, z, cont_idxs, meanB, stdB), real_books)

    for tag, (gen_rows, gen_books) in variants.items():
        write_dir(tag, real_rows, real_books, cond_rows, cond_books, gen_rows, gen_books)
        print(f"  wrote {tag}: gen={len(gen_rows)}")
    print("BASELINES_DONE ->", CMP)


if __name__ == "__main__":
    main()
