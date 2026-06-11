"""P0 head-to-head AR lob_bench builder — MODEL-switchable fork of
build_lobbench_eval_ar.py so the SAME autoregressive-rollout + matching-engine
protocol can score BOTH a single-asset TRADES engine and a shared MA_TRADES
(no-graph) engine on the same real ETH test windows.

Difference vs build_lobbench_eval_ar.py: MODEL env selects the engine and the
conditioning shape. TRADES => single asset, no peer. MA_TRADES => target asset
generated AR, peer asset kept real (graph inert). Everything else (engine seed
from real book, order_id tracking, tick quantization, window bootstrap) is
identical so the two models are scored under one protocol.

Env: MODEL(TRADES|MA_TRADES), CKPT, OUT, ASSET(=ETHUSDT), PEER(=BTCUSDT for MA),
     N_EVENTS, WINDOWS, STRIDE, SAMPLING(DDIM), DDIM_NSTEPS, SIZE_SCALE, TICK, SEED.
"""
from __future__ import annotations
import os, csv, math
import numpy as np
import torch

import constants as cst
from constants import LearningHyperParameter as LHP
from configuration import Configuration
from run import HP_DICT_MODEL
from models.diffusers.diffusion_engine import DiffusionEngine
from models.diffusers.multi_asset.ma_diffusion_engine import MultiAssetDiffusionEngine
from preprocessing.AssetUniverse import AssetUniverse
from lob_engine import MatchingEngine

ROOT = "/DATA/DATANAS1/wpj24/AIQuant/DeepMarket"
RAW = f"{ROOT}/data/_adapter_raw"
DATAZ = f"{ROOT}/data"
MODEL = os.environ.get("MODEL", "TRADES")
ASSET = os.environ.get("ASSET", "ETHUSDT")
PEER = os.environ.get("PEER", "BTCUSDT")
CKPT = os.environ["CKPT"]
DATE = "2024-01-01"
SPLIT = (.85, .05, .10)
CONT = [0, 2, 3, 5] + list(range(6, 46))
OUT = os.environ.get("OUT", f"{ROOT}/data/lobbench/p0/{MODEL}_{ASSET}")
N_EVENTS = int(os.environ.get("N_EVENTS", "500"))
WINDOWS = int(os.environ.get("WINDOWS", "8"))
STRIDE = int(os.environ.get("STRIDE", "0"))
SIZE_SCALE = float(os.environ.get("SIZE_SCALE", "100"))
SAMPLING = os.environ.get("SAMPLING", "DDIM")
SEED = int(os.environ.get("SEED", "0"))
MA = MODEL == "MA_TRADES"
LOGTIME = os.environ.get("LOGTIME", "0") == "1"   # option-b: time col is log1p(inter-arrival)
ASSETS = [PEER, ASSET] if MA else [ASSET]
TARGET_IDX = ASSETS.index(ASSET)

SEQ = 256; KGEN = 1; KCOND = SEQ - KGEN; STE = 3
LO = cst.LEN_ORDER


def col_stats(asset):
    arr = np.load(f"{RAW}/{asset}.npy").astype(np.float64)
    if LOGTIME:
        arr[:, 0] = np.log1p(np.clip(arr[:, 0], 0, None))
    n = len(arr); a = int(n * SPLIT[0]); tr = arr[:a]
    mean = np.zeros(46); std = np.ones(46)
    m = tr[:, CONT].mean(0); s = tr[:, CONT].std(0); s = np.where(s < 1e-8, 1.0, s)
    mean[CONT] = m; std[CONT] = s
    return mean, std


def est_tick(asset):
    if os.environ.get("TICK"):
        return float(os.environ["TICK"])
    arr = np.load(f"{RAW}/{asset}.npy").astype(np.float64)
    ask = arr[:50000, 6:6 + 40:4]
    gaps = np.diff(ask, axis=1).ravel(); gaps = gaps[gaps > 1e-9]
    return 10.0 ** round(math.log10(float(np.median(gaps))))


def build_engine():
    cfg = Configuration()
    cfg.IS_WANDB = False; cfg.IS_TRAINING = False
    cfg.SAMPLING_TYPE = SAMPLING
    cfg.FILENAME_CKPT = "lobbench_eval_ar_p0"
    if os.environ.get("DDIM_NSTEPS"):
        cfg.HYPER_PARAMETERS[LHP.DDIM_NSTEPS] = int(os.environ["DDIM_NSTEPS"])
    if MA:
        cfg.CHOSEN_MODEL = cst.Models.MA_TRADES; cfg.MULTI_ASSET = True
    else:
        cfg.CHOSEN_MODEL = cst.Models.TRADES; cfg.MULTI_ASSET = False
    fixed = HP_DICT_MODEL[cfg.CHOSEN_MODEL].fixed
    for p in cst.LearningHyperParameter:
        if p.value in fixed:
            cfg.HYPER_PARAMETERS[p] = fixed[p.value]
    if MA:
        universe = AssetUniverse(assets=ASSETS, relation_types={(0, 1): 0, (1, 0): 1})
        eng = MultiAssetDiffusionEngine(cfg, universe).to(cst.DEVICE)
    else:
        eng = DiffusionEngine(cfg).to(cst.DEVICE)
    sd = torch.load(CKPT, map_location=cst.DEVICE)
    missing, unexpected = eng.load_state_dict(sd.get("state_dict", sd), strict=False)
    print(f"[ckpt] {os.path.basename(CKPT)} missing={len(missing)} unexpected={len(unexpected)}")
    eng.eval()
    return eng


def main():
    for sub in ("data_real", "data_gen", "data_cond"):
        os.makedirs(f"{OUT}/{sub}", exist_ok=True)
    if SEED:
        torch.manual_seed(SEED); np.random.seed(SEED)
    mean, std = col_stats(ASSET)
    tick = est_tick(ASSET)
    print(f"[cfg] MODEL={MODEL} ASSET={ASSET} tick={tick} N_EVENTS={N_EVENTS} WINDOWS={WINDOWS}")

    eng = build_engine()
    temb = eng.type_embedder.weight.data.float().cpu().numpy()

    z = {a: np.load(f"{DATAZ}/{a}/test.npy").astype(np.float32) for a in ASSETS}
    L = min(z[a].shape[0] for a in ASSETS)
    first = KCOND + 1; last_start = L - N_EVENTS - 1
    if STRIDE > 0:
        starts = [first + w * STRIDE for w in range(WINDOWS) if first + w * STRIDE <= last_start]
    else:
        span = max(0, last_start - first)
        starts = [first + (span * w) // max(1, WINDOWS - 1) for w in range(WINDOWS)] if WINDOWS > 1 else [first]
    starts = sorted(set(starts))
    print(f"[data] test len={L}, {len(starts)} windows")

    def denorm_order(zc):
        etype_class = int(round(zc[1]))
        size = zc[2] * std[2] + mean[2]
        price = zc[3] * std[3] + mean[3]
        t = zc[0] * std[0] + mean[0]
        if LOGTIME:
            t = math.expm1(min(t, 50.0))   # invert log1p (clamp to avoid overflow)
        direction = 1 if zc[4] >= 0 else -1
        etype = {0: 1, 1: 3, 2: 4}.get(etype_class, 1)
        return float(max(t, 1e-7)), etype, max(float(size), 1e-9), float(price), direction

    def book_raw_to_int(braw):
        out = np.zeros(40, dtype=np.int64)
        for lvl in range(10):
            ap, asz, bp, bsz = braw[4 * lvl:4 * lvl + 4]
            out[4 * lvl] = int(round(ap / tick)); out[4 * lvl + 1] = int(round(asz * SIZE_SCALE))
            out[4 * lvl + 2] = int(round(bp / tick)); out[4 * lvl + 3] = int(round(bsz * SIZE_SCALE))
        return out

    def write_period(subdir, idsuffix, rows, books):
        t_abs = 0.0; msg = []
        for (t, et, oid, size, price, d) in rows:
            t_abs += max(t, 1e-7)
            msg.append([f"{t_abs:.9f}", et, oid, max(int(round(size * SIZE_SCALE)), 1), int(round(price / tick)), d])
        with open(f"{OUT}/{subdir}/{ASSET}_{DATE}_34200000_57600000_message_{idsuffix}.csv", "w", newline="") as f:
            w = csv.writer(f)
            for m in msg:
                w.writerow(m)
        np.savetxt(f"{OUT}/{subdir}/{ASSET}_{DATE}_34200000_57600000_orderbook_{idsuffix}.csv",
                   np.asarray(books), delimiter=",", fmt="%d")

    def reconstruct_real(z_rows, seed_book_raw):
        eng2 = MatchingEngine(tick=tick, n_levels=10)
        eng2.seed_from_l2(seed_book_raw)
        rows, books = [], []
        for zc in z_rows:
            t, et, size, price, d = denorm_order(zc)
            oid, mprice = eng2.step(et, price, size, d)
            rows.append((t, et, oid, size, mprice if mprice > 0 else price, d))
            books.append(book_raw_to_int(eng2.l2()))
        return rows, books

    nW = len(starts)
    engines = [MatchingEngine(tick=tick, n_levels=10) for _ in range(nW)]
    for w, s in enumerate(starts):
        engines[w].seed_from_l2((z[ASSET][s - 1, LO:] * std[6:] + mean[6:]).astype(np.float64))

    ord_buf = [z[ASSET][s - KCOND:s, :LO].copy() for s in starts]
    lob_buf = [z[ASSET][s - KCOND - 1:s, LO:].copy() for s in starts]
    gen_rows = [[] for _ in range(nW)]; gen_books = [[] for _ in range(nW)]

    # option (c): point-process time head -- if PP_CKPT set, the time field is
    # sampled from the conditional point-process model instead of the diffusion.
    PP = None
    if os.environ.get("PP_CKPT"):
        from pp_model import PointProcess
        PP = PointProcess.load(os.environ["PP_CKPT"], device=str(cst.DEVICE))
        pp_ld = []; pp_et = []
        for s in starts:
            rdt = z[ASSET][s - 64:s, 0] * std[0] + mean[0]      # raw inter-arrival (LOGTIME=0 here)
            pp_ld.append(list(np.log1p(np.clip(rdt, 1e-6, None))))
            pp_et.append(list(np.clip(np.round(z[ASSET][s - 64:s, 1]), 0, 2).astype(int)))
        print(f"[pp] point-process time head loaded: {os.environ['PP_CKPT']}")

    with torch.no_grad():
        for k in range(N_EVENTS):
            cond_o, cond_l, x0 = [], [], []
            for w, s in enumerate(starts):
                pos = s + k
                if MA:
                    co = np.empty((len(ASSETS), KCOND, LO), np.float32)
                    cl = np.empty((len(ASSETS), KCOND + 1, 40), np.float32)
                    for ai, a in enumerate(ASSETS):
                        if ai == TARGET_IDX:
                            co[ai] = ord_buf[w]; cl[ai] = lob_buf[w]
                        else:
                            co[ai] = z[a][pos - KCOND:pos, :LO]
                            cl[ai] = z[a][pos - KCOND - 1:pos, LO:]
                    x0.append(torch.zeros(len(ASSETS), KGEN, LO))
                else:
                    co = ord_buf[w].astype(np.float32)
                    cl = lob_buf[w].astype(np.float32)
                    x0.append(torch.zeros(KGEN, LO))
                cond_o.append(torch.from_numpy(co)); cond_l.append(torch.from_numpy(cl))
            cond_o = torch.stack(cond_o).to(cst.DEVICE)
            cond_l = torch.stack(cond_l).to(cst.DEVICE)
            x0 = torch.stack(x0).to(cst.DEVICE)
            gen = eng.sample(cond_orders=cond_o, x=x0, cond_lob=cond_l)
            g = (gen[:, TARGET_IDX, 0, :] if MA else gen[:, 0, :]).float().cpu().numpy()

            for w in range(nW):
                gv = g[w]
                etype_class = int(np.argmin(np.sum(np.abs(temb - gv[1:1 + STE]), axis=1)))
                if PP is not None:
                    pp_dt = PP.sample(pp_ld[w], pp_et[w])
                    zc0 = (pp_dt - mean[0]) / std[0]            # decodes back to pp_dt (LOGTIME=0)
                    pp_ld[w].append(float(np.log1p(max(pp_dt, 1e-6)))); pp_et[w].append(etype_class)
                else:
                    zc0 = gv[0]
                zc = np.array([zc0, etype_class, gv[STE + 1], gv[STE + 2],
                               1.0 if gv[STE + 3] >= 0 else -1.0, gv[-1]], np.float32)
                t, et, size, price, d = denorm_order(zc)
                oid, mprice = engines[w].step(et, price, size, d)
                braw = np.asarray(engines[w].l2(), dtype=np.float64)
                gen_rows[w].append((t, et, oid, size, mprice if mprice > 0 else price, d))
                gen_books[w].append(book_raw_to_int(braw))
                ord_buf[w] = np.vstack([ord_buf[w][1:], zc[None, :]])
                bz = (braw - mean[6:]) / std[6:]
                lob_buf[w] = np.vstack([lob_buf[w][1:], bz[None, :].astype(np.float32)])
            if (k + 1) % 100 == 0:
                print(f"  rollout {k + 1}/{N_EVENTS}")

    for w, s in enumerate(starts):
        seed = (z[ASSET][s - 1, LO:] * std[6:] + mean[6:]).astype(np.float64)
        real_rows, real_books = reconstruct_real(z[ASSET][s:s + N_EVENTS], seed)
        seed_cond = (z[ASSET][s - KCOND - 1, LO:] * std[6:] + mean[6:]).astype(np.float64)
        cond_rows, cond_books = reconstruct_real(z[ASSET][s - KCOND:s], seed_cond)
        write_period("data_real", f"real_id_{w}", real_rows, real_books)
        write_period("data_gen", f"real_id_{w}_gen_id_0", gen_rows[w], gen_books[w])
        write_period("data_cond", f"real_id_{w}", cond_rows, cond_books)

    gt = np.array([r[1] for w in gen_rows for r in w])
    gd = np.array([r[5] for w in gen_rows for r in w])
    print(f"[gen] type mix limit/cancel/exec = "
          f"{(gt==1).mean():.3f}/{(gt==3).mean():.3f}/{(gt==4).mean():.3f} | buy-frac={(gd==1).mean():.3f}")
    print(f"GEN_DONE windows={nW} -> {OUT}")


if __name__ == "__main__":
    main()
