"""Phase-2 lob_bench eval builder v2 -- AUTOREGRESSIVE rollout + matching engine.

Replaces the teacher-forced single-step builder (`build_lobbench_eval.py`). Key
differences, addressing NEXT_WORK_ZH.md task-bucket 1:

  (1.1) Stateful rollout: the target asset is generated AUTOREGRESSIVELY -- each
        generated order is decoded, applied to a price-time matching engine, and
        the resulting book is fed back as conditioning for the next step. The
        gen orderbook written for lob_bench is the engine's reconstruction, so
        spread/imbalance/volume/ofi are a genuine function of the gen messages
        (no longer trivially equal to the real book). The peer asset is kept on
        real data as exogenous context (graph coupling is ~inert, gamma~0).

  (1.2) order_ids: the engine assigns ids to resting limit orders and links
        cancels/executions to them, so log_time_to_cancel / *_ask_order_depth /
        ask_cancellation_depth stop being degenerate. Both real and gen streams
        go through the SAME engine (seeded from the same real book) so book
        metrics reflect message differences only, with no real/gen process bias.

  (1.3) coverage: ASSET selects BTCUSDT or ETHUSDT; WINDOWS independent rollout
        windows become separate lob_bench periods -> bootstrap CI across windows.

Env knobs:
  CKPT, OUT, ASSET(BTCUSDT|ETHUSDT), N_EVENTS, WINDOWS, STRIDE, SAMPLING(DDIM|DDPM),
  DDIM_NSTEPS, SIZE_SCALE, TICK(override auto), SEED.
DDIM (default, 10 steps) keeps the sequential rollout fast.
"""
from __future__ import annotations
import os, sys, csv, math
import numpy as np
import torch

import constants as cst
from constants import LearningHyperParameter as LHP
from configuration import Configuration
from run import HP_DICT_MODEL
from models.diffusers.multi_asset.ma_diffusion_engine import MultiAssetDiffusionEngine
from preprocessing.AssetUniverse import AssetUniverse
from lob_engine import MatchingEngine

# ---- config --------------------------------------------------------------
ROOT = "/DATA/DATANAS1/wpj24/AIQuant/DeepMarket"
CKPT = os.environ.get("CKPT", f"{ROOT}/data/checkpoints/MA_TRADES/val_ema=0.963_epoch=4_MA_BTCUSDT_ETHUSDT_ng_s1234_lr3e4_e5.ckpt")
RAW = f"{ROOT}/data/_adapter_raw"
DATAZ = f"{ROOT}/data"
ASSET = os.environ.get("ASSET", "BTCUSDT")
ASSETS = ["BTCUSDT", "ETHUSDT"]
TARGET_IDX = ASSETS.index(ASSET)
DATE = "2024-01-01"
SPLIT = (.85, .05, .10)
CONT = [0, 2, 3, 5] + list(range(6, 46))
OUT = os.environ.get("OUT", f"{ROOT}/data/lobbench/ar/{ASSET}/2024-01-01")

N_EVENTS = int(os.environ.get("N_EVENTS", "500"))
WINDOWS = int(os.environ.get("WINDOWS", "8"))
STRIDE = int(os.environ.get("STRIDE", "0"))          # 0 => auto-spread across test
SIZE_SCALE = float(os.environ.get("SIZE_SCALE", "100"))
SAMPLING = os.environ.get("SAMPLING", "DDIM")
SEED = int(os.environ.get("SEED", "0"))

SEQ = 256
KGEN = 1
KCOND = SEQ - KGEN          # 255
STE = 3                     # SIZE_TYPE_EMB


def col_stats(asset):
    arr = np.load(f"{RAW}/{asset}.npy").astype(np.float64)
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
    med = float(np.median(gaps))
    return 10.0 ** round(math.log10(med))


def main():
    for sub in ("data_real", "data_gen", "data_cond"):
        os.makedirs(f"{OUT}/{sub}", exist_ok=True)

    if SEED:
        torch.manual_seed(SEED); np.random.seed(SEED)

    stats = {a: col_stats(a) for a in ASSETS}
    mean, std = stats[ASSET]
    tick = est_tick(ASSET)
    print(f"[cfg] ASSET={ASSET} idx={TARGET_IDX} tick={tick} N_EVENTS={N_EVENTS} "
          f"WINDOWS={WINDOWS} SAMPLING={SAMPLING} SIZE_SCALE={SIZE_SCALE}")

    # ---- engine + checkpoint ------------------------------------------
    cfg = Configuration()
    cfg.CHOSEN_MODEL = cst.Models.MA_TRADES
    cfg.MULTI_ASSET = True
    cfg.IS_WANDB = False
    cfg.IS_TRAINING = False
    cfg.FILENAME_CKPT = "lobbench_eval_ar"
    cfg.SAMPLING_TYPE = SAMPLING
    if os.environ.get("DDIM_NSTEPS"):
        cfg.HYPER_PARAMETERS[LHP.DDIM_NSTEPS] = int(os.environ["DDIM_NSTEPS"])
    fixed = HP_DICT_MODEL[cfg.CHOSEN_MODEL].fixed
    for p in cst.LearningHyperParameter:
        if p.value in fixed:
            cfg.HYPER_PARAMETERS[p] = fixed[p.value]
    universe = AssetUniverse(assets=ASSETS, relation_types={(0, 1): 0, (1, 0): 1})
    engine = MultiAssetDiffusionEngine(cfg, universe).to(cst.DEVICE)
    sd = torch.load(CKPT, map_location=cst.DEVICE)
    state = sd.get("state_dict", sd)
    missing, unexpected = engine.load_state_dict(state, strict=False)
    print(f"[ckpt] {os.path.basename(CKPT)} missing={len(missing)} unexpected={len(unexpected)}")
    engine.eval()
    temb = engine.type_embedder.weight.data.float().cpu()

    # ---- z-scored test data -------------------------------------------
    z = {a: np.load(f"{DATAZ}/{a}/test.npy").astype(np.float32) for a in ASSETS}
    L = min(z[a].shape[0] for a in ASSETS)
    first = KCOND + 1
    last_start = L - N_EVENTS - 1
    if STRIDE > 0:
        starts = [first + w * STRIDE for w in range(WINDOWS) if first + w * STRIDE <= last_start]
    else:
        span = max(0, last_start - first)
        starts = [first + (span * w) // max(1, WINDOWS - 1) for w in range(WINDOWS)] if WINDOWS > 1 else [first]
    starts = sorted(set(starts))
    print(f"[data] test len={L}, {len(starts)} windows starts={starts[:8]}")

    def denorm_order(zc):
        """continuous z order-vec [t,etype,size,price,dir,depth] -> raw fields."""
        etype_class = int(round(zc[1]))
        size = zc[2] * std[2] + mean[2]
        price = zc[3] * std[3] + mean[3]
        t = zc[0] * std[0] + mean[0]
        direction = 1 if zc[4] >= 0 else -1
        etype = {0: 1, 1: 3, 2: 4}.get(etype_class, 1)
        return float(max(t, 1e-7)), etype, max(float(size), 1e-9), float(price), direction

    def book_raw_to_int(braw):
        out = np.zeros(40, dtype=np.int64)
        for lvl in range(10):
            ap, asz, bp, bsz = braw[4 * lvl:4 * lvl + 4]
            out[4 * lvl] = int(round(ap / tick))
            out[4 * lvl + 1] = int(round(asz * SIZE_SCALE))
            out[4 * lvl + 2] = int(round(bp / tick))
            out[4 * lvl + 3] = int(round(bsz * SIZE_SCALE))
        return out

    def write_period(subdir, idsuffix, rows, books):
        """rows: list of (inter_arrival_t, etype, order_id, size_raw, price_raw, dir)."""
        t_abs = 0.0; msg = []
        for (t, et, oid, size, price, d) in rows:
            t_abs += max(t, 1e-7)
            msg.append([f"{t_abs:.9f}", et, oid,
                        max(int(round(size * SIZE_SCALE)), 1), int(round(price / tick)), d])
        with open(f"{OUT}/{subdir}/{ASSET}_{DATE}_34200000_57600000_message_{idsuffix}.csv", "w", newline="") as f:
            w = csv.writer(f)
            for m in msg:
                w.writerow(m)
        np.savetxt(f"{OUT}/{subdir}/{ASSET}_{DATE}_34200000_57600000_orderbook_{idsuffix}.csv",
                   np.asarray(books), delimiter=",", fmt="%d")

    def reconstruct_real(z_rows, seed_book_raw):
        """Run the engine on a real continuous z order stream -> messages+books."""
        eng = MatchingEngine(tick=tick, n_levels=10)
        eng.seed_from_l2(seed_book_raw)
        rows, books = [], []
        for zc in z_rows:
            t, et, size, price, d = denorm_order(zc)
            oid, mprice = eng.step(et, price, size, d)
            rows.append((t, et, oid, size, mprice if mprice > 0 else price, d))
            books.append(book_raw_to_int(eng.l2()))
        return rows, books

    # ---- AR rollout (batched across windows) --------------------------
    nW = len(starts)
    engines = [MatchingEngine(tick=tick, n_levels=10) for _ in range(nW)]
    for w, s in enumerate(starts):
        engines[w].seed_from_l2((z[ASSET][s - 1, cst.LEN_ORDER:] * std[6:] + mean[6:]).astype(np.float64))

    # rolling conditioning buffers for the TARGET asset, per window (z units)
    ord_buf = [z[ASSET][s - KCOND:s, :cst.LEN_ORDER].copy() for s in starts]          # (KCOND,6)
    lob_buf = [z[ASSET][s - KCOND - 1:s, cst.LEN_ORDER:].copy() for s in starts]       # (KCOND+1,40) book-before each
    gen_rows = [[] for _ in range(nW)]
    gen_books = [[] for _ in range(nW)]

    with torch.no_grad():
        for k in range(N_EVENTS):
            cond_o, cond_l, x0 = [], [], []
            for w, s in enumerate(starts):
                pos = s + k
                # target asset: AR buffers; peer asset: real, advancing with k
                co = np.empty((len(ASSETS), KCOND, cst.LEN_ORDER), np.float32)
                cl = np.empty((len(ASSETS), KCOND + 1, 40), np.float32)
                for ai, a in enumerate(ASSETS):
                    if ai == TARGET_IDX:
                        co[ai] = ord_buf[w]; cl[ai] = lob_buf[w]
                    else:
                        co[ai] = z[a][pos - KCOND:pos, :cst.LEN_ORDER]
                        cl[ai] = z[a][pos - KCOND - 1:pos, cst.LEN_ORDER:]
                cond_o.append(torch.from_numpy(co)); cond_l.append(torch.from_numpy(cl))
                x0.append(torch.zeros(len(ASSETS), KGEN, cst.LEN_ORDER))
            cond_o = torch.stack(cond_o).to(cst.DEVICE)
            cond_l = torch.stack(cond_l).to(cst.DEVICE)
            x0 = torch.stack(x0).to(cst.DEVICE)
            gen = engine.sample(cond_orders=cond_o, x=x0, cond_lob=cond_l)             # (B,N,KGEN,8)
            g = gen[:, TARGET_IDX, 0, :].float().cpu().numpy()                          # (B,8)

            for w in range(nW):
                gv = g[w]
                etype_class = int(np.argmin(np.sum(np.abs(temb.numpy() - gv[1:1 + STE]), axis=1)))
                # continuous z order-vec to feed back (no rounding noise)
                zc = np.array([gv[0], etype_class, gv[STE + 1], gv[STE + 2],
                               1.0 if gv[STE + 3] >= 0 else -1.0, gv[-1]], np.float32)
                t, et, size, price, d = denorm_order(zc)
                oid, mprice = engines[w].step(et, price, size, d)
                braw = np.asarray(engines[w].l2(), dtype=np.float64)
                gen_rows[w].append((t, et, oid, size, mprice if mprice > 0 else price, d))
                gen_books[w].append(book_raw_to_int(braw))
                # slide buffers
                ord_buf[w] = np.vstack([ord_buf[w][1:], zc[None, :]])
                bz = (braw - mean[6:]) / std[6:]
                lob_buf[w] = np.vstack([lob_buf[w][1:], bz[None, :].astype(np.float32)])
            if (k + 1) % 100 == 0:
                print(f"  rollout {k + 1}/{N_EVENTS}")

    # ---- write all periods --------------------------------------------
    for w, s in enumerate(starts):
        seed = (z[ASSET][s - 1, cst.LEN_ORDER:] * std[6:] + mean[6:]).astype(np.float64)
        # real continuation (same engine machinery, seeded identically)
        real_rows, real_books = reconstruct_real(z[ASSET][s:s + N_EVENTS], seed)
        # cond prefix (real)
        seed_cond = (z[ASSET][s - KCOND - 1, cst.LEN_ORDER:] * std[6:] + mean[6:]).astype(np.float64)
        cond_rows, cond_books = reconstruct_real(z[ASSET][s - KCOND:s], seed_cond)
        write_period("data_real", f"real_id_{w}", real_rows, real_books)
        write_period("data_gen", f"real_id_{w}_gen_id_0", gen_rows[w], gen_books[w])
        write_period("data_cond", f"real_id_{w}", cond_rows, cond_books)

    # diagnostics: gen order-type mix + book sanity
    gt = np.array([r[1] for w in gen_rows for r in w])
    gd = np.array([r[5] for w in gen_rows for r in w])
    buy_frac = float((gd == 1).mean())
    print(f"[gen] type mix limit/cancel/exec = "
          f"{(gt==1).mean():.3f}/{(gt==3).mean():.3f}/{(gt==4).mean():.3f} | "
          f"dir buy-frac={buy_frac:.3f}")
    print(f"GEN_DONE windows={nW} events/window={N_EVENTS} -> {OUT}")


if __name__ == "__main__":
    main()
