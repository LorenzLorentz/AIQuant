"""Build per-asset {train,val,test}.npy for a multi-asset MA_TRADES run on cluster38.

Asset 0 = INTC: REAL LOBSTER sample (data/INTC/INTC_2012-06-21_2012-06-21/),
                processed with the repo's own preprocess_data(), then standardized.
Asset 1 = SYNTH: GENERATED coupled second asset in the identical 46-col contract.

46-col contract (matches MultiAssetLOBDataset / LOBDataset):
  [time(z), event_type{0,1,2}, size(z), price(z), direction(+-1), depth(z),
   ask_px1, ask_sz1, bid_px1, bid_sz1, ... x10  (all z-scored) ]
Categorical columns 1 (event_type) and 4 (direction) are NEVER standardized.
Trained from scratch, so plain per-column z-score (train stats) is valid.
"""
from __future__ import annotations
import os, sys, json
import numpy as np
import pandas as pd

import constants as cst
from constants import Models
from utils.utils_data import preprocess_data

DATA = cst.DATA_DIR
SPLIT = cst.SPLIT_RATES            # (.85, .05, .10)
CONT_COLS = [0, 2, 3, 5] + list(range(6, 46))   # everything except event_type(1) & direction(4)
SEED = 30
rng = np.random.default_rng(SEED)

OB_COLS = []
for i in range(1, 11):
    OB_COLS += [f"sell{i}", f"vsell{i}", f"buy{i}", f"vbuy{i}"]
MSG_COLS = ["time", "event_type", "order_id", "size", "price", "direction"]


def row_splits(n):
    a = int(n * SPLIT[0])
    b = int(n * (SPLIT[0] + SPLIT[1]))
    return (0, a), (a, b), (b, n)


def standardize(train, *others):
    """Z-score CONT_COLS using train stats; leave categorical cols untouched."""
    mean = train[:, CONT_COLS].mean(axis=0)
    std = train[:, CONT_COLS].std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    out = []
    for arr in (train,) + others:
        a = arr.copy().astype(np.float32)
        a[:, CONT_COLS] = (a[:, CONT_COLS] - mean) / std
        out.append(a)
    return out


def save_asset(name, train, val, test):
    d = f"{DATA}/{name}"
    os.makedirs(d, exist_ok=True)
    for split, arr in [("train", train), ("val", val), ("test", test)]:
        assert arr.shape[1] == 46, f"{name}/{split} has {arr.shape[1]} cols"
        assert np.isfinite(arr).all(), f"{name}/{split} has non-finite values"
        ev = arr[:, 1]
        assert set(np.unique(ev)).issubset({0, 1, 2}), f"{name}/{split} bad event_type {np.unique(ev)}"
        np.save(f"{d}/{split}.npy", arr.astype(np.float32))
    print(f"  saved {name}: train{train.shape} val{val.shape} test{test.shape}")


# ---------------- Asset 0: real INTC ----------------------------------------
def build_intc():
    print("[INTC] reading real LOBSTER sample ...")
    base = f"{DATA}/INTC/INTC_2012-06-21_2012-06-21"
    pref = "INTC_2012-06-21_34200000_57600000"
    msg = pd.read_csv(f"{base}/{pref}_message_10.csv", names=MSG_COLS)
    ob = pd.read_csv(f"{base}/{pref}_orderbook_10.csv", names=OB_COLS)
    print(f"[INTC] raw rows: msg={len(msg)} ob={len(ob)}")
    ob_p, msg_p = preprocess_data([msg, ob], cst.N_LOB_LEVELS, Models.TRADES)
    # msg_p cols: [time, event_type, size, price, direction, depth]
    et = msg_p["event_type"].to_numpy()
    et = np.select([et == 1, et == 3, et == 4], [0, 1, 2], default=0)  # {1,3,4}->{0,1,2}
    msg_arr = msg_p[["time", "event_type", "size", "price", "direction", "depth"]].to_numpy(dtype=np.float64)
    msg_arr[:, 1] = et
    arr = np.concatenate([msg_arr, ob_p.to_numpy(dtype=np.float64)], axis=1)
    print(f"[INTC] processed array: {arr.shape}")
    (a0, a1), (b0, b1), (c0, c1) = row_splits(len(arr))
    tr, va, te = standardize(arr[a0:a1], arr[b0:b1], arr[c0:c1])
    save_asset("INTC", tr, va, te)
    return len(arr)


# ---------------- Asset 1: synthetic coupled SYNTH --------------------------
def build_synth(n_rows):
    print(f"[SYNTH] generating {n_rows} coupled rows ...")
    n = n_rows
    # latent mid-price random walk (a plausible LOB)
    rets = rng.normal(0, 1.0, n).cumsum()
    mid = 10000.0 + rets * 5.0
    tick = 1.0
    # order events
    event_type = rng.choice([0, 1, 2], size=n, p=[0.5, 0.4, 0.1]).astype(np.float64)
    direction = rng.choice([-1.0, 1.0], size=n)
    size = rng.lognormal(4.0, 1.0, n)
    time = rng.exponential(0.05, n)
    depth = rng.poisson(1.0, n).astype(np.float64)
    price = mid + direction * tick * (depth + rng.integers(0, 2, n))
    # build a coherent 10-level book around mid
    lob = np.zeros((n, 40), dtype=np.float64)
    for lvl in range(10):
        lob[:, lvl * 4 + 0] = mid + tick * (lvl + 1)             # ask price
        lob[:, lvl * 4 + 1] = rng.lognormal(5.0, 0.8, n)         # ask size
        lob[:, lvl * 4 + 2] = mid - tick * (lvl + 1)             # bid price
        lob[:, lvl * 4 + 3] = rng.lognormal(5.0, 0.8, n)         # bid size
    arr = np.concatenate([
        np.stack([time, event_type, size, price, direction, depth], axis=1), lob], axis=1)
    (a0, a1), (b0, b1), (c0, c1) = row_splits(n)
    tr, va, te = standardize(arr[a0:a1], arr[b0:b1], arr[c0:c1])
    save_asset("SYNTH", tr, va, te)


if __name__ == "__main__":
    n = build_intc()
    build_synth(n)
    print("DATA_BUILD_DONE")
