"""Turn the rsync'd adapter npy (normalized:false) into per-asset
{train,val,test}.npy for MA_TRADES, using the same standardization as
build_c38_data.py: z-score continuous cols on the train split, leave the
categorical columns (1=event_type, 4=direction) untouched.

Sources (real data verified in an earlier local session):
  BTCUSDT/ETHUSDT  -- Tardis binance-futures book_snapshot_25 (top-10 -> 40/40)
  AAPL_A/AAPL_B    -- Databento AAPL mbp-10 (real 10-level equity), 2 halves
"""
import os
import numpy as np

RAW = "data/_adapter_raw"
SPLIT = (.85, .05, .10)
CONT = [0, 2, 3, 5] + list(range(6, 46))   # all except event_type(1) & direction(4)


def splits(n):
    a = int(n * SPLIT[0]); b = int(n * (SPLIT[0] + SPLIT[1]))
    return (0, a), (a, b), (b, n)


def standardize(train, *others):
    m = train[:, CONT].mean(0); s = train[:, CONT].std(0)
    s = np.where(s < 1e-8, 1.0, s)
    out = []
    for a in (train,) + others:
        a = a.copy().astype(np.float32)
        a[:, CONT] = (a[:, CONT] - m) / s
        out.append(a)
    return out


def build(name, src):
    arr = np.load(f"{RAW}/{src}").astype(np.float64)
    (a0, a1), (b0, b1), (c0, c1) = splits(len(arr))
    tr, va, te = standardize(arr[a0:a1], arr[b0:b1], arr[c0:c1])
    d = f"data/{name}"; os.makedirs(d, exist_ok=True)
    for sp, x in [("train", tr), ("val", va), ("test", te)]:
        assert x.shape[1] == 46, f"{name}/{sp} cols={x.shape[1]}"
        assert np.isfinite(x).all(), f"{name}/{sp} non-finite"
        assert set(np.unique(x[:, 1])).issubset({0, 1, 2}), f"{name}/{sp} bad evt"
        np.save(f"{d}/{sp}.npy", x.astype(np.float32))
    print(f"  {name}: train{tr.shape} val{va.shape} test{te.shape}")


if __name__ == "__main__":
    for name, src in [("BTCUSDT", "BTCUSDT.npy"), ("ETHUSDT", "ETHUSDT.npy"),
                      ("AAPL_A", "AAPL_A.npy"), ("AAPL_B", "AAPL_B.npy")]:
        build(name, src)
    print("REAL_DATA_DONE")
