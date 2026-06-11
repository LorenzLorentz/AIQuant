"""Option-b (task 2.2): build log-time variants of BTC/ETH datasets.
Time col (idx 0, raw inter-arrival) -> log1p, THEN the same train-split z-score
as build_real_datasets.py. Heavy-tailed inter-arrivals become near-Gaussian, so
the diffusion model should match the inter-arrival DISTRIBUTION better than raw-z.
Also symlinks RAW/<name>.npy so the AR eval's col_stats/est_tick (with LOGTIME=1)
work against the raw inter-arrival source.
"""
import os
import numpy as np

RAW = "data/_adapter_raw"
SPLIT = (.85, .05, .10)
CONT = [0, 2, 3, 5] + list(range(6, 46))


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
    arr[:, 0] = np.log1p(np.clip(arr[:, 0], 0, None))     # log-time
    (a0, a1), (b0, b1), (c0, c1) = splits(len(arr))
    tr, va, te = standardize(arr[a0:a1], arr[b0:b1], arr[c0:c1])
    d = f"data/{name}"; os.makedirs(d, exist_ok=True)
    for sp, x in [("train", tr), ("val", va), ("test", te)]:
        assert x.shape[1] == 46 and np.isfinite(x).all()
        np.save(f"{d}/{sp}.npy", x.astype(np.float32))
    link = f"{RAW}/{name}.npy"
    if not os.path.exists(link):
        os.symlink(os.path.abspath(f"{RAW}/{src}"), link)
    print(f"  {name}: train{tr.shape} val{va.shape} test{te.shape} (log-time)")


if __name__ == "__main__":
    for name, src in [("BTCUSDT_lt", "BTCUSDT.npy"), ("ETHUSDT_lt", "ETHUSDT.npy")]:
        build(name, src)
    print("LOGTIME_DATA_DONE")
