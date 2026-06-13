"""Clip aligned equity adapter arrays to US regular trading hours (09:30-16:00 ET).

Databento mbp-10 spans ~15h/session (incl pre/post-market), where the ETF<->NAV
relation is thin/erratic. For the scheme-C no-arb experiment we want RTH only.
The structured assets are LOCF-aligned on one shared grid, so a single time mask
(from any leg's ``.t.npy``) applies to all legs and keeps them row-aligned.

No re-download / no cost -- operates on the already-saved ``_adapter_raw`` npy.

    python -m preprocessing.clip_rth QQQ AAPL MSFT NVDA AMZN --in data/_adapter_raw

Writes ``<asset>_rth.npy`` (+ ``.t.npy``); feed to build_real_datasets.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RTH_OPEN_MIN = 9 * 60 + 30   # 09:30 ET
RTH_CLOSE_MIN = 16 * 60      # 16:00 ET


def rth_mask(t_epoch_s: np.ndarray, tz: str = "America/New_York") -> np.ndarray:
    """Boolean mask of rows whose ET wall-clock is within [09:30, 16:00)."""
    ts = pd.to_datetime(np.asarray(t_epoch_s, dtype=np.float64), unit="s", utc=True).tz_convert(tz)
    mins = ts.hour * 60 + ts.minute
    weekday = ts.dayofweek  # 0=Mon..6=Sun
    return (mins >= RTH_OPEN_MIN) & (mins < RTH_CLOSE_MIN) & (weekday < 5)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("assets", nargs="+", help="asset names present in --in as <name>.npy")
    ap.add_argument("--in", dest="in_dir", default="data/_adapter_raw")
    ap.add_argument("--tref", default=None, help="asset whose .t.npy defines the mask (default: first)")
    ap.add_argument("--suffix", default="_rth")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    tref = args.tref or args.assets[0]
    t = np.load(in_dir / f"{tref}.t.npy")
    mask = rth_mask(t)
    print(f"RTH mask from {tref}.t.npy: {mask.sum()}/{len(mask)} rows kept "
          f"({100 * mask.mean():.1f}%)")

    for a in args.assets:
        arr = np.load(in_dir / f"{a}.npy")
        ta = np.load(in_dir / f"{a}.t.npy")
        if len(arr) != len(mask):
            raise SystemExit(f"{a}: {len(arr)} rows != mask {len(mask)} -- not on the shared grid?")
        np.save(in_dir / f"{a}{args.suffix}.npy", arr[mask])
        np.save(in_dir / f"{a}{args.suffix}.t.npy", ta[mask])
        print(f"  wrote {a}{args.suffix}.npy rows={int(mask.sum())}")
    print("CLIP_RTH_DONE")


if __name__ == "__main__":
    main()
