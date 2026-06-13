"""Fetch + adapt + (optionally) time-align a structured asset universe.

Task bucket 4 data driver. Given a name from :mod:`structured_universes`, this:

  1. downloads each leg's Tardis ``book_snapshot_25`` daily free sample
     (scheme A spot/perp, scheme B cross-venue; scheme C / Databento is left to
     a separate paid path and only validated here, not auto-fetched),
  2. converts each leg to a DeepMarket adapter array via the Tardis-aware
     ``iex_or_databento_adapter`` (top-10 of the 25 levels),
  3. optionally **time-aligns** the legs onto a shared regular grid (asof LOCF)
     -- the missing piece flagged in DATA_SOURCES_ZH 3.4 that basis / no-arb
     need, since ``MultiAssetLOBDataset`` only truncates to min length, and
  4. writes per-asset ``data/_adapter_raw/<asset>.npy`` (+ ``.t.npy`` absolute
     seconds sidecar) and runs the lead-lag verifier across the legs.

The un-normalized ``_adapter_raw`` outputs are the same contract
``build_real_datasets.py`` consumes (z-score + split). Run that next.

Example (on cluster38, deep_market env)::

    python -m preprocessing.build_structured_pairs btc_spot_perp \
        --date 2024/01/01 --max-rows 300000 --align-freq-ms 100
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import constants as cst
from preprocessing import lead_lag, structured_universes
from preprocessing.adapters.common import (
    save_lobster_like_npy,
    synthesize_orders_from_lob,
)
from preprocessing.adapters.iex_or_databento_adapter import (
    TIMESTAMP_CANDIDATES,
    equity_frame_to_lob,
    _find_column,
)

TARDIS_URL = "https://datasets.tardis.dev/v1/{venue}/book_snapshot_25/{date}/{symbol}.csv.gz"


def download_tardis(venue: str, symbol: str, date: str, cache_dir: Path) -> Path:
    """Download (and cache) one Tardis daily snapshot file. ``date`` = YYYY/MM/DD."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / f"{venue}_{symbol}_{date.replace('/', '-')}.csv.gz"
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  cached {dest.name}")
        return dest
    url = TARDIS_URL.format(venue=venue, symbol=symbol, date=date)
    print(f"  GET {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "aiquant/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = resp.read()
    # Tardis serves 4xx as a tiny JSON body; a real file is gzipped CSV.
    if len(data) < 1024 or not data[:2] == b"\x1f\x8b":
        raise RuntimeError(f"{venue}/{symbol} {date}: not a gzip payload ({len(data)} bytes): {data[:200]!r}")
    dest.write_bytes(data)
    print(f"  saved {dest.name} ({len(data)/1e6:.1f} MB)")
    return dest


def _abs_seconds(frame: pd.DataFrame) -> np.ndarray:
    """Absolute epoch seconds (NOT relative) from a timestamp column."""
    col = _find_column(frame, TIMESTAMP_CANDIDATES)
    if col is None:
        raise ValueError(f"no timestamp column among {TIMESTAMP_CANDIDATES}")
    vals = frame[col]
    if pd.api.types.is_numeric_dtype(vals):
        raw = vals.to_numpy(dtype=np.float64)
        mag = np.nanmedian(np.abs(raw)) if len(raw) else 0.0
        if mag > 1e17:
            return raw / 1e9   # ns
        if mag > 1e14:
            return raw / 1e6   # us  (Tardis)
        if mag > 1e11:
            return raw / 1e3   # ms
        return raw             # already seconds
    return pd.to_datetime(vals, utc=True).to_numpy("datetime64[ns]").astype("int64") / 1e9


def load_leg(gz_path: Path, max_rows: int | None) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(abs_seconds, lob[T,40])`` for one Tardis file."""
    with gzip.open(gz_path, "rt") as fh:
        frame = pd.read_csv(fh, nrows=max_rows)
    lob = equity_frame_to_lob(frame, levels=cst.N_LOB_LEVELS)
    t = _abs_seconds(frame)
    return t, lob


def align_legs(
    legs: List[Tuple[np.ndarray, np.ndarray]], freq_ms: float
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """LOCF-resample every leg onto one shared regular grid (row-aligned)."""
    t0 = max(t.min() for t, _ in legs)
    t1 = min(t.max() for t, _ in legs)
    if not (t1 > t0):
        raise ValueError("legs do not overlap in time; cannot align")
    grid = np.arange(t0, t1, freq_ms / 1000.0)
    out = []
    for t, lob in legs:
        idx = np.clip(np.searchsorted(t, grid, side="right") - 1, 0, len(lob) - 1)
        out.append((grid.copy(), lob[idx]))
    print(f"  aligned {len(legs)} legs onto {len(grid)} grid points "
          f"@ {freq_ms}ms over {t1 - t0:.0f}s")
    return out


def leg_to_array(t: np.ndarray, lob: np.ndarray) -> np.ndarray:
    orders = synthesize_orders_from_lob(lob, timestamps=t)
    return np.concatenate([orders, lob], axis=1).astype(np.float32, copy=False)


def build(
    name: str,
    *,
    dates: List[str],
    out_dir: Path,
    cache_dir: Path,
    max_rows: int | None,
    align_freq_ms: float | None,
    ll_grid_ms: float = 10.0,
    ll_max_lag_ms: float = 500.0,
) -> Dict[str, object]:
    spec = structured_universes.get(name)
    print(f"[{name}] scheme {spec.scheme}: {[l.name for l in spec.legs]}  dates={dates}")
    if not spec.free:
        raise SystemExit(
            f"{name} is scheme {spec.scheme} ({spec.legs[0].source}); not auto-fetched. "
            "Pull via the Databento path (fetch_databento.py: get_cost + download), "
            "then point build_real_datasets.py at the resulting _adapter_raw npy."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    # Per leg, accumulate (T,46) arrays + abs-time across all dates, and collect
    # one lead-lag verdict per date (measured on that day's event-native ticks).
    leg_arrays: Dict[str, List[np.ndarray]] = {l.name: [] for l in spec.legs}
    leg_times: Dict[str, List[np.ndarray]] = {l.name: [] for l in spec.legs}
    ref = spec.legs[0].name
    verdicts: List[Dict[str, object]] = []

    for date in dates:
        print(f"-- {date} --")
        raw_legs: List[Tuple[np.ndarray, np.ndarray]] = []
        for leg in spec.legs:
            gz = download_tardis(leg.venue, leg.symbol, date, cache_dir)
            raw_legs.append(load_leg(gz, max_rows))
        legs = align_legs(raw_legs, align_freq_ms) if align_freq_ms else raw_legs

        # Per-day, full-resolution lead-lag (don't span days at 10ms -> blowup).
        ev = {l.name: (t, 0.5 * (lob[:, 0] + lob[:, 2])) for l, (t, lob) in zip(spec.legs, raw_legs)}
        for other in spec.legs[1:]:
            v = lead_lag.analyze(*ev[ref], *ev[other.name], name_a=ref, name_b=other.name,
                                 grid_ms=ll_grid_ms, max_lag_ms=ll_max_lag_ms)
            v["date"] = date
            verdicts.append(v)
            print(f"  lead-lag {ref} vs {other.name}: leader={v['leader']} "
                  f"peak_lag={v['xcorr_peak_lag_ms']}ms corr={v['xcorr_peak_corr']:.3f} "
                  f"hy_corr={v.get('hy_corr', float('nan')):.3f} significant={v['significant']}")

        for leg, (t, lob) in zip(spec.legs, legs):
            leg_arrays[leg.name].append(leg_to_array(t, lob))
            leg_times[leg.name].append(t.astype(np.float64))

    # Concatenate dates per leg and save (z-score/split is build_real_datasets.py).
    for leg in spec.legs:
        arr = np.concatenate(leg_arrays[leg.name], axis=0)
        save_lobster_like_npy(
            arr[:, : cst.LEN_ORDER], arr[:, cst.LEN_ORDER :],
            out_dir / f"{leg.name}.npy",
            manifest={
                "adapter": "build_structured_pairs",
                "source": f"tardis {leg.venue} book_snapshot_25 {leg.symbol}",
                "dates": dates, "scheme": spec.scheme, "universe": name,
                "aligned": bool(align_freq_ms), "align_freq_ms": align_freq_ms,
                "warning": "Snapshot-derived order tokens are proxies, not Level-3 messages.",
            },
        )
        np.save(out_dir / f"{leg.name}.t.npy", np.concatenate(leg_times[leg.name]))
        print(f"  wrote {leg.name}.npy  rows={arr.shape[0]} ({len(dates)} dates)")

    report = {"universe": name, "scheme": spec.scheme, "dates": dates,
              "notes": spec.notes, "lead_lag": verdicts}
    (out_dir / f"{name}.leadlag.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _resolve_dates(args) -> List[str]:
    """Build the date list from --months (range) | --dates (list) | --date."""
    if args.months:
        lo, hi = args.months.split(":")
        y0, m0 = (int(x) for x in lo.split("-"))
        y1, m1 = (int(x) for x in hi.split("-"))
        out = []
        y, m = y0, m0
        while (y, m) <= (y1, m1):
            out.append(f"{y:04d}/{m:02d}/01")
            m += 1
            if m > 12:
                y, m = y + 1, 1
        return out
    if args.dates:
        return [d.strip() for d in args.dates.split(",") if d.strip()]
    return [args.date]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("name", choices=structured_universes.names())
    ap.add_argument("--date", default="2024/01/01", help="single Tardis free sample = 1st of month")
    ap.add_argument("--dates", default=None,
                    help="comma list of YYYY/MM/DD (overrides --date)")
    ap.add_argument("--months", default=None,
                    help="YYYY-MM:YYYY-MM -> 1st of each month inclusive (free samples)")
    ap.add_argument("--out-dir", default="data/_adapter_raw")
    ap.add_argument("--cache-dir", default="data/_tardis_cache")
    ap.add_argument("--max-rows", type=int, default=300_000, help="0 = all rows")
    ap.add_argument("--align-freq-ms", type=float, default=None,
                    help="if set, LOCF-align all legs onto a shared grid")
    ap.add_argument("--ll-grid-ms", type=float, default=10.0,
                    help="lead-lag grid (fine; spot/perp lead-lag is sub-100ms)")
    ap.add_argument("--ll-max-lag-ms", type=float, default=500.0)
    args = ap.parse_args()
    dates = _resolve_dates(args)
    build(
        args.name,
        dates=dates,
        out_dir=Path(args.out_dir),
        cache_dir=Path(args.cache_dir),
        max_rows=None if args.max_rows == 0 else args.max_rows,
        align_freq_ms=args.align_freq_ms,
        ll_grid_ms=args.ll_grid_ms,
        ll_max_lag_ms=args.ll_max_lag_ms,
    )
    print("STRUCTURED_BUILD_DONE")


if __name__ == "__main__":
    main()
