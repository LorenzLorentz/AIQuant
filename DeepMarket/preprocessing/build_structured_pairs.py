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
    date: str,
    out_dir: Path,
    cache_dir: Path,
    max_rows: int | None,
    align_freq_ms: float | None,
    ll_grid_ms: float = 10.0,
    ll_max_lag_ms: float = 500.0,
) -> Dict[str, object]:
    spec = structured_universes.get(name)
    print(f"[{name}] scheme {spec.scheme}: {[l.name for l in spec.legs]}")
    if not spec.free:
        raise SystemExit(
            f"{name} is scheme {spec.scheme} ({spec.legs[0].source}); not auto-fetched. "
            "Pull via the Databento path (API key + get_cost), then point "
            "build_real_datasets.py at the resulting _adapter_raw npy."
        )

    raw_legs: List[Tuple[np.ndarray, np.ndarray]] = []
    for leg in spec.legs:
        gz = download_tardis(leg.venue, leg.symbol, date, cache_dir)
        raw_legs.append(load_leg(gz, max_rows))

    if align_freq_ms:
        legs = align_legs(raw_legs, align_freq_ms)
    else:
        legs = raw_legs

    out_dir.mkdir(parents=True, exist_ok=True)
    times: Dict[str, np.ndarray] = {}
    mids: Dict[str, np.ndarray] = {}
    for leg, (t, lob) in zip(spec.legs, legs):
        arr = leg_to_array(t, lob)
        save_lobster_like_npy(
            arr[:, : cst.LEN_ORDER], arr[:, cst.LEN_ORDER :],
            out_dir / f"{leg.name}.npy",
            manifest={
                "adapter": "build_structured_pairs",
                "source": f"tardis {leg.venue} book_snapshot_25 {leg.symbol} {date}",
                "scheme": spec.scheme,
                "universe": name,
                "aligned": bool(align_freq_ms),
                "align_freq_ms": align_freq_ms,
                "warning": "Snapshot-derived order tokens are proxies, not Level-3 messages.",
            },
        )
        np.save(out_dir / f"{leg.name}.t.npy", t.astype(np.float64))
        print(f"  wrote {leg.name}.npy  rows={arr.shape[0]}")

    # Lead-lag is measured on the *event-native* ticks (raw_legs), not the
    # LOCF-aligned grid, so sub-grid lead-lag (perp leads spot by ~10-50ms) is
    # not washed out by the alignment step.
    for leg, (t, lob) in zip(spec.legs, raw_legs):
        times[leg.name] = t
        mids[leg.name] = 0.5 * (lob[:, 0] + lob[:, 2])

    # Lead-lag: reference leg (index 0) vs every other leg.
    ref = spec.legs[0].name
    verdicts = []
    for other in spec.legs[1:]:
        v = lead_lag.analyze(
            times[ref], mids[ref], times[other.name], mids[other.name],
            name_a=ref, name_b=other.name,
            grid_ms=ll_grid_ms, max_lag_ms=ll_max_lag_ms,
        )
        verdicts.append(v)
        print(f"  lead-lag {ref} vs {other.name}: "
              f"leader={v['leader']} peak_lag={v['xcorr_peak_lag_ms']}ms "
              f"corr={v['xcorr_peak_corr']:.3f} hy_corr={v.get('hy_corr', float('nan')):.3f} "
              f"significant={v['significant']}")
    report = {"universe": name, "scheme": spec.scheme, "date": date,
              "notes": spec.notes, "lead_lag": verdicts}
    (out_dir / f"{name}.leadlag.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("name", choices=structured_universes.names())
    ap.add_argument("--date", default="2024/01/01", help="Tardis free sample = 1st of month")
    ap.add_argument("--out-dir", default="data/_adapter_raw")
    ap.add_argument("--cache-dir", default="data/_tardis_cache")
    ap.add_argument("--max-rows", type=int, default=300_000, help="0 = all rows")
    ap.add_argument("--align-freq-ms", type=float, default=None,
                    help="if set, LOCF-align all legs onto a shared grid")
    ap.add_argument("--ll-grid-ms", type=float, default=10.0,
                    help="lead-lag grid (fine; spot/perp lead-lag is sub-100ms)")
    ap.add_argument("--ll-max-lag-ms", type=float, default=500.0)
    args = ap.parse_args()
    build(
        args.name,
        date=args.date,
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
