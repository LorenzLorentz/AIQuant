"""Databento fetch for scheme C (ETF vs constituents) -- estimate-first.

Strict budget ($125 credit): this ALWAYS prints a ``get_cost`` estimate and only
downloads when ``--pull`` is passed, so you never spend blind. The API key is
read from ``DATABENTO_API_KEY`` (env or a ``.env`` at repo root / cwd).

Estimate::

    python -m preprocessing.fetch_databento qqq_basket --start 2024-06-03 --end 2024-06-04

Pull (after you accept the printed cost)::

    python -m preprocessing.fetch_databento qqq_basket --start 2024-06-03 --end 2024-06-04 --pull

``mbp-10`` gives top-10 book snapshots (order tokens are proxies, like Tardis);
true L3 would need ``mbo`` + a new adapter. Output: ``_adapter_raw/<sym>.npy``
(+ ``.t.npy``), ready for ``build_real_datasets.py``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np

import constants as cst
from preprocessing import lead_lag, structured_universes
from preprocessing.adapters.common import save_lobster_like_npy
from preprocessing.adapters.iex_or_databento_adapter import equity_frame_to_lob
from preprocessing.build_structured_pairs import _abs_seconds, align_legs, leg_to_array


def _load_api_key() -> str:
    key = os.environ.get("DATABENTO_API_KEY")
    if key:
        return key
    for env in (Path(".env"), Path(__file__).resolve().parents[2] / ".env"):
        if env.exists():
            for line in env.read_text().splitlines():
                if line.startswith("DATABENTO_API_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("DATABENTO_API_KEY not found in env or .env")


def estimate_cost(client, dataset, symbols, schema, start, end) -> float:
    cost = client.metadata.get_cost(
        dataset=dataset, symbols=symbols, schema=schema, start=start, end=end
    )
    return float(cost)


def _symbol_t_lob(client, dataset, sym, schema, start, end) -> Tuple[np.ndarray, np.ndarray]:
    """Download one symbol/day and return (abs_seconds, lob[T,40])."""
    data = client.timeseries.get_range(
        dataset=dataset, symbols=[sym], schema=schema, start=start, end=end
    )
    frame = data.to_df(price_type="float", pretty_ts=True).reset_index()
    return _abs_seconds(frame), equity_frame_to_lob(frame)


def pull(client, name, dataset, symbols, schema, dates, out_dir: Path,
         align_freq_ms: float | None) -> None:
    """Pull each (symbol, day), LOCF-align the basket per day, concat days, save.

    Alignment matters here: the ETF NAV basis ``QQQ - sum w_i constituent_i``
    is only meaningful when the legs share a clock (P3 energy compares mids at
    matched instants). Same machinery as the crypto driver.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = structured_universes.get(name)
    leg_arrays: dict = {s: [] for s in symbols}
    leg_times: dict = {s: [] for s in symbols}
    ref = symbols[0]
    verdicts = []
    for day in dates:
        nxt = (np.datetime64(day) + np.timedelta64(1, "D")).astype(str)
        print(f"-- {day} --")
        raw = []
        for sym in symbols:
            print(f"  download {sym} {schema}")
            raw.append(_symbol_t_lob(client, dataset, sym, schema, day, nxt))
        legs = align_legs(raw, align_freq_ms) if align_freq_ms else raw
        ev = {s: (t, 0.5 * (lob[:, 0] + lob[:, 2])) for s, (t, lob) in zip(symbols, raw)}
        for other in symbols[1:]:
            v = lead_lag.analyze(*ev[ref], *ev[other], name_a=ref, name_b=other,
                                 grid_ms=100.0, max_lag_ms=5000.0)
            v["date"] = day
            verdicts.append(v)
            print(f"  lead-lag {ref} vs {other}: leader={v['leader']} "
                  f"peak_lag={v['xcorr_peak_lag_ms']}ms corr={v['xcorr_peak_corr']:.3f} "
                  f"hy_corr={v.get('hy_corr', float('nan')):.3f}")
        for sym, (t, lob) in zip(symbols, legs):
            leg_arrays[sym].append(leg_to_array(t, lob))
            leg_times[sym].append(t.astype(np.float64))

    for sym in symbols:
        arr = np.concatenate(leg_arrays[sym], axis=0)
        save_lobster_like_npy(
            arr[:, : cst.LEN_ORDER], arr[:, cst.LEN_ORDER :],
            out_dir / f"{sym}.npy",
            manifest={"adapter": "fetch_databento", "source": f"databento {dataset} {schema} {sym}",
                      "dates": dates, "scheme": spec.scheme, "universe": name,
                      "aligned": bool(align_freq_ms), "align_freq_ms": align_freq_ms,
                      "warning": "mbp-10 order tokens are proxies, not Level-3 messages."},
        )
        np.save(out_dir / f"{sym}.t.npy", np.concatenate(leg_times[sym]))
        print(f"    wrote {sym}.npy rows={arr.shape[0]} ({len(dates)} dates)")
    (out_dir / f"{name}.leadlag.json").write_text(
        json.dumps({"universe": name, "dates": dates, "lead_lag": verdicts}, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("name", help="structured universe name (scheme C), e.g. qqq_basket")
    ap.add_argument("--dataset", default="XNAS.ITCH")
    ap.add_argument("--schema", default="mbp-10")
    ap.add_argument("--dates", required=True,
                    help="comma list of trading days YYYY-MM-DD (each = 1 RTH session)")
    ap.add_argument("--align-freq-ms", type=float, default=100.0)
    ap.add_argument("--out-dir", default="data/_adapter_raw")
    ap.add_argument("--pull", action="store_true", help="actually download (spends credit)")
    args = ap.parse_args()

    import databento as db

    spec = structured_universes.get(args.name)
    symbols: List[str] = [l.symbol for l in spec.legs]
    client = db.Historical(_load_api_key())
    dates = [d.strip() for d in args.dates.split(",") if d.strip()]

    print(f"[{args.name}] {args.dataset} {args.schema} {symbols}")
    total = 0.0
    for day in dates:
        nxt = (np.datetime64(day) + np.timedelta64(1, "D")).astype(str)
        c = estimate_cost(client, args.dataset, symbols, args.schema, day, nxt)
        total += c
        print(f"  {day}..{nxt}  ${c:.2f}")
    print(f"  TOTAL ESTIMATED COST = ${total:.2f}  ({len(dates)} day(s))")
    if not args.pull:
        print("  (estimate only; pass --pull to download)")
        return
    pull(client, args.name, args.dataset, symbols, args.schema, dates,
         Path(args.out_dir), args.align_freq_ms)
    print("DATABENTO_PULL_DONE")


if __name__ == "__main__":
    main()
