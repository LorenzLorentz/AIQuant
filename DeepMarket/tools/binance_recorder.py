"""Record Binance partial-book depth (20 levels @ 100ms) to adapter-ready CSV.

Each symbol gets one CSV whose columns match the IEX/Databento snapshot adapter
schema (``ts_event, ask_px_00.., ask_sz_00.., bid_px_00.., bid_sz_00..``), so the
output feeds straight into ``equity_file_to_npy`` / ``equity_file_to_splits`` with
no renaming. Record a correlated basket over time to build a multi-asset cluster.

Usage (run wherever you have Binance access; spot or USDT-margined futures):

    python tools/binance_recorder.py --symbols BTCUSDT,ETHUSDT,SOLUSDT \
        --out data/crypto_raw --market futures --hours 12

Then convert, e.g.:

    from preprocessing.adapters import equity_file_to_splits
    equity_file_to_splits("data/crypto_raw/BTCUSDT.csv", "data/crypto/BTCUSDT")

Stop anytime with Ctrl-C; CSVs are flushed continuously and remain valid.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import signal
import time
from pathlib import Path

import websockets  # pip install websockets

LEVELS = 20
HOSTS = {
    "spot": "wss://stream.binance.com:9443/stream?streams=",
    "futures": "wss://fstream.binance.com/stream?streams=",
}


def _header() -> list[str]:
    cols = ["ts_event"]
    for i in range(LEVELS):
        cols += [f"ask_px_{i:02d}", f"ask_sz_{i:02d}", f"bid_px_{i:02d}", f"bid_sz_{i:02d}"]
    return cols


def _row(data: dict) -> list:
    # Partial-book payload: spot uses bids/asks (no time); futures uses b/a + E/T.
    asks = data.get("a") or data.get("asks") or []
    bids = data.get("b") or data.get("bids") or []
    ts = data.get("E") or data.get("T") or int(time.time() * 1000)
    row = [ts]
    for i in range(LEVELS):
        ap, asz = (asks[i][0], asks[i][1]) if i < len(asks) else ("", "")
        bp, bsz = (bids[i][0], bids[i][1]) if i < len(bids) else ("", "")
        row += [ap, asz, bp, bsz]
    return row


async def record(symbols: list[str], out: Path, market: str, deadline: float | None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    writers, files = {}, {}
    for s in symbols:
        f = open(out / f"{s.upper()}.csv", "a", newline="")
        w = csv.writer(f)
        if f.tell() == 0:
            w.writerow(_header())
        files[s.upper()], writers[s.upper()] = f, w

    streams = "/".join(f"{s.lower()}@depth{LEVELS}@100ms" for s in symbols)
    url = HOSTS[market] + streams
    counts = {s.upper(): 0 for s in symbols}
    try:
        while deadline is None or time.time() < deadline:
            try:
                async with websockets.connect(url, ping_interval=15, max_queue=None) as ws:
                    print(f"connected: {market} {symbols}")
                    async for raw in ws:
                        msg = json.loads(raw)
                        stream = msg.get("stream", "")
                        data = msg.get("data", msg)
                        sym = stream.split("@", 1)[0].upper()
                        if sym in writers:
                            writers[sym].writerow(_row(data))
                            counts[sym] += 1
                            if counts[sym] % 200 == 0:
                                files[sym].flush()
                        if deadline is not None and time.time() >= deadline:
                            break
            except Exception as e:  # network drop -> reconnect
                print(f"reconnect after: {type(e).__name__}: {str(e)[:120]}")
                await asyncio.sleep(2)
    finally:
        for f in files.values():
            f.flush(); f.close()
        print("rows written:", counts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", required=True, help="comma-separated, e.g. BTCUSDT,ETHUSDT")
    ap.add_argument("--out", default="data/crypto_raw")
    ap.add_argument("--market", choices=["spot", "futures"], default="futures")
    ap.add_argument("--hours", type=float, default=None, help="stop after N hours (default: run until Ctrl-C)")
    ap.add_argument("--seconds", type=float, default=None, help="stop after N seconds (overrides --hours; for smoke tests)")
    a = ap.parse_args()

    dur = a.seconds if a.seconds is not None else (a.hours * 3600 if a.hours else None)
    deadline = (time.time() + dur) if dur else None
    symbols = [s.strip() for s in a.symbols.split(",") if s.strip()]

    loop = asyncio.new_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, loop.stop)
        except NotImplementedError:
            pass
    loop.run_until_complete(record(symbols, Path(a.out), a.market, deadline))


if __name__ == "__main__":
    main()
