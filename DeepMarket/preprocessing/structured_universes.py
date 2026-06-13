"""Registry of *economically structured* multi-asset universes (task bucket 4).

Buckets 1-3 ran on BTC+ETH, a pair with neither a dominant lead-lag relation
nor an arbitrage constraint, so the P1 graph (gamma self-collapsed to 0) and P3
no-arb energy (``spread_groups`` empty -> energy 0/NaN) had nothing to learn.
This module declares pairs/baskets that *do* have structure, and pins down both:

  1. the model-facing :class:`AssetUniverse` (fills ``etf_basket_weights`` ->
     ``spread_groups`` for P2/P3 and ``relation_types`` for the P1 graph), and
  2. the data-source ``legs`` (where each asset is fetched from) so the build
     driver and the lead-lag verifier share one source of truth.

Schemes (see ``NEXT_WORK_ZH.md`` 4.2):
  A  spot vs perpetual      -- zero cost (Tardis), strongest/cleanest lead-lag.
  B  same coin cross-venue  -- zero cost (Tardis), enables the N>2 graph.
  C  ETF vs constituents    -- Databento ($0-125), the textbook NAV no-arb case.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from preprocessing.AssetUniverse import AssetUniverse


@dataclass(frozen=True)
class LegSpec:
    """Where one asset of a structured universe is fetched from.

    ``source == "tardis"``   -> ``venue`` is a Tardis exchange id (``binance``,
    ``binance-futures``, ``coinbase``, ``okex`` ...) and ``symbol`` the venue's
    instrument id; the daily ``book_snapshot_25`` free sample is downloadable.
    ``source == "databento"`` -> ``venue`` is a Databento dataset (e.g.
    ``XNAS.ITCH``) and ``symbol`` the ticker; needs an API key + cost approval.
    """

    name: str            # canonical asset name (matches AssetUniverse.assets)
    source: str          # "tardis" | "databento"
    venue: str           # tardis exchange id, or databento dataset
    symbol: str          # venue-native instrument id
    weight: float = 1.0  # basket weight (scheme C); 1.0 for basis legs


@dataclass(frozen=True)
class StructuredSpec:
    scheme: str                 # "A" | "B" | "C"
    universe: AssetUniverse
    legs: List[LegSpec]
    notes: str = ""
    free: bool = True           # True if fetchable at zero cost (Tardis)


# --- Scheme A: spot vs perpetual (zero cost, strongest signal) ---------------

def _spot_perp(coin: str) -> StructuredSpec:
    spot, perp = f"{coin}_spot", f"{coin}_perp"
    return StructuredSpec(
        scheme="A",
        universe=AssetUniverse.basis_pair(reference=perp, leg=spot),
        legs=[
            LegSpec(perp, "tardis", "binance-futures", coin),
            LegSpec(spot, "tardis", "binance", coin),
        ],
        notes="perp leads spot (price discovery in derivatives); basis=perp-spot "
        "mean-reverts ~0 (funding/carry give it drift, so P3 delta is dynamic).",
    )


# --- Scheme B: same coin across venues (zero cost, exercises N>2 graph) -------

def _cross_venue(coin: str, legs: List[LegSpec]) -> StructuredSpec:
    return StructuredSpec(
        scheme="B",
        universe=AssetUniverse.cross_venue([l.name for l in legs]),
        legs=legs,
        notes="cross-venue price equality pinned by arbitrageurs; most-liquid "
        "venue tends to lead. lead-lag weaker/time-varying than spot/perp.",
    )


# --- Scheme C: ETF vs constituents (Databento, NAV no-arb) -------------------

def _etf_basket(
    etf: str,
    constituents: List[str],
    weights: List[float],
    dataset: str = "XNAS.ITCH",
) -> StructuredSpec:
    return StructuredSpec(
        scheme="C",
        universe=AssetUniverse.etf_basket(etf, constituents, weights),
        legs=[LegSpec(etf, "databento", dataset, etf)]
        + [
            LegSpec(c, "databento", dataset, c, weight=w)
            for c, w in zip(constituents, weights)
        ],
        notes="ETF mid ~ sum w_i * constituent mid (NAV); energy.py's "
        "relu(|ETF - basket| - delta)^2 is written for exactly this.",
        free=False,
    )


REGISTRY: Dict[str, StructuredSpec] = {
    # A
    "btc_spot_perp": _spot_perp("BTCUSDT"),
    "eth_spot_perp": _spot_perp("ETHUSDT"),
    # B  -- clean USDT-quoted venues (no cross-quote contamination): the basis
    # P_a - P_b is a true cross-exchange spread because both are BTC/USDT.
    "btc_cross_usdt": _cross_venue(
        "BTC",
        [
            LegSpec("BTC_binance", "tardis", "binance", "BTCUSDT"),
            LegSpec("BTC_okex", "tardis", "okex", "BTC-USDT"),
        ],
    ),
    "btc_cross_usdt3": _cross_venue(  # N=3 exercises the >2-asset graph
        "BTC",
        [
            LegSpec("BTC_binance", "tardis", "binance", "BTCUSDT"),
            LegSpec("BTC_okex", "tardis", "okex", "BTC-USDT"),
            LegSpec("BTC_bybit", "tardis", "bybit", "BTCUSDT"),
        ],
    ),
    # B (mixed-quote, kept for reference): coinbase is BTC/USD, so the basis vs
    # binance/okex (BTC/USDT) carries the slowly-drifting USDT/USD peg as a
    # large offset -> weak/biased. Use btc_cross_usdt for the clean experiment,
    # or quote-convert coinbase by USDT/USD before differencing.
    "btc_cross_exchange": _cross_venue(
        "BTC",
        [
            LegSpec("BTC_binance", "tardis", "binance", "BTCUSDT"),
            LegSpec("BTC_coinbase", "tardis", "coinbase", "BTC-USD"),
            LegSpec("BTC_okex", "tardis", "okex", "BTC-USDT"),
        ],
    ),
    # C  (QQQ top weights, approximate; refine with official holdings before use)
    "qqq_basket": _etf_basket(
        "QQQ",
        ["AAPL", "MSFT", "NVDA", "AMZN"],
        [0.09, 0.08, 0.08, 0.05],
    ),
}


def get(name: str) -> StructuredSpec:
    if name not in REGISTRY:
        raise KeyError(f"unknown structured universe {name!r}; have {sorted(REGISTRY)}")
    return REGISTRY[name]


def names() -> List[str]:
    return sorted(REGISTRY)
