"""P4 posterior report driver.

The driver consumes held-out real LOB trajectories plus one or more generated
ablation-corner trajectories and emits a JSON-serializable report covering:

- Layer-3 spread posterior consistency.
- Realized cross-asset correlation matrix and CCF.
- Lead-lag error against the real CCF.
- Spread/arbitrage distribution metrics.
- Bidirectional conditional spillover.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
DEEPMARKET_ROOT = Path(__file__).resolve().parents[2]
for path in (str(REPO_ROOT), str(DEEPMARKET_ROOT)):
    if path not in sys.path:
        sys.path.append(path)

from lob_bench.cross_asset import (  # noqa: E402
    bidirectional_spillover,
    ccf,
    ccf_lags,
    compare_arbitrage_metrics,
    lead_lag_error,
    realized_corr,
    realized_corr_matrix,
)
from models.diffusers.multi_asset.arbitrage.consistency_check import (  # noqa: E402
    consistency_check,
    mid_prices_from_lob,
    reference_from_lob,
    spread_from_mid,
)
from preprocessing.AssetUniverse import AssetUniverse  # noqa: E402


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def returns_from_mid(mid: np.ndarray) -> np.ndarray:
    """Return log returns shaped ``(N, B*(T-1))`` from ``(B,T,N)`` mid."""

    arr = np.asarray(mid, dtype=float)
    if arr.ndim != 3:
        raise ValueError(f"mid must be (B,T,N), got {arr.shape}")
    log_mid = np.log(np.maximum(np.abs(arr), 1e-12))
    returns = np.diff(log_mid, axis=1)
    return np.transpose(returns, (2, 0, 1)).reshape(arr.shape[2], -1)


def stress_proxy_from_mid(mid: np.ndarray, num_groups: int) -> np.ndarray:
    """Simple realized-volatility proxy shaped like spread ``(B,T,G)``."""

    arr = np.asarray(mid, dtype=float)
    log_mid = np.log(np.maximum(np.abs(arr), 1e-12))
    returns = np.abs(np.diff(log_mid, axis=1)).mean(axis=-1)
    if returns.shape[1] == 0:
        stress = np.zeros(arr.shape[:2], dtype=float)
    else:
        stress = np.concatenate([returns[:, :1], returns], axis=1)
    return np.repeat(stress[:, :, None], num_groups, axis=2)


def cross_asset_report(real_mid, generated_mid, max_lag: int = 5, bar_size: int = 1) -> Dict[str, object]:
    """Compute P4 cross-asset metrics for one generated corner."""

    real_returns = returns_from_mid(real_mid)
    gen_returns = returns_from_mid(generated_mid)
    if real_returns.shape[0] < 2 or gen_returns.shape[0] < 2:
        raise ValueError("P4 cross-asset report requires at least two assets")

    real_pair_ccf = ccf(real_returns[0], real_returns[1], max_lag=max_lag, dt=bar_size)
    gen_pair_ccf = ccf(gen_returns[0], gen_returns[1], max_lag=max_lag, dt=bar_size)
    real_corr_value = realized_corr(real_returns[0], real_returns[1], dt=bar_size)
    gen_corr_value = realized_corr(gen_returns[0], gen_returns[1], dt=bar_size)
    return {
        "realized_corr": gen_corr_value,
        "realized_corr_abs_error": abs(gen_corr_value - real_corr_value)
        if np.isfinite(gen_corr_value) and np.isfinite(real_corr_value)
        else np.nan,
        "corr_matrix": realized_corr_matrix(gen_returns, dt=bar_size),
        "ccf_lags": ccf_lags(max_lag),
        "ccf": gen_pair_ccf,
        "lead_lag_error": lead_lag_error(real_pair_ccf, gen_pair_ccf, max_lag=max_lag),
        "spillover": bidirectional_spillover(gen_returns[0], gen_returns[1]),
    }


def real_reference_report(real_mid, max_lag: int = 5, bar_size: int = 1) -> Dict[str, object]:
    """Reference-only cross-asset values for held-out real data."""

    real_returns = returns_from_mid(real_mid)
    pair_ccf = ccf(real_returns[0], real_returns[1], max_lag=max_lag, dt=bar_size)
    return {
        "realized_corr": realized_corr(real_returns[0], real_returns[1], dt=bar_size),
        "corr_matrix": realized_corr_matrix(real_returns, dt=bar_size),
        "ccf_lags": ccf_lags(max_lag),
        "ccf": pair_ccf,
        "spillover": bidirectional_spillover(real_returns[0], real_returns[1]),
    }


def trend_checks(corners: Mapping[str, Mapping[str, object]]) -> Dict[str, object]:
    """Summarize the P4 exit-criterion trend checks when corners are present."""

    checks: Dict[str, object] = {
        "all_four_corners_present": all(name in corners for name in ("P0", "P1", "P2", "P3"))
    }
    if "P0" in corners and "P1" in corners:
        p0_err = corners["P0"]["cross_asset"]["realized_corr_abs_error"]
        p1_err = corners["P1"]["cross_asset"]["realized_corr_abs_error"]
        checks["p1_corr_closer_than_p0"] = bool(np.isfinite(p0_err) and np.isfinite(p1_err) and p1_err < p0_err)
    if "P1" in corners and "P2" in corners:
        p1_ks = corners["P1"]["arbitrage"]["ks_distance"]
        p2_ks = corners["P2"]["arbitrage"]["ks_distance"]
        checks["p2_spread_ks_below_p1"] = bool(np.isfinite(p1_ks) and np.isfinite(p2_ks) and p2_ks < p1_ks)
    if "P2" in corners and "P3" in corners:
        p2_tail = corners["P2"]["arbitrage"]["generated_excursions"]["p95_duration"]
        p3_tail = corners["P3"]["arbitrage"]["generated_excursions"]["p95_duration"]
        checks["p3_excursion_p95_below_p2"] = bool(np.isfinite(p2_tail) and np.isfinite(p3_tail) and p3_tail < p2_tail)
    return checks


def build_posterior_report(
    real_lob,
    generated_by_corner: Mapping[str, np.ndarray],
    asset_universe,
    reference_lob=None,
    delta_base=None,
    max_lag: int = 5,
    bar_size: int = 1,
    tolerances: Mapping[str, float] | None = None,
    mid_kwargs: Mapping[str, int] | None = None,
) -> Dict[str, object]:
    """Build the full P4 JSON-compatible report."""

    mid_kwargs = {} if mid_kwargs is None else dict(mid_kwargs)
    reference_lob = real_lob if reference_lob is None else reference_lob
    reference = reference_from_lob(
        reference_lob,
        asset_universe,
        delta_base=delta_base,
        tolerances=tolerances,
        **mid_kwargs,
    )
    real_mid = mid_prices_from_lob(real_lob, asset_universe=asset_universe, **mid_kwargs)
    real_spread = spread_from_mid(real_mid, asset_universe)
    real_stress = stress_proxy_from_mid(real_mid, num_groups=real_spread.shape[-1])

    corners: Dict[str, Dict[str, object]] = {}
    for name, generated_lob in generated_by_corner.items():
        gen_mid = mid_prices_from_lob(generated_lob, asset_universe=asset_universe, **mid_kwargs)
        gen_spread = spread_from_mid(gen_mid, asset_universe)
        gen_stress = stress_proxy_from_mid(gen_mid, num_groups=gen_spread.shape[-1])
        corners[str(name)] = {
            "posterior_consistency": consistency_check(
                generated_lob,
                asset_universe,
                reference,
                delta_base=reference.stats["delta_base"],
                **mid_kwargs,
            ),
            "cross_asset": cross_asset_report(
                real_mid,
                gen_mid,
                max_lag=max_lag,
                bar_size=bar_size,
            ),
            "arbitrage": compare_arbitrage_metrics(
                real_spread,
                gen_spread,
                delta_base=reference.stats["delta_base"],
                real_stress=real_stress,
                generated_stress=gen_stress,
                time_axis=1,
            ),
        }

    real_consistency = consistency_check(
        real_lob,
        asset_universe,
        reference,
        delta_base=reference.stats["delta_base"],
        **mid_kwargs,
    )
    report = {
        "reference": {
            "posterior_consistency": real_consistency,
            "cross_asset": real_reference_report(real_mid, max_lag=max_lag, bar_size=bar_size),
            "spread": {
                "delta_base": reference.stats["delta_base"],
                "shape": list(real_spread.shape),
            },
        },
        "corners": corners,
        "trend_checks": trend_checks(corners),
        "settings": {
            "max_lag": int(max_lag),
            "bar_size": int(bar_size),
            "asset_universe": {
                "assets": list(asset_universe.assets),
                "etf_basket_weights": {
                    str(k): {str(i): float(w) for i, w in v.items()}
                    for k, v in asset_universe.etf_basket_weights.items()
                },
            },
        },
    }
    return _json_safe(report)


def _load_npy(path: str) -> np.ndarray:
    return np.load(Path(path))


def _parse_corner_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--corner must be NAME=/path/to/lob.npy")
    name, path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("corner name cannot be empty")
    return name, Path(path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-lob", required=True, help="Held-out real LOB .npy")
    parser.add_argument("--reference-lob", default=None, help="Training/reference LOB .npy; defaults to --real-lob")
    parser.add_argument(
        "--corner",
        action="append",
        type=_parse_corner_arg,
        default=[],
        help="Generated corner as NAME=/path/to/generated_lob.npy. Repeat for P0/P1/P2/P3.",
    )
    parser.add_argument("--assets", nargs=2, default=["ETF", "CONST"], metavar=("ETF", "CONSTITUENT"))
    parser.add_argument("--basket-weight", type=float, default=1.0)
    parser.add_argument("--max-lag", type=int, default=5)
    parser.add_argument("--bar-size", type=int, default=1)
    parser.add_argument("--delta-base", type=float, default=None)
    parser.add_argument("--ask-price-col", type=int, default=0)
    parser.add_argument("--bid-price-col", type=int, default=2)
    parser.add_argument("--out", default=None, help="Optional output JSON path")
    args = parser.parse_args(argv)

    if not args.corner:
        raise SystemExit("at least one --corner NAME=path is required")

    universe = AssetUniverse(
        assets=list(args.assets),
        relation_types={(0, 1): 0, (1, 0): 1},
        etf_basket_weights={0: {1: float(args.basket_weight)}},
    )
    real_lob = _load_npy(args.real_lob)
    reference_lob = None if args.reference_lob is None else _load_npy(args.reference_lob)
    generated = {name: _load_npy(str(path)) for name, path in args.corner}
    report = build_posterior_report(
        real_lob,
        generated,
        universe,
        reference_lob=reference_lob,
        delta_base=args.delta_base,
        max_lag=args.max_lag,
        bar_size=args.bar_size,
        mid_kwargs={"ask_price_col": args.ask_price_col, "bid_price_col": args.bid_price_col},
    )
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.out is not None:
        Path(args.out).write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
