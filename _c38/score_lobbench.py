"""Run lob_bench scoring on the generated vs real BTC sequences (lob env)."""
import sys, json, os
import numpy as np
ROOT = "/DATA/DATANAS1/wpj24/AIQuant"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + "/lob_bench")

from lob_bench import data_loading, scoring, run_bench

OUT = os.environ.get("OUT", ROOT + "/DeepMarket/data/lobbench/DeepMarketMA/BTCUSDT/2024-01-01")

ASSET = next((a for a in ("BTCUSDT", "ETHUSDT", "INTC", "AAPL_A", "AAPL_B") if a in OUT), "?")
loader = data_loading.Simple_Loader(OUT + "/data_real", OUT + "/data_gen", OUT + "/data_cond")
print(f"[loader] {len(loader)} period(s)")
for s in loader:
    s.materialize()

scores, score_dfs, plot_fns = scoring.run_benchmark(
    loader,
    run_bench.DEFAULT_SCORING_CONFIG,
    default_metric=run_bench.DEFAULT_METRICS,
)

print(f"\n==== lob_bench divergence scores (gen vs real, {ASSET}) ====")
print(f"{'metric':28s} {'L1':>8s} {'L1_CI':>17s} {'Wass':>8s}")
print("-" * 66)
rows = {}
for k, v in scores.items():
    try:
        l1 = v['l1']; ws = v['wasserstein']
        l1p = float(l1[0]); ci = l1[1]; wsp = float(ws[0])
        rows[k] = {'l1': l1p, 'l1_ci': [float(ci[0]), float(ci[1])], 'wasserstein': wsp}
        print(f"{k:28s} {l1p:8.3f} [{float(ci[0]):6.3f},{float(ci[1]):6.3f}] {wsp:8.3f}")
    except Exception as e:
        rows[k] = {'l1': float('nan'), 'wasserstein': float('nan'), 'err': str(e)}
        print(f"{k:28s} <err: {e}>")

valid = [r['l1'] for r in rows.values() if r['l1'] == r['l1']]
mean_l1 = float(np.mean(valid)) if valid else float('nan')
valid_lt1 = [r['l1'] for r in rows.values() if r['l1'] == r['l1'] and r['l1'] < 1.0]
mean_l1_lt1 = float(np.mean(valid_lt1)) if valid_lt1 else float('nan')
wvalid = [r['wasserstein'] for r in rows.values() if r['wasserstein'] == r['wasserstein']]
mean_ws = float(np.mean(wvalid)) if wvalid else float('nan')
print("-" * 66)
print(f"{'MEAN(all)':28s} {mean_l1:8.3f} {'':17s} {mean_ws:8.3f}")
print(f"{'MEAN(L1<1,non-degen)':28s} {mean_l1_lt1:8.3f}")
rows['_mean'] = {'l1': mean_l1, 'l1_valid_lt1': mean_l1_lt1, 'wasserstein': mean_ws}

with open(OUT + "/lobbench_scores.json", "w") as f:
    json.dump(rows, f, indent=2, default=str)
print("\nSCORE_DONE ->", OUT + "/lobbench_scores.json")
