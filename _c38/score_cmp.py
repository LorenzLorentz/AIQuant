import sys, json, numpy as np, warnings
warnings.filterwarnings("ignore")
ROOT="/DATA/DATANAS1/wpj24/AIQuant"
sys.path.insert(0,ROOT); sys.path.insert(0,ROOT+"/lob_bench")
from lob_bench import data_loading, scoring, run_bench
CMP=ROOT+"/DeepMarket/data/lobbench/cmp"
TAGS=["floor","naive","nograph","graph"]
CFG=run_bench.DEFAULT_SCORING_CONFIG
# order-flow / placement metrics (book-only ones excluded; identical by design)
CURATED=["log_inter_arrival_time","limit_ask_order_depth","limit_bid_order_depth",
         "ask_cancellation_depth","bid_cancellation_depth","limit_ask_order_levels",
         "limit_bid_order_levels","ask_cancellation_levels","bid_cancellation_levels"]
res={t:{} for t in TAGS}
for tag in TAGS:
    d=f"{CMP}/{tag}"
    loader=data_loading.Simple_Loader(d+"/data_real",d+"/data_gen",d+"/data_cond")
    for s in loader: s.materialize()
    for m in CURATED:
        try:
            sc,_,_=scoring.run_benchmark(loader,{m:CFG[m]},default_metric=run_bench.DEFAULT_METRICS)
            res[tag][m]=float(sc[m]['l1'][0])
        except Exception as e:
            res[tag][m]=float('nan')
    print("scored",tag)
print("\n==== lob_bench L1 (order-flow; lower=closer to real; book-only metrics excluded) ====")
hdr=f"{'metric':26s}"+"".join(f"{t:>9s}" for t in TAGS); print(hdr); print("-"*len(hdr))
for m in CURATED:
    print(f"{m:26s}"+"".join((f"{res[t][m]:9.3f}" if res[t][m]==res[t][m] else f"{'nan':>9s}") for t in TAGS))
print("-"*len(hdr))
def mean_valid(t):
    vals=[res[t][m] for m in CURATED if res[t][m]==res[t][m] and res[t][m]<1.0]  # drop nan + degenerate 1.0
    return float(np.mean(vals)) if vals else float('nan')
mv={t:mean_valid(t) for t in TAGS}
print(f"{'MEAN(valid,<1.0)':26s}"+"".join(f"{mv[t]:9.3f}" for t in TAGS))
json.dump({"per_metric":res,"mean_valid":mv},open(CMP+"/cmp_scores.json","w"),indent=2)
print("\nsaved",CMP+"/cmp_scores.json")
