import sys, numpy as np, pickle
ROOT="/DATA/DATANAS1/wpj24/AIQuant"
sys.path.insert(0,ROOT); sys.path.insert(0,ROOT+"/lob_bench")
from lob_bench import data_loading, scoring, run_bench
OUT=ROOT+"/DeepMarket/data/lobbench/DeepMarketMA/BTCUSDT/2024-01-01"
loader=data_loading.Simple_Loader(OUT+"/data_real",OUT+"/data_gen",OUT+"/data_cond")
for s in loader: s.materialize()
scores,score_dfs,_=scoring.run_benchmark(loader,run_bench.DEFAULT_SCORING_CONFIG,default_metric=run_bench.DEFAULT_METRICS)
k="log_inter_arrival_time"
print("TYPE:",type(scores[k]))
print("REPR:",repr(scores[k])[:600])
print("--- keys if dict ---")
if isinstance(scores[k],dict):
    for kk,vv in scores[k].items():
        print(kk, type(vv), repr(vv)[:200])
pickle.dump(scores, open(OUT+"/scores_raw.pkl","wb"))
print("saved raw")
