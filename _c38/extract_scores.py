import pickle, json, numpy as np
OUT="/DATA/DATANAS1/wpj24/AIQuant/DeepMarket/data/lobbench/DeepMarketMA/BTCUSDT/2024-01-01"
scores=pickle.load(open(OUT+"/scores_raw.pkl","rb"))
rows={}
print(f"{'metric':28s} {'L1':>8s} {'L1_CI':>17s} {'Wass':>8s}")
print("-"*66)
for k,v in scores.items():
    l1=v['l1']; ws=v['wasserstein']
    l1p=float(l1[0]); ci=l1[1]; wsp=float(ws[0])
    rows[k]={'l1':l1p,'l1_ci':[float(ci[0]),float(ci[1])],'wasserstein':wsp}
    print(f"{k:28s} {l1p:8.3f} [{float(ci[0]):6.3f},{float(ci[1]):6.3f}] {wsp:8.3f}")
mean_l1=float(np.mean([r['l1'] for r in rows.values()]))
mean_ws=float(np.mean([r['wasserstein'] for r in rows.values()]))
print("-"*66)
print(f"{'MEAN':28s} {mean_l1:8.3f} {'':17s} {mean_ws:8.3f}")
rows['_mean']={'l1':mean_l1,'wasserstein':mean_ws}
json.dump(rows, open(OUT+"/lobbench_scores.json","w"), indent=2)
print("saved", OUT+"/lobbench_scores.json")
