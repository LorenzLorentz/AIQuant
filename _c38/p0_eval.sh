#!/bin/bash
# Evaluate every P0 head-to-head checkpoint on the SAME real ETH test positions
# (fixed eval SEED) with eval_p0.py. Writes one JSON per ckpt + a summary table.
set -u
cd /DATA/DATANAS1/wpj24/AIQuant/DeepMarket
source /DATA/DATANAS1/wpj24/miniconda3/etc/profile.d/conda.sh
conda activate deep_market
export MPLCONFIGDIR=/DATA/DATANAS1/wpj24/.mplcache TMPDIR=/tmp/wpj24_tmp
mkdir -p "$TMPDIR" data/eval_p0
ESEED=2024            # fixes the test positions -> identical across all models
NPOS=${NPOS:-3000}
GPU=${GPU:-0}
TR=data/checkpoints/TRADES
MA=data/checkpoints/MA_TRADES

ev(){ # $1=model $2=ckpt $3=tag
  [ -z "$2" ] && { echo "SKIP $3 (no ckpt)"; return; }
  echo "=== EVAL $3  ($1)  $(basename $2) ==="
  CUDA_VISIBLE_DEVICES=$GPU MODEL=$1 ASSET=ETHUSDT PEER=BTCUSDT N_POS=$NPOS BATCH=512 \
    DDIM_NSTEPS=10 SAMPLING=DDIM SEED=$ESEED CKPT=$2 OUT=data/eval_p0/$3.json \
    python eval_p0.py 2>&1 | grep -E "RESULT|Traceback|Error" | tail -3
}
pick(){ ls -t $1 2>/dev/null | head -1; }

for s in 1234 42; do
  ev TRADES    "$(pick "$TR/*trades_eth_full_s$s.ckpt")"  A_full_s$s
  ev TRADES    "$(pick "$TR/*trades_eth_p20_s$s.ckpt")"   A_p20_s$s
  ev TRADES    "$(pick "$TR/*trades_eth_p10_s$s.ckpt")"   A_p10_s$s
  ev MA_TRADES "$(pick "$MA/*ng_s${s}_ms.ckpt")"          B_full_s$s
  ev MA_TRADES "$(pick "$MA/*ng_btc_ethp20_s$s.ckpt")"    B_p20_s$s
  ev MA_TRADES "$(pick "$MA/*ng_btc_ethp10_s$s.ckpt")"    B_p10_s$s
done

echo "===================== SUMMARY (cont_l1_z, etype_acc, l1_time_z) ====================="
python - <<'PY'
import json, glob, os
rows=[]
for p in sorted(glob.glob("data/eval_p0/*.json")):
    d=json.load(open(p))
    rows.append((os.path.basename(p)[:-5], d["cont_l1_z"], d["etype_acc"], d["l1_time_z"], d["l1_size_z"], d["l1_price_z"]))
print(f"{'tag':22s} {'cont_l1':>8s} {'etype_acc':>9s} {'l1_time':>8s} {'l1_size':>8s} {'l1_price':>8s}")
for r in rows:
    print(f"{r[0]:22s} {r[1]:8.4f} {r[2]:9.3f} {r[3]:8.4f} {r[4]:8.4f} {r[5]:8.4f}")
PY
echo P0_EVAL_DONE
