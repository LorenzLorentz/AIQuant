#!/bin/bash
# Task 3.2 — eval_p0 (teacher-forced predictive L1, reduction-invariant) on the
# 4 stability ckpts + the established LR=3e-4 norm baseline anchor. Scores ETH
# (target, BTC peer) and BTC (target, ETH peer). Prints RESULT json lines.
set -u
source /DATA/DATANAS1/wpj24/miniconda3/etc/profile.d/conda.sh
conda activate deep_market
export MPLCONFIGDIR=/DATA/DATANAS1/wpj24/.mpl PIP_CACHE_DIR=/DATA/DATANAS1/wpj24/.pipcache HF_HOME=/DATA/DATANAS1/wpj24/.hf TMPDIR=/tmp/wpj24_tmp
mkdir -p $TMPDIR
CKDIR=/DATA/DATANAS1/wpj24/AIQuant/DeepMarket/data/checkpoints/MA_TRADES
cd /DATA/DATANAS1/wpj24/AIQuant/DeepMarket

eval_ckpt() {  # $1=label $2=glob-suffix
  local label=$1 suf=$2
  local ckpt=$(ls -t $CKDIR/*_c38_${suf}.ckpt 2>/dev/null | head -1)
  if [ -z "$ckpt" ]; then echo "MISSING $label ($suf)"; return; fi
  echo "### $label :: $(basename $ckpt)"
  for pair in "ETHUSDT BTCUSDT" "BTCUSDT ETHUSDT"; do
    set -- $pair
    CKPT=$ckpt MODEL=MA_TRADES ASSET=$1 PEER=$2 SEED=0 N_POS=3000 DDIM_NSTEPS=10 \
      python eval_p0.py 2>/dev/null | grep "^RESULT"
  done
}

eval_ckpt "T_mean+cosine_s1234" mc_lr1e3_s1234
eval_ckpt "T_mean+cosine_s42"   mc_lr1e3_s42
eval_ckpt "A_mean-only_s1234"   mean_lr1e3_s1234
eval_ckpt "B_baseline_s1234"    base_lr1e3_s1234
# anchor: established LR=3e-4 norm no-graph baseline (memory: ETH cont_l1~0.196)
ANCHOR=$(ls -t $CKDIR/*ng_s1234*lr3e4*e5*.ckpt $CKDIR/*ng_s1234_lr3e4*.ckpt 2>/dev/null | head -1)
if [ -n "$ANCHOR" ]; then
  echo "### ANCHOR_lr3e4_norm_s1234 :: $(basename $ANCHOR)"
  for pair in "ETHUSDT BTCUSDT" "BTCUSDT ETHUSDT"; do
    set -- $pair
    CKPT=$ANCHOR MODEL=MA_TRADES ASSET=$1 PEER=$2 SEED=0 N_POS=3000 DDIM_NSTEPS=10 \
      python eval_p0.py 2>/dev/null | grep "^RESULT"
  done
else echo "ANCHOR ckpt not found"; fi
echo "EVAL_DONE"
