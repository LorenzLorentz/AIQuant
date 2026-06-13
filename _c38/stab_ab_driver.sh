#!/bin/bash
# Task 3.2 stability A/B driver — 4 runs, 2 rounds on 2 A100s.
# R1: T(mean+cosine) vs B(baseline) both LR=1e-3 seed1234.
# R2: T seed42 vs A(mean-only) LR=1e-3 seed1234.
# All: MA_TRADES no-graph, BTC+ETH, 5ep, arb-guidance off.
set -u
source /DATA/DATANAS1/wpj24/miniconda3/etc/profile.d/conda.sh
conda activate deep_market
export MPLCONFIGDIR=/DATA/DATANAS1/wpj24/.mpl PIP_CACHE_DIR=/DATA/DATANAS1/wpj24/.pipcache HF_HOME=/DATA/DATANAS1/wpj24/.hf TMPDIR=/tmp/wpj24_tmp
mkdir -p $TMPDIR
LOGD=/DATA/DATANAS1/wpj24/AIQuant_logs
mkdir -p $LOGD
cd /DATA/DATANAS1/wpj24/AIQuant/DeepMarket

COMMON="SMOKE=0 NDEV=1 EPOCHS=5 ASSETS=BTCUSDT,ETHUSDT DISABLE_GRAPH=1 DISABLE_ARB_GUIDANCE=1"

run() {  # $1=gpu $2=tag $3=extra-env
  local gpu=$1 tag=$2 extra=$3
  echo "[launch] gpu=$gpu tag=$tag :: $extra"
  env CUDA_VISIBLE_DEVICES=$gpu $COMMON $extra CKPT_SUFFIX=$tag \
    setsid python run_ma_c38.py > $LOGD/stab_${tag}.log 2>&1 &
  echo $!
}

echo "===== ROUND 1 ====="
run 0 mc_lr1e3_s1234   "SEED=1234 LR=1e-3 MSE_REDUCE=mean LR_SCHED=cosine"
run 1 base_lr1e3_s1234 "SEED=1234 LR=1e-3 MSE_REDUCE=norm LR_SCHED=plateau"
wait
echo "===== ROUND 1 DONE ====="

echo "===== ROUND 2 ====="
run 0 mc_lr1e3_s42     "SEED=42 LR=1e-3 MSE_REDUCE=mean LR_SCHED=cosine"
run 1 mean_lr1e3_s1234 "SEED=1234 LR=1e-3 MSE_REDUCE=mean LR_SCHED=plateau"
wait
echo "===== ROUND 2 DONE ====="
echo "ALL_STAB_DONE"
