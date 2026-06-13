#!/bin/bash
# Task 3.2 round 3 — disambiguator/payoff: norm + cosine (isolate the good lever
# from the harmful mean reduction). LR=1e-3, seeds 1234 + 42, 2 A100s.
set -u
source /DATA/DATANAS1/wpj24/miniconda3/etc/profile.d/conda.sh
conda activate deep_market
export MPLCONFIGDIR=/DATA/DATANAS1/wpj24/.mpl PIP_CACHE_DIR=/DATA/DATANAS1/wpj24/.pipcache HF_HOME=/DATA/DATANAS1/wpj24/.hf TMPDIR=/tmp/wpj24_tmp
mkdir -p $TMPDIR
LOGD=/DATA/DATANAS1/wpj24/AIQuant_logs
cd /DATA/DATANAS1/wpj24/AIQuant/DeepMarket
COMMON="SMOKE=0 NDEV=1 EPOCHS=5 ASSETS=BTCUSDT,ETHUSDT DISABLE_GRAPH=1 DISABLE_ARB_GUIDANCE=1 MSE_REDUCE=norm LR_SCHED=cosine LR=1e-3"
run() {
  local gpu=$1 tag=$2 seed=$3
  echo "[launch] gpu=$gpu tag=$tag seed=$seed"
  env CUDA_VISIBLE_DEVICES=$gpu $COMMON SEED=$seed CKPT_SUFFIX=$tag \
    setsid python run_ma_c38.py > $LOGD/stab_${tag}.log 2>&1 &
}
echo "===== ROUND 3 ====="
run 0 nc_lr1e3_s1234 1234
run 1 nc_lr1e3_s42   42
wait
echo "ALL_R3_DONE"
