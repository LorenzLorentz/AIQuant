#!/bin/bash
# Task 2.2: time-channel upweighting A/B. Treatment = TIME_LOSS_W=5 on the MA
# no-graph BTC+ETH full model, 2 seeds. Baseline = existing ng_s{1234,42}_ms
# (TIME_LOSS_W unset == 1.0). Both GPUs in parallel.
set -u
cd /DATA/DATANAS1/wpj24/AIQuant/DeepMarket
source /DATA/DATANAS1/wpj24/miniconda3/etc/profile.d/conda.sh
conda activate deep_market
export MPLCONFIGDIR=/DATA/DATANAS1/wpj24/.mplcache TMPDIR=/tmp/wpj24_tmp
mkdir -p "$TMPDIR"
LOGD=/DATA/DATANAS1/wpj24/AIQuant_logs
COMMON="SMOKE=0 NDEV=1 LR=3e-4 EPOCHS=5 DISABLE_ARB_GUIDANCE=1 DISABLE_GRAPH=1 VAL_CHECK_INTERVAL=1.0 LIMIT_VAL_BATCHES=20 MODEL=MA_TRADES ASSETS=BTCUSDT,ETHUSDT"
TW=${TW:-5}

run(){ # $1=seed $2=gpu
  env $COMMON TIME_LOSS_W=$TW CUDA_VISIBLE_DEVICES=$2 SEED=$1 CKPT_SUFFIX=ng_timew${TW}_s$1 \
    python run_ma_c38.py > $LOGD/p2_ng_timew${TW}_s$1.log 2>&1
}
run 1234 0 & P0=$!
run 42   1 & P1=$!
wait $P0 $P1
echo TIMEW_TRAIN_DONE
