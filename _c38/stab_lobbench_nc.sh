#!/bin/bash
# lob_bench AR rollout on the bucket-3 winner (nc = norm+cosine@1e-3, s1234).
# Canonical §3.4i protocol: 8 windows x 500 events, DDIM-10, BTC + ETH.
set -u
ROOT=/DATA/DATANAS1/wpj24/AIQuant
CK=$ROOT/DeepMarket/data/checkpoints/MA_TRADES/val_ema=0.897_epoch=4_MA_BTCUSDT_ETHUSDT_c38_nc_lr1e3_s1234.ckpt
source /DATA/DATANAS1/wpj24/miniconda3/etc/profile.d/conda.sh
export MPLCONFIGDIR=/DATA/DATANAS1/wpj24/.mpl HF_HOME=/DATA/DATANAS1/wpj24/.hf TMPDIR=/tmp/wpj24_tmp CUDA_VISIBLE_DEVICES=1
mkdir -p $TMPDIR
cd $ROOT/DeepMarket

for ASSET in BTCUSDT ETHUSDT; do
  OUT=$ROOT/DeepMarket/data/lobbench/ar_nc/$ASSET/2024-01-01
  echo "===== GEN $ASSET -> $OUT ====="
  conda activate deep_market
  CKPT=$CK OUT=$OUT ASSET=$ASSET N_EVENTS=500 WINDOWS=8 SAMPLING=DDIM DDIM_NSTEPS=10 SEED=1234 \
    python build_lobbench_eval_ar.py 2>&1 | tail -5
  echo "===== SCORE $ASSET ====="
  conda activate lob
  OUT=$OUT python $ROOT/lob_bench/score_lobbench.py 2>&1 | tail -40
done
echo "ALL_LOBBENCH_NC_DONE"
