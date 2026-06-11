#!/bin/bash
# Canonical lob_bench AR confirmation for the P0 head-to-head (ETH), seed 1234:
# independent TRADES vs shared MA at full / p20 / p10 data. Generates via the
# MODEL-switchable AR builder then scores. Runs on one GPU (eval is light).
set -u
ROOTABS=/DATA/DATANAS1/wpj24/AIQuant/DeepMarket
cd $ROOTABS
source /DATA/DATANAS1/wpj24/miniconda3/etc/profile.d/conda.sh
export MPLCONFIGDIR=/DATA/DATANAS1/wpj24/.mplcache TMPDIR=/tmp/wpj24_tmp
mkdir -p "$TMPDIR"
GPU=${GPU:-0}
TR=data/checkpoints/TRADES
MA=data/checkpoints/MA_TRADES
pick(){ ls -t $1 2>/dev/null | head -1; }

gen_score(){ # $1=MODEL $2=ckpt $3=tag
  if [ -z "$2" ]; then echo "SKIP $3 (no ckpt)"; return; fi
  local OUT=$ROOTABS/data/lobbench/p0/${3}_ETHUSDT
  echo "===================== $3 ($1) $(basename $2) ====================="
  conda activate deep_market
  MODEL=$1 CKPT=$2 ASSET=ETHUSDT PEER=BTCUSDT N_EVENTS=500 WINDOWS=8 DDIM_NSTEPS=10 \
    SAMPLING=DDIM SEED=0 OUT=$OUT CUDA_VISIBLE_DEVICES=$GPU \
    python build_lobbench_eval_ar_p0.py 2>&1 | grep -E "GEN_DONE|type mix|Traceback|Error" | tail -3
  conda activate lob
  OUT=$OUT python /DATA/DATANAS1/wpj24/AIQuant/lob_bench/score_lobbench.py 2>&1 \
    | grep -E "MEAN|inter_arrival|spread|orderbook_imb|SCORE_DONE" | tail -8
}

S=${S:-1234}
gen_score TRADES    "$(pick "$TR/*trades_eth_full_s$S.ckpt")"  A_full_s$S
gen_score MA_TRADES "$(pick "$MA/*ng_s${S}_ms.ckpt")"          B_full_s$S
gen_score TRADES    "$(pick "$TR/*trades_eth_p20_s$S.ckpt")"   A_p20_s$S
gen_score MA_TRADES "$(pick "$MA/*ng_btc_ethp20_s$S.ckpt")"    B_p20_s$S
gen_score TRADES    "$(pick "$TR/*trades_eth_p10_s$S.ckpt")"   A_p10_s$S
gen_score MA_TRADES "$(pick "$MA/*ng_btc_ethp10_s$S.ckpt")"    B_p10_s$S
echo P0_LOBBENCH_DONE
