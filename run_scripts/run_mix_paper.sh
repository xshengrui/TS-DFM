#!/bin/bash

# Custom mixed RGD1 + Transition1x experiment.
# This is not the Transition1x-only setup reported in the main paper Table 1.
# Set Configs/Dynamics_mixed.yml data paths before running.

source ~/.bashrc
conda activate reactot

cd /inspire/qb-ilm/project/chemicalreaction/czxs25220150/projects/TS-DFM

GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-2025}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
LOG_PREFIX="${LOG_PREFIX:-tsdfm_mix_fixed}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" python Scripts/train_flow_matching_dist_ts1x.py \
  --config_file Configs/Dynamics_mixed.yml \
  --log_prefix "${LOG_PREFIX}_seed${SEED}" \
  --device cuda \
  --seed "${SEED}" \
  --num_workers "${NUM_WORKERS}" \
  --prefetch_factor "${PREFETCH_FACTOR}"
