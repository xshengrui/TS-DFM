#!/bin/bash

# Fine-tune a mixed RGD1 + Transition1x checkpoint on Transition1x only.
# The checkpoint initializes model weights only. Optimizer and scheduler state
# intentionally start fresh from the fine-tuning configuration.

source ~/.bashrc
conda activate reactot
set -euo pipefail

cd /inspire/qb-ilm/project/chemicalreaction/czxs25220150/projects/TS-DFM

GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-2025}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
CONFIG_FILE="${CONFIG_FILE:-Configs/Dynamics_t1x_finetune_from_mix.yml}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-logs/dynamics_flow_mixed/tsdfm_mix_fixed_seed2025_2026_08_08__14_58_47/checkpoints/checkpoint_best.pth}"
LOG_PREFIX="${LOG_PREFIX:-tsdfm_mix_to_t1x_finetune_seed${SEED}}"

if [[ ! -f "${INIT_CHECKPOINT}" ]]; then
  echo "Initial mixed checkpoint not found: ${INIT_CHECKPOINT}" >&2
  exit 2
fi

CUDA_VISIBLE_DEVICES="${GPU_ID}" python Scripts/train_flow_matching_dist_ts1x.py \
  --config_file "${CONFIG_FILE}" \
  --init_checkpoint "${INIT_CHECKPOINT}" \
  --log_prefix "${LOG_PREFIX}" \
  --device cuda \
  --seed "${SEED}" \
  --num_workers "${NUM_WORKERS}" \
  --prefetch_factor "${PREFETCH_FACTOR}"
