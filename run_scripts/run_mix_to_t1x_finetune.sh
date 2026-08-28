#!/bin/bash

# Fine-tune a mixed RGD1 + Transition1x checkpoint on Transition1x only.
# The checkpoint initializes model weights only. Optimizer and scheduler state
# intentionally start fresh from the fine-tuning configuration.

source ~/.bashrc
conda activate reactot
set -euo pipefail

cd /inspire/qb-ilm/project/chemicalreaction/czxs25220150/projects/TS-DFM

export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"

N_RUNS="${N_RUNS:-2}"
BASE_SEED="${BASE_SEED:-${SEED:-2025}}"
NUM_WORKERS_PER_RUN="${NUM_WORKERS_PER_RUN:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
CONFIG_FILE="${CONFIG_FILE:-Configs/Dynamics_t1x_finetune_from_mix.yml}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-logs/dynamics_flow_mixed/tsdfm_mix_fixed_seed2025_2026_08_08__14_58_47/checkpoints/checkpoint_best.pth}"
LOG_PREFIX_BASE="${LOG_PREFIX_BASE:-${LOG_PREFIX:-tsdfm_mix_to_t1x_finetune}}"

if [[ ! -f "${INIT_CHECKPOINT}" ]]; then
  echo "Initial mixed checkpoint not found: ${INIT_CHECKPOINT}" >&2
  exit 2
fi

mkdir -p logs

pids=()
for ((i = 0; i < N_RUNS; i++)); do
  seed=$((BASE_SEED + i))
  prefix="${LOG_PREFIX_BASE}_seed${seed}"
  python Scripts/train_flow_matching_dist_ts1x.py \
    --config_file "${CONFIG_FILE}" \
    --init_checkpoint "${INIT_CHECKPOINT}" \
    --log_prefix "${prefix}" \
    --device cuda \
    --seed "${seed}" \
    --num_workers "${NUM_WORKERS_PER_RUN}" \
    --prefetch_factor "${PREFETCH_FACTOR}" \
    > "logs/${prefix}.stdout.log" 2>&1 &
  pids+=("$!")
  sleep 20
done

exit_code=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    exit_code=1
  fi
done

exit "$exit_code"
