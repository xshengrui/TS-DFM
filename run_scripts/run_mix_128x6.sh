#!/usr/bin/env bash
# Run in a prepared training environment (or set PYTHON_BIN explicitly).
# Long training runs should use a qz submitted job.
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SEED="${SEED:-2025}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
LOG_PREFIX="${LOG_PREFIX:-tsdfm_mix_128x6}"

# No init_checkpoint/resume_status: train the smaller model from scratch.
command=(
  "${PYTHON_BIN}" Scripts/train_flow_matching_dist_ts1x.py
  --config_file Configs/Dynamics_mixed_128x6.yml
  --log_prefix "${LOG_PREFIX}_seed${SEED}"
  --device cuda
  --seed "${SEED}"
  --num_workers "${NUM_WORKERS}"
  --prefetch_factor "${PREFETCH_FACTOR}"
)

printf 'CUDA_VISIBLE_DEVICES=%q ' "${CUDA_VISIBLE_DEVICES}"
printf '%q ' "${command[@]}"
printf '\n'
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi
exec "${command[@]}"
