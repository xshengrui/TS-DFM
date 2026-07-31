#!/bin/bash

# Run multiple independent paper-setting TS-DFM trainings on one GPU.
# This keeps each run at batch_size=32; parallelism is only used to satisfy
# cluster utilization rules for this small model.
source ~/.bashrc
conda activate reactot

cd /inspire/qb-ilm/project/chemicalreaction/czxs25220150/projects/TS-DFM
mkdir -p logs

export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"

N_RUNS="${N_RUNS:-2}"
BASE_SEED="${BASE_SEED:-2025}"
NUM_WORKERS_PER_RUN="${NUM_WORKERS_PER_RUN:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"

pids=()
for ((i = 0; i < N_RUNS; i++)); do
  seed=$((BASE_SEED + i))
  prefix="tsdfm_ts1x_paper_seed${seed}"
  python Scripts/train_flow_matching_dist_ts1x.py \
    --config_file Configs/Dynamics.yml \
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
