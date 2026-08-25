#!/bin/bash

# Run Transition1x test-set inference and evaluation for a mixed-data checkpoint.
# source ~/.bashrc
# conda activate reactot

cd /inspire/qb-ilm/project/chemicalreaction/czxs25220150/projects/TS-DFM

RUN_PREFIX="${RUN_PREFIX:-tsdfm_mix_fixed_seed2025}"
RUN_DIR="${RUN_DIR:-}"
if [ -z "${RUN_DIR}" ]; then
  RUN_DIR="$(ls -td "logs/dynamics_flow_mixed/${RUN_PREFIX}"_* 2>/dev/null | head -n 1)"
fi
if [ -z "${RUN_DIR}" ]; then
  echo "No mixed run directory found. Set RUN_DIR or RUN_PREFIX." >&2
  exit 1
fi
CHECKPOINT="${CHECKPOINT:-${RUN_DIR}/checkpoints/checkpoint_best.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_DIR}/test_xyz_t1x}"
HDF5="${HDF5:-Data/Transition1x.h5}"
CONFIG="${CONFIG:-Configs/Dynamics_mixed.yml}"
GPU_ID="${GPU_ID:-0}"
RESTARTS="${RESTARTS:-1}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" python -m Scripts.infer_ts1x_xyz \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --hdf5 "${HDF5}" \
  --output "${OUTPUT_DIR}" \
  --device cuda \
  --reconstruction-restarts "${RESTARTS}" \
  --overwrite

python -m Scripts.evaluate_ts1x_xyz \
  --hdf5 "${HDF5}" \
  --pred-dir "${OUTPUT_DIR}" \
  --output-csv "${OUTPUT_DIR}/metrics.csv"
