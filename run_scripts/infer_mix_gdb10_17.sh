#!/bin/bash
set -eo pipefail

# Infer GDB-10 and GDB-17 with a mixed RGD1 + Transition1x checkpoint.
if [[ "${CONDA_DEFAULT_ENV:-}" != "reactot" ]]; then
  source ~/.bashrc
  conda activate reactot
fi
set -u

cd /inspire/qb-ilm/project/chemicalreaction/czxs25220150/projects/TS-DFM

GPU_ID="${GPU_ID:-0}"
CONFIG="${CONFIG:-Configs/Dynamics_mixed.yml}"
RUN_PREFIX="${RUN_PREFIX:-tsdfm_mix_fixed_seed2025}"
RUN_DIR="${RUN_DIR:-}"
CHECKPOINT="${CHECKPOINT:-}"
GDB10_ARCHIVE="${GDB10_ARCHIVE:-Data/GDB-10-rxn_raw.tar.gz}"
GDB17_ARCHIVE="${GDB17_ARCHIVE:-Data/GDB-17-rxn_raw.tar.gz}"
STEP_SIZE="${STEP_SIZE:-0.05}"
RESTARTS="${RESTARTS:-8}"
NOISE_SCALE="${NOISE_SCALE:-0.05}"
LBFGS_MAX_ITER="${LBFGS_MAX_ITER:-100}"
LBFGS_LR="${LBFGS_LR:-0.1}"
LIMIT="${LIMIT:-}"
PARALLEL_ARCHIVES="${PARALLEL_ARCHIVES:-1}"

if [[ -z "${CHECKPOINT}" ]]; then
  if [[ -z "${RUN_DIR}" ]]; then
    RUN_DIR="$(ls -td "logs/dynamics_flow_mixed/${RUN_PREFIX}"_* 2>/dev/null | head -n 1 || true)"
  fi
  if [[ -z "${RUN_DIR}" ]]; then
    echo "No mixed run directory found. Set RUN_DIR, CHECKPOINT, or RUN_PREFIX." >&2
    exit 1
  fi
  CHECKPOINT="${RUN_DIR}/checkpoints/checkpoint_best.pth"
elif [[ -z "${RUN_DIR}" ]]; then
  checkpoint_dir="$(dirname "${CHECKPOINT}")"
  if [[ "$(basename "${checkpoint_dir}")" == "checkpoints" ]]; then
    RUN_DIR="$(dirname "${checkpoint_dir}")"
  fi
fi

OUTPUT="${OUTPUT:-${RUN_DIR:-logs/dynamics_flow_mixed}/gdb10_17_xyz_restarts${RESTARTS}}"

for required_file in "${CONFIG}" "${CHECKPOINT}" "${GDB10_ARCHIVE}" "${GDB17_ARCHIVE}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "Required file does not exist: ${required_file}" >&2
    exit 1
  fi
done

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

common_args=(
  --config "${CONFIG}"
  --checkpoint "${CHECKPOINT}"
  --output "${OUTPUT}"
  --device cuda
  --step-size "${STEP_SIZE}"
  --lbfgs-max-iter "${LBFGS_MAX_ITER}"
  --lbfgs-lr "${LBFGS_LR}"
  --reconstruction-restarts "${RESTARTS}"
  --reconstruction-noise-scale "${NOISE_SCALE}"
  --skip-existing
)

if [[ -n "${LIMIT}" ]]; then
  common_args+=(--limit "${LIMIT}")
fi

echo "Mixed checkpoint: ${CHECKPOINT}"
echo "Output root: ${OUTPUT}"

if [[ "${PARALLEL_ARCHIVES}" == "1" ]]; then
  python -m Scripts.infer_reaction_archives_xyz \
    "${common_args[@]}" --archives "${GDB10_ARCHIVE}" &
  pid_gdb10=$!
  python -m Scripts.infer_reaction_archives_xyz \
    "${common_args[@]}" --archives "${GDB17_ARCHIVE}" &
  pid_gdb17=$!

  status=0
  wait "${pid_gdb10}" || status=$?
  wait "${pid_gdb17}" || status=$?
  exit "${status}"
fi

python -m Scripts.infer_reaction_archives_xyz \
  "${common_args[@]}" \
  --archives "${GDB10_ARCHIVE}" "${GDB17_ARCHIVE}"
