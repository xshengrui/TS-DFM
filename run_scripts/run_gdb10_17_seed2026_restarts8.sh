# #!/bin/bash
# set -euo pipefail

# # Infer GDB-10 and GDB-17 with the seed-2026 Transition1x checkpoint and
# # the robust 8-restart coordinate reconstruction used in the latest TS1x run.
# source ~/.bashrc
# conda activate reactot

cd /inspire/qb-ilm/project/chemicalreaction/czxs25220150/projects/TS-DFM

export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"

CONFIG="${CONFIG:-Configs/Dynamics.yml}"
CHECKPOINT="${CHECKPOINT:-logs/dynamics_flow/tsdfm_ts1x_paper_seed2026_2026_07_31__06_56_15/checkpoints/checkpoint_best.pth}"
OUTPUT="${OUTPUT:-logs/dynamics_flow/tsdfm_ts1x_paper_seed2026_2026_07_31__06_56_15/gdb10_17_xyz_restarts8}"
STEP_SIZE="${STEP_SIZE:-0.05}"
RESTARTS="${RESTARTS:-8}"
NOISE_SCALE="${NOISE_SCALE:-0.05}"
LBFGS_MAX_ITER="${LBFGS_MAX_ITER:-100}"
LBFGS_LR="${LBFGS_LR:-0.1}"
LIMIT="${LIMIT:-}"

cmd=(
  python -m Scripts.infer_reaction_archives_xyz
  --config "${CONFIG}"
  --checkpoint "${CHECKPOINT}"
  --archives
    Data/GDB-10-rxn_raw.tar.gz
    Data/GDB-17-rxn_raw.tar.gz
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
  cmd+=(--limit "${LIMIT}")
fi

"${cmd[@]}"
