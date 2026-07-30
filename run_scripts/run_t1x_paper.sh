#!/bin/bash
set -euo pipefail

# Paper Transition1x TS-DFM reproduction.
# Set Configs/Dynamics.yml:data.path to the downloaded Transition1x.h5 first.
source ~/.bashrc
conda activate reactot

cd /inspire/qb-ilm/project/chemicalreaction/czxs25220150/projects/TS-DFM

python Scripts/train_flow_matching_dist_ts1x.py \
  --config_file Configs/Dynamics.yml \
  --log_prefix tsdfm_ts1x \
  --device cuda
