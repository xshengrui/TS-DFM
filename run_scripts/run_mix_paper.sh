#!/bin/bash

# Custom mixed RGD1 + Transition1x experiment.
# This is not the Transition1x-only setup reported in the main paper Table 1.
# Set Configs/Dynamics_mixed.yml data paths before running.
source ~/.bashrc
conda activate reactot

cd /inspire/qb-ilm/project/chemicalreaction/czxs25220150/projects/TS-DFM

python Scripts/train_flow_matching_dist_ts1x.py \
  --config_file Configs/Dynamics_mixed.yml \
  --log_prefix tsdfm_mix \
  --device cuda
