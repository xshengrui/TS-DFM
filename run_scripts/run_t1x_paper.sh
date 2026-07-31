#!/bin/bash

# Paper Transition1x TS-DFM reproduction.
# Set Configs/Dynamics.yml:data.path to the downloaded Transition1x.h5 first.
# Keep data.batch_size=32 for paper reproduction. Configs/Dynamics.yml defaults
# to dataloader workers tuned for 4090-class GPUs.
source ~/.bashrc
conda activate reactot

cd /inspire/qb-ilm/project/chemicalreaction/czxs25220150/projects/TS-DFM

export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"

python Scripts/train_flow_matching_dist_ts1x.py \
  --config_file Configs/Dynamics.yml \
  --log_prefix tsdfm_ts1x_paper \
  --device cuda
