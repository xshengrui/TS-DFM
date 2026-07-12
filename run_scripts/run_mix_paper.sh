#!/bin/bash

# 1. 加载服务器上的 conda 环境。
source ~/.bashrc
conda activate reactot

# 2. 进入 TS-DFM 项目目录。
cd /inspire/qb-ilm/project/chemicalreaction/czxs25220150/projects/TS-DFM

# 3. 使用 RGD1 + Transition1x 混合数据配置启动正式训练。
python Scripts/train_flow_matching_dist_ts1x.py --config_file Configs/Dynamics_mixed.yml --log_prefix tsdfm_mix --device cuda
