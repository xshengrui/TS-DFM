#!/bin/bash

# 1. 加载 conda 环境 (reactot)
# 注意：如果你的平台没有初始化 conda，可能需要 source /path/to/conda/bin/activate
source ~/.bashrc
conda activate reactot

# 2. 切换到项目目录 (假设脚本执行时默认在 home 目录，保险起见写全路径)
# 如果你的工作目录已经挂载好了，直接 cd 即可
cd /inspire/qb-ilm/project/chemicalreaction/czxs25220150/projects/TS-DFM

# 3. 运行你的 GPU 命令
# 注意：这里去掉了行尾的反斜杠，保证在脚本里能正常运行
python Scripts/train_flow_matching_dist_ts1x.py --config_file Configs/Dynamics.yml --log_prefix tsdfm_ts1x --device cuda