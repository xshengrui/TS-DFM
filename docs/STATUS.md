# TS-DFM — 当前进度

更新时间：2026-09-09（UTC）。

## 当前目标

增加正常容量（128维／6层）的 RGD1 + T1x mix 训练，与现有
256维／12层 mix 比较容量对泛化的影响。用户明确选择完整标准模型。

## 本次完成

- 新增 `Configs/Dynamics_mixed_128x6.yml`，保持原 mix 的数据、batch 128、
  noise 0.1、学习率、scheduler 和停止规则，仅改变模型容量与输出根目录。
- 新增 `run_scripts/run_mix_128x6.sh`：随机初始化、默认 seed 2025，支持
  SEED/GPU_ID/PYTHON_BIN/NUM_WORKERS/PREFETCH_FACTOR/LOG_PREFIX 与 DRY_RUN。
- 产物根目录：`logs/dynamics_flow_mixed_128x6`，不复用旧模型 checkpoint。
- 实验定义、版本、数据协议、运行命令和解释边界见
  `docs/mix_128x6_experiment.md`；README 已增加入口。
- 保留开始时已有的未跟踪 `AGENTS.md`，更新已有未跟踪 STATUS。

## 验证（2026-09-09）

- `bash -n run_scripts/run_mix_128x6.sh` 通过。
- `DRY_RUN=1 bash run_scripts/run_mix_128x6.sh` 通过，正确指向新配置。
- Python/PyYAML 递归比较新旧配置通过：仅 hidden_dim、num_layers、
  train.save_path 不同。
- Python 调用 factory 并 mock mixed backend 通过：正确选择 mixed loader，
  全部数据加载参数正确透传；未执行真实 loader。
- Python 文件检查通过：T1x HDF5 存在；RGD1 manifest 声明 176,898 条，
  87 个 shard 文件均存在。未验证 shard 内容或 HDF5 数据。

## 运行边界与下一步

- 本轮未查询平台作业、未选择 GPU/镜像、未提交训练，无新实验指标。
- 启动前核实可用训练环境及平台已有作业，使用提交式任务，避免重复运行。
- 两组按相同 seed、训练与推理设置比较；验证集选择 checkpoint，最终报告
  严格 RMSD、reflection RMSD、DMAE 和尾部误差。
- 小模型表现更好不独立证明过拟合；train flow MSE 与 val 距离 L1 的
  绝对值不可直接作差。此实验同时改变宽度和深度。
- 全量 RGD1 cache 仍包含其 CSV val/test；本次为保持容量对照未改 split，
  不能将这些数据作为独立 RGD1 holdout。
