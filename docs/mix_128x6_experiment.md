# 标准容量 mix 对照

日期：2026-09-09（UTC）。状态：配置与启动入口已准备，未提交训练。

## 实验定义

目标：比较完整标准模型 128维／6层与现有 256维／12层 mix，判断
较大容量是否与较差的 T1x 泛化有关。用户明确选择同时恢复维度和层数。
这不是单独的宽度消融，也不是 T1x-only 论文复现。

基准代码：`25e4cac1029c794504e4b17b8e18ffdec9b99db9`。
本次未提交修改：新增 `Configs/Dynamics_mixed_128x6.yml`、
`run_scripts/run_mix_128x6.sh`、本记录，更新 README 和 STATUS。
开始时已有未跟踪的 `AGENTS.md`、`docs/STATUS.md`，予以保留。

| 设置 | 现有 mix | 新 mix |
| --- | --- | --- |
| hidden_dim / num_layers | 256 / 12 | 128 / 6 |
| batch size | 128 | 128 |
| 初始学习率 / noise | 5e-4 / 0.1 | 相同 |
| 最大 epochs / stop_tolerance | 2000 / 400 | 相同 |
| 初始化 | 随机 | 随机，不加载旧 checkpoint |
| 默认 seed | 2025 | 2025，可用 SEED 覆盖 |

其余 YAML 设置与 `Configs/Dynamics_mixed.yml` 一致，仅模型维度、层数、
输出根目录不同。沿用 RGD1 全量 cache（manifest 声明 176,898 条）+
T1x train（仓库协议 9,561 条）的自然比例混合；验证/测试仍为 T1x。
本次没有重建 cache 或更改 split。RGD1 CSV val/test 仍包含在全量 cache 中，
不能将它们作为独立 RGD1 holdout 报告。

## 启动与产物

在准备好的训练镜像/环境中，以项目根目录为工作目录运行：

```bash
DRY_RUN=1 bash run_scripts/run_mix_128x6.sh
SEED=2025 GPU_ID=0 bash run_scripts/run_mix_128x6.sh
```

脚本使用当前 `python`；可设置 `PYTHON_BIN=/path/to/python`。
不自动激活旧 `reactot` 环境。长训练使用平台提交式任务，以上第二条作为
作业执行命令；运行前核对平台现有任务、镜像、GPU 与数据路径。
本轮未选择镜像或提交作业，无任务 ID、训练指标或新 checkpoint。

产物：`logs/dynamics_flow_mixed_128x6/tsdfm_mix_128x6_seed2025_<timestamp>/`，
由主训练入口保存配置日志及 `checkpoints/checkpoint_best.pth`。
新 checkpoint 推理必须使用 `Configs/Dynamics_mixed_128x6.yml`。

## 比较方法

优先比较同 seed、相同训练设置的 mix；增加 seed 时两组保持配对。
不能把新模型的 raw mix 指标直接与旧模型的 T1x 微调结果视为容量对照。
按 T1x validation 选择 checkpoint，推理时两组使用一致 ODE 和重建参数，
报告严格 RMSD、reflection RMSD、DMAE，并观察 p90/离群样本。
历史 mix 使用过不同重建 restart 数，比较前应统一。

较小模型更好只能支持容量影响的假设。判断过拟合还需观察各自训练与验证
轨迹；当前 train 是含 noise 的 flow MSE，val 是 ODE 终点距离 L1，
不能用两者绝对值相减当作泛化间隙。仅此实验不能区分宽度与深度作用，
也不能确定 RGD1 本身是否有益；后者需要同架构 T1x-only 对照。

## 已执行验证（2026-09-09）

- `bash -n run_scripts/run_mix_128x6.sh`：通过。
- `DRY_RUN=1 bash run_scripts/run_mix_128x6.sh`：通过；命令指向新配置，
  seed 2025，无 init/resume 参数。
- Python/PyYAML 读取两份配置并递归比较：仅维度、层数、日志根目录不同。
- Python 调用 `build_dynamics_dataloaders`，mock mixed backend：正确选择
  mixed 路径，batch、seed 及全部 worker 选项正确透传。未实例化真实 loader。
- Python 路径检查：T1x HDF5 存在；RGD1 manifest 与 87 个 shard 文件存在。
  未加载 HDF5/shard 内容，未进行 GPU smoke test 或完整训练。
