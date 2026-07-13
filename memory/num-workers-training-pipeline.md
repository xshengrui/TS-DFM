---
name: num-workers-training-pipeline
description: num_workers 原理、训练流水线、GPU/CPU 瓶颈判断方法
metadata:
  type: reference
---

# num_workers 与训练流水线

## 训练循环的流水线

一次迭代分为 CPU 阶段和 GPU 阶段：

```
CPU 做的事:                     GPU 做的事:
  读 HDF5 / pickle               神经网络前向计算
  解析分子坐标                   loss 计算
  Kabsch 对齐（纯CPU）           backward 反向传播
  拼接 batch                     optimizer.step() 更新参数
  搬到 GPU
```

## num_workers 是什么

`num_workers` = 用几个 CPU 子进程提前准备下一批数据。

### num_workers=0（串行）

```
CPU: [加载0] → 空闲等GPU → [加载1] → 空闲等GPU → [加载2] → ...
GPU: 空闲等CPU → [计算0]  → 空闲等CPU → [计算1]  → 空闲等CPU → ...
```
**CPU 和 GPU 互相等待，轮流空闲。**

### num_workers=4（并行）

```
子进程1-4: [预加载1] [预加载2] [预加载3] [预加载4] ... (一直在后台)
主进程:    取batch0送GPU → 取batch1送GPU → 取batch2送GPU → ...
GPU:       [计算0]       → [计算1]       → [计算2]       → ... (永不停)
```
**GPU 永远在算，不等 CPU。**

## 加大 num_workers 的影响

| 好处 | 代价 |
|------|------|
| GPU 利用率 ↑ | 每个 worker 多占 ~500MB 内存 |
| 每 iter 耗时 ↓ | CPU 占用 ↑ |
| 训练加速 | 多进程同时读磁盘，I/O 可能成为瓶颈 |

## 如何判断瓶颈

在服务器上开两个终端：

```bash
nvidia-smi    # 看 GPU-Util
top           # 看 CPU 利用率
```

### 判断表

| 现象 | 结论 | 操作 |
|------|------|------|
| GPU-Util < 60%，CPU 空闲 | **GPU 在等 CPU 喂数据** | 加大 num_workers |
| GPU-Util > 90%，worker 进程 CPU 低 | **num_workers 够用** | 不动 |
| 显存快满了 | batch_size 到上限 | 不能加大 batch_size |
| 内存快满了 | num_workers 太多 | 减小 num_workers |

### 实际案例（TS-DFM mixed 训练，batch_size=64, num_workers=4）

```
GPU-Util: 100%     ← 满载，无瓶颈
显存:    47.9/48GB ← 97.7%，不能加 batch_size
Worker:  4个进程，CPU 各占 2-3%  ← 很闲，数据早准备好了

结论：当前配置已是最优，不需要改动。
```

## 服务器查看命令

```bash
# 安装 htop（如有权限）
sudo apt install htop -y   # Ubuntu/Debian
sudo yum install htop -y   # CentOS/RHEL
conda install -c conda-forge htop  # 无 sudo 时用 conda

# 或用系统自带 top
top          # 看 CPU、内存
nvidia-smi   # 看 GPU 利用率、显存、温度、功率
```

## 训练时间估算公式

```
1. 每 epoch 迭代次数 = 总数据量 ÷ batch_size
2. 单次迭代耗时 = 两条相邻日志的时间差 ÷ 间隔 iter 数
3. 每 epoch 耗时 = 迭代次数 × 单次耗时
4. 总时间 = 每 epoch 耗时 × 预期 epoch 数

示例（TS-DFM mixed）：
  186,459 条 ÷ 64 batch = 2,914 iter/epoch
  0.56 秒/iter × 2,914 ≈ 27 分钟/epoch
  27 min × 100~300 epochs ≈ 1.9~5.6 天
```

**Why:** 用户询问 num_workers 原理、训练流水线、GPU/CPU 瓶颈判断，需要一份系统笔记。

**How to apply:** 调试训练性能时参考判断表；修改配置后可用公式重新估算训练时间。