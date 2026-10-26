# 多智能体进化强化学习

当前主方法为 **Carbon-Dividend CDA V3 + All-task Mean-HV-guided Evolutionary MADDPG**。
六个任务均参与偏好进化，并根据预测的平均超体积贡献选择下一代偏好。

## 目录结构

- `src/envs/`: 综合能源系统环境、设备模型、交易机制。
- `src/emomarl/`: 多智能体强化学习与进化优化组件。
- `scripts/`: 训练、数据预处理、快速自检入口。
- `configs/main_method.json`: 当前主方法及正式实验结果索引。
- `data/raw/`: 原始 CSV 数据。
- `data/processed/`: 预处理后的 `.npy` 数据。
- `outputs/runs/`: 训练结果。
- `outputs/figures/`: 数据图像。
- `tests/`: 轻量测试。

## 常用命令

使用当前项目验证过的解释器：

```powershell
D:\ANACONDA\envs\venv1\python.exe scripts\smoke_test.py
```

运行主方法的完整训练：

```powershell
D:\ANACONDA\envs\venv1\python.exe scripts\train.py
```

运行主方法的小规模链路测试：

```powershell
D:\ANACONDA\envs\venv1\python.exe scripts\train.py --generations 3 --warmup-episodes 1 --generation-episodes 1 --hidden-dim 32 --batch-size 64 --replay-size 2000 --update-every 32 --candidate-count 4 --save-dir outputs/runs/smoke_main
```

旧算法仍可通过显式参数运行，例如 MAPPO：

```powershell
D:\ANACONDA\envs\venv1\python.exe scripts\train.py --algorithm mappo --save_dir outputs/runs/legacy_mappo
```

重新生成处理后的数据：

```powershell
D:\ANACONDA\envs\venv1\python.exe scripts\prepare_data.py
```
