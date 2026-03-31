# AIPSA-GM

**Adaptive Islanded Parallel Simulated Annealing with Guided Migration**

EE/CSCI 451 Course Project — Pengxiang Li, Weilun Qiu, Jinkuo Ha

---

## 项目结构

```
451-project/
├── problems/
│   ├── base.py              # 抽象基类，定义统一问题接口
│   ├── tsp.py               # TSP 旅行商问题
│   └── rastrigin.py         # Rastrigin 连续优化函数
├── solvers/
│   ├── serial_sa.py         # 串行 SA 基准
│   ├── baseline_a.py        # Baseline A：独立并行（无通信）
│   ├── baseline_b.py        # Baseline B：同步岛屿（有 barrier）
│   └── aipsa_gm.py          # AIPSA-GM：我们的算法
├── experiments/
│   └── run_experiment.py    # 统一实验入口脚本
├── utils/
│   └── logger.py            # 日志工具
└── results/                 # 实验结果 CSV（自动生成）
```

---

## 环境配置

### 安装依赖

```bash
pip install numpy tqdm
```

### Python 版本

Python 3.12（Windows 用 `python3.exe` 或 `py`）

---

## 快速开始

### 验证四个 solver 能跑通

```bash
python3.exe -m solvers.serial_sa
python3.exe -m solvers.baseline_a
python3.exe -m solvers.baseline_b
python3.exe -m solvers.aipsa_gm
```

---

## 实验脚本使用说明

所有实验统一通过 `experiments/run_experiment.py` 运行。

### 基础用法

```bash
# TSP 1000城市，跑3次取均值（默认）
python3.exe -m experiments.run_experiment

# TSP 指定城市数量
python3.exe -m experiments.run_experiment --cities 200
python3.exe -m experiments.run_experiment --cities 500
python3.exe -m experiments.run_experiment --cities 1000
python3.exe -m experiments.run_experiment --cities 2000

# Rastrigin 函数
python3.exe -m experiments.run_experiment --problem rastrigin

# 指定运行次数
python3.exe -m experiments.run_experiment --runs 5
python3.exe -m experiments.run_experiment --runs 20

# 指定 island 数量
python3.exe -m experiments.run_experiment --islands 8

# 完整参数示例
python3.exe -m experiments.run_experiment --problem tsp --cities 1000 --runs 5 --islands 4
```

### Scalability 实验（自动跑 2/4/8/16 islands）

```bash
# TSP 1000城市，每个配置跑3次
python3.exe -m experiments.run_experiment --scale --cities 1000 --runs 3

# Rastrigin
python3.exe -m experiments.run_experiment --scale --problem rastrigin --runs 3
```

结果自动保存到 `results/tsp_1000cities_scalability.csv`

### 所有参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--problem` | `tsp` | 问题类型：`tsp` 或 `rastrigin` |
| `--cities` | `1000` | TSP 城市数量 |
| `--dims` | `10` | Rastrigin 维度数 |
| `--runs` | `3` | 独立运行次数（取均值） |
| `--seed` | `42` | 随机种子基准值 |
| `--islands` | `4` | 并行 island 数量 |
| `--scale` | 关 | 开启 scalability 模式 |

---

## 当前最优配置（已验证）

`run_experiment.py` 中的关键参数：

```python
TOPOLOGY           = 'full'    # full > ring > random_k
MIGRATION_INTERVAL = 300
ADAPTIVE_HEAT_TSP        = False   # TSP 用单调冷却
ADAPTIVE_HEAT_RASTRIGIN  = True    # Rastrigin 用自适应温度
```

`aipsa_gm.py` 中的关键参数：

```python
best_utility > 0.3    # diversity injection 阈值（原来是 0.5）
alpha_w = 0.5         # quality 权重
beta_w  = 0.5         # diversity 权重
```

---

## 已有实验结果

### TSP 不同规模对比（4 islands, full topology, 5 runs）

| 城市数 | Serial SA | Baseline A | Baseline B | **AIPSA-GM** |
|--------|-----------|------------|------------|-------------|
| 200 | 11539 | 11392 | 11536 | 11703 |
| 1000 | 61657 | 60496 | 57766 | **40694** |
| 2000 | 163122 | 161896 | 155223 | **110216** |

> 规模越大，AIPSA-GM 优势越明显（1000城市好 34%，2000城市好 32%）

### Rastrigin 10维（50 runs，全局最优 = 0）

| Solver | Mean Cost | Best Cost |
|--------|-----------|-----------|
| Serial SA | 19.00 | 9.16 |
| Baseline A | 13.93 | 5.36 |
| Baseline B | 7.33 | 2.98 |
| **AIPSA-GM** | **4.87** | **2.09** |

### Scalability 实验（TSP 1000城市，3 runs）

| n_islands | Baseline B Cost | Baseline B Time | AIPSA-GM Cost | AIPSA-GM Time |
|-----------|----------------|-----------------|---------------|---------------|
| 2 | 35023 | 23.7s | 40005 | 13.5s |
| 4 | 33525 | 27.0s | 35338 | 14.8s |
| 8 | 32069 | 32.8s | 33655 | 18.0s |
| 16 | 30973 | **45.5s** | 32585 | **24.2s** |

> island 增加时，Baseline B 时间急剧增长（barrier 开销），AIPSA-GM 时间增长平缓

### Topology 对比（TSP 1000城市，4 islands，3 runs）

| Topology | AIPSA-GM Mean Cost | Time |
|----------|-------------------|------|
| ring | 35338 | 14.8s |
| random_k | 35607 | 15.2s |
| **full** | **34278** | **15.1s** |

---

## 还需要完成的实验

根据 proposal，以下实验还未完成：

- [ ] **Experiment 2**：Migration policy 对比（random / best-only / quality-only / quality+diversity）
- [ ] **Experiment 3**：Adaptive vs Fixed cooling 专项对比
- [ ] **Experiment 6**：GPU 版本开发与对比
- [ ] 画图可视化（所有对比图）

---

## 注意事项

### Windows 多进程
所有多进程代码必须在 `if __name__ == '__main__':` 保护下运行，否则 Windows 会报错。实验脚本已处理好，直接用 `-m` 方式运行即可。

### manager.Queue() vs mp.Queue()
Windows 上 `mp.Queue().empty()` 会挂死，项目中统一使用 `manager.Queue()` + `get_nowait()` 方式。

### manager.Barrier() vs mp.Barrier()
`mp.Barrier` 无法被 pickle 传给 `mp.Pool`，项目中统一使用 `manager.Barrier()`。

### 运行时间参考（i9-12900K）
| 实验 | 预计时间 |
|------|---------|
| TSP 200城市 1次 | ~10s |
| TSP 1000城市 1次 | ~35s |
| TSP 2000城市 1次 | ~80s |
| Rastrigin 1次 | ~8s |
| Scalability (2/4/8/16) 3次 | ~20min |

---

## 算法说明

### AIPSA-GM 核心特性

**1. 自适应温度控制**
```
接受率 < r_low  → 升温 T *= 1.1（仅 Rastrigin）
接受率 > r_high → 降温 T *= 0.9
否则            → 标准冷却 T *= alpha
```

**2. Guided Migration（引导迁移）**
```
U(x, i) = α × q(x) + β × d(x, i)
q(x) = 解的质量（归一化）
d(x, i) = 解与目标 island 的多样性距离
```

**3. 异步通信**
- 无全局 barrier
- 每隔 migration_interval 步推送最优解
- 非阻塞接收，计算与通信重叠

### 四个 Solver 对比

| | Serial SA | Baseline A | Baseline B | AIPSA-GM |
|--|-----------|------------|------------|----------|
| 并行 | ❌ | ✅ | ✅ | ✅ |
| 通信 | ❌ | ❌ | ✅ 同步 | ✅ 异步 |
| Barrier | - | - | ✅ 有 | ❌ 无 |
| 自适应温度 | ❌ | ❌ | ❌ | ✅ |
| Guided Migration | ❌ | ❌ | ❌ | ✅ |
