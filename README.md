# AIPSA-GM: Adaptive Islanded Parallel Simulated Annealing with Guided Migration

A parallel simulated annealing (SA) framework that combines adaptive temperature control, guided migration with a quality + diversity utility, asynchronous buffered communication, and topology-aware island models, plus a CUDA/CuPy GPU multi-chain solver.

## TL;DR

- **Island-model SA (CPU):** On TSP (1,000 cities, 4 islands), AIPSA-GM reduces mean tour cost by **37.4%** vs. serial SA and **13.0%** vs. synchronous islands. On Rastrigin (10-D) the reduction is **79.7%** vs. serial SA.
- **GPU multi-chain SA (RTX 4090):** Under an equal 60 s wall-clock budget, the GPU solver reaches up to **67.4% lower cost** on Rastrigin (10-D) and **17.6% lower cost** on TSP (10,000 cities) than a single-chain CPU SA. On Rastrigin 10-D it executes **2,114× more aggregate iterations** (summed over 1,024 parallel chains) in the same time.
- **Design insight:** Benefits are problem-dependent. Adaptive reheating helps multimodal continuous problems (Rastrigin: −47% mean cost) but hurts TSP; full topology is best for TSP while ring is best for Rastrigin; GPU gain shrinks with dimensionality on Rastrigin but grows with city count on TSP.

## Overview

We evaluate AIPSA-GM against three baselines on two benchmarks (TSP and Rastrigin), testing five hypotheses about parallelism, migration strategy, adaptive temperature, synchronization / communication topology, and GPU acceleration.

## Contributions

**Pengxiang (Alex) Li** ([@pengxiangli666](https://github.com/pengxiangli666)) designed and implemented the entire codebase: the AIPSA-GM algorithm (guided migration, adaptive heating, async buffered communication), the baseline solvers, the TSP and Rastrigin benchmarks, the experiment runner, and the CUDA/CuPy GPU kernels, including the chain-count sweep and GPU experiment design.

Teammates helped collect experimental data and wrote the course project report.

---

## Project Structure

```
AIPSA-GM/
├── experiments/
│   ├── __init__.py
│   ├── run_experiment.py      # Unified experiment runner (Exp 1–5)
│   ├── run_gpu_final.py       # Rastrigin GPU experiment runner (Exp 6)
│   └── run_gpu_tsp.py         # TSP GPU experiment runner (Exp 6)
├── problems/
│   ├── __init__.py
│   ├── base.py                # Unified Problem interface
│   ├── rastrigin.py           # Rastrigin function benchmark
│   └── tsp.py                 # TSP benchmark
├── solvers/
│   ├── __init__.py
│   ├── serial_sa.py           # Serial SA baseline
│   ├── baseline_a.py          # Baseline A: Independent Replicas
│   ├── baseline_b.py          # Baseline B: Synchronous Islands
│   ├── aipsa_gm.py            # AIPSA-GM (Baseline C)
│   ├── gpu_sa_final.py        # Rastrigin CUDA kernel (multi-chain SA)
│   └── gpu_tsp_kernel.py      # TSP CUDA kernel (parallel 2-opt SA chains)
├── utils/
│   ├── __init__.py
│   └── logger.py              # SA logging utility
├── results/                   # CSV outputs from experiments
├── run_tsp_chain_sweep.py     # N_CHAINS optimization sweep (GPU)
├── README.md
└── requirements.txt
```

---

## Quick Start

### Requirements

```bash
pip install numpy tqdm
```

(GPU experiments additionally need CuPy and CUDA. See [GPU Requirements](#gpu-requirements).)

### Run All Experiments

```bash
# ─── Experiment 1: Solver Comparison (TSP) ───────────────────
python -m experiments.run_experiment --problem tsp --cities 1000 --runs 10 --outdir results/exp1

# ─── Experiment 1: Solver Comparison (Rastrigin) ─────────────
python -m experiments.run_experiment --problem rastrigin --dims 10 --runs 10 --outdir results/exp1

# ─── Experiment 2: Migration Policy (TSP) ────────────────────
python -m experiments.run_experiment --exp2 --problem tsp --cities 2000 --runs 10 --outdir results/exp2

# ─── Experiment 2: Migration Policy (Rastrigin 10d, ring) ────
# Set TOPOLOGY = 'ring' first
python -m experiments.run_experiment --exp2 --problem rastrigin --dims 10 --runs 10 --outdir results/exp2

# ─── Experiment 2: Migration Policy (Rastrigin 30d, ring) ────
python -m experiments.run_experiment --exp2 --problem rastrigin --dims 30 --runs 10 --outdir results/exp2

# ─── Experiment 3: Adaptive Temperature (Rastrigin) ──────────
# Set TOPOLOGY = 'full' first
python -m experiments.run_experiment --exp3 --problem rastrigin --dims 10 --runs 10 --outdir results/exp3

# ─── Experiment 3: Adaptive Temperature (TSP) ────────────────
python -m experiments.run_experiment --exp3 --problem tsp --cities 1000 --runs 10 --outdir results/exp3

# ─── Experiment 4: Async vs Sync (TSP) ───────────────────────
python -m experiments.run_experiment --exp4 --problem tsp --cities 1000 --runs 10 --outdir results/exp4

# ─── Experiment 4: Async vs Sync (Rastrigin) ─────────────────
python -m experiments.run_experiment --exp4 --problem rastrigin --dims 10 --runs 10 --outdir results/exp4

# ─── Experiment 5: Topology Comparison (TSP) ─────────────────
python -m experiments.run_experiment --exp5 --problem tsp --cities 1000 --runs 10 --outdir results/exp5

# ─── Experiment 5: Topology Comparison (Rastrigin) ───────────
python -m experiments.run_experiment --exp5 --problem rastrigin --dims 10 --runs 10 --outdir results/exp5

# ─── Experiment 5: Scalability (TSP, full topology) ──────────
python -m experiments.run_experiment --scale --problem tsp --cities 1000 --runs 10 --outdir results/exp5_scale_tsp

# ─── Experiment 5: Scalability (Rastrigin, ring) ─────────────
# Set TOPOLOGY = 'ring' first
python -m experiments.run_experiment --scale --problem rastrigin --dims 10 --runs 10 --outdir results/exp5_scale_ring

# ─── Experiment 5: Scalability (Rastrigin, full) ─────────────
# Set TOPOLOGY = 'full' first
python -m experiments.run_experiment --scale --problem rastrigin --dims 10 --runs 10 --outdir results/exp5_scale_full

# ─── Experiment 5: Scalability (Rastrigin, random_k) ─────────
# Set TOPOLOGY = 'random_k' first
python -m experiments.run_experiment --scale --problem rastrigin --dims 10 --runs 10 --outdir results/exp5_scale_randomk
```

---

## Experiment Guide

### Configuration (in `run_experiment.py`)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `TOTAL_ITERS` | 500,000 | Serial SA iteration budget |
| `PER_ISLAND` | 500,000 | Iterations per island |
| `T0` | 5000.0 | Initial temperature |
| `ALPHA` | 0.995 | Cooling rate |
| `L` | 100 | Inner loop length |
| `T_MIN` | 1e-3 | Minimum temperature |
| `MIGRATION_INTERVAL` | 300 | Steps between migrations |
| `TOPOLOGY` | full | Default topology (set to ring for Rastrigin Exp2 and scalability) |
| `ADAPTIVE_HEAT_TSP` | False | No reheating for TSP |
| `ADAPTIVE_HEAT_RASTRIGIN` | True | Reheating enabled for Rastrigin |

### CLI Reference

```
python -m experiments.run_experiment [OPTIONS]

Options:
  --problem {tsp,rastrigin}    Benchmark problem (default: tsp)
  --cities N                   TSP city count (default: 1000)
  --dims N                     Rastrigin dimensions (default: 10)
  --runs N                     Number of independent runs (default: 3)
  --seed N                     Base random seed (default: 42)
  --islands N                  Number of islands (default: 4)
  --migration {guided,random,best_only,quality_only}
                               Migration policy (default: guided)
  --exp2                       Experiment 2: migration policy comparison
  --exp3                       Experiment 3: fixed vs adaptive cooling
  --exp4                       Experiment 4: async vs sync migration
  --exp5                       Experiment 5: topology comparison
  --scale                      Scalability: test n_islands = 2, 4, 8, 16
  --outdir PATH                Output directory for CSV results (default: results/)
```

---

## Experiment Results

### Experiment 1: Is Parallel SA Beneficial?

**TSP 1000 cities | 4 islands | Full topology | 10 runs**

| Solver | Mean Cost | Best Cost | Std Dev | Avg Time |
|--------|-----------|-----------|---------|----------|
| Serial SA | 46,882 | 46,049 | 422 | 14.5s |
| Baseline A | 46,660 | 45,624 | 596 | 15.8s |
| Baseline B | 33,728 | 32,915 | 417 | 26.4s |
| **AIPSA-GM (guided)** | **29,330** | **28,775** | **399** | **22.7s** ◀ |

**Rastrigin 10dims | 4 islands | Full topology | 10 runs**

| Solver | Mean Cost | Best Cost | Std Dev | Avg Time |
|--------|-----------|-----------|---------|----------|
| Serial SA | 21.2449 | 12.7841 | 7.06 | 1.65s |
| Baseline A | 12.4737 | 4.9738 | 3.01 | 1.95s |
| Baseline B | 5.6278 | 2.4763 | 2.87 | 3.43s |
| **AIPSA-GM (guided)** | **4.3036** | **2.5255** | **0.97** | **4.84s** ◀ |

**Key findings:**
- **TSP:** AIPSA-GM achieves 37.4% improvement over Serial SA and 13.0% over Baseline B. Std Dev is the lowest among all solvers, demonstrating superior stability.
- **Rastrigin:** AIPSA-GM achieves 79.7% improvement over Serial SA and 23.5% over Baseline B. Std Dev (0.97) is far lower than all other solvers.
- Baseline A barely improves over Serial SA on TSP, indicating that **migration, not just parallelism, is the key driver of improvement**.

<!-- TODO: Baseline B on Rastrigin is 5.6278 here but 8.8867 in Exp 4. If the configuration differs (e.g. iteration budget or cooling), add a one-line note under the Exp 4 table; if not, re-check the data. -->

---

### Experiment 2: Migration Policy Effect

**TSP 2000 cities | 4 islands | Full topology | 10 runs**

| Policy | Mean Cost | Best Cost | Std Dev | Avg Time |
|--------|-----------|-----------|---------|----------|
| random | 73,548 | 72,519 | 699 | 34.1s |
| best_only | 71,811 | 70,845 | 768 | 34.1s |
| **quality_only** | **70,849** | **69,777** | 766 | **34.7s** ◀ |
| guided | 71,093 | 69,905 | 780 | 39.2s |

**Rastrigin 10dims | 4 islands | Ring topology | 10 runs**

| Policy | Mean Cost | Best Cost | Std Dev | Avg Time |
|--------|-----------|-----------|---------|----------|
| random | 4.97 | 1.80 | 1.45 | 5.00s |
| **best_only** | **4.41** | **2.30** | **1.19** | **5.06s** ◀ |
| quality_only | 5.32 | 2.76 | 2.03 | 5.11s |
| guided | 5.03 | 2.90 | 1.68 | 4.84s |

**Rastrigin 30dims | 4 islands | Ring topology | 10 runs**

| Policy | Mean Cost | Best Cost | Std Dev | Avg Time |
|--------|-----------|-----------|---------|----------|
| random | 75.12 | 59.94 | 11.63 | 8.98s |
| best_only | 70.09 | 56.65 | 7.27 | 9.39s |
| quality_only | 69.11 | 48.43 | 12.16 | 8.96s |
| **guided** | **66.70** | **52.86** | **8.29** | **9.02s** ◀ |

**Key findings:**
- **TSP 2000 cities:** guided and quality_only are close (<0.5% gap in mean cost), suggesting that diversity provides limited benefit for combinatorial optimization where solution-space diversity is structurally constrained.
- **Rastrigin 10dims:** All policies perform comparably. Low dimensionality means fewer local optima, making diversity-aware selection unnecessary.
- **Rastrigin 30dims:** guided achieves the lowest mean cost (66.70), beating random by 11.2% and best_only by 4.8%. Higher dimensionality creates more local optima where diversity tie-breaking is valuable.
- Ring topology is used for Rastrigin Exp 2 to preserve inter-island diversity, consistent with Exp 5 findings. Using full topology suppresses diversity effects and conflates topology choice with migration policy effects.

---

### Experiment 3: Adaptive Temperature

**Rastrigin 10dims | 4 islands | Full topology | 10 runs**

| Solver | Mean Cost | Best Cost | Std Dev | Avg Time |
|--------|-----------|-----------|---------|----------|
| Fixed cooling | 8.8874 | 4.6493 | 2.16 | 3.18s |
| **Adaptive cooling** | **4.6893** | **3.6353** | **0.89** | **4.89s** ◀ |

**TSP 1000 cities | 4 islands | Full topology | 10 runs**

| Solver | Mean Cost | Best Cost | Std Dev | Avg Time |
|--------|-----------|-----------|---------|----------|
| **Fixed cooling** | **29,463** | **28,715** | **471** | **22.0s** ◀ |
| Adaptive cooling | 36,833 | 36,075 | 629 | 27.0s |

**Key findings:**
- **Rastrigin:** Adaptive cooling cuts Mean Cost by 47.2% (8.89 → 4.69) and reduces Std Dev by 58.6% (2.16 → 0.89), improving robustness across random seeds.
- **TSP:** Fixed cooling outperforms adaptive by 20.0% (29,463 vs 36,833), winning all 10/10 runs. Reheating disrupts the monotone convergence that combinatorial optimization requires.
- **Conclusion:** Adaptive temperature is effective for multimodal continuous problems but detrimental for combinatorial optimization. This motivates the per-problem `adaptive_heat` switch in AIPSA-GM.

---

### Experiment 4: Async vs Sync

**TSP 1000 cities | 4 islands | Full topology | 10 runs**

| Solver | Mean Cost | Best Cost | Std Dev | Avg Time | Speedup |
|--------|-----------|-----------|---------|----------|---------|
| Baseline B (sync) | 33,728 | 32,915 | 417 | 23.3s | 1.00x |
| **AIPSA-GM (async)** | **29,326** | **28,987** | **213** | **22.0s** | **1.06x** |

**Rastrigin 10dims | 4 islands | Full topology | 10 runs**

| Solver | Mean Cost | Best Cost | Std Dev | Avg Time | Speedup |
|--------|-----------|-----------|---------|----------|---------|
| Baseline B (sync) | 8.8867 | 5.3346 | 2.38 | 3.36s | 1.00x |
| **AIPSA-GM (async)** | **4.9239** | **2.6002** | **1.46** | **4.82s** | 0.70x |

*Async advantage grows with island count (TSP scalability data):*

| n_islands | Baseline B (sync) | AIPSA-GM (async) | Improvement |
|-----------|--------------------|-------------------|-------------|
| 2         | 35,316             | 33,742            | 4.5%        |
| 4         | 33,728             | 29,556            | 12.4%       |
| 8         | 32,350             | 26,958            | 16.7%       |
| 16        | 31,181             | 26,063            | 16.4%       |

**Key findings:**
- AIPSA-GM wins **10/10 runs** on both benchmarks in solution quality.
- **TSP:** async is both faster (1.06x) and better in quality (12.9%), since eliminating barriers reduces idle time with no tradeoff.
- **Rastrigin:** async is slower (0.70x) because auto-calibrated cooling runs the full iteration budget, but it delivers a 44.6% quality improvement, a worthwhile tradeoff.
- The async quality advantage grows with island count on TSP (4.5% → 16.7%), consistent with synchronous barriers causing increasing idle time at higher parallelism.

---

### Experiment 5: Topology Comparison and Scalability

**TSP 1000 cities | 4 islands | 10 runs**

| Topology | Mean Cost | Best Cost | Std Dev | Avg Time |
|----------|-----------|-----------|---------|----------|
| ring | 30,035 | 29,390 | 261 | 21.3s |
| **full** | **29,353** | **28,740** | 308 | **22.7s** ◀ |
| random_k | 30,147 | 29,653 | 274 | 21.5s |

**Rastrigin 10dims | 4 islands | 10 runs**

| Topology | Mean Cost | Best Cost | Std Dev | Avg Time |
|----------|-----------|-----------|---------|----------|
| **ring** | **4.61** | **2.51** | 1.41 | **4.52s** ◀ |
| full | 4.77 | 3.54 | **0.64** | 4.80s |
| random_k | 5.20 | 2.95 | 1.55 | 4.55s |

**Rastrigin 10dims | Scalability across topologies | 10 runs**

| n_islands | ring (cost) | full (cost) | random_k (cost) | ring (time) | full (time) | random_k (time) |
|-----------|------------|------------|----------------|------------|------------|----------------|
| 2  | 7.33 | **5.98** | 6.37 | 4.42s | 4.24s | 4.23s |
| 4  | **4.45** | 4.86 | 4.33 | 4.67s | 4.89s | 4.34s |
| 8  | **3.70** | 3.66 | 3.80 | 6.31s | 10.26s | 5.30s |
| 16 | **3.14** | 2.46 | 3.63 | 11.86s | 35.54s | 9.29s |

**TSP 1000 cities | Scalability | Full topology | 10 runs**

| n_islands | Serial SA | Baseline A | Baseline B | AIPSA-GM | AIPSA-GM Time |
|-----------|-----------|------------|------------|----------|--------------|
| 2  | 46,882 | 46,788 | 35,316 | **33,742** | 20.5s |
| 4  | 46,882 | 46,660 | 33,728 | **29,556** | 23.5s |
| 8  | 46,882 | 46,285 | 32,350 | **26,958** | 34.4s |
| 16 | 46,882 | 45,963 | 31,181 | **26,063** | 75.4s |

**Key findings:**
- **Topology is problem-dependent:** Full topology is best for TSP (10/10 wins, 2.3% better than ring), while ring is best for Rastrigin at 4 islands (preserves inter-island diversity for multimodal search).
- **Full topology has O(n²) communication overhead:** Time grows sharply at 16 islands for both TSP (75s vs 23s at 4 islands) and Rastrigin (35s vs 4.9s). Ring and random_k scale roughly linearly.
- **Ring is the most balanced choice for scalability:** Stable quality across island counts, roughly linear time growth, no communication blow-up.
- **Random_k is less stable:** at 16 islands on Rastrigin it reaches 3.63, worse than ring (3.14) and full (2.46).
- **Improvement over Serial SA grows with island count on TSP:** 28% → 37% → 42% → 44%.

<!-- TODO: The original text said random_k "loses to Baseline B at 16 islands (3.63 vs 3.56)", but Baseline B's Rastrigin scalability numbers are not in any table. Add them or drop that comparison. -->

---

### Experiment 6 (H5): GPU Parallelism Tradeoffs

**H5 Hypothesis:** GPU parallelism provides meaningful acceleration for SA, but its effectiveness is problem-dependent and requires sufficient problem scale to amortize hardware overhead.

---

#### GPU Implementation Design

We extend AIPSA-GM with a GPU-accelerated SA solver implemented in CUDA C via CuPy's `RawModule` interface. The implementation consists of two benchmark-specific kernels:

**Rastrigin GPU Kernel: Multi-Chain Parallel SA**

The entire SA main loop runs inside a single CUDA kernel with zero per-step CPU↔GPU communication:

```
Each CUDA thread = one independent SA chain
1024 threads run simultaneously on GPU
CPU launches kernel once → reads result once
```

Key design choices:
- LCG (Linear Congruential Generator) for on-device random number generation (no curand dependency)
- Auto-calibrated cooling: `alpha = (T_min/T0)^(1/max_iter)` computed inside kernel
- Per-thread local arrays store current and best solutions in register/local memory
- Warmup-based time calibration: measures GPU speed first, then sets `max_iter` to match the time budget

**TSP GPU Kernel: Parallel 2-opt SA Chains**

```
Each CUDA thread = one independent TSP SA chain
Each chain uses standard random 2-opt with Boltzmann (Metropolis) acceptance
32 chains run in parallel (best on RTX 4090 per the N_CHAINS sweep)
```

Unlike a greedy best-2-opt step (which turns SA into hill climbing), each chain preserves full SA stochastic acceptance:
- Each step selects a random 2-opt swap
- Acceptance follows `P(accept) = exp(-delta/T)`, identical to standard SA
- Multiple chains provide diversity without changing the per-chain SA semantics

The acceptance rule follows Metropolis et al. (1953).

---

#### Experimental Setup

**Fair comparison protocol:** CPU and GPU receive the same wall-clock time budget (60 seconds). GPU speed is measured via a warmup run, and `max_iter` is set to fully utilize the time budget. The CPU baseline is a single-chain serial SA (`solvers/serial_sa.py`).

<!-- TODO: State the CPU baseline's implementation language / threading (e.g. single-threaded Python + NumPy) so readers can interpret the CPU vs. GPU comparison. -->

| Parameter | Rastrigin | TSP |
|-----------|-----------|-----|
| Time budget | 60s | 60s |
| GPU chains | 1024 | 32 |
| Runs | 5 | 10 |
| Platform | RTX 4090 (WSL2, CUDA 12.6) | RTX 4090 (WSL2, CUDA 12.6) |

**Why N_CHAINS differs:**
- Rastrigin: the cost function is lightweight (vector ops); 1024 chains each get ~20M iterations
- TSP: each step reads the distance matrix (memory-bound); 1024 chains saturate bandwidth. N_CHAINS=32 was found best via a sweep over [32, 64, 128, 256, 512, 1024, 2048]

---

#### Results

**Rastrigin: GPU vs CPU (60s time budget, 5 runs)**

*GPU Iters are aggregate iterations summed over all 1,024 chains; CPU runs a single chain. Iter Ratio therefore measures aggregate throughput, not the speedup of a single SA chain.*

| Dims | CPU Cost | GPU Cost | Quality Gain | CPU Iters | GPU Iters (aggregate) | Iter Ratio |
|------|----------|----------|-------------|-----------|-----------|-----------|
| 10   | 4.2624   | 1.3877   | **67.4%**   | 10,361,068 | 21,904,954,163 | 2114x |
| 50   | 229.2561 | 103.7956 | **54.7%**   | 8,937,155  | 6,260,950,425  | 700x  |
| 100  | 792.2807 | 419.6162 | **47.0%**   | 7,574,789  | 2,983,547,699  | 393x  |
| 200  | 2132.4544| 1470.3526| **31.1%**   | 5,695,989  | 1,466,571,776  | 257x  |
| 500  | 6734.9075| 5645.8206| **16.2%**   | 3,390,088  | 577,015,808    | 170x  |
| 1000 | 14951.1336|13594.4674| **9.1%**   | 2,002,279  | 284,870,041    | 142x  |

**TSP: GPU vs CPU (60s time budget, 10 runs)**

| Cities | CPU Cost   | GPU Cost   | Quality Gain |
|--------|------------|------------|-------------|
| 500    | 18,667.6   | 17,646.6   | **5.5%**    |
| 1000   | 29,054.7   | 27,502.3   | **5.3%**    |
| 2000   | 81,236.8   | 73,912.8   | **9.0%**    |
| 3000   | 178,866.9  | 166,324.4  | **7.0%**    |
| 5000   | 494,954.7  | 452,871.6  | **8.5%**    |
| 8000   | 1,335,795.9| 1,140,296.2| **14.6%**   |
| 10,000 | 2,143,243.2| 1,767,074.8| **17.6%**   |

The GPU solver reaches a lower mean cost than the CPU baseline at **every** tested scale on both benchmarks.

---

#### Key Findings

**Finding 1: Rastrigin: GPU advantage decreases with dimensionality**

Gain drops monotonically from 67.4% (dims=10) to 9.1% (dims=1000). This reflects the tradeoff between GPU breadth and CPU depth:
- Low dims: 1024 chains effectively cover the search space, diversity dominates
- High dims: the search space is `[-5.12, 5.12]^1000`, each chain is severely undersampled; both CPU and GPU struggle, narrowing the gap

The aggregate Iter Ratio (2114x at dims=10, 142x at dims=1000) also shrinks with dimensionality because each step becomes more expensive.

**Finding 2: TSP: GPU advantage increases with city count**

Gain grows from 5.5% (500 cities) to 17.6% (10,000 cities), with a dip around 3,000–5,000 cities. Two competing effects are likely at play:

- **Positive:** larger TSP instances have many more local optima; a single CPU chain gets trapped more easily, while multiple GPU chains help escape
- **Negative:** a larger dist_matrix (100MB at 5,000 cities) stresses GPU L2 cache and memory bandwidth, increasing latency per step

Beyond 5,000 cities the diversity advantage appears to dominate and gain resumes growing.

**Finding 3: Optimal parallelism is hardware-constrained**

Chain sweep results for TSP (30s budget, cities=5000):

| N_CHAINS | Gain | Iters/Chain |
|----------|------|------------|
| 32       | **+22.4%** | 157,793 |
| 64       | +22.3% | 153,020 |
| 128      | +20.8% | 146,301 |
| 512      | +17.7% | 132,943 |
| 2048     | +14.3% | 114,247 |

<!-- TODO: The sweep reports +22.4% at 5,000 cities (30s budget) while the main TSP table reports 8.5% at 5,000 cities (60s budget). Add one line explaining the difference (budget, baseline, or number of runs), or re-check. -->

More chains means fewer iterations per chain, hence less convergence depth. On the RTX 4090, N_CHAINS=32 was best in our sweep, consistent with the 100MB dist_matrix saturating memory bandwidth beyond that point. On higher-bandwidth hardware (e.g., A100 at 2039 GB/s vs 4090 at 1008 GB/s), the optimal N_CHAINS would likely be higher (not tested).

**Finding 4: Benchmark structure determines GPU benefit pattern**

| | Rastrigin | TSP |
|--|-----------|-----|
| GPU benefit | Decreases with scale | Increases with scale |
| Key driver | Multi-chain diversity | Multi-chain diversity vs depth |
| Bottleneck | Dimensionality curse | Memory bandwidth |
| Best at | Low dims (67% gain) | Large cities (17.6% gain) |

This supports H5: GPU parallelism tradeoffs are problem-dependent. Rastrigin (continuous multimodal) benefits most from breadth at small scale; TSP (discrete combinatorial) benefits most from diversity at large scale.

---

#### GPU Output Files

| File | Description |
|------|-------------|
| `results/gpu_rastrigin_60time_5runs/gpu_final_rastrigin.csv` | Rastrigin GPU vs CPU (60s, 5 runs) |
| `results/gpu_TSP_N_CHAINS_32_10runs/gpu_tsp_boltzmann.csv` | TSP GPU vs CPU (60s, 10 runs, N_CHAINS=32) |
| `results/gpu_sweep/chain_sweep.csv` | N_CHAINS sweep results |

#### GPU Source Files

| File | Description |
|------|-------------|
| `solvers/gpu_sa_final.py` | Rastrigin CUDA kernel (multi-chain SA) |
| `solvers/gpu_tsp_kernel.py` | TSP CUDA kernel (parallel 2-opt SA chains) |
| `experiments/run_gpu_final.py` | Rastrigin GPU experiment runner |
| `experiments/run_gpu_tsp.py` | TSP GPU experiment runner |
| `run_tsp_chain_sweep.py` | N_CHAINS optimization sweep |

#### GPU Requirements

```bash
pip install cupy-cuda12x
pip install nvidia-curand-cu12
pip install nvidia-cuda-nvrtc-cu12
export CUDA_PATH=/usr/local/cuda-12.6
export PATH=$CUDA_PATH/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH
```

#### Run GPU Experiments

```bash
# Rastrigin: 60s budget, 5 runs
python3 -m experiments.run_gpu_final --time 60 --runs 5 --outdir results/gpu_rastrigin

# TSP: 60s budget, 10 runs, N_CHAINS=32
python3 -m experiments.run_gpu_tsp --time 60 --runs 10 --outdir results/gpu_tsp

# N_CHAINS sweep (find optimal chain count)
python3 run_tsp_chain_sweep.py
```

---

## Hypothesis Scorecard

| # | Hypothesis | Result | Evidence |
|---|-----------|--------|----------|
| H1 | Guided migration outperforms random/best-only | **Partially confirmed** | Rastrigin 30d: guided has the lowest mean cost (better than random by 11.2%, best_only by 4.8%). TSP 2000: guided ≈ quality_only (<0.5% gap), both better than random. Rastrigin 10d: no clear winner (best_only lowest). |
| H2 | Adaptive temperature improves robustness | **Confirmed (problem-dependent)** | Rastrigin: −47% mean cost, −59% Std Dev. TSP: fixed cooling wins 10/10 runs by 20%, so adaptive heating is disabled for TSP. |
| H3 | Async migration improves performance | **Partially confirmed** | TSP: 1.06x speedup + 12.9% better cost, and the advantage grows with island count (4.5% at 2 islands → 16.7% at 8). Rastrigin: 44.6% better cost but 0.70x speed (slower). |
| H4 | Topology affects convergence | **Confirmed** | Full topology best for TSP (10/10 wins). Ring best for Rastrigin at 4 islands. Full topology costs ~3x time at 16 islands. |
| H5 | GPU parallelism provides meaningful acceleration with problem-dependent tradeoffs | **Confirmed** | Rastrigin: up to 67.4% quality gain (dims=10). TSP: up to 17.6% quality gain (10K cities). GPU is better at all tested scales; optimal chain count is constrained by memory bandwidth. |

---

## AIPSA-GM Design Highlights

### Quality-First Guided Migration (Multiplicative Utility)

Unlike the additive formulation `U = α·q + β·d` proposed in the original design, we adopt a multiplicative utility:

```
U = q × (1 + β·d)
```

This makes quality a hard prerequisite for selection: when `q ≈ 0`, the utility collapses to near zero regardless of diversity, preventing low-quality but high-diversity candidates from being selected. Diversity serves as a tie-breaker bonus when candidates have similar cost. This avoids the failure mode of the additive form, where a diverse-but-poor solution can outscore a high-quality one when the diversity weight is non-trivial.

### Pool-Normalized Diversity

Diversity is normalized against the current incoming migration pool rather than the entire search-space diameter. This provides stable [0,1] diversity scores regardless of problem scale or dimensionality.

### Auto-Calibrated Cooling Schedule

The cooling rate `alpha_cool` is computed from `max_iter` so that temperature reaches `T_min` exactly when iterations are exhausted. This prevents wasted computation from premature temperature depletion.

### Phase-Aware Adaptive Heating

We extend the basic adaptive scheme with a phase-aware design that addresses the runaway-reheating problem of naive adaptive temperature control:

- **Reheating is restricted to the first 60% of iterations** to avoid disrupting late-stage convergence.
- **The reheat factor decays linearly from 1.10 to 1.02** with iteration progress, reducing aggressiveness over time.
- **Temperature is capped at T0 × 0.3** to prevent runaway reheating.

This is motivated by the observation that aggressive reheating in later phases increases variance without improving solution quality. Experiment 3 supports this design on Rastrigin (−47% mean cost, −59% Std Dev), while fixed cooling remains better for TSP, where reheating disrupts monotone convergence.

### Asynchronous Buffered Migration

Islands communicate via non-blocking queues with no global barriers. Migration is checked both during the inner loop (every `migration_interval` steps) and at cooling-step boundaries for faster response. This removes synchronization idle time and lets computation and communication overlap; on TSP the quality advantage over synchronous islands grows from 4.5% at 2 islands to 16.7% at 8 islands.

---

## Output Files

All results are saved to `results/` as CSV files:

| File/Directory | Experiment |
|----------------|------------|
| `exp1/` | Exp 1: Solver comparison |
| `exp2/` | Exp 2: Migration policy |
| `exp3/` | Exp 3: Adaptive temperature |
| `exp4/` | Exp 4: Async vs sync |
| `exp5/` | Exp 5: Topology comparison |
| `exp5_scale_tsp/` | Exp 5: TSP scalability (full topology) |
| `exp5_scale_ring/` | Exp 5: Rastrigin scalability (ring topology) |
| `exp5_scale_full/` | Exp 5: Rastrigin scalability (full topology) |
| `exp5_scale_randomk/` | Exp 5: Rastrigin scalability (random_k topology) |
| `gpu_rastrigin_60time_5runs/` | Exp 6: Rastrigin GPU vs CPU |
| `gpu_TSP_N_CHAINS_32_10runs/` | Exp 6: TSP GPU vs CPU |
| `gpu_sweep/` | Exp 6: N_CHAINS sweep |
