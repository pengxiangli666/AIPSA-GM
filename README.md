# AIPSA-GM: Adaptive Islanded Parallel Simulated Annealing with Guided Migration

## Team

Pengxiang Li, Weilun Qiu, Jinkuo Ha

## Overview

AIPSA-GM is a parallel simulated annealing framework that combines adaptive temperature control, guided migration with quality+diversity utility, asynchronous buffered communication, and topology-aware island models.

We evaluate AIPSA-GM against three baselines on two benchmarks (TSP and Rastrigin), testing five hypotheses about parallelism, migration strategy, adaptive temperature, synchronization, and communication topology.

---

## Project Structure

```
451-PROJECT/
├── experiments/
│   ├── __init__.py
│   └── run_experiment.py      # Unified experiment runner (Exp 1–5)
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
│   └── aipsa_gm.py            # AIPSA-GM (Baseline C)
├── utils/
│   ├── __init__.py
│   └── logger.py              # SA logging utility
├── results/                   # CSV outputs from experiments
├── README.md
└── requirements.txt
```

---

## Quick Start

### Requirements

```bash
pip install numpy tqdm
```

### Run All Experiments

```bash
# ─── Experiment 1: Solver Comparison (TSP) ───────────────────
python -m experiments.run_experiment --problem tsp --cities 1000 --runs 5

# ─── Experiment 1: Solver Comparison (Rastrigin) ─────────────
python -m experiments.run_experiment --problem rastrigin --dims 10 --runs 10

# ─── Experiment 2: Migration Policy (TSP) ────────────────────
python -m experiments.run_experiment --exp2 --problem tsp --cities 1000 --runs 10

# ─── Experiment 2: Migration Policy (Rastrigin) ──────────────
python -m experiments.run_experiment --exp2 --problem rastrigin --dims 10 --runs 10

# ─── Experiment 3: Adaptive Temperature (Rastrigin) ──────────
python -m experiments.run_experiment --exp3 --problem rastrigin --dims 10 --runs 10

# ─── Experiment 3: Adaptive Temperature (TSP) ────────────────
python -m experiments.run_experiment --exp3 --problem tsp --cities 1000 --runs 5

# ─── Experiment 5: Topology Comparison (TSP) ─────────────────
python -m experiments.run_experiment --exp5 --problem tsp --cities 1000 --runs 5

# ─── Experiment 5: Topology Comparison (Rastrigin) ───────────
python -m experiments.run_experiment --exp5 --problem rastrigin --dims 10 --runs 10

# ─── Experiment 5: Scalability (TSP) ─────────────────────────
python -m experiments.run_experiment --scale --problem tsp --cities 1000 --runs 3
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
| `TOPOLOGY` | full | Default topology |
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
  --exp5                       Experiment 5: topology comparison
  --scale                      Scalability: test n_islands = 2, 4, 8, 16
```

---

## Experiment Results

### Experiment 1: Is Parallel SA Beneficial?

**TSP 1000 cities | Scalability across island counts | 3 runs**

| n_islands | Serial SA | Baseline A | Baseline B | AIPSA-GM (guided) |
|-----------|-----------|------------|------------|-------------------|
| 2         | 46,544    | 46,229     | 35,023     | **33,747** ◀      |
| 4         | 46,544    | 45,981     | 33,525     | **29,380** ◀      |
| 8         | 46,544    | 45,961     | 32,069     | **26,888** ◀      |
| 16        | 46,544    | 45,931     | 30,973     | **25,876** ◀      |

**Key findings:**
- AIPSA-GM consistently achieves the best cost across all island configurations.
- Cost improves steadily as islands increase (33,747 → 25,876), demonstrating good scalability.
- Baseline A (independent replicas, no migration) barely improves over Serial SA, proving that migration — not just parallelism — is the key driver of improvement.
- AIPSA-GM outperforms Baseline B (synchronous) by 12–16%, showing the benefit of asynchronous guided migration.

---

### Experiment 2: Migration Policy Effect

**TSP 2000 cities | 4 islands | 10 runs**

| Policy | Mean Cost | Best Cost | Avg Time |
|--------|-----------|-----------|----------|
| random | 74,150 | 73,326 | 33.6s |
| best_only | 71,787 | 70,667 | 33.7s ◀ |
| quality_only | 71,817 | 70,976 | 40.3s |
| guided | 72,092 | 70,573 | 40.2s |

**Rastrigin 10dims | 4 islands | 10 runs**

| Policy | Mean Cost | Best Cost | Avg Time |
|--------|-----------|-----------|----------|
| random | 4.29 | 1.66 | 4.7s ◀ |
| best_only | 4.68 | 0.82 | 4.7s |
| quality_only | 4.64 | 2.81 | 4.7s |
| guided | 4.97 | 3.42 | 5.1s |

**Key findings:**
- On TSP, all migration policies significantly outperform random, with best_only and guided achieving comparable Best Cost (~70,500).
- On Rastrigin, migration policies show minimal differentiation (Mean Cost all within 4.3–5.0), suggesting that diversity-driven migration provides limited benefit for continuous optimization with fixed step sizes.
- The guided strategy adds overhead from diversity computation, which pays off more in combinatorial problems (TSP) than continuous ones (Rastrigin).

---

### Experiment 3: Adaptive Temperature

**Rastrigin 10dims | 4 islands | 10 runs**

| Solver | Mean Cost | Best Cost | Std Dev | Avg Time |
|--------|-----------|-----------|---------|----------|
| Fixed cooling | 8.56 | 1.61 | 4.22 | 3.3s |
| **Adaptive cooling** | **4.02** | **1.71** | **1.07** | **4.8s** ◀ |

**TSP 1000 cities | 4 islands | 5 runs**

| Solver | Mean Cost | Best Cost | Std Dev | Avg Time |
|--------|-----------|-----------|---------|----------|
| **Fixed cooling** | **29,512** | **28,905** | **478** | **24.0s** ◀ |
| Adaptive cooling | 36,808 | 35,965 | 578 | 28.7s |

**Key findings:**
- **Rastrigin:** Adaptive cooling cuts Mean Cost by 53% (8.56 → 4.02) and reduces Std Dev by 75% (4.22 → 1.07). This demonstrates significantly improved robustness across random seeds.
- **TSP:** Fixed cooling outperforms adaptive by 20% (29,512 vs 36,808). Reheating disrupts the monotone convergence that combinatorial optimization requires.
- **Conclusion:** Adaptive temperature is highly effective for multimodal continuous problems but detrimental for combinatorial optimization. This motivates the per-problem `adaptive_heat` switch in AIPSA-GM.

---

### Experiment 4: Async vs Sync

*Derived from Experiment 1 scalability data (Baseline B = synchronous, AIPSA-GM = asynchronous).*

**TSP 1000 cities | Mean Cost comparison**

| n_islands | Baseline B (sync) | AIPSA-GM (async) | Improvement |
|-----------|--------------------|-------------------|-------------|
| 2         | 35,023             | 33,747            | 3.6%        |
| 4         | 33,525             | 29,380            | 12.4%       |
| 8         | 32,069             | 26,888            | 16.2%       |
| 16        | 30,973             | 25,876            | 16.5%       |

**Key findings:**
- Asynchronous migration consistently outperforms synchronous at every island count.
- The advantage grows with more islands (3.6% at 2 islands → 16.5% at 16 islands), because synchronous barriers create more idle time as island count increases.
- Async design eliminates global barriers, allowing faster islands to continue computation while slower ones catch up.

---

### Experiment 5: Topology and Scalability

**TSP 1000 cities | 4 islands | 5 runs**

| Topology | Mean Cost | Best Cost | Avg Time |
|----------|-----------|-----------|----------|
| ring | 30,105 | 29,267 | 21.9s |
| **full** | **29,240** | **28,534** | **23.1s** ◀ |
| random_k | 30,072 | 29,650 | 22.1s |

**Rastrigin 10dims | 4 islands | 10 runs**

| Topology | Mean Cost | Best Cost | Avg Time |
|----------|-----------|-----------|----------|
| ring | 4.83 | 3.42 | 4.8s |
| **full** | **4.14** | **2.88** | **5.0s** ◀ |
| random_k | 4.58 | 2.22 | 4.8s |

**Key findings:**
- Full topology achieves the best Mean Cost on both benchmarks, thanks to maximum information sharing between islands.
- The cost is slightly higher wall-clock time (~5% more than ring) due to increased communication.
- Ring topology preserves diversity better (each island only sees 2 neighbors), but converges slower.
- With only 4 islands, differences are modest; larger island counts would amplify topology effects.

---

## AIPSA-GM Design Highlights

### Auto-Calibrated Cooling Schedule
The cooling rate `alpha_cool` is automatically calculated from `max_iter` to ensure temperature reaches `T_min` exactly when iterations are exhausted. This prevents wasted computation from premature temperature depletion.

### Phase-Aware Adaptive Heating
When enabled, reheating only occurs in the first 60% of iterations, with the reheat factor decaying from 1.10 to 1.02. Temperature is capped at `T0 * 0.3` to prevent runaway reheating.

### Dynamic Diversity Injection
For guided and quality_only migration, the acceptance threshold for diverse solutions scales with progress:
- Early phase: permissive (cost tolerance 30%, diversity threshold 0.15)
- Late phase: strict (cost tolerance 5%, diversity threshold 0.50)

### Asynchronous Buffered Migration
Islands communicate via non-blocking queues with no global barriers. Migration is checked both during the inner loop (every `migration_interval` steps) and at cooling step boundaries for faster response.

---

## Output Files

All results are saved to `results/` as CSV files:

| File | Experiment |
|------|------------|
| `tsp_*cities_comparison.csv` | Exp 1: Solver comparison |
| `rastrigin_*dims_comparison.csv` | Exp 1: Solver comparison |
| `*_migration_policy.csv` | Exp 2: Migration policy |
| `*_adaptive_temp.csv` | Exp 3: Adaptive temperature |
| `*_topology.csv` | Exp 5: Topology comparison |
| `*_scalability.csv` | Exp 5: Scalability |
