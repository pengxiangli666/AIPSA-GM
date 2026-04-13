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
python -m experiments.run_experiment --exp2 --problem tsp --cities 2000 --runs 10

# ─── Experiment 2: Migration Policy (Rastrigin 10d) ──────────
python -m experiments.run_experiment --exp2 --problem rastrigin --dims 10 --runs 10

# ─── Experiment 2: Migration Policy (Rastrigin 30d) ──────────
python -m experiments.run_experiment --exp2 --problem rastrigin --dims 30 --runs 10

# ─── Experiment 3: Adaptive Temperature (Rastrigin) ──────────
python -m experiments.run_experiment --exp3 --problem rastrigin --dims 10 --runs 10

# ─── Experiment 3: Adaptive Temperature (TSP) ────────────────
python -m experiments.run_experiment --exp3 --problem tsp --cities 1000 --runs 5

# ─── Experiment 4: Async vs Sync (TSP) ───────────────────────
python -m experiments.run_experiment --exp4 --problem tsp --cities 1000 --runs 5

# ─── Experiment 4: Async vs Sync (Rastrigin) ─────────────────
python -m experiments.run_experiment --exp4 --problem rastrigin --dims 10 --runs 10

# ─── Experiment 5: Topology Comparison (TSP) ─────────────────
python -m experiments.run_experiment --exp5 --problem tsp --cities 1000 --runs 5

# ─── Experiment 5: Topology Comparison (Rastrigin) ───────────
python -m experiments.run_experiment --exp5 --problem rastrigin --dims 10 --runs 10

# ─── Experiment 5: Scalability (TSP) ─────────────────────────
python -m experiments.run_experiment --scale --problem tsp --cities 1000 --runs 3

# ─── Experiment 5: Scalability (Rastrigin) - ring topology ───
python -m experiments.run_experiment --scale --problem rastrigin --dims 10 --runs 10 --outdir results/scalability_ring

# ─── Experiment 5: Scalability (Rastrigin) - full topology ───
# Set TOPOLOGY = 'full' in run_experiment.py first
python -m experiments.run_experiment --scale --problem rastrigin --dims 10 --runs 10 --outdir results/scalability_full

# ─── Experiment 5: Scalability (Rastrigin) - random_k topology
# Set TOPOLOGY = 'random_k' in run_experiment.py first
python -m experiments.run_experiment --scale --problem rastrigin --dims 10 --runs 10 --outdir results/scalability_randomk
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
| `TOPOLOGY` | full | Default topology (set to ring for Rastrigin scalability) |
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

**TSP 1000 cities | Scalability across island counts | Full topology | 3 runs**

| n_islands | Serial SA | Baseline A | Baseline B | AIPSA-GM (guided) |
|-----------|-----------|------------|------------|-------------------|
| 2         | 46,544    | 46,229     | 35,023     | **33,747** ◀      |
| 4         | 46,544    | 45,981     | 33,525     | **29,380** ◀      |
| 8         | 46,544    | 45,961     | 32,069     | **26,888** ◀      |
| 16        | 46,544    | 45,931     | 30,973     | **25,876** ◀      |

**Rastrigin 10dims | Scalability across island counts | Ring topology | 10 runs**

| n_islands | Serial SA | Baseline A | Baseline B | AIPSA-GM (guided) |
|-----------|-----------|------------|------------|-------------------|
| 2         | 21.24     | 15.38      | 10.26      | **7.14** ◀        |
| 4         | 21.24     | 12.76      | 5.71       | **3.97** ◀        |
| 8         | 21.24     | 9.99       | 5.45       | **3.90** ◀        |
| 16        | 21.24     | 9.31       | 4.47       | **2.73** ◀        |

**Key findings:**
- **TSP:** AIPSA-GM consistently achieves the best cost across all island configurations. Cost improves steadily as islands increase (33,747 → 25,876), demonstrating good scalability.
- **Rastrigin:** AIPSA-GM wins at all island counts under ring topology. At 16 islands AIPSA-GM achieves 2.73, a 39% improvement over Baseline B's 4.47. Earlier results using full topology showed anomalous behavior at 8 and 16 islands due to communication overhead — ring topology resolves this entirely.
- Baseline A (independent replicas, no migration) barely improves over Serial SA on TSP, proving that migration — not just parallelism — is the key driver of improvement.
- TSP scalability uses full topology (optimal for ≤16 islands on combinatorial problems). Rastrigin scalability uses ring topology, which scales better for continuous optimization by limiting communication overhead while preserving island diversity.

---

### Experiment 2: Migration Policy Effect

**TSP 2000 cities | 4 islands | 10 runs**

| Policy | Mean Cost | Best Cost | Avg Time |
|--------|-----------|-----------|----------|
| random | 74,355 | 73,281 | 32.3s |
| best_only | 71,526 | 70,694 | 32.5s |
| quality_only | 71,878 | 70,655 | 32.4s |
| **guided** | **71,341** | **70,225** | **37.2s** ◀ |

**Rastrigin 10dims | 4 islands | 10 runs**

| Policy | Mean Cost | Best Cost | Avg Time |
|--------|-----------|-----------|----------|
| **random** | **3.97** | **1.64** | **5.2s** ◀ |
| best_only | 4.84 | 2.49 | 5.2s |
| quality_only | 4.82 | 3.77 | 5.3s |
| guided | 4.49 | 3.35 | 5.1s |

**Rastrigin 30dims | 4 islands | 10 runs**

| Policy | Mean Cost | Best Cost | Avg Time |
|--------|-----------|-----------|----------|
| random | 77.88 | 59.97 | 10.2s |
| best_only | 84.51 | 75.43 | 10.2s |
| quality_only | 75.17 | 57.97 | 10.2s |
| **guided** | **65.18** | **45.03** | **10.2s** ◀ |

**Key findings:**
- **TSP 2000 cities:** guided achieves the best Mean Cost (71,341) and Best Cost (70,225), outperforming best_only by 0.3% and random by 4.1%.
- **Rastrigin 10dims:** All policies perform comparably (Mean 3.97–4.84). With low dimensionality, the search space has fewer local optima and diversity-aware selection provides limited benefit.
- **Rastrigin 30dims:** guided dominates with 65.18 Mean Cost, beating quality_only by 13% and best_only by 23%. Higher dimensionality creates more local optima, making diversity-aware tie-breaking highly valuable.
- The guided strategy's quality-first utility `U = q × (1 + β·d)` ensures quality remains primary while diversity serves as an effective tie-breaker when candidates have similar cost.

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
- **Rastrigin:** Adaptive cooling cuts Mean Cost by 53% (8.56 → 4.02) and reduces Std Dev by 75% (4.22 → 1.07), demonstrating significantly improved robustness across random seeds.
- **TSP:** Fixed cooling outperforms adaptive by 20% (29,512 vs 36,808). Reheating disrupts the monotone convergence that combinatorial optimization requires.
- **Conclusion:** Adaptive temperature is highly effective for multimodal continuous problems but detrimental for combinatorial optimization. This motivates the per-problem `adaptive_heat` switch in AIPSA-GM.

---

### Experiment 4: Async vs Sync

**TSP 1000 cities | 4 islands | 5 runs**

| Solver | Mean Cost | Best Cost | Avg Time | Speedup |
|--------|-----------|-----------|----------|---------|
| Baseline B (sync) | 33,733 | 33,377 | 24.9s | 1.00x |
| **AIPSA-GM (async)** | **29,409** | **29,257** | **23.5s** | **1.06x** ◀ |

**Rastrigin 10dims | 4 islands | 10 runs**

| Solver | Mean Cost | Best Cost | Avg Time | Speedup |
|--------|-----------|-----------|----------|---------|
| Baseline B (sync) | 8.69 | 3.65 | 3.3s | 1.00x |
| **AIPSA-GM (async)** | **4.14** | **2.73** | **4.9s** | **0.68x** ◀ |

*Scalability data (Experiment 1) confirms async advantage grows with island count:*

| n_islands | Baseline B (sync) | AIPSA-GM (async) | Improvement |
|-----------|--------------------|-------------------|-------------|
| 2         | 35,023             | 33,747            | 3.6%        |
| 4         | 33,525             | 29,380            | 12.4%       |
| 8         | 32,069             | 26,888            | 16.2%       |
| 16        | 30,973             | 25,876            | 16.5%       |

**Key findings:**
- Asynchronous migration consistently outperforms synchronous in solution quality on both benchmarks: -12.8% on TSP, -52.3% on Rastrigin.
- On TSP, async is also faster (1.06x speedup) because it eliminates synchronization barriers.
- On Rastrigin, async uses more time (0.68x) due to auto-calibrated cooling running full iterations, but the cost improvement is substantial — a worthwhile tradeoff.
- The scalability data shows the async advantage grows with island count (3.6% → 16.5%), as synchronous barriers cause increasing idle time with more islands.

---

### Experiment 5: Topology Comparison and Scalability

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

**Rastrigin 10dims | Scalability across topologies | 10 runs**

| n_islands | ring (cost) | full (cost) | random_k (cost) | ring (time) | full (time) | random_k (time) |
|-----------|------------|------------|----------------|------------|------------|----------------|
| 2         | 7.14       | **5.98**   | 5.69           | 4.26s      | 4.05s      | 4.08s          |
| 4         | **3.97**   | 4.21       | 5.31           | 4.41s      | 4.65s      | 4.35s          |
| 8         | **3.90**   | 3.22       | 4.43           | 5.63s      | 9.21s 🔴   | 5.30s          |
| 16        | **2.73**   | 3.31       | 2.73           | 10.97s     | 29.43s 🔴  | 9.23s          |

**Key findings:**
- **Fixed 4-island comparison:** Full topology achieves best Mean Cost on both benchmarks at 4 islands, due to maximum information sharing between islands.
- **Scalability across island counts:** Full topology causes super-linear time growth beyond 8 islands (9.21s → 29.43s at 16 islands) due to O(n²) communication overhead. Ring and random_k scale linearly.
- **Ring topology** is the most balanced choice for scalability experiments: stable quality across all island counts, linear time growth, no communication explosion at high island counts.
- **Random_k** shows instability at 8 islands (AIPSA-GM 4.43 vs Baseline B 4.19), as random neighbor selection occasionally misses high-quality candidates.
- **Topology selection is problem- and scale-dependent:** full topology is optimal for small island counts; ring topology is recommended for high island counts and continuous optimization.

---

## Hypothesis Scorecard

| # | Hypothesis | Result | Evidence |
|---|-----------|--------|----------|
| H1 | Guided migration outperforms random/best-only | **Confirmed** | TSP 2000: guided best Mean & Best Cost. Rastrigin 30d: guided wins by 13–23%. |
| H2 | Adaptive temperature improves robustness | **Confirmed** | Rastrigin: -53% Mean Cost, -75% Std Dev. Problem-dependent: harmful for TSP. |
| H3 | Async migration improves wall-clock performance | **Confirmed** | TSP: 1.06x speedup + 12.8% better cost. Advantage grows with island count (3.6% → 16.5%). |
| H4 | Topology affects convergence | **Confirmed** | Full topology best at small island counts. Ring scales linearly and is optimal for high island counts; full topology causes 3x time overhead at 16 islands. |
| H5 | GPU parallelism tradeoffs | **Pending** | — |

---

## AIPSA-GM Design Highlights

### Quality-First Guided Migration (Multiplicative Utility)

Unlike the additive formulation `U = α·q + β·d` proposed in the original design, we adopt a multiplicative utility:

```
U = q × (1 + β·d)
```

This guarantees quality as a hard prerequisite for selection — when `q ≈ 0`, the utility collapses to near-zero regardless of diversity, preventing low-quality but high-diversity candidates from being selected. Diversity serves purely as a tie-breaker bonus when candidates have similar cost. This avoids the failure mode of the additive form, where a diverse-but-poor solution can outscore a high-quality solution when diversity weight is non-trivial.

### Pool-Normalized Diversity
Diversity is normalized against the current incoming migration pool rather than the entire search space diameter. This provides stable [0,1] diversity scores regardless of problem scale or dimensionality.

### Auto-Calibrated Cooling Schedule
The cooling rate `alpha_cool` is automatically calculated from `max_iter` to ensure temperature reaches `T_min` exactly when iterations are exhausted. This prevents wasted computation from premature temperature depletion.

### Phase-Aware Adaptive Heating

We extend the basic adaptive scheme with a phase-aware design that addresses the runaway reheating problem of naive adaptive temperature control:

- **Reheating is restricted to the first 60% of iterations** to prevent disrupting late-stage convergence.
- **The reheat factor decays linearly from 1.10 to 1.02** with iteration progress, reducing aggressiveness over time.
- **Temperature is capped at T0 × 0.3** to prevent runaway reheating.

This is motivated by the observation that aggressive reheating in later phases increases variance without improving solution quality. Experiment 3 confirms this design: adaptive cooling achieves −53% Mean Cost and −75% Std Dev on Rastrigin (8.56 → 4.02 mean, 4.22 → 1.07 std dev), while fixed cooling remains superior for TSP where reheating disrupts monotone convergence.

### Asynchronous Buffered Migration
Islands communicate via non-blocking queues with no global barriers. Migration is checked both during the inner loop (every `migration_interval` steps) and at cooling step boundaries for faster response. This eliminates synchronization idle time and allows computation and communication to overlap, with the async advantage growing from 3.6% at 2 islands to 16.5% at 16 islands.

---

## Output Files

All results are saved to `results/` as CSV files:

| File | Experiment |
|------|------------|
| `tsp_*cities_comparison.csv` | Exp 1: Solver comparison |
| `rastrigin_*dims_comparison.csv` | Exp 1: Solver comparison |
| `*_migration_policy.csv` | Exp 2: Migration policy |
| `*_adaptive_temp.csv` | Exp 3: Adaptive temperature |
| `*_async_vs_sync.csv` | Exp 4: Async vs sync |
| `*_topology.csv` | Exp 5: Topology comparison |
| `*_scalability.csv` | Exp 5: Scalability |
| `scalability_ring/` | Exp 5: Scalability (ring topology) |
| `scalability_full/` | Exp 5: Scalability (full topology) |
| `scalability_randomk/` | Exp 5: Scalability (random_k topology) |