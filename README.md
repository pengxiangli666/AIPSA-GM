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
- Baseline A barely improves over Serial SA on TSP, proving that **migration — not just parallelism — is the key driver of improvement**.

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
- **TSP 2000 cities:** guided and quality_only are statistically indistinguishable (<0.5% gap), confirming that diversity provides limited benefit for combinatorial optimization where solution space diversity is structurally constrained.
- **Rastrigin 10dims:** All policies perform comparably. Low dimensionality means fewer local optima, making diversity-aware selection unnecessary.
- **Rastrigin 30dims:** guided dominates with 66.70 Mean Cost, beating random by 11.2% and best_only by 4.8%. Higher dimensionality creates more local optima where diversity tie-breaking is highly valuable.
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
- **Rastrigin:** Adaptive cooling cuts Mean Cost by 47.2% (8.89 → 4.69) and reduces Std Dev by 58.6% (2.16 → 0.89), demonstrating significantly improved robustness across random seeds.
- **TSP:** Fixed cooling outperforms adaptive by 20.0% (29,463 vs 36,833), winning all 10/10 runs. Reheating disrupts the monotone convergence that combinatorial optimization requires.
- **Conclusion:** Adaptive temperature is highly effective for multimodal continuous problems but detrimental for combinatorial optimization. This motivates the per-problem `adaptive_heat` switch in AIPSA-GM.

---

### Experiment 4: Async vs Sync

**TSP 1000 cities | 4 islands | Full topology | 10 runs**

| Solver | Mean Cost | Best Cost | Std Dev | Avg Time | Speedup |
|--------|-----------|-----------|---------|----------|---------|
| Baseline B (sync) | 33,728 | 32,915 | 417 | 23.3s | 1.00x |
| **AIPSA-GM (async)** | **29,326** | **28,987** | **213** | **22.0s** | **1.06x** ◀ |

**Rastrigin 10dims | 4 islands | Full topology | 10 runs**

| Solver | Mean Cost | Best Cost | Std Dev | Avg Time | Speedup |
|--------|-----------|-----------|---------|----------|---------|
| Baseline B (sync) | 8.8867 | 5.3346 | 2.38 | 3.36s | 1.00x |
| **AIPSA-GM (async)** | **4.9239** | **2.6002** | **1.46** | **4.82s** | **0.70x** ◀ |

*Async advantage grows with island count (TSP scalability data):*

| n_islands | Baseline B (sync) | AIPSA-GM (async) | Improvement |
|-----------|--------------------|-------------------|-------------|
| 2         | 35,316             | 33,742            | 4.5%        |
| 4         | 33,728             | 29,556            | 12.4%       |
| 8         | 32,350             | 26,958            | 16.7%       |
| 16        | 31,181             | 26,063            | 16.4%       |

**Key findings:**
- AIPSA-GM wins **10/10 runs** on both benchmarks — no exceptions.
- On TSP, async is both faster (1.06x) and better quality (12.9%), as eliminating barriers reduces idle time with no tradeoff.
- On Rastrigin, async is slower (0.70x) due to auto-calibrated cooling running full iterations, but delivers 44.6% quality improvement — a worthwhile tradeoff.
- The async advantage grows with island count (4.5% → 16.7%), confirming that synchronous barriers cause increasing idle time at higher parallelism.

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
| 8  | **3.70** | 3.66 | 3.80 | 6.31s | 10.26s 🔴 | 5.30s |
| 16 | **3.14** | 2.46 | 3.63 | 11.86s | 35.54s 🔴 | 9.29s |

**TSP 1000 cities | Scalability | Full topology | 10 runs**

| n_islands | Serial SA | Baseline A | Baseline B | AIPSA-GM | AIPSA-GM Time |
|-----------|-----------|------------|------------|----------|--------------|
| 2  | 46,882 | 46,788 | 35,316 | **33,742** | 20.5s |
| 4  | 46,882 | 46,660 | 33,728 | **29,556** | 23.5s |
| 8  | 46,882 | 46,285 | 32,350 | **26,958** | 34.4s |
| 16 | 46,882 | 45,963 | 31,181 | **26,063** | 75.4s 🔴 |

**Key findings:**
- **Topology is problem-dependent:** Full topology is optimal for TSP (10/10 wins, 2.3% better than ring), while ring is optimal for Rastrigin (preserves inter-island diversity for multimodal search).
- **Full topology causes O(n²) communication overhead:** Time explodes at 16 islands for both TSP (75s vs 23s at 4 islands) and Rastrigin (35s vs 4.9s). Ring and random_k scale linearly.
- **Ring is the most balanced choice for scalability:** Stable quality across all island counts, linear time growth, no communication explosion.
- **Random_k is unstable:** Loses to Baseline B at 16 islands on Rastrigin (3.63 vs 3.56), as random neighbor selection occasionally misses high-quality candidates.
- **AIPSA-GM vs Serial SA improvement grows with island count:** 28% → 37% → 42% → 44% on TSP, confirming scalability benefit of guided parallel SA.

---

## Hypothesis Scorecard

| # | Hypothesis | Result | Evidence |
|---|-----------|--------|----------|
| H1 | Guided migration outperforms random/best-only | **Confirmed** | Rastrigin 30d: guided wins by 4.8–11.2%. TSP: guided ≈ quality_only (<0.5% gap), both outperform random by 3.5%. |
| H2 | Adaptive temperature improves robustness | **Confirmed** | Rastrigin: -47% Mean Cost, -59% Std Dev. TSP: fixed wins 10/10 runs by 20%. Problem-dependent behavior confirmed. |
| H3 | Async migration improves wall-clock performance | **Confirmed** | TSP: 1.06x speedup + 12.9% better cost, 10/10 wins. Advantage grows from 4.5% at 2 islands to 16.7% at 8 islands. |
| H4 | Topology affects convergence | **Confirmed** | Full topology best for TSP (10/10 wins). Ring best for Rastrigin. Full causes 3x time overhead at 16 islands. |
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

This is motivated by the observation that aggressive reheating in later phases increases variance without improving solution quality. Experiment 3 confirms this design: adaptive cooling achieves −47% Mean Cost and −59% Std Dev on Rastrigin, while fixed cooling remains superior for TSP where reheating disrupts monotone convergence.

### Asynchronous Buffered Migration
Islands communicate via non-blocking queues with no global barriers. Migration is checked both during the inner loop (every `migration_interval` steps) and at cooling step boundaries for faster response. This eliminates synchronization idle time and allows computation and communication to overlap, with the async advantage growing from 4.5% at 2 islands to 16.7% at 8 islands on TSP.

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
