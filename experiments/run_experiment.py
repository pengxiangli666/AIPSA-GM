"""
Unified experiment runner for AIPSA-GM project.

Runs all four solvers on the same problem with the same total computation budget,
then prints a comparison table and saves results to CSV.

Usage:
    python -m experiments.run_experiment
    python -m experiments.run_experiment --problem rastrigin
    python -m experiments.run_experiment --problem tsp --cities 50 --runs 5
"""

import sys
import os
import csv
import time
import argparse
import random

sys.path.insert(0, '.')

from problems.tsp import TSP
from problems.rastrigin import Rastrigin
from solvers.serial_sa import serial_sa
from solvers.baseline_a import independent_replicas
from solvers.baseline_b import synchronous_islands
from solvers.aipsa_gm import aipsa_gm


# ─────────────────────────────────────────────
# Shared experiment configuration
# ─────────────────────────────────────────────

N_ISLANDS   = 4       # number of parallel islands (used by A, B, AIPSA-GM)
TOTAL_ITERS = 200_000 # serial SA iteration budget

# SA hyperparameters (same for all solvers)
T0          = 5000.0
ALPHA       = 0.995
L           = 100       # inner loop length (steps per temperature)
T_MIN       = 1e-3

# Each parallel island gets the SAME budget as serial SA (not divided).
# Parallel solvers use more total compute, but each island has a fair
# chance to converge — realistic deployment scenario where we have
# multiple cores available and care about wall-clock time, not total flops.
PER_ISLAND  = TOTAL_ITERS

# Baseline B sync parameters
STEPS_PER_ROUND = 50
N_ROUNDS        = PER_ISLAND // (STEPS_PER_ROUND * L)  # auto-calculated

# AIPSA-GM specific
MIGRATION_INTERVAL = 300
TOPOLOGY           = 'ring'
# adaptive_heat=True  → allow temperature to rise (good for Rastrigin)
# adaptive_heat=False → monotone cooling only   (good for TSP)
ADAPTIVE_HEAT_TSP        = False
ADAPTIVE_HEAT_RASTRIGIN  = True


def run_once(problem, seed_base, adaptive_heat=True):
    """Run all four solvers on the given problem, return results dict."""
    seeds = [seed_base + i for i in range(N_ISLANDS)]

    results = {}

    # ── Serial SA ──────────────────────────────
    print("  [1/4] Serial SA ...", end=' ', flush=True)
    t0 = time.time()
    _, cost, _ = serial_sa(
        problem,
        T0=T0, alpha=ALPHA, L=L, T_min=T_MIN,
        max_iter=TOTAL_ITERS,
        seed=seed_base
    )
    elapsed = time.time() - t0
    results['Serial SA'] = {'cost': cost, 'time': elapsed}
    print(f"cost={cost:.2f}  ({elapsed:.1f}s)")

    # ── Baseline A: Independent Replicas ───────
    print("  [2/4] Baseline A (Independent Replicas) ...", end=' ', flush=True)
    t0 = time.time()
    _, cost, _ = independent_replicas(
        problem,
        n_islands=N_ISLANDS,
        T0=T0, alpha=ALPHA, L=L, T_min=T_MIN,
        max_iter=PER_ISLAND,
        seeds=seeds
    )
    elapsed = time.time() - t0
    results['Baseline A'] = {'cost': cost, 'time': elapsed}
    print(f"cost={cost:.2f}  ({elapsed:.1f}s)")

    # ── Baseline B: Synchronous Islands ────────
    print("  [3/4] Baseline B (Synchronous Islands) ...", end=' ', flush=True)
    t0 = time.time()
    _, cost, _ = synchronous_islands(
        problem,
        n_islands=N_ISLANDS,
        T0=T0, alpha=ALPHA, L=L,
        steps_per_round=STEPS_PER_ROUND,
        n_rounds=max(N_ROUNDS, 10),  # at least 10 rounds
        seeds=seeds
    )
    elapsed = time.time() - t0
    results['Baseline B'] = {'cost': cost, 'time': elapsed}
    print(f"cost={cost:.2f}  ({elapsed:.1f}s)")

    # ── AIPSA-GM ───────────────────────────────
    print("  [4/4] AIPSA-GM ...", end=' ', flush=True)
    t0 = time.time()
    _, cost, _ = aipsa_gm(
        problem,
        n_islands=N_ISLANDS,
        T0=T0, alpha_cool=ALPHA, L=L, T_min=T_MIN,
        max_iter=PER_ISLAND,
        topology=TOPOLOGY,
        migration_interval=MIGRATION_INTERVAL,
        adaptive_heat=adaptive_heat,
        seeds=seeds
    )
    elapsed = time.time() - t0
    results['AIPSA-GM'] = {'cost': cost, 'time': elapsed}
    print(f"cost={cost:.2f}  ({elapsed:.1f}s)")

    return results


def print_table(all_results, solver_names):
    """Print a nicely formatted comparison table."""
    n_runs = len(all_results)

    # Compute mean and best across runs
    summary = {}
    for name in solver_names:
        costs = [r[name]['cost'] for r in all_results]
        times = [r[name]['time'] for r in all_results]
        summary[name] = {
            'mean_cost': sum(costs) / n_runs,
            'best_cost': min(costs),
            'mean_time': sum(times) / n_runs,
        }

    # Find best mean cost for highlighting
    best_mean = min(v['mean_cost'] for v in summary.values())

    print()
    print("=" * 65)
    print(f"{'Solver':<22} {'Mean Cost':>12} {'Best Cost':>12} {'Avg Time':>10}")
    print("-" * 65)
    for name in solver_names:
        s = summary[name]
        marker = " ◀" if s['mean_cost'] == best_mean else ""
        print(f"{name:<22} {s['mean_cost']:>12.2f} {s['best_cost']:>12.2f} "
              f"{s['mean_time']:>9.1f}s{marker}")
    print("=" * 65)
    print()

    return summary


def save_csv(all_results, solver_names, problem_name, output_dir="results"):
    """Save per-run results to CSV."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{problem_name}_comparison.csv")

    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['run', 'solver', 'cost', 'time_sec'])
        for run_idx, results in enumerate(all_results):
            for name in solver_names:
                writer.writerow([
                    run_idx + 1,
                    name,
                    round(results[name]['cost'], 4),
                    round(results[name]['time'], 2),
                ])

    print(f"Results saved to: {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description='AIPSA-GM Experiment Runner')
    parser.add_argument('--problem', choices=['tsp', 'rastrigin'], default='tsp')
    parser.add_argument('--cities', type=int, default=200, help='TSP city count')
    parser.add_argument('--dims', type=int, default=10, help='Rastrigin dimensions')
    parser.add_argument('--runs', type=int, default=3, help='Number of independent runs')
    parser.add_argument('--seed', type=int, default=42, help='Base random seed')
    args = parser.parse_args()

    solver_names = ['Serial SA', 'Baseline A', 'Baseline B', 'AIPSA-GM']

    # ── Build problem ──────────────────────────
    if args.problem == 'tsp':
        problem = TSP(n_cities=args.cities, seed=args.seed)
        problem_name = f"tsp_{args.cities}cities"
        print(f"\nProblem: TSP ({args.cities} cities)")
    else:
        problem = Rastrigin(n_dims=args.dims, seed=args.seed)
        problem_name = f"rastrigin_{args.dims}dims"
        print(f"\nProblem: Rastrigin ({args.dims} dims)")

    print(f"Total iters per solver: {TOTAL_ITERS:,}  "
          f"(serial) / {PER_ISLAND:,} per island (parallel)")
    print(f"Runs: {args.runs}  |  N_islands: {N_ISLANDS}  |  Topology: {TOPOLOGY}")
    print()

    # ── Run experiment ─────────────────────────
    all_results = []
    adaptive_heat = ADAPTIVE_HEAT_TSP if args.problem == 'tsp' else ADAPTIVE_HEAT_RASTRIGIN
    for run in range(args.runs):
        seed = args.seed + run * 100
        print(f"── Run {run + 1}/{args.runs}  (seed={seed}) ──────────────────")

        # Rebuild problem each run with the new seed so city layout also varies
        if args.problem == 'tsp':
            problem = TSP(n_cities=args.cities, seed=seed)
        else:
            problem = Rastrigin(n_dims=args.dims, seed=seed)

        results = run_once(problem, seed_base=seed, adaptive_heat=adaptive_heat)
        all_results.append(results)
        print()

    # ── Print summary table ────────────────────
    print_table(all_results, solver_names)

    # ── Save CSV ───────────────────────────────
    save_csv(all_results, solver_names, problem_name)


if __name__ == '__main__':
    main()