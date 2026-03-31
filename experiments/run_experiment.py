"""
Unified experiment runner for AIPSA-GM project.

Runs all four solvers on the same problem with the same total computation budget,
then prints a comparison table and saves results to CSV.

Usage:
    python -m experiments.run_experiment
    python -m experiments.run_experiment --problem rastrigin
    python -m experiments.run_experiment --cities 1000 --runs 5
    python -m experiments.run_experiment --islands 8
    python -m experiments.run_experiment --scale --cities 1000 --runs 3
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

N_ISLANDS   = 4       # default number of parallel islands
TOTAL_ITERS = 500_000 # serial SA iteration budget

# SA hyperparameters (same for all solvers)
T0          = 5000.0
ALPHA       = 0.995
L           = 100
T_MIN       = 1e-3

PER_ISLAND  = TOTAL_ITERS

# Baseline B sync parameters
STEPS_PER_ROUND = 50
N_ROUNDS        = PER_ISLAND // (STEPS_PER_ROUND * L)

# AIPSA-GM specific
MIGRATION_INTERVAL = 300
TOPOLOGY           = 'full' # 'ring', 'full', or 'random_k'
ADAPTIVE_HEAT_TSP        = False
ADAPTIVE_HEAT_RASTRIGIN  = True


def run_once(problem, seed_base, n_islands, adaptive_heat=True):
    """Run all four solvers on the given problem, return results dict."""
    seeds = [seed_base + i for i in range(n_islands)]
    n_rounds = max(PER_ISLAND // (STEPS_PER_ROUND * L), 10)

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
        n_islands=n_islands,
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
        n_islands=n_islands,
        T0=T0, alpha=ALPHA, L=L,
        steps_per_round=STEPS_PER_ROUND,
        n_rounds=n_rounds,
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
        n_islands=n_islands,
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

    summary = {}
    for name in solver_names:
        costs = [r[name]['cost'] for r in all_results]
        times = [r[name]['time'] for r in all_results]
        summary[name] = {
            'mean_cost': sum(costs) / n_runs,
            'best_cost': min(costs),
            'mean_time': sum(times) / n_runs,
        }

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


def run_scalability(problem_factory, problem_name, n_islands_list,
                    runs, seed, adaptive_heat, output_dir="results"):
    """
    Scalability experiment: run all solvers with different island counts.
    Shows how solution quality and wall-clock time change as islands scale up.
    Results saved to {problem_name}_scalability.csv
    """
    solver_names = ['Serial SA', 'Baseline A', 'Baseline B', 'AIPSA-GM']
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{problem_name}_scalability.csv")

    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['n_islands', 'solver', 'mean_cost', 'best_cost', 'mean_time'])

        for n in n_islands_list:
            print(f"\n{'='*60}")
            print(f"  n_islands = {n}")
            print(f"{'='*60}")
            all_results = []

            for run in range(runs):
                s = seed + run * 100
                print(f"  ── Run {run+1}/{runs}  (seed={s})")
                problem = problem_factory(s)
                result = run_once(problem, seed_base=s,
                                  n_islands=n, adaptive_heat=adaptive_heat)
                all_results.append(result)

            summary = print_table(all_results, solver_names)

            for name in solver_names:
                writer.writerow([
                    n, name,
                    round(summary[name]['mean_cost'], 4),
                    round(summary[name]['best_cost'], 4),
                    round(summary[name]['mean_time'], 2),
                ])

    print(f"\nScalability results saved to: {path}")


def main():
    parser = argparse.ArgumentParser(description='AIPSA-GM Experiment Runner')
    parser.add_argument('--problem', choices=['tsp', 'rastrigin'], default='tsp')
    parser.add_argument('--cities', type=int, default=1000, help='TSP city count')
    parser.add_argument('--dims', type=int, default=10, help='Rastrigin dimensions')
    parser.add_argument('--runs', type=int, default=3, help='Number of independent runs')
    parser.add_argument('--seed', type=int, default=42, help='Base random seed')
    parser.add_argument('--islands', type=int, default=None,
                        help='Number of islands (default: 4). e.g. --islands 8')
    parser.add_argument('--scale', action='store_true',
                        help='Scalability mode: test n_islands = 2, 4, 8, 16 automatically')
    args = parser.parse_args()

    n_islands = args.islands if args.islands else N_ISLANDS
    solver_names = ['Serial SA', 'Baseline A', 'Baseline B', 'AIPSA-GM']

    # Problem factory
    if args.problem == 'tsp':
        problem_name = f"tsp_{args.cities}cities"
        problem_factory = lambda s: TSP(n_cities=args.cities, seed=s)
        adaptive_heat = ADAPTIVE_HEAT_TSP
        print(f"\nProblem: TSP ({args.cities} cities)")
    else:
        problem_name = f"rastrigin_{args.dims}dims"
        problem_factory = lambda s: Rastrigin(n_dims=args.dims, seed=s)
        adaptive_heat = ADAPTIVE_HEAT_RASTRIGIN
        print(f"\nProblem: Rastrigin ({args.dims} dims)")

    # ── Scalability mode ───────────────────────
    if args.scale:
        print(f"Mode: Scalability  (n_islands = 2, 4, 8, 16)")
        print(f"Runs per config: {args.runs}  |  Topology: {TOPOLOGY}\n")
        run_scalability(
            problem_factory=problem_factory,
            problem_name=problem_name,
            n_islands_list=[2, 4, 8, 16],
            runs=args.runs,
            seed=args.seed,
            adaptive_heat=adaptive_heat,
        )
        return

    # ── Normal mode ────────────────────────────
    print(f"Total iters: {TOTAL_ITERS:,} (serial) / {PER_ISLAND:,} per island")
    print(f"Runs: {args.runs}  |  N_islands: {n_islands}  |  Topology: {TOPOLOGY}")
    print()

    all_results = []
    for run in range(args.runs):
        seed = args.seed + run * 100
        print(f"── Run {run + 1}/{args.runs}  (seed={seed}) ──────────────────")
        problem = problem_factory(seed)
        results = run_once(problem, seed_base=seed,
                           n_islands=n_islands, adaptive_heat=adaptive_heat)
        all_results.append(results)
        print()

    print_table(all_results, solver_names)

    # 文件名加上island数量（如果用了--islands参数）
    csv_name = f"{problem_name}_islands{n_islands}" if args.islands else problem_name
    save_csv(all_results, solver_names, csv_name)


if __name__ == '__main__':
    main()