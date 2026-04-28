
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
TOPOLOGY           = 'ring' # ring, full, or random_k
ADAPTIVE_HEAT_TSP        = False
ADAPTIVE_HEAT_RASTRIGIN  = True

# Migration policies for Experiment 2
ALL_MIGRATION_POLICIES = ['random', 'best_only', 'quality_only', 'guided']


def run_once(problem, seed_base, n_islands, adaptive_heat=True,
             migration_policy='guided'):
    """Run all four solvers on the given problem, return results dict."""
    seeds = [seed_base + i for i in range(n_islands)]
    n_rounds = max(PER_ISLAND // (STEPS_PER_ROUND * L), 10)

    results = {}

    #Serial SA
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

    #Baseline A: Independent Replicas
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

    #Baseline B: Synchronous Islands
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

    #AIPSA-GM
    label = f"AIPSA-GM ({migration_policy})"
    print(f"  [4/4] {label} ...", end=' ', flush=True)
    t0 = time.time()
    _, cost, _ = aipsa_gm(
        problem,
        n_islands=n_islands,
        T0=T0, alpha_cool=ALPHA, L=L, T_min=T_MIN,
        max_iter=PER_ISLAND,
        topology=TOPOLOGY,
        migration_interval=MIGRATION_INTERVAL,
        adaptive_heat=adaptive_heat,
        migration_policy=migration_policy,
        seeds=seeds
    )
    elapsed = time.time() - t0
    results[label] = {'cost': cost, 'time': elapsed}
    print(f"cost={cost:.2f}  ({elapsed:.1f}s)")

    return results


def run_once_migration_only(problem, seed_base, n_islands, adaptive_heat,
                            migration_policy):
    """
    Experiment 2: run only AIPSA-GM with a specific migration policy.
    Returns single result dict keyed by policy name.
    """
    seeds = [seed_base + i for i in range(n_islands)]
    label = f"AIPSA-GM ({migration_policy})"
    print(f"  [{migration_policy}] ...", end=' ', flush=True)
    t0 = time.time()
    _, cost, _ = aipsa_gm(
        problem,
        n_islands=n_islands,
        T0=T0, alpha_cool=ALPHA, L=L, T_min=T_MIN,
        max_iter=PER_ISLAND,
        topology=TOPOLOGY,
        migration_interval=MIGRATION_INTERVAL,
        adaptive_heat=adaptive_heat,
        migration_policy=migration_policy,
        seeds=seeds
    )
    elapsed = time.time() - t0
    print(f"cost={cost:.2f}  ({elapsed:.1f}s)")
    return {label: {'cost': cost, 'time': elapsed}}


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
    print(f"{'Solver':<30} {'Mean Cost':>12} {'Best Cost':>12} {'Avg Time':>10}")
    print("-" * 65)
    for name in solver_names:
        s = summary[name]
        marker = " ◀" if s['mean_cost'] == best_mean else ""
        print(f"{name:<30} {s['mean_cost']:>12.2f} {s['best_cost']:>12.2f} "
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
    """
    solver_names = ['Serial SA', 'Baseline A', 'Baseline B', 'AIPSA-GM (guided)']
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
                # rename key for consistency
                if 'AIPSA-GM (guided)' not in result:
                    old_key = [k for k in result if k.startswith('AIPSA-GM')][0]
                    result['AIPSA-GM (guided)'] = result.pop(old_key)
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


def run_migration_experiment(problem_factory, problem_name, n_islands,
                              runs, seed, adaptive_heat, output_dir="results"):
    """
    Experiment 2: Compare four migration policies on AIPSA-GM.
    Saves results to {problem_name}_migration_policy.csv
    """
    policy_labels = [f"AIPSA-GM ({p})" for p in ALL_MIGRATION_POLICIES]
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{problem_name}_migration_policy.csv")

    # Collect results per policy across all runs
    all_results = {label: [] for label in policy_labels}

    for run in range(runs):
        s = seed + run * 100
        print(f"\n── Run {run+1}/{runs}  (seed={s}) ──────────────────")
        problem = problem_factory(s)
        for policy in ALL_MIGRATION_POLICIES:
            label = f"AIPSA-GM ({policy})"
            result = run_once_migration_only(
                problem, seed_base=s, n_islands=n_islands,
                adaptive_heat=adaptive_heat, migration_policy=policy
            )
            all_results[label].append(result[label])

    # Print summary table
    print()
    print("=" * 65)
    print(f"Experiment 2: Migration Policy Comparison")
    print(f"Problem: {problem_name} | Islands: {n_islands} | Runs: {runs}")
    print("=" * 65)
    print(f"{'Policy':<30} {'Mean Cost':>12} {'Best Cost':>12} {'Avg Time':>10}")
    print("-" * 65)

    summary_rows = []
    best_mean = min(
        sum(v['cost'] for v in vals) / runs
        for vals in all_results.values()
    )
    for label in policy_labels:
        vals = all_results[label]
        mean_cost = sum(v['cost'] for v in vals) / runs
        best_cost = min(v['cost'] for v in vals)
        mean_time = sum(v['time'] for v in vals) / runs
        marker = " ◀" if mean_cost == best_mean else ""
        print(f"{label:<30} {mean_cost:>12.2f} {best_cost:>12.2f} "
              f"{mean_time:>9.1f}s{marker}")
        summary_rows.append((label, mean_cost, best_cost, mean_time))

    print("=" * 65)

    # Save CSV
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['run', 'policy', 'cost', 'time_sec'])
        for run_idx in range(runs):
            for policy in ALL_MIGRATION_POLICIES:
                label = f"AIPSA-GM ({policy})"
                v = all_results[label][run_idx]
                writer.writerow([
                    run_idx + 1, label,
                    round(v['cost'], 4),
                    round(v['time'], 2),
                ])

    print(f"\nMigration policy results saved to: {path}")


def run_topology_experiment(problem_factory, problem_name, n_islands,
                            runs, seed, adaptive_heat, output_dir="results"):
    """
    Experiment 5: Compare communication topologies on AIPSA-GM.
    Tests ring, full, and random_k topologies with guided migration.
    Saves results to {problem_name}_topology.csv
    """
    ALL_TOPOLOGIES = ['ring', 'full', 'random_k']
    topo_labels = [f"AIPSA-GM ({t})" for t in ALL_TOPOLOGIES]
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{problem_name}_topology.csv")

    all_results = {label: [] for label in topo_labels}

    for run in range(runs):
        s = seed + run * 100
        seeds = [s + i for i in range(n_islands)]
        print(f"\n── Run {run+1}/{runs}  (seed={s}) ──────────────────")
        problem = problem_factory(s)

        for topo in ALL_TOPOLOGIES:
            label = f"AIPSA-GM ({topo})"
            print(f"  [{topo}] ...", end=' ', flush=True)
            t0 = time.time()
            _, cost, _ = aipsa_gm(
                problem,
                n_islands=n_islands,
                T0=T0, alpha_cool=ALPHA, L=L, T_min=T_MIN,
                max_iter=PER_ISLAND,
                topology=topo,
                migration_interval=MIGRATION_INTERVAL,
                adaptive_heat=adaptive_heat,
                migration_policy='guided',
                seeds=seeds
            )
            elapsed = time.time() - t0
            print(f"cost={cost:.2f}  ({elapsed:.1f}s)")
            all_results[label].append({'cost': cost, 'time': elapsed})

    # Print summary table
    print()
    print("=" * 65)
    print(f"Experiment 5: Topology Comparison")
    print(f"Problem: {problem_name} | Islands: {n_islands} | Runs: {runs}")
    print("=" * 65)
    print(f"{'Topology':<30} {'Mean Cost':>12} {'Best Cost':>12} {'Avg Time':>10}")
    print("-" * 65)

    best_mean = min(
        sum(v['cost'] for v in vals) / runs
        for vals in all_results.values()
    )
    for label in topo_labels:
        vals = all_results[label]
        mean_cost = sum(v['cost'] for v in vals) / runs
        best_cost = min(v['cost'] for v in vals)
        mean_time = sum(v['time'] for v in vals) / runs
        marker = " ◀" if mean_cost == best_mean else ""
        print(f"{label:<30} {mean_cost:>12.2f} {best_cost:>12.2f} "
              f"{mean_time:>9.1f}s{marker}")

    print("=" * 65)

    # Save CSV
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['run', 'topology', 'cost', 'time_sec'])
        for run_idx in range(runs):
            for topo in ALL_TOPOLOGIES:
                label = f"AIPSA-GM ({topo})"
                v = all_results[label][run_idx]
                writer.writerow([
                    run_idx + 1, label,
                    round(v['cost'], 4),
                    round(v['time'], 2),
                ])

    print(f"\nTopology results saved to: {path}")


def run_async_experiment(problem_factory, problem_name, n_islands,
                         runs, seed, adaptive_heat, output_dir="results"):
    """
    Experiment 4: Compare synchronous (Baseline B) vs asynchronous (AIPSA-GM) migration.
    Both use guided migration policy for fair comparison.
    Saves results to {problem_name}_async_vs_sync.csv
    """
    labels = ['Baseline B (sync)', 'AIPSA-GM (async)']
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{problem_name}_async_vs_sync.csv")

    all_results = {label: [] for label in labels}
    n_rounds = max(PER_ISLAND // (STEPS_PER_ROUND * L), 10)

    for run in range(runs):
        s = seed + run * 100
        seeds = [s + i for i in range(n_islands)]
        print(f"\n── Run {run+1}/{runs}  (seed={s}) ──────────────────")
        problem = problem_factory(s)

        # ── Baseline B: Synchronous Islands ────
        print(f"  [Baseline B (sync)] ...", end=' ', flush=True)
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
        print(f"cost={cost:.2f}  ({elapsed:.1f}s)")
        all_results['Baseline B (sync)'].append({'cost': cost, 'time': elapsed})

        # ── AIPSA-GM: Asynchronous Migration ───
        print(f"  [AIPSA-GM (async)] ...", end=' ', flush=True)
        t0 = time.time()
        _, cost, _ = aipsa_gm(
            problem,
            n_islands=n_islands,
            T0=T0, alpha_cool=ALPHA, L=L, T_min=T_MIN,
            max_iter=PER_ISLAND,
            topology=TOPOLOGY,
            migration_interval=MIGRATION_INTERVAL,
            adaptive_heat=adaptive_heat,
            migration_policy='guided',
            seeds=seeds
        )
        elapsed = time.time() - t0
        print(f"cost={cost:.2f}  ({elapsed:.1f}s)")
        all_results['AIPSA-GM (async)'].append({'cost': cost, 'time': elapsed})

    # Print summary table
    print()
    print("=" * 70)
    print(f"Experiment 4: Async vs Sync Migration")
    print(f"Problem: {problem_name} | Islands: {n_islands} | Runs: {runs}")
    print("=" * 70)
    print(f"{'Solver':<30} {'Mean Cost':>12} {'Best Cost':>12} "
          f"{'Avg Time':>10} {'Speedup':>10}")
    print("-" * 70)

    sync_mean_time = sum(v['time'] for v in all_results['Baseline B (sync)']) / runs
    best_mean = min(
        sum(v['cost'] for v in vals) / runs
        for vals in all_results.values()
    )
    for label in labels:
        vals = all_results[label]
        mean_cost = sum(v['cost'] for v in vals) / runs
        best_cost = min(v['cost'] for v in vals)
        mean_time = sum(v['time'] for v in vals) / runs
        speedup = sync_mean_time / mean_time if mean_time > 0 else 0
        marker = " ◀" if mean_cost == best_mean else ""
        print(f"{label:<30} {mean_cost:>12.2f} {best_cost:>12.2f} "
              f"{mean_time:>9.1f}s {speedup:>9.2f}x{marker}")

    print("=" * 70)

    # Save CSV
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['run', 'solver', 'cost', 'time_sec'])
        for run_idx in range(runs):
            for label in labels:
                v = all_results[label][run_idx]
                writer.writerow([
                    run_idx + 1, label,
                    round(v['cost'], 4),
                    round(v['time'], 2),
                ])

    print(f"\nAsync vs Sync results saved to: {path}")


def run_adaptive_experiment(problem_factory, problem_name, n_islands,
                            runs, seed, output_dir="results"):
    """
    Experiment 3: Compare fixed cooling vs adaptive cooling on AIPSA-GM.
    Runs AIPSA-GM (guided) twice: once with adaptive_heat=True, once False.
    Saves results to {problem_name}_adaptive_temp.csv
    """
    labels = ['AIPSA-GM (fixed cooling)', 'AIPSA-GM (adaptive cooling)']
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{problem_name}_adaptive_temp.csv")

    all_results = {label: [] for label in labels}

    for run in range(runs):
        s = seed + run * 100
        seeds = [s + i for i in range(n_islands)]
        print(f"\n── Run {run+1}/{runs}  (seed={s}) ──────────────────")
        problem = problem_factory(s)

        for adaptive_heat, label in [(False, labels[0]), (True, labels[1])]:
            print(f"  [{label}] ...", end=' ', flush=True)
            t0 = time.time()
            _, cost, _ = aipsa_gm(
                problem,
                n_islands=n_islands,
                T0=T0, alpha_cool=ALPHA, L=L, T_min=T_MIN,
                max_iter=PER_ISLAND,
                topology=TOPOLOGY,
                migration_interval=MIGRATION_INTERVAL,
                adaptive_heat=adaptive_heat,
                migration_policy='guided',
                seeds=seeds
            )
            elapsed = time.time() - t0
            print(f"cost={cost:.2f}  ({elapsed:.1f}s)")
            all_results[label].append({'cost': cost, 'time': elapsed})

    # Print summary table
    print()
    print("=" * 65)
    print(f"Experiment 3: Adaptive Temperature Comparison")
    print(f"Problem: {problem_name} | Islands: {n_islands} | Runs: {runs}")
    print("=" * 65)
    print(f"{'Solver':<35} {'Mean Cost':>10} {'Best Cost':>10} "
          f"{'Std Dev':>10} {'Avg Time':>10}")
    print("-" * 65)

    best_mean = min(
        sum(v['cost'] for v in vals) / runs
        for vals in all_results.values()
    )
    for label in labels:
        vals = all_results[label]
        costs = [v['cost'] for v in vals]
        times = [v['time'] for v in vals]
        mean_cost = sum(costs) / runs
        best_cost = min(costs)
        mean_time = sum(times) / runs
        variance = sum((c - mean_cost) ** 2 for c in costs) / runs
        std_dev = variance ** 0.5
        marker = " ◀" if mean_cost == best_mean else ""
        print(f"{label:<35} {mean_cost:>10.2f} {best_cost:>10.2f} "
              f"{std_dev:>10.2f} {mean_time:>9.1f}s{marker}")

    print("=" * 65)

    # Save CSV with per-run data
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['run', 'solver', 'cost', 'time_sec'])
        for run_idx in range(runs):
            for label in labels:
                v = all_results[label][run_idx]
                writer.writerow([
                    run_idx + 1, label,
                    round(v['cost'], 4),
                    round(v['time'], 2),
                ])

    print(f"\nAdaptive temperature results saved to: {path}")


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
    parser.add_argument('--exp2', action='store_true',
                        help='Experiment 2: compare all four migration policies')
    parser.add_argument('--exp3', action='store_true',
                        help='Experiment 3: compare fixed vs adaptive cooling')
    parser.add_argument('--exp4', action='store_true',
                        help='Experiment 4: compare sync vs async migration')
    parser.add_argument('--exp5', action='store_true',
                        help='Experiment 5: compare ring, full, random_k topologies')
    parser.add_argument('--migration',
                        choices=ALL_MIGRATION_POLICIES,
                        default='guided',
                        help='Migration policy for AIPSA-GM (default: guided)')
    #NEW: custom output directory
    parser.add_argument('--outdir', type=str, default=None,
                        help='Output directory for CSV results. '
                             'Defaults to "results". '
                             'Example: --outdir results/2025-01-15')
    args = parser.parse_args()

    n_islands = args.islands if args.islands else N_ISLANDS

    #Resolve output directory
    output_dir = args.outdir if args.outdir else "results"
    os.makedirs(output_dir, exist_ok=True)

    solver_names = [
        'Serial SA', 'Baseline A', 'Baseline B',
        f'AIPSA-GM ({args.migration})'
    ]

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

    print(f"Output directory: {output_dir}")

    #Experiment 2: Migration Policy
    if args.exp2:
        print(f"Mode: Experiment 2 — Migration Policy Comparison")
        print(f"Policies: {ALL_MIGRATION_POLICIES}")
        print(f"Runs: {args.runs}  |  N_islands: {n_islands}  |  Topology: {TOPOLOGY}\n")
        run_migration_experiment(
            problem_factory=problem_factory,
            problem_name=problem_name,
            n_islands=n_islands,
            runs=args.runs,
            seed=args.seed,
            adaptive_heat=adaptive_heat,
            output_dir=output_dir,
        )
        return

    #Experiment 3: Adaptive Temperature
    if args.exp3:
        print(f"Mode: Experiment 3 — Fixed vs Adaptive Cooling")
        print(f"Runs: {args.runs}  |  N_islands: {n_islands}  |  Topology: {TOPOLOGY}\n")
        run_adaptive_experiment(
            problem_factory=problem_factory,
            problem_name=problem_name,
            n_islands=n_islands,
            runs=args.runs,
            seed=args.seed,
            output_dir=output_dir,
        )
        return

    #Experiment 4: Async vs Sync
    if args.exp4:
        print(f"Mode: Experiment 4 — Async vs Sync Migration")
        print(f"Runs: {args.runs}  |  N_islands: {n_islands}  |  Topology: {TOPOLOGY}\n")
        run_async_experiment(
            problem_factory=problem_factory,
            problem_name=problem_name,
            n_islands=n_islands,
            runs=args.runs,
            seed=args.seed,
            adaptive_heat=adaptive_heat,
            output_dir=output_dir,
        )
        return

    #Experiment 5: Topology Comparison
    if args.exp5:
        print(f"Mode: Experiment 5 — Topology Comparison (ring, full, random_k)")
        print(f"Runs: {args.runs}  |  N_islands: {n_islands}\n")
        run_topology_experiment(
            problem_factory=problem_factory,
            problem_name=problem_name,
            n_islands=n_islands,
            runs=args.runs,
            seed=args.seed,
            adaptive_heat=adaptive_heat,
            output_dir=output_dir,
        )
        return

    #Scalability mode
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
            output_dir=output_dir,
        )
        return

    # Normal mode
    print(f"Total iters: {TOTAL_ITERS:,} (serial) / {PER_ISLAND:,} per island")
    print(f"Runs: {args.runs}  |  N_islands: {n_islands}  |  Topology: {TOPOLOGY}")
    print(f"Migration policy: {args.migration}")
    print()

    all_results = []
    for run in range(args.runs):
        seed = args.seed + run * 100
        print(f"── Run {run + 1}/{args.runs}  (seed={seed}) ──────────────────")
        problem = problem_factory(seed)
        results = run_once(problem, seed_base=seed,
                           n_islands=n_islands, adaptive_heat=adaptive_heat,
                           migration_policy=args.migration)
        all_results.append(results)
        print()

    print_table(all_results, solver_names)

    csv_name = f"{problem_name}_islands{n_islands}" if args.islands else problem_name
    save_csv(all_results, solver_names, csv_name, output_dir=output_dir)


if __name__ == '__main__':
    main()