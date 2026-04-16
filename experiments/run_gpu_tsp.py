"""
H5: GPU vs CPU — TSP Boltzmann-weighted 2-opt Experiment
"""

import sys, os, csv, argparse
import numpy as np
sys.path.insert(0, '.')

from solvers import gpu_tsp_kernel as _m
gpu_sa_tsp_timed  = _m.gpu_sa_tsp_timed
cpu_sa_tsp_timed  = _m.cpu_sa_tsp_timed
CUPY_AVAILABLE    = _m.CUPY_AVAILABLE

N_CHAINS = 32
THREADS_PER_CHAIN = 64
#TSP_CITIES = [500, 1000, 2000, 3000, 5000, 8000, 10000]
TSP_CITIES = [2000]
T0 = 5000.0; T_MIN = 1e-3; L = 100


def run(runs=3, seed=42, time_budget=30.0, output_dir="results/gpu"):
    print(f"\n{'='*75}")
    print(f"H5: GPU TSP (Boltzmann 2-opt) vs CPU SA — Time Budget ({time_budget}s)")
    print(f"GPU: {N_CHAINS} chains × {THREADS_PER_CHAIN} threads/chain")
    print(f"CPU: 1 serial chain, random 2-opt")
    print(f"{'='*75}")

    os.makedirs(output_dir, exist_ok=True)
    rows = []

    for n_cities in TSP_CITIES:
        cpu_costs, gpu_costs = [], []

        for run_i in range(runs):
            s = seed + run_i * 100
            np.random.seed(s)
            coords = (np.random.rand(n_cities, 2) * 1000).tolist()
            print(f"  cities={n_cities:>5}  run={run_i+1}/{runs} ", end='', flush=True)

            _, cost_cpu, _, iters_cpu = cpu_sa_tsp_timed(
                n_cities=n_cities, seed=s, coords=coords,
                T0=T0, T_min=T_MIN, L=L, time_budget=time_budget
            )
            cpu_costs.append(cost_cpu)
            print(f"  CPU: {cost_cpu:.1f} ({iters_cpu:,})", end='', flush=True)

            _, cost_gpu, _, iters_gpu = gpu_sa_tsp_timed(
                n_cities=n_cities, seed=s, coords=coords,
                n_chains=N_CHAINS, threads_per_chain=THREADS_PER_CHAIN,
                T0=T0, T_min=T_MIN, L=L, time_budget=time_budget
            )
            gpu_costs.append(cost_gpu)
            print(f"  GPU: {cost_gpu:.1f} ({iters_gpu:,})")

        mean_cpu = sum(cpu_costs)/runs
        mean_gpu = sum(gpu_costs)/runs
        gain = (mean_cpu - mean_gpu) / (mean_cpu + 1e-10) * 100

        rows.append({
            'n_cities': n_cities,
            'cpu_cost': round(mean_cpu, 1),
            'gpu_cost': round(mean_gpu, 1),
            'quality_gain_pct': round(gain, 2),
        })

    print(f"\n{'='*60}")
    print(f"{'Cities':>8} {'CPU Cost':>12} {'GPU Cost':>12} {'Gain':>10}")
    print(f"{'-'*60}")
    for r in rows:
        marker = " ◀" if r['quality_gain_pct'] > 0 else "  "
        print(f"{r['n_cities']:>8} {r['cpu_cost']:>12.1f} {r['gpu_cost']:>12.1f} "
              f"{r['quality_gain_pct']:>9.1f}%{marker}")
    print(f"{'='*60}")

    path = os.path.join(output_dir, "gpu_tsp_boltzmann.csv")
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print(f"Results saved to: {path}")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--time', type=float, default=30.0)
    parser.add_argument('--outdir', type=str, default='results/gpu')
    args = parser.parse_args()

    if not CUPY_AVAILABLE:
        print("[ERROR] CuPy not available.")
        return

    run(runs=args.runs, seed=args.seed,
        time_budget=args.time, output_dir=args.outdir)


if __name__ == '__main__':
    main()