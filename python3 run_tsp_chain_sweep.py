"""
TSP Chain Count Sweep
Find optimal N_CHAINS for GPU TSP SA.
Tests different chain counts on a fixed problem to find the sweet spot
between diversity (more chains) and convergence depth (fewer chains).
"""

import sys, os, csv
import numpy as np
sys.path.insert(0, '.')

from solvers import gpu_tsp_kernel as _m
gpu_sa_tsp_timed = _m.gpu_sa_tsp_timed
cpu_sa_tsp_timed = _m.cpu_sa_tsp_timed

# Test on representative city counts
TEST_CITIES = [1000, 3000, 5000]
CHAIN_COUNTS = [32, 64, 128, 256, 512, 1024, 2048]
TIME_BUDGET = 30.0
RUNS = 3
SEED = 42
T0 = 5000.0; T_MIN = 1e-3; L = 100


def main():
    print(f"\n{'='*70}")
    print(f"TSP Chain Count Sweep — Finding Optimal N_CHAINS")
    print(f"Time budget: {TIME_BUDGET}s | Runs: {RUNS}")
    print(f"{'='*70}")

    os.makedirs("results/gpu_sweep", exist_ok=True)
    rows = []

    for n_cities in TEST_CITIES:
        print(f"\n── cities={n_cities} ──────────────────────────────")

        # CPU baseline
        cpu_costs = []
        for run_i in range(RUNS):
            s = SEED + run_i * 100
            np.random.seed(s)
            coords = (np.random.rand(n_cities, 2) * 1000).tolist()
            _, cost_cpu, _, iters_cpu = cpu_sa_tsp_timed(
                n_cities=n_cities, seed=s, coords=coords,
                T0=T0, T_min=T_MIN, L=L, time_budget=TIME_BUDGET
            )
            cpu_costs.append(cost_cpu)
        mean_cpu = sum(cpu_costs) / RUNS
        print(f"  CPU baseline: {mean_cpu:.1f}")

        for n_chains in CHAIN_COUNTS:
            gpu_costs = []
            for run_i in range(RUNS):
                s = SEED + run_i * 100
                np.random.seed(s)
                coords = (np.random.rand(n_cities, 2) * 1000).tolist()
                _, cost_gpu, _, iters_gpu = gpu_sa_tsp_timed(
                    n_cities=n_cities, seed=s, coords=coords,
                    n_chains=n_chains,
                    T0=T0, T_min=T_MIN, L=L, time_budget=TIME_BUDGET
                )
                gpu_costs.append(cost_gpu)

            mean_gpu = sum(gpu_costs) / RUNS
            gain = (mean_cpu - mean_gpu) / (mean_cpu + 1e-10) * 100
            iters_per_chain = iters_gpu // n_chains

            marker = " ◀" if gain > 0 else "  "
            print(f"  chains={n_chains:>5}: GPU={mean_gpu:.1f}  gain={gain:+.1f}%  "
                  f"iters/chain={iters_per_chain:,}{marker}")

            rows.append({
                'n_cities': n_cities,
                'n_chains': n_chains,
                'cpu_cost': round(mean_cpu, 1),
                'gpu_cost': round(mean_gpu, 1),
                'gain_pct': round(gain, 2),
                'iters_per_chain': iters_per_chain,
            })

    # Save
    path = "results/gpu_sweep/chain_sweep.csv"
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print(f"\nResults saved to: {path}")

    # Print best chain count per city size
    print(f"\n{'='*50}")
    print("Optimal N_CHAINS per city size:")
    for n_cities in TEST_CITIES:
        city_rows = [r for r in rows if r['n_cities'] == n_cities]
        best = max(city_rows, key=lambda r: r['gain_pct'])
        print(f"  cities={n_cities}: best N_CHAINS={best['n_chains']} "
              f"(gain={best['gain_pct']:+.1f}%, iters/chain={best['iters_per_chain']:,})")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()