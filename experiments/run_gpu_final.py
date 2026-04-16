"""
H5: GPU vs CPU — Final Time-Budget Experiment
Both solvers get the same wall-clock time.
GPU: CUDA kernel, 1024 chains, zero per-step sync
CPU: serial SA, 1 chain
"""

import sys, os, csv, argparse
import numpy as np
sys.path.insert(0, '.')

from solvers import gpu_sa_final as _m
gpu_sa_rastrigin_timed = _m.gpu_sa_rastrigin_timed
cpu_sa_rastrigin_timed = _m.cpu_sa_rastrigin_timed
CUPY_AVAILABLE         = _m.CUPY_AVAILABLE

N_CHAINS = 1024
RASTRIGIN_DIMS = [10, 50, 100, 200, 500, 1000]
T0 = 5000.0; T_MIN = 1e-3; L = 100


def run(runs=3, seed=42, time_budget=30.0, output_dir="results/gpu"):
    print(f"\n{'='*75}")
    print(f"H5: GPU Kernel vs CPU SA — Fixed Time Budget ({time_budget}s each)")
    print(f"GPU: {N_CHAINS} parallel chains (CUDA kernel, zero CPU sync)")
    print(f"CPU: 1 serial chain")
    print(f"{'='*75}")

    os.makedirs(output_dir, exist_ok=True)
    rows = []

    for n_dims in RASTRIGIN_DIMS:
        cpu_costs, gpu_costs = [], []
        cpu_iters, gpu_iters = [], []

        for run_i in range(runs):
            s = seed + run_i * 100
            print(f"  dims={n_dims:>5}  run={run_i+1}/{runs} ", end='', flush=True)

            _, cost_cpu, t_cpu, iters_cpu = cpu_sa_rastrigin_timed(
                n_dims=n_dims, T0=T0, T_min=T_MIN, L=L,
                time_budget=time_budget, seed=s
            )
            cpu_costs.append(cost_cpu); cpu_iters.append(iters_cpu)
            print(f"  CPU: {cost_cpu:.4f} ({iters_cpu:,} iters)", end='', flush=True)

            _, cost_gpu, t_gpu, iters_gpu = gpu_sa_rastrigin_timed(
                n_dims=n_dims, n_chains=N_CHAINS,
                T0=T0, T_min=T_MIN, L=L,
                time_budget=time_budget, seed=s
            )
            gpu_costs.append(cost_gpu); gpu_iters.append(iters_gpu)
            print(f"  GPU: {cost_gpu:.4f} ({iters_gpu:,} iters)")

        mean_cpu = sum(cpu_costs)/runs
        mean_gpu = sum(gpu_costs)/runs
        mean_cpu_iters = int(sum(cpu_iters)/runs)
        mean_gpu_iters = int(sum(gpu_iters)/runs)
        quality_gain = (mean_cpu - mean_gpu) / (mean_cpu + 1e-10) * 100

        rows.append({
            'n_dims': n_dims,
            'cpu_cost': round(mean_cpu, 4),
            'gpu_cost': round(mean_gpu, 4),
            'quality_gain_pct': round(quality_gain, 2),
            'cpu_iters': mean_cpu_iters,
            'gpu_iters': mean_gpu_iters,
            'iter_ratio': round(mean_gpu_iters / (mean_cpu_iters + 1), 1),
        })

    print(f"\n{'='*85}")
    print(f"{'Dims':>6} {'CPU Cost':>12} {'GPU Cost':>12} {'Gain':>8} "
          f"{'CPU Iters':>12} {'GPU Iters':>12} {'Ratio':>8}")
    print(f"{'-'*85}")
    for r in rows:
        marker = " ◀" if r['quality_gain_pct'] > 0 else "  "
        print(f"{r['n_dims']:>6} {r['cpu_cost']:>12.4f} {r['gpu_cost']:>12.4f} "
              f"{r['quality_gain_pct']:>7.1f}% {r['cpu_iters']:>12,} "
              f"{r['gpu_iters']:>12,} {r['iter_ratio']:>7.1f}x{marker}")
    print(f"{'='*85}")
    print("(◀ = GPU found better solution | Gain = (CPU-GPU)/CPU×100%)")

    path = os.path.join(output_dir, "gpu_final_rastrigin.csv")
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