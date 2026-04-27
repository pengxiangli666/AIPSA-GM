
import multiprocessing as mp
import random
from solvers.serial_sa import serial_sa


def _worker(args):
    """Worker function for each independent replica."""
    problem, T0, alpha, L, T_min, max_iter, seed, island_id, log_file = args
    best, best_cost, logger = serial_sa(
        problem, T0=T0, alpha=alpha, L=L, T_min=T_min,
        max_iter=max_iter, log_file=log_file, seed=seed
    )
    return island_id, best, best_cost, logger.get_records()


def independent_replicas(problem, n_islands=4, T0=1000.0, alpha=0.99,
                         L=100, T_min=1e-3, max_iter=None,
                         seeds=None, log_dir=None):
    """
    Run n_islands independent SA processes in parallel.

    Returns:
        best_solution, best_cost, all_records (list of per-island records)
    """
    if seeds is None:
        seeds = [random.randint(0, 10000) for _ in range(n_islands)]

    args_list = []
    for i in range(n_islands):
        log_file = f"{log_dir}/island_{i}.csv" if log_dir else None
        args_list.append((problem, T0, alpha, L, T_min, max_iter,
                          seeds[i], i, log_file))

    with mp.Pool(processes=n_islands) as pool:
        results = pool.map(_worker, args_list)

    # Collect best overall solution
    all_records = []
    best_solution = None
    best_cost = float('inf')

    for island_id, solution, cost, records in results:
        all_records.append({'island_id': island_id, 'records': records})
        if cost < best_cost:
            best_cost = cost
            best_solution = solution

    return best_solution, best_cost, all_records


if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')
    from problems.tsp import TSP

    print("=== Baseline A: Independent Replicas (4 islands, TSP 50 cities) ===")
    tsp = TSP(n_cities=50, seed=0)
    best, cost, records = independent_replicas(
        tsp, n_islands=4, T0=5000, alpha=0.995, L=200, T_min=1e-3
    )
    print(f"Best tour cost across all islands: {cost:.2f}")
