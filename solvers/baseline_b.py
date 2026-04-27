
import multiprocessing as mp
import math
import random
from utils.logger import SALogger


def _island_worker(args):
    """
    Run one island for multiple synchronous rounds.
    Each round: run M SA steps, then put best solution in result queue.
    """
    (island_id, problem, T0, alpha, L, steps_per_round, n_rounds,
     seed, migrate_count, barrier, shared_pool, lock, log_file) = args

    if seed is not None:
        random.seed(seed)

    logger = SALogger(log_file=log_file, island_id=island_id)
    logger.start()

    # Initialize
    current = problem.init_solution()
    current_cost = problem.cost(current)
    best = current[:]
    best_cost = current_cost
    T = T0

    total_iter = 0
    accepted = 0
    total_moves = 0

    for round_idx in range(n_rounds):
        # Run steps_per_round SA steps
        for _ in range(steps_per_round):
            for _ in range(L):
                candidate = problem.neighbor(current)
                candidate_cost = problem.cost(candidate)
                delta = candidate_cost - current_cost

                if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
                    current = candidate
                    current_cost = candidate_cost
                    accepted += 1

                    if current_cost < best_cost:
                        best = current[:]
                        best_cost = current_cost

                total_moves += 1
                total_iter += 1

            T *= alpha

        # Synchronization point
        # Push our best solution to shared pool
        with lock:
            shared_pool.append((best_cost, best[:]))

        barrier.wait()  # Wait for all islands to push

        # Pull best solution from pool and potentially adopt it
        with lock:
            if shared_pool:
                pool_sorted = sorted(shared_pool, key=lambda x: x[0])
                top_cost, top_sol = pool_sorted[0]
                if top_cost < best_cost:
                    best = top_sol[:]
                    best_cost = top_cost
                    current = best[:]
                    current_cost = best_cost

        barrier.wait()  # Wait before clearing pool

        # Only island 0 clears the pool
        if island_id == 0:
            with lock:
                shared_pool[:] = []

        barrier.wait()  # Sync before next round

        # Log
        acc_rate = accepted / total_moves if total_moves > 0 else 0.0
        logger.log(total_iter, T, current_cost, best_cost, acc_rate)

    return island_id, best, best_cost, logger.get_records()


def synchronous_islands(problem, n_islands=4, T0=1000.0, alpha=0.99,
                        L=100, steps_per_round=50, n_rounds=20,
                        migrate_count=1, seeds=None, log_dir=None):
    """
    Run n_islands SA processes with synchronous migration every steps_per_round steps.

    Returns:
        best_solution, best_cost, all_records
    """
    if seeds is None:
        seeds = [random.randint(0, 10000) for _ in range(n_islands)]

    # Shared state for migration
    # FIX: use manager.Barrier() instead of mp.Barrier() so it can be
    # passed through mp.Pool (which requires pickling all arguments).
    manager = mp.Manager()
    shared_pool = manager.list()
    lock = manager.Lock()
    barrier = manager.Barrier(n_islands)  # was mp.Barrier(n_islands)

    args_list = []
    for i in range(n_islands):
        log_file = f"{log_dir}/island_{i}.csv" if log_dir else None
        args_list.append((
            i, problem, T0, alpha, L, steps_per_round, n_rounds,
            seeds[i], migrate_count, barrier, shared_pool, lock, log_file
        ))

    with mp.Pool(processes=n_islands) as pool:
        results = pool.map(_island_worker, args_list)

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

    print("=== Baseline B: Synchronous Islands (4 islands, TSP 50 cities) ===")
    tsp = TSP(n_cities=50, seed=0)
    best, cost, records = synchronous_islands(
        tsp, n_islands=4, T0=5000, alpha=0.995, L=100,
        steps_per_round=20, n_rounds=30
    )
    print(f"Best tour cost: {cost:.2f}")