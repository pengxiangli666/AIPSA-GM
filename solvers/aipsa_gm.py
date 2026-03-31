"""
Baseline C: AIPSA-GM
Adaptive Islanded Parallel Simulated Annealing with Guided Migration

Features:
  - Adaptive temperature control based on acceptance rate
  - Guided migration using utility function U(x,i) = alpha*q(x) + beta*d(x,i)
  - Asynchronous buffered migration (non-blocking queue, no global barrier)
  - Topology-aware communication: 'ring', 'full', 'random_k'
"""

import multiprocessing as mp
import math
import random
import time
from utils.logger import SALogger


def _get_neighbors_topology(island_id, n_islands, topology, k=2):
    """Return list of neighbor island IDs based on topology."""
    if topology == 'ring':
        return [(island_id - 1) % n_islands, (island_id + 1) % n_islands]
    elif topology == 'full':
        return [i for i in range(n_islands) if i != island_id]
    elif topology == 'random_k':
        others = [i for i in range(n_islands) if i != island_id]
        return random.sample(others, min(k, len(others)))
    else:
        raise ValueError(f"Unknown topology: {topology}")


def _utility(sol, sol_cost, recipient_best, recipient_cost,
             all_costs, problem, alpha_w=0.5, beta_w=0.5):
    """
    Compute utility of migrating sol to a recipient island.
    U(x, i) = alpha * q(x) + beta * d(x, i)

    q(x) = (f_max - f(x)) / (f_max - f_min + eps)   [quality, higher=better]
    d(x, i) = distance(x, current_i) / d_max          [diversity, higher=better]
    """
    eps = 1e-10
    f_max = max(all_costs) if all_costs else sol_cost
    f_min = min(all_costs) if all_costs else sol_cost

    q = (f_max - sol_cost) / (f_max - f_min + eps)
    d = problem.distance(sol, recipient_best)
    # d is already normalized [0,1] by problem.distance()

    return alpha_w * q + beta_w * d


def _island_worker(args):
    """
    AIPSA-GM island worker with adaptive temperature and async migration.
    """
    (island_id, problem, T0, alpha_cool, L, T_min, max_iter,
     r_low, r_high, migration_interval, migrate_count,
     topology, k_neighbors, alpha_w, beta_w,
     queues, n_islands, seed, log_file) = args

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

    iteration = 0
    accepted = 0
    total_moves = 0
    window_accepted = 0
    window_moves = 0
    window_size = L * 5  # adaptive temperature window

    my_queue = queues[island_id]

    while T > T_min:
        for _ in range(L):
            if max_iter and iteration >= max_iter:
                break

            # SA step
            candidate = problem.neighbor(current)
            candidate_cost = problem.cost(candidate)
            delta = candidate_cost - current_cost

            if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
                current = candidate
                current_cost = candidate_cost
                accepted += 1
                window_accepted += 1

                if current_cost < best_cost:
                    best = current[:]
                    best_cost = current_cost

            total_moves += 1
            window_moves += 1
            iteration += 1

            # --- Adaptive temperature adjustment ---
            if window_moves >= window_size:
                r = window_accepted / window_moves
                if r < r_low:
                    T *= 1.1   # too cold, heat up
                elif r > r_high:
                    T *= 0.9   # too hot, cool down
                window_accepted = 0
                window_moves = 0

            # --- Async outgoing migration ---
            if iteration % migration_interval == 0:
                neighbor_ids = _get_neighbors_topology(
                    island_id, n_islands, topology, k_neighbors)
                for nid in neighbor_ids:
                    queues[nid].put({
                        'from': island_id,
                        'solution': best[:],
                        'cost': best_cost
                    })

            # --- Async incoming migration ---
            if not my_queue.empty():
                incoming = []
                while not my_queue.empty():
                    try:
                        msg = my_queue.get_nowait()
                        incoming.append(msg)
                    except Exception:
                        break

                if incoming:
                    # Compute utility for each incoming candidate
                    all_costs = [m['cost'] for m in incoming] + [best_cost]
                    best_utility = -1
                    best_candidate = None

                    for msg in incoming:
                        u = _utility(
                            msg['solution'], msg['cost'],
                            best, best_cost, all_costs,
                            problem, alpha_w, beta_w
                        )
                        if u > best_utility:
                            best_utility = u
                            best_candidate = msg

                    # Accept migration if it improves quality or adds diversity
                    if best_candidate and best_candidate['cost'] < best_cost:
                        best = best_candidate['solution'][:]
                        best_cost = best_candidate['cost']
                        current = best[:]
                        current_cost = best_cost

        # Standard cooling
        T *= alpha_cool

        # Log
        acc_rate = accepted / total_moves if total_moves > 0 else 0.0
        logger.log(iteration, T, current_cost, best_cost, acc_rate)

        if max_iter and iteration >= max_iter:
            break

    return island_id, best, best_cost, logger.get_records()


def aipsa_gm(problem, n_islands=4, T0=1000.0, alpha_cool=0.99, L=100,
             T_min=1e-3, max_iter=None, r_low=0.1, r_high=0.4,
             migration_interval=500, migrate_count=1,
             topology='ring', k_neighbors=2,
             alpha_w=0.5, beta_w=0.5,
             seeds=None, log_dir=None):
    """
    Run AIPSA-GM with async guided migration.

    Args:
        topology: 'ring', 'full', or 'random_k'
        r_low, r_high: acceptance rate thresholds for adaptive temperature
        migration_interval: steps between migration attempts
        alpha_w, beta_w: weights for quality vs diversity in utility function

    Returns:
        best_solution, best_cost, all_records
    """
    if seeds is None:
        seeds = [random.randint(0, 10000) for _ in range(n_islands)]

    # One async queue per island (non-blocking communication)
    manager = mp.Manager()
    queues = [manager.Queue() for _ in range(n_islands)]

    args_list = []
    for i in range(n_islands):
        log_file = f"{log_dir}/island_{i}.csv" if log_dir else None
        args_list.append((
            i, problem, T0, alpha_cool, L, T_min, max_iter,
            r_low, r_high, migration_interval, migrate_count,
            topology, k_neighbors, alpha_w, beta_w,
            queues, n_islands, seeds[i], log_file
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

    print("=== AIPSA-GM (4 islands, ring topology, TSP 50 cities) ===")
    tsp = TSP(n_cities=50, seed=0)
    best, cost, records = aipsa_gm(
        tsp, n_islands=4, T0=5000, alpha_cool=0.995, L=100,
        T_min=1e-3, topology='ring', migration_interval=300
    )
    print(f"Best tour cost: {cost:.2f}")
