"""
Serial Simulated Annealing baseline.

Parameters:
    T0       : initial temperature
    alpha    : cooling rate (T <- alpha * T)
    L        : inner loop length (steps per temperature)
    T_min    : stopping temperature
    max_iter : max total iterations (alternative stopping criterion)
"""

import math
import random
from utils.logger import SALogger


def serial_sa(problem, T0=1000.0, alpha=0.99, L=100, T_min=1e-3,
              max_iter=None, log_file=None, log_interval=100, seed=None):
    """
    Run serial Simulated Annealing on the given problem.

    Returns:
        best_solution, best_cost, logger
    """
    if seed is not None:
        random.seed(seed)

    logger = SALogger(log_file=log_file)
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

    while T > T_min:
        for _ in range(L):
            if max_iter and iteration >= max_iter:
                break

            # Generate neighbor
            candidate = problem.neighbor(current)
            candidate_cost = problem.cost(candidate)
            delta = candidate_cost - current_cost

            # Accept or reject
            if delta < 0 or random.random() < math.exp(-delta / T):
                current = candidate
                current_cost = candidate_cost
                accepted += 1

                if current_cost < best_cost:
                    best = current[:]
                    best_cost = current_cost

            total_moves += 1
            iteration += 1

            # Log periodically
            if iteration % log_interval == 0:
                acc_rate = accepted / total_moves if total_moves > 0 else 0.0
                logger.log(iteration, T, current_cost, best_cost, acc_rate)

        # Cool down
        T *= alpha

        if max_iter and iteration >= max_iter:
            break

    # Final log entry
    acc_rate = accepted / total_moves if total_moves > 0 else 0.0
    logger.log(iteration, T, current_cost, best_cost, acc_rate)

    return best, best_cost, logger


if __name__ == '__main__':
    # Quick sanity check
    import sys
    sys.path.insert(0, '.')
    from problems.tsp import TSP
    from problems.rastrigin import Rastrigin

    print("=== Serial SA on TSP (50 cities) ===")
    tsp = TSP(n_cities=50, seed=42)
    best, cost, _ = serial_sa(tsp, T0=5000, alpha=0.995, L=200, T_min=1e-3, seed=42)
    print(f"Best tour cost: {cost:.2f}")

    print("\n=== Serial SA on Rastrigin (10 dims) ===")
    rastrigin = Rastrigin(n_dims=10, seed=42)
    best, cost, _ = serial_sa(rastrigin, T0=100, alpha=0.99, L=100, T_min=1e-4, seed=42)
    print(f"Best Rastrigin cost: {cost:.6f}")
