"""
Baseline C: AIPSA-GM
Adaptive Islanded Parallel Simulated Annealing with Guided Migration

Features:
  - Adaptive temperature control based on acceptance rate
  - Guided migration using utility function U(x,i) = alpha*q(x) + beta*d(x,i)
  - Asynchronous buffered migration (non-blocking queue, no global barrier)
  - Topology-aware communication: 'ring', 'full', 'random_k'
  - Migration policy: 'guided', 'quality_only', 'best_only', 'random'

Fixes (v3):
  - Auto-calculate alpha_cool so temperature schedule spans full iteration budget.
  - Phase-aware adaptive heating: only allow reheating in first 60% of iterations,
    force monotone cooling in final 40% for fine-grained convergence.
  - Reheat factor decays linearly with progress (aggressive early, gentle later).
  - Dynamic diversity injection threshold: scales with temperature ratio,
    more permissive at high T (exploration), stricter at low T (exploitation).
  - Migration check happens both inside inner loop (every migration_interval)
    AND at each cooling step boundary, for faster response.
  - Adaptive heating cap at T0 * 0.3 (tighter than v2).
"""

import multiprocessing as mp
import math
import random
import time
from utils.logger import SALogger

try:
    from tqdm import tqdm
    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False


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
    d(x, i) = distance(x, current_i)                  [diversity, higher=better, already [0,1]]
    """
    eps = 1e-10
    f_max = max(all_costs) if all_costs else sol_cost
    f_min = min(all_costs) if all_costs else sol_cost

    q = (f_max - sol_cost) / (f_max - f_min + eps)
    d = problem.distance(sol, recipient_best)

    return alpha_w * q + beta_w * d


def _select_migrant(incoming, best, best_cost, problem,
                    migration_policy, alpha_w, beta_w):
    """
    Select best candidate from incoming migrants based on migration_policy.

    Policies:
        'guided'       : quality + diversity (default AIPSA-GM strategy)
        'quality_only' : quality only (alpha_w=1.0, beta_w=0.0)
        'best_only'    : lowest cost only, no diversity consideration
        'random'       : random selection from incoming
    """
    if not incoming:
        return None

    if migration_policy == 'random':
        return random.choice(incoming)

    elif migration_policy == 'best_only':
        return min(incoming, key=lambda m: m['cost'])

    elif migration_policy == 'quality_only':
        all_costs = [m['cost'] for m in incoming] + [best_cost]
        return max(
            incoming,
            key=lambda m: _utility(
                m['solution'], m['cost'],
                best, best_cost,
                all_costs, problem,
                alpha_w=1.0, beta_w=0.0
            )
        )

    else:  # 'guided' = quality + diversity (default)
        all_costs = [m['cost'] for m in incoming] + [best_cost]
        return max(
            incoming,
            key=lambda m: _utility(
                m['solution'], m['cost'],
                best, best_cost,
                all_costs, problem,
                alpha_w=alpha_w, beta_w=beta_w
            )
        )


def _process_incoming(my_queue, best, best_cost, current, current_cost,
                      problem, migration_policy, alpha_w, beta_w,
                      progress_ratio):
    """
    Drain the incoming migration queue and apply the best candidate.
    Returns updated (best, best_cost, current, current_cost, migrated_flag).

    progress_ratio: float in [0, 1] indicating how far along we are.
        Used to scale diversity injection threshold:
        - Early (progress ~0): more permissive, accept diverse solutions easily
        - Late  (progress ~1): strict, only accept clear improvements
    """
    incoming = []
    while True:
        try:
            msg = my_queue.get_nowait()
            incoming.append(msg)
        except Exception:
            break

    if not incoming:
        return best, best_cost, current, current_cost, False

    best_candidate = _select_migrant(
        incoming, best, best_cost, problem,
        migration_policy, alpha_w, beta_w
    )

    if best_candidate is None:
        return best, best_cost, current, current_cost, False

    migrated = False

    if best_candidate['cost'] < best_cost:
        # Strict improvement: update both best and current
        best = best_candidate['solution'][:]
        best_cost = best_candidate['cost']
        current = best[:]
        current_cost = best_cost
        migrated = True

    elif migration_policy in ('guided', 'quality_only'):
        # Dynamic diversity injection:
        # - cost_threshold: starts at 1.3 (30% worse OK), decays to 1.05
        # - diversity_threshold: starts at 0.15 (easy to trigger), rises to 0.5
        cost_threshold = 1.30 - 0.25 * progress_ratio      # 1.30 -> 1.05
        diversity_threshold = 0.15 + 0.35 * progress_ratio  # 0.15 -> 0.50

        diversity = problem.distance(best_candidate['solution'], best)
        cost_ratio = best_candidate['cost'] / (best_cost + 1e-10)

        if diversity > diversity_threshold and cost_ratio < cost_threshold:
            current = best_candidate['solution'][:]
            current_cost = best_candidate['cost']
            migrated = True

    return best, best_cost, current, current_cost, migrated


def _island_worker(args):
    """
    AIPSA-GM island worker with adaptive temperature and async migration.
    """
    (island_id, problem, T0, alpha_cool, L, T_min, max_iter,
     r_low, r_high, migration_interval, migrate_count,
     topology, k_neighbors, alpha_w, beta_w,
     queues, n_islands, seed, log_file, adaptive_heat,
     migration_policy) = args

    if seed is not None:
        random.seed(seed)

    logger = SALogger(log_file=log_file, island_id=island_id)
    logger.start()

    # ── Auto-calculate alpha_cool so T reaches T_min when max_iter is hit ──
    if max_iter and max_iter > 0:
        total_cooling_steps = max(max_iter // L, 1)
        alpha_cool_auto = (T_min / T0) ** (1.0 / total_cooling_steps)
        # Use the slower (larger) of user-specified and auto-calculated
        alpha_cool = max(alpha_cool, alpha_cool_auto)

    # Initialize
    current = problem.init_solution()
    current_cost = problem.cost(current)
    best = current[:]
    best_cost = current_cost
    T = T0

    # Adaptive heating parameters
    T_heat_cap = T0 * 0.3          # never reheat above 30% of T0
    HEAT_PHASE_END = 0.6           # stop reheating after 60% of iterations

    iteration = 0
    accepted = 0
    total_moves = 0
    window_accepted = 0
    window_moves = 0
    window_size = L * 5

    effective_max_iter = max_iter if max_iter else float('inf')
    my_queue = queues[island_id]

    # Progress bar (island 0 only)
    estimated_steps = int(math.log(T_min / T0) / math.log(alpha_cool)) if alpha_cool < 1 else 1000
    show_bar = (island_id == 0 and _TQDM_AVAILABLE)
    pbar = tqdm(total=estimated_steps, desc="Island-0", unit="cool",
                dynamic_ncols=True) if show_bar else None

    while T > T_min:
        if pbar is not None:
            pbar.set_postfix(T=f"{T:.2f}", best=f"{best_cost:.2f}")
            pbar.update(1)

        for _ in range(L):
            if iteration >= effective_max_iter:
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

            # ── Adaptive temperature adjustment ──
            if window_moves >= window_size:
                progress = iteration / effective_max_iter
                r = window_accepted / window_moves

                if r < r_low and adaptive_heat and progress < HEAT_PHASE_END:
                    # Reheat factor decays with progress: 1.10 -> 1.02
                    heat_factor = 1.10 - 0.08 * (progress / HEAT_PHASE_END)
                    T = min(T * heat_factor, T_heat_cap)
                elif r > r_high:
                    T *= 0.9

                window_accepted = 0
                window_moves = 0

            # ── Async outgoing migration ──
            if iteration % migration_interval == 0:
                neighbor_ids = _get_neighbors_topology(
                    island_id, n_islands, topology, k_neighbors)
                for nid in neighbor_ids:
                    queues[nid].put({
                        'from': island_id,
                        'solution': best[:],
                        'cost': best_cost
                    })

                # Also check incoming during inner loop for faster response
                progress = iteration / effective_max_iter
                best, best_cost, current, current_cost, _ = _process_incoming(
                    my_queue, best, best_cost, current, current_cost,
                    problem, migration_policy, alpha_w, beta_w,
                    progress
                )

        # ── Also check incoming at cooling step boundary ──
        progress = iteration / effective_max_iter
        best, best_cost, current, current_cost, _ = _process_incoming(
            my_queue, best, best_cost, current, current_cost,
            problem, migration_policy, alpha_w, beta_w,
            progress
        )

        # Standard cooling
        T *= alpha_cool

        # Log
        acc_rate = accepted / total_moves if total_moves > 0 else 0.0
        logger.log(iteration, T, current_cost, best_cost, acc_rate)

        if iteration >= effective_max_iter:
            break

    if pbar is not None:
        pbar.close()

    return island_id, best, best_cost, logger.get_records()


def aipsa_gm(problem, n_islands=4, T0=1000.0, alpha_cool=0.99, L=100,
             T_min=1e-3, max_iter=None, r_low=0.1, r_high=0.4,
             migration_interval=500, migrate_count=1,
             topology='ring', k_neighbors=2,
             alpha_w=0.5, beta_w=0.5,
             adaptive_heat=True,
             migration_policy='guided',
             seeds=None, log_dir=None):
    """
    Run AIPSA-GM with async guided migration.

    Args:
        topology: 'ring', 'full', or 'random_k'
        r_low, r_high: acceptance rate thresholds for adaptive temperature
        migration_interval: steps between migration attempts
        alpha_w, beta_w: weights for quality vs diversity in utility function
            alpha_w + beta_w should equal 1.0
            higher beta_w => more diversity-driven migration
        adaptive_heat: if True, allow temperature to increase when acceptance
            rate is too low. Reheating is phase-aware:
            - Only active in first 60% of iterations
            - Reheat factor decays linearly (1.10 -> 1.02)
            - Temperature capped at T0 * 0.3
            Set to False for combinatorial problems like TSP where monotone
            cooling is more stable.
        migration_policy: 'guided' (default), 'quality_only', 'best_only', 'random'
            'guided'       - quality + diversity (AIPSA-GM full strategy)
            'quality_only' - select by solution quality only
            'best_only'    - select strictly lowest cost solution
            'random'       - select random incoming solution

    Note:
        alpha_cool is auto-adjusted if max_iter is set, to ensure the
        temperature schedule spans the full iteration budget. The slower
        (larger) of the user-supplied and auto-calculated alpha_cool is used.

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
            queues, n_islands, seeds[i], log_file, adaptive_heat,
            migration_policy
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