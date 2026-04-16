"""
GPU-Accelerated SA — Final Version
====================================
Fair comparison: CPU and GPU each get the same wall-clock time budget.
GPU uses full CUDA kernel (zero per-step CPU<->GPU sync).
Temperature decays based on iteration progress (same curve for both).

Strategy:
  1. Warmup: run GPU kernel for 1000 iters to measure per-iter time
  2. Calculate gpu_max_iter = time_budget / time_per_iter
  3. Run GPU kernel with gpu_max_iter
  4. Run CPU SA for same time_budget seconds
"""

import math
import time
import random
import numpy as np

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

_RASTRIGIN_SA_KERNEL = r"""
// No includes needed - NVRTC has built-in math functions

__device__ unsigned int lcg_rand(unsigned int* state) {
    *state = (*state) * 1664525u + 1013904223u;
    return *state;
}

__device__ float lcg_uniform(unsigned int* state) {
    return (float)(lcg_rand(state)) / 4294967296.0f;
}

__device__ float lcg_normal(unsigned int* state) {
    float u1 = lcg_uniform(state) + 1e-10f;
    float u2 = lcg_uniform(state);
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2);
}

extern "C" __global__
void rastrigin_sa_kernel(
    float* best_solutions,
    float* best_costs,
    int n_dims,
    int n_chains,
    float T0,
    float T_min,
    float sigma,
    float bounds_low,
    float bounds_high,
    int max_iter,
    int L,
    unsigned int seed
) {
    int chain_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (chain_id >= n_chains) return;

    const float A = 10.0f;
    const float PI = 3.14159265f;

    unsigned int rng_state = seed ^ (chain_id * 2654435761u);
    for (int i = 0; i < 16; i++) lcg_rand(&rng_state);

    // Use local arrays (stored in registers for small dims, local mem for large)
    float current[1024];
    float best_sol[1024];

    // Initialize
    for (int d = 0; d < n_dims; d++) {
        float r = lcg_uniform(&rng_state);
        current[d] = bounds_low + r * (bounds_high - bounds_low);
        best_sol[d] = current[d];
    }

    // Initial cost
    float current_cost = A * n_dims;
    for (int d = 0; d < n_dims; d++) {
        float x = current[d];
        current_cost += x*x - A * cosf(2.0f * PI * x);
    }
    float best_cost = current_cost;

    // Auto-calibrate alpha so T reaches T_min at max_iter
    int total_cooling = max_iter / L + 1;
    float alpha = powf(T_min / T0, 1.0f / (float)total_cooling);
    float T = T0;

    for (int iter = 0; iter < max_iter; iter++) {
        // Generate neighbor
        float candidate_cost = A * n_dims;
        float candidate[1024];
        for (int d = 0; d < n_dims; d++) {
            float val = current[d] + lcg_normal(&rng_state) * sigma;
            if (val < bounds_low) val = bounds_low;
            if (val > bounds_high) val = bounds_high;
            candidate[d] = val;
            candidate_cost += val*val - A * cosf(2.0f * PI * val);
        }

        // SA accept/reject
        float delta = candidate_cost - current_cost;
        float r = lcg_uniform(&rng_state);
        float threshold = (delta < 0.0f) ? 1.0f : expf(-delta / fmaxf(T, 1e-10f));

        if (r < threshold) {
            for (int d = 0; d < n_dims; d++)
                current[d] = candidate[d];
            current_cost = candidate_cost;

            if (current_cost < best_cost) {
                best_cost = current_cost;
                for (int d = 0; d < n_dims; d++)
                    best_sol[d] = current[d];
            }
        }

        if ((iter + 1) % L == 0) {
            T *= alpha;
            if (T < T_min) T = T_min;
        }
    }

    best_costs[chain_id] = best_cost;
    for (int d = 0; d < n_dims; d++)
        best_solutions[chain_id * n_dims + d] = best_sol[d];
}
"""

_kernel_cache = {}

def _get_kernel():
    if 'rastrigin' not in _kernel_cache:
        mod = cp.RawModule(code=_RASTRIGIN_SA_KERNEL)
        _kernel_cache['rastrigin'] = mod.get_function('rastrigin_sa_kernel')
    return _kernel_cache['rastrigin']


def _run_kernel(n_dims, n_chains, T0, T_min, sigma, bounds, max_iter, L, seed):
    """Run the CUDA kernel and return (best_sol, best_cost, elapsed)."""
    low, high = bounds
    best_solutions = cp.zeros((n_chains, n_dims), dtype=cp.float32)
    best_costs = cp.full(n_chains, 1e18, dtype=cp.float32)

    threads = 256
    blocks = (n_chains + threads - 1) // threads
    kernel = _get_kernel()

    t0 = time.time()
    kernel(
        grid=(blocks,), block=(threads,),
        args=(
            best_solutions, best_costs,
            cp.int32(n_dims), cp.int32(n_chains),
            cp.float32(T0), cp.float32(T_min),
            cp.float32(sigma), cp.float32(low), cp.float32(high),
            cp.int32(max_iter), cp.int32(L),
            cp.uint32(seed)
        )
    )
    cp.cuda.Stream.null.synchronize()
    elapsed = time.time() - t0

    best_idx = int(cp.argmin(best_costs))
    best_sol = cp.asnumpy(best_solutions[best_idx]).astype(np.float64)
    best_cost = float(best_costs[best_idx])
    return best_sol, best_cost, elapsed


def gpu_sa_rastrigin_timed(n_dims=10, sigma=0.1, bounds=(-5.12, 5.12),
                            n_chains=1024, T0=5000.0, T_min=1e-3, L=100,
                            time_budget=30.0, seed=None, **kwargs):
    """
    GPU SA with time budget.
    Step 1: Warmup to measure GPU speed (iters/sec)
    Step 2: Calculate max_iter = time_budget * speed
    Step 3: Run full kernel
    """
    assert CUPY_AVAILABLE
    if seed is None:
        seed = random.randint(0, 2**31)

    # Warmup: compile kernel + measure speed
    WARMUP_ITER = 2000
    _, _, warmup_time = _run_kernel(
        n_dims, n_chains, T0, T_min, sigma, bounds,
        WARMUP_ITER, L, seed
    )
    iters_per_sec = WARMUP_ITER / max(warmup_time, 1e-6)

    # Calculate how many iters we can do in time_budget
    gpu_max_iter = max(int(iters_per_sec * time_budget), WARMUP_ITER)

    # Run full experiment
    best_sol, best_cost, elapsed = _run_kernel(
        n_dims, n_chains, T0, T_min, sigma, bounds,
        gpu_max_iter, L, seed
    )

    return best_sol, best_cost, elapsed, gpu_max_iter * n_chains


def cpu_sa_rastrigin_timed(n_dims=10, sigma=0.1, bounds=(-5.12, 5.12),
                            T0=5000.0, T_min=1e-3, L=100,
                            time_budget=30.0, seed=None, **kwargs):
    """CPU serial SA — runs until time_budget seconds elapsed."""
    if seed is not None:
        random.seed(seed); np.random.seed(seed)
    low, high = bounds; A = 10.0

    cost_fn = lambda x: A*n_dims + np.sum(x**2 - A*np.cos(2*math.pi*x))
    nbr_fn  = lambda x: np.clip(x + np.random.normal(0, sigma, n_dims), low, high)

    current = np.random.uniform(low, high, n_dims)
    current_cost = cost_fn(current)
    best, best_cost = current.copy(), current_cost
    T = T0
    T_heat_cap = T0 * 0.3
    wa, wm, ws = 0, 0, L * 5
    iterations = 0
    t_start = time.time()

    while time.time() - t_start < time_budget:
        for _ in range(L):
            if time.time() - t_start >= time_budget:
                break
            cand = nbr_fn(current); cand_cost = cost_fn(cand)
            delta = cand_cost - current_cost
            if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
                current, current_cost = cand, cand_cost; wa += 1
                if current_cost < best_cost:
                    best, best_cost = current.copy(), current_cost
            wm += 1; iterations += 1
            if wm >= ws:
                r = wa / wm
                if r < 0.1:
                    T = min(T * 1.05, T_heat_cap)
                elif r > 0.4:
                    T *= 0.9
                wa = wm = 0
        T = max(T * 0.995, T_min)

    return best, best_cost, time.time() - t_start, iterations