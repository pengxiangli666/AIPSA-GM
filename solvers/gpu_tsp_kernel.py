"""
GPU TSP SA — Boltzmann weighted 2-opt, fixed version
Each chain initialized properly with Fisher-Yates on GPU.
"""

import math, time, random
import numpy as np

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

_TSP_KERNEL = r"""
__device__ unsigned int lcg(unsigned int* s) {
    *s = (*s) * 1664525u + 1013904223u;
    return *s;
}
__device__ float lcg_f(unsigned int* s) {
    return (float)lcg(s) / 4294967296.0f;
}

extern "C" __global__
void tsp_sa_kernel(
    int*   tours,           // (N_CHAINS, n_cities) in/out
    float* costs,           // (N_CHAINS,) out
    const float* dist,      // (n_cities, n_cities)
    int n_cities,
    int n_chains,
    float T0,
    float T_min,
    int max_iter,
    int L,
    unsigned int seed
) {
    int cid = blockIdx.x * blockDim.x + threadIdx.x;
    if (cid >= n_chains) return;

    unsigned int rng = seed ^ (cid * 2654435761u);
    for (int i = 0; i < 16; i++) lcg(&rng);

    int* tour = tours + cid * n_cities;

    // Fisher-Yates shuffle for this chain
    for (int i = 0; i < n_cities; i++) tour[i] = i;
    for (int i = n_cities - 1; i > 0; i--) {
        int j = (int)(lcg(&rng) % (unsigned int)(i + 1));
        int tmp = tour[i]; tour[i] = tour[j]; tour[j] = tmp;
    }

    // Compute initial cost
    float cur_cost = 0.0f;
    for (int i = 0; i < n_cities; i++) {
        int a = tour[i], b = tour[(i+1) % n_cities];
        cur_cost += dist[a * n_cities + b];
    }
    float best_cost = cur_cost;

    // Auto alpha
    int n_cool = max_iter / L + 1;
    float alpha = powf(T_min / T0, 1.0f / (float)n_cool);
    float T = T0;

    for (int iter = 0; iter < max_iter; iter++) {
        // Sample random 2-opt (i, j) with i < j-1
        int i = (int)(lcg(&rng) % (unsigned int)(n_cities - 2));
        int j = i + 1 + (int)(lcg(&rng) % (unsigned int)(n_cities - 1 - i));

        int a = tour[i], b = tour[i+1];
        int c = tour[j], d = tour[(j+1) % n_cities];

        float delta = dist[a*n_cities+c] + dist[b*n_cities+d]
                    - dist[a*n_cities+b] - dist[c*n_cities+d];

        // Boltzmann accept
        float threshold = (delta < 0.0f) ? 1.0f : expf(-delta / fmaxf(T, 1e-10f));
        if (lcg_f(&rng) < threshold) {
            // Reverse segment [i+1 .. j]
            int l = i+1, r = j;
            while (l < r) {
                int tmp = tour[l]; tour[l] = tour[r]; tour[r] = tmp;
                l++; r--;
            }
            cur_cost += delta;
            if (cur_cost < best_cost) best_cost = cur_cost;
        }

        if ((iter+1) % L == 0) {
            T *= alpha;
            if (T < T_min) T = T_min;
        }
    }

    costs[cid] = best_cost;
}
"""

_cache = {}

def _get_kernel():
    if 'tsp' not in _cache:
        mod = cp.RawModule(code=_TSP_KERNEL)
        _cache['tsp'] = mod.get_function('tsp_sa_kernel')
    return _cache['tsp']


def _run(n_cities, n_chains, dist_gpu, T0, T_min, L, max_iter, seed):
    tours = cp.zeros((n_chains, n_cities), dtype=cp.int32)
    costs = cp.full(n_chains, 1e18, dtype=cp.float32)
    kernel = _get_kernel()
    threads = 256
    blocks = (n_chains + threads - 1) // threads
    t0 = time.time()
    kernel(
        grid=(blocks,), block=(threads,),
        args=(tours, costs, dist_gpu,
              cp.int32(n_cities), cp.int32(n_chains),
              cp.float32(T0), cp.float32(T_min),
              cp.int32(max_iter), cp.int32(L),
              cp.uint32(seed))
    )
    cp.cuda.Stream.null.synchronize()
    return tours, costs, time.time() - t0


def gpu_sa_tsp_timed(n_cities=1000, seed=None, coords=None,
                     n_chains=512, T0=5000.0, T_min=1e-3, L=100,
                     time_budget=30.0, **kwargs):
    assert CUPY_AVAILABLE
    if seed is None: seed = random.randint(0, 2**31)
    if seed is not None: np.random.seed(seed)

    if coords is not None:
        coords_np = np.array(coords, dtype=np.float32)
    else:
        coords_np = np.random.rand(n_cities, 2).astype(np.float32) * 1000

    n = len(coords_np)
    c = coords_np
    diff = c[:,np.newaxis,:] - c[np.newaxis,:,:]
    dist_np = np.sqrt((diff**2).sum(axis=2)).astype(np.float32)
    dist_gpu = cp.asarray(dist_np)

    # Warmup
    WARMUP = 1000
    _, _, wt = _run(n, n_chains, dist_gpu, T0, T_min, L, WARMUP, seed)
    iters_per_sec = WARMUP / max(wt, 1e-6)
    gpu_max_iter = max(int(iters_per_sec * time_budget), WARMUP)

    # Full run
    tours, costs, elapsed = _run(n, n_chains, dist_gpu, T0, T_min, L, gpu_max_iter, seed)

    best_idx = int(cp.argmin(costs))
    best_tour = cp.asnumpy(tours[best_idx])
    best_cost = float(costs[best_idx])
    return best_tour, best_cost, elapsed, gpu_max_iter * n_chains


def cpu_sa_tsp_timed(n_cities=1000, seed=None, coords=None,
                     T0=5000.0, T_min=1e-3, L=100,
                     time_budget=30.0, **kwargs):
    if seed is not None:
        random.seed(seed); np.random.seed(seed)
    if coords is not None:
        coords_np = np.array(coords, dtype=np.float64)
    else:
        coords_np = np.random.rand(n_cities, 2).astype(np.float64) * 1000
    n = len(coords_np); c = coords_np
    dist_matrix = np.sqrt(((c[:,np.newaxis,:]-c[np.newaxis,:,:])**2).sum(axis=2))
    cost_fn = lambda t: float(dist_matrix[np.array(t), np.roll(np.array(t),-1)].sum())

    current = list(range(n)); random.shuffle(current)
    current_cost = cost_fn(current)
    best, best_cost = current[:], current_cost
    T = T0; iterations = 0
    t_start = time.time()

    while time.time() - t_start < time_budget:
        for _ in range(L):
            if time.time() - t_start >= time_budget: break
            i, j = sorted(random.sample(range(n), 2))
            new_tour = current[:]; new_tour[i:j+1] = reversed(new_tour[i:j+1])
            new_cost = cost_fn(new_tour); delta = new_cost - current_cost
            if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
                current, current_cost = new_tour, new_cost
                if current_cost < best_cost:
                    best, best_cost = current[:], current_cost
            iterations += 1
        T = max(T * 0.995, T_min)

    return best, best_cost, time.time() - t_start, iterations