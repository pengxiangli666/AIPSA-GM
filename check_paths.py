import os

files = [
    'results/exp1_problem_TSP/tsp_1000cities_comparison.csv',
    'results/exp1_problem_RASTRIGIN/rastrigin_10dims_comparison.csv',
    'results/exp2_problem_TSP/tsp_2000cities_migration_policy.csv',
    'results/exp2_problem_RASTRIGIN_dims_10_ring/rastrigin_10dims_migration_policy.csv',
    'results/exp2_problem_RASTRIGIN_dims_30_ring/rastrigin_30dims_migration_policy.csv',
    'results/exp2_problem_RASTRIGIN_dims_30_full/rastrigin_30dims_migration_policy.csv',
    'results/exp3_problem_RASTRIGIN_10/rastrigin_10dims_adaptive_temp.csv',
    'results/exp3_problem_TSP_1000/tsp_1000cities_adaptive_temp.csv',
    'results/exp4_problem_TSP_1000/tsp_1000cities_async_vs_sync.csv',
    'results/exp4_problem_RASTRIGIN_10/rastrigin_10dims_async_vs_sync.csv',
    'results/exp5_3TOPOLOGY_TSP/tsp_1000cities_topology.csv',
    'results/exp5_3TOPOLOGY_RASTRIGIN/rastrigin_10dims_topology.csv',
    'results/exp5_scale_TSP_full_part2/tsp_1000cities_scalability.csv',
    'results/exp5_scale_ring_RASTRIGIN_part2/rastrigin_10dims_scalability.csv',
    'results/exp5_scale_full_RASTRIGIN_part2/rastrigin_10dims_scalability.csv',
    'results/exp5_scale_randomk_RASTRIGIN_part2/rastrigin_10dims_scalability.csv',
    'results/gpu_rastrigin_60time_5runs/gpu_final_rastrigin.csv',
    'results/gpu_TSP_N_CHAINS_32_10runs/gpu_tsp_boltzmann.csv',
    'results/gpu_sweep/chain_sweep.csv',
]

for f in files:
    print(("OK   " if os.path.exists(f) else "MISS "), f)