#!/bin/bash

set -e

CITIES=1000
DIMS=10
RUNS=10
ISLANDS=4
SEED=42

# GPU configuration
GPU_TIME=60
GPU_RUNS_RASTRIGIN=5
GPU_RUNS_TSP=10
GPU_CHAINS=32

DRY_RUN=false
SKIP_GPU=false
for arg in "$@"; do
    [[ "$arg" == "--dry-run"  ]] && DRY_RUN=true
    [[ "$arg" == "--skip-gpu" ]] && SKIP_GPU=true
done

$DRY_RUN  && echo "[dry-run mode] printing commands only, not executing"
$SKIP_GPU && echo "[skip-gpu mode] skipping Exp 6 GPU experiments"

TIMESTAMP=$(date +"%Y-%m-%d_%H%M%S")
OUTDIR="results/${TIMESTAMP}"

echo ""
echo "============================================================"
echo "  AIPSA-GM Full Experiment Run (Exp 1-6)"
echo "  Output directory: ${OUTDIR}"
echo "  Config: TSP ${CITIES} cities | Rastrigin ${DIMS} dims"
echo "          runs=${RUNS} | islands=${ISLANDS} | seed=${SEED}"
echo "============================================================"
echo ""

run_cmd() {
    echo ""
    echo ">>> $*"
    if [ "$DRY_RUN" = false ]; then
        eval "$@"
    fi
}

# Archive old CSV files if present
if [ "$DRY_RUN" = false ] && ls results/*.csv 2>/dev/null | grep -q .; then
    echo "-- Archiving existing CSVs to results/archive/ --"
    mkdir -p results/archive
    mv results/*.csv results/archive/
fi

mkdir -p "${OUTDIR}"

BASE_TSP="python -m experiments.run_experiment --problem tsp      --cities ${CITIES} --runs ${RUNS} --seed ${SEED} --islands ${ISLANDS} --outdir ${OUTDIR}"
BASE_RAS="python -m experiments.run_experiment --problem rastrigin --dims   ${DIMS}   --runs ${RUNS} --seed ${SEED} --islands ${ISLANDS} --outdir ${OUTDIR}"


echo ""
echo "==== Exp 1: Solver Comparison (Serial / Baseline A / B / AIPSA-GM) ===="
# Default topology: full
run_cmd "${BASE_TSP}"
run_cmd "${BASE_RAS}"

echo ""
echo "==== Exp 2: Migration Policy (random / best_only / quality_only / guided) ===="
# TSP: full topology (default)
run_cmd "${BASE_TSP} --exp2 --cities 2000"
# Rastrigin 10d: requires TOPOLOGY='ring' in run_experiment.py
echo ">>> [NOTE] Set TOPOLOGY='ring' in run_experiment.py before running Rastrigin Exp2"
run_cmd "${BASE_RAS} --exp2"
# Rastrigin 30d: ring topology
run_cmd "${BASE_RAS} --exp2 --dims 30"

echo ""
echo "==== Exp 3: Adaptive Temperature (fixed vs adaptive cooling) ===="
# Rastrigin: requires TOPOLOGY='full' in run_experiment.py
echo ">>> [NOTE] Set TOPOLOGY='full' in run_experiment.py before running Rastrigin Exp3"
run_cmd "${BASE_RAS} --exp3"
# TSP: full topology (default)
run_cmd "${BASE_TSP} --exp3"


echo ""
echo "==== Exp 4: Async vs Sync Migration ===="
run_cmd "${BASE_TSP} --exp4"
run_cmd "${BASE_RAS} --exp4"


echo ""
echo "==== Exp 5: Topology Comparison (ring / full / random_k) ===="
run_cmd "${BASE_TSP} --exp5"
run_cmd "${BASE_RAS} --exp5"

echo ""
echo "==== Exp 5: Scalability (n_islands = 2, 4, 8, 16) ===="
# TSP scalability: full topology (default)
run_cmd "python -m experiments.run_experiment --problem tsp      --cities ${CITIES} --runs ${RUNS} --seed ${SEED} --outdir ${OUTDIR}/exp5_scale_tsp     --scale"
# Rastrigin scalability: ring topology
echo ">>> [NOTE] Set TOPOLOGY='ring' in run_experiment.py before running Rastrigin scalability (ring)"
run_cmd "python -m experiments.run_experiment --problem rastrigin --dims ${DIMS}    --runs ${RUNS} --seed ${SEED} --outdir ${OUTDIR}/exp5_scale_ring    --scale"
# Rastrigin scalability: full topology
echo ">>> [NOTE] Set TOPOLOGY='full' in run_experiment.py before running Rastrigin scalability (full)"
run_cmd "python -m experiments.run_experiment --problem rastrigin --dims ${DIMS}    --runs ${RUNS} --seed ${SEED} --outdir ${OUTDIR}/exp5_scale_full    --scale"
# Rastrigin scalability: random_k topology
echo ">>> [NOTE] Set TOPOLOGY='random_k' in run_experiment.py before running Rastrigin scalability (random_k)"
run_cmd "python -m experiments.run_experiment --problem rastrigin --dims ${DIMS}    --runs ${RUNS} --seed ${SEED} --outdir ${OUTDIR}/exp5_scale_randomk --scale"


if [ "$SKIP_GPU" = false ]; then
    echo ""
    echo "==== Exp 6: GPU Parallelism ===="

    # Set CUDA environment
    export CUDA_PATH=/usr/local/cuda-12.6
    export PATH=$CUDA_PATH/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH

    # Rastrigin GPU: 60s budget, 5 runs
    run_cmd "python3 -m experiments.run_gpu_final --time ${GPU_TIME} --runs ${GPU_RUNS_RASTRIGIN} --outdir ${OUTDIR}/gpu_rastrigin"

    # TSP GPU: 60s budget, 10 runs, N_CHAINS=32
    run_cmd "python3 -m experiments.run_gpu_tsp --time ${GPU_TIME} --runs ${GPU_RUNS_TSP} --outdir ${OUTDIR}/gpu_tsp"

    # N_CHAINS sweep to find optimal chain count
    run_cmd "python3 run_tsp_chain_sweep.py"
else
    echo ""
    echo "==== Exp 6: GPU experiments skipped ===="
fi


echo ""
echo "============================================================"
if [ "$DRY_RUN" = false ]; then
    echo "  All experiments complete. Results saved to: ${OUTDIR}/"
    echo ""
    echo "  Output files:"
    find "${OUTDIR}" -name "*.csv" 2>/dev/null | sed 's/^/    /' || echo "    (no files found)"
else
    echo "  [dry-run] Command preview complete, nothing was executed."
fi
echo "============================================================"
echo ""