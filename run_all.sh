#!/bin/bash
# ============================================================
# run_all.sh — AIPSA-GM 全实验一键重跑脚本
#
# Usage:
#   bash run_all.sh              # Full run
#   bash run_all.sh --dry-run    # Print commands only, no execution
#   bash run_all.sh --skip-gpu   # Skip GPU experiments
#
# Output directory: results/YYYY-MM-DD_HHMMSS/
# ============================================================

set -e

# ── Configuration ───────────────────────────────────────────────
CITIES=1000
DIMS=10
RUNS=5
ISLANDS=4
SEED=42

# GPU configuration
GPU_TIME=60
GPU_RUNS_RASTRIGIN=5
GPU_RUNS_TSP=10
GPU_CHAINS_TSP=32
CUDA_PATH_VAL=/usr/local/cuda-12.6
# ─────────────────────────────────────────────────────────────

DRY_RUN=false
SKIP_GPU=false

for arg in "$@"; do
    case $arg in
        --dry-run)  DRY_RUN=true ;;
        --skip-gpu) SKIP_GPU=true ;;
    esac
done

if [ "$DRY_RUN" = true ]; then
    echo "[dry-run mode] Print commands only, not executing"
fi

TIMESTAMP=$(date +"%Y-%m-%d_%H%M%S")
OUTDIR="results/${TIMESTAMP}"

echo ""
echo "============================================================"
echo "  AIPSA-GM 全实验重跑"
echo "  Output dir: ${OUTDIR}"
echo "  Config: TSP ${CITIES} cities | Rastrigin ${DIMS} dims"
echo "        runs=${RUNS} | islands=${ISLANDS} | seed=${SEED}"
echo "============================================================"
echo ""

run_cmd() {
    echo ""
    echo ">>> $*"
    if [ "$DRY_RUN" = false ]; then
        eval "$@"
    fi
}

mkdir -p "${OUTDIR}"

BASE_TSP="python3 -m experiments.run_experiment --problem tsp      --cities ${CITIES} --runs ${RUNS} --seed ${SEED} --islands ${ISLANDS} --outdir ${OUTDIR}"
BASE_RAS="python3 -m experiments.run_experiment --problem rastrigin --dims   ${DIMS}   --runs ${RUNS} --seed ${SEED} --islands ${ISLANDS} --outdir ${OUTDIR}"
BASE_RAS30="python3 -m experiments.run_experiment --problem rastrigin --dims 30 --runs ${RUNS} --seed ${SEED} --islands ${ISLANDS} --outdir ${OUTDIR}"

# ────────────────────────────────────────────────────────────
echo "════════════════════════════════════════════════════════════"
echo "  Exp 1: Solver Comparison"
echo "════════════════════════════════════════════════════════════"
run_cmd "${BASE_TSP}"
run_cmd "${BASE_RAS}"

# ────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Exp 2: Migration Policy"
echo "  TSP 2000 cities | Rastrigin 10d ring | Rastrigin 30d ring"
echo "════════════════════════════════════════════════════════════"
# TSP 2000 cities
run_cmd "python3 -m experiments.run_experiment --problem tsp --cities 2000 --runs ${RUNS} --seed ${SEED} --islands ${ISLANDS} --outdir ${OUTDIR} --exp2"
# Rastrigin 10d ring topology
run_cmd "${BASE_RAS} --exp2"
# Rastrigin 30d ring topology
run_cmd "${BASE_RAS30} --exp2"

# ────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Exp 3: Adaptive Temperature"
echo "════════════════════════════════════════════════════════════"
run_cmd "${BASE_TSP} --exp3"
run_cmd "${BASE_RAS} --exp3"

# ────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Exp 4: Async vs Sync Migration"
echo "════════════════════════════════════════════════════════════"
run_cmd "${BASE_TSP} --exp4"
run_cmd "${BASE_RAS} --exp4"

# ────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Exp 5: Topology Comparison (ring / full / random_k)"
echo "════════════════════════════════════════════════════════════"
run_cmd "${BASE_TSP} --exp5"
run_cmd "${BASE_RAS} --exp5"

# ────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Exp 5: Scalability (n_islands = 2, 4, 8, 16)"
echo "  TSP: full topology"
echo "  Rastrigin: ring / full / random_k 分别跑"
echo "════════════════════════════════════════════════════════════"
# TSP scalability (full topology — default)
run_cmd "python3 -m experiments.run_experiment --problem tsp --cities ${CITIES} --runs ${RUNS} --seed ${SEED} --outdir ${OUTDIR} --scale"

# Rastrigin scalability — ring
run_cmd "python3 -m experiments.run_experiment --problem rastrigin --dims ${DIMS} --runs ${RUNS} --seed ${SEED} --outdir ${OUTDIR}/scale_ring --scale"

# Rastrigin scalability — full
# Note: set TOPOLOGY="full" in run_experiment.py before running
echo ""
echo "  [NOTE] Before running Rastrigin full/random_k scalability,"
echo "         set TOPOLOGY to the corresponding value in run_experiment.py"
echo ""
run_cmd "python3 -m experiments.run_experiment --problem rastrigin --dims ${DIMS} --runs ${RUNS} --seed ${SEED} --outdir ${OUTDIR}/scale_full --scale"

# Rastrigin scalability — random_k
run_cmd "python3 -m experiments.run_experiment --problem rastrigin --dims ${DIMS} --runs ${RUNS} --seed ${SEED} --outdir ${OUTDIR}/scale_randomk --scale"

# ────────────────────────────────────────────────────────────
if [ "$SKIP_GPU" = false ]; then
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  Exp 6 (H5): GPU vs CPU — Rastrigin & TSP"
    echo "  Requires CUDA: RTX GPU + CUDA 12.6 + CuPy"
    echo "════════════════════════════════════════════════════════════"

    # Set CUDA environment
    run_cmd "export CUDA_PATH=${CUDA_PATH_VAL} && export PATH=\$CUDA_PATH/bin:\$PATH && export LD_LIBRARY_PATH=\$CUDA_PATH/lib64:\$LD_LIBRARY_PATH"

    # Rastrigin GPU experiment
    run_cmd "CUDA_PATH=${CUDA_PATH_VAL} PATH=${CUDA_PATH_VAL}/bin:\$PATH LD_LIBRARY_PATH=${CUDA_PATH_VAL}/lib64:\$LD_LIBRARY_PATH \
        python3 -m experiments.run_gpu_final \
        --time ${GPU_TIME} \
        --runs ${GPU_RUNS_RASTRIGIN} \
        --seed ${SEED} \
        --outdir ${OUTDIR}/gpu_rastrigin"

    # TSP GPU experiment
    run_cmd "CUDA_PATH=${CUDA_PATH_VAL} PATH=${CUDA_PATH_VAL}/bin:\$PATH LD_LIBRARY_PATH=${CUDA_PATH_VAL}/lib64:\$LD_LIBRARY_PATH \
        python3 -m experiments.run_gpu_tsp \
        --time ${GPU_TIME} \
        --runs ${GPU_RUNS_TSP} \
        --seed ${SEED} \
        --outdir ${OUTDIR}/gpu_tsp"

    # N_CHAINS sweep
    run_cmd "CUDA_PATH=${CUDA_PATH_VAL} PATH=${CUDA_PATH_VAL}/bin:\$PATH LD_LIBRARY_PATH=${CUDA_PATH_VAL}/lib64:\$LD_LIBRARY_PATH \
        python3 run_tsp_chain_sweep.py"
else
    echo ""
    echo "  [GPU skipped] Use --skip-gpu to skip GPU experiments"
fi

# ────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════"
if [ "$DRY_RUN" = false ]; then
    echo "  All done! Results saved to: ${OUTDIR}/"
    echo ""
    echo "  Generated directories:"
    ls -d "${OUTDIR}"/*/  2>/dev/null | sed 's/^/    /' || echo "    (none)"
else
    echo "  [dry-run] Command preview complete, nothing executed"
fi
echo "════════════════════════════════════════════════════════════"
echo ""