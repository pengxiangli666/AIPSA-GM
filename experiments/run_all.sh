#!/bin/bash
# ============================================================
# run_all.sh — AIPSA-GM 全实验一键重跑脚本
#
# 用法:
#   bash run_all.sh              # 正式版: TSP 1000 + Rastrigin 10, 5 runs
#   bash run_all.sh --dry-run    # 只打印命令，不实际执行
#
# 输出目录: results/YYYY-MM-DD_HHMMSS/
# 旧 results/ 下的 CSV 不会被删除或覆盖
# ============================================================

set -e  # 任何命令失败立即退出

# ── 配置区（按需修改）────────────────────────────────────────
CITIES=1000
DIMS=10
RUNS=5
ISLANDS=4
SEED=42
SCALE_ISLANDS="--scale"        # 去掉这行如果不想跑 scalability
# ─────────────────────────────────────────────────────────────

DRY_RUN=false
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "[dry-run mode] 只打印命令，不执行"
fi

# 生成带时间戳的输出目录
TIMESTAMP=$(date +"%Y-%m-%d_%H%M%S")
OUTDIR="results/${TIMESTAMP}"

echo ""
echo "============================================================"
echo "  AIPSA-GM 全实验重跑"
echo "  输出目录: ${OUTDIR}"
echo "  配置: TSP ${CITIES} cities | Rastrigin ${DIMS} dims"
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

# 先把旧数据 archive（只在非 dry-run 且旧数据存在时执行）
if [ "$DRY_RUN" = false ] && ls results/*.csv 2>/dev/null | grep -q .; then
    echo "── 归档旧 CSV 到 results/archive/ ──"
    mkdir -p results/archive
    mv results/*.csv results/archive/
    echo "   已移动 $(ls results/archive/*.csv 2>/dev/null | wc -l) 个文件到 results/archive/"
fi

mkdir -p "${OUTDIR}"

BASE_TSP="python -m experiments.run_experiment --problem tsp   --cities ${CITIES} --runs ${RUNS} --seed ${SEED} --islands ${ISLANDS} --outdir ${OUTDIR}"
BASE_RAS="python -m experiments.run_experiment --problem rastrigin --dims ${DIMS}  --runs ${RUNS} --seed ${SEED} --islands ${ISLANDS} --outdir ${OUTDIR}"

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Exp 1: Solver Comparison (Serial / Baseline A / B / AIPSA-GM)"
echo "════════════════════════════════════════════════════════════"
run_cmd "${BASE_TSP}"
run_cmd "${BASE_RAS}"

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Exp 2: Migration Policy (random / best_only / quality_only / guided)"
echo "════════════════════════════════════════════════════════════"
run_cmd "${BASE_TSP} --exp2"
run_cmd "${BASE_RAS} --exp2"

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Exp 3: Adaptive Temperature (fixed vs adaptive cooling)"
echo "════════════════════════════════════════════════════════════"
run_cmd "${BASE_TSP} --exp3"
run_cmd "${BASE_RAS} --exp3"

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Exp 4: Async vs Sync Migration"
echo "════════════════════════════════════════════════════════════"
run_cmd "${BASE_TSP} --exp4"
run_cmd "${BASE_RAS} --exp4"

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Exp 5: Topology (ring / full / random_k)"
echo "════════════════════════════════════════════════════════════"
run_cmd "${BASE_TSP} --exp5"
run_cmd "${BASE_RAS} --exp5"

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Exp 5 (Scalability): n_islands = 2, 4, 8, 16"
echo "════════════════════════════════════════════════════════════"
run_cmd "python -m experiments.run_experiment --problem tsp      --cities ${CITIES} --runs ${RUNS} --seed ${SEED} --outdir ${OUTDIR} ${SCALE_ISLANDS}"
run_cmd "python -m experiments.run_experiment --problem rastrigin --dims ${DIMS}   --runs ${RUNS} --seed ${SEED} --outdir ${OUTDIR} ${SCALE_ISLANDS}"

echo ""
echo "════════════════════════════════════════════════════════════"
if [ "$DRY_RUN" = false ]; then
    echo "  全部完成！结果保存在: ${OUTDIR}/"
    echo ""
    echo "  生成的 CSV 文件:"
    ls "${OUTDIR}"/*.csv 2>/dev/null | sed 's/^/    /' || echo "    (无文件)"
else
    echo "  [dry-run] 以上为完整命令预览，未实际执行"
fi
echo "════════════════════════════════════════════════════════════"
echo ""