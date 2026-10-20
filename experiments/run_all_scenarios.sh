#!/bin/bash
# 逐场景跑四臂基准并分片落盘 (中断只丢一个场景), 跑完合并。
# 用法: bash experiments/run_all_scenarios.sh
#   可用环境变量覆盖: PY (python 可执行文件), SEEDS, EPISODES, EVAL_WORLDS, OUTDIR
set -u
cd "$(dirname "$0")/.."
PY=${PY:-python}
SEEDS=${SEEDS:-42,142,242}
EPISODES=${EPISODES:-2000}
EVAL_WORLDS=${EVAL_WORLDS:-10}
OUTDIR=${OUTDIR:-results/parts}
mkdir -p "$OUTDIR"

for S in A B C D E; do
  echo "=== scenario $S start $(date) ==="
  PYTHONIOENCODING=utf-8 "$PY" experiments/run_four_arm_benchmark.py \
      --scenarios "$S" --seeds "$SEEDS" --episodes "$EPISODES" --eval-worlds "$EVAL_WORLDS" \
      --model-subdir four_arm --out "$OUTDIR/four_arm_$S.json"
  echo "=== scenario $S done rc=$? $(date) ==="
done

PYTHONIOENCODING=utf-8 "$PY" experiments/merge_four_arm_parts.py \
    --parts "$OUTDIR/four_arm_*.json" --out results/four_arm_benchmark.json
