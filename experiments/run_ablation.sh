#!/bin/bash
# 消融实验: 逐个关闭方法论组件, 检查完整方法的增益是否依赖这些组件。
# 场景选 A —— 四臂基准里学习臂相对两条规则臂都显著胜出的唯一场景之一。
# 规则臂不使用 SAC 奖励, 不受这些开关影响, 因此只重跑两条学习臂;
# 对照直接取 results/four_arm_benchmark.json 里场景 A 的完整方法结果。
cd "$(dirname "$0")/.."
PY=/d/Miniconda/envs/power_trading/python.exe

run_variant() {
  local V=$1; shift
  echo "=== variant $V start $(date) ==="
  PYTHONIOENCODING=utf-8 "$PY" experiments/run_four_arm_benchmark.py \
    --scenarios A --seeds 42,142,242 --episodes 2000 --eval-worlds 10 \
    --arms sac_rule,dual_sac --model-subdir "ablation_$V" \
    --out "results/parts/ablation_$V.json" "$@"
  echo "=== variant $V done rc=$? $(date) ==="
}

run_variant no_exposure_shaping --no-exposure-shaping
run_variant no_spot_reward_scale --no-spot-reward-scale
run_variant no_alternating --no-alternating --arms dual_sac
