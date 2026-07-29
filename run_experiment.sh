#!/bin/bash
set -euo pipefail

SHUTDOWN_ON_EXIT="${MMRL_SHUTDOWN_ON_EXIT:-1}"

cancel_shutdown_on_interrupt() {
  SHUTDOWN_ON_EXIT=0
  trap - EXIT
  echo "[INT] 检测到 Ctrl+C，已取消自动关机。"
  exit 130
}

shutdown_on_exit() {
  local exit_code=$?
  if [ "${SHUTDOWN_ON_EXIT:-1}" != "1" ]; then
    return "$exit_code"
  fi
  echo "[EXIT] 脚本退出，exit_code=$exit_code"
  echo "[EXIT] 600 秒后自动关机。"
  echo "[EXIT] 如需取消自动关机，请在倒计时内按 Ctrl+C。"
  sleep 600
  /usr/bin/shutdown
}
trap shutdown_on_exit EXIT
trap cancel_shutdown_on_interrupt INT

ROOT_DIR="/root/autodl-tmp/Qwen3-VL-modify-test"
TRAIN_DIR="$ROOT_DIR/train"
OUTPUT_ROOT="$ROOT_DIR/experiment_outputs"
CHECKPOINT_ROOT="$OUTPUT_ROOT/output"
SLAKE_OUTPUT_ROOT="${SLAKE_OUTPUT_ROOT:-$ROOT_DIR/slake/outputs}"
SEED_STATE_FILE="${MMRL_SEED_STATE_FILE:-$OUTPUT_ROOT/next_seed.txt}"
INITIAL_SEED="${MMRL_INITIAL_SEED:-42}"
AUTO_INCREMENT_SEED="${MMRL_AUTO_INCREMENT_SEED:-0}"
FIXED_SEED="${MMRL_FIXED_SEED:-44}"
RUN_TARGET="${1:-${MMRL_RUN_TARGET:-all}}"

mkdir -p "$OUTPUT_ROOT" "$CHECKPOINT_ROOT" "$SLAKE_OUTPUT_ROOT"

RUN_SUFFIX="${MMRL_RUN_SUFFIX:-}"
RUN_DATE="${MMRL_RUN_DATE:-$(date +%Y%m%d)}"

# 持久分配实验 seed；脚本重启后继续递增，不重复使用已分配 seed。
allocate_seed() {
  local seed
  if [ "$AUTO_INCREMENT_SEED" = "0" ]; then
    if ! [[ "$FIXED_SEED" =~ ^[0-9]+$ ]]; then
      echo "[ERR] 非法固定 seed: '$FIXED_SEED'" >&2
      return 1
    fi
    printf '%s\n' "$FIXED_SEED"
    return 0
  fi
  if [ "$AUTO_INCREMENT_SEED" != "1" ]; then
    echo "[ERR] MMRL_AUTO_INCREMENT_SEED 必须为 0 或 1，当前为 '$AUTO_INCREMENT_SEED'" >&2
    return 1
  fi

  if [ -f "$SEED_STATE_FILE" ]; then
    read -r seed < "$SEED_STATE_FILE"
  else
    seed="$INITIAL_SEED"
  fi

  if ! [[ "$seed" =~ ^[0-9]+$ ]]; then
    echo "[ERR] 非法 seed 状态: $SEED_STATE_FILE 内容为 '$seed'" >&2
    return 1
  fi

  printf '%s\n' "$((seed + 1))" > "${SEED_STATE_FILE}.tmp"
  mv "${SEED_STATE_FILE}.tmp" "$SEED_STATE_FILE"
  printf '%s\n' "$seed"
}

with_run_suffix() {
  local base_tag="$1"
  if [ -n "$RUN_SUFFIX" ]; then
    echo "${base_tag}_${RUN_SUFFIX}_${RUN_DATE}"
  else
    echo "${base_tag}_${RUN_DATE}"
  fi
}

# 若目录已存在，自动追加 _1, _2, ... 直到找到可用名称
find_available_tag() {
  local base_tag="$1"
  local candidate="$base_tag"
  local i=1
  while [ -d "$CHECKPOINT_ROOT/$candidate" ]; do
    candidate="${base_tag}_${i}"
    i=$((i + 1))
  done
  echo "$candidate"
}

# 只保留当前仍有模型的并列最高分和并列最低分 checkpoint。
# 第一次运行时唯一 checkpoint 同时是最高和最低，因此必定保留。
# 无有效分数的 final 不会出现在计划中，也不会被误删。
prune_middle_final_dirs() {
  local retention_plan
  if ! retention_plan="$(cd "$ROOT_DIR" && python get_score.py --checkpoint-retention-plan)"; then
    echo "[WARN] 获取 checkpoint 保留计划失败，本轮不删除任何 final 目录。"
    return 0
  fi
  if [ -z "$retention_plan" ]; then
    echo "[WARN] 没有找到可比较的有效分数，本轮不删除任何 final 目录。"
    return 0
  fi

  local action tag score final_dir
  while IFS=$'\t' read -r action tag score; do
    [ -n "$action" ] || continue
    final_dir="$CHECKPOINT_ROOT/$tag/final"
    case "$action" in
      KEEP)
        echo "[KEEP] 保留极值 checkpoint: $tag score=$score"
        ;;
      PRUNE)
        if [ -d "$final_dir" ]; then
          echo "[PRUNE] 删除中间分 checkpoint: $tag score=$score"
          rm -rf -- "$final_dir"
        fi
        ;;
      *)
        echo "[WARN] 未知保留动作: $action tag=$tag，本条跳过。"
        ;;
    esac
  done <<< "$retention_plan"
}

run_one() {
  local experiment_name="$1"
  local raw_tag="$2"
  local decouple_stage_pooling="${MMRL_DECOUPLE_STAGE_POOLING:-0}"
  if [ "$decouple_stage_pooling" != "0" ] && [ "$decouple_stage_pooling" != "1" ]; then
    echo "[ERR] MMRL_DECOUPLE_STAGE_POOLING 必须为 0 或 1，当前为 '$decouple_stage_pooling'" >&2
    return 1
  fi
  local base_tag
  base_tag="$(with_run_suffix "$raw_tag")"
  local tag
  tag="$(find_available_tag "$base_tag")"
  local output_dir="$CHECKPOINT_ROOT/$tag"
  local eval_dir="$output_dir/eval"
  local experiment_seed
  experiment_seed="$(allocate_seed)"

  echo "============================================================"
  echo "[EXP] 开始实验: $tag"
  echo "[EXP] MMRL_EXPERIMENT=$experiment_name"
  echo "[EXP] seed=$experiment_seed"
  if [ "$AUTO_INCREMENT_SEED" = "1" ]; then
    echo "[EXP] seed_mode=auto_increment"
  else
    echo "[EXP] seed_mode=fixed"
  fi
  echo "[EXP] data_sampling_seed=42"
  echo "[EXP] data_order_seed=42"
  echo "[EXP] decouple_stage_pooling=$decouple_stage_pooling"
  echo "[EXP] checkpoint目录: $output_dir"
  echo "============================================================"

  mkdir -p "$output_dir" "$eval_dir"

  if ! (
    cd "$TRAIN_DIR"
    MMRL_OUTPUT_DIR="$output_dir" \
    MMRL_EXPERIMENT="$experiment_name" \
    MMRL_SEED="$experiment_seed" \
    MMRL_DATA_SAMPLING_SEED="42" \
    MMRL_DATA_ORDER_SEED="42" \
    MMRL_DETERMINISTIC_SAMPLING="1" \
    MMRL_DECOUPLE_STAGE_POOLING="$decouple_stage_pooling" \
    MMRL_EVAL_EACH_EPOCH="0" \
    MMRL_LIVE_FINAL_EVAL="1" \
    MMRL_SAVE_EXTREMA_CHECKPOINTS="1" \
    python train.py 2>&1 | tee "$output_dir/train.log"
  ); then
    echo "[ERR] 实验训练或在线测评失败: $tag" >&2
    return 1
  fi

  if [ ! -f "$eval_dir/test.log" ]; then
    echo "[ERR] 训练完成后未找到在线测评日志: $eval_dir/test.log"
    return 1
  fi

  # 在线测评后只保留全局并列最高/最低，清理已经失去极值资格的 final。
  # prune_middle_final_dirs
}

run_slake_full_all() {
  local data_root="${SLAKE_DATA_ROOT:-/root/autodl-tmp/dataset/slake}"
  local model_path="${MMRL_MODEL_PATH:-/root/autodl-tmp/model}"
  local slake_output_root="$SLAKE_OUTPUT_ROOT/mmrl"
  local base_tag
  base_tag="$(with_run_suffix "slake_mmrl_full_all_seed44")"
  local tag="$base_tag"
  local i=1
  while [ -d "$slake_output_root/$tag" ]; do
    tag="${base_tag}_${i}"
    i=$((i + 1))
  done
  local output_dir="$slake_output_root/$tag"
  local eval_dir="$output_dir/eval"
  local status_file="$output_dir/run_status.txt"
  local latest_pointer="$SLAKE_OUTPUT_ROOT/last_overnight_run.txt"

  mkdir -p "$output_dir" "$eval_dir"
  {
    echo "status=STARTED"
    echo "target=overnight_slake_pooling_pair_seed44"
    echo "tag=$tag"
    echo "output_dir=$output_dir"
    echo "started_at=$(date --iso-8601=seconds)"
  } > "$status_file"
  {
    echo "target=overnight_slake_pooling_pair_seed44"
    echo "tag=$tag"
    echo "output_dir=$output_dir"
    echo "status_file=$status_file"
    echo "train_log=$output_dir/train.log"
    echo "eval_log=$output_dir/eval.log"
  } > "$latest_pointer"
  echo "============================================================"
  echo "[SLAKE] 开始全量训练与官方测评: $tag"
  echo "[SLAKE] data_root=$data_root"
  echo "[SLAKE] language=all"
  echo "[SLAKE] seed=44 data_seed=42"
  echo "[SLAKE] decouple_stage_pooling=0"
  echo "[SLAKE] output_dir=$output_dir"
  echo "============================================================"

  if ! (
    cd "$ROOT_DIR"
    python slake/train_mmrl.py \
      --data-root "$data_root" \
      --model-path "$model_path" \
      --output-dir "$output_dir" \
      --experiment-name "slake_mmrl_full_all_seed44" \
      --language all \
      --seed 44 \
      --data-seed 42 \
      2>&1 | tee "$output_dir/train.log"
  ); then
    {
      echo "status=TRAIN_FAILED"
      echo "target=overnight_slake_pooling_pair_seed44"
      echo "tag=$tag"
      echo "output_dir=$output_dir"
      echo "failed_at=$(date --iso-8601=seconds)"
    } > "$status_file"
    echo "[ERR] SLAKE 全量训练失败: $tag" >&2
    return 1
  fi

  if [ ! -d "$output_dir/final" ]; then
    {
      echo "status=CHECKPOINT_MISSING"
      echo "target=overnight_slake_pooling_pair_seed44"
      echo "tag=$tag"
      echo "output_dir=$output_dir"
      echo "failed_at=$(date --iso-8601=seconds)"
    } > "$status_file"
    echo "[ERR] SLAKE 训练完成后未找到 checkpoint: $output_dir/final" >&2
    return 1
  fi

  {
    echo "status=EVALUATING"
    echo "target=overnight_slake_pooling_pair_seed44"
    echo "tag=$tag"
    echo "output_dir=$output_dir"
    echo "evaluation_started_at=$(date --iso-8601=seconds)"
  } > "$status_file"

  if ! (
    cd "$ROOT_DIR"
    python slake/slake_official_eval.py \
      --backend mmrl \
      --base-model "$model_path" \
      --checkpoint "$output_dir/final" \
      --questions "$data_root/test.json" \
      --image-root "$data_root/imgs" \
      --output-dir "$eval_dir" \
      --language all \
      --overwrite \
      2>&1 | tee "$output_dir/eval.log"
  ); then
    {
      echo "status=EVAL_FAILED"
      echo "target=overnight_slake_pooling_pair_seed44"
      echo "tag=$tag"
      echo "output_dir=$output_dir"
      echo "failed_at=$(date --iso-8601=seconds)"
    } > "$status_file"
    echo "[ERR] SLAKE 官方测评失败: $tag" >&2
    return 1
  fi

  {
    echo "status=COMPLETED"
    echo "target=overnight_slake_pooling_pair_seed44"
    echo "tag=$tag"
    echo "output_dir=$output_dir"
    echo "completed_at=$(date --iso-8601=seconds)"
  } > "$status_file"
  echo "[SLAKE] 全量训练与官方测评完成: $output_dir"
}

# 重复跑 N 次同一实验；目录命名由 find_available_tag 自动处理，不会覆写
# 用法: run_N <experiment_name> <raw_tag> <N>
run_N() {
  local experiment_name="$1"
  local raw_tag="$2"
  local n="$3"
  local i
  for ((i = 1; i <= n; i++)); do
    run_one "$experiment_name" "$raw_tag"
  done
}

# 依次运行显式给定的固定 seed；不读取或修改自动递增状态文件。
run_fixed_seed_sequence() {
  local experiment_name="$1"
  local raw_tag_prefix="$2"
  shift 2
  local seed
  AUTO_INCREMENT_SEED=0
  for seed in "$@"; do
    FIXED_SEED="$seed"
    run_one "$experiment_name" "${raw_tag_prefix}_seed${seed}"
  done
}

# 常用实验入口。
#   bash run_experiment.sh relation44
#   bash run_experiment.sh relation47
#   bash run_experiment.sh heterogeneous_lr_pair
#   bash run_experiment.sh heterogeneous_relation_100_102
#   bash run_experiment.sh multiturn_relation_100_102
#   bash run_experiment.sh joint_cosine_seed100
#   bash run_experiment.sh legacy_3cdf58d_seed100
#   bash run_experiment.sh fixed_stage1_constant_seed100
#   bash run_experiment.sh fixed_stage1_lr5e5_seed100
#   bash run_experiment.sh fixed_stage1_pooling_lr1e5_seed44
#   bash run_experiment.sh fixed_stage1_global_lr_half_seed44
#   bash run_experiment.sh fixed_stage1_mmrl_only_lr3e5_seed44
#   bash run_experiment.sh overnight_slake_pooling_pair_seed44
#   bash run_experiment.sh joint_cosine_1_4
#   bash run_experiment.sh joint_cosine_44_46
#   bash run_experiment.sh spatial_grounding_seed44
#   bash run_experiment.sh ce_only_seed44
#   bash run_experiment.sh usage_only_seed44
#   bash run_experiment.sh overnight_route_loss_3plus1
#   bash run_experiment.sh relation_2p5e3_repeat2
#   bash run_experiment.sh relation_5e3_repeat2
#   bash run_experiment.sh tune_mmrl_lr4e5_seed44
#   bash run_experiment.sh tune_mmrl_lr3e5_seed44
#   bash run_experiment.sh heterogeneous_no_relation_100_102
#   bash run_experiment.sh raw_adapter47
# 当前激进消融默认固定 seed 44：
#   bash run_experiment.sh direct_mmrl
#   bash run_experiment.sh two_adapter
#   bash run_experiment.sh single_adapter
case "$RUN_TARGET" in
  relation44)
    AUTO_INCREMENT_SEED=0
    FIXED_SEED=44
    run_one "visual_router_relation_alignment_diag" "visual_router_relation_retest_seed44"
    ;;
  relation47)
    AUTO_INCREMENT_SEED=0
    FIXED_SEED=47
    run_one "visual_router_relation_alignment_diag" "visual_router_relation_retest_seed47"
    ;;
  heterogeneous_lr_pair)
    AUTO_INCREMENT_SEED=0
    FIXED_SEED=44
    run_one "visual_router_relation_heterogeneous_adapter_lr_v1" "visual_router_relation_heterogeneous_adapter_lr_v1_seed44"
    FIXED_SEED=48
    run_one "visual_router_relation_heterogeneous_adapter_lr_v1" "visual_router_relation_heterogeneous_adapter_lr_v1_seed48"
    ;;
  heterogeneous_relation_100_102)
    run_fixed_seed_sequence \
      "visual_router_relation_heterogeneous_adapter_lr_v1" \
      "visual_router_relation_heterogeneous_adapter_lr_v1_paired" \
      100 101 102
    ;;
  multiturn_relation_seed100)
    AUTO_INCREMENT_SEED=0
    FIXED_SEED=100
    run_one \
      "visual_router_relation_multiturn_fixed_v1" \
      "visual_router_relation_multiturn_fixed_v1_seed100"
    ;;
  multiturn_relation_seed101)
    AUTO_INCREMENT_SEED=0
    FIXED_SEED=101
    run_one \
      "visual_router_relation_multiturn_fixed_v1" \
      "visual_router_relation_multiturn_fixed_v1_seed101"
    ;;
  multiturn_relation_100_102)
    run_fixed_seed_sequence \
      "visual_router_relation_multiturn_fixed_v1" \
      "visual_router_relation_multiturn_fixed_v1" \
      100 101 102
    ;;
  joint_cosine_seed100)
    AUTO_INCREMENT_SEED=0
    FIXED_SEED=100
    run_one \
      "visual_router_relation_joint_multiturn_cosine_v2" \
      "visual_router_relation_joint_multiturn_cosine_v2_seed100"
    ;;
  legacy_3cdf58d_seed100)
    AUTO_INCREMENT_SEED=0
    FIXED_SEED=100
    run_one \
      "visual_router_legacy_3cdf58d_joint_cosine_v2" \
      "visual_router_legacy_3cdf58d_joint_cosine_seed100"
    ;;
  fixed_stage1_constant_seed100)
    AUTO_INCREMENT_SEED=0
    FIXED_SEED=100
    run_one \
      "visual_router_fixed_stage1_constant_v1" \
      "visual_router_fixed_stage1_constant_v1_seed100"
    ;;
  fixed_stage1_lr5e5_seed100)
    AUTO_INCREMENT_SEED=0
    FIXED_SEED=100
    run_one \
      "visual_router_fixed_stage1_lr5e5_v1" \
      "visual_router_fixed_stage1_lr5e5_v1_seed100"
    ;;
  fixed_stage1_pooling_lr1e5_seed44)
    AUTO_INCREMENT_SEED=0
    FIXED_SEED=44
    run_one \
      "visual_router_fixed_stage1_pooling_lr1e5_v1" \
      "visual_router_fixed_stage1_pooling_lr1e5_v1_seed44"
    ;;
  fixed_stage1_global_lr_half_seed44)
    AUTO_INCREMENT_SEED=0
    FIXED_SEED=44
    run_one \
      "visual_router_fixed_stage1_global_lr_half_v1" \
      "visual_router_fixed_stage1_global_lr_half_v1_seed44"
    ;;
  fixed_stage1_mmrl_only_lr3e5_seed44)
    AUTO_INCREMENT_SEED=0
    FIXED_SEED=44
    run_one \
      "visual_router_fixed_stage1_mmrl_only_lr3e5_v1" \
      "visual_router_fixed_stage1_mmrl_only_lr3e5_v1_seed44"
    ;;
  overnight_slake_pooling_pair_seed44)
    echo "[OVERNIGHT] target=$RUN_TARGET"
    echo "[OVERNIGHT] SLAKE output root=$SLAKE_OUTPUT_ROOT/mmrl"
    echo "[OVERNIGHT] latest run pointer=$SLAKE_OUTPUT_ROOT/last_overnight_run.txt"
    overnight_failures=0
    if ! run_slake_full_all; then
      echo "[OVERNIGHT-WARN] SLAKE 全量实验失败，继续共享 pooling 实验。" >&2
      overnight_failures=$((overnight_failures + 1))
    fi
    AUTO_INCREMENT_SEED=0
    FIXED_SEED=44
    MMRL_DECOUPLE_STAGE_POOLING=0
    if ! run_one \
      "visual_router_fixed_stage1_constant_v1" \
      "visual_router_fixed_stage1_constant_shared_pooling_seed44"; then
      echo "[OVERNIGHT-WARN] 共享 pooling 实验失败，继续解耦 pooling 实验。" >&2
      overnight_failures=$((overnight_failures + 1))
    fi
    MMRL_DECOUPLE_STAGE_POOLING=1
    if ! run_one \
      "visual_router_fixed_stage1_constant_v1" \
      "visual_router_fixed_stage1_constant_decoupled_pooling_seed44"; then
      echo "[OVERNIGHT-WARN] 解耦 pooling 实验失败。" >&2
      overnight_failures=$((overnight_failures + 1))
    fi
    if [ "$overnight_failures" -ne 0 ]; then
      echo "[OVERNIGHT-SUMMARY] 三轮均已尝试，失败数=$overnight_failures；即将按原计划自动关机。" >&2
      exit 1
    fi
    echo "[OVERNIGHT-SUMMARY] 三轮实验全部成功；即将按原计划自动关机。"
    ;;
  joint_cosine_1_4)
    run_fixed_seed_sequence \
      "visual_router_relation_joint_multiturn_cosine_v2" \
      "visual_router_relation_joint_multiturn_cosine" \
      1 2 3 4
    ;;
  joint_cosine_44_46)
    run_fixed_seed_sequence \
      "visual_router_relation_joint_multiturn_cosine_v2" \
      "visual_router_relation_joint_multiturn_cosine_clean_1_14" \
      44 45 46
    ;;
  spatial_grounding_seed44)
    AUTO_INCREMENT_SEED=0
    FIXED_SEED=44
    run_one \
      "visual_router_spatial_grounding_v1" \
      "visual_router_spatial_grounding_v1_seed44"
    ;;
  ce_only_seed44)
    AUTO_INCREMENT_SEED=0
    FIXED_SEED=44
    run_one \
      "visual_router_joint_multiturn_cosine_ce_only_v1" \
      "visual_router_joint_multiturn_cosine_ce_only_v1_seed44"
    ;;
  usage_only_seed44)
    AUTO_INCREMENT_SEED=0
    FIXED_SEED=44
    run_one \
      "visual_router_joint_multiturn_cosine_usage_only_v1" \
      "visual_router_joint_multiturn_cosine_usage_only_v1_seed44"
    ;;
  overnight_route_loss_3plus1)
    AUTO_INCREMENT_SEED=0
    FIXED_SEED=44
    run_one \
      "visual_router_joint_multiturn_cosine_usage_1e2_v1" \
      "visual_router_joint_multiturn_cosine_usage_1e2_v1_seed44"
    run_one \
      "visual_router_joint_multiturn_cosine_entropy_only_v1" \
      "visual_router_joint_multiturn_cosine_entropy_only_v1_seed44"
    run_one \
      "visual_router_joint_multiturn_cosine_route_pair_v1" \
      "visual_router_joint_multiturn_cosine_route_pair_v1_seed44_repeat_a"
    run_one \
      "visual_router_joint_multiturn_cosine_route_pair_v1" \
      "visual_router_joint_multiturn_cosine_route_pair_v1_seed44_repeat_b"
    ;;
  relation_2p5e3_repeat2)
    AUTO_INCREMENT_SEED=0
    FIXED_SEED=44
    run_one \
      "visual_router_joint_multiturn_cosine_route_pair_relation_2p5e3_v1" \
      "visual_router_route_pair_relation_2p5e3_seed44_repeat_a"
    run_one \
      "visual_router_joint_multiturn_cosine_route_pair_relation_2p5e3_v1" \
      "visual_router_route_pair_relation_2p5e3_seed44_repeat_b"
    ;;
  relation_5e3_repeat2)
    AUTO_INCREMENT_SEED=0
    FIXED_SEED=44
    run_one \
      "visual_router_joint_multiturn_cosine_route_pair_relation_5e3_v1" \
      "visual_router_route_pair_relation_5e3_seed44_repeat_a"
    run_one \
      "visual_router_joint_multiturn_cosine_route_pair_relation_5e3_v1" \
      "visual_router_route_pair_relation_5e3_seed44_repeat_b"
    ;;
  tune_mmrl_lr4e5_seed44)
    AUTO_INCREMENT_SEED=0
    FIXED_SEED=44
    run_one \
      "visual_router_relation_joint_multiturn_cosine_mmrl_lr4e5_v1" \
      "visual_router_relation_joint_multiturn_cosine_mmrl_lr4e5_v1_seed44"
    ;;
  tune_mmrl_lr3e5_seed44)
    AUTO_INCREMENT_SEED=0
    FIXED_SEED=44
    run_one \
      "visual_router_relation_joint_multiturn_cosine_mmrl_lr3e5_v1" \
      "visual_router_relation_joint_multiturn_cosine_mmrl_lr3e5_v1_seed44"
    ;;
  joint_cosine_no_usage_seed44)
    AUTO_INCREMENT_SEED=0
    FIXED_SEED=44
    run_one \
      "visual_router_relation_joint_multiturn_cosine_no_usage_v3" \
      "visual_router_relation_joint_multiturn_cosine_no_usage_v3_seed44"
    ;;
  heterogeneous_no_relation_100_102)
    run_fixed_seed_sequence \
      "visual_router_no_relation_heterogeneous_adapter_lr_v1" \
      "visual_router_no_relation_heterogeneous_adapter_lr_v1_paired" \
      100 101 102
    ;;
  raw_adapter47)
    AUTO_INCREMENT_SEED=0
    FIXED_SEED=47
    run_one "visual_router_raw_adapter_v1" "visual_router_raw_adapter_v1_seed47"
    ;;
  direct_mmrl)
    run_one "visual_router_direct_mmrl_seed44" "visual_router_direct_mmrl_seed44"
    ;;
  two_adapter)
    run_one "visual_router_two_adapter_seed44" "visual_router_two_adapter_seed44"
    ;;
  single_adapter)
    run_one "visual_router_single_adapter_seed44" "visual_router_single_adapter_seed44"
    ;;
  all)
    AUTO_INCREMENT_SEED=0
    FIXED_SEED=47
    run_one "visual_router_relation_alignment_diag" "visual_router_relation_retest_seed47"
    run_one "visual_router_raw_adapter_v1" "visual_router_raw_adapter_v1_seed47"
    ;;
  *)
    echo "[ERR] 未知实验目标: $RUN_TARGET（可选: relation44, relation47, heterogeneous_lr_pair, heterogeneous_relation_100_102, multiturn_relation_seed100, multiturn_relation_seed101, multiturn_relation_100_102, joint_cosine_seed100, legacy_3cdf58d_seed100, fixed_stage1_constant_seed100, fixed_stage1_lr5e5_seed100, fixed_stage1_pooling_lr1e5_seed44, fixed_stage1_global_lr_half_seed44, fixed_stage1_mmrl_only_lr3e5_seed44, overnight_slake_pooling_pair_seed44, joint_cosine_1_4, joint_cosine_44_46, spatial_grounding_seed44, heterogeneous_no_relation_100_102, raw_adapter47, direct_mmrl, two_adapter, single_adapter, all）" >&2
    exit 2
    ;;
esac




# run_one "ablation_full_model" "ablation_full_model"
# run_one "ablation_wo_visual_gate" "ablation_wo_visual_gate"
# run_one "ablation_replace_mmrl_with_40_learnable_tokens" "ablation_replace_mmrl_with_40_learnable_tokens"

echo "[DONE] 已完成实验目标: $RUN_TARGET"
