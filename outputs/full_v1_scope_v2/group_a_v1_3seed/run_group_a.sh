#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/moshiling/Adam-NSCL-C-Flat/Adam-NSCL-main"
OUTPUT_ROOT="/home/moshiling/Adam-NSCL-C-Flat/outputs/full_v1_scope_v2/group_a_v1_3seed"
PYTHON_BIN="/home/moshiling/miniconda3/envs/torch118/bin/python"
AGG_SCRIPT="/home/moshiling/Adam-NSCL-C-Flat/outputs/full_v1_scope_v2/aggregate_full_results.py"
GPU_ID="${GPU_ID:-0}"

mkdir -p "$OUTPUT_ROOT"
QUEUE_LOG="$OUTPUT_ROOT/queue_status.log"
HEARTBEAT="$OUTPUT_ROOT/heartbeat.log"

COMMON_ARGS=(
  --schedule 30 60 80
  --reg_coef 100
  --model_lr 1e-4
  --head_lr 1e-3
  --svd_lr 5e-5
  --bn_lr 5e-4
  --svd_thres 10
  --model_weight_decay 5e-5
  --agent_type svd_based
  --agent_name svd_based
  --dataset CIFAR100
  --gpuid 0
  --repeat 3
  --seed 0
  --model_optimizer Adam
  --force_out_dim 0
  --first_split_size 10
  --other_split_size 10
  --batch_size 32
  --model_name resnet18
  --model_type resnet
  --workers 0
  --print_freq 10
)

RUN_NAMES=(
  baseline_repeat3
  v1_repeat3
)

run_cmd() {
  local name="$1"
  shift
  local run_dir="$OUTPUT_ROOT/$name"
  mkdir -p "$run_dir"
  printf 'CUDA_VISIBLE_DEVICES=%s %q -u %q ' "$GPU_ID" "$PYTHON_BIN" "$PROJECT_DIR/main.py" > "$run_dir/command.sh"
  printf '%q ' "$@" >> "$run_dir/command.sh"
  printf '\n' >> "$run_dir/command.sh"
  chmod +x "$run_dir/command.sh"

  if [[ -f "$run_dir/final_metrics.json" ]]; then
    echo "$(date -Iseconds) SKIP $name existing_final_metrics" | tee -a "$QUEUE_LOG"
    "$PYTHON_BIN" "$AGG_SCRIPT" "$OUTPUT_ROOT" >/dev/null
    return
  fi

  if [[ -f "$run_dir/run.log" || -f "$run_dir/config_resolved.json" ]]; then
    echo "$(date -Iseconds) RESTART $name incomplete_previous_run" | tee -a "$QUEUE_LOG"
  else
    echo "$(date -Iseconds) START $name" | tee -a "$QUEUE_LOG"
  fi
  echo "$(date -Iseconds) running=$name gpu=$GPU_ID" >> "$HEARTBEAT"
  set +e
  (
    cd "$PROJECT_DIR"
    bash "$run_dir/command.sh" 2>&1 | tee "$run_dir/run.log"
  )
  local rc=$?
  set -e
  if [[ $rc -eq 0 && -f "$run_dir/final_metrics.json" ]]; then
    echo "$(date -Iseconds) DONE $name" | tee -a "$QUEUE_LOG"
  else
    echo "$(date -Iseconds) FAIL $name rc=$rc" | tee -a "$QUEUE_LOG"
    return $rc
  fi
  "$PYTHON_BIN" "$AGG_SCRIPT" "$OUTPUT_ROOT" >/dev/null
}

echo "$(date -Iseconds) queue_started gpu=$GPU_ID" | tee -a "$QUEUE_LOG"

run_cmd baseline_repeat3 \
  "${COMMON_ARGS[@]}" \
  --output_dir "$OUTPUT_ROOT/baseline_repeat3"

run_cmd v1_repeat3 \
  "${COMMON_ARGS[@]}" \
  --use_pls_cflat \
  --pls_cflat_mode layer_selective \
  --cflat_target_scope deep_plus_classifier \
  --deep_layer_rule last_stage \
  --cflat_rho 0.02 \
  --cflat_lambda 0.02 \
  --pls_cflat_debug \
  --output_dir "$OUTPUT_ROOT/v1_repeat3"

echo "$(date -Iseconds) queue_finished gpu=$GPU_ID" | tee -a "$QUEUE_LOG"
echo "$(date -Iseconds) idle gpu=$GPU_ID" >> "$HEARTBEAT"
