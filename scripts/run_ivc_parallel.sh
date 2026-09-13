#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
CODE_ROOT="$PROJECT_ROOT"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/data/corrected_split}"
PYTHON="${PYTHON:-python3}"
LOG_DIR="$CODE_ROOT/logs/parallel"
JOB_DIR="/tmp/ivc_parallel_jobs"
MAX_JOBS=${MAX_JOBS:-3}

mkdir -p "$LOG_DIR" "$JOB_DIR"
cd "$CODE_ROOT"
export PYTHONPATH=src

echo "[ivc-parallel] start $(date -Is) max_jobs=$MAX_JOBS"

write_job() {
  local run_name=$1
  local model=$2
  local seed=$3
  local batch=$4
  local lr=$5
  local results_dir=$6
  local out="$JOB_DIR/${run_name}_${model}_seed${seed}.json"
  cat > "$out" <<JSON
{
  "run_name": "$run_name",
  "data_root": "$DATA_ROOT",
  "label_mode": "three_class",
  "image_size": 224,
  "models": ["$model"],
  "seeds": [$seed],
  "num_epochs": 30,
  "checkpoint_metric": "val_viral_recall",
  "loss": "weighted_cross_entropy",
  "class_weight_power": 0.5,
  "train_fraction": 1.0,
  "batch_size": $batch,
  "learning_rate": $lr,
  "weight_decay": 0.0001,
  "dropout": 0.2,
  "pretrained_baselines": true,
  "num_workers": 2,
  "device": "auto",
  "calibrate": true,
  "save_predictions": true,
  "results_dir": "$results_dir"
}
JSON
  echo "$out"
}

JOBS=()
for seed in 42 2026 3407; do
  JOBS+=("$(write_job pedilite_ms_ablation pedilite_ms_se $seed 32 0.001 results/pedilite_ms_ablation)")
  JOBS+=("$(write_job pedilite_ms_ablation pedilite_ms_k5 $seed 32 0.001 results/pedilite_ms_ablation)")
  JOBS+=("$(write_job light_family_weighted mobilenet_v2 $seed 16 0.0003 results/light_family_weighted)")
  JOBS+=("$(write_job light_family_weighted shufflenet_v2_x1_0 $seed 16 0.0003 results/light_family_weighted)")
  JOBS+=("$(write_job light_family_weighted efficientnet_b0 $seed 16 0.0003 results/light_family_weighted)")
done

echo "[ivc-parallel] queued ${#JOBS[@]} runs"

running=0
for job in "${JOBS[@]}"; do
  tag=$(basename "$job" .json)
  metrics_guess=""
  while [ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]; do
    wait -n || true
  done
  echo "[ivc-parallel] launch $tag $(date -Is)"
  "$PYTHON" scripts/02_run_experiment.py --config "$job" > "$LOG_DIR/${tag}.out" 2>&1 &
done

wait
echo "[ivc-parallel] all finished $(date -Is)"
