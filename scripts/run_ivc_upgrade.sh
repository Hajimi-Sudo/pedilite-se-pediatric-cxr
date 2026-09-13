#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
CODE_ROOT="$PROJECT_ROOT"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/data/corrected_split}"
PYTHON="${PYTHON:-python3}"
LOG_DIR="$CODE_ROOT/logs"
RESULTS_ROOT="$CODE_ROOT/results"

mkdir -p "$LOG_DIR" "$RESULTS_ROOT"
cd "$CODE_ROOT"
export PYTHONPATH=src

echo "[ivc] start $(date -Is) host=$(hostname)"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv || true

"$PYTHON" - <<'PY'
from pedilite.models import build_model, parse_pedilite_ms_name
from pedilite.utils import count_parameters, estimate_flops
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device", device)
names = [
    "pedilite_ms",
    "pedilite_ms_se",
    "pedilite_ms_k5",
    "pedilite_ms_k3",
    "pedilite_se_v2",
    "pedilite_se_v2_none",
    "mobilenet_v2",
    "shufflenet_v2_x1_0",
    "efficientnet_b0",
]
for name in names:
    model = build_model(name, 3, pretrained_baselines=False).to(device)
    dummy = torch.zeros(1, 3, 224, 224, device=device)
    out = model(dummy)
    params = count_parameters(model)
    flops = estimate_flops(model, 224, device)
    print(f"{name:24s} params={params:9d} flops={flops:12d} out={tuple(out.shape)}")
    if name.startswith("pedilite_ms"):
        print("  parse", parse_pedilite_ms_name(name))
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
print("smoke ok")
PY

prepare_config() {
  local source=$1
  local target=$2
  sed "s|CHANGE_ME_TO_CORRECTED_SPLIT_PATH|$DATA_ROOT|" "$CODE_ROOT/configs/$source" > "/tmp/$target"
}

run_config() {
  local source=$1
  local tag=$2
  prepare_config "$source" "$tag.json"
  echo "[ivc] $tag started $(date -Is)"
  "$PYTHON" scripts/02_run_experiment.py --config "/tmp/$tag.json" \
    > "$LOG_DIR/${tag}.out" 2>&1
  echo "[ivc] $tag finished $(date -Is)"
}

run_config pedilite_ms_matched_weighted_3seed.json pedilite_ms_matched
run_config pedilite_ms_ablation_3seed.json pedilite_ms_ablation
run_config light_family_weighted_3seed.json light_family
run_config recent_models_weighted_3seed.json recent_models

echo "[ivc] all queues finished $(date -Is)"
