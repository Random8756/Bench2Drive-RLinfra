#!/bin/bash
# On-policy (PPO / A2C) resume training.
#
# Loads a PPO/A2C checkpoint and continues training with the full policy
# weights intact.
#
# Usage:
#   bash tools/launch/baseline/resume_on_policy.sh ppo
#   bash tools/launch/baseline/resume_on_policy.sh a2c /path/to/checkpoint

set -euo pipefail

export CUDA_VISIBLE_DEVICES=0

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

export CARLA_ROOT=/path/to/carla
export TRAINING_RUNTIME="$REPO_ROOT/vendor/carla/training-runtime"
export SCENARIO_RUNNER_ROOT="$TRAINING_RUNTIME/scenario_runner"
export PYTHONPATH="$REPO_ROOT:$TRAINING_RUNTIME:$SCENARIO_RUNNER_ROOT:$CARLA_ROOT/PythonAPI/carla${PYTHONPATH:+:${PYTHONPATH}}"
# ============================================================
# Resume parameters (edit as needed)
# ============================================================
ARG="${1:-ppo}"
CHECKPOINT_PATH="${2:-path/to/checkpoint_dir}"

case "$ARG" in
    ppo)  CONFIG_PATH="${REPO_ROOT}/configs/ppo_bev_example.yaml" ;;
    a2c)  CONFIG_PATH="${REPO_ROOT}/configs/a2c_bev_example.yaml" ;;
    *)    CONFIG_PATH="$ARG" ;;   # treat as a literal path
esac
# ============================================================

if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "[resume_on_policy.sh] ERROR: config not found: $CONFIG_PATH" >&2
    exit 1
fi

CONFIG_STEM="$(basename "${CONFIG_PATH%.yaml}")"

LOG_DIR="output/${CONFIG_STEM}"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/$(date +%Y-%m-%d_%H-%M-%S).log"

echo "Config:      $CONFIG_PATH"
echo "Checkpoint:  $CHECKPOINT_PATH"
echo "Log:         $LOG_FILE"

python -m b2d_rlinfra.learning.training.resume_on_policy \
    --config "$CONFIG_PATH" \
    --checkpoint "$CHECKPOINT_PATH" \
    2>&1 | tee "$LOG_FILE"
