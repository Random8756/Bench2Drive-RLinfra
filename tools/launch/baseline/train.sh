#!/bin/bash
# Unified training script.
#
# Usage:
#   bash train.sh                          # PPO (default config)
#   bash train.sh ppo|a2c|sac|td3          # config alias
#   bash train.sh /path/to/custom.yaml     # arbitrary config path
#
# GPU can be overridden via env:
#   CUDA_VISIBLE_DEVICES=2 bash train.sh ppo

set -euo pipefail

# ── GPU ─────────────────────────────────────────────────────────────────────
: "${CUDA_VISIBLE_DEVICES:=0}"
export CUDA_VISIBLE_DEVICES

# ── Paths ────────────────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

export CARLA_ROOT=/path/to/carla
export TRAINING_RUNTIME="$REPO_ROOT/vendor/carla/training-runtime"
export SCENARIO_RUNNER_ROOT="$TRAINING_RUNTIME/scenario_runner"
export PYTHONPATH="$REPO_ROOT:$TRAINING_RUNTIME:$SCENARIO_RUNNER_ROOT:$CARLA_ROOT/PythonAPI/carla${PYTHONPATH:+:${PYTHONPATH}}"
# ── Config resolution ────────────────────────────────────────────────────────
ARG="${1:-ppo}"   # first positional arg: algorithm shortname or full path

case "$ARG" in
    ppo)  CONFIG_PATH="${REPO_ROOT}/configs/ppo_bev_example.yaml" ;;
    a2c)  CONFIG_PATH="${REPO_ROOT}/configs/a2c_bev_example.yaml" ;;
    sac)  CONFIG_PATH="${REPO_ROOT}/configs/sac_bev_example.yaml" ;;
    td3)  CONFIG_PATH="${REPO_ROOT}/configs/td3_bev_example.yaml" ;;
    *)    CONFIG_PATH="$ARG" ;;   # treat as a literal path
esac

if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "[train.sh] ERROR: config not found: $CONFIG_PATH" >&2
    exit 1
fi

CONFIG_STEM="$(basename "${CONFIG_PATH%.yaml}")"

# ── Logging ──────────────────────────────────────────────────────────────────
LOG_DIR="${REPO_ROOT}/output/${CONFIG_STEM}"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/$(date +%Y-%m-%d_%H-%M-%S).log"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " Config : $CONFIG_PATH"
echo " GPU    : $CUDA_VISIBLE_DEVICES"
echo " Log    : $LOG_FILE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── Launch ────────────────────────────────────────────────────────────────────
python -m b2d_rlinfra.learning.training.train \
    --config "$CONFIG_PATH" \
    2>&1 | tee "$LOG_FILE"
