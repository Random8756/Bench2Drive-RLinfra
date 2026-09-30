#!/bin/bash
# Environment rollout demo.
#
# Usage:
#   bash b2d_rlinfra/framework/env_rollout_demo.sh
#
# Rollout behavior is configured in:
#   configs/env_rollout_demo.yaml

set -euo pipefail

# ── GPU ─────────────────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=0

# ── Paths ────────────────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

export CARLA_ROOT=/path/to/carla
export TRAINING_RUNTIME="$REPO_ROOT/vendor/carla/training-runtime"
export SCENARIO_RUNNER_ROOT="$TRAINING_RUNTIME/scenario_runner"
export PYTHONPATH="$REPO_ROOT:$TRAINING_RUNTIME:$SCENARIO_RUNNER_ROOT:$CARLA_ROOT/PythonAPI/carla${PYTHONPATH:+:${PYTHONPATH}}"

# ── Config ───────────────────────────────────────────────────────────────────
CONFIG_PATH="${REPO_ROOT}/configs/env_rollout_demo.yaml"
if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "[env_rollout_demo.sh] ERROR: config not found: $CONFIG_PATH" >&2
    exit 1
fi

# ── Logging ──────────────────────────────────────────────────────────────────
LOG_DIR="${REPO_ROOT}/outputs/env_rollout_demo/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/$(date +%Y-%m-%d_%H-%M-%S).log"

echo "============================================================"
echo " ENV ROLLOUT DEMO"
echo "============================================================"
echo " Config : $CONFIG_PATH"
echo " GPU    : $CUDA_VISIBLE_DEVICES"
echo " CARLA  : $CARLA_ROOT"
echo " Log    : $LOG_FILE"
echo "============================================================"

# ── Launch ────────────────────────────────────────────────────────────────────
python -m b2d_rlinfra.framework.env_rollout_demo \
    --config "$CONFIG_PATH" \
    2>&1 | tee "$LOG_FILE"
