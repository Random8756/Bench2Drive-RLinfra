#!/bin/bash
# Off-policy (SAC / TD3) resume training.
#
# Loads checkpoint weights selectively while everything else (buffer size,
# entropy, BC, routes, rewards, ...) is taken from a fresh YAML config.
#
# Edit the parameters at the top of this script:
#   CONFIG_PATH     - new YAML config; algorithm.name must be sac or td3.
#   CHECKPOINT_PATH - checkpoint dir (must contain policy.pth + metadata.json).
#   LOAD_ACTOR      - load actor weights (and the shared features_extractor).
#   LOAD_CRITIC     - load critic / critic_target weights.
#   RESET_TIMESTEPS - reset num_timesteps so warmup runs again (default true).
#
# warmup_source and adaptive route sampling parameters
# (sample_mode=adaptive + window_size / success_threshold / ...) live in the
# YAML config.
#
# Usage:
#   bash tools/launch/baseline/resume_off_policy.sh sac
#   bash tools/launch/baseline/resume_off_policy.sh td3 /path/to/checkpoint

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
ARG="${1:-sac}"
CHECKPOINT_PATH="${2:-path/to/checkpoint_dir}"

case "$ARG" in
    sac)  CONFIG_PATH="${REPO_ROOT}/configs/sac_bev_example.yaml" ;;
    td3)  CONFIG_PATH="${REPO_ROOT}/configs/td3_bev_example.yaml" ;;
    *)    CONFIG_PATH="$ARG" ;;   # treat as a literal path
esac

# Load actor / critic from the checkpoint independently. Branches that are
# not loaded are re-initialised from the new config. The shared
# features_extractor is loaded as long as either actor or critic is.
LOAD_ACTOR=true
LOAD_CRITIC=true

# Default true: reset num_timesteps so warmup runs again.
RESET_TIMESTEPS=true
# ============================================================

if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "[resume_off_policy.sh] ERROR: config not found: $CONFIG_PATH" >&2
    exit 1
fi

CONFIG_STEM="$(basename "${CONFIG_PATH%.yaml}")"
LOG_DIR="output/${CONFIG_STEM}"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/$(date +%Y-%m-%d_%H-%M-%S).log"

echo "Config:          $CONFIG_PATH"
echo "Checkpoint:      $CHECKPOINT_PATH"
echo "Load actor:      $LOAD_ACTOR"
echo "Load critic:     $LOAD_CRITIC"
echo "Reset timesteps: $RESET_TIMESTEPS"
echo "Log:             $LOG_FILE"

python -m b2d_rlinfra.learning.training.resume_off_policy \
    --config "$CONFIG_PATH" \
    --checkpoint "$CHECKPOINT_PATH" \
    --load-actor "$LOAD_ACTOR" \
    --load-critic "$LOAD_CRITIC" \
    --reset-timesteps "$RESET_TIMESTEPS" \
    2>&1 | tee "$LOG_FILE"
