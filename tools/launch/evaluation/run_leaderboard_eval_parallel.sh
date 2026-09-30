#!/bin/bash
# Parallel leaderboard evaluation script.
#
# Usage:
#   bash run_leaderboard_eval_parallel.sh

set -euo pipefail

# ── Paths ────────────────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

export PROJECT_ROOT="$REPO_ROOT"
export CARLA_ROOT="/path/to/carla"

# Use the evaluation runtime expected by leaderboard_evaluator.py.
export EVALUATION_RUNTIME="$REPO_ROOT/vendor/carla/evaluation-runtime"
export SCENARIO_RUNNER_ROOT="$EVALUATION_RUNTIME/scenario_runner"
export LEADERBOARD_ROOT="$EVALUATION_RUNTIME/leaderboard"
export PYTHONPATH="$REPO_ROOT:$EVALUATION_RUNTIME:$LEADERBOARD_ROOT:$SCENARIO_RUNNER_ROOT:$CARLA_ROOT/PythonAPI/carla${PYTHONPATH:+:${PYTHONPATH}}"

# ── Config ───────────────────────────────────────────────────────────────────
# Edit these values directly for a new evaluation run.
CONFIG_PATH="/path/to/eval_config.yaml"
CHECKPOINT_PATH="/path/to/checkpoint_dir"
ROUTES_FILE="/path/to/routes/eval_routes.xml"

AGENT_PY="$REPO_ROOT/b2d_rlinfra/evaluation/leaderboard/agent.py"
STOCHASTIC=false
REPETITIONS=1
NUM_WORKERS=16
TRACK="SENSORS"

# Watchdog timeout per route: agent setup loads a large pre-rasterized HDF5
# map, and parallel startup can be I/O bound.
TIMEOUT=1200
DEBUG=0

if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "[run_leaderboard_eval_parallel.sh] ERROR: config not found: $CONFIG_PATH" >&2
    exit 1
fi

if [[ ! -e "$CHECKPOINT_PATH" ]]; then
    echo "[run_leaderboard_eval_parallel.sh] ERROR: checkpoint not found: $CHECKPOINT_PATH" >&2
    exit 1
fi

if [[ ! -f "$ROUTES_FILE" ]]; then
    echo "[run_leaderboard_eval_parallel.sh] ERROR: routes file not found: $ROUTES_FILE" >&2
    exit 1
fi

CONFIG_STEM="$(basename "${CONFIG_PATH%.yaml}")"
OUTPUT_ROOT="$REPO_ROOT/leaderboard_eval_logs/${CONFIG_STEM}_$(date +%Y-%m-%d_%H-%M-%S)_parallel"
RECORD_DIR="${OUTPUT_ROOT}/records"

export CHECKPOINT_PATH
export STOCHASTIC
export OUTPUT_ROOT

if [[ "${STOCHASTIC,,}" == "true" || "${STOCHASTIC}" == "1" || "${STOCHASTIC,,}" == "yes" || "${STOCHASTIC,,}" == "on" ]]; then
    STOCHASTIC_LABEL="True"
else
    STOCHASTIC_LABEL="False"
fi

# ── Logging ──────────────────────────────────────────────────────────────────
mkdir -p "$OUTPUT_ROOT"
LOG_FILE="$OUTPUT_ROOT/run.log"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " Config     : $CONFIG_PATH"
echo " Checkpoint : $CHECKPOINT_PATH"
echo " Routes     : $ROUTES_FILE"
echo " Workers    : $NUM_WORKERS"
echo " Stochastic : $STOCHASTIC_LABEL"
echo " Output     : $OUTPUT_ROOT"
echo " Log        : $LOG_FILE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── Launch ────────────────────────────────────────────────────────────────────
python3 -m b2d_rlinfra.evaluation.leaderboard.run_leaderboard_eval_parallel \
    --config "$CONFIG_PATH" \
    --checkpoint "$CHECKPOINT_PATH" \
    --routes "$ROUTES_FILE" \
    --output-root "$OUTPUT_ROOT" \
    --leaderboard-root "$LEADERBOARD_ROOT" \
    --agent "$AGENT_PY" \
    --repetitions "$REPETITIONS" \
    --num-workers "$NUM_WORKERS" \
    --track "$TRACK" \
    --timeout "$TIMEOUT" \
    --debug "$DEBUG" \
    --stochastic "$STOCHASTIC" \
    --record-dir "$RECORD_DIR" \
    2>&1 | tee "$LOG_FILE"
