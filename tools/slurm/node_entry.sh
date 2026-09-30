#!/bin/bash
# Per-node entrypoint for distributed rl_finetune.
#
# The submission scripts pass the resolved runtime paths, Slurm node rank,
# config path, run directory, and optional checkpoint to this entrypoint.
# This entrypoint prepends VULKAN_LIB_DIR when provided and
# otherwise checks third_party/vulkan-tools for libvulkan.so.1 and vulkaninfo.

set -euo pipefail

# -- Arguments ----------------------------------------------------------------
CONFIG_PATH="${1:?usage: $0 <config.yaml> <run_dir> [checkpoint.pt]}"
RUN_DIR="${2:?usage: $0 <config.yaml> <run_dir> [checkpoint.pt]}"
INIT_FROM_CHECKPOINT="${3:-}"

# -- Repository ---------------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# -- Job environment: Python --------------------------------------------------
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1

CONDA_ENV_PREFIX="${B2D_RLFT_CONDA_PREFIX:-}"
if [[ -z "$CONDA_ENV_PREFIX" ]]; then
    echo "[node_entry] Conda environment prefix was not propagated by the submission script." >&2
    exit 2
fi
if [[ ! -x "$CONDA_ENV_PREFIX/bin/python" ]]; then
    echo "[node_entry] Python not found under the configured Conda prefix: $CONDA_ENV_PREFIX/bin/python" >&2
    exit 2
fi
export CONDA_PREFIX="$CONDA_ENV_PREFIX"
export PATH="$CONDA_PREFIX/bin:${PATH}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${B2D_RLFT_LD_LIBRARY_PATH:+:${B2D_RLFT_LD_LIBRARY_PATH}}"

# -- Job environment: GPU and Vulkan -----------------------------------------
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "[node_entry] CUDA_VISIBLE_DEVICES is empty; verify the Slurm GRES allocation." >&2
    exit 2
fi
export CUDA_VISIBLE_DEVICES
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/nvidia_icd.json}"

VULKAN_LIB_DIR="${VULKAN_LIB_DIR:-${REPO_ROOT}/third_party/vulkan-tools}"
if [[ -f "$VULKAN_LIB_DIR/libvulkan.so.1" ]]; then
    export LD_LIBRARY_PATH="$VULKAN_LIB_DIR:${LD_LIBRARY_PATH}"
fi
if [[ -x "$VULKAN_LIB_DIR/vulkaninfo" ]]; then
    export PATH="$VULKAN_LIB_DIR:${PATH}"
fi

# -- Job environment: project paths ------------------------------------------
export CARLA_ROOT="${B2D_RLFT_CARLA_ROOT:-${CARLA_ROOT:-}}"
export TRAINING_RUNTIME="$REPO_ROOT/vendor/carla/training-runtime"
export SCENARIO_RUNNER_ROOT="$TRAINING_RUNTIME/scenario_runner"
export PYTHONPATH="$REPO_ROOT:$TRAINING_RUNTIME:$SCENARIO_RUNNER_ROOT:$CARLA_ROOT/PythonAPI/carla${B2D_RLFT_EXTRA_PYTHONPATH:+:${B2D_RLFT_EXTRA_PYTHONPATH}}"

# -- Node role ----------------------------------------------------------------
ROLE="collector"
if [[ "${SLURM_NODEID:-0}" == "0" ]]; then
    ROLE="learner"
fi

# -- Launch -------------------------------------------------------------------
CMD=(
    python -m b2d_rlinfra.finetuning.node_agent
    --config "$CONFIG_PATH"
    --run-dir "$RUN_DIR"
    --node-rank "${SLURM_NODEID:-0}"
    --role "$ROLE"
)
if [[ -n "$INIT_FROM_CHECKPOINT" ]]; then
    CMD+=(--init-from-checkpoint "$INIT_FROM_CHECKPOINT")
fi

exec "${CMD[@]}"
