#!/bin/bash
# RL finetune training script.
#
# Usage:
#   bash train.sh                         # configs/rl_finetune_rgb_example.yaml
#   bash train.sh example                 # configs/rl_finetune_rgb_example.yaml
#   bash train.sh minddrive               # configs/minddrive_rl_finetune.yaml
#   bash train.sh /path/to/custom.yaml    # arbitrary config path
#   bash train.sh example --init-from-checkpoint /path/to/checkpoint.pt
#
# Vulkan runtime:
#   CARLA requires libvulkan.so.1 when using Vulkan rendering. If it is not
#   installed system-wide, set VULKAN_LIB_DIR to a compatible loader directory.
#   The default fallback directory is third_party/vulkan-tools.
#
# Edit the USER SETTINGS sections below before launching.

set -euo pipefail

# -- Repository ---------------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# -- Required user settings ---------------------------------------------------
CARLA_ROOT="/path/to/CARLA"

# -- Optional user settings ---------------------------------------------------
# Visible GPUs must match collector_devices and the learner device in the YAML.
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

# Change these only when the NVIDIA ICD or Vulkan loader lives elsewhere.
VK_ICD_FILENAMES="/usr/share/vulkan/icd.d/nvidia_icd.json"
VULKAN_LIB_DIR="${REPO_ROOT}/third_party/vulkan-tools"

# Optional diagnostics and allocator settings; empty means disabled.
PYTORCH_CUDA_ALLOC_CONF=""
PYTHONFAULTHANDLER=""
B2D_CARLA_LOG_DIR=""

# -- Runtime environment ------------------------------------------------------
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES
export CARLA_ROOT
export VK_ICD_FILENAMES

if [[ -n "$PYTORCH_CUDA_ALLOC_CONF" ]]; then
    export PYTORCH_CUDA_ALLOC_CONF
fi
if [[ -n "$PYTHONFAULTHANDLER" ]]; then
    export PYTHONFAULTHANDLER
fi
if [[ -n "$B2D_CARLA_LOG_DIR" ]]; then
    export B2D_CARLA_LOG_DIR
fi
if [[ -f "$VULKAN_LIB_DIR/libvulkan.so.1" ]]; then
    export LD_LIBRARY_PATH="$VULKAN_LIB_DIR:${LD_LIBRARY_PATH:-}"
fi
if [[ -x "$VULKAN_LIB_DIR/vulkaninfo" ]]; then
    export PATH="$VULKAN_LIB_DIR:${PATH}"
fi

export TRAINING_RUNTIME="$REPO_ROOT/vendor/carla/training-runtime"
export SCENARIO_RUNNER_ROOT="$TRAINING_RUNTIME/scenario_runner"
export PYTHONPATH="$REPO_ROOT:$TRAINING_RUNTIME:$SCENARIO_RUNNER_ROOT:$CARLA_ROOT/PythonAPI/carla${PYTHONPATH:+:${PYTHONPATH}}"

# -- Config resolution --------------------------------------------------------
ARG="${1:-example}"   # first positional arg: config alias or full path

case "$ARG" in
    minddrive)     CONFIG_PATH="${REPO_ROOT}/configs/minddrive_rl_finetune.yaml" ;;
    drivepi0)      CONFIG_PATH="${REPO_ROOT}/configs/drivepi0_rl_finetune.yaml" ;;
    example)       CONFIG_PATH="${REPO_ROOT}/configs/rl_finetune_rgb_example.yaml" ;;
    *)             CONFIG_PATH="$ARG" ;;
esac

if [[ $# -gt 0 ]]; then
    shift
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "[rl_finetune/train.sh] ERROR: config not found: $CONFIG_PATH" >&2
    exit 1
fi

CONFIG_STEM="$(basename "${CONFIG_PATH%.yaml}")"

# -- Logging ------------------------------------------------------------------
LOG_DIR="${REPO_ROOT}/output/${CONFIG_STEM}"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/$(date +%Y-%m-%d_%H-%M-%S).log"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " Config      : $CONFIG_PATH"
echo " GPU         : $CUDA_VISIBLE_DEVICES"
echo " CARLA_ROOT  : $CARLA_ROOT"
echo " Log         : $LOG_FILE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# -- Launch -------------------------------------------------------------------
CMD=(python -m b2d_rlinfra.finetuning.train --config "$CONFIG_PATH")
CMD+=("$@")

"${CMD[@]}" 2>&1 | tee "$LOG_FILE"
