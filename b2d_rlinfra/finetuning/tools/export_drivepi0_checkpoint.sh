#!/bin/bash
# Export DrivePi0 RL finetune weights into a DriveMoE official eval checkpoint.
#
# Usage:
#   bash b2d_rlinfra/finetuning/tools/export_drivepi0_checkpoint.sh
#
# Paths can be overridden via env:
#   RL_CHECKPOINT=/path/to/rl_finetune_update_000100.pt \
#   BASE_CHECKPOINT=/path/to/DrivePi0_Base_bf16.pt \
#   OUTPUT_CHECKPOINT=/path/to/drivepi0_rl_export.pt \
#   bash b2d_rlinfra/finetuning/tools/export_drivepi0_checkpoint.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:${PYTHONPATH}}"

: "${RL_CHECKPOINT:=$REPO_ROOT/path/to/rl_finetune_update_000100.pt}"
: "${BASE_CHECKPOINT:=$REPO_ROOT/../DriveMoE/ckpts/DrivePi0_Base_bf16.pt}"
: "${OUTPUT_CHECKPOINT:=$REPO_ROOT/output/drivepi0_rl_export/$(date +%Y-%m-%d_%H-%M-%S).pt}"

if [[ ! -f "$RL_CHECKPOINT" ]]; then
    echo "[export_drivepi0_checkpoint.sh] ERROR: RL checkpoint not found: $RL_CHECKPOINT" >&2
    exit 1
fi

if [[ ! -f "$BASE_CHECKPOINT" ]]; then
    echo "[export_drivepi0_checkpoint.sh] ERROR: base checkpoint not found: $BASE_CHECKPOINT" >&2
    exit 1
fi

mkdir -p "$(dirname "$OUTPUT_CHECKPOINT")"

echo "RL checkpoint   : $RL_CHECKPOINT"
echo "Base checkpoint : $BASE_CHECKPOINT"
echo "Output          : $OUTPUT_CHECKPOINT"

python -m b2d_rlinfra.finetuning.tools.export_drivepi0_checkpoint \
    --rl-checkpoint "$RL_CHECKPOINT" \
    --base-checkpoint "$BASE_CHECKPOINT" \
    --output "$OUTPUT_CHECKPOINT"
