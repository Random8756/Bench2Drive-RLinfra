#!/bin/bash
# Export MindDrive RL finetune weights into a MindDrive eval checkpoint.
#
# Usage:
#   bash b2d_rlinfra/finetuning/tools/export_minddrive_checkpoint.sh
#
# Paths can be overridden via env:
#   RL_CHECKPOINT=/path/to/rl_finetune_update_000054.pt \
#   BASE_CHECKPOINT=/path/to/minddrive_rltrain.pth \
#   OUTPUT_CHECKPOINT=/path/to/minddrive_rl_export.pth \
#   bash b2d_rlinfra/finetuning/tools/export_minddrive_checkpoint.sh
#
# rl_finetune_update_000054.pt means 54 PPO updates have completed.

set -euo pipefail

# ── Paths ────────────────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:${PYTHONPATH}}"

# ── Config ───────────────────────────────────────────────────────────────────
# Edit these values directly for a new export, or override them via env.
: "${RL_CHECKPOINT:=$REPO_ROOT/path/to/rl_finetune_update_000054.pt}"
: "${BASE_CHECKPOINT:=$REPO_ROOT/path/to/minddrive_rltrain.pth}"
: "${OUTPUT_CHECKPOINT:=$REPO_ROOT/output/minddrive_rl_export/$(date +%Y-%m-%d_%H-%M-%S).pth}"

if [[ ! -f "$RL_CHECKPOINT" ]]; then
    echo "[export_minddrive_checkpoint.sh] ERROR: RL checkpoint not found: $RL_CHECKPOINT" >&2
    exit 1
fi

if [[ ! -f "$BASE_CHECKPOINT" ]]; then
    echo "[export_minddrive_checkpoint.sh] ERROR: base checkpoint not found: $BASE_CHECKPOINT" >&2
    exit 1
fi

mkdir -p "$(dirname "$OUTPUT_CHECKPOINT")"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " RL checkpoint   : $RL_CHECKPOINT"
echo " Base checkpoint : $BASE_CHECKPOINT"
echo " Output          : $OUTPUT_CHECKPOINT"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── Launch ───────────────────────────────────────────────────────────────────
python -m b2d_rlinfra.finetuning.tools.export_minddrive_checkpoint \
    --rl-checkpoint "$RL_CHECKPOINT" \
    --base-checkpoint "$BASE_CHECKPOINT" \
    --output "$OUTPUT_CHECKPOINT"
