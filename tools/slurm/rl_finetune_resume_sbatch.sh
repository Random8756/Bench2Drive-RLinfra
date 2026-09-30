#!/bin/bash
# Submit a distributed rl_finetune job initialized from an RL checkpoint.
#
# Usage:
#   bash tools/slurm/rl_finetune_resume_sbatch.sh configs/<distributed_config>.yaml logs/.../checkpoints/rl_finetune_update_000010.pt
#
# Vulkan runtime:
#   CARLA requires libvulkan.so.1 on every compute node. If it is not installed
#   system-wide, set VULKAN_LIB_DIR to a compatible loader directory. The
#   default fallback directory is third_party/vulkan-tools. VK_ICD_FILENAMES
#   defaults to the standard NVIDIA ICD path and can be overridden.
#
# Edit the USER SETTINGS sections below before invoking this script.

set -euo pipefail

# -- Input --------------------------------------------------------------------
if [[ $# -lt 2 ]]; then
    echo "usage: $0 <config.yaml> <checkpoint.pt>" >&2
    exit 2
fi

CONFIG="$1"
CHECKPOINT="$2"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# -- Config and checkpoint paths ----------------------------------------------
CONFIG_PATH="$CONFIG"
if [[ "$CONFIG_PATH" != /* ]]; then
    CONFIG_PATH="${REPO_ROOT}/${CONFIG_PATH}"
fi
if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "[rl_finetune_resume_sbatch] config not found: $CONFIG_PATH" >&2
    exit 1
fi

CHECKPOINT_PATH="$CHECKPOINT"
if [[ "$CHECKPOINT_PATH" != /* ]]; then
    CHECKPOINT_PATH="${REPO_ROOT}/${CHECKPOINT_PATH}"
fi
if [[ ! -f "$CHECKPOINT_PATH" ]]; then
    echo "[rl_finetune_resume_sbatch] checkpoint not found: $CHECKPOINT_PATH" >&2
    exit 1
fi

# -- Required user settings ---------------------------------------------------
CONDA_ENV_PREFIX="/path/to/conda/env"
CARLA_ROOT_VALUE="/path/to/CARLA"

# -- Optional user settings: runtime -----------------------------------------
# Change the Vulkan paths only when the ICD or loader lives elsewhere.
VULKAN_LIB_DIR_VALUE="${REPO_ROOT}/third_party/vulkan-tools"
VK_ICD_VALUE="/usr/share/vulkan/icd.d/nvidia_icd.json"
EXTRA_PYTHONPATH_VALUE=""
BASE_LD_LIBRARY_PATH_VALUE=""

# -- Optional user settings: Slurm resources ---------------------------------
PARTITION="gpu"
NODES="2"
GRES="gpu:8"
CPUS="100"
MEM="0"
TIME_LIMIT="48:00:00"

# -- Validation and serialization --------------------------------------------
if [[ -z "$CONDA_ENV_PREFIX" || "$CONDA_ENV_PREFIX" != /* ]]; then
    echo "[rl_finetune_resume_sbatch] set CONDA_ENV_PREFIX to an absolute prefix in USER SETTINGS" >&2
    exit 2
fi
if [[ ! -x "$CONDA_ENV_PREFIX/bin/python" ]]; then
    echo "[rl_finetune_resume_sbatch] Python not found: $CONDA_ENV_PREFIX/bin/python" >&2
    exit 2
fi

printf -v CONDA_ENV_Q '%q' "$CONDA_ENV_PREFIX"
printf -v CARLA_ROOT_Q '%q' "$CARLA_ROOT_VALUE"
printf -v VULKAN_LIB_DIR_Q '%q' "$VULKAN_LIB_DIR_VALUE"
printf -v VK_ICD_Q '%q' "$VK_ICD_VALUE"
printf -v EXTRA_PYTHONPATH_Q '%q' "$EXTRA_PYTHONPATH_VALUE"
printf -v BASE_LD_LIBRARY_PATH_Q '%q' "$BASE_LD_LIBRARY_PATH_VALUE"

# -- Submission ---------------------------------------------------------------
mkdir -p "${REPO_ROOT}/output/slurm"

sbatch <<SBATCH
#!/bin/bash
#SBATCH --job-name=b2d-rlft-resume
#SBATCH --partition=${PARTITION}
#SBATCH --nodes=${NODES}
#SBATCH --ntasks-per-node=1
#SBATCH --gres=${GRES}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --signal=TERM@180
#SBATCH --output=${REPO_ROOT}/output/slurm/%x-%j-launch.out

set -euo pipefail

# -- Runtime environment ------------------------------------------------------
REPO_ROOT="${REPO_ROOT}"
CONFIG_PATH="${CONFIG_PATH}"
CHECKPOINT_PATH="${CHECKPOINT_PATH}"
export B2D_RLFT_CONDA_PREFIX=${CONDA_ENV_Q}
export B2D_RLFT_CARLA_ROOT=${CARLA_ROOT_Q}
export VULKAN_LIB_DIR=${VULKAN_LIB_DIR_Q}
export VK_ICD_FILENAMES=${VK_ICD_Q}
export B2D_RLFT_EXTRA_PYTHONPATH=${EXTRA_PYTHONPATH_Q}
export B2D_RLFT_LD_LIBRARY_PATH=${BASE_LD_LIBRARY_PATH_Q}

# -- Run directory ------------------------------------------------------------
RUN_STAMP="\$(date +%Y-%m-%d_%H-%M-%S)"
RUN_ID="\${RUN_STAMP}_\${SLURM_JOB_ID}"
CONFIG_STEM="\$(basename "\${CONFIG_PATH%.yaml}")"
RUN_DIR="\${REPO_ROOT}/logs/\${CONFIG_STEM}/rl_finetune_resume_\${RUN_ID}"
mkdir -p "\${RUN_DIR}/logs/nodes"

# -- Launch -------------------------------------------------------------------
srun --kill-on-bad-exit=0 \\
     --output="\${RUN_DIR}/logs/nodes/node_%n.out" \\
     bash "\${REPO_ROOT}/tools/slurm/node_entry.sh" "\${CONFIG_PATH}" "\${RUN_DIR}" "\${CHECKPOINT_PATH}"
SBATCH
