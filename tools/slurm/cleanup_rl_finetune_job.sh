#!/usr/bin/env bash
# Fan out run-scoped rl_finetune cleanup across the original Slurm nodes.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  cleanup_rl_finetune_job.sh --run-dir RUN_DIR [options]
  cleanup_rl_finetune_job.sh --job-id JOB_ID --run-dir RUN_DIR [options]
  cleanup_rl_finetune_job.sh --nodelist NODELIST --run-dir RUN_DIR [options]

Options:
  --job-id JOB_ID       Fallback node discovery, or allocation reuse target.
  --run-dir RUN_DIR     Distributed run directory; also used for node discovery.
  --force               Actually kill tagged processes and unlink run SHM.
  --nodelist NODELIST   Override node discovery from sacct/scontrol.
  --partition NAME      Partition for a new cleanup allocation (default: gpu).
  --gres SPEC           Optional GRES request for a new cleanup allocation.
  --time LIMIT          Cleanup allocation time limit (default: 00:10:00).
  --reuse-allocation    Run as a new step inside the still-active JOB_ID.
  --print-only          Print the srun command without executing it.
  -h, --help            Show this help.

Without --force the node cleanup runs in dry-run mode.
EOF
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
NODE_CLEANUP="$SCRIPT_DIR/cleanup_rl_finetune_node.py"
PYTHON_BIN="${B2D_RLFT_CLEANUP_PYTHON:-python3}"

JOB_ID=""
RUN_DIR=""
NODELIST=""
PARTITION="${B2D_RLFT_SLURM_PARTITION:-gpu}"
GRES="${B2D_RLFT_CLEANUP_GRES:-}"
TIME_LIMIT="${B2D_RLFT_CLEANUP_TIME:-00:10:00}"
FORCE=0
REUSE_ALLOCATION=0
PRINT_ONLY=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --job-id) JOB_ID="${2:?--job-id requires a value}"; shift 2 ;;
        --run-dir) RUN_DIR="${2:?--run-dir requires a value}"; shift 2 ;;
        --nodelist) NODELIST="${2:?--nodelist requires a value}"; shift 2 ;;
        --partition) PARTITION="${2:?--partition requires a value}"; shift 2 ;;
        --gres) GRES="${2:?--gres requires a value}"; shift 2 ;;
        --time) TIME_LIMIT="${2:?--time requires a value}"; shift 2 ;;
        --force) FORCE=1; shift ;;
        --reuse-allocation) REUSE_ALLOCATION=1; shift ;;
        --print-only) PRINT_ONLY=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -z "$RUN_DIR" ]]; then
    echo "--run-dir is required" >&2
    usage >&2
    exit 2
fi
if [[ "$RUN_DIR" != /* ]]; then
    RUN_DIR="$REPO_ROOT/$RUN_DIR"
fi
if [[ ! -d "$RUN_DIR" ]]; then
    echo "run directory does not exist: $RUN_DIR" >&2
    exit 2
fi
RUN_DIR="$(cd -- "$RUN_DIR" && pwd)"

if [[ -z "$NODELIST" ]]; then
    mapfile -t RECORDED_NODES < <("$PYTHON_BIN" "$NODE_CLEANUP" --run-dir "$RUN_DIR" --print-hosts 2>/dev/null || true)
    if [[ ${#RECORDED_NODES[@]} -gt 0 ]]; then
        NODELIST="$(IFS=,; echo "${RECORDED_NODES[*]}")"
    fi
fi

if [[ -z "$NODELIST" && -n "$JOB_ID" ]]; then
    if command -v sacct >/dev/null 2>&1; then
        NODELIST="$(
            sacct -X -j "$JOB_ID" -n -P -o JobIDRaw,NodeList 2>/dev/null |
                awk -F'|' -v job="$JOB_ID" '$1 == job && $2 != "" && $2 != "None assigned" { print $2; exit }'
        )"
    fi
    if [[ -z "$NODELIST" ]] && command -v scontrol >/dev/null 2>&1; then
        NODELIST="$(
            scontrol show job -o "$JOB_ID" 2>/dev/null |
                sed -n 's/.*NodeList=\([^ ]*\).*/\1/p' |
                head -n1
        )"
    fi
fi

if [[ -z "$NODELIST" || "$NODELIST" == "(null)" ]]; then
    echo "could not resolve nodes; pass --nodelist explicitly" >&2
    exit 1
fi
if ! command -v scontrol >/dev/null 2>&1; then
    echo "scontrol is required to expand the node list" >&2
    exit 1
fi
mapfile -t NODES < <(scontrol show hostnames "$NODELIST")
if [[ ${#NODES[@]} -eq 0 ]]; then
    echo "node list expanded to zero hosts: $NODELIST" >&2
    exit 1
fi

CLEANUP_ARGS=(--run-dir "$RUN_DIR")
MODE="DRY-RUN"
if [[ $FORCE -eq 1 ]]; then
    CLEANUP_ARGS+=(--force)
    MODE="FORCE"
fi

SRUN=(srun)
if [[ $REUSE_ALLOCATION -eq 1 ]]; then
    if [[ -z "$JOB_ID" ]]; then
        echo "--reuse-allocation requires --job-id" >&2
        exit 2
    fi
    SRUN+=(--jobid="$JOB_ID" --overlap)
else
    SRUN+=(--partition="$PARTITION" --nodelist="$NODELIST")
    if [[ -n "$GRES" ]]; then
        SRUN+=(--gres="$GRES")
    fi
fi
SRUN+=(
    --nodes="${#NODES[@]}"
    --ntasks="${#NODES[@]}"
    --ntasks-per-node=1
    --cpus-per-task=1
    --time="$TIME_LIMIT"
    --label
    "$PYTHON_BIN" "$NODE_CLEANUP" "${CLEANUP_ARGS[@]}"
)

echo "Mode: $MODE"
echo "Nodes: $NODELIST (${#NODES[@]})"
echo "Run dir: $RUN_DIR"
printf 'Command:'
printf ' %q' "${SRUN[@]}"
printf '\n'

if [[ $PRINT_ONLY -eq 1 ]]; then
    exit 0
fi
exec "${SRUN[@]}"
