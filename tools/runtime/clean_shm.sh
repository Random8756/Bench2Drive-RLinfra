#!/bin/bash
# Clean up orphaned /dev/shm files left by off-policy RL training processes.
#
# Files targeted:
#   rl_buf_*    — SharedPrioritizedReplayBuffer mmap files
#   rl_weights_* — policy weight flat-buffer files
#   carla_rgb_* — RGB observation SHM IPC files
#   rl_finetune_rgb_* — RL finetune RGB observation SHM IPC files
#
# A file is considered orphaned if no running process has it open.
# fuser(1) is used for this check; lsof(1) is used as a fallback.
#
# Usage:
#   ./clean_shm.sh            # interactive (asks for confirmation)
#   ./clean_shm.sh -n         # dry-run, show what would be deleted
#   ./clean_shm.sh -f         # force, skip confirmation prompt
#   ./clean_shm.sh -n -v      # dry-run with verbose per-file listing
#   ./clean_shm.sh --all      # also delete files that ARE in use (dangerous!)

set -euo pipefail

SHM_DIR="/dev/shm"
DRY_RUN=0
FORCE=0
VERBOSE=0
DELETE_ACTIVE=0

# ── argument parsing ─────────────────────────────────────────────────────────
for arg in "$@"; do
    case "$arg" in
        -n|--dry-run)   DRY_RUN=1 ;;
        -f|--force)     FORCE=1 ;;
        -v|--verbose)   VERBOSE=1 ;;
        --all)          DELETE_ACTIVE=1 ;;
        -h|--help)
            sed -n '2,17p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *)
            echo "Unknown option: $arg  (use -h for help)" >&2
            exit 1
            ;;
    esac
done

# ── check if a file is currently open by any process ─────────────────────────
file_in_use() {
    local f="$1"
    if command -v fuser &>/dev/null; then
        fuser "$f" &>/dev/null
    elif command -v lsof &>/dev/null; then
        lsof "$f" &>/dev/null
    else
        # Cannot determine; treat as in-use to be safe
        return 0
    fi
}

# ── collect target files ──────────────────────────────────────────────────────
mapfile -t ALL_FILES < <(
    find "$SHM_DIR" -maxdepth 1 \( -name 'rl_buf_*' -o -name 'rl_weights_*' -o -name 'carla_rgb_*' -o -name 'rl_finetune_rgb_*' \) \
        -type f 2>/dev/null | sort
)

if [[ ${#ALL_FILES[@]} -eq 0 ]]; then
    echo "No rl_buf_*, rl_weights_*, carla_rgb_*, or rl_finetune_rgb_* files found in $SHM_DIR. Nothing to do."
    exit 0
fi

# ── categorise into orphaned / active ────────────────────────────────────────
ORPHANED=()
ACTIVE=()

echo "Scanning ${#ALL_FILES[@]} file(s) in $SHM_DIR ..."
for f in "${ALL_FILES[@]}"; do
    if file_in_use "$f"; then
        ACTIVE+=("$f")
    else
        ORPHANED+=("$f")
    fi
done

# ── summary before action ─────────────────────────────────────────────────────
print_file_list() {
    local label="$1"; shift
    local -a files=("$@")
    if [[ ${#files[@]} -eq 0 ]]; then return; fi
    local total_bytes=0
    echo ""
    echo "  $label (${#files[@]} file(s)):"
    for f in "${files[@]}"; do
        local sz
        sz=$(stat -c '%s' "$f" 2>/dev/null || echo 0)
        total_bytes=$(( total_bytes + sz ))
        if [[ $VERBOSE -eq 1 ]]; then
            local hr
            hr=$(numfmt --to=iec-i --suffix=B "$sz" 2>/dev/null || echo "${sz}B")
            printf "    %-10s  %s\n" "$hr" "$(basename "$f")"
        fi
    done
    local hr_total
    hr_total=$(numfmt --to=iec-i --suffix=B "$total_bytes" 2>/dev/null || echo "${total_bytes}B")
    echo "    Total: $hr_total"
}

print_file_list "Orphaned (will be deleted)" "${ORPHANED[@]+"${ORPHANED[@]}"}"

if [[ ${#ACTIVE[@]} -gt 0 ]]; then
    print_file_list "In use  (will be SKIPPED)" "${ACTIVE[@]}"
    if [[ $DELETE_ACTIVE -eq 1 ]]; then
        echo ""
        echo "  WARNING: --all specified; in-use files will also be deleted!"
    fi
fi

echo ""

# ── determine final delete list ───────────────────────────────────────────────
TO_DELETE=("${ORPHANED[@]+"${ORPHANED[@]}"}")
if [[ $DELETE_ACTIVE -eq 1 ]]; then
    TO_DELETE+=("${ACTIVE[@]+"${ACTIVE[@]}"}")
fi

if [[ ${#TO_DELETE[@]} -eq 0 ]]; then
    echo "Nothing to delete."
    exit 0
fi

# ── dry-run: stop here ────────────────────────────────────────────────────────
if [[ $DRY_RUN -eq 1 ]]; then
    echo "[dry-run] Would delete ${#TO_DELETE[@]} file(s). No changes made."
    exit 0
fi

# ── confirmation prompt ───────────────────────────────────────────────────────
if [[ $FORCE -eq 0 ]]; then
    read -rp "Delete ${#TO_DELETE[@]} file(s)? [y/N] " ans
    case "$ans" in
        [Yy]*) ;;
        *) echo "Aborted."; exit 0 ;;
    esac
fi

# ── delete ────────────────────────────────────────────────────────────────────
DELETED=0
FAILED=0
FREED_BYTES=0

for f in "${TO_DELETE[@]}"; do
    sz=$(stat -c '%s' "$f" 2>/dev/null || echo 0)
    if rm -f "$f"; then
        DELETED=$(( DELETED + 1 ))
        FREED_BYTES=$(( FREED_BYTES + sz ))
        [[ $VERBOSE -eq 1 ]] && echo "  deleted: $(basename "$f")"
    else
        echo "  FAILED to delete: $f" >&2
        FAILED=$(( FAILED + 1 ))
    fi
done

FREED_HR=$(numfmt --to=iec-i --suffix=B "$FREED_BYTES" 2>/dev/null || echo "${FREED_BYTES}B")
echo "Done. Deleted $DELETED file(s), freed $FREED_HR of /dev/shm (memory)."
[[ $FAILED -gt 0 ]] && echo "WARNING: $FAILED file(s) could not be deleted." >&2

exit $(( FAILED > 0 ? 1 : 0 ))
