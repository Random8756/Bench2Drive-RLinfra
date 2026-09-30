#!/usr/bin/env bash

# Usage: bash tools/runtime/kill_all.sh [--dry-run]

set -u

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
status=0

for octet in {135..166}; do
    "$SCRIPT_DIR/kill_by_host.sh" "$@" "127.0.1.${octet}" || status=1
done

exit "$status"
