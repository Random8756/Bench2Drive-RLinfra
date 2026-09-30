#!/bin/bash

# Kill CARLA processes associated with a specific fake-bind host IP.
# Usage: ./kill_by_host.sh [--dry-run] <host_ip>
# Example: ./kill_by_host.sh --dry-run 127.0.0.108

DRY_RUN=0
if [ "$1" = "--dry-run" ] || [ "$1" = "-n" ]; then
    DRY_RUN=1
    shift
fi

if [ -z "$1" ]; then
    echo "Usage: $0 [--dry-run] <host_ip>"
    echo "Example: $0 --dry-run 127.0.0.108"
    echo ""
    echo "Current CARLA processes:"
    pgrep -a -f "CarlaUE4|carla-rpc-port"
    exit 1
fi

HOST_IP="$1"
CARLA_PATTERN="CarlaUE4|carla-rpc-port"

echo "Searching for CARLA processes bound to $HOST_IP..."
if [ "$DRY_RUN" -eq 1 ]; then
    echo "Dry run enabled; no processes will be killed."
fi

host_matches() {
    local text="$1"
    local host_re="${HOST_IP//./\\.}"
    printf '%s\n' "$text" | grep -Eq "(^|[^0-9.])${host_re}([^0-9.]|$)"
}

fake_bind_matches() {
    local text="$1"
    printf '%s\n' "$text" | grep -Fxq "FAKE_BIND_IP=$HOST_IP"
}

read_cmdline() {
    local pid="$1"
    tr '\0' ' ' 2>/dev/null < "/proc/$pid/cmdline"
}

collect_descendants() {
    local parent="$1"
    local child

    for child in $(pgrep -P "$parent" 2>/dev/null); do
        echo "$child"
        collect_descendants "$child"
    done
}

PIDS_BY_NETWORK=""
while IFS= read -r line; do
    if host_matches "$line"; then
        PIDS_BY_NETWORK="$PIDS_BY_NETWORK $(printf '%s\n' "$line" | grep -oP 'pid=\K[0-9]+' | sort -u)"
    fi
done < <(ss -tlnp 2>/dev/null)

PIDS_BY_ENV=""
for pid in $(pgrep -f "$CARLA_PATTERN" 2>/dev/null); do
    if [ -f "/proc/$pid/environ" ]; then
        if fake_bind_matches "$(tr '\0' '\n' 2>/dev/null < "/proc/$pid/environ")"; then
            PIDS_BY_ENV="$PIDS_BY_ENV $pid"
        fi
    fi
done

PIDS_BY_CMD=""
for pid in $(pgrep -f "FAKE_BIND_IP=" 2>/dev/null); do
    cmdline="$(read_cmdline "$pid")"
    if host_matches "$cmdline" && printf '%s\n' "$cmdline" | grep -Eq "$CARLA_PATTERN"; then
        PIDS_BY_CMD="$PIDS_BY_CMD $pid"
    fi
done

PIDS_BY_PORT=""
for pid in $PIDS_BY_CMD; do
    cmdline="$(read_cmdline "$pid")"
    port="$(echo "$cmdline" | grep -oE -- '-carla-rpc-port=[0-9]+' | head -n1 | cut -d= -f2)"
    if [ -n "$port" ]; then
        for port_pid in $(pgrep -f -- "-carla-rpc-port=$port" 2>/dev/null); do
            PIDS_BY_PORT="$PIDS_BY_PORT $port_pid"
        done
    fi
done

PIDS_BY_CHILDREN=""
for pid in $PIDS_BY_NETWORK $PIDS_BY_ENV $PIDS_BY_CMD $PIDS_BY_PORT; do
    if [ -d "/proc/$pid" ]; then
        PIDS_BY_CHILDREN="$PIDS_BY_CHILDREN $(collect_descendants "$pid")"
    fi
done

ALL_PIDS=$(echo "$PIDS_BY_NETWORK $PIDS_BY_ENV $PIDS_BY_CMD $PIDS_BY_PORT $PIDS_BY_CHILDREN" | tr ' ' '\n' | sort -u | grep -v '^$')

if [ -z "$ALL_PIDS" ]; then
    echo "No CARLA processes bound to $HOST_IP were found"
    echo ""
    echo "All current CARLA-related processes:"
    pgrep -a -f "$CARLA_PATTERN" || echo "  none"
    exit 0
fi

echo "Found processes:"
for pid in $ALL_PIDS; do
    if [ -d "/proc/$pid" ]; then
        echo "  PID: $pid - $(cat /proc/$pid/comm 2>/dev/null)"
    fi
done

echo ""
if [ "$DRY_RUN" -eq 1 ]; then
    echo "Dry run complete; no processes were killed."
else
    echo "Terminating processes..."

    for pid in $ALL_PIDS; do
        if [ -d "/proc/$pid" ]; then
            echo "  Killing PID: $pid"
            kill -9 $pid 2>/dev/null
        fi
    done
fi

echo "Done!"
echo ""
echo "Remaining CARLA processes:"
pgrep -a -f "$CARLA_PATTERN" || echo "  none"
