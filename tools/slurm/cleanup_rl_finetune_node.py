#!/usr/bin/env python3
"""Clean one rl_finetune run from the current compute node."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import socket
import sys
import time
from pathlib import Path
from typing import Iterable, List, Set


RUN_ENV_KEY = "B2D_RL_FINETUNE_RUN_DIR"
SHM_NAME_PREFIX = "rl_finetune_rgb_"


def run_shm_prefix(run_dir: Path) -> str:
    run_key = str(run_dir.resolve()).encode("utf-8")
    return "{}{}".format(SHM_NAME_PREFIX, hashlib.sha256(run_key).hexdigest()[:12])


def _read_proc_bytes(proc_dir: Path, pid: int, name: str) -> bytes:
    try:
        return (proc_dir / str(pid) / name).read_bytes()
    except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
        return b""


def _pid_has_run_marker(proc_dir: Path, pid: int, run_dir: Path) -> bool:
    marker = "{}={}".format(RUN_ENV_KEY, run_dir).encode("utf-8")
    return marker in _read_proc_bytes(proc_dir, pid, "environ").split(b"\0")


def _pid_has_run_argument(proc_dir: Path, pid: int, run_dir: Path) -> bool:
    args = [
        item.decode("utf-8", errors="replace")
        for item in _read_proc_bytes(proc_dir, pid, "cmdline").split(b"\0")
        if item
    ]
    run_dir_text = str(run_dir)
    for index, arg in enumerate(args):
        if arg == "--run-dir" and index + 1 < len(args):
            try:
                return str(Path(args[index + 1]).expanduser().resolve()) == run_dir_text
            except OSError:
                return False
        if arg.startswith("--run-dir="):
            try:
                return str(Path(arg.split("=", 1)[1]).expanduser().resolve()) == run_dir_text
            except OSError:
                return False
    return False


def _pid_matches_run(proc_dir: Path, pid: int, run_dir: Path) -> bool:
    return _pid_has_run_marker(proc_dir, pid, run_dir) or _pid_has_run_argument(proc_dir, pid, run_dir)


def _parent_pid(proc_dir: Path, pid: int) -> int:
    status = _read_proc_bytes(proc_dir, pid, "status").decode("utf-8", errors="replace")
    for line in status.splitlines():
        if line.startswith("PPid:"):
            try:
                return int(line.split(":", 1)[1].strip())
            except ValueError:
                return 0
    return 0


def _ancestor_pids(proc_dir: Path, pid: int) -> Set[int]:
    ancestors: Set[int] = set()
    current = int(pid)
    while current > 1:
        current = _parent_pid(proc_dir, current)
        if current <= 1 or current in ancestors:
            break
        ancestors.add(current)
    return ancestors


def find_run_pids(proc_dir: Path, run_dir: Path, exclude: Iterable[int] = ()) -> List[int]:
    if not proc_dir.is_dir():
        return []
    excluded = {int(pid) for pid in exclude}
    pids: List[int] = []
    for entry in proc_dir.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid in excluded:
            continue
        if _pid_matches_run(proc_dir, pid, run_dir):
            pids.append(pid)
    return sorted(pids)


def process_command(proc_dir: Path, pid: int) -> str:
    command = _read_proc_bytes(proc_dir, pid, "cmdline").replace(b"\0", b" ").strip()
    return command.decode("utf-8", errors="replace") or "<unavailable>"


def find_run_shm(shm_dir: Path, prefix: str) -> List[Path]:
    if not shm_dir.is_dir():
        return []
    return sorted(path for path in shm_dir.glob("{}_*".format(prefix)) if path.is_file())


def run_hostnames(run_dir: Path) -> List[str]:
    try:
        topology = json.loads((run_dir / "topology_resolved.json").read_text(encoding="utf-8"))
        num_nodes = int(topology["num_nodes"])
    except (KeyError, OSError, TypeError, ValueError):
        return []
    if num_nodes <= 0:
        return []

    hostnames: List[str] = []
    for node_rank in range(num_nodes):
        path = run_dir / "nodes" / "node_{:03d}.json".format(node_rank)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        try:
            recorded_rank = int(data["node_rank"])
        except (KeyError, TypeError, ValueError):
            return []
        hostname = str(data.get("hostname", "")).strip()
        if recorded_rank != node_rank or not hostname or hostname in hostnames:
            return []
        hostnames.append(hostname)
    return hostnames


def _signal_matching_pids(proc_dir: Path, run_dir: Path, pids: Iterable[int], sig: int) -> None:
    for pid in pids:
        if not _pid_matches_run(proc_dir, pid, run_dir):
            continue
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            continue
        except PermissionError as exc:
            print("permission denied signaling pid {}: {}".format(pid, exc), file=sys.stderr)


def _wait_for_exit(proc_dir: Path, run_dir: Path, exclude: Set[int], timeout: float) -> List[int]:
    deadline = time.monotonic() + max(0.0, float(timeout))
    remaining = find_run_pids(proc_dir, run_dir, exclude)
    while remaining and time.monotonic() < deadline:
        time.sleep(0.05)
        remaining = find_run_pids(proc_dir, run_dir, exclude)
    return remaining


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean processes and RGB SHM for one rl_finetune run on the current node."
    )
    parser.add_argument("--run-dir", required=True, help="Distributed rl_finetune run directory")
    parser.add_argument("--force", action="store_true", help="Actually signal processes and unlink SHM")
    parser.add_argument("--term-wait", type=float, default=2.0, help="Seconds to wait between TERM and KILL")
    parser.add_argument("--print-hosts", action="store_true", help="Print recorded Slurm hostnames and exit")
    parser.add_argument("--proc-dir", type=Path, default=Path("/proc"), help=argparse.SUPPRESS)
    parser.add_argument("--shm-dir", type=Path, default=Path("/dev/shm"), help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.is_dir():
        print("run directory does not exist: {}".format(run_dir), file=sys.stderr)
        return 2
    if args.print_hosts:
        hostnames = run_hostnames(run_dir)
        for hostname in hostnames:
            print(hostname)
        return 0 if hostnames else 1

    proc_dir = args.proc_dir.resolve()
    shm_dir = args.shm_dir.resolve()
    prefix = run_shm_prefix(run_dir)
    exclude = {os.getpid()}
    exclude.update(_ancestor_pids(proc_dir, os.getpid()))

    pids = find_run_pids(proc_dir, run_dir, exclude)
    shm_paths = find_run_shm(shm_dir, prefix)
    mode = "FORCE" if args.force else "DRY-RUN"
    print("[{}] host={} run_dir={}".format(mode, socket.gethostname(), run_dir))
    print("SHM prefix: {}".format(prefix))
    print("Tagged processes: {}".format(len(pids)))
    for pid in pids:
        print("  pid={} {}".format(pid, process_command(proc_dir, pid)))
    print("SHM files: {}".format(len(shm_paths)))
    for path in shm_paths:
        print("  {}".format(path))

    if not args.force:
        return 0

    _signal_matching_pids(proc_dir, run_dir, pids, signal.SIGTERM)
    remaining = _wait_for_exit(proc_dir, run_dir, exclude, args.term_wait)
    if remaining:
        print("Force-killing {} process(es): {}".format(len(remaining), remaining))
        _signal_matching_pids(proc_dir, run_dir, remaining, signal.SIGKILL)
        _wait_for_exit(proc_dir, run_dir, exclude, 1.0)

    for path in find_run_shm(shm_dir, prefix):
        try:
            path.unlink()
            print("unlinked {}".format(path))
        except FileNotFoundError:
            continue
        except OSError as exc:
            print("failed to unlink {}: {}".format(path, exc), file=sys.stderr)

    residual_pids = find_run_pids(proc_dir, run_dir, exclude)
    residual_shm = find_run_shm(shm_dir, prefix)
    if residual_pids or residual_shm:
        print(
            "cleanup incomplete: residual_pids={} residual_shm={}".format(
                residual_pids,
                [str(path) for path in residual_shm],
            ),
            file=sys.stderr,
        )
        return 1

    print("cleanup complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
