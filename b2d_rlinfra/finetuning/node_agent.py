#!/usr/bin/env python3
"""Per-node launcher for distributed rl_finetune jobs."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import logging
import multiprocessing as mp
import os
import signal
import socket
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

from b2d_rlinfra.finetuning.config_schema import normalize_rl_finetune_config
from b2d_rlinfra.finetuning.coordination import (
    FileControlPlane,
    FileEventBus,
    HeartbeatWriter,
    atomic_write_json,
)
from b2d_rlinfra.finetuning.gpu_mapping import detect_cuda_to_vulkan_mapping
from b2d_rlinfra.finetuning.topology import (
    NodePlan,
    TopologyPlan,
    apply_carla_gpu_mapping,
    build_topology,
    topology_from_file,
)

logger = logging.getLogger("RLFinetune.NodeAgent")
_RUN_ENV_KEY = "B2D_RL_FINETUNE_RUN_DIR"
_SHM_PREFIX_ENV_KEY = "B2D_RL_FINETUNE_SHM_PREFIX"
_SIGNAL_SHUTDOWN_TIMEOUT = 60.0


class _NodeAgentStopRequested(RuntimeError):
    pass


def _run_shm_prefix(run_dir: Path) -> str:
    run_key = str(run_dir.resolve()).encode("utf-8")
    return "rl_finetune_rgb_{}".format(hashlib.sha256(run_key).hexdigest()[:12])


def _configure_logging(verbose: int = 1) -> None:
    level = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}.get(int(verbose), logging.INFO)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _role_for_rank(node_rank: int, role: Optional[str]) -> str:
    resolved = (role or ("learner" if int(node_rank) == 0 else "collector")).lower()
    if resolved not in {"learner", "collector"}:
        raise ValueError("--role must be 'learner' or 'collector'")
    if int(node_rank) == 0 and resolved != "learner":
        raise ValueError("node_rank=0 must use role=learner")
    if int(node_rank) != 0 and resolved != "collector":
        raise ValueError("node_rank!=0 must use role=collector")
    return resolved


def _slurm_job_num_nodes() -> Optional[int]:
    for key in ("SLURM_JOB_NUM_NODES", "SLURM_NNODES"):
        value = os.environ.get(key)
        if value:
            try:
                return int(value)
            except ValueError:
                raise ValueError(f"{key} must be an integer, got {value!r}") from None
    return None


def _validate_slurm_node_count(dist_cfg: Mapping[str, Any]) -> None:
    allocated = _slurm_job_num_nodes()
    if allocated is None:
        return
    configured = int(dist_cfg.get("num_nodes", 1))
    if allocated != configured:
        raise ValueError(
            f"distributed.num_nodes={configured} does not match Slurm allocation "
            f"node count {allocated}; update the config or sbatch --nodes"
        )


def _make_run_dirs(run_dir: Path) -> None:
    for rel in (
        "rollouts",
        "weights",
        "checkpoints",
        "events",
        "heartbeats",
        "control",
        "nodes",
        "logs/nodes",
        "logs/collectors",
        "logs/carla",
        "crash_events",
    ):
        (run_dir / rel).mkdir(parents=True, exist_ok=True)


def _non_empty_dir(path: Path) -> bool:
    if not path.exists():
        return False
    if not path.is_dir():
        return True
    try:
        next(path.iterdir())
    except StopIteration:
        return False
    return True


def _assert_fresh_run_dir(run_dir: Path) -> None:
    stale_paths = [
        run_dir / "control" / "run_state.json",
        run_dir / "topology_resolved.json",
        run_dir / "rollouts" / "manifest.jsonl",
        run_dir / "weights" / "latest.json",
        run_dir / "weights" / "policy_latest.pt",
    ]
    for path in stale_paths:
        if path.exists():
            raise FileExistsError(f"Refusing to reuse existing distributed run artifact: {path}")
    if _non_empty_dir(run_dir / "events"):
        raise FileExistsError(f"Refusing to reuse non-empty distributed events directory: {run_dir / 'events'}")


def _write_rank0_run_files(raw_config: Mapping[str, Any], topology: TopologyPlan, run_dir: Path) -> None:
    _assert_fresh_run_dir(run_dir)
    _make_run_dirs(run_dir)
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(dict(raw_config), f, default_flow_style=False, allow_unicode=True)
    atomic_write_json(run_dir / "topology_resolved.json", topology.to_dict())
    FileControlPlane(run_dir / "control").publish_state("starting", reason="preflight")


def _wait_for_rank0_files(run_dir: Path, timeout: float) -> None:
    deadline = time.monotonic() + float(timeout)
    while time.monotonic() < deadline:
        if (run_dir / "topology_resolved.json").exists() and (run_dir / "control" / "run_state.json").exists():
            return
        time.sleep(1.0)
    raise TimeoutError(f"Timed out waiting for rank0 run files in {run_dir}")


def _env_type(raw_config: Mapping[str, Any]) -> str:
    rl_cfg = dict(raw_config.get("rl_finetune", {}) or {})
    sim_actor = dict(rl_cfg.get("sim_actor", {}) or {})
    return str(rl_cfg.get("env_type", sim_actor.get("env_type", "carla"))).lower()


def _requires_vulkan_loader(raw_config: Mapping[str, Any]) -> bool:
    """Return whether this node will launch a Vulkan-backed CARLA server."""

    if _env_type(raw_config) == "fake":
        return False
    rl_cfg = dict(raw_config.get("rl_finetune", {}) or {})
    sim_actor = dict(rl_cfg.get("sim_actor", {}) or {})
    if not bool(sim_actor.get("manage_servers", True)):
        return False
    env_cfg = dict(raw_config.get("env", {}) or {})
    carla_cfg = dict(env_cfg.get("carla", {}) or {})
    return not bool(carla_cfg.get("null_rhi", False) or carla_cfg.get("opengl", False))


def _validate_vulkan_loader() -> None:
    try:
        ctypes.CDLL("libvulkan.so.1")
    except OSError as exc:
        raise RuntimeError(
            "libvulkan.so.1 is not loadable. Install the system Vulkan loader or set "
            "VULKAN_LIB_DIR to a compatible directory before launching the Slurm job."
        ) from exc


def _resolve_fake_bind_path(raw_config: Mapping[str, Any]) -> Path:
    configured = (raw_config.get("rl_finetune", {}) or {}).get("fake_bind_path")
    return Path(configured) if configured else (_PROJECT_ROOT / "tools" / "fake_bind.so")


def _run_preflight(
    *,
    raw_config: Mapping[str, Any],
    topology: TopologyPlan,
    node_rank: int,
    run_dir: Path,
) -> NodePlan:
    status = "preflight_ok"
    error = ""
    diagnostics: Dict[str, Any] = {
        "hostname": socket.gethostname(),
        "node_rank": int(node_rank),
        "pid": os.getpid(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "slurm_nodeid": os.environ.get("SLURM_NODEID", ""),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "python_no_user_site": os.environ.get("PYTHONNOUSERSITE", ""),
    }
    fallback_node = NodePlan(node_rank=int(node_rank), collectors=[])
    try:
        import numpy  # noqa: F401

        if not bool(os.environ.get("PYTHONNOUSERSITE")):
            diagnostics["warning"] = "PYTHONNOUSERSITE is not set"
        if _env_type(raw_config) != "fake":
            carla_root = os.environ.get("CARLA_ROOT", "")
            if not carla_root:
                raise RuntimeError("CARLA_ROOT is required for non-fake distributed rl_finetune")
            if not Path(carla_root).exists():
                raise RuntimeError(f"CARLA_ROOT does not exist: {carla_root}")
            fake_bind = _resolve_fake_bind_path(raw_config)
            if not fake_bind.exists():
                raise RuntimeError(f"fake_bind.so not found: {fake_bind}")
        if _requires_vulkan_loader(raw_config):
            _validate_vulkan_loader()
            diagnostics["vulkan_loader"] = "libvulkan.so.1"
        else:
            diagnostics["vulkan_loader"] = "not_required"
        probe = run_dir / "nodes" / f"probe_node_{int(node_rank):03d}.json"
        atomic_write_json(probe, {"node_rank": int(node_rank), "probe": "ok"})
        unresolved_node = topology.node(node_rank)
        if int(node_rank) == 0 and unresolved_node.learners:
            import torch

            cuda_count = int(torch.cuda.device_count())
            for learner in unresolved_node.learners:
                device = torch.device(learner.device)
                if device.type != "cuda":
                    continue
                if device.index is None or device.index < 0 or device.index >= cuda_count:
                    raise RuntimeError(
                        f"learner rank {learner.global_rank} requests {learner.device}, "
                        f"but only {cuda_count} CUDA device(s) are visible"
                    )
            diagnostics["learners"] = [learner.to_dict() for learner in unresolved_node.learners]
        needs_mapping = any(str(plan.carla_gpu_id).lower() == "auto" for plan in unresolved_node.collectors)
        if needs_mapping:
            dist_cfg = dict((raw_config.get("rl_finetune", {}) or {}).get("distributed", {}) or {})
            mapping_result = detect_cuda_to_vulkan_mapping(
                project_root=_PROJECT_ROOT,
                env=os.environ,
                env_type=_env_type(raw_config),
                allow_identity_fallback=bool(dist_cfg.get("cuda_vulkan_identity_fallback", True)),
            )
            diagnostics.update(mapping_result.to_dict())
            local_topology = apply_carla_gpu_mapping(topology, mapping_result.cuda_to_vulkan)
            node_plan = local_topology.node(node_rank)
        else:
            diagnostics["cuda_to_vulkan"] = {}
            diagnostics["mapping_source"] = "not_required"
            node_plan = unresolved_node
        diagnostics["collectors"] = [collector.to_dict() for collector in node_plan.collectors]
    except Exception as exc:
        status = "preflight_failed"
        error = str(exc)
        try:
            node_plan = topology.node(node_rank)
        except Exception:
            node_plan = fallback_node
    diagnostics.update(
        {
            "preflight_status": status,
            "error": error,
            "updated_at": time.time(),
        }
    )
    atomic_write_json(run_dir / "nodes" / f"node_{int(node_rank):03d}.json", diagnostics)
    if status != "preflight_ok":
        raise RuntimeError(error)
    return node_plan


def _wait_for_preflight(
    run_dir: Path,
    topology: TopologyPlan,
    timeout: float,
    control: FileControlPlane,
) -> None:
    deadline = time.monotonic() + float(timeout)
    expected = int(topology.num_nodes)
    while time.monotonic() < deadline:
        control_state = str(control.read_state(default={}).get("state", "")).lower()
        if control_state == "failed":
            raise RuntimeError("Distributed run failed while waiting for preflight")
        if control_state in FileControlPlane.STOP_STATES:
            raise _NodeAgentStopRequested(f"run_state={control_state} while waiting for preflight")
        ok = 0
        for rank in range(expected):
            path = run_dir / "nodes" / f"node_{rank:03d}.json"
            if not path.exists():
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue
            status = str(data.get("preflight_status", ""))
            if status == "preflight_failed":
                raise RuntimeError(f"node {rank} preflight failed: {data.get('error', '')}")
            if status == "preflight_ok":
                ok += 1
        if ok >= expected:
            return
        time.sleep(1.0)
    raise TimeoutError("Timed out waiting for all node preflight checks")


def _start_collectors(
    *,
    config: Any,
    rl_config: Mapping[str, Any],
    node_plan: NodePlan,
    run_dir: Path,
    shm_prefix: str,
    stop_event: Any,
) -> List[mp.Process]:
    from b2d_rlinfra.finetuning.collector import collector_entry

    ctx = mp.get_context(str(rl_config.get("start_method", "spawn")))
    processes: List[mp.Process] = []
    previous_run_env = os.environ.get(_RUN_ENV_KEY)
    os.environ[_RUN_ENV_KEY] = str(run_dir.resolve())
    os.environ[_SHM_PREFIX_ENV_KEY] = str(shm_prefix)
    if previous_run_env is not None:
        os.environ[f"{_RUN_ENV_KEY}_PREVIOUS"] = previous_run_env
    stagger = float((rl_config.get("distributed", {}) or {}).get("stagger_seconds", 0.0))
    for plan in node_plan.collectors:
        process = ctx.Process(
            target=collector_entry,
            args=(
                int(plan.collector_id),
                config,
                dict(rl_config),
                str(run_dir / "rollouts"),
                str(run_dir / "weights"),
                str(plan.device),
                None,
                stop_event,
                plan.to_dict(),
                str(run_dir),
                "file",
                str(run_dir / "heartbeats"),
            ),
            daemon=False,
        )
        process.start()
        processes.append(process)
        if stagger > 0:
            time.sleep(stagger)
    return processes


def _join_processes(processes: List[mp.Process], timeout: float) -> None:
    deadline = time.monotonic() + max(0.0, float(timeout))
    for process in processes:
        remaining = max(0.0, deadline - time.monotonic())
        process.join(timeout=remaining)


def _stop_collectors(processes: List[mp.Process], stop_event: Any, timeout: float) -> List[int]:
    stop_event.set()
    _join_processes(processes, timeout)

    alive = [process for process in processes if process.is_alive()]
    if alive:
        logger.warning(
            "%s collector(s) did not stop within the shared %.1fs timeout; terminating pids=%s",
            len(alive),
            timeout,
            [process.pid for process in alive],
        )
        for process in alive:
            process.terminate()
        _join_processes(alive, 5.0)

    alive = [process for process in alive if process.is_alive()]
    if alive:
        logger.warning("Force-killing collector pids=%s", [process.pid for process in alive])
        for process in alive:
            try:
                process.kill()
            except (AttributeError, ProcessLookupError):
                if process.pid is not None:
                    try:
                        os.kill(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
        _join_processes(alive, 5.0)

    remaining = [int(process.pid) for process in alive if process.is_alive() and process.pid is not None]
    if remaining:
        logger.warning("Collector pids still alive after SIGKILL: %s", remaining)
    return remaining


def _cleanup_local_rgb_shm(shm_prefix: str, shm_dir: Path = Path("/dev/shm")) -> List[Path]:
    if not shm_dir.exists():
        return []
    removed: List[Path] = []
    for path in sorted(shm_dir.glob("{}_*".format(shm_prefix))):
        try:
            path.unlink()
            removed.append(path)
        except FileNotFoundError:
            continue
        except OSError as exc:
            logger.warning("Failed to unlink RGB shm %s: %s", path, exc)
    if removed:
        logger.info("Removed %s RGB shm file(s) for prefix %s", len(removed), shm_prefix)
    return removed


def _cleanup_local_carla(raw_config: Mapping[str, Any], rl_config: Mapping[str, Any], node_plan: NodePlan) -> None:
    sim_actor_cfg = dict(rl_config.get("sim_actor", {}) or {})
    if not bool(sim_actor_cfg.get("manage_servers", True)) or _env_type(raw_config) == "fake":
        return
    script = _PROJECT_ROOT / "tools" / "runtime" / "kill_by_host.sh"
    if not script.exists():
        logger.warning("CARLA cleanup script not found: %s", script)
        return
    seen_hosts = []
    for plan in node_plan.collectors:
        if plan.host not in seen_hosts:
            seen_hosts.append(plan.host)
    cleanup_timeout = float(rl_config.get("carla_cleanup_timeout", 30.0))

    def _cleanup_host(host: str) -> None:
        try:
            subprocess.run(
                ["bash", str(script), str(host)],
                cwd=str(_PROJECT_ROOT),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=cleanup_timeout,
                check=False,
            )
        except Exception as exc:
            logger.warning("Failed to cleanup CARLA processes for host %s: %s", host, exc)

    if seen_hosts:
        max_workers = min(len(seen_hosts), 16)
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="carla-cleanup") as executor:
            list(executor.map(_cleanup_host, seen_hosts))


def _cleanup_tagged_child_processes(run_dir: Path) -> None:
    run_dir_text = str(run_dir.resolve())
    pids = _find_processes_by_env(_RUN_ENV_KEY, run_dir_text)
    pids = [pid for pid in pids if pid != os.getpid()]
    if not pids:
        return
    logger.warning("Cleaning %s lingering rl_finetune child process(es): %s", len(pids), pids)
    for sig, wait_time in ((signal.SIGTERM, 1.0), (signal.SIGKILL, 0.0)):
        remaining = []
        for pid in pids:
            if not _is_process_running(pid):
                continue
            try:
                os.kill(pid, sig)
                remaining.append(pid)
            except ProcessLookupError:
                continue
            except PermissionError as exc:
                logger.warning("No permission to signal lingering process pid=%s: %s", pid, exc)
        if wait_time > 0.0:
            deadline = time.time() + wait_time
            while time.time() < deadline and any(_is_process_running(pid) for pid in remaining):
                time.sleep(0.05)
        pids = [pid for pid in remaining if _is_process_running(pid)]
        if not pids:
            return
    if pids:
        logger.warning("Lingering rl_finetune child process(es) still alive after cleanup: %s", pids)


def _find_processes_by_env(key: str, value: str) -> List[int]:
    marker = f"{key}={value}".encode()
    pids: List[int] = []
    proc_root = Path("/proc")
    if not proc_root.exists():
        return pids
    for proc_dir in proc_root.iterdir():
        if not proc_dir.name.isdigit():
            continue
        try:
            environ = (proc_dir / "environ").read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if marker in environ.split(b"\0"):
            pids.append(int(proc_dir.name))
    return sorted(pids)


def _is_process_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _wait_for_stop(control: FileControlPlane, *, timeout: float) -> None:
    last_seq = -1
    last_seen = time.monotonic()
    while True:
        state = control.read_state(default={})
        if str(state.get("state", "")).lower() in FileControlPlane.STOP_STATES:
            return
        heartbeat = control.read_heartbeat(default={})
        seq = int(heartbeat.get("learner_heartbeat_seq", -1))
        if seq > last_seq:
            last_seq = seq
            last_seen = time.monotonic()
        elif time.monotonic() - last_seen > float(timeout):
            control.publish_state("failed", reason="learner_heartbeat_timeout")
            return
        time.sleep(1.0)


def run_node_agent(args: argparse.Namespace) -> int:
    from b2d_rlinfra.learning.utils.config import load_config

    config = load_config(args.config)
    raw_config = normalize_rl_finetune_config(config.to_dict())
    config.raw_config = raw_config
    config.env_config = raw_config.get("env")
    rl_config = dict(raw_config.get("rl_finetune", {}) or {})
    dist_cfg = dict(rl_config.get("distributed", {}) or {})
    if not bool(dist_cfg.get("enabled", False)):
        raise ValueError("node_agent requires rl_finetune.distributed.enabled=true")
    _validate_slurm_node_count(dist_cfg)
    _configure_logging(int(rl_config.get("verbose", 1)))

    node_rank = int(args.node_rank if args.node_rank is not None else os.environ.get("SLURM_NODEID", 0))
    role = _role_for_rank(node_rank, args.role)
    run_dir = Path(args.run_dir).resolve()
    shm_prefix = _run_shm_prefix(run_dir)
    if args.init_from_checkpoint is not None and not Path(args.init_from_checkpoint).is_file():
        raise ValueError(f"--init-from-checkpoint must point to a checkpoint file: {args.init_from_checkpoint}")
    startup_timeout = float(dist_cfg.get("startup_timeout", 600.0))
    control = FileControlPlane(run_dir / "control")
    topology: TopologyPlan
    phase = {"value": "starting"}
    node_heartbeat = HeartbeatWriter(
        run_dir / "heartbeats" / f"node_{node_rank:03d}.json",
        interval=float(dist_cfg.get("heartbeat_interval", 10.0)),
        payload_factory=lambda seq: {
            "seq": seq,
            "node_rank": node_rank,
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "phase": phase["value"],
        },
    )
    stop_event = mp.get_context(str(rl_config.get("start_method", "spawn"))).Event()
    processes: List[mp.Process] = []
    shutting_down = {"value": False}

    def _signal_handler(signum: int, _frame: Any) -> None:
        shutting_down["value"] = True
        phase["value"] = f"signal_{signum}"
        if role == "learner":
            control.request_stop(reason=f"signal_{signum}")
        stop_event.set()

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    node_heartbeat.start()
    try:
        if node_rank == 0:
            topology = build_topology(raw_config)
            _write_rank0_run_files(raw_config, topology, run_dir)
        else:
            _wait_for_rank0_files(run_dir, startup_timeout)
            topology = topology_from_file(run_dir / "topology_resolved.json")

        node_plan = _run_preflight(
            raw_config=raw_config,
            topology=topology,
            node_rank=node_rank,
            run_dir=run_dir,
        )
        if node_rank == 0:
            _wait_for_preflight(run_dir, topology, startup_timeout, control)
            control.publish_state("running", reason="preflight_ok")
        else:
            deadline = time.monotonic() + startup_timeout
            while time.monotonic() < deadline:
                state = str(control.read_state(default={}).get("state", "")).lower()
                if state == "running":
                    break
                if state == "failed":
                    return 1
                if state in FileControlPlane.STOP_STATES:
                    raise _NodeAgentStopRequested(f"run_state={state} before launch")
                time.sleep(1.0)
            else:
                raise TimeoutError("Timed out waiting for distributed run_state=running")

        phase["value"] = "launching_collectors"
        processes = _start_collectors(
            config=config,
            rl_config=rl_config,
            node_plan=node_plan,
            run_dir=run_dir,
            shm_prefix=shm_prefix,
            stop_event=stop_event,
        )
        phase["value"] = "running"
        if role == "learner":
            learner_cfg = dict(rl_config.get("learner", {}) or {})
            if str(learner_cfg.get("strategy", "single")).lower() == "ddp":
                from b2d_rlinfra.finetuning.learner_distributed import run_node0_learner_group

                run_node0_learner_group(
                    config_path=str(Path(args.config).resolve()),
                    run_dir=str(run_dir),
                    world_size=topology.learner_world_size,
                    max_updates=args.max_updates,
                    init_from_checkpoint=args.init_from_checkpoint,
                )
            else:
                import torch

                from b2d_rlinfra.finetuning.coordinator import Coordinator

                coordinator = Coordinator(
                    config=config,
                    config_path=args.config,
                    run_dir=str(run_dir),
                    device=torch.device(str(rl_config.get("device", "cuda:0"))),
                    topology=topology,
                    event_bus=FileEventBus(run_dir / "events"),
                    control_plane=control,
                    spawn_collectors=False,
                    init_from_checkpoint=args.init_from_checkpoint,
                )
                coordinator.run(max_updates=args.max_updates)
        else:
            _wait_for_stop(control, timeout=float(dist_cfg.get("heartbeat_timeout", 900.0)))
        return 0 if not shutting_down["value"] else 143
    except _NodeAgentStopRequested as exc:
        logger.info("node_agent stopping on control-plane request: %s", exc)
        return 143 if shutting_down["value"] else 0
    except Exception as exc:
        logger.exception("node_agent failed: %s", exc)
        try:
            control.publish_state("failed", reason=str(exc))
        except Exception:
            pass
        return 1
    finally:
        phase["value"] = "stopping"
        shutdown_timeout = float(rl_config.get("shutdown_timeout", 120.0))
        if shutting_down["value"]:
            shutdown_timeout = min(shutdown_timeout, _SIGNAL_SHUTDOWN_TIMEOUT)
        try:
            _stop_collectors(processes, stop_event, shutdown_timeout)
        except Exception:
            logger.exception("Failed while stopping collector processes")
        try:
            try:
                _cleanup_tagged_child_processes(run_dir)
            except Exception:
                logger.exception("Failed while cleaning tagged child processes")
            try:
                _cleanup_local_rgb_shm(shm_prefix)
            except Exception:
                logger.exception("Failed while cleaning local RGB shm")
            if "node_plan" in locals():
                try:
                    _cleanup_local_carla(raw_config, rl_config, node_plan)
                except Exception:
                    logger.exception("Failed while cleaning local CARLA processes")
        finally:
            atomic_write_json(
                run_dir / "nodes" / f"node_{node_rank:03d}_exit.json",
                {
                    "node_rank": node_rank,
                    "role": role,
                    "phase": "exited",
                    "pid": os.getpid(),
                    "updated_at": time.time(),
                },
            )
            node_heartbeat.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Distributed rl_finetune node agent")
    parser.add_argument("--config", "-c", required=True, type=str, help="Path to YAML config")
    parser.add_argument("--run-dir", required=True, type=str, help="Shared run directory")
    parser.add_argument("--node-rank", type=int, default=None, help="Node rank; defaults to SLURM_NODEID or 0")
    parser.add_argument("--role", type=str, default=None, help="learner on rank0, collector on other ranks")
    parser.add_argument("--max-updates", type=int, default=None, help="Override rl_finetune.max_updates on learner")
    parser.add_argument("--init-from-checkpoint", type=str, default=None, help="Initialize learner from RL checkpoint")
    args = parser.parse_args()
    raise SystemExit(run_node_agent(args))


if __name__ == "__main__":
    main()
