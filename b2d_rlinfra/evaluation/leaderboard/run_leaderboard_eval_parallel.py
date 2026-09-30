#!/usr/bin/env python3
"""Parallel Leaderboard and Bench2Drive evaluation entry point."""
from __future__ import annotations

import argparse
import copy
import collections
import importlib
import importlib.util
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import yaml

from b2d_rlinfra.simulation.runners.carla_server_manager import CARLAServerManager

__layer__ = (5, "Evaluation")


_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parents[2]
_RUNNER_PARALLEL_UTILS_PATH = (
    _PROJECT_ROOT / "b2d_rlinfra" / "learning" / "training" / "parallel_utils.py"
)
_SCORE_VERSIONS_PATH = _THIS_DIR / "score_versions.py"
_PARALLEL_HEAVY_TOWNS = ("Town11", "Town12", "Town13")
_DEFAULT_ROUTE_RETRY_ON_EMPTY_RECORD = 1
_HOST_CLEANUP_WAIT_TIMEOUT = 15.0
_SHARD_SERVER_EXIT_WAIT_AFTER_CRASH = 120.0
_SHARD_SERVER_MONITOR_POLL_INTERVAL = 10.0
_SHARD_SERVER_MONITOR_TERMINATE_GRACE = 10.0
_B2D_SUCCESS_RULE = (
    "B2D: status in {Completed, Perfect} and no infractions except min_speed_infractions"
)
_ENTRY_STATUS_PRIORITY = {
    "Finished": 0,
    "Started": 1,
    "Rejected": 2,
    "Crashed": 3,
    "Invalid": 4,
}
_ROUTE_RECORD_ID_RE = re.compile(r"^RouteScenario_(.+)_rep(\d+)$")


def _load_runner_parallel_utils():
    module_name = "rl_runner_parallel_utils"
    spec = importlib.util.spec_from_file_location(
        module_name,
        str(_RUNNER_PARALLEL_UTILS_PATH),
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load parallel_utils from {_RUNNER_PARALLEL_UTILS_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_RUNNER_PARALLEL_UTILS = _load_runner_parallel_utils()
kill_hosts = _RUNNER_PARALLEL_UTILS.kill_hosts
load_json_if_exists = _RUNNER_PARALLEL_UTILS.load_json_if_exists
load_selected_route_elements = _RUNNER_PARALLEL_UTILS.load_selected_route_elements
materialize_episode_routes = _RUNNER_PARALLEL_UTILS.materialize_episode_routes
normalize_carla_slots = _RUNNER_PARALLEL_UTILS.normalize_carla_slots
now_iso = _RUNNER_PARALLEL_UTILS.now_iso
resolve_path = _RUNNER_PARALLEL_UTILS.resolve_path
sanitize_tag = _RUNNER_PARALLEL_UTILS.sanitize_tag
split_episodes_balanced_by_town = _RUNNER_PARALLEL_UTILS.split_episodes_balanced_by_town
validate_parallel_slots = _RUNNER_PARALLEL_UTILS.validate_parallel_slots
write_json_file = _RUNNER_PARALLEL_UTILS.write_json_file
write_route_shard = _RUNNER_PARALLEL_UTILS.write_route_shard


def _load_statistics_manager_symbols():
    module = importlib.import_module("leaderboard.utils.statistics_manager")
    return module.RouteRecord, module.StatisticsManager, module.to_route_record


def _load_score_versions():
    module_name = "rl_leaderboard_score_versions"
    spec = importlib.util.spec_from_file_location(
        module_name,
        str(_SCORE_VERSIONS_PATH),
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load score_versions from {_SCORE_VERSIONS_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_SCORE_VERSIONS = _load_score_versions()
ScoreVersionError = _SCORE_VERSIONS.ScoreVersionError
build_v20_statistics_from_payload = _SCORE_VERSIONS.build_v20_statistics_from_payload
extract_summary_metrics_from_statistics_payload = _SCORE_VERSIONS.extract_summary_metrics_from_statistics_payload


def _enable_line_buffering() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(line_buffering=True)


def _parse_bool(raw_value: str | bool) -> bool:
    if isinstance(raw_value, bool):
        return raw_value
    return str(raw_value).strip().lower() in ("1", "true", "yes", "on")


def _parse_csv_list(raw_value: Optional[str], *, cast=None) -> Optional[List[Any]]:
    if raw_value is None:
        return None
    value = str(raw_value).strip()
    if not value:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    if cast is None:
        return items
    return [cast(item) for item in items]


def _parse_extra_args(raw_value: Any) -> List[str]:
    if raw_value is None:
        return []
    if isinstance(raw_value, str):
        try:
            return shlex.split(raw_value)
        except Exception:
            return [token for token in raw_value.split() if token]
    if isinstance(raw_value, (list, tuple)):
        return [str(item) for item in raw_value if str(item).strip()]
    return []


def _build_balanced_episode_order(
    episodes: Sequence[Any],
    worker_count: int,
) -> List[Any]:
    chunks = split_episodes_balanced_by_town(
        episodes,
        worker_count,
        heavy_towns=_PARALLEL_HEAVY_TOWNS,
    )
    ordered: List[Any] = []
    max_len = max((len(chunk) for chunk in chunks), default=0)
    for offset in range(max_len):
        for chunk in chunks:
            if offset < len(chunk):
                ordered.append(chunk[offset])
    return ordered


def _build_route_jobs(
    ordered_episodes: Sequence[Any],
) -> List[Dict[str, Any]]:
    repetitions_by_route: Dict[str, int] = {}
    jobs: List[Dict[str, Any]] = []
    for job_index, episode in enumerate(ordered_episodes):
        route_numeric_id = str(episode.route_id or "").strip()
        repetition_index = repetitions_by_route.get(route_numeric_id, 0)
        repetitions_by_route[route_numeric_id] = repetition_index + 1
        town = str(episode.element.attrib.get("town", "Unknown") or "Unknown")
        scenarios = episode.element.findall(".//scenario")
        scenario_type = scenarios[0].attrib.get("type", "Unknown") if scenarios else "Unknown"
        jobs.append({
            "job_index": job_index,
            "route_id": route_numeric_id,
            "repetition_index": repetition_index,
            "record_id": _build_route_record_id(route_numeric_id, repetition_index),
            "episode": episode,
            "town": town,
            "scenario_type": scenario_type,
            "status": "pending",
            "attempts": [],
            "assigned_worker": None,
            "retry_count": 0,
            "result_source": None,
            "final_record_found": False,
            "final_record": None,
            "final_artifacts": {},
            "started_at": None,
            "finished_at": None,
        })
    return jobs


def _build_job_dir_name(job: Dict[str, Any]) -> str:
    route_tag = sanitize_tag(str(job.get("route_id", "unknown")))
    repetition_index = int(job.get("repetition_index", 0))
    return f"job_{int(job.get('job_index', 0)):03d}_route_{route_tag}_rep{repetition_index}"


def _extract_route_identity(route_record_id: Any) -> tuple[str, Optional[int]]:
    raw = str(route_record_id or "").strip()
    if not raw:
        return "", None
    match = _ROUTE_RECORD_ID_RE.match(raw)
    if match:
        return match.group(1), int(match.group(2))
    if raw.startswith("RouteScenario_"):
        return raw[len("RouteScenario_"):], None
    return raw, None


def _extract_attempt_record(
    *,
    statistics_path: Optional[str | Path],
    expected_route_id: str,
    expected_record_id: str,
) -> tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    payload = load_json_if_exists(statistics_path)
    if not payload:
        return None, None

    records = payload.get("_checkpoint", {}).get("records", []) or []
    selected_record: Optional[Dict[str, Any]] = None
    for record_dict in records:
        route_id, _ = _extract_route_identity(record_dict.get("route_id", ""))
        if route_id == expected_route_id:
            selected_record = dict(record_dict)
            break
    if selected_record is None:
        return payload, None

    selected_record["route_id"] = expected_record_id
    selected_record["index"] = -1
    return payload, selected_record


def _summarize_statistics_payload(payload: Optional[Mapping[str, Any]]) -> str:
    if not isinstance(payload, Mapping):
        return "payload=none"
    checkpoint = payload.get("_checkpoint", {}) if isinstance(payload.get("_checkpoint", {}), Mapping) else {}
    progress = checkpoint.get("progress", [])
    records = checkpoint.get("records", [])
    entry_status = payload.get("entry_status", "unknown")
    if not isinstance(progress, list):
        progress = []
    if not isinstance(records, list):
        records = []
    return (
        f"entry_status={entry_status!r}, progress={progress}, "
        f"records={len(records)}"
    )


def _describe_returncode(returncode: Optional[int]) -> str:
    if returncode is None:
        return "None"
    if returncode > 128:
        signed = returncode - 256
        return f"{returncode} (signed={signed})"
    return str(returncode)


def _wait_for_host_carla_exit(
    host: str,
    timeout: float = _HOST_CLEANUP_WAIT_TIMEOUT,
) -> None:
    manager = CARLAServerManager.__new__(CARLAServerManager)
    remaining = manager._find_carla_pids_by_host(str(host))
    if not remaining:
        return
    remaining, _ = manager._wait_for_processes_exit(remaining, timeout=max(0.0, float(timeout)))
    if remaining:
        print(
            f"[leaderboard-worker] WARNING: lingering CARLA processes remain on host={host}: {remaining}",
            flush=True,
        )


def _resolve_fake_bind_lib() -> Optional[Path]:
    env_override = os.environ.get("FAKE_BIND_LIB")
    if env_override:
        candidate = Path(env_override).expanduser()
        if candidate.exists():
            return candidate.resolve()

    default_path = _PROJECT_ROOT / "tools" / "fake_bind.so"
    if default_path.exists():
        return default_path

    return None


def _prepare_parallel_env_config(
    *,
    config: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    child = copy.deepcopy(config)
    env_cfg = child.get("env", child)
    carla_cfg = env_cfg.setdefault("carla", {})

    hosts = _parse_csv_list(os.environ.get("CARLA_HOSTS"))
    if hosts is not None:
        carla_cfg["host"] = hosts

    ports = _parse_csv_list(os.environ.get("CARLA_PORTS"), cast=int)
    if ports is not None:
        carla_cfg["port"] = ports
    elif os.environ.get("CARLA_PORT"):
        carla_cfg["port"] = int(os.environ["CARLA_PORT"])

    tm_ports = _parse_csv_list(os.environ.get("TM_PORTS"), cast=int)
    if tm_ports is not None:
        carla_cfg["traffic_manager_port"] = tm_ports
    elif os.environ.get("TM_PORT"):
        carla_cfg["traffic_manager_port"] = int(os.environ["TM_PORT"])

    tm_seeds = _parse_csv_list(os.environ.get("TM_SEEDS"), cast=int)
    if tm_seeds is not None:
        carla_cfg["traffic_manager_seed"] = tm_seeds

    gpu_ids = _parse_csv_list(os.environ.get("GPU_IDS"))
    if gpu_ids is not None:
        normalized_gpu_ids: List[Any] = []
        for value in gpu_ids:
            try:
                normalized_gpu_ids.append(int(value))
            except ValueError:
                normalized_gpu_ids.append(value)
        carla_cfg["gpu_id"] = normalized_gpu_ids

    return child


def _start_single_carla_server_for_slot(
    *,
    carla_root: str,
    carla_cfg: Dict[str, Any],
    slot: Any,
) -> CARLAServerManager:
    fake_bind_lib = _resolve_fake_bind_lib()
    quality_level = "Default"
    server_manager = CARLAServerManager(
        carla_root=carla_root,
        # The upstream evaluator startup path does not pass ``-fps``.
        default_fps=None,
        default_quality=quality_level,
        auto_restart=True,
        fake_bind_lib=str(fake_bind_lib) if fake_bind_lib is not None else None,
    )
    results = server_manager.start_servers(
        ports=[int(slot.port)],
        gpu_ids=[slot.gpu_id],
        hosts=[str(slot.host)],
        timeout=float(carla_cfg.get("server_wait_timeout", 120.0) or 120.0),
        quality_level=quality_level,
        multihome=carla_cfg.get("multihome"),
        render_offscreen=bool(carla_cfg.get("render_offscreen", True)),
        no_sound=bool(carla_cfg.get("no_sound", True)),
        null_rhi=bool(carla_cfg.get("null_rhi", True)),
        opengl=bool(carla_cfg.get("opengl", False)),
        vulkan=bool(carla_cfg.get("vulkan", False)),
        no_steam=bool(carla_cfg.get("no_steam", False)),
        extra_args=_parse_extra_args(carla_cfg.get("extra_args")),
    )
    success = bool(results.get((str(slot.host), int(slot.port)), False))
    if not success:
        raise RuntimeError(f"Failed to start CARLA server on {slot.host}:{slot.port}")
    return server_manager


def _run_shard_worker(args: argparse.Namespace) -> int:
    config_path = resolve_path(args.config)
    checkpoint_path = resolve_path(args.checkpoint)
    evaluator_path = resolve_path(Path(args.leaderboard_root) / "leaderboard" / "leaderboard_evaluator.py")
    agent_path = resolve_path(args.agent)
    routes_file = resolve_path(args.child_routes_file)
    statistics_file = resolve_path(args.child_statistics_file)
    live_results_file = resolve_path(args.child_live_file)
    shard_output_root = resolve_path(args.child_output_root)
    record_dir = resolve_path(args.child_record_dir) if args.child_record_dir else None
    carla_root = os.environ.get("CARLA_ROOT")
    if not carla_root:
        raise ValueError("CARLA_ROOT must be set to auto-start CARLA servers")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    parallel_env_config = _prepare_parallel_env_config(config=config, args=args)
    env_cfg = parallel_env_config.get("env", parallel_env_config)
    carla_cfg = env_cfg.get("carla", {})
    slot = type("ShardSlot", (), {
        "host": str(args.child_host),
        "port": int(args.child_port),
        "traffic_manager_port": int(args.child_tm_port),
        "traffic_manager_seed": int(args.child_tm_seed),
        "gpu_id": int(args.child_gpu_id) if str(args.child_gpu_id).isdigit() else args.child_gpu_id,
    })()

    server_manager: Optional[CARLAServerManager] = None
    evaluator_proc: Optional[subprocess.Popen] = None
    server_exit_deadline: Optional[float] = None
    previous_sigint = signal.getsignal(signal.SIGINT)
    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def _signal_handler(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}")

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    try:
        server_manager = _start_single_carla_server_for_slot(
            carla_root=carla_root,
            carla_cfg=carla_cfg,
            slot=slot,
        )
        child_env = os.environ.copy()
        child_env["CHECKPOINT_PATH"] = str(checkpoint_path)
        child_env["STOCHASTIC"] = "true" if args.stochastic else "false"
        child_env["OUTPUT_ROOT"] = str(shard_output_root)
        # Preserve evaluator output emitted immediately before a crash.
        child_env["PYTHONUNBUFFERED"] = "1"
        child_env["PYTHONFAULTHANDLER"] = "1"

        cmd = [
            sys.executable,
            str(evaluator_path),
            "--host", str(slot.host),
            "--port", str(slot.port),
            "--traffic-manager-port", str(slot.traffic_manager_port),
            "--traffic-manager-seed", str(slot.traffic_manager_seed),
            "--routes", str(routes_file),
            "--repetitions", str(args.repetitions),
            "--agent", str(agent_path),
            "--agent-config", str(config_path),
            "--track", str(args.track),
            "--checkpoint", str(statistics_file),
            "--debug-checkpoint", str(live_results_file),
            "--timeout", str(args.timeout),
            "--debug", str(args.debug),
        ]
        if record_dir is not None:
            record_dir.mkdir(parents=True, exist_ok=True)
            cmd.extend(["--record", str(record_dir)])

        evaluator_proc = subprocess.Popen(
            cmd,
            cwd=str(_PROJECT_ROOT),
            env=child_env,
        )
        while True:
            returncode = evaluator_proc.poll()
            if returncode is not None:
                return returncode

            if server_manager is not None and not server_manager.is_server_running(str(slot.host), int(slot.port)):
                if server_exit_deadline is None:
                    server_exit_deadline = time.time() + _SHARD_SERVER_EXIT_WAIT_AFTER_CRASH
                    print(
                        f"[leaderboard-shard] detected CARLA server exit for {slot.host}:{slot.port}, "
                        f"waiting up to {_SHARD_SERVER_EXIT_WAIT_AFTER_CRASH:.0f}s for evaluator to exit",
                        flush=True,
                    )
                elif time.time() >= server_exit_deadline:
                    print(
                        f"[leaderboard-shard] evaluator still alive {_SHARD_SERVER_EXIT_WAIT_AFTER_CRASH:.0f}s "
                        f"after CARLA server exit for {slot.host}:{slot.port}, terminating evaluator",
                        flush=True,
                    )
                    try:
                        evaluator_proc.terminate()
                    except Exception:
                        pass

                    deadline = time.time() + _SHARD_SERVER_MONITOR_TERMINATE_GRACE
                    while time.time() < deadline:
                        returncode = evaluator_proc.poll()
                        if returncode is not None:
                            return returncode
                        time.sleep(0.2)

                    if evaluator_proc.poll() is None:
                        try:
                            evaluator_proc.kill()
                        except Exception:
                            pass
                    return evaluator_proc.wait()

            time.sleep(_SHARD_SERVER_MONITOR_POLL_INTERVAL)
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)
        if evaluator_proc is not None and evaluator_proc.poll() is None:
            try:
                evaluator_proc.terminate()
            except Exception:
                pass
        if server_manager is not None:
            try:
                server_manager.stop_all()
            except Exception as exc:
                print(f"[leaderboard-shard] WARNING: manager stop failed: {exc}", flush=True)
        kill_hosts([str(slot.host)], verbose=True)
        _wait_for_host_carla_exit(str(slot.host))


def _build_route_record_id(route_numeric_id: str, repetition_index: int) -> str:
    return f"RouteScenario_{route_numeric_id}_rep{repetition_index}"


def _build_missing_route_record(route_id: str, reason: str):
    RouteRecord, _, _ = _load_statistics_manager_symbols()
    record = RouteRecord()
    record.route_id = route_id
    record.status = f"Failed - {reason}"
    record.index = -1
    return record


def _route_record_is_b2d_success(record: Optional[Mapping[str, Any]]) -> bool:
    if not record:
        return False

    status = str(record.get("status", "")).strip()
    if status not in ("Completed", "Perfect"):
        return False

    infractions = record.get("infractions", {}) or {}
    for key, value in infractions.items():
        if key == "min_speed_infractions":
            continue
        if value:
            return False
    return True


def _job_is_b2d_success(job: Mapping[str, Any]) -> bool:
    if not bool(job.get("final_record_found")):
        return False
    return _route_record_is_b2d_success(job.get("final_record"))


def _build_b2d_success_statistics(
    job_entries: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    planned_route_count = len(job_entries)
    success_count = 0
    scenario_buckets: Dict[str, Dict[str, int]] = collections.defaultdict(
        lambda: {"success_count": 0, "total_count": 0}
    )

    for job in job_entries:
        scenario_type = str(job.get("scenario_type") or "Unknown")
        is_success = _job_is_b2d_success(job)
        scenario_bucket = scenario_buckets[scenario_type]
        scenario_bucket["total_count"] += 1
        if is_success:
            success_count += 1
            scenario_bucket["success_count"] += 1

    scenario_success_summary = []
    for scenario_type in sorted(scenario_buckets):
        bucket = scenario_buckets[scenario_type]
        total_count = int(bucket["total_count"])
        scenario_success_summary.append({
            "scenario_type": scenario_type,
            "success_count": int(bucket["success_count"]),
            "total_count": total_count,
            "success_rate": (
                float(bucket["success_count"]) / float(total_count)
                if total_count > 0 else 0.0
            ),
        })

    return {
        "success_rule": _B2D_SUCCESS_RULE,
        "planned_route_count": planned_route_count,
        "success_count": success_count,
        "non_success_count": planned_route_count - success_count,
        "success_rate": (
            float(success_count) / float(planned_route_count)
            if planned_route_count > 0 else 0.0
        ),
        "scenario_success_summary": scenario_success_summary,
    }


def _select_entry_status(
    child_payloads: Sequence[Optional[Dict[str, Any]]],
    *,
    has_missing_route_records: bool,
) -> str:
    if has_missing_route_records:
        return "Crashed"

    candidate_statuses: List[str] = []
    for payload in child_payloads:
        if not payload:
            continue
        status = str(payload.get("entry_status", "")).strip()
        if status:
            candidate_statuses.append(status)

    if not candidate_statuses:
        return "Finished"

    return max(candidate_statuses, key=lambda status: _ENTRY_STATUS_PRIORITY.get(status, -1))


def _merge_statistics(
    *,
    job_entries: Sequence[Dict[str, Any]],
    planned_route_ids: Sequence[str],
    output_path: Path,
    live_output_path: Path,
) -> Dict[str, Any]:
    _, StatisticsManager, to_route_record = _load_statistics_manager_symbols()
    record_by_route_id: Dict[str, Dict[str, Any]] = {}
    sensors: List[str] = []
    child_payloads: List[Optional[Dict[str, Any]]] = []
    duplicate_route_ids: List[str] = []

    for job in job_entries:
        payload = job.get("final_payload")
        child_payloads.append(payload)
        if payload and not sensors:
            child_sensors = payload.get("sensors") or []
            if child_sensors:
                sensors = list(child_sensors)

        record_dict = job.get("final_record")
        route_id = str(job.get("record_id", "")).strip()
        if not route_id:
            continue
        if record_dict is not None:
            if route_id in record_by_route_id:
                duplicate_route_ids.append(route_id)
            record_by_route_id[route_id] = dict(record_dict)

    merged_records = []
    missing_route_ids: List[str] = []
    for route_id in planned_route_ids:
        record_dict = record_by_route_id.get(route_id)
        if record_dict is None:
            missing_route_ids.append(route_id)
            merged_records.append(_build_missing_route_record(route_id, "Missing route result after retries"))
        else:
            merged_records.append(to_route_record(record_dict))

    manager = StatisticsManager(str(output_path), str(live_output_path))
    manager._results.checkpoint.records = merged_records
    manager.save_progress(len(planned_route_ids), len(planned_route_ids))
    manager.save_sensors(sensors)
    manager.sort_records()
    manager.compute_global_statistics()
    manager.save_entry_status(
        _select_entry_status(
            child_payloads,
            has_missing_route_records=bool(missing_route_ids),
        )
    )
    manager.write_statistics()

    merged_payload = load_json_if_exists(output_path)
    return {
        "statistics_path": str(output_path),
        "entry_status": (merged_payload or {}).get("entry_status"),
        "eligible": (merged_payload or {}).get("eligible"),
        "reported_records": len(record_by_route_id),
        "planned_records": len(planned_route_ids),
        "missing_route_ids": missing_route_ids,
        "duplicate_route_ids": sorted(set(duplicate_route_ids)),
        "labels": (merged_payload or {}).get("labels", []),
        "values": (merged_payload or {}).get("values", []),
        "global_record": (merged_payload or {}).get("_checkpoint", {}).get("global_record", {}),
        "payload": merged_payload or {},
    }


def _consolidate_text_artifacts(
    *,
    entries: Sequence[Dict[str, Any]],
    key: str,
    output_path: Path,
    header_label: str,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as out:
        for entry in entries:
            artifact_path = entry.get("artifacts", {}).get(key)
            idx = entry.get("index", "?")
            out.write(f"\n{'=' * 70}\n")
            out.write(f"  {header_label} {idx:>3}  {key}\n")
            out.write(f"{'=' * 70}\n")
            if artifact_path and Path(artifact_path).exists():
                try:
                    out.write(Path(artifact_path).read_text(encoding="utf-8", errors="replace"))
                except Exception as exc:
                    out.write(f"[ERROR reading {artifact_path}: {exc}]\n")
            else:
                out.write("[artifact missing]\n")
            out.write("\n")
    return output_path


def _consolidate_shard_videos(
    *,
    entries: Sequence[Dict[str, Any]],
    output_dir: Path,
) -> int:
    import shutil

    output_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for entry in entries:
        entry_index = int(entry.get("index", -1))
        videos_dir = entry.get("artifacts", {}).get("videos_dir")
        if not videos_dir:
            continue
        src_dir = Path(videos_dir)
        if not src_dir.is_dir():
            continue
        for video_file in sorted(src_dir.glob("*.mp4")):
            dst_file = output_dir / video_file.name
            if dst_file.exists() and entry_index >= 0:
                dst_file = output_dir / f"{video_file.stem}_job{entry_index:03d}{video_file.suffix}"
            shutil.copy2(str(video_file), str(dst_file))
            copied += 1
    return copied


def _discover_child_artifacts(
    *,
    statistics_file: Path,
    live_file: Path,
    output_root: Path,
    record_dir: Optional[Path],
) -> Dict[str, Optional[str]]:
    videos_dir = output_root / "videos"
    return {
        "statistics": str(statistics_file) if statistics_file.exists() else None,
        "live_results": str(live_file) if live_file.exists() else None,
        "output_root": str(output_root),
        "videos_dir": str(videos_dir) if videos_dir.is_dir() else None,
        "record_dir": str(record_dir) if record_dir and record_dir.exists() else None,
    }


def _terminate_child_processes(running: List[Dict[str, Any]]) -> None:
    live_entries = [entry for entry in running if entry["proc"].poll() is None]
    if not live_entries:
        return

    for entry in live_entries:
        try:
            entry["proc"].terminate()
        except Exception:
            pass

    deadline = time.time() + 15.0
    while time.time() < deadline:
        if all(entry["proc"].poll() is not None for entry in live_entries):
            break
        time.sleep(0.5)

    for entry in live_entries:
        if entry["proc"].poll() is None:
            try:
                entry["proc"].kill()
            except Exception:
                pass


def _serialize_worker_entry(worker: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "index": worker.get("index"),
        "status": worker.get("status"),
        "host": worker.get("host"),
        "port": worker.get("port"),
        "traffic_manager_port": worker.get("traffic_manager_port"),
        "traffic_manager_seed": worker.get("traffic_manager_seed"),
        "gpu_id": worker.get("gpu_id"),
        "jobs_started": worker.get("jobs_started"),
        "jobs_completed": worker.get("jobs_completed"),
        "jobs_with_retry": worker.get("jobs_with_retry"),
        "current_job_index": worker.get("current_job_index"),
        "started_at": worker.get("started_at"),
        "finished_at": worker.get("finished_at"),
        "paths": dict(worker.get("paths", {})),
    }


def _serialize_attempt_entry(attempt: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "attempt_index": attempt.get("attempt_index"),
        "status": attempt.get("status"),
        "returncode": attempt.get("returncode"),
        "record_found": attempt.get("record_found"),
        "started_at": attempt.get("started_at"),
        "finished_at": attempt.get("finished_at"),
        "paths": dict(attempt.get("paths", {})),
        "artifacts": dict(attempt.get("artifacts", {})),
    }


def _serialize_job_entry(job: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "job_index": job.get("job_index"),
        "route_id": job.get("route_id"),
        "repetition_index": job.get("repetition_index"),
        "record_id": job.get("record_id"),
        "town": job.get("town"),
        "scenario_type": job.get("scenario_type"),
        "status": job.get("status"),
        "assigned_worker": job.get("assigned_worker"),
        "retry_count": job.get("retry_count"),
        "result_source": job.get("result_source"),
        "final_record_found": job.get("final_record_found"),
        "started_at": job.get("started_at"),
        "finished_at": job.get("finished_at"),
        "paths": dict(job.get("paths", {})),
        "final_artifacts": dict(job.get("final_artifacts", {})),
        "attempts": [_serialize_attempt_entry(attempt) for attempt in job.get("attempts", [])],
    }


def _build_manifest(
    *,
    config_path: Path,
    checkpoint_path: Path,
    routes_file: Path,
    output_root: Path,
    requested_shards: int,
    launched_workers: int,
    repetitions: int,
    planned_route_ids: Sequence[str],
    selected_route_count: int,
    workers: Sequence[Dict[str, Any]],
    jobs: Sequence[Dict[str, Any]],
    started_at: str,
) -> Dict[str, Any]:
    return {
        "created_at": now_iso(),
        "started_at": started_at,
        "config_path": str(config_path),
        "checkpoint_path": str(checkpoint_path),
        "routes_file": str(routes_file),
        "output_root": str(output_root),
        "requested_shards": requested_shards,
        "launched_shards": launched_workers,
        "launched_workers": launched_workers,
        "repetitions": repetitions,
        "selected_route_count": selected_route_count,
        "planned_record_count": len(planned_route_ids),
        "planned_route_ids": list(planned_route_ids),
        "workers": [_serialize_worker_entry(worker) for worker in workers],
        "jobs": [_serialize_job_entry(job) for job in jobs],
        "shards": [_serialize_worker_entry(worker) for worker in workers],
    }

def _build_summary(
    *,
    output_root: Path,
    requested_shards: int,
    launched_workers: int,
    started_at: str,
    finished_at: str,
    interrupted: bool,
    worker_entries: Sequence[Dict[str, Any]],
    job_entries: Sequence[Dict[str, Any]],
    merged_statistics: Dict[str, Any],
    statistics_payload: Dict[str, Any],
    statistics_path: Path,
    live_results_path: Path,
    combined_log_path: Path,
    videos_dir: Path,
    manifest_path: Path,
    score_version_default: str,
) -> Dict[str, Any]:
    completed_workers = sum(1 for worker in worker_entries if worker.get("status") == "completed")
    failed_workers = sum(1 for worker in worker_entries if worker.get("status") == "failed")
    interrupted_workers = sum(1 for worker in worker_entries if worker.get("status") == "interrupted")
    job_retry_count = sum(int(job.get("retry_count", 0) or 0) for job in job_entries)
    routes_failed_without_record = sum(1 for job in job_entries if job.get("result_source") == "synthetic_missing")
    job_success_count = sum(1 for job in job_entries if job.get("final_record_found"))
    success_stats = _build_b2d_success_statistics(job_entries)

    if interrupted:
        overall_status = "interrupted"
    elif failed_workers or int(merged_statistics.get("planned_records", 0)) != int(merged_statistics.get("reported_records", 0) + len(merged_statistics.get("missing_route_ids", []) or [])):
        overall_status = "failed"
    else:
        overall_status = "completed"

    metrics = extract_summary_metrics_from_statistics_payload(statistics_payload)
    summary_workers = []
    for worker in worker_entries:
        summary_workers.append({
            "index": worker.get("index"),
            "status": worker.get("status"),
            "host": worker.get("host"),
            "port": worker.get("port"),
            "traffic_manager_port": worker.get("traffic_manager_port"),
            "jobs_started": worker.get("jobs_started"),
            "jobs_completed": worker.get("jobs_completed"),
            "jobs_with_retry": worker.get("jobs_with_retry"),
            "worker_dir": worker.get("paths", {}).get("worker_dir"),
        })
    summary_jobs = []
    for job in job_entries:
        summary_jobs.append({
            "job_index": job.get("job_index"),
            "record_id": job.get("record_id"),
            "route_id": job.get("route_id"),
            "repetition_index": job.get("repetition_index"),
            "town": job.get("town"),
            "scenario_type": job.get("scenario_type"),
            "status": job.get("status"),
            "assigned_worker": job.get("assigned_worker"),
            "retry_count": job.get("retry_count"),
            "result_source": job.get("result_source"),
            "reported_record_count": 1 if job.get("final_record") else 0,
            "statistics": job.get("final_artifacts", {}).get("statistics"),
            "live_results": job.get("final_artifacts", {}).get("live_results"),
            "log_file": job.get("final_artifacts", {}).get("log_file"),
            "job_dir": job.get("paths", {}).get("job_dir"),
        })

    return {
        "status": overall_status,
        "started_at": started_at,
        "finished_at": finished_at,
        "output_root": str(output_root),
        "requested_shards": requested_shards,
        "launched_shards": launched_workers,
        "launched_workers": launched_workers,
        "completed_shards": completed_workers,
        "failed_shards": failed_workers,
        "interrupted_shards": interrupted_workers,
        "completed_workers": completed_workers,
        "failed_workers": failed_workers,
        "interrupted_workers": interrupted_workers,
        "job_count": len(job_entries),
        "job_success_count": job_success_count,
        "job_retry_count": job_retry_count,
        "routes_failed_without_record": routes_failed_without_record,
        "planned_record_count": int(merged_statistics.get("planned_records", 0)),
        "reported_record_count": int(merged_statistics.get("reported_records", 0)),
        "missing_record_count": len(merged_statistics.get("missing_route_ids", []) or []),
        "duplicate_record_count": len(merged_statistics.get("duplicate_route_ids", []) or []),
        "entry_status": metrics.get("entry_status"),
        "eligible": metrics.get("eligible"),
        "success_rule": success_stats.get("success_rule"),
        "success_count": success_stats.get("success_count"),
        "non_success_count": success_stats.get("non_success_count"),
        "success_rate": success_stats.get("success_rate"),
        "scenario_success_summary": success_stats.get("scenario_success_summary"),
        "score_mean": metrics.get("score_mean"),
        "route_completion_mean": metrics.get("route_completion_mean"),
        "score_penalty_mean": metrics.get("score_penalty_mean"),
        "collisions_pedestrian": metrics.get("collisions_pedestrian"),
        "collisions_vehicle": metrics.get("collisions_vehicle"),
        "min_speed_infractions": metrics.get("min_speed_infractions"),
        "global_record": metrics.get("global_record"),
        "score_version_default": score_version_default,
        "statistics": str(statistics_path),
        "live_results": str(live_results_path),
        "combined_log": str(combined_log_path),
        "videos_dir": str(videos_dir),
        "manifest": str(manifest_path),
        "workers": summary_workers,
        "jobs": summary_jobs,
        "shards": summary_workers,
    }


def _build_score_version_node(
    *,
    summary: Dict[str, Any],
    statistics_path: Path,
    summary_path: Path,
) -> Dict[str, Any]:
    return {
        "status": "ok",
        "score_mean": summary.get("score_mean"),
        "route_completion_mean": summary.get("route_completion_mean"),
        "score_penalty_mean": summary.get("score_penalty_mean"),
        "success_rate": summary.get("success_rate"),
        "success_count": summary.get("success_count"),
        "planned_route_count": summary.get("planned_record_count"),
        "collisions_pedestrian": summary.get("collisions_pedestrian"),
        "collisions_vehicle": summary.get("collisions_vehicle"),
        "min_speed_infractions": summary.get("min_speed_infractions"),
        "statistics": str(statistics_path),
        "summary": str(summary_path),
    }


def _build_attempt_paths(
    *,
    output_root: Path,
    record_root: Optional[Path],
    worker_index: int,
    job: Dict[str, Any],
    attempt_index: int,
) -> Dict[str, str]:
    worker_dir = output_root / "workers" / f"worker_{worker_index:02d}"
    job_dir = worker_dir / _build_job_dir_name(job)
    attempt_dir = job_dir / f"attempt_{attempt_index:02d}"
    attempt_output_root = attempt_dir / "output"
    attempt_record_dir = (record_root / f"worker_{worker_index:02d}" / _build_job_dir_name(job) / f"attempt_{attempt_index:02d}") if record_root else None
    return {
        "worker_dir": str(worker_dir),
        "job_dir": str(job_dir),
        "attempt_dir": str(attempt_dir),
        "routes_file": str(attempt_dir / "routes.xml"),
        "log_file": str(attempt_dir / "eval.log"),
        "statistics_file": str(attempt_dir / "statistics.json"),
        "live_results_file": str(attempt_dir / "live_results.txt"),
        "output_root": str(attempt_output_root),
        "record_dir": str(attempt_record_dir) if attempt_record_dir else "",
    }


def _build_child_runner_cmd(
    *,
    args: argparse.Namespace,
    config_path: Path,
    checkpoint_path: Path,
    output_root: Path,
    agent_path: Path,
    worker: Dict[str, Any],
    attempt_paths: Dict[str, str],
) -> List[str]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child-runner",
        "--config", str(config_path),
        "--checkpoint", str(checkpoint_path),
        "--routes", str(args.routes),
        "--output-root", str(output_root),
        "--leaderboard-root", str(args.leaderboard_root),
        "--agent", str(agent_path),
        "--repetitions", "1",
        "--track", str(args.track),
        "--timeout", str(args.timeout),
        "--debug", str(args.debug),
        "--stochastic", "true" if args.stochastic else "false",
        "--child-routes-file", attempt_paths["routes_file"],
        "--child-statistics-file", attempt_paths["statistics_file"],
        "--child-live-file", attempt_paths["live_results_file"],
        "--child-output-root", attempt_paths["output_root"],
        "--child-host", str(worker["host"]),
        "--child-port", str(worker["port"]),
        "--child-tm-port", str(worker["traffic_manager_port"]),
        "--child-tm-seed", str(worker["traffic_manager_seed"]),
        "--child-gpu-id", str(worker["gpu_id"]),
    ]
    if attempt_paths.get("record_dir"):
        cmd.extend(["--child-record-dir", attempt_paths["record_dir"]])
    return cmd


def evaluate_parallel(args: argparse.Namespace) -> int:
    config_path = resolve_path(args.config)
    checkpoint_path = resolve_path(args.checkpoint)
    routes_file = resolve_path(args.routes)
    output_root = resolve_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    record_root = resolve_path(args.record_dir) if args.record_dir else None

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    parallel_env_config = _prepare_parallel_env_config(config=config, args=args)
    env_cfg = parallel_env_config.get("env", parallel_env_config)
    requested_shards = int(args.num_workers or 1)
    if requested_shards <= 0:
        raise ValueError(f"Parallel shard count must be positive, got {requested_shards}")

    template_root, selected_routes = load_selected_route_elements(
        routes_file,
    )
    if not selected_routes:
        raise ValueError("No routes selected for leaderboard parallel evaluation")

    route_episodes = materialize_episode_routes(
        selected_routes,
        repetitions=args.repetitions,
        n_eval_episodes=0,
    )
    actual_workers = min(requested_shards, len(route_episodes))
    if actual_workers < requested_shards:
        print(
            f"[leaderboard-parallel] requested {requested_shards} workers but only "
            f"{len(route_episodes)} jobs are available; launching {actual_workers} worker(s).",
            flush=True,
        )

    slots = normalize_carla_slots(env_cfg, actual_workers)
    validate_parallel_slots(slots)
    ordered_episodes = _build_balanced_episode_order(route_episodes, actual_workers)
    job_entries = _build_route_jobs(ordered_episodes)
    planned_route_ids = [job["record_id"] for job in job_entries]
    workers_dir = output_root / "workers"
    workers_dir.mkdir(parents=True, exist_ok=True)

    agent_path = resolve_path(args.agent)

    worker_entries: List[Dict[str, Any]] = []
    for idx, slot in enumerate(slots):
        worker_dir = workers_dir / f"worker_{idx:02d}"
        worker_entries.append({
            "index": idx,
            "status": "prepared",
            "host": slot.host,
            "port": slot.port,
            "traffic_manager_port": slot.traffic_manager_port,
            "traffic_manager_seed": slot.traffic_manager_seed,
            "gpu_id": slot.gpu_id,
            "jobs_started": 0,
            "jobs_completed": 0,
            "jobs_with_retry": 0,
            "current_job_index": None,
            "retry_job": None,
            "started_at": None,
            "finished_at": None,
            "paths": {
                "worker_dir": str(worker_dir),
            },
        })

    started_at = now_iso()
    manifest_path = output_root / "manifest.json"
    manifest_args = dict(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        routes_file=routes_file,
        output_root=output_root,
        requested_shards=requested_shards,
        launched_workers=actual_workers,
        repetitions=args.repetitions,
        planned_route_ids=planned_route_ids,
        selected_route_count=len(selected_routes),
        workers=worker_entries,
        jobs=job_entries,
        started_at=started_at,
    )
    write_json_file(manifest_path, _build_manifest(**manifest_args))

    hosts = [entry["host"] for entry in worker_entries]
    print("=" * 70, flush=True)
    print("  Leaderboard Parallel Evaluation", flush=True)
    print("=" * 70, flush=True)
    print(f"Config:             {config_path}", flush=True)
    print(f"Checkpoint:         {checkpoint_path}", flush=True)
    print(f"Routes file:        {routes_file}", flush=True)
    print(f"Output root:        {output_root}", flush=True)
    print(f"Workers launched:   {actual_workers}/{requested_shards}", flush=True)
    print(f"Job count:          {len(job_entries)}", flush=True)
    print(f"Route count:        {len(selected_routes)}", flush=True)
    print(f"Repetitions:        {args.repetitions}", flush=True)
    print(f"Stochastic:         {args.stochastic}", flush=True)

    interrupted = False
    fatal_error = False
    pending_jobs: collections.deque[Dict[str, Any]] = collections.deque(job_entries)
    all_attempt_entries: List[Dict[str, Any]] = []
    running: List[Dict[str, Any]] = []

    def _signal_handler(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}")

    previous_sigint = signal.getsignal(signal.SIGINT)
    previous_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    try:
        print("\n[leaderboard-parallel] pre-cleaning fake bind hosts ...", flush=True)
        kill_hosts(hosts, verbose=True)
        for host in hosts:
            _wait_for_host_carla_exit(host)

        while pending_jobs or running or any(worker.get("retry_job") is not None for worker in worker_entries):
            busy_worker_indexes = {entry["worker"]["index"] for entry in running}

            for worker in worker_entries:
                if worker["index"] in busy_worker_indexes:
                    continue

                job = worker.get("retry_job")
                if job is not None:
                    worker["retry_job"] = None
                elif pending_jobs:
                    job = pending_jobs.popleft()
                else:
                    continue

                attempt_index = len(job["attempts"])
                attempt_paths = _build_attempt_paths(
                    output_root=output_root,
                    record_root=record_root,
                    worker_index=int(worker["index"]),
                    job=job,
                    attempt_index=attempt_index,
                )
                attempt_dir = Path(attempt_paths["attempt_dir"])
                attempt_dir.mkdir(parents=True, exist_ok=True)
                write_route_shard(template_root, [job["episode"]], Path(attempt_paths["routes_file"]))

                job["assigned_worker"] = worker["index"]
                if job["started_at"] is None:
                    job["started_at"] = now_iso()
                    worker["jobs_started"] += 1
                job["paths"] = {
                    "job_dir": attempt_paths["job_dir"],
                }

                attempt_entry = {
                    "index": len(all_attempt_entries),
                    "attempt_index": attempt_index,
                    "status": "running",
                    "returncode": None,
                    "record_found": False,
                    "started_at": now_iso(),
                    "finished_at": None,
                    "paths": dict(attempt_paths),
                    "artifacts": {},
                }
                job["attempts"].append(attempt_entry)
                all_attempt_entries.append(attempt_entry)

                cmd = _build_child_runner_cmd(
                    args=args,
                    config_path=config_path,
                    checkpoint_path=checkpoint_path,
                    output_root=output_root,
                    agent_path=agent_path,
                    worker=worker,
                    attempt_paths=attempt_paths,
                )
                log_handle = open(attempt_paths["log_file"], "w", encoding="utf-8")
                proc = subprocess.Popen(
                    cmd,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    cwd=str(_PROJECT_ROOT),
                    env=os.environ.copy(),
                )
                worker["status"] = "running"
                worker["current_job_index"] = job["job_index"]
                if worker["started_at"] is None:
                    worker["started_at"] = now_iso()
                job["status"] = "running"
                running.append({
                    "proc": proc,
                    "log_handle": log_handle,
                    "worker": worker,
                    "job": job,
                    "attempt": attempt_entry,
                })
                print(
                    f"[leaderboard-parallel] worker {worker['index']:02d} started "
                    f"job {job['job_index']:03d} route={job['route_id']} rep={job['repetition_index']} "
                    f"(attempt={attempt_index}, pid={proc.pid}, host={worker['host']}, port={worker['port']})",
                    flush=True,
                )

            write_json_file(manifest_path, _build_manifest(**manifest_args))

            progress_made = False
            for entry in list(running):
                proc = entry["proc"]
                returncode = proc.poll()
                if returncode is None:
                    continue

                progress_made = True
                running.remove(entry)
                entry["log_handle"].close()

                worker = entry["worker"]
                job = entry["job"]
                attempt = entry["attempt"]
                attempt["returncode"] = returncode
                attempt["finished_at"] = now_iso()
                attempt["artifacts"] = _discover_child_artifacts(
                    statistics_file=Path(attempt["paths"]["statistics_file"]),
                    live_file=Path(attempt["paths"]["live_results_file"]),
                    output_root=Path(attempt["paths"]["output_root"]),
                    record_dir=Path(attempt["paths"]["record_dir"]) if attempt["paths"]["record_dir"] else None,
                )
                attempt["artifacts"]["log_file"] = attempt["paths"]["log_file"]
                _wait_for_host_carla_exit(str(worker["host"]))

                payload, record = _extract_attempt_record(
                    statistics_path=attempt["artifacts"].get("statistics"),
                    expected_route_id=str(job["route_id"]),
                    expected_record_id=str(job["record_id"]),
                )
                attempt["record_found"] = record is not None
                if record is None:
                    print(
                        f"[leaderboard-parallel][debug] worker {worker['index']:02d} "
                        f"job {job['job_index']:03d} missing record: "
                        f"rc={_describe_returncode(returncode)}, "
                        f"{_summarize_statistics_payload(payload)}, "
                        f"stats={attempt['artifacts'].get('statistics')}, "
                        f"live={attempt['artifacts'].get('live_results')}, "
                        f"log={attempt['paths']['log_file']}",
                        flush=True,
                    )

                if record is not None:
                    attempt["status"] = "completed"
                    job["final_payload"] = payload
                    job["final_record"] = record
                    job["final_record_found"] = True
                    job["final_artifacts"] = dict(attempt["artifacts"])
                    job["status"] = "completed"
                    job["result_source"] = "attempt"
                    job["finished_at"] = now_iso()
                    worker["jobs_completed"] += 1
                    if job.get("retry_count", 0):
                        worker["jobs_with_retry"] += 1
                    worker["status"] = "idle"
                    worker["current_job_index"] = None
                    print(
                        f"[leaderboard-parallel] worker {worker['index']:02d} completed "
                        f"job {job['job_index']:03d} (rc={returncode}, retries={job['retry_count']})",
                        flush=True,
                    )
                elif int(job.get("retry_count", 0)) < _DEFAULT_ROUTE_RETRY_ON_EMPTY_RECORD:
                    job["retry_count"] = int(job.get("retry_count", 0)) + 1
                    attempt["status"] = "missing_record"
                    job["status"] = "retry_pending"
                    worker["status"] = "idle"
                    worker["current_job_index"] = None
                    worker["retry_job"] = job
                    print(
                        f"[leaderboard-parallel] worker {worker['index']:02d} retrying "
                        f"job {job['job_index']:03d} route={job['route_id']} "
                        f"(attempt={attempt['attempt_index']}, rc={returncode})",
                        flush=True,
                    )
                else:
                    attempt["status"] = "missing_record"
                    job["status"] = "synthetic_missing"
                    job["result_source"] = "synthetic_missing"
                    job["final_artifacts"] = dict(attempt["artifacts"])
                    job["finished_at"] = now_iso()
                    worker["jobs_completed"] += 1
                    if job.get("retry_count", 0):
                        worker["jobs_with_retry"] += 1
                    worker["status"] = "idle"
                    worker["current_job_index"] = None
                    print(
                        f"[leaderboard-parallel] worker {worker['index']:02d} exhausted retries "
                        f"for job {job['job_index']:03d} route={job['route_id']}; "
                        f"will synthesize zero-score record during merge",
                        flush=True,
                    )

                worker["finished_at"] = now_iso()
                write_json_file(manifest_path, _build_manifest(**manifest_args))

            if running and not progress_made:
                time.sleep(0.5)

    except KeyboardInterrupt:
        interrupted = True
        print("\n[leaderboard-parallel] interrupt received, stopping workers ...", flush=True)
        _terminate_child_processes(running)
        for entry in list(running):
            try:
                entry["log_handle"].close()
            except Exception:
                pass
            worker = entry["worker"]
            job = entry["job"]
            attempt = entry["attempt"]
            attempt["status"] = "interrupted"
            attempt["returncode"] = entry["proc"].poll()
            attempt["finished_at"] = now_iso()
            attempt["artifacts"] = _discover_child_artifacts(
                statistics_file=Path(attempt["paths"]["statistics_file"]),
                live_file=Path(attempt["paths"]["live_results_file"]),
                output_root=Path(attempt["paths"]["output_root"]),
                record_dir=Path(attempt["paths"]["record_dir"]) if attempt["paths"]["record_dir"] else None,
            )
            attempt["artifacts"]["log_file"] = attempt["paths"]["log_file"]
            worker["status"] = "interrupted"
            worker["current_job_index"] = None
            job["status"] = "interrupted"
            _wait_for_host_carla_exit(str(worker["host"]))
        running.clear()
    except Exception:
        fatal_error = True
        print("\n[leaderboard-parallel] fatal error, stopping workers ...", flush=True)
        _terminate_child_processes(running)
        for entry in list(running):
            try:
                entry["log_handle"].close()
            except Exception:
                pass
            worker = entry["worker"]
            job = entry["job"]
            attempt = entry["attempt"]
            attempt["status"] = "failed"
            attempt["returncode"] = entry["proc"].poll()
            attempt["finished_at"] = now_iso()
            attempt["artifacts"] = _discover_child_artifacts(
                statistics_file=Path(attempt["paths"]["statistics_file"]),
                live_file=Path(attempt["paths"]["live_results_file"]),
                output_root=Path(attempt["paths"]["output_root"]),
                record_dir=Path(attempt["paths"]["record_dir"]) if attempt["paths"]["record_dir"] else None,
            )
            attempt["artifacts"]["log_file"] = attempt["paths"]["log_file"]
            worker["status"] = "failed"
            worker["current_job_index"] = None
            job["status"] = "failed"
            _wait_for_host_carla_exit(str(worker["host"]))
        running.clear()
        raise
    finally:
        try:
            for entry in running:
                try:
                    entry["log_handle"].close()
                except Exception:
                    pass
        finally:
            signal.signal(signal.SIGINT, previous_sigint)
            signal.signal(signal.SIGTERM, previous_sigterm)

        print("\n[leaderboard-parallel] final host sweep ...", flush=True)
        kill_hosts(hosts, verbose=True)
        for host in hosts:
            _wait_for_host_carla_exit(host)
        for worker in worker_entries:
            if worker["status"] in ("prepared", "idle", "running"):
                worker["status"] = "failed" if fatal_error else ("interrupted" if interrupted else "completed")
                worker["finished_at"] = now_iso()
        write_json_file(manifest_path, _build_manifest(**manifest_args))

    statistics_path = output_root / "statistics.json"
    statistics_v20_path = output_root / "statistics_v20.json"
    live_results_path = output_root / "live_results.txt"
    combined_log_path = output_root / "combined_eval.log"
    videos_dir = output_root / "videos"

    merged_statistics = _merge_statistics(
        job_entries=job_entries,
        planned_route_ids=planned_route_ids,
        output_path=statistics_path,
        live_output_path=live_results_path,
    )
    _consolidate_text_artifacts(
        entries=all_attempt_entries,
        key="live_results",
        output_path=live_results_path,
        header_label="attempt",
    )
    _consolidate_text_artifacts(
        entries=all_attempt_entries,
        key="log_file",
        output_path=combined_log_path,
        header_label="attempt",
    )
    video_count = _consolidate_shard_videos(
        entries=[
            {
                "index": job["job_index"],
                "artifacts": job.get("final_artifacts", {}),
            }
            for job in job_entries
        ],
        output_dir=videos_dir,
    )

    summary_path = output_root / "summary.json"
    summary_v20_path = output_root / "summary_v20.json"
    summary = _build_summary(
        output_root=output_root,
        requested_shards=requested_shards,
        launched_workers=actual_workers,
        started_at=started_at,
        finished_at=now_iso(),
        interrupted=interrupted,
        worker_entries=worker_entries,
        job_entries=job_entries,
        merged_statistics=merged_statistics,
        statistics_payload=merged_statistics.get("payload", {}) or {},
        statistics_path=statistics_path,
        live_results_path=live_results_path,
        combined_log_path=combined_log_path,
        videos_dir=videos_dir,
        manifest_path=manifest_path,
        score_version_default="v2.1",
    )
    summary["video_count"] = video_count
    summary["score_versions"] = {
        "v2.1": _build_score_version_node(
            summary=summary,
            statistics_path=statistics_path,
            summary_path=summary_path,
        ),
        "v2.0": {
            "status": "error",
            "success_rate": summary.get("success_rate"),
            "success_count": summary.get("success_count"),
            "planned_route_count": summary.get("planned_record_count"),
            "statistics": str(statistics_v20_path),
            "summary": str(summary_v20_path),
            "error": None,
        },
    }

    summary_v20: Optional[Dict[str, Any]] = None
    v20_error: Optional[str] = None
    try:
        statistics_v20_payload = build_v20_statistics_from_payload(merged_statistics.get("payload", {}) or {})
        write_json_file(statistics_v20_path, statistics_v20_payload)
        summary_v20 = _build_summary(
            output_root=output_root,
            requested_shards=requested_shards,
            launched_workers=actual_workers,
            started_at=started_at,
            finished_at=summary["finished_at"],
            interrupted=interrupted,
            worker_entries=worker_entries,
            job_entries=job_entries,
            merged_statistics=merged_statistics,
            statistics_payload=statistics_v20_payload,
            statistics_path=statistics_v20_path,
            live_results_path=live_results_path,
            combined_log_path=combined_log_path,
            videos_dir=videos_dir,
            manifest_path=manifest_path,
            score_version_default="v2.0",
        )
        summary_v20["video_count"] = video_count
        write_json_file(summary_v20_path, summary_v20)
        summary["score_versions"]["v2.0"] = _build_score_version_node(
            summary=summary_v20,
            statistics_path=statistics_v20_path,
            summary_path=summary_v20_path,
        )
    except (ScoreVersionError, KeyError, TypeError, ValueError) as exc:
        v20_error = str(exc)
        summary["score_versions"]["v2.0"]["error"] = v20_error

    write_json_file(summary_path, summary)

    print("\n" + "=" * 70, flush=True)
    print("  Leaderboard Parallel Evaluation Summary", flush=True)
    print("=" * 70, flush=True)
    print(f"status:              {summary['status']}", flush=True)
    print(f"output:              {output_root}", flush=True)
    print(
        f"workers:             {summary['completed_workers']} completed / "
        f"{summary['failed_workers']} failed / {summary['interrupted_workers']} interrupted",
        flush=True,
    )
    print(
        f"jobs:                {summary['job_success_count']} with records / "
        f"{summary['job_count']} total",
        flush=True,
    )
    print(
        f"missing child records:{summary['routes_failed_without_record']}",
        flush=True,
    )
    print(
        f"records:             {summary['reported_record_count']} actual / "
        f"{summary['planned_record_count']} planned",
        flush=True,
    )
    print(f"entry_status:        {summary['entry_status']}", flush=True)
    print(
        f"success rate:        {summary['success_rate'] * 100.0:.1f}% "
        f"({summary['success_count']}/{summary['planned_record_count']})",
        flush=True,
    )
    if summary.get("score_mean") is not None:
        print(f"score v2.1:          {summary['score_mean']:.3f}", flush=True)
    if summary_v20 and summary_v20.get("score_mean") is not None:
        print(f"score v2.0:          {summary_v20['score_mean']:.3f}", flush=True)
    else:
        print("score v2.0:          unavailable", flush=True)
    if summary.get("route_completion_mean") is not None:
        print(f"route completion:    {summary['route_completion_mean']:.3f}", flush=True)
    if summary.get("collisions_pedestrian") is not None:
        print(f"Ped:                 {summary['collisions_pedestrian']:.3f}", flush=True)
    if summary.get("collisions_vehicle") is not None:
        print(f"Veh:                 {summary['collisions_vehicle']:.3f}", flush=True)
    if summary.get("min_speed_infractions") is not None:
        print(f"MS:                  {summary['min_speed_infractions']:.3f}", flush=True)
    print(f"statistics:          {statistics_path}", flush=True)
    if summary_v20 is not None:
        print(f"statistics v2.0:     {statistics_v20_path}", flush=True)
    elif v20_error:
        print(f"statistics v2.0:     unavailable ({v20_error})", flush=True)
    print(f"live results:        {live_results_path}", flush=True)
    print(f"combined log:        {combined_log_path}", flush=True)
    print(f"videos:              {videos_dir} ({video_count} files)", flush=True)
    print(f"manifest:            {manifest_path}", flush=True)
    print(f"summary:             {summary_path}", flush=True)
    if summary_v20 is not None:
        print(f"summary v2.0:        {summary_v20_path}", flush=True)
    print("=" * 70, flush=True)

    if interrupted:
        return 130
    return 1 if summary["status"] == "failed" else 0


def main() -> None:
    _enable_line_buffering()

    parser = argparse.ArgumentParser(description="Parallel leaderboard evaluation for RL agents")
    parser.add_argument("--config", type=str, required=True, help="Path to the RL config YAML")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint directory")
    parser.add_argument("--routes", type=str, required=True, help="Path to routes XML")
    parser.add_argument("--output-root", type=str, required=True, help="Run output directory")
    parser.add_argument(
        "--leaderboard-root",
        type=str,
        required=True,
        help="Path to vendor/carla/evaluation-runtime/leaderboard",
    )
    parser.add_argument(
        "--agent",
        type=str,
        default=str(_THIS_DIR / "agent.py"),
        help="Path to leaderboard agent entry script",
    )
    parser.add_argument("--repetitions", type=int, default=1, help="Leaderboard repetitions")
    parser.add_argument("--num-workers", type=int, default=1, help="Number of parallel leaderboard workers")
    parser.add_argument("--track", type=str, default="SENSORS", help="Leaderboard track")
    parser.add_argument("--timeout", type=int, default=300, help="Leaderboard evaluator timeout")
    parser.add_argument("--debug", type=int, default=0, help="Leaderboard debug level")
    parser.add_argument("--record-dir", type=str, default="", help="Optional top-level recorder output root")
    parser.add_argument(
        "--stochastic",
        type=str,
        default="false",
        help="Whether to enable stochastic policy execution",
    )
    parser.add_argument("--child-runner", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--child-routes-file", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument("--child-statistics-file", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument("--child-live-file", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument("--child-output-root", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument("--child-record-dir", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument("--child-host", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument("--child-port", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--child-tm-port", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--child-tm-seed", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--child-gpu-id", type=str, default="0", help=argparse.SUPPRESS)

    args = parser.parse_args()
    args.stochastic = _parse_bool(args.stochastic)

    if args.child_runner:
        raise SystemExit(_run_shard_worker(args))

    raise SystemExit(evaluate_parallel(args))


if __name__ == "__main__":
    main()
