"""Pure helpers for ``CARLAEnvPool``.

Crash-file format, per-worker IPC keys, and other side-effect-free
utilities used by the worker pool. Kept pure so they can be reused by
tests and introspection tools.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

__layer__ = (3, "Simulation")

EPISODE_CONTEXT_KEYS = ("route_id", "scenario_name", "scenario_instance_name", "town")
EPISODE_CONTEXT_ATTR_ALIASES = {
    "route_id": ("route_id", "_current_route_id"),
    "scenario_name": ("scenario_name",),
    "scenario_instance_name": ("scenario_instance_name",),
    "town": ("town", "_current_town"),
}


def parse_list_config(
    override: Optional[List],
    config: Dict,
    key: str,
    num_envs: int,
    default_value: Any,
    auto_increment: Optional[int] = None,
    fallback_base: Optional[List] = None,
    fallback_offset: Optional[int] = None,
    fail_on_short_list: bool = False,
) -> List:
    if override is not None:
        result = list(override[:num_envs])
        if len(result) < num_envs:
            if fail_on_short_list:
                raise ValueError(
                    f"Connection config '{key}' expects at least {num_envs} values, "
                    f"but override only provided {len(result)}."
                )
            fill_value = default_value if default_value is not None else 0
            result.extend([fill_value] * (num_envs - len(result)))
        return result

    if key in config:
        value = config[key]
        if isinstance(value, list):
            result = list(value[:num_envs])
            if len(result) < num_envs:
                if fail_on_short_list:
                    raise ValueError(
                        f"carla.{key} expects at least {num_envs} values, "
                        f"but only {len(result)} were configured."
                    )
                fill_value = result[-1] if result else (default_value if default_value is not None else 0)
                result.extend([fill_value] * (num_envs - len(result)))
            return result
        if auto_increment is not None:
            return [value + i * auto_increment for i in range(num_envs)]
        return [value] * num_envs

    if fallback_base is not None and fallback_offset is not None:
        return [base + fallback_offset for base in fallback_base]

    if default_value is not None:
        if auto_increment is not None:
            return [default_value + i * auto_increment for i in range(num_envs)]
        return [default_value] * num_envs

    return [0] * num_envs


def parse_connection_config(
    config: Dict,
    num_envs: int,
    carla_ports_override: Optional[List[int]],
    tm_ports_override: Optional[List[int]],
    gpu_ids_override: Optional[List[int]],
    hosts_override: Optional[List[str]] = None,
) -> Dict[str, List[Any]]:
    carla_config = config.get("carla", {})
    hosts = parse_list_config(
        override=hosts_override,
        config=carla_config,
        key="host",
        num_envs=num_envs,
        default_value="localhost",
        fail_on_short_list=True,
    )
    carla_ports = parse_list_config(
        override=carla_ports_override,
        config=carla_config,
        key="port",
        num_envs=num_envs,
        default_value=2000,
        auto_increment=10,
        fail_on_short_list=True,
    )
    tm_ports = parse_list_config(
        override=tm_ports_override,
        config=carla_config,
        key="traffic_manager_port",
        num_envs=num_envs,
        default_value=None,
        fallback_base=carla_ports,
        fallback_offset=6000,
        fail_on_short_list=True,
    )
    tm_seeds = parse_list_config(
        override=None,
        config=carla_config,
        key="traffic_manager_seed",
        num_envs=num_envs,
        default_value=0,
        fail_on_short_list=True,
    )
    gpu_ids = parse_list_config(
        override=gpu_ids_override,
        config=carla_config,
        key="gpu_id",
        num_envs=num_envs,
        default_value=0,
        fail_on_short_list=True,
    )
    return {
        "hosts": hosts,
        "ports": carla_ports,
        "tm_ports": tm_ports,
        "tm_seeds": tm_seeds,
        "gpu_ids": gpu_ids,
    }


def is_current_epoch(expected_epoch: int, msg: Dict[str, Any]) -> bool:
    return msg.get("worker_epoch", 0) == expected_epoch


def is_reset_obs(info: Optional[Dict[str, Any]]) -> bool:
    return bool(info and (info.get("from_reset") or info.get("episode_start")))


def extract_episode_context(source: Optional[Any], preserve_empty: bool = False) -> Dict[str, str]:
    if source is None:
        return {}

    if isinstance(source, dict):
        def get_value(key: str) -> Tuple[bool, Optional[Any]]:
            return key in source, source.get(key)
    else:
        get_context = getattr(source, "get_episode_context", None)
        if callable(get_context):
            try:
                direct_context = get_context()
            except Exception:
                direct_context = None
            if isinstance(direct_context, dict):
                normalized_context = extract_episode_context(
                    direct_context,
                    preserve_empty=preserve_empty,
                )
                if normalized_context:
                    return normalized_context

        for attr_name in ("unwrapped", "env"):
            nested = getattr(source, attr_name, None)
            if nested is not None and nested is not source:
                nested_context = extract_episode_context(nested, preserve_empty=preserve_empty)
                if nested_context:
                    return nested_context

        def get_value(key: str) -> Tuple[bool, Optional[Any]]:
            for attr_name in EPISODE_CONTEXT_ATTR_ALIASES.get(key, (key,)):
                if hasattr(source, attr_name):
                    return True, getattr(source, attr_name, None)
            return False, None

    context: Dict[str, str] = {}
    for key in EPISODE_CONTEXT_KEYS:
        exists, value = get_value(key)
        if not exists or value is None:
            continue
        text = str(value).strip()
        if text or preserve_empty:
            context[key] = text
    return context


def format_episode_context(source: Optional[Any]) -> str:
    context = extract_episode_context(source)
    if not context:
        return ""

    route_id = context.get("route_id")
    scenario_name = context.get("scenario_name")
    scenario_instance_name = context.get("scenario_instance_name")
    town = context.get("town")

    scenario_display = scenario_name or scenario_instance_name or ""
    if scenario_name and scenario_instance_name and scenario_instance_name != scenario_name:
        scenario_display = f"{scenario_display} ({scenario_instance_name})"

    parts = []
    if route_id:
        parts.append(f"route={route_id}")
    if scenario_display:
        parts.append(f"scenario={scenario_display}")
    if town:
        parts.append(f"town={town}")
    return " | ".join(parts)


def build_worker_crash_file_path(
    result_dir: Union[str, Path],
    worker_id: int,
    worker_epoch: int,
) -> Path:
    base_dir = Path(result_dir).expanduser()
    if not base_dir.is_absolute():
        base_dir = (Path.cwd() / base_dir).resolve()
    return base_dir / "crash_events" / f"worker_{worker_id}_epoch_{worker_epoch}.json"


def write_worker_crash_file(
    result_dir: Union[str, Path],
    payload: Dict[str, Any],
) -> Path:
    worker_id = int(payload["worker_id"])
    worker_epoch = int(payload["worker_epoch"])
    crash_path = build_worker_crash_file_path(result_dir, worker_id, worker_epoch)
    crash_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = crash_path.with_suffix(f".tmp.{os.getpid()}")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp_path, crash_path)
    return crash_path


def read_worker_crash_file(
    result_dir: Union[str, Path],
    worker_id: int,
    worker_epoch: int,
    *,
    delete_after_read: bool = True,
) -> Optional[Dict[str, Any]]:
    crash_path = build_worker_crash_file_path(result_dir, worker_id, worker_epoch)
    if not crash_path.exists():
        return None
    with open(crash_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if delete_after_read:
        try:
            crash_path.unlink()
        except FileNotFoundError:
            pass
    return payload


def update_worker_episode_context(worker_info: Any, source: Optional[Any]) -> Dict[str, str]:
    context = extract_episode_context(source, preserve_empty=True)
    clear_missing = isinstance(source, dict) and not source
    for key in EPISODE_CONTEXT_KEYS:
        if key in context:
            setattr(worker_info, key, context[key])
        elif clear_missing:
            setattr(worker_info, key, "")
    return context


def build_crash_step_result(
    worker_id: int,
    crash_reason: str,
    crash_type: str,
    error_msg: str,
    crash_detail: Optional[str] = None,
    traceback_str: Optional[str] = None,
    worker_epoch: Optional[int] = None,
    episode_step: Optional[int] = None,
    episode_reward: Optional[float] = None,
    worker_info: Optional[Any] = None,
    context_source: Optional[Any] = None,
) -> Dict[str, Any]:
    info = {
        "crashed": True,
        "crash_reason": crash_reason,
        "crash_type": crash_type,
        "crash_detail": crash_detail or error_msg,
        "error": error_msg,
    }
    if traceback_str:
        info["traceback"] = traceback_str
    if episode_step is None:
        episode_step = getattr(worker_info, "current_episode_step", 0)
    if episode_reward is None:
        episode_reward = getattr(worker_info, "current_episode_reward", 0.0)
    if worker_epoch is None:
        worker_epoch = getattr(worker_info, "epoch", 0)
    if worker_info is not None:
        info.update(extract_episode_context(worker_info, preserve_empty=True))
    if context_source is not None:
        info.update(extract_episode_context(context_source, preserve_empty=True))
    return {
        "type": "step_result",
        "worker_id": worker_id,
        "worker_epoch": worker_epoch,
        "observation": None,
        "reward": 0.0,
        "terminated": True,
        "truncated": False,
        "info": info,
        "episode_step": episode_step,
        "episode_reward": episode_reward,
        "crashed": True,
        "crash_reason": crash_reason,
        "crash_type": crash_type,
    }


def enrich_step_info(result: Dict[str, Any]) -> Dict[str, Any]:
    info = dict(result.get("info") or {})
    info.update(extract_episode_context(result))
    info["episode_reward"] = result.get("episode_reward", 0.0)
    info["episode_length"] = result.get("episode_step", 0)
    info["total_reward"] = result.get("episode_reward", 0.0)
    info["total_steps"] = result.get("episode_step", 0)
    if result.get("crashed", False):
        info["crashed"] = True
        info["crash_reason"] = result.get("crash_reason", "process_died")
        info["crash_type"] = result.get("crash_type", "unknown")
        info["crash_detail"] = result.get("info", {}).get("crash_detail") or info.get("error", "")
        info["episode_ended"] = True
        info["end_reason"] = "crashed"
    elif result.get("terminated", False):
        info["episode_ended"] = True
        info["end_reason"] = "terminated"
    elif result.get("truncated", False):
        info["episode_ended"] = True
        info["truncated_mark"] = True
        info["end_reason"] = "truncated"
    return info
