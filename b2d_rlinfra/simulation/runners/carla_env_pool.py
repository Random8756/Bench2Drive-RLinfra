"""CARLA async environment pool.

* AsyncVecEnv worker pool: one process per CARLA server, dict-based async
  ``step()/reset()`` keyed by ``worker_id``.
* Min-ready async stepping: ``step(actions, min_ready=k)`` returns as soon
  as any ``k`` workers have transitions ready.
* Port pool & health checks: managed via ``CARLAServerManager``.
* Fault-tolerant restart: workers detect CARLA crashes, mark the affected
  episode for discard, and respawn the underlying simulator.
"""

import copy
import logging
import multiprocessing as mp
from multiprocessing import Queue
import os
from pathlib import Path
import queue
import random
import threading
import time
import traceback
import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Deque, Dict, List, Optional, Set, Tuple, Union
from collections import OrderedDict, deque

import numpy as np
import yaml

from b2d_rlinfra.simulation.runners.carla_env_pool_utils import (
    EPISODE_CONTEXT_KEYS,
    build_worker_crash_file_path,
    build_crash_step_result,
    enrich_step_info,
    extract_episode_context,
    format_episode_context,
    is_current_epoch,
    is_reset_obs,
    parse_connection_config,
    read_worker_crash_file,
    update_worker_episode_context,
    write_worker_crash_file,
)
from b2d_rlinfra.simulation.runners.shm_rgb_buffer import ShmRgbBuffer
from b2d_rlinfra.simulation.crash_utils import refine_crash_payload

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

logger = logging.getLogger("Env Pool")

__layer__ = (3, "Simulation")


def _worker_log_prefix(worker_id: int, source: Optional[Any] = None) -> str:
    context = format_episode_context(source)
    if context:
        return f"Worker {worker_id} | {context}"
    return f"Worker {worker_id}"


_TERMINAL_FALLBACK_EVENT_TYPES = {
    "COLLISION_PEDESTRIAN",
    "COLLISION_VEHICLE",
    "COLLISION_STATIC",
    "TRAFFIC_LIGHT_INFRACTION",
    "STOP_INFRACTION",
    "VEHICLE_BLOCKED",
    "ROUTE_DEVIATION",
    "ROUTE_TIMEOUT",
    "SCENARIO_TIMEOUT",
}


def _normalize_event_type(value: Any) -> str:
    if value is None:
        return ""
    if hasattr(value, "name"):
        value = value.name
    text = str(value).strip()
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    return text


def _event_type_from_dict(event: Any) -> str:
    if not isinstance(event, dict):
        return ""
    return _normalize_event_type(event.get("event_type") or event.get("type"))


def _route_completion_percent_from_info(info: Optional[Dict[str, Any]]) -> Optional[float]:
    if not isinstance(info, dict):
        return None

    for key in ("route_completed_ratio", "truncation_route_completed_ratio"):
        value = info.get(key)
        if isinstance(value, (int, float, np.number)):
            return max(0.0, min(float(value) * 100.0, 100.0))

    best_percent = None
    for event in info.get("all_events", []) or []:
        if not isinstance(event, dict):
            continue
        if _event_type_from_dict(event) != "ROUTE_COMPLETION":
            continue
        details = event.get("details") or {}
        if not isinstance(details, dict):
            continue
        value = details.get("route_completed")
        if isinstance(value, (int, float, np.number)):
            percent = max(0.0, min(float(value), 100.0))
            best_percent = percent if best_percent is None else max(best_percent, percent)
    return best_percent


def _main_terminal_type(
    info: Optional[Dict[str, Any]],
    *,
    terminated: bool,
    truncated: bool,
) -> str:
    if truncated:
        return "TRUNCATED"
    if isinstance(info, dict):
        terminate_events = info.get("terminate_events")
        if isinstance(terminate_events, list) and terminate_events:
            event_type = _event_type_from_dict(terminate_events[0])
            if event_type:
                return event_type

        for event in info.get("all_events", []) or []:
            event_type = _event_type_from_dict(event)
            if event_type in _TERMINAL_FALLBACK_EVENT_TYPES:
                return event_type
        route_completion = _route_completion_percent_from_info(info)
        if terminated and route_completion is not None and route_completion >= 99.9:
            return "ROUTE_COMPLETION"
    if terminated:
        return "TERMINATED"
    return "UNKNOWN"


def _format_route_completion(percent: Optional[float]) -> str:
    if percent is None:
        return "nan%"
    return f"{percent:.1f}%"


def _format_episode_boundary_summary(
    info: Optional[Dict[str, Any]],
    *,
    episode_reward: float,
    terminated: bool,
    truncated: bool,
) -> str:
    term = _main_terminal_type(info, terminated=terminated, truncated=truncated)
    route_completion = _route_completion_percent_from_info(info)
    return (
        f" | reward={float(episode_reward):.1f}"
        f" | term={term}"
        f" | rc={_format_route_completion(route_completion)}"
    )


def _configure_worker_logging(log_level: int) -> None:
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    logging.basicConfig(
        level=log_level,
        format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    for logger_name in (
        "Carla",
        "Carla Env",
        "Carla Manager",
        "Env Pool",
        "Carla Server",
        "Interface Wrapper",
        "Training Loop",
        "Policy",
        "Visualization",
    ):
        named_logger = logging.getLogger(logger_name)
        named_logger.setLevel(log_level)
        for handler in named_logger.handlers:
            handler.setLevel(log_level)
            handler.setFormatter(formatter)


class EnvPoolState(Enum):
    INITIALIZING = auto()
    READY = auto()
    RUNNING = auto()
    CLOSING = auto()
    CLOSED = auto()


class WorkerState(Enum):
    STARTING = auto()
    IDLE = auto()
    RESETTING = auto()
    STEPPING = auto()
    WAITING_ACTION = auto()
    ERROR = auto()
    RECOVERING = auto()
    STOPPED = auto()


@dataclass
class WorkerInfo:
    worker_id: int
    state: WorkerState
    carla_host: str
    carla_port: int
    traffic_manager_port: int = 8000
    traffic_manager_seed: int = 0
    epoch: int = 0
    restart_count: int = 0
    last_restart_time: float = 0.0
    episode_count: int = 0
    episodes_since_restart: int = 0
    next_recycle_episode: int = 0
    planned_recycle_count: int = 0
    total_steps: int = 0
    last_error: Optional[str] = None
    last_update: float = 0.0
    current_episode_step: int = 0
    current_episode_reward: float = 0.0
    route_id: str = ""
    scenario_name: str = ""
    scenario_instance_name: str = ""
    town: str = ""


class HealthWorker:
    """Background health-check worker for a ``CARLAEnvPool`` instance."""

    def __init__(
        self,
        check_fn: Callable[[], None],
        interval: float,
        info_fn: Optional[Callable[[], Any]] = None,
        worker_logger: Optional[logging.Logger] = None,
    ) -> None:
        self.check_fn = check_fn
        self.interval = interval
        self.info_fn = info_fn
        self.logger = worker_logger or logger
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()

    def start(self) -> None:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._stop.clear()
            thread = threading.Thread(target=self._run, daemon=True)
            self._thread = thread
        thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        with self._lock:
            thread = self._thread
        self._stop.set()
        if thread:
            thread.join(timeout=timeout)
        with self._lock:
            if self._thread is thread and (thread is None or not thread.is_alive()):
                self._thread = None

    def is_alive(self) -> bool:
        with self._lock:
            return bool(self._thread and self._thread.is_alive())

    def _run(self) -> None:
        check_count = 0
        info_every_checks = (
            max(1, int(round(300.0 / self.interval)))
            if self.interval > 0
            else 1
        )
        while not self._stop.is_set():
            self.check_fn()
            check_count += 1
            active_workers_info = self.info_fn() if self.info_fn else None
            self.logger.debug(
                "[MainProcess] Health check #%s, %s",
                check_count,
                active_workers_info,
            )
            if check_count % info_every_checks == 0:
                self.logger.info(
                    "[MainProcess] Health check #%s, %s",
                    check_count,
                    active_workers_info,
                )
            if self._stop.wait(self.interval):
                break


def _load_config_from_yaml(config_path: Union[str, Path]) -> Dict:
    """Load and validate a YAML config file for ``CARLAEnvPool``.

    Strict checks:
    - File must exist.
    - Suffix must be ``.yaml`` or ``.yml`` (case-insensitive).
    - Top-level YAML document must deserialize to a mapping (``dict``).
    """
    path = Path(config_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(
            f"CARLAEnvPool config path does not exist: {path}"
        )
    if path.suffix.lower() not in (".yaml", ".yml"):
        raise ValueError(
            f"CARLAEnvPool config path must be a .yaml/.yml file, got: {path}"
        )
    with open(path, "r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    if not isinstance(loaded, dict):
        raise ValueError(
            f"CARLAEnvPool config YAML must deserialize to a mapping/dict, "
            f"got {type(loaded).__name__} from {path}"
        )
    return loaded


class CARLAEnvPool:
    def __init__(
        self,
        env_fn: Callable,
        config: Union[Dict, str, Path],
        num_envs: Optional[int] = None,
        carla_hosts: Optional[List[str]] = None,
        carla_ports: Optional[List[int]] = None,
        traffic_manager_ports: Optional[List[int]] = None,
        auto_reset: bool = True,
        max_episode_steps: int = 10000,
        health_check_interval: float = 5.0,
        worker_timeout: float = 120.0,
        server_wait_timeout: float = 120.0,
        start_method: str = "spawn",
        manage_servers: Optional[bool] = None,
        carla_root: Optional[str] = None,
        gpu_ids: Optional[List[int]] = None,
        server_fps: Optional[int] = None,
        server_timeout: float = 120.0,
        fake_bind_lib: Optional[str] = None,
    ):
        self.env_fn = env_fn
        self.auto_reset = auto_reset
        self.max_episode_steps = max_episode_steps
        self.health_check_interval = health_check_interval
        self.worker_timeout = worker_timeout
        self.server_wait_timeout = server_wait_timeout
        self._worker_log_level = logging.root.level

        if isinstance(config, (str, Path)):
            logger.info("CARLAEnvPool: loading config from YAML file %s", config)
            config = _load_config_from_yaml(config)
        elif not isinstance(config, dict):
            raise TypeError(
                f"CARLAEnvPool 'config' must be a dict or a path to a YAML file, "
                f"got {type(config).__name__}"
            )

        self.config = copy.deepcopy(config)
        env_config = self.config.setdefault("environment", {})
        self.worker_recycle_interval_episodes = max(
            0,
            int(env_config.get("worker_recycle_interval_episodes", 0) or 0),
        )
        self.worker_recycle_jitter_episodes = max(
            0,
            int(env_config.get("worker_recycle_jitter_episodes", 0) or 0),
        )
        use_timestamp = env_config.get("use_timestamp", True)
        if use_timestamp and "_experiment_timestamp" not in env_config:
            env_config["_experiment_timestamp"] = datetime.now().strftime("%Y%m%d_%H%M%S")
            logger.info("Experiment timestamp: %s", env_config["_experiment_timestamp"])
        self._result_dir = Path(os.path.abspath(env_config.get("result_dir", "./results")))
        self._crash_event_dir = self._result_dir / "crash_events"
        self._adaptive_manager = None
        self._adaptive_shared_stats = None
        self._adaptive_shared_lock = None

        routes_config = self.config.setdefault("routes", {})
        sample_mode = str(routes_config.get("sample_mode", "sequential")).strip().lower()
        if sample_mode == "adaptive":
            self._adaptive_manager = mp.Manager()
            self._adaptive_shared_stats = self._adaptive_manager.dict()
            self._adaptive_shared_lock = self._adaptive_manager.Lock()
            logger.info("CARLAEnvPool adaptive shared sampler state initialized")

        carla_config = config.get("carla", {})
        self.server_wait_timeout = carla_config.get("server_wait_timeout", self.server_wait_timeout)
        if num_envs is None:
            num_envs = carla_config.get("num_envs", 1)
        self.num_envs = num_envs
        rgb_cfg = config.get("observation_space", {}).get("rgb", {})
        rgb_enabled = bool(rgb_cfg.get("enable", False))

        if carla_root is None:
            carla_root = os.environ.get("CARLA_ROOT", "/data_216/carla_fix")
        if server_fps is None:
            server_fps = carla_config.get("server_fps", carla_config.get("frequency_hz", 10))
        if isinstance(server_fps, str):
            stripped_server_fps = server_fps.strip().lower()
            if stripped_server_fps in ("", "none", "null"):
                server_fps = None
            else:
                server_fps = int(float(server_fps))
        server_timeout = carla_config.get("server_wait_timeout", server_timeout)
        # Keep CARLA launch aligned with the previously stable behavior:
        # do not force -quality-level from YAML during server startup.
        quality_level = "Default"
        render_offscreen = bool(carla_config.get("render_offscreen", True))
        no_sound = bool(carla_config.get("no_sound", True))
        null_rhi = bool(carla_config.get("null_rhi", True))
        no_rendering_mode = bool(carla_config.get("no_rendering_mode", True))
        opengl = bool(carla_config.get("opengl", False))
        vulkan = bool(carla_config.get("vulkan", False))
        no_steam = bool(carla_config.get("no_steam", False))
        multihome = carla_config.get("multihome")
        if rgb_enabled and no_rendering_mode:
            raise ValueError("RGB observation requires env.carla.no_rendering_mode=false")
        if rgb_enabled and null_rhi:
            raise ValueError("RGB observation requires env.carla.null_rhi=false")

        raw_extra_args = carla_config.get("extra_args")
        if isinstance(raw_extra_args, str):
            try:
                import shlex
                extra_args = shlex.split(raw_extra_args)
            except Exception:
                extra_args = raw_extra_args.split()
        elif isinstance(raw_extra_args, (list, tuple)):
            extra_args = list(raw_extra_args)
        else:
            extra_args = []

        if manage_servers is None:
            manage_servers = carla_root is not None and os.path.exists(carla_root)

        if fake_bind_lib is None:
            project_root = Path(__file__).resolve().parents[3]
            default_fake_bind = project_root / "tools" / "fake_bind.so"
            fake_bind_lib = str(default_fake_bind) if default_fake_bind.exists() else None

        parsed = parse_connection_config(
            config=config,
            num_envs=num_envs,
            carla_ports_override=carla_ports,
            tm_ports_override=traffic_manager_ports,
            gpu_ids_override=gpu_ids,
            hosts_override=carla_hosts,
        )
        self.carla_hosts = parsed["hosts"]
        self.carla_ports = parsed["ports"]
        self.traffic_manager_ports = parsed["tm_ports"]
        self.traffic_manager_seeds = parsed["tm_seeds"]
        self.gpu_ids = parsed["gpu_ids"]

        logger.info("CARLAEnvPool: %s environments", num_envs)
        logger.info("  CARLA hosts: %s", self.carla_hosts)
        logger.info("  CARLA ports: %s", self.carla_ports)
        logger.info("  TM ports: %s", self.traffic_manager_ports)
        logger.info("  TM seeds: %s", self.traffic_manager_seeds)
        logger.info("  GPU IDs (graphicsadapter): %s", self.gpu_ids)
        logger.info("  manage_servers: %s", manage_servers)
        if carla_root:
            logger.info("  carla_root: %s", carla_root)
        if fake_bind_lib:
            logger.info("  fake_bind_lib: %s", fake_bind_lib)
        if extra_args:
            logger.info("  extra_args: %s", extra_args)
        logger.info("  server_fps: %s", "disabled" if server_fps is None else server_fps)
        logger.info("  quality_level: %s", quality_level)
        logger.info("  render_offscreen: %s", render_offscreen)
        logger.info("  no_rendering_mode: %s", no_rendering_mode)
        logger.info("  no_sound: %s", no_sound)
        logger.info("  null_rhi: %s", null_rhi)
        logger.info("  opengl: %s", opengl)
        logger.info("  vulkan: %s", vulkan)
        logger.info("  no_steam: %s", no_steam)
        if self.worker_recycle_interval_episodes > 0:
            logger.info(
                "  worker planned recycle: every %s episodes + jitter [0, %s]",
                self.worker_recycle_interval_episodes,
                self.worker_recycle_jitter_episodes,
            )
        else:
            logger.info("  worker planned recycle: disabled")

        self.manage_servers = manage_servers
        self._server_manager = None
        if manage_servers:
            if carla_root is None or not os.path.exists(carla_root):
                raise ValueError(
                    "carla_root is required when manage_servers=True. "
                    "Please set CARLA_ROOT to a valid path."
                )

            from b2d_rlinfra.simulation.runners.carla_server_manager import CARLAServerManager

            self._server_manager = CARLAServerManager(
                carla_root=carla_root,
                default_fps=server_fps,
                default_quality=quality_level,
                auto_restart=True,
                fake_bind_lib=fake_bind_lib,
            )
            logger.info("Starting %s CARLA servers...", num_envs)
            results = self._server_manager.start_servers(
                ports=self.carla_ports,
                gpu_ids=self.gpu_ids,
                hosts=self.carla_hosts,
                timeout=server_timeout,
                quality_level=quality_level,
                multihome=multihome,
                render_offscreen=render_offscreen,
                no_sound=no_sound,
                null_rhi=null_rhi,
                opengl=opengl,
                vulkan=vulkan,
                no_steam=no_steam,
                extra_args=extra_args,
            )
            failed = [(h, p) for (h, p), success in results.items() if not success]
            if failed:
                self._server_manager.stop_all()
                raise RuntimeError(f"Failed to start CARLA servers: {failed}")
            logger.info("All CARLA servers started successfully!")

        self._ctx = mp.get_context(start_method)
        self._workers: Dict[int, Process] = {}
        self._worker_infos: Dict[int, WorkerInfo] = {}

        self._event_queues: Dict[int, Queue] = {}
        self._action_queues: Dict[int, Queue] = {}
        self._control_queues: Dict[int, Queue] = {}
        self._internal_queue: queue.Queue = queue.Queue()

        self._state = EnvPoolState.INITIALIZING
        # All runtime state shared across the main thread, health-check thread,
        # and recovery thread goes through this single re-entrant lock.
        self._runtime_lock = threading.RLock()
        self._restart_in_progress: Set[int] = set()
        self._handled_dead_pids: Dict[int, Optional[int]] = {}
        # Once a worker generation is observed dead, treat its IPC queues as
        # unsafe. We replace them on restart instead of draining them because a
        # broken multiprocessing queue can wedge the reader.
        self._tainted_queues: Set[int] = set()
        self._event_processing_workers: Set[int] = set()
        self._blocked_reset_obs_counts: Dict[int, int] = {}
        self._last_handled_crash_epoch: Dict[int, int] = {}

        self._pending_reset_obs: "OrderedDict[int, Tuple[Any, Dict[str, Any]]]" = OrderedDict()
        self._queued_step_results: Deque[Dict[str, Any]] = deque()

        self._total_episodes = 0
        self._total_steps = 0

        self._health_worker = HealthWorker(
            check_fn=self._check_workers_health,
            interval=self.health_check_interval,
            info_fn=self._get_active_workers_info,
            worker_logger=logger,
        )

        self._recovery_stop = threading.Event()
        self._recovery_queue: queue.Queue = queue.Queue()
        self._recovery_thread: Optional[threading.Thread] = None
        self._recovery_thread_enabled = False
        self._recovery_task_counts: Dict[str, int] = {
            "total": 0,
            "restart_worker": 0,
            "inline": 0,
        }

        self._action_space = None
        self._observation_space = None
        self._use_rgb_shm = False
        self._rgb_obs_key = "rgb"
        self._rgb_shm_prefix: Optional[str] = None
        self._rgb_shm_shape: Optional[Tuple[int, ...]] = None
        self._rgb_shm_dtype: Optional[np.dtype] = None
        self._rgb_shm_num_slots = 2
        self._rgb_shm_buffers: Dict[int, ShmRgbBuffer] = {}

        try:
            self._initialize_rgb_shm_buffers()
            self._start_recovery_worker()
            self._start_workers()
            with self._runtime_lock:
                self._state = EnvPoolState.READY
        except Exception:
            logger.exception("CARLAEnvPool initialization failed, cleaning up partial startup state")
            try:
                self.close()
            except Exception as cleanup_error:
                logger.error("CARLAEnvPool cleanup after init failure failed: %s", cleanup_error)
            raise

    @staticmethod
    def _normalize_rgb_shm_mode(value: Any) -> str:
        if value is None:
            return "auto"
        if isinstance(value, bool):
            return "true" if value else "false"
        mode = str(value).strip().lower()
        if mode in ("", "auto"):
            return "auto"
        if mode in ("1", "true", "yes", "on", "enable", "enabled"):
            return "true"
        if mode in ("0", "false", "no", "off", "disable", "disabled", "none", "null"):
            return "false"
        raise ValueError(
            "ipc.use_rgb_shm must be one of auto/true/false, got {!r}".format(value)
        )

    def _create_rgb_shm_buffer(self, worker_id: int) -> ShmRgbBuffer:
        if self._rgb_shm_shape is None or self._rgb_shm_dtype is None or self._rgb_shm_prefix is None:
            raise RuntimeError("RGB shm metadata is not initialized")
        return ShmRgbBuffer(
            worker_id,
            rgb_shape=self._rgb_shm_shape,
            rgb_dtype=self._rgb_shm_dtype,
            prefix=self._rgb_shm_prefix,
            num_slots=self._rgb_shm_num_slots,
            create=True,
        )

    def _cleanup_rgb_shm_buffers(self) -> None:
        with self._runtime_lock:
            buffers = list(self._rgb_shm_buffers.values())
            self._rgb_shm_buffers = {}
        for buffer in buffers:
            try:
                buffer.close()
            except Exception as exc:
                logger.warning("Failed to close RGB shm buffer %s: %s", buffer.path, exc)
            try:
                buffer.unlink()
            except Exception as exc:
                logger.warning("Failed to unlink RGB shm buffer %s: %s", buffer.path, exc)

    def _initialize_rgb_shm_buffers(self) -> None:
        ipc_cfg = self.config.get("ipc", {}) if isinstance(self.config, dict) else {}
        mode = self._normalize_rgb_shm_mode(ipc_cfg.get("use_rgb_shm", "auto"))
        if mode == "false":
            logger.info("RGB SHM IPC: disabled by config")
            return

        algo_cfg = self.config.get("algorithm", {}) if isinstance(self.config, dict) else {}
        algo_name = str(algo_cfg.get("name", "") or "").strip().lower() if isinstance(algo_cfg, dict) else ""
        if algo_name and algo_name not in ("ppo", "a2c"):
            if mode == "true":
                raise ValueError(
                    "ipc.use_rgb_shm is currently supported only for A2C/PPO on-policy training, "
                    "got algorithm.name={!r}".format(algo_name)
                )
            logger.info(
                "RGB SHM IPC: disabled for algorithm.name=%s (only A2C/PPO are enabled in auto mode)",
                algo_name,
            )
            return

        obs_cfg = self.config.get("observation_space", {}) if isinstance(self.config, dict) else {}
        rgb_cfg = obs_cfg.get("rgb", {}) if isinstance(obs_cfg, dict) else {}
        rgb_enabled = bool(rgb_cfg.get("enable", False)) if isinstance(rgb_cfg, dict) else False

        if not rgb_enabled:
            if mode == "true":
                raise ValueError("ipc.use_rgb_shm is enabled but observation_space.rgb.enable is false")
            logger.info("RGB SHM IPC: disabled (RGB observation is not enabled)")
            return

        from b2d_rlinfra.environment.spaces import build_observation_space_dict, resolve_rgb_obs_key

        observation_space = build_observation_space_dict(self.config)
        self._rgb_obs_key = resolve_rgb_obs_key(rgb_cfg) if isinstance(rgb_cfg, dict) else "rgb"
        rgb_space = observation_space.spaces.get(self._rgb_obs_key)
        if rgb_space is None:
            if mode == "true":
                raise ValueError(
                    "ipc.use_rgb_shm is enabled but RGB observation space key {!r} is absent".format(
                        self._rgb_obs_key
                    )
                )
            logger.info(
                "RGB SHM IPC: disabled (RGB observation space key %r is absent)",
                self._rgb_obs_key,
            )
            return

        self._use_rgb_shm = True
        self._rgb_shm_prefix = "carla_rgb_{}".format(uuid.uuid4().hex[:12])
        self._rgb_shm_shape = tuple(int(dim) for dim in rgb_space.shape)
        self._rgb_shm_dtype = np.dtype(rgb_space.dtype)
        self._rgb_shm_num_slots = 2
        buffers: Dict[int, ShmRgbBuffer] = {}
        try:
            for worker_id in range(self.num_envs):
                buffers[worker_id] = self._create_rgb_shm_buffer(worker_id)
            with self._runtime_lock:
                self._rgb_shm_buffers = buffers
        except Exception:
            for buffer in buffers.values():
                try:
                    buffer.close()
                except Exception:
                    pass
                try:
                    buffer.unlink()
                except Exception:
                    pass
            self._cleanup_rgb_shm_buffers()
            raise

        per_worker_mib = (
            self._rgb_shm_num_slots
            * int(np.prod(self._rgb_shm_shape))
            * self._rgb_shm_dtype.itemsize
            / (1024.0 * 1024.0)
        )
        logger.info(
            "RGB SHM IPC: enabled key=%s shape=%s dtype=%s slots=%s workers=%s %.1f MiB/worker",
            self._rgb_obs_key,
            self._rgb_shm_shape,
            self._rgb_shm_dtype,
            self._rgb_shm_num_slots,
            self.num_envs,
            per_worker_mib,
        )

    def _replace_rgb_shm_buffer(self, worker_id: int) -> None:
        if not self._use_rgb_shm:
            return
        with self._runtime_lock:
            old_buffer = self._rgb_shm_buffers.pop(worker_id, None)
        if old_buffer is not None:
            try:
                old_buffer.close()
            except Exception as exc:
                logger.warning("Worker %s: failed to close old RGB shm %s: %s", worker_id, old_buffer.path, exc)
            try:
                old_buffer.unlink()
            except Exception as exc:
                logger.warning("Worker %s: failed to unlink old RGB shm %s: %s", worker_id, old_buffer.path, exc)

        new_buffer = self._create_rgb_shm_buffer(worker_id)
        with self._runtime_lock:
            self._rgb_shm_buffers[worker_id] = new_buffer

    def _rgb_shm_metadata_for_worker(self, worker_id: int) -> Optional[Dict[str, Any]]:
        if not self._use_rgb_shm:
            return None
        with self._runtime_lock:
            buffer = self._rgb_shm_buffers.get(worker_id)
            rgb_obs_key = self._rgb_obs_key
        if buffer is None:
            raise RuntimeError("RGB shm buffer for worker {} is not initialized".format(worker_id))
        return buffer.metadata(rgb_obs_key)

    def _restore_rgb_from_shm(self, worker_id: int, msg: Dict[str, Any]) -> None:
        slot = msg.pop("rgb_shm_slot", None)
        expected_seq = msg.pop("rgb_shm_seq", None)
        if slot is None:
            return
        if expected_seq is None:
            raise RuntimeError(
                "Received RGB shm slot without rgb_shm_seq from worker {}".format(worker_id)
            )
        if not self._use_rgb_shm:
            raise RuntimeError(
                "Received RGB shm metadata from worker {} while RGB SHM IPC is disabled".format(worker_id)
            )
        with self._runtime_lock:
            buffer = self._rgb_shm_buffers.get(worker_id)
            rgb_obs_key = self._rgb_obs_key
        if buffer is None:
            raise RuntimeError("Missing RGB shm buffer for worker {}".format(worker_id))

        rgb = buffer.read_rgb(slot, expected_seq=expected_seq)
        obs = msg.get("observation")
        if obs is None:
            obs = {}
        if not isinstance(obs, dict):
            raise TypeError(
                "RGB SHM IPC expects dict observations, got {} from worker {}".format(
                    type(obs).__name__,
                    worker_id,
                )
            )
        restored_obs = dict(obs)
        restored_obs[rgb_obs_key] = rgb
        msg["observation"] = restored_obs

    def _mark_event_processing_locked(self, worker_id: int) -> None:
        self._event_processing_workers.add(worker_id)

    def _unmark_event_processing(self, worker_id: int) -> None:
        with self._runtime_lock:
            self._event_processing_workers.discard(worker_id)

    def _start_workers(self) -> None:
        for worker_id in range(self.num_envs):
            with self._runtime_lock:
                info = WorkerInfo(
                    worker_id=worker_id,
                    state=WorkerState.STARTING,
                    carla_host=self.carla_hosts[worker_id],
                    carla_port=self.carla_ports[worker_id],
                    traffic_manager_port=self.traffic_manager_ports[worker_id],
                    traffic_manager_seed=self.traffic_manager_seeds[worker_id],
                    last_update=time.time(),
                )
                self._schedule_worker_recycle_locked(info)
                self._worker_infos[worker_id] = info
            self._spawn_worker_process(worker_id)

        self._start_health_check()
        worker_ready_timeout = self.config.get("carla", {}).get("worker_ready_timeout", 600.0)
        self._wait_for_workers_ready(timeout=worker_ready_timeout)

    def _next_recycle_episode_target(self) -> int:
        if self.worker_recycle_interval_episodes <= 0:
            return 0
        jitter = (
            random.randint(0, self.worker_recycle_jitter_episodes)
            if self.worker_recycle_jitter_episodes > 0
            else 0
        )
        return self.worker_recycle_interval_episodes + jitter

    def _schedule_worker_recycle_locked(self, info: WorkerInfo) -> None:
        info.episodes_since_restart = 0
        info.next_recycle_episode = self._next_recycle_episode_target()

    def _planned_recycle_reason_on_terminal_locked(
        self,
        info: WorkerInfo,
        result: Dict[str, Any],
    ) -> Optional[str]:
        if result.get("crashed", False):
            return None

        result_info = result.get("info") or {}
        episodes_since_restart = result.get(
            "episodes_since_restart",
            result_info.get("episodes_since_restart"),
        )
        recycle_target = result.get(
            "worker_recycle_target_episode",
            result_info.get("worker_recycle_target_episode"),
        )
        if episodes_since_restart is not None:
            try:
                info.episodes_since_restart = int(episodes_since_restart)
            except (TypeError, ValueError):
                pass
        else:
            info.episodes_since_restart += 1
        if recycle_target is not None:
            try:
                info.next_recycle_episode = int(recycle_target)
            except (TypeError, ValueError):
                pass

        planned_recycle = bool(result.get("planned_recycle") or result_info.get("planned_recycle"))
        if not planned_recycle:
            return None

        info.planned_recycle_count += 1
        return (
            "planned_episode_recycle "
            f"episodes_since_restart={info.episodes_since_restart} "
            f"target={info.next_recycle_episode} "
            f"total_worker_episodes={info.episode_count} "
            f"planned_recycle_count={info.planned_recycle_count}"
        )

    def _replace_worker_queues(self, worker_id: int) -> Tuple[Queue, Queue, Queue]:
        with self._runtime_lock:
            old_queues = [
                self._event_queues.get(worker_id),
                self._action_queues.get(worker_id),
                self._control_queues.get(worker_id),
            ]

            event_queue = self._ctx.Queue()
            action_queue = self._ctx.Queue()
            control_queue = self._ctx.Queue()

            self._event_queues[worker_id] = event_queue
            self._action_queues[worker_id] = action_queue
            self._control_queues[worker_id] = control_queue

        for old_q in old_queues:
            if old_q is not None:
                try:
                    old_q.close()
                    old_q.cancel_join_thread()
                except Exception:
                    pass

        return event_queue, action_queue, control_queue

    def _spawn_worker_process(self, worker_id: int) -> None:
        event_queue, action_queue, control_queue = self._replace_worker_queues(worker_id)
        with self._runtime_lock:
            info = self._worker_infos[worker_id]
            worker_epoch = info.epoch
            worker_host = info.carla_host
            worker_port = info.carla_port
            tm_port = info.traffic_manager_port
            worker_recycle_target_episode = info.next_recycle_episode
        rgb_shm_meta = self._rgb_shm_metadata_for_worker(worker_id)
        worker = self._ctx.Process(
            target=self._worker_main,
            args=(
                worker_id,
                worker_host,
                worker_port,
                tm_port,
                info.traffic_manager_seed,
                self.env_fn,
                self.config,
                event_queue,
                action_queue,
                control_queue,
                self.auto_reset,
                self.max_episode_steps,
                self._adaptive_shared_stats,
                self._adaptive_shared_lock,
                self.server_wait_timeout,
                worker_epoch,
                worker_recycle_target_episode,
                rgb_shm_meta,
                self._worker_log_level,
            ),
            daemon=True,
        )
        worker.start()
        with self._runtime_lock:
            info = self._worker_infos.get(worker_id)
            self._workers[worker_id] = worker
            self._handled_dead_pids.pop(worker_id, None)
            self._tainted_queues.discard(worker_id)
            if info is not None:
                info.last_update = time.time()
                worker_epoch = info.epoch
                worker_host = info.carla_host
                worker_port = info.carla_port
                tm_port = info.traffic_manager_port
        logger.info(
            "Worker %s process started pid=%s host=%s port=%s tm_port=%s epoch=%s",
            worker_id,
            worker.pid,
            worker_host,
            worker_port,
            tm_port,
            worker_epoch,
        )

    def _consume_worker_crash_file(
        self,
        worker_id: int,
        worker_epoch: int,
    ) -> Optional[Dict[str, Any]]:
        try:
            return read_worker_crash_file(
                self._result_dir,
                worker_id,
                worker_epoch,
                delete_after_read=True,
            )
        except Exception as exc:
            crash_path = build_worker_crash_file_path(self._result_dir, worker_id, worker_epoch)
            logger.warning(
                "Failed to read crash file for worker %s epoch %s at %s: %s",
                worker_id,
                worker_epoch,
                crash_path,
                exc,
            )
            return None

    def _wait_for_workers_ready(self, timeout: float = 300.0) -> None:
        start_time = time.time()
        workers_initialized: Set[int] = set()
        workers_ready: Set[int] = set()
        logger.info("Waiting for %s workers to be ready...", self.num_envs)

        while len(workers_ready) < self.num_envs:
            if time.time() - start_time > timeout:
                raise TimeoutError(
                    f"Timeout waiting for workers. Initialized: {len(workers_initialized)}/{self.num_envs}, "
                    f"Ready: {len(workers_ready)}/{self.num_envs}"
                )

            with self._runtime_lock:
                event_items = list(self._event_queues.items())
            random.shuffle(event_items)
            for wid, event_q in event_items:
                while True:
                    log_msg = None
                    worker_id = None
                    try:
                        with self._runtime_lock:
                            if self._event_queues.get(wid) is not event_q:
                                break
                            if wid in self._tainted_queues:
                                break
                            worker = self._workers.get(wid)
                            if worker is not None and not worker.is_alive():
                                self._tainted_queues.add(wid)
                                break
                            msg = event_q.get_nowait()
                            worker_id = msg["worker_id"]
                            info = self._worker_infos.get(worker_id)
                            if info is None or not is_current_epoch(info.epoch, msg):
                                continue
                            msg_type = msg.get("type")
                            self._mark_event_processing_locked(worker_id)

                        try:
                            if msg_type in ("reset_obs", "step_result"):
                                self._restore_rgb_from_shm(worker_id, msg)

                            with self._runtime_lock:
                                if self._event_queues.get(wid) is not event_q:
                                    break
                                if wid in self._tainted_queues:
                                    break
                                info = self._worker_infos.get(worker_id)
                                if info is None or not is_current_epoch(info.epoch, msg):
                                    continue
                                if msg_type == "ready":
                                    workers_initialized.add(worker_id)
                                    info.state = WorkerState.IDLE
                                    info.last_update = time.time()
                                    if self._action_space is None:
                                        self._action_space = msg.get("action_space")
                                        self._observation_space = msg.get("observation_space")
                                    log_msg = (
                                        "Worker %s initialized (%s/%s)",
                                        worker_id,
                                        len(workers_initialized),
                                        self.num_envs,
                                    )
                                elif msg_type == "reset_obs":
                                    workers_ready.add(worker_id)
                                    self._store_reset_obs(worker_id, msg["observation"], msg.get("info") or {})
                                    update_worker_episode_context(info, msg.get("info") or {})
                                    info.current_episode_step = 0
                                    info.current_episode_reward = 0.0
                                    info.state = WorkerState.WAITING_ACTION
                                    info.last_update = time.time()
                                    log_msg = (
                                        "%s ready (%s/%s)",
                                        _worker_log_prefix(worker_id, info),
                                        len(workers_ready),
                                        self.num_envs,
                                    )
                        finally:
                            self._unmark_event_processing(worker_id)
                    except queue.Empty:
                        break
                    except OSError:
                        break
                    if log_msg is not None:
                        logger.info(*log_msg)

            time.sleep(0.1)

        logger.info("All workers ready!")

    @staticmethod
    def _worker_main(
        worker_id: int,
        carla_host: str,
        carla_port: int,
        traffic_manager_port: int,
        traffic_manager_seed: int,
        env_fn: Callable,
        config: Dict,
        event_queue: Queue,
        action_queue: Queue,
        control_queue: Queue,
        auto_reset: bool,
        max_episode_steps: int,
        adaptive_shared_stats=None,
        adaptive_shared_lock=None,
        server_wait_timeout: float = 120.0,
        worker_epoch: int = 0,
        worker_recycle_target_episode: int = 0,
        rgb_shm_meta: Optional[Dict[str, Any]] = None,
        log_level: int = logging.INFO,
    ) -> None:
        _configure_worker_logging(log_level)

        import inspect
        import socket
        import time as time_module

        env = None
        episode_step = 0
        episode_reward = 0.0
        global_step = 0
        early_term_streak = 0
        env_cfg = config.get("environment", {}) if isinstance(config, dict) else {}
        result_dir = os.path.abspath(env_cfg.get("result_dir", "./results"))
        host = carla_host
        exit_reason_override: Optional[str] = None
        rgb_shm_buffer: Optional[ShmRgbBuffer] = None
        rgb_obs_key = "rgb"
        rgb_shm_num_slots = 2
        next_rgb_slot = 0
        rgb_shm_seq = 0
        if isinstance(rgb_shm_meta, dict) and rgb_shm_meta.get("enabled"):
            rgb_obs_key = str(rgb_shm_meta.get("rgb_obs_key", "rgb"))
            rgb_shm_num_slots = int(rgb_shm_meta.get("num_slots", 2))
            rgb_shm_buffer = ShmRgbBuffer(
                worker_id=int(rgb_shm_meta.get("worker_id", worker_id)),
                rgb_shape=tuple(rgb_shm_meta["rgb_shape"]),
                rgb_dtype=np.dtype(rgb_shm_meta["rgb_dtype"]),
                prefix=str(rgb_shm_meta["prefix"]),
                num_slots=rgb_shm_num_slots,
                create=False,
            )
            logger.info(
                "Worker %s: attached RGB shm key=%s path=%s shape=%s dtype=%s slots=%s",
                worker_id,
                rgb_obs_key,
                rgb_shm_buffer.path,
                rgb_shm_buffer.rgb_shape,
                rgb_shm_buffer.rgb_dtype,
                rgb_shm_num_slots,
            )
        try:
            worker_recycle_target_episode = max(0, int(worker_recycle_target_episode or 0))
        except (TypeError, ValueError):
            worker_recycle_target_episode = 0
        worker_episodes_since_restart = 0

        def _put_event(msg: Dict[str, Any]) -> None:
            msg.setdefault("worker_id", worker_id)
            msg.setdefault("worker_epoch", worker_epoch)
            event_queue.put(msg)

        def _worker_prefix(source: Optional[Any] = None) -> str:
            return _worker_log_prefix(worker_id, source if source is not None else env)

        def _classify_exception(
            exc: BaseException,
            *,
            default_reason: str,
            default_type: str,
        ) -> Tuple[str, str, str]:
            crash_reason = str(getattr(exc, "crash_reason", "") or default_reason).strip()
            crash_type = str(getattr(exc, "crash_type", "") or default_type).strip()
            crash_detail = str(getattr(exc, "crash_detail", "") or str(exc) or default_reason).strip()
            if exc.__class__.__name__ == "RouteScenarioSetupError":
                crash_reason = "skip_setup_error"
                crash_type = "route_scenario_setup"
                crash_detail = str(exc).strip() or crash_detail
            return refine_crash_payload(crash_reason, crash_type, crash_detail)

        def _write_crash_record(
            *,
            crash_reason: str,
            crash_type: str,
            crash_detail: str,
            traceback_str: Optional[str] = None,
            source: Optional[Any] = None,
            episode_step_override: Optional[int] = None,
            episode_reward_override: Optional[float] = None,
        ) -> None:
            crash_reason, crash_type, crash_detail = refine_crash_payload(
                crash_reason,
                crash_type,
                crash_detail,
            )
            crash_context = extract_episode_context(
                source if source is not None else env,
                preserve_empty=True,
            )
            payload = {
                "worker_id": worker_id,
                "worker_epoch": worker_epoch,
                "episode_step": episode_step if episode_step_override is None else episode_step_override,
                "episode_reward": episode_reward if episode_reward_override is None else episode_reward_override,
                "crash_reason": crash_reason,
                "crash_type": crash_type,
                "crash_detail": crash_detail,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }
            payload.update(crash_context)
            if traceback_str:
                payload["traceback"] = traceback_str
            try:
                crash_path = write_worker_crash_file(result_dir, payload)
                logger.info(
                    "%s | wrote crash record -> %s (reason=%s, type=%s)",
                    _worker_prefix(source),
                    crash_path,
                    crash_reason,
                    crash_type,
                )
            except Exception as crash_write_error:
                logger.error(
                    "%s | failed to write crash record (reason=%s, type=%s): %s",
                    _worker_prefix(source),
                    crash_reason,
                    crash_type,
                    crash_write_error,
                )

        def check_server_available(host: str, port: int) -> bool:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(2.0)
                result = sock.connect_ex((host, port))
                sock.close()
                return result == 0
            except Exception:
                return False

        def wait_for_server(host: str, port: int, timeout: float) -> bool:
            start_time = time_module.time()
            while time_module.time() - start_time < timeout:
                if check_server_available(host, port):
                    time_module.sleep(3)
                    return True
                time_module.sleep(2)
            return False

        def _exit_for_restart(reason: str, exit_reason: str) -> None:
            nonlocal exit_reason_override
            exit_reason_override = exit_reason
            _write_crash_record(
                crash_reason="server_connection_error",
                crash_type=exit_reason,
                crash_detail=reason,
                source=env,
            )
            logger.warning(
                "%s | exiting for health-check restart (%s) for %s:%s...",
                _worker_prefix(),
                reason,
                host,
                carla_port,
            )
            raise SystemExit(1)

        def _wait_for_planned_recycle_stop(source: Optional[Any] = None) -> None:
            logger.info(
                "%s | planned recycle pending; waiting for main process stop command",
                _worker_prefix(source),
            )
            while True:
                try:
                    cmd = control_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                except (EOFError, OSError):
                    return
                if cmd.get("type") == "stop":
                    logger.info("Worker %s: received planned recycle stop command", worker_id)
                    return

        def create_env_with_retry(max_retries: int = 3) -> Any:
            env_fn_signature = inspect.signature(env_fn)
            env_fn_params = env_fn_signature.parameters
            supports_var_kwargs = any(
                param.kind == inspect.Parameter.VAR_KEYWORD
                for param in env_fn_params.values()
            )
            last_error = None
            for attempt in range(max_retries):
                if not check_server_available(host, carla_port):
                    _exit_for_restart(
                        f"server not available (attempt {attempt + 1})",
                        "server_not_available",
                    )

                try:
                    worker_config = copy.deepcopy(config)
                    worker_env_cfg = worker_config.setdefault("environment", {})
                    worker_env_cfg["host"] = host
                    worker_env_cfg["port"] = carla_port
                    worker_env_cfg["traffic_manager_port"] = traffic_manager_port
                    worker_env_cfg["traffic_manager_seed"] = traffic_manager_seed
                    if "carla" in worker_config:
                        worker_config["carla"]["host"] = host
                        worker_config["carla"]["port"] = carla_port
                        worker_config["carla"]["traffic_manager_port"] = traffic_manager_port
                        worker_config["carla"]["traffic_manager_seed"] = traffic_manager_seed
                    env_kwargs = {}
                    if adaptive_shared_stats is not None and (
                        supports_var_kwargs or "adaptive_shared_stats" in env_fn_params
                    ):
                        env_kwargs["adaptive_shared_stats"] = adaptive_shared_stats
                    if adaptive_shared_lock is not None and (
                        supports_var_kwargs or "adaptive_shared_lock" in env_fn_params
                    ):
                        env_kwargs["adaptive_shared_lock"] = adaptive_shared_lock
                    return env_fn(
                        worker_config,
                        worker_id,
                        carla_port,
                        traffic_manager_port,
                        **env_kwargs,
                    )
                except Exception as exc:
                    last_error = exc
                    error_msg = str(exc).lower()
                    is_connection_error = any(
                        kw in error_msg for kw in ["connection", "timeout", "socket", "rpc", "not ready"]
                    )
                    should_restart = (
                        is_connection_error
                        and not any(
                            kw in error_msg
                            for kw in ["not ready", "server not ready", "timeout waiting for carla server"]
                        )
                    )
                    if should_restart:
                        _exit_for_restart(
                            f"env creation connection error: {exc}",
                            "env_creation_connection_error",
                        )
                    if attempt < max_retries - 1:
                        wait_for_server(host, carla_port, server_wait_timeout)
            raise RuntimeError(f"Failed to create env after {max_retries} attempts: {last_error}")

        def _prepare_observation_for_queue(obs: Any) -> Tuple[Any, Optional[int], Optional[int]]:
            nonlocal next_rgb_slot, rgb_shm_seq
            if rgb_shm_buffer is None:
                return obs, None, None
            if not isinstance(obs, dict):
                raise TypeError(
                    "RGB SHM IPC requires dict observations, got {}".format(type(obs).__name__)
                )
            if rgb_obs_key not in obs:
                raise KeyError(
                    "RGB SHM IPC enabled but observation key {!r} is absent".format(rgb_obs_key)
                )
            slot = next_rgb_slot
            next_rgb_slot = (next_rgb_slot + 1) % rgb_shm_num_slots
            seq = rgb_shm_seq
            rgb_shm_seq += 1
            rgb_shm_buffer.write_rgb(slot, obs[rgb_obs_key], seq=seq)
            queue_obs = dict(obs)
            queue_obs.pop(rgb_obs_key, None)
            return queue_obs, slot, seq

        def _make_reset_event(obs: Any, info: Optional[Dict[str, Any]], reason: str) -> Dict[str, Any]:
            reset_info = dict(info or {})
            reset_info["episode_start"] = True
            reset_info["from_reset"] = True
            reset_info["reset_reason"] = reason
            queue_obs, rgb_slot, rgb_seq = _prepare_observation_for_queue(obs)
            event = {"type": "reset_obs", "observation": queue_obs, "info": reset_info}
            if rgb_slot is not None:
                event["rgb_shm_slot"] = rgb_slot
                event["rgb_shm_seq"] = rgb_seq
            return event

        try:
            logger.info("Worker %s: waiting for CARLA server at %s:%s...", worker_id, host, carla_port)
            if not wait_for_server(host, carla_port, server_wait_timeout):
                raise RuntimeError(f"Timeout waiting for CARLA server at {host}:{carla_port}")

            env = create_env_with_retry()
            _put_event(
                {
                    "type": "ready",
                    "carla_port": carla_port,
                    "action_space": env.action_space,
                    "observation_space": env.observation_space,
                    "worker_recycle_target_episode": worker_recycle_target_episode,
                }
            )

            logger.info("Worker %s: performing initial reset...", worker_id)
            try:
                obs, info = env.reset()
            except Exception as exc:
                tb = traceback.format_exc()
                crash_reason, crash_type, crash_detail = _classify_exception(
                    exc,
                    default_reason="initial_reset_failed",
                    default_type=exc.__class__.__name__,
                )
                _write_crash_record(
                    crash_reason=crash_reason,
                    crash_type=crash_type,
                    crash_detail=crash_detail,
                    traceback_str=tb,
                    source=env,
                )
                detail = " ".join(str(crash_detail).split())
                is_load_world_failure = (
                    crash_type == "load_world_failed"
                    or "Failed to load world" in detail
                )
                if is_load_world_failure:
                    logger.error(
                        "%s | initial reset failed: %s - %s",
                        _worker_prefix(),
                        crash_type,
                        detail,
                    )
                else:
                    logger.error("%s | initial reset failed!\n%s", _worker_prefix(), tb)
                raise SystemExit(1)
            logger.info("%s | initial reset completed", _worker_prefix(info))
            episode_step = 0
            episode_reward = 0.0
            _put_event(_make_reset_event(obs, info, "initial"))

            while True:
                try:
                    cmd = control_queue.get_nowait()
                    if cmd.get("type") == "stop":
                        logger.info("Worker %s: received stop command", worker_id)
                        break
                except queue.Empty:
                    pass

                try:
                    action_msg = action_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                action = action_msg["action"]

                try:
                    obs, reward, terminated, truncated, info = env.step(action)

                    termination_reason = info.get("terminate_reason") or info.get("reason")
                    if terminated and not termination_reason:
                        try:
                            scenario = getattr(env, "scenario", None)
                            if scenario is not None and getattr(scenario, "timeout_node", None):
                                if scenario.timeout_node.timeout:
                                    termination_reason = "timeout"
                        except Exception:
                            pass
                    if terminated and not termination_reason:
                        try:
                            from srunner.scenariomanager.traffic_events import TrafficEventType

                            scenario = getattr(env, "scenario", None)
                            if scenario is not None and hasattr(scenario, "get_criteria"):
                                for node in scenario.get_criteria():
                                    for event in node.events:
                                        if event.get_type() in (
                                            TrafficEventType.ROUTE_TIMEOUT,
                                            TrafficEventType.SCENARIO_TIMEOUT,
                                        ):
                                            termination_reason = "timeout"
                                            break
                                    if termination_reason:
                                        break
                        except Exception:
                            pass
                    if termination_reason:
                        info["terminate_reason"] = termination_reason

                    next_step = episode_step + 1
                    early_term_steps = int(env_cfg.get("early_termination_steps", 1))
                    early_term_restart = int(env_cfg.get("early_termination_restart", 3))
                    terminate_events = info.get("terminate_events")
                    no_term_events = isinstance(terminate_events, list) and len(terminate_events) == 0
                    mem_restart_mb = float(env_cfg.get("worker_mem_restart_mb", 0) or 0)
                    mem_mb = None
                    if HAS_PSUTIL:
                        try:
                            mem_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
                            info["proc_mem_mb"] = mem_mb
                        except Exception:
                            mem_mb = None

                    if terminated and not info.get("crashed", False) and next_step <= early_term_steps:
                        if termination_reason == "timeout":
                            info["crashed"] = True
                            if mem_restart_mb > 0 and mem_mb is not None and mem_mb >= mem_restart_mb:
                                info["crash_type"] = "timeout_high_mem"
                                info["error"] = (
                                    f"Timeout with high memory: {mem_mb:.1f}MB >= {mem_restart_mb}MB"
                                )
                            else:
                                info["crash_type"] = "timeout_termination"
                                info["error"] = f"Timeout termination at step {next_step}"
                        elif no_term_events:
                            early_term_streak += 1
                            if early_term_streak >= early_term_restart:
                                info["crashed"] = True
                                info["crash_type"] = "early_termination"
                                info["error"] = (
                                    f"Early termination streak {early_term_streak} "
                                    f"at step {next_step} without terminate events"
                                )
                    else:
                        if next_step > early_term_steps or terminated or truncated:
                            early_term_streak = 0

                    if info.get("crashed", False):
                        crash_reason = str(info.get("crash_reason") or "simulation_crashed").strip()
                        crash_type = str(info.get("crash_type") or "unknown").strip()
                        crash_detail = str(
                            info.get("crash_detail")
                            or info.get("error")
                            or info.get("crash_message")
                            or "Simulation crashed"
                        ).strip()
                        exit_reason_override = crash_reason
                        _write_crash_record(
                            crash_reason=crash_reason,
                            crash_type=crash_type,
                            crash_detail=crash_detail,
                            traceback_str=info.get("traceback"),
                            source=info,
                            episode_step_override=next_step,
                            episode_reward_override=episode_reward + float(reward),
                        )
                        logger.error(
                            "%s | simulation crashed: type=%s, error=%s",
                            _worker_prefix(info),
                            info.get("crash_type", "unknown"),
                            info.get("error", "Simulation crashed"),
                        )
                        raise SystemExit(1)

                    episode_step += 1
                    episode_reward += reward
                    global_step += 1

                    if (not terminated) and episode_step >= max_episode_steps:
                        truncated = True
                        info["TimeLimit.truncated"] = True

                    planned_recycle_after_step = False
                    if (terminated or truncated) and worker_recycle_target_episode > 0:
                        worker_episodes_since_restart += 1
                        if worker_episodes_since_restart >= worker_recycle_target_episode:
                            planned_recycle_after_step = True
                            info["planned_recycle"] = True
                            info["planned_recycle_reason"] = "planned_episode_recycle"
                            info["episodes_since_restart"] = worker_episodes_since_restart
                            info["worker_recycle_target_episode"] = worker_recycle_target_episode

                    reason_str = f", reason={termination_reason}" if terminated and termination_reason else ""
                    logger.debug(
                        "Worker %s: step %s, global_step=%s, reward=%.3f, episode_reward=%.3f, "
                        "terminated=%s, truncated=%s%s",
                        worker_id,
                        episode_step,
                        global_step,
                        reward,
                        episode_reward,
                        terminated,
                        truncated,
                        reason_str,
                    )

                    queue_obs, rgb_slot, rgb_seq = _prepare_observation_for_queue(obs)
                    step_event = {
                        "type": "step_result",
                        "observation": queue_obs,
                        "reward": reward,
                        "terminated": terminated,
                        "truncated": truncated,
                        "info": info,
                        "episode_step": episode_step,
                        "episode_reward": episode_reward,
                        "crashed": False,
                        "planned_recycle": planned_recycle_after_step,
                        "episodes_since_restart": worker_episodes_since_restart,
                        "worker_recycle_target_episode": worker_recycle_target_episode,
                    }
                    if rgb_slot is not None:
                        step_event["rgb_shm_slot"] = rgb_slot
                        step_event["rgb_shm_seq"] = rgb_seq
                    _put_event(step_event)

                    if terminated or truncated:
                        boundary_summary = _format_episode_boundary_summary(
                            info,
                            episode_reward=episode_reward,
                            terminated=terminated,
                            truncated=truncated,
                        )
                        if terminated:
                            logger.info(
                                "%s | episode terminated at step %s%s%s",
                                _worker_prefix(info),
                                episode_step,
                                reason_str,
                                boundary_summary,
                            )
                        else:
                            logger.info(
                                "%s | episode truncated at step %s, resetting%s",
                                _worker_prefix(info),
                                episode_step,
                                boundary_summary,
                            )
                        if isinstance(info, dict):
                            terminate_events = info.get("terminate_events")
                            if isinstance(terminate_events, list) and len(terminate_events) > 1:
                                logger.debug(
                                    "%s | terminate_events=%s",
                                    _worker_prefix(info),
                                    terminate_events,
                                )
                        if planned_recycle_after_step:
                            exit_reason_override = "planned_episode_recycle"
                            logger.info(
                                "%s | planned recycle reached target episode %s; skip auto_reset",
                                _worker_prefix(info),
                                worker_recycle_target_episode,
                            )
                            _wait_for_planned_recycle_stop(info)
                            break
                        if auto_reset:
                            reset_reason = "auto" if terminated else "truncated"
                            start_tag = "async" if terminated else "truncated"
                            logger.debug("Worker %s: auto_reset starting (%s)...", worker_id, start_tag)
                            try:
                                reset_obs, reset_info = env.reset()
                            except Exception as exc:
                                tb = traceback.format_exc()
                                crash_reason, crash_type, crash_detail = _classify_exception(
                                    exc,
                                    default_reason="auto_reset_failed",
                                    default_type=exc.__class__.__name__,
                                )
                                _write_crash_record(
                                    crash_reason=crash_reason,
                                    crash_type=crash_type,
                                    crash_detail=crash_detail,
                                    traceback_str=tb,
                                    source=env,
                                )
                                logger.error("%s | auto reset failed!\n%s", _worker_prefix(), tb)
                                raise SystemExit(1)
                            complete_suffix = "" if terminated else " (truncated)"
                            logger.debug("Worker %s: auto_reset completed%s", worker_id, complete_suffix)
                            episode_step = 0
                            episode_reward = 0.0
                            _put_event(_make_reset_event(reset_obs, reset_info, reset_reason))
                    else:
                        # Normal step: next action can be sent immediately after the
                        # step_result is consumed in the main process.
                        pass

                except Exception as exc:
                    error_msg = str(exc)
                    tb = traceback.format_exc()
                    is_connection_error = any(
                        keyword in error_msg.lower()
                        for keyword in ["connection", "timeout", "socket", "rpc", "disconnected"]
                    )
                    if is_connection_error:
                        if env is not None:
                            try:
                                env.crash_message = "Simulation crashed"
                            except Exception:
                                pass
                        exit_reason_override = "server_connection_error"
                        _write_crash_record(
                            crash_reason="server_connection_error",
                            crash_type=exc.__class__.__name__,
                            crash_detail=error_msg,
                            traceback_str=tb,
                            source=env,
                        )
                        logger.error("%s | connection error, exiting for restart:\n%s", _worker_prefix(), tb)
                        raise SystemExit(1)
                    exit_reason_override = "step_error"
                    _write_crash_record(
                        crash_reason="step_error",
                        crash_type=exc.__class__.__name__,
                        crash_detail=error_msg,
                        traceback_str=tb,
                        source=env,
                    )
                    logger.error("%s | step error:\n%s", _worker_prefix(), tb)
                    raise SystemExit(1)

        except Exception as exc:
            error_msg = str(exc)
            tb = traceback.format_exc()
            crash_reason, crash_type, crash_detail = _classify_exception(
                exc,
                default_reason="server_connection_error" if env is None and episode_step == 0 else "step_error",
                default_type=exc.__class__.__name__,
            )
            _write_crash_record(
                crash_reason=crash_reason,
                crash_type=crash_type,
                crash_detail=crash_detail,
                traceback_str=tb,
                source=env,
            )
            logger.error("%s | fatal exception: %s\n%s", _worker_prefix(), error_msg, tb)
            exit_reason = f"fatal_exception: {error_msg}"
        else:
            exit_reason = "normal_exit (stop command received)"
        finally:
            import sys

            if exit_reason_override is not None:
                exit_reason = exit_reason_override
            elif "exit_reason" not in locals():
                exit_reason = "unknown"
            exc_info = sys.exc_info()
            if exit_reason_override is None and exc_info[0] is not None:
                exit_reason = f"exception: {exc_info[0].__name__}: {exc_info[1]}"
            logger.info("%s | process ending (reason=%s), cleaning up...", _worker_prefix(), exit_reason)
            if env is not None:
                try:
                    env.close()
                except Exception as close_error:
                    logger.warning("%s | env.close() failed: %s", _worker_prefix(), close_error)
            if rgb_shm_buffer is not None:
                try:
                    rgb_shm_buffer.close()
                except Exception as close_error:
                    logger.warning("%s | RGB shm close failed: %s", _worker_prefix(), close_error)
            logger.info("%s | process exiting (reason=%s)", _worker_prefix(), exit_reason)

    def _reset_worker_buffers(self, worker_id: int) -> None:
        self._pending_reset_obs.pop(worker_id, None)
        info = self._worker_infos.get(worker_id)
        if info is not None:
            info.current_episode_step = 0
            info.current_episode_reward = 0.0
            info.last_update = time.time()
            update_worker_episode_context(info, {})

    def _is_reset_obs_blocked(self, worker_id: int) -> bool:
        return self._blocked_reset_obs_counts.get(worker_id, 0) > 0

    def _block_reset_obs(self, worker_id: int) -> None:
        self._blocked_reset_obs_counts[worker_id] = self._blocked_reset_obs_counts.get(worker_id, 0) + 1

    def _unblock_reset_obs(self, worker_id: int) -> None:
        remaining = self._blocked_reset_obs_counts.get(worker_id, 0) - 1
        if remaining > 0:
            self._blocked_reset_obs_counts[worker_id] = remaining
        else:
            self._blocked_reset_obs_counts.pop(worker_id, None)

    def _classify_pending_results_for_epoch(
        self,
        worker_id: int,
        worker_epoch: Optional[int],
    ) -> Tuple[bool, bool]:
        has_non_terminal = False
        has_terminal = False
        for result in self._queued_step_results:
            if result.get("worker_id") != worker_id:
                continue
            if worker_epoch is not None and result.get("worker_epoch") != worker_epoch:
                continue
            if result.get("crashed", False):
                continue
            if result.get("terminated", False) or result.get("truncated", False):
                has_terminal = True
            else:
                has_non_terminal = True
        return has_non_terminal, has_terminal

    def _drop_non_terminal_results_for_epoch(
        self,
        worker_id: int,
        worker_epoch: Optional[int],
    ) -> None:
        if not self._queued_step_results:
            return
        self._queued_step_results = deque(
            result
            for result in self._queued_step_results
            if not (
                result.get("worker_id") == worker_id
                and (worker_epoch is None or result.get("worker_epoch") == worker_epoch)
                and not result.get("crashed", False)
                and not result.get("terminated", False)
                and not result.get("truncated", False)
            )
        )

    def _has_terminal_result_for_route(
        self,
        worker_id: int,
        worker_epoch: Optional[int],
        route_id: str,
    ) -> bool:
        normalized_route_id = str(route_id or "").strip()
        if not normalized_route_id:
            return False
        for result in self._queued_step_results:
            if result.get("worker_id") != worker_id:
                continue
            if worker_epoch is not None and result.get("worker_epoch") != worker_epoch:
                continue
            if not (
                result.get("crashed", False)
                or result.get("terminated", False)
                or result.get("truncated", False)
            ):
                continue
            result_route_id = extract_episode_context(result.get("info") or {}, preserve_empty=True).get("route_id", "")
            if str(result_route_id).strip() == normalized_route_id:
                return True
        return False

    def _stop_worker_process(self, worker_id: int, timeout: float = 5.0) -> None:
        with self._runtime_lock:
            worker = self._workers.get(worker_id)
            info = self._worker_infos.get(worker_id)
            control_queue = self._control_queues.get(worker_id)
        if worker is None:
            return
        if worker.is_alive():
            try:
                logger.info(
                    "[MainProcess] Sending stop to worker %s (pid=%s, state=%s, reason=%s)",
                    worker_id,
                    worker.pid,
                    getattr(info, "state", None),
                    getattr(info, "last_error", None),
                )
                if control_queue is not None:
                    control_queue.put({"type": "stop"})
            except Exception:
                pass
            worker.join(timeout=timeout)
        if worker.is_alive():
            try:
                worker.terminate()
            except Exception:
                pass
            worker.join(timeout=timeout)

    def _restart_server_for_worker(self, worker_id: int) -> bool:
        host = self.carla_hosts[worker_id]
        port = self.carla_ports[worker_id]
        gpu_id = self.gpu_ids[worker_id] if self.gpu_ids else 0
        with self._runtime_lock:
            server_manager = self._server_manager
        if not (self.manage_servers and server_manager is not None):
            logger.warning("Worker %s: server restart skipped (server manager not enabled)", worker_id)
            return True
        try:
            logger.info("Restarting CARLA server for worker %s at %s:%s...", worker_id, host, port)
            return server_manager.restart_server(
                host,
                port,
                gpu_id=gpu_id,
                timeout=self.server_wait_timeout,
            )
        except Exception as exc:
            logger.error("Worker %s: server restart failed at %s:%s: %s", worker_id, host, port, exc)
            return False

    def _restart_worker(self, worker_id: int, reason: str) -> None:
        with self._runtime_lock:
            if self._state in (EnvPoolState.CLOSING, EnvPoolState.CLOSED):
                return
            if worker_id in self._restart_in_progress:
                return
            info = self._worker_infos.get(worker_id)
            if info is None:
                return
            self._restart_in_progress.add(worker_id)
            self._tainted_queues.add(worker_id)
            info.state = WorkerState.RECOVERING
            info.last_error = reason
            info.epoch += 1

        def _restart_task() -> None:
            retry_same_dead_pid = False
            try:
                self._stop_worker_process(worker_id)
                with self._runtime_lock:
                    self._reset_worker_buffers(worker_id)
                if not self._restart_server_for_worker(worker_id):
                    with self._runtime_lock:
                        info = self._worker_infos.get(worker_id)
                        if info is not None:
                            info.state = WorkerState.ERROR
                            info.last_error = "Server restart failed"
                    retry_same_dead_pid = True
                    return

                self._replace_rgb_shm_buffer(worker_id)
                with self._runtime_lock:
                    if self._state in (EnvPoolState.CLOSING, EnvPoolState.CLOSED):
                        return
                    info = self._worker_infos.get(worker_id)
                    if info is None:
                        return
                    info.restart_count += 1
                    info.last_restart_time = time.time()
                    info.state = WorkerState.STARTING
                    self._schedule_worker_recycle_locked(info)
                self._spawn_worker_process(worker_id)
            except Exception as exc:
                logger.error("Worker %s: restart failed: %s", worker_id, exc)
                with self._runtime_lock:
                    info = self._worker_infos.get(worker_id)
                    if info is not None:
                        info.state = WorkerState.ERROR
                        info.last_error = str(exc)
            finally:
                with self._runtime_lock:
                    self._restart_in_progress.discard(worker_id)
                    if retry_same_dead_pid:
                        self._handled_dead_pids.pop(worker_id, None)

        logger.info("[MainProcess] Submit worker restart task for worker %s", worker_id)
        self._submit_recovery_task(
            _restart_task,
            name=f"restart_worker_{worker_id}",
            task_type="restart_worker",
        )

    def _get_active_workers_info(self) -> str:
        with self._runtime_lock:
            active_workers = []
            for wid, info in self._worker_infos.items():
                worker = self._workers.get(wid)
                is_alive = worker.is_alive() if worker else False
                state_short = info.state.name[:3] if info.state else "UNK"
                if is_alive and info.state not in (WorkerState.ERROR, WorkerState.STOPPED):
                    active_workers.append(f"{wid}({state_short})")
            total_workers = len(self._worker_infos)
        return f"active_workers=[{','.join(active_workers)}]({len(active_workers)}/{total_workers})"

    def _start_recovery_worker(self) -> None:
        def recovery_loop() -> None:
            while not self._recovery_stop.is_set():
                try:
                    task = self._recovery_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                if task is None:
                    break
                fn, name = task
                try:
                    fn()
                except Exception as exc:
                    logger.error("[RecoveryWorker] Task %s failed: %s", name, exc)
                finally:
                    try:
                        self._recovery_queue.task_done()
                    except Exception:
                        pass

        try:
            thread = threading.Thread(
                target=recovery_loop,
                daemon=True,
                name="carla_recovery_worker",
            )
            with self._runtime_lock:
                self._recovery_thread = thread
            thread.start()
            with self._runtime_lock:
                self._recovery_thread_enabled = True
            logger.info("[MainProcess] Recovery worker thread started")
        except RuntimeError as exc:
            with self._runtime_lock:
                self._recovery_thread_enabled = False
            logger.error("[MainProcess] Failed to start recovery worker thread: %s", exc)

    def _submit_recovery_task(self, fn: callable, name: str, task_type: str) -> None:
        with self._runtime_lock:
            self._recovery_task_counts["total"] = self._recovery_task_counts.get("total", 0) + 1
            self._recovery_task_counts[task_type] = self._recovery_task_counts.get(task_type, 0) + 1
            recovery_thread_enabled = self._recovery_thread_enabled
            recovery_thread = self._recovery_thread
            counts_snapshot = dict(self._recovery_task_counts)
        try:
            queue_size = self._recovery_queue.qsize()
        except Exception:
            queue_size = -1
        logger.info(
            "[RecoveryWorker] enqueue task=%s, type=%s, counts=%s, queue=%s",
            name,
            task_type,
            counts_snapshot,
            queue_size,
        )

        if recovery_thread_enabled and recovery_thread and recovery_thread.is_alive():
            try:
                self._recovery_queue.put((fn, name), timeout=1.0)
                return
            except Exception as exc:
                logger.warning("[RecoveryWorker] Enqueue failed for %s: %s", name, exc)

        with self._runtime_lock:
            self._recovery_task_counts["inline"] = self._recovery_task_counts.get("inline", 0) + 1
        logger.warning("[RecoveryWorker] Running task inline: %s", name)
        fn()

    def _start_health_check(self) -> None:
        self._health_worker.start()

    def _check_workers_health(self) -> None:
        with self._runtime_lock:
            worker_ids = list(self._worker_infos.keys())
        for worker_id in worker_ids:
            crash_msg = None
            server_lookup = None
            log_context_source = None
            with self._runtime_lock:
                info = self._worker_infos.get(worker_id)
                worker = self._workers.get(worker_id)
                if worker_id in self._event_processing_workers:
                    continue
                if worker_id in self._restart_in_progress or (
                    info is not None and info.state == WorkerState.RECOVERING
                ):
                    continue
                if info is None or worker is None or worker.is_alive():
                    continue
                dead_pid = worker.pid
                if self._handled_dead_pids.get(worker_id) == dead_pid:
                    continue
                self._handled_dead_pids[worker_id] = dead_pid
                exitcode = worker.exitcode
                error_msg = f"worker process died (exitcode={exitcode})"
                # Never read this worker generation's IPC queues again after
                # death is observed; recovery swaps in fresh queues instead.
                self._tainted_queues.add(worker_id)
                crash_payload = self._consume_worker_crash_file(worker_id, info.epoch)
                crash_context = extract_episode_context(info, preserve_empty=True)
                crash_msg = {
                    "type": "worker_crashed",
                    "worker_id": worker_id,
                    "worker_epoch": info.epoch,
                    "episode_step": info.current_episode_step,
                    "episode_reward": info.current_episode_reward,
                    "dead_pid": dead_pid,
                    "error": error_msg,
                    "crash_reason": "process_died",
                    "crash_type": "process_died",
                }
                crash_msg.update(crash_context)
                if crash_payload:
                    payload_context = {
                        key: str(crash_payload.get(key, "") or "").strip()
                        for key in EPISODE_CONTEXT_KEYS
                    }
                    error_msg = str(
                        crash_payload.get("crash_detail")
                        or crash_payload.get("crash_reason")
                        or error_msg
                    )
                    crash_msg.update({
                        "episode_step": crash_payload.get("episode_step", info.current_episode_step),
                        "episode_reward": crash_payload.get("episode_reward", info.current_episode_reward),
                        "error": error_msg,
                        "crash_reason": crash_payload.get("crash_reason", "process_died"),
                        "crash_type": crash_payload.get("crash_type", "process_died"),
                        "crash_detail": crash_payload.get("crash_detail"),
                        "traceback": crash_payload.get("traceback"),
                    })
                    crash_msg.update(payload_context)
                    log_context_source = {**crash_payload, **payload_context}
                else:
                    crash_msg["crash_detail"] = error_msg
                    log_context_source = info
                if self._server_manager is not None:
                    server_lookup = (self._server_manager, info.carla_host, info.carla_port)
                info.state = WorkerState.STOPPED
                info.last_error = error_msg
                should_report_startup_crash = (
                    self._state != EnvPoolState.INITIALIZING
                )
            logger.error(
                "%s | process died! pid=%s exitcode=%s host=%s port=%s tm_port=%s "
                "state=%s last_error=%s last_update=%.2f",
                _worker_log_prefix(worker_id, log_context_source or info),
                worker.pid,
                exitcode,
                info.carla_host,
                info.carla_port,
                info.traffic_manager_port,
                info.state,
                info.last_error,
                info.last_update,
            )
            if crash_msg is not None and should_report_startup_crash:
                try:
                    self._internal_queue.put(crash_msg)
                except Exception:
                    pass
            if server_lookup is not None:
                server_manager, host, port = server_lookup
                server_info = server_manager.get_server_info(host, port)
                if server_info is not None:
                    proc_alive = None
                    if server_info.process is not None:
                        try:
                            proc_alive = server_info.process.poll() is None
                        except Exception:
                            proc_alive = None
                logger.error(
                    "%s | server state=%s pid=%s proc_alive=%s error=%s",
                    _worker_log_prefix(worker_id, info),
                    server_info.state,
                    server_info.pid,
                    proc_alive,
                    server_info.error_message,
                )
            self._restart_worker(worker_id, reason=error_msg)

    def _store_reset_obs(self, worker_id: int, obs: Any, info: Dict[str, Any]) -> None:
        self._pending_reset_obs[worker_id] = (obs, info)

    def _process_internal_queue(self) -> None:
        while True:
            try:
                msg = self._internal_queue.get_nowait()
            except queue.Empty:
                break

            msg_type = msg.get("type")
            if msg_type != "worker_crashed":
                continue
            with self._runtime_lock:
                worker_id = msg["worker_id"]
                worker_epoch = msg.get("worker_epoch")
                last_handled_epoch = self._last_handled_crash_epoch.get(worker_id)
                if worker_epoch is not None and last_handled_epoch is not None and worker_epoch <= last_handled_epoch:
                    continue
                has_non_terminal, _ = self._classify_pending_results_for_epoch(worker_id, worker_epoch)
                crash_route_id = str(msg.get("route_id", "") or "").strip()
                if self._has_terminal_result_for_route(worker_id, worker_epoch, crash_route_id):
                    self._last_handled_crash_epoch[worker_id] = worker_epoch if worker_epoch is not None else -1
                    continue
                self._last_handled_crash_epoch[worker_id] = worker_epoch if worker_epoch is not None else -1
                if has_non_terminal:
                    self._drop_non_terminal_results_for_epoch(worker_id, worker_epoch)
                self._queued_step_results.append(
                    build_crash_step_result(
                        worker_id=worker_id,
                        crash_reason=msg.get("crash_reason", "process_died"),
                        crash_type=msg.get("crash_type", "process_died"),
                        error_msg=msg.get("error", "worker process died"),
                        crash_detail=msg.get("crash_detail"),
                        traceback_str=msg.get("traceback"),
                        worker_epoch=worker_epoch,
                        episode_step=msg.get("episode_step"),
                        episode_reward=msg.get("episode_reward"),
                        worker_info=self._worker_infos.get(worker_id),
                        context_source=msg,
                    )
                )
                self._block_reset_obs(worker_id)

    def _process_event_queues(self) -> None:
        with self._runtime_lock:
            event_items = list(self._event_queues.items())
        random.shuffle(event_items)
        for wid, event_q in event_items:
            while True:
                planned_recycle_request = None
                worker_id = None
                try:
                    with self._runtime_lock:
                        if self._event_queues.get(wid) is not event_q:
                            break
                        if wid in self._tainted_queues:
                            break
                        worker = self._workers.get(wid)
                        if worker is not None and not worker.is_alive():
                            self._tainted_queues.add(wid)
                            break
                        msg = event_q.get_nowait()
                        worker_id = msg["worker_id"]
                        info = self._worker_infos.get(worker_id)
                        if info is None or not is_current_epoch(info.epoch, msg):
                            continue

                        msg_type = msg.get("type")
                        self._mark_event_processing_locked(worker_id)

                    try:
                        if msg_type in ("reset_obs", "step_result"):
                            self._restore_rgb_from_shm(worker_id, msg)

                        with self._runtime_lock:
                            if self._event_queues.get(wid) is not event_q:
                                break
                            if wid in self._tainted_queues:
                                break
                            info = self._worker_infos.get(worker_id)
                            if info is None or not is_current_epoch(info.epoch, msg):
                                continue
                            now = time.time()
                            if msg_type == "ready":
                                info.last_update = now
                                if self._action_space is None:
                                    self._action_space = msg.get("action_space")
                                    self._observation_space = msg.get("observation_space")
                            elif msg_type == "reset_obs":
                                self._store_reset_obs(worker_id, msg["observation"], msg.get("info") or {})
                                update_worker_episode_context(info, msg.get("info") or {})
                                info.current_episode_step = 0
                                info.current_episode_reward = 0.0
                                info.state = WorkerState.WAITING_ACTION
                                info.last_update = now
                            elif msg_type == "step_result":
                                self._queued_step_results.append(msg)
                                update_worker_episode_context(info, msg.get("info") or {})
                                info.current_episode_step = msg.get("episode_step", 0)
                                info.current_episode_reward = msg.get("episode_reward", 0.0)
                                info.total_steps += 1
                                self._total_steps += 1
                                if msg.get("terminated", False) or msg.get("truncated", False):
                                    info.episode_count += 1
                                    self._total_episodes += 1
                                    recycle_reason = self._planned_recycle_reason_on_terminal_locked(info, msg)
                                    if recycle_reason is not None:
                                        planned_recycle_request = (worker_id, recycle_reason, info)
                                        info.state = WorkerState.RECOVERING
                                    else:
                                        info.state = WorkerState.RESETTING if self.auto_reset else WorkerState.WAITING_ACTION
                                else:
                                    info.state = WorkerState.WAITING_ACTION
                                info.last_update = now
                    finally:
                        self._unmark_event_processing(worker_id)
                    if planned_recycle_request is not None:
                        planned_worker_id, recycle_reason, planned_info = planned_recycle_request
                        logger.info(
                            "%s | requesting planned recycle after terminal result: %s",
                            _worker_log_prefix(planned_worker_id, planned_info),
                            recycle_reason,
                        )
                        self._restart_worker(planned_worker_id, reason=recycle_reason)
                except queue.Empty:
                    break
                except OSError:
                    break

    def _process_queues(self) -> None:
        self._process_internal_queue()
        self._process_event_queues()

    def _drain_step_results(
        self,
        process_queues: bool = True,
        exclude_workers: Optional[Set[int]] = None,
    ) -> List[Dict[str, Any]]:
        if process_queues:
            self._process_queues()
        with self._runtime_lock:
            results: List[Dict[str, Any]] = []
            seen_workers: Set[int] = set(exclude_workers or set())
            remaining: Deque[Dict[str, Any]] = deque()
            while self._queued_step_results:
                result = self._queued_step_results.popleft()
                worker_id = result.get("worker_id")
                if worker_id in seen_workers:
                    remaining.append(result)
                    continue
                results.append(result)
                seen_workers.add(worker_id)
                if result.get("crashed", False) and worker_id is not None:
                    self._unblock_reset_obs(worker_id)
            self._queued_step_results = remaining
        random.shuffle(results)
        return results

    def _consume_pending_reset_obs(
        self,
        obs_dict: Dict[int, Any],
        info_dict: Dict[int, Dict[str, Any]],
        ready_ids: Set[int],
        action_worker_ids: Optional[Set[int]] = None,
        terminal_worker_ids: Optional[Set[int]] = None,
        crashed_worker_ids: Optional[Set[int]] = None,
    ) -> None:
        action_worker_ids = action_worker_ids or set()
        terminal_worker_ids = terminal_worker_ids or set()
        crashed_worker_ids = crashed_worker_ids or set()

        with self._runtime_lock:
            candidate_ids = list(self._pending_reset_obs.keys())
            random.shuffle(candidate_ids)
            for wid in candidate_ids:
                if self._is_reset_obs_blocked(wid):
                    continue
                if wid in ready_ids or wid in action_worker_ids or wid in terminal_worker_ids or wid in crashed_worker_ids:
                    continue
                obs, info = self._pending_reset_obs.pop(wid)
                if not is_reset_obs(info):
                    continue
                obs_dict[wid] = obs
                info_dict[wid] = dict(info)
                ready_ids.add(wid)

    def _collect_reset_observations(
        self,
        min_ready: int,
        timeout: float,
    ) -> Tuple[Dict[int, Any], Dict[int, Dict[str, Any]]]:
        start_time = time.time()
        while True:
            self._process_queues()
            with self._runtime_lock:
                available_ids = [wid for wid in self._pending_reset_obs.keys() if not self._is_reset_obs_blocked(wid)]
            if len(available_ids) >= min_ready:
                break
            if time.time() - start_time > timeout:
                logger.warning(
                    "[reset] Timeout after %.1fs, returning current available observations: %s",
                    timeout,
                    available_ids,
                )
                break
            time.sleep(0.001)

        obs_dict: Dict[int, Any] = {}
        info_dict: Dict[int, Dict[str, Any]] = {}
        with self._runtime_lock:
            candidate_ids = list(self._pending_reset_obs.keys())
            random.shuffle(candidate_ids)
            for wid in candidate_ids:
                if self._is_reset_obs_blocked(wid):
                    continue
                obs, info = self._pending_reset_obs.pop(wid)
                obs_dict[wid] = obs
                info_dict[wid] = dict(info)
                if len(obs_dict) >= self.num_envs:
                    break
        return obs_dict, info_dict

    def send_actions(self, worker_ids: List[int], actions: Dict[int, Any]) -> None:
        timestamp = time.time()
        with self._runtime_lock:
            for worker_id in worker_ids:
                if worker_id not in actions:
                    raise ValueError(f"Missing action for worker {worker_id}")
                info = self._worker_infos.get(worker_id)
                if info is None:
                    continue
                if info.state != WorkerState.WAITING_ACTION:
                    logger.warning("Skip action for worker %s: state=%s", worker_id, info.state.name)
                    continue
                action_queue = self._action_queues.get(worker_id)
                if action_queue is None:
                    continue
                try:
                    action_queue.put({"action": actions[worker_id], "timestamp": timestamp})
                except (OSError, ValueError):
                    logger.warning("send_actions: queue closed for worker %s, skipping", worker_id)
                    continue
                info.state = WorkerState.STEPPING
                info.last_update = timestamp

    def _apply_step_result(
        self,
        result: Dict[str, Any],
        obs_dict: Dict[int, Any],
        reward_dict: Dict[int, float],
        terminated_dict: Dict[int, bool],
        truncated_dict: Dict[int, bool],
        info_dict: Dict[int, Dict[str, Any]],
        ready_ids: Set[int],
        crashed_ids: Set[int],
        terminal_ids: Set[int],
    ) -> None:
        if result.get("type") != "step_result":
            return
        wid = result["worker_id"]
        ready_ids.add(wid)
        reward_dict[wid] = result["reward"]
        terminated_dict[wid] = bool(result["terminated"])
        truncated_dict[wid] = bool(result["truncated"])
        info_dict[wid] = enrich_step_info(result)

        if result.get("crashed", False):
            crashed_ids.add(wid)
            return

        obs_dict[wid] = result["observation"]
        if result.get("terminated", False) or result.get("truncated", False):
            terminal_ids.add(wid)

    def step(
        self,
        actions: Dict[int, Any],
        min_ready: int = 1,
        timeout: float = 30.0,
    ) -> Tuple[Dict[int, Any], Dict[int, float], Dict[int, bool], Dict[int, bool], Dict[int, Dict[str, Any]]]:
        with self._runtime_lock:
            if self._state == EnvPoolState.READY:
                self._state = EnvPoolState.RUNNING

        obs_dict: Dict[int, Any] = {}
        reward_dict: Dict[int, float] = {}
        terminated_dict: Dict[int, bool] = {}
        truncated_dict: Dict[int, bool] = {}
        info_dict: Dict[int, Dict[str, Any]] = {}
        ready_ids: Set[int] = set()
        crashed_ids: Set[int] = set()
        terminal_ids: Set[int] = set()
        seen_result_workers: Set[int] = set()
        action_worker_ids = set(actions.keys())
        if actions:
            self.send_actions(list(action_worker_ids), actions)

        start_time = time.time()

        def count_ready() -> int:
            return len(ready_ids)

        while count_ready() < min_ready:
            self._process_queues()
            for result in self._drain_step_results(
                process_queues=False,
                exclude_workers=seen_result_workers,
            ):
                self._apply_step_result(
                    result,
                    obs_dict,
                    reward_dict,
                    terminated_dict,
                    truncated_dict,
                    info_dict,
                    ready_ids,
                    crashed_ids,
                    terminal_ids,
                )
                seen_result_workers.add(result.get("worker_id"))
            self._consume_pending_reset_obs(
                obs_dict,
                info_dict,
                ready_ids,
                action_worker_ids=action_worker_ids,
                terminal_worker_ids=terminal_ids,
                crashed_worker_ids=crashed_ids,
            )
            if count_ready() >= min_ready:
                break
            if time.time() - start_time > timeout:
                logger.warning("step timeout: got %s ready results, need %s", count_ready(), min_ready)
                break
            time.sleep(0.001)

        self._process_queues()
        for result in self._drain_step_results(
            process_queues=False,
            exclude_workers=seen_result_workers,
        ):
            self._apply_step_result(
                result,
                obs_dict,
                reward_dict,
                terminated_dict,
                truncated_dict,
                info_dict,
                ready_ids,
                crashed_ids,
                terminal_ids,
            )
            seen_result_workers.add(result.get("worker_id"))
        self._consume_pending_reset_obs(
            obs_dict,
            info_dict,
            ready_ids,
            action_worker_ids=action_worker_ids,
            terminal_worker_ids=terminal_ids,
            crashed_worker_ids=crashed_ids,
        )

        return obs_dict, reward_dict, terminated_dict, truncated_dict, info_dict

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
        min_ready: int = 1,
        timeout: float = 60.0,
    ) -> Tuple[Dict[int, Any], Dict[int, Dict[str, Any]]]:
        del seed, options
        with self._runtime_lock:
            if self._state == EnvPoolState.READY:
                self._state = EnvPoolState.RUNNING
        return self._collect_reset_observations(min_ready=min_ready, timeout=timeout)

    @property
    def single_observation_space(self):
        with self._runtime_lock:
            return self._observation_space

    @property
    def single_action_space(self):
        with self._runtime_lock:
            return self._action_space

    def __len__(self) -> int:
        return self.num_envs

    def get_stats(self) -> Dict[str, Any]:
        with self._runtime_lock:
            active_count = 0
            ready_count = 0
            stepping_count = 0
            resetting_count = 0
            error_count = 0
            recovering_count = 0
            idle_count = 0
            stopped_count = 0
            for info in self._worker_infos.values():
                worker = self._workers.get(info.worker_id)
                is_alive = worker.is_alive() if worker else False
                if is_alive and info.state not in (WorkerState.ERROR, WorkerState.STOPPED):
                    active_count += 1

                if info.state == WorkerState.WAITING_ACTION:
                    ready_count += 1
                elif info.state == WorkerState.STEPPING:
                    stepping_count += 1
                elif info.state == WorkerState.RESETTING:
                    resetting_count += 1
                elif info.state == WorkerState.ERROR:
                    error_count += 1
                elif info.state in (WorkerState.RECOVERING, WorkerState.STARTING):
                    recovering_count += 1
                elif info.state == WorkerState.IDLE:
                    idle_count += 1
                elif info.state == WorkerState.STOPPED:
                    stopped_count += 1

            return {
                "pool_state": self._state.name,
                "num_envs": self.num_envs,
                "total_episodes": self._total_episodes,
                "total_steps": self._total_steps,
                "active_workers": active_count,
                "ready_workers": ready_count,
                "stepping_workers": stepping_count,
                "resetting_workers": resetting_count,
                "error_workers": error_count,
                "recovering_workers": recovering_count,
                "idle_workers": idle_count,
                "stopped_workers": stopped_count,
                "pending_reset_observations": len(self._pending_reset_obs),
                "queued_step_results": len(self._queued_step_results),
                "worker_details": {
                    wid: {
                        "state": info.state.name,
                        "host": info.carla_host,
                        "port": info.carla_port,
                        "tm_port": info.traffic_manager_port,
                        "epoch": info.epoch,
                        "restart_count": info.restart_count,
                        "episode_count": info.episode_count,
                        "episodes_since_restart": info.episodes_since_restart,
                        "next_recycle_episode": info.next_recycle_episode,
                        "planned_recycle_count": info.planned_recycle_count,
                        "total_steps": info.total_steps,
                        "current_episode_step": info.current_episode_step,
                        "current_episode_reward": info.current_episode_reward,
                        "route_id": info.route_id,
                        "scenario_name": info.scenario_name,
                        "scenario_instance_name": info.scenario_instance_name,
                        "town": info.town,
                        "last_error": info.last_error,
                    }
                    for wid, info in self._worker_infos.items()
                },
            }

    @property
    def action_space(self):
        with self._runtime_lock:
            return self._action_space

    @property
    def observation_space(self):
        with self._runtime_lock:
            return self._observation_space

    def close(self) -> None:
        with self._runtime_lock:
            if self._state == EnvPoolState.CLOSED:
                return
            if self._state == EnvPoolState.CLOSING:
                return
            self._state = EnvPoolState.CLOSING
            health_worker = self._health_worker
            recovery_thread = self._recovery_thread
        health_worker.stop(timeout=5.0)

        self._recovery_stop.set()
        if recovery_thread:
            try:
                self._recovery_queue.put(None)
            except Exception:
                pass
            recovery_thread.join(timeout=5.0)

        with self._runtime_lock:
            control_items = list(self._control_queues.items())
            workers = list(self._workers.values())
            queue_objs = [queue_obj for queue_dict in (self._event_queues, self._action_queues, self._control_queues) for queue_obj in queue_dict.values()]
            rgb_buffers = list(self._rgb_shm_buffers.values())
            self._rgb_shm_buffers = {}
            server_manager = self._server_manager
            adaptive_manager = self._adaptive_manager
            adaptive_shared_stats = self._adaptive_shared_stats
            adaptive_shared_lock = self._adaptive_shared_lock
            if self.manage_servers:
                self._server_manager = None
            self._recovery_thread = None

        for worker_id, control_queue in control_items:
            try:
                control_queue.put({"type": "stop"})
            except Exception:
                pass

        for worker in workers:
            worker.join(timeout=5.0)
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=5.0)

        for queue_obj in queue_objs:
            try:
                queue_obj.close()
                queue_obj.cancel_join_thread()
            except Exception:
                pass

        for buffer in rgb_buffers:
            try:
                buffer.close()
            except Exception as close_error:
                logger.warning("Failed to close RGB shm buffer %s: %s", buffer.path, close_error)
            try:
                buffer.unlink()
            except Exception as unlink_error:
                logger.warning("Failed to unlink RGB shm buffer %s: %s", buffer.path, unlink_error)

        if self.manage_servers and server_manager is not None:
            logger.info("Stopping CARLA servers...")
            server_manager.stop_all()

        if adaptive_manager is not None:
            # Keep the shared proxy objects alive until every worker has
            # finished env.close(). Dropping the parent references earlier
            # can let multiprocessing.Manager dispose the dict/lock while
            # worker cleanup is still updating adaptive sampler stats.
            adaptive_shared_stats = None
            adaptive_shared_lock = None
            self._adaptive_manager = None
            self._adaptive_shared_stats = None
            self._adaptive_shared_lock = None
            try:
                adaptive_manager.shutdown()
            except Exception:
                pass

        with self._runtime_lock:
            self._state = EnvPoolState.CLOSED
        logger.info("CARLAEnvPool closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
