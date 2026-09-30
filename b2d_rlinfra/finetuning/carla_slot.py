"""Single-environment slot abstraction for rl finetune collectors."""

from __future__ import annotations

import copy
import logging
import multiprocessing as mp
import os
import queue
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from b2d_rlinfra.simulation.runners.carla_env_pool_utils import parse_connection_config
from b2d_rlinfra.simulation.runners.shm_rgb_buffer import ShmRgbBuffer
from b2d_rlinfra.finetuning.sim_actor import SimActorProcess, sim_actor_entry

logger = logging.getLogger("RLFinetune.CarlaSlot")
_SHM_PREFIX_ENV_KEY = "B2D_RL_FINETUNE_SHM_PREFIX"


def _new_rgb_shm_prefix() -> str:
    base_prefix = os.environ.get(_SHM_PREFIX_ENV_KEY, "rl_finetune_rgb")
    return "{}_{}".format(base_prefix, uuid.uuid4().hex[:12])


class ShmIpcError(RuntimeError):
    pass


class SlotCrashed(RuntimeError):
    def __init__(self, reason: str, info: Optional[Dict[str, Any]] = None):
        super().__init__(reason)
        self.reason = reason
        self.info = dict(info or {})


class CarlaSlot:
    """A single-env, restartable CARLA slot.

    It owns one SimActor process and, when configured, one RGB shared-memory
    buffer. ``step()`` returns the terminal/truncation observation for the
    current action; non-recycle reset observations are cached until
    ``consume_reset_obs()`` is called.
    """

    def __init__(
        self,
        *,
        collector_id: int,
        env_config: Dict[str, Any],
        algorithm_config: Optional[Dict[str, Any]] = None,
        slot_config: Optional[Dict[str, Any]] = None,
        env_fn: Optional[Any] = None,
        connection_plan: Optional[Dict[str, Any]] = None,
    ):
        self.collector_id = int(collector_id)
        self.config = copy.deepcopy(env_config)
        if algorithm_config:
            self.config["algorithm"] = dict(algorithm_config)
        self.slot_config = dict(slot_config or {})
        self.auto_reset = bool(self.slot_config.get("auto_reset", True))
        self.use_rgb_shm = bool(self.slot_config.get("use_rgb_shm", True))
        self.manage_servers = bool(self.slot_config.get("manage_servers", True))
        self.server_wait_timeout = float(self.slot_config.get("server_wait_timeout", 120.0))
        self.event_timeout = float(self.slot_config.get("event_timeout", 120.0))
        self.max_episode_steps = int(
            self.config.get("environment", {}).get("max_episode_steps", self.slot_config.get("max_episode_steps", 10000))
        )
        self.result_dir = str(self.config.get("environment", {}).get("result_dir", "./results"))
        self._validate_rgb_server_config()
        self.env_fn = env_fn
        if self.env_fn is None:
            from b2d_rlinfra.learning.training.runner_utils import make_env

            self.env_fn = make_env

        carla_cfg = self.config.get("carla", {}) if isinstance(self.config, dict) else {}
        if connection_plan is not None:
            plan = dict(connection_plan)
            self.host = str(plan["host"])
            self.carla_port = int(plan["port"])
            self.traffic_manager_port = int(plan["traffic_manager_port"])
            self.traffic_manager_seed = int(plan.get("traffic_manager_seed", 0))
            self.gpu_id = int(plan["carla_gpu_id"])
        else:
            configured_num_envs = int(carla_cfg.get("num_envs", self.collector_id + 1) or (self.collector_id + 1))
            connection_slots = max(configured_num_envs, self.collector_id + 1)
            conn = parse_connection_config(
                self.config,
                num_envs=connection_slots,
                carla_ports_override=None,
                tm_ports_override=None,
                gpu_ids_override=None,
                hosts_override=None,
            )
            self.host = conn["hosts"][self.collector_id]
            self.carla_port = int(conn["ports"][self.collector_id])
            self.traffic_manager_port = int(conn["tm_ports"][self.collector_id])
            self.traffic_manager_seed = int(conn["tm_seeds"][self.collector_id])
            self.gpu_id = int(conn["gpu_ids"][self.collector_id])

        self._server_manager = None
        self._worker_epoch = 0
        self._process: Optional[mp.Process] = None
        self._event_queue: Optional[mp.Queue] = None
        self._action_queue: Optional[mp.Queue] = None
        self._control_queue: Optional[mp.Queue] = None
        self._reset_cache: Optional[Tuple[Any, Dict[str, Any]]] = None
        self._rgb_shm_buffer: Optional[ShmRgbBuffer] = None
        self._rgb_obs_key = "rgb"
        self._rgb_shape: Optional[Tuple[int, ...]] = None
        self._rgb_dtype: Optional[np.dtype] = None
        self._rgb_prefix: Optional[str] = None

    def _validate_rgb_server_config(self) -> None:
        obs_cfg = self.config.get("observation_space", {}) if isinstance(self.config, dict) else {}
        rgb_cfg = obs_cfg.get("rgb", {}) if isinstance(obs_cfg, dict) else {}
        if not bool(rgb_cfg.get("enable", False)):
            return
        carla_cfg = self.config.get("carla", {}) if isinstance(self.config, dict) else {}
        if bool(carla_cfg.get("no_rendering_mode", True)):
            raise ValueError("RGB observation requires env.carla.no_rendering_mode=false")
        if bool(carla_cfg.get("null_rhi", True)):
            raise ValueError("RGB observation requires env.carla.null_rhi=false")

    def reset(self) -> Tuple[Any, Dict[str, Any]]:
        return self.restart(reason="initial")

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        if self._process is None or not self._process.is_alive():
            return self._crash_result("sim_actor_not_alive")
        assert self._action_queue is not None
        self._action_queue.put({"type": "action", "action": action})
        while True:
            event = self._get_event()
            event_type = event.get("type")
            if event_type == "reset_obs":
                self._reset_cache = self._event_obs_info(event)
                continue
            if event_type != "step_result":
                continue
            if event.get("crashed", False) or (event.get("info") or {}).get("crashed", False):
                obs = self._attach_rgb_or_crash(event)
                info = dict(event.get("info") or {})
                info["crashed"] = True
                return obs, float(event.get("reward", 0.0)), True, False, info
            obs = self._attach_rgb_or_crash(event)
            info = dict(event.get("info") or {})
            info["planned_recycle"] = bool(event.get("planned_recycle", info.get("planned_recycle", False)))
            return (
                obs,
                float(event.get("reward", 0.0)),
                bool(event.get("terminated", False)),
                bool(event.get("truncated", False)),
                info,
            )

    def consume_reset_obs(self) -> Tuple[Any, Dict[str, Any]]:
        if self._reset_cache is None:
            deadline = time.time() + self.event_timeout
            while time.time() < deadline:
                event = self._get_event(timeout=max(0.1, deadline - time.time()))
                if event.get("type") == "reset_obs":
                    self._reset_cache = self._event_obs_info(event)
                    break
                if event.get("type") == "step_result" and event.get("crashed", False):
                    info = dict(event.get("info") or {})
                    reason = info.get("crash_reason") or info.get("crash_type") or "crashed"
                    raise SlotCrashed(f"SimActor crashed before reset obs: {reason}", info)
        if self._reset_cache is None:
            raise RuntimeError("reset observation is not available")
        obs, info = self._reset_cache
        self._reset_cache = None
        info = dict(info or {})
        info["episode_start"] = True
        info["from_reset"] = True
        return obs, info

    def restart(self, reason: Optional[str] = None) -> Tuple[Any, Dict[str, Any]]:
        self._stop_actor()
        if self.manage_servers:
            self._restart_server()
        self._replace_rgb_shm()
        self._start_actor()
        ready = self._get_event()
        if ready.get("type") == "step_result" and ready.get("crashed", False):
            raise self._slot_crashed_from_event(ready, "SimActor crashed during restart")
        if ready.get("type") != "ready":
            raise RuntimeError(f"expected SimActor ready event, got {ready.get('type')!r}")
        reset_event = self._get_event()
        if reset_event.get("type") == "step_result" and reset_event.get("crashed", False):
            raise self._slot_crashed_from_event(reset_event, "SimActor crashed before initial reset obs")
        if reset_event.get("type") != "reset_obs":
            raise RuntimeError(f"expected SimActor reset_obs event, got {reset_event.get('type')!r}")
        self._reset_cache = None
        obs, info = self._event_obs_info(reset_event)
        info = dict(info or {})
        info["episode_start"] = True
        info["from_reset"] = True
        return obs, info

    def close(self) -> None:
        try:
            self._stop_actor()
        except Exception:
            logger.exception("Failed to stop SimActor")
        if self._server_manager is not None:
            try:
                self._server_manager.stop_server(self.host, self.carla_port)
            except Exception:
                logger.exception("Failed to stop CARLA server")
        try:
            self._cleanup_rgb_shm()
        except Exception:
            logger.exception("Failed to cleanup RGB shm")

    def _restart_server(self) -> None:
        carla_root = (
            self.slot_config.get("carla_root")
            or self.config.get("carla", {}).get("root")
            or os.environ.get("CARLA_ROOT")
        )
        if not carla_root or not Path(str(carla_root)).exists():
            raise ValueError(
                "carla_root is required when rl_finetune.sim_actor.manage_servers=true. "
                "Set CARLA_ROOT or rl_finetune.sim_actor.carla_root to a valid CARLA directory."
            )
        from b2d_rlinfra.simulation.runners.carla_server_manager import CARLAServerManager

        if self._server_manager is None:
            carla_cfg = dict(self.config.get("carla", {}) or {})
            # Do NOT pass default_quality here. CARLAServerManager defaults
            # to "Default" which means the -quality-level flag is omitted
            # from the CarlaUE4 command line. Some CARLA builds are
            # incompatible with that flag and will fail to start.
            self._server_manager = CARLAServerManager(
                carla_root=carla_root,
                default_fps=int(carla_cfg.get("server_fps", carla_cfg.get("frequency_hz", 10))),
            )
        # Forward carla config as kwargs so server-critical flags like
        # null_rhi, render_offscreen, no_sound etc. reach start_server().
        # Excluded keys:
        #   host/port/gpu_id  – managed by carla_slot per-collector
        #   traffic_manager_* – irrelevant to the CARLA server process
        #   quality_level     – some CARLA builds crash with -quality-level;
        #                       omitting it lets CARLAServerManager use its
        #                       default ("Default" = flag not passed)
        kwargs = dict(self.config.get("carla", {}) or {})
        kwargs.pop("host", None)
        kwargs.pop("port", None)
        kwargs.pop("traffic_manager_port", None)
        kwargs.pop("gpu_id", None)
        kwargs.pop("quality_level", None)
        success = self._server_manager.restart_server(self.host, self.carla_port, gpu_id=self.gpu_id, **kwargs)
        if not success:
            raise SlotCrashed(
                f"Failed to start CARLA server at {self.host}:{self.carla_port} (gpu_id={self.gpu_id})",
                {"crash_reason": "carla_server_start_failed", "crash_type": "server_startup"},
            )

    def _start_actor(self) -> None:
        ctx = mp.get_context(str(self.slot_config.get("start_method", "spawn")))
        self._event_queue = ctx.Queue(maxsize=int(self.slot_config.get("queue_maxsize", 64)))
        self._action_queue = ctx.Queue(maxsize=1)
        self._control_queue = ctx.Queue(maxsize=4)
        actor = SimActorProcess(
            worker_id=self.collector_id,
            worker_epoch=self._worker_epoch,
            env_fn=self.env_fn,
            config=self.config,
            host=self.host,
            carla_port=self.carla_port,
            traffic_manager_port=self.traffic_manager_port,
            traffic_manager_seed=self.traffic_manager_seed,
            max_episode_steps=self.max_episode_steps,
            auto_reset=self.auto_reset,
            result_dir=self.result_dir,
            rgb_shm_meta=self._rgb_shm_buffer.metadata(self._rgb_obs_key) if self._rgb_shm_buffer else None,
            server_wait_timeout=self.server_wait_timeout,
        )
        self._process = ctx.Process(
            target=sim_actor_entry,
            args=(actor, self._event_queue, self._action_queue, self._control_queue),
            daemon=True,
        )
        self._process.start()
        self._worker_epoch += 1

    def _stop_actor(self) -> None:
        if self._process is None:
            return
        if getattr(self._process, "_popen", None) is None:
            self._process = None
            self._reset_cache = None
            return
        if self._process.is_alive() and self._control_queue is not None:
            try:
                self._control_queue.put({"type": "stop"}, timeout=0.5)
            except Exception:
                pass
        self._process.join(timeout=float(self.slot_config.get("actor_stop_timeout", 5.0)))
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=5.0)
        self._process = None
        self._reset_cache = None

    def _replace_rgb_shm(self) -> None:
        self._cleanup_rgb_shm()
        if not self.use_rgb_shm:
            return
        obs_cfg = self.config.get("observation_space", {}) if isinstance(self.config, dict) else {}
        rgb_cfg = obs_cfg.get("rgb", {}) if isinstance(obs_cfg, dict) else {}
        if not bool(rgb_cfg.get("enable", False)):
            return
        from b2d_rlinfra.environment.spaces import build_observation_space_dict, resolve_rgb_obs_key

        self._rgb_obs_key = resolve_rgb_obs_key(rgb_cfg)
        rgb_space = build_observation_space_dict(self.config).spaces.get(self._rgb_obs_key)
        if rgb_space is None:
            raise ValueError(f"RGB SHM enabled but observation space has no key {self._rgb_obs_key!r}")
        self._rgb_shape = tuple(int(dim) for dim in rgb_space.shape)
        self._rgb_dtype = np.dtype(rgb_space.dtype)
        self._rgb_prefix = _new_rgb_shm_prefix()
        self._rgb_shm_buffer = ShmRgbBuffer(
            self.collector_id,
            rgb_shape=self._rgb_shape,
            rgb_dtype=self._rgb_dtype,
            prefix=self._rgb_prefix,
            num_slots=int(self.slot_config.get("rgb_num_slots", 2)),
            create=True,
        )

    def _cleanup_rgb_shm(self) -> None:
        if self._rgb_shm_buffer is None:
            return
        try:
            self._rgb_shm_buffer.close()
        finally:
            try:
                self._rgb_shm_buffer.unlink()
            except Exception:
                logger.exception("Failed to unlink RGB shm")
            self._rgb_shm_buffer = None

    def _attach_rgb(self, event: Dict[str, Any]) -> Any:
        obs = event.get("observation")
        if self._rgb_shm_buffer is None or "rgb_shm_slot" not in event:
            return obs
        if obs is None:
            obs = {}
        if not isinstance(obs, dict):
            raise ShmIpcError("RGB SHM event cannot attach RGB to non-dict observation")
        try:
            rgb = self._rgb_shm_buffer.read_rgb(event["rgb_shm_slot"], expected_seq=event["rgb_shm_seq"])
        except Exception as exc:
            raise ShmIpcError(str(exc)) from exc
        merged = dict(obs)
        merged[self._rgb_obs_key] = rgb
        return merged

    def _attach_rgb_or_crash(self, event: Dict[str, Any]) -> Any:
        try:
            return self._attach_rgb(event)
        except ShmIpcError as exc:
            raise SlotCrashed(
                "RGB SHM IPC error",
                {
                    "crashed": True,
                    "crash_reason": "rgb_shm_ipc_error",
                    "crash_type": exc.__class__.__name__,
                    "crash_detail": str(exc),
                },
            ) from exc

    def _event_obs_info(self, event: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        return self._attach_rgb_or_crash(event), dict(event.get("info") or {})

    def _slot_crashed_from_event(self, event: Dict[str, Any], prefix: str) -> SlotCrashed:
        info = dict(event.get("info") or {})
        reason = info.get("crash_reason") or info.get("crash_type") or event.get("crash_reason") or "crashed"
        info["crashed"] = True
        return SlotCrashed(f"{prefix}: {reason}", info)

    def _get_event(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        if self._event_queue is None:
            raise RuntimeError("SimActor event queue is not initialized")
        try:
            return self._event_queue.get(timeout=self.event_timeout if timeout is None else timeout)
        except queue.Empty as exc:
            raise TimeoutError("Timed out waiting for SimActor event") from exc

    def _crash_result(self, reason: str) -> Tuple[None, float, bool, bool, Dict[str, Any]]:
        return None, 0.0, True, False, {"crashed": True, "crash_reason": reason, "crash_type": reason}


class FakeCarlaSlot:
    """Deterministic fake slot used by tests."""

    def __init__(self, *, collector_id: int, slot_config: Optional[Dict[str, Any]] = None, **_: Any):
        self.collector_id = int(collector_id)
        cfg = dict(slot_config or {})
        self.episode_length = int(cfg.get("episode_length", 4))
        self.obs_dim = int(cfg.get("obs_dim", 4))
        self.obs_type = str(cfg.get("obs_type", "vector")).lower()
        self.rgb_key = str(cfg.get("rgb_key", "rgb"))
        self.rgb_shape = tuple(int(dim) for dim in cfg.get("rgb_shape", (6, 32, 64, 3)))
        self.scalars_dim = int(cfg.get("scalars_dim", 12))
        self.truncate_every = int(cfg.get("truncate_every", 0))
        self.crash_every = int(cfg.get("crash_every", 0))
        self.crash_once = bool(cfg.get("crash_once", False))
        self.crash_at_step = int(cfg.get("crash_at_step", 2))
        self.planned_recycle_every = int(cfg.get("planned_recycle_every", 0))
        self._episode_id = 0
        self._step = 0
        self._reset_cache: Optional[Tuple[Any, Dict[str, Any]]] = None
        self.reset_calls = 0
        self.restart_calls = 0
        self.consume_reset_calls = 0
        self._crash_emitted = False

    def reset(self) -> Tuple[Any, Dict[str, Any]]:
        self.reset_calls += 1
        self._episode_id += 1
        self._step = 0
        return self._obs(), self._reset_info("initial")

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        self._step += 1
        should_crash_once = self.crash_once and not self._crash_emitted and self._step == self.crash_at_step
        should_crash_periodic = (
            self.crash_every > 0
            and self._episode_id % self.crash_every == 0
            and self._step == self.crash_at_step
        )
        if should_crash_once or should_crash_periodic:
            self._crash_emitted = True
            return None, 0.0, True, False, {
                "crashed": True,
                "crash_reason": "fake_crash",
                "crash_type": "fake_crash",
                "crash_detail": "synthetic fake slot crash",
            }
        terminated = self._step >= self.episode_length
        truncated = False
        if self.truncate_every > 0 and self._episode_id % self.truncate_every == 0 and terminated:
            terminated = False
            truncated = True
        obs = self._obs()
        reward = float(1.0 - 0.05 * np.linalg.norm(np.asarray(action, dtype=np.float32)))
        info: Dict[str, Any] = {
            "route_id": f"fake_route_{self._episode_id}",
            "scenario_name": "fake_scenario",
            "town": "TownFake",
            "crashed": False,
        }
        if (terminated or truncated) and self.planned_recycle_every > 0 and self._episode_id % self.planned_recycle_every == 0:
            info["planned_recycle"] = True
            info["planned_recycle_reason"] = "fake_planned_recycle"
        if terminated or truncated:
            info["episode_ended"] = True
            info["end_reason"] = "truncated" if truncated else "terminated"
            if not info.get("planned_recycle"):
                self._episode_id += 1
                self._step = 0
                self._reset_cache = (self._obs(), self._reset_info("truncated" if truncated else "auto"))
        return obs, reward, terminated, truncated, info

    def consume_reset_obs(self) -> Tuple[Any, Dict[str, Any]]:
        self.consume_reset_calls += 1
        if self._reset_cache is None:
            raise RuntimeError("fake reset cache is empty")
        obs, info = self._reset_cache
        self._reset_cache = None
        return obs, info

    def restart(self, reason: Optional[str] = None) -> Tuple[Any, Dict[str, Any]]:
        self.restart_calls += 1
        self._episode_id += 1
        self._step = 0
        self._reset_cache = None
        return self._obs(), self._reset_info(reason or "restart")

    def close(self) -> None:
        self._reset_cache = None

    def _obs(self) -> Dict[str, np.ndarray]:
        if self.obs_type == "rgb":
            fill = int((self._episode_id * 17 + self._step * 7 + self.collector_id) % 256)
            rgb = np.full(self.rgb_shape, fill, dtype=np.uint8)
            scalars = (
                np.arange(self.scalars_dim, dtype=np.float32)
                + float(self._episode_id)
                + float(self._step) * 0.1
            )
            return {self.rgb_key: rgb, "scalars": scalars}
        base = np.arange(self.obs_dim, dtype=np.float32)
        return {"vector": base + float(self._episode_id) + float(self._step) * 0.1}

    def _reset_info(self, reason: str) -> Dict[str, Any]:
        return {
            "episode_start": True,
            "from_reset": True,
            "reset_reason": reason,
            "route_id": f"fake_route_{self._episode_id}",
            "scenario_name": "fake_scenario",
            "town": "TownFake",
        }


def make_slot(
    *,
    collector_id: int,
    env_config: Dict[str, Any],
    algorithm_config: Optional[Dict[str, Any]],
    rl_config: Dict[str, Any],
    collector_plan: Optional[Dict[str, Any]] = None,
) -> Any:
    slot_cfg = dict(rl_config.get("sim_actor", {}) or {})
    env_type = str(rl_config.get("env_type", slot_cfg.get("env_type", "carla"))).lower()
    if env_type == "fake":
        fake_cfg = dict(slot_cfg.get("fake", {}) or {})
        fake_cfg.setdefault("obs_dim", rl_config.get("mock_obs_dim", 4))
        return FakeCarlaSlot(collector_id=collector_id, slot_config=fake_cfg)
    return CarlaSlot(
        collector_id=collector_id,
        env_config=env_config,
        algorithm_config=algorithm_config,
        slot_config=slot_cfg,
        connection_plan=collector_plan,
    )
