"""Single-env simulator actor used by :mod:`carla_slot`.

The actor intentionally mirrors the small subset of ``CARLAEnvPool`` semantics
needed by rl finetune collectors: initial reset, step result, terminal
auto-reset, planned recycle stop, crash event, and optional RGB shared memory.
"""

from __future__ import annotations

import copy
import logging
import multiprocessing as mp
import os
import queue
import random
import socket
import time
import traceback
from typing import Any, Dict, Optional, Tuple

import numpy as np

from b2d_rlinfra.simulation.runners.carla_env_pool_utils import write_worker_crash_file
from b2d_rlinfra.simulation.crash_utils import refine_crash_payload
from b2d_rlinfra.simulation.runners.shm_rgb_buffer import ShmRgbBuffer

logger = logging.getLogger("RLFinetune.SimActor")
_ENV_RESULTS_DIRNAME = "env_results"


def _server_available(host: str, port: int, timeout: float = 2.0) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        return sock.connect_ex((host, int(port))) == 0
    finally:
        sock.close()


class SimActorProcess:
    def __init__(
        self,
        *,
        worker_id: int,
        worker_epoch: int,
        env_fn: Any,
        config: Dict[str, Any],
        host: str,
        carla_port: int,
        traffic_manager_port: int,
        traffic_manager_seed: int,
        max_episode_steps: int,
        auto_reset: bool,
        result_dir: str,
        rgb_shm_meta: Optional[Dict[str, Any]] = None,
        server_wait_timeout: float = 120.0,
    ):
        self.worker_id = int(worker_id)
        self.worker_epoch = int(worker_epoch)
        self.env_fn = env_fn
        self.config = copy.deepcopy(config)
        self.host = host
        self.carla_port = int(carla_port)
        self.traffic_manager_port = int(traffic_manager_port)
        self.traffic_manager_seed = int(traffic_manager_seed)
        self.max_episode_steps = int(max_episode_steps)
        self.auto_reset = bool(auto_reset)
        self.result_dir = os.path.abspath(result_dir)
        self.rgb_shm_meta = dict(rgb_shm_meta or {})
        self.server_wait_timeout = float(server_wait_timeout)

    def __call__(self, event_queue: mp.Queue, action_queue: mp.Queue, control_queue: mp.Queue) -> None:
        env = None
        rgb_shm_buffer = None
        rgb_obs_key = "rgb"
        rgb_num_slots = 2
        next_rgb_slot = 0
        rgb_seq = 0
        episode_step = 0
        episode_reward = 0.0
        exit_reason = "normal_exit"
        env_cfg = self.config.get("environment", {}) if isinstance(self.config, dict) else {}
        recycle_interval = max(0, int(env_cfg.get("worker_recycle_interval_episodes", 0) or 0))
        recycle_jitter = max(0, int(env_cfg.get("worker_recycle_jitter_episodes", 0) or 0))
        recycle_target = 0
        if recycle_interval > 0:
            recycle_target = recycle_interval + (random.randint(0, recycle_jitter) if recycle_jitter > 0 else 0)
        episodes_since_restart = 0

        if self.rgb_shm_meta.get("enabled"):
            rgb_obs_key = str(self.rgb_shm_meta.get("rgb_obs_key", "rgb"))
            rgb_num_slots = int(self.rgb_shm_meta.get("num_slots", 2))
            rgb_shm_buffer = ShmRgbBuffer(
                worker_id=int(self.rgb_shm_meta.get("worker_id", self.worker_id)),
                rgb_shape=tuple(self.rgb_shm_meta["rgb_shape"]),
                rgb_dtype=np.dtype(self.rgb_shm_meta["rgb_dtype"]),
                prefix=str(self.rgb_shm_meta["prefix"]),
                num_slots=rgb_num_slots,
                create=False,
            )

        def wait_for_server(host: str, port: int, timeout: float) -> bool:
            start_time = time.time()
            while time.time() - start_time < timeout:
                if _server_available(host, port):
                    # Match pool behavior: add a short settle delay once port is up.
                    time.sleep(3.0)
                    return True
                time.sleep(2.0)
            return False

        def create_env_with_retry(worker_config, max_retries: int = 3):
            last_error = None
            for attempt in range(max_retries):
                if not _server_available(self.host, self.carla_port):
                    # Align with pool: server-level connectivity failure should
                    # fail fast and let the outer restart path handle recovery.
                    raise RuntimeError(
                        f"server not available (attempt {attempt + 1}) at {self.host}:{self.carla_port}"
                    )
                try:
                    return self.env_fn(
                        worker_config, self.worker_id,
                        self.carla_port, self.traffic_manager_port,
                    )
                except Exception as exc:
                    last_error = exc
                    error_msg = str(exc).lower()
                    is_connection_error = any(
                        kw in error_msg for kw in ("connection", "timeout", "socket", "rpc", "not ready")
                    )
                    should_fast_fail = is_connection_error and not any(
                        kw in error_msg for kw in ("not ready", "server not ready", "timeout waiting for carla server")
                    )
                    logger.warning(
                        "SimActor %s env creation attempt %s/%s failed: %s",
                        self.worker_id, attempt + 1, max_retries, exc,
                    )
                    if should_fast_fail:
                        # Align with pool: connection-type creation failures
                        # should trigger immediate outer restart rather than local retries.
                        raise RuntimeError(f"env creation connection error: {exc}") from exc
                    if attempt < max_retries - 1:
                        wait_for_server(self.host, self.carla_port, self.server_wait_timeout)
            raise RuntimeError(f"Failed to create env after {max_retries} attempts: {last_error}")

        def put_event(msg: Dict[str, Any]) -> None:
            msg.setdefault("worker_id", self.worker_id)
            msg.setdefault("worker_epoch", self.worker_epoch)
            event_queue.put(msg)

        def prepare_obs(obs: Any) -> Tuple[Any, Optional[int], Optional[int]]:
            nonlocal next_rgb_slot, rgb_seq
            if rgb_shm_buffer is None:
                return obs, None, None
            if not isinstance(obs, dict):
                raise TypeError("RGB SHM requires dict observations")
            if rgb_obs_key not in obs:
                raise KeyError(f"RGB SHM enabled but observation has no key {rgb_obs_key!r}")
            slot = next_rgb_slot
            next_rgb_slot = (next_rgb_slot + 1) % rgb_num_slots
            seq = rgb_seq
            rgb_seq += 1
            rgb_shm_buffer.write_rgb(slot, obs[rgb_obs_key], seq=seq)
            queue_obs = dict(obs)
            queue_obs.pop(rgb_obs_key, None)
            return queue_obs, slot, seq

        def reset_event(obs: Any, info: Optional[Dict[str, Any]], reason: str) -> Dict[str, Any]:
            queue_obs, rgb_slot, seq = prepare_obs(obs)
            reset_info = dict(info or {})
            reset_info["episode_start"] = True
            reset_info["from_reset"] = True
            reset_info["reset_reason"] = reason
            event = {"type": "reset_obs", "observation": queue_obs, "info": reset_info}
            if rgb_slot is not None:
                event["rgb_shm_slot"] = rgb_slot
                event["rgb_shm_seq"] = seq
            return event

        def crash_event(reason: str, exc: BaseException, tb: str, source_info: Optional[Dict[str, Any]] = None) -> None:
            crash_reason, crash_type, crash_detail = refine_crash_payload(
                reason,
                exc.__class__.__name__,
                str(exc),
            )
            info = dict(source_info or {})
            info.update(
                {
                    "crashed": True,
                    "crash_reason": crash_reason,
                    "crash_type": crash_type,
                    "crash_detail": crash_detail,
                    "error": crash_detail,
                    "traceback": tb,
                    "episode_ended": True,
                    "end_reason": "crashed",
                }
            )
            try:
                write_worker_crash_file(
                    self.result_dir,
                    {
                        "worker_id": self.worker_id,
                        "worker_epoch": self.worker_epoch,
                        "episode_step": episode_step,
                        "episode_reward": episode_reward,
                        "crash_reason": crash_reason,
                        "crash_type": crash_type,
                        "crash_detail": crash_detail,
                        "traceback": tb,
                    },
                )
            except Exception:
                logger.exception("Failed to write crash record")
            put_event(
                {
                    "type": "step_result",
                    "observation": None,
                    "reward": 0.0,
                    "terminated": True,
                    "truncated": False,
                    "info": info,
                    "episode_step": episode_step,
                    "episode_reward": episode_reward,
                    "crashed": True,
                }
            )

        try:
            if not wait_for_server(self.host, self.carla_port, self.server_wait_timeout):
                raise RuntimeError(f"Timeout waiting for CARLA server at {self.host}:{self.carla_port}")

            worker_config = copy.deepcopy(self.config)
            worker_env_cfg = worker_config.setdefault("environment", {})
            worker_env_cfg["host"] = self.host
            worker_env_cfg["port"] = self.carla_port
            worker_env_cfg["traffic_manager_port"] = self.traffic_manager_port
            worker_env_cfg["traffic_manager_seed"] = self.traffic_manager_seed
            worker_env_cfg["result_dir"] = os.path.join(self.result_dir, _ENV_RESULTS_DIRNAME)
            if "carla" in worker_config:
                worker_config["carla"]["host"] = self.host
                worker_config["carla"]["port"] = self.carla_port
                worker_config["carla"]["traffic_manager_port"] = self.traffic_manager_port
                worker_config["carla"]["traffic_manager_seed"] = self.traffic_manager_seed

            env = create_env_with_retry(worker_config)
            put_event(
                {
                    "type": "ready",
                    "action_space": getattr(env, "action_space", None),
                    "observation_space": getattr(env, "observation_space", None),
                }
            )
            obs, info = env.reset()
            put_event(reset_event(obs, info, "initial"))

            while True:
                try:
                    cmd = control_queue.get_nowait()
                    if cmd.get("type") == "stop":
                        exit_reason = "stop"
                        break
                except queue.Empty:
                    pass

                try:
                    action_msg = action_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                try:
                    obs, reward, terminated, truncated, info = env.step(action_msg["action"])
                    info = dict(info or {})
                    episode_step += 1
                    episode_reward += float(reward)
                    if (not terminated) and episode_step >= self.max_episode_steps:
                        truncated = True
                        info["TimeLimit.truncated"] = True
                    planned_recycle = bool(info.get("planned_recycle", False))
                    if (terminated or truncated) and recycle_target > 0:
                        episodes_since_restart += 1
                        if episodes_since_restart >= recycle_target:
                            planned_recycle = True
                            info["planned_recycle"] = True
                            info["planned_recycle_reason"] = "planned_episode_recycle"
                            info["episodes_since_restart"] = episodes_since_restart
                            info["worker_recycle_target_episode"] = recycle_target
                    queue_obs, rgb_slot, seq = prepare_obs(obs)
                    event = {
                        "type": "step_result",
                        "observation": queue_obs,
                        "reward": float(reward),
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                        "info": info,
                        "episode_step": episode_step,
                        "episode_reward": episode_reward,
                        "crashed": bool(info.get("crashed", False)),
                        "planned_recycle": planned_recycle,
                        "episodes_since_restart": episodes_since_restart,
                        "worker_recycle_target_episode": recycle_target,
                    }
                    if rgb_slot is not None:
                        event["rgb_shm_slot"] = rgb_slot
                        event["rgb_shm_seq"] = seq
                    put_event(event)

                    if info.get("crashed", False):
                        exit_reason = str(info.get("crash_reason") or "simulation_crashed")
                        break

                    if terminated or truncated:
                        if planned_recycle:
                            exit_reason = "planned_episode_recycle"
                            while True:
                                try:
                                    cmd = control_queue.get(timeout=0.1)
                                except queue.Empty:
                                    continue
                                if cmd.get("type") == "stop":
                                    break
                            break
                        if self.auto_reset:
                            reset_reason = "auto" if terminated else "truncated"
                            try:
                                obs, info = env.reset()
                            except Exception as exc:
                                tb = traceback.format_exc()
                                crash_event("auto_reset_failed", exc, tb, source_info=info)
                                exit_reason = "auto_reset_failed"
                                break
                            episode_step = 0
                            episode_reward = 0.0
                            put_event(reset_event(obs, info, reset_reason))
                except Exception as exc:
                    tb = traceback.format_exc()
                    crash_event("step_error", exc, tb)
                    exit_reason = "step_error"
                    break

        except Exception as exc:
            tb = traceback.format_exc()
            crash_event("initialization_failed", exc, tb)
            exit_reason = "initialization_failed"
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception:
                    logger.exception("env.close failed")
            if rgb_shm_buffer is not None:
                try:
                    rgb_shm_buffer.close()
                except Exception:
                    logger.exception("RGB shm close failed")
            logger.info("SimActor %s epoch %s exiting reason=%s", self.worker_id, self.worker_epoch, exit_reason)


def sim_actor_entry(actor: SimActorProcess, event_queue: mp.Queue, action_queue: mp.Queue, control_queue: mp.Queue) -> None:
    actor(event_queue, action_queue, control_queue)
