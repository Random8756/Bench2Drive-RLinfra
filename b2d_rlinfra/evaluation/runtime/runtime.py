"""Leaderboard-2.0 / Bench2Drive evaluation runtime.

* Plug-in runtime that turns a trained model into a Leaderboard-2.0 /
  Bench2Drive ``AutonomousAgent.run_step()`` implementation.
* Decouples policy frequency from simulator frequency via a fixed
  ``decision_stride`` so the same model trained at 10 Hz drives a 20 Hz
  evaluator without re-tuning.
* Exports BEV videos for qualitative inspection.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, Optional, Union

import carla
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

__layer__ = (5, "Evaluation")

from b2d_rlinfra.environment.handlers.birdview_obs_handler import BirdViewObsManager
from b2d_rlinfra.environment.handlers.scalars_obs_handler import ScalarObsHandler

from .action_adapter import ActionAdapter
from .provider_compat import ProviderCompat
from .route_tracker import RouteTracker
from .video_export import export_bev_video


class LeaderboardEvalRuntime:
    def __init__(
        self,
        *,
        model,
        env_config: Dict,
        route: Iterable,
        route_name: str,
        output_root: Union[str, Path],
        save_video: bool = True,
        video_fps: int = 10,
        stochastic: bool = False,
    ):
        self._model = model
        self._env_config = deepcopy(env_config)
        self._route = list(route)
        self._route_name = str(route_name)
        self._output_root = Path(output_root).expanduser().resolve()
        self._save_video = bool(save_video)
        self._video_fps = int(video_fps)
        self._deterministic = not bool(stochastic)

        self._provider_compat = ProviderCompat()
        self._route_tracker = RouteTracker(window_size=5)
        self._action_adapter = ActionAdapter(self._env_config["action_space"])
        self._policy_frequency_hz = float(self._env_config["carla"]["frequency_hz"])
        self._simulator_frequency_hz: Optional[float] = None
        self._decision_stride = 1
        self._sim_step_index = 0
        self._cached_control: Optional[carla.VehicleControl] = None

        obs_cfg = self._env_config["observation_space"]
        vector_cfg = obs_cfg["vector"]
        self._vector_type = vector_cfg["type"]
        if self._vector_type not in ("bev_mask", "bev_image"):
            raise ValueError(f"Unsupported vector observation type: {self._vector_type}")
        self._vector_handler = BirdViewObsManager(vector_cfg)

        scalars_cfg = obs_cfg.get("scalars")
        self._scalar_handler = ScalarObsHandler(scalars_cfg) if scalars_cfg else None

        self._video_frames = []
        self._last_debug_frame = {}

    def _resolve_simulator_frequency_hz(self) -> float:
        world = CarlaDataProvider.get_world()
        if world is None:
            raise ValueError("CARLA world is not available when resetting leaderboard eval runtime")

        fixed_delta_seconds = world.get_settings().fixed_delta_seconds
        if fixed_delta_seconds is None or fixed_delta_seconds <= 0.0:
            raise ValueError("Leaderboard eval runtime requires a positive fixed_delta_seconds")
        return 1.0 / fixed_delta_seconds

    @staticmethod
    def _clone_control(control: carla.VehicleControl) -> carla.VehicleControl:
        cloned = carla.VehicleControl()
        cloned.throttle = float(control.throttle)
        cloned.steer = float(control.steer)
        cloned.brake = float(control.brake)
        cloned.hand_brake = bool(control.hand_brake)
        cloned.reverse = bool(control.reverse)
        cloned.manual_gear_shift = bool(control.manual_gear_shift)
        cloned.gear = int(control.gear)
        return cloned

    def reset(self) -> None:
        self._provider_compat.bootstrap()
        self._provider_compat.refresh()
        self._simulator_frequency_hz = self._resolve_simulator_frequency_hz()
        decision_stride = self._simulator_frequency_hz / self._policy_frequency_hz
        rounded_stride = int(round(decision_stride))
        if rounded_stride < 1 or abs(decision_stride - rounded_stride) > 1e-6:
            raise ValueError(
                "Leaderboard eval runtime requires simulator_frequency_hz/policy_frequency_hz "
                "to be a positive integer"
            )
        self._decision_stride = rounded_stride
        self._sim_step_index = 0
        self._cached_control = None
        self._route_tracker.reset(self._route)
        self._vector_handler.reset()
        if self._scalar_handler is not None:
            self._scalar_handler.reset()
        self._video_frames.clear()
        self._last_debug_frame = {}

    def _build_observation(self):
        vector_result = self._vector_handler.get_observation()
        observation = {
            "vector": vector_result["bev_mask"] if self._vector_type == "bev_mask" else vector_result["bev_image"],
        }

        if self._scalar_handler is not None:
            observation.update(self._scalar_handler.get_observation())

        debug_frame = {
            "bev_image": vector_result.get("bev_image"),
            "emergency_vehicles_in_vision": vector_result.get("emergency_vehicles_in_vision", False),
        }
        return observation, debug_frame

    def run_step(self) -> carla.VehicleControl:
        ego_actor = self._provider_compat.refresh()
        if ego_actor is None or not ego_actor.is_alive:
            self._cached_control = None
            return carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0)

        should_predict = (
            self._cached_control is None
            or self._sim_step_index % self._decision_stride == 0
        )

        if should_predict:
            # Only decision ticks advance route/history so policy-facing time stays at 10Hz.
            self._route_tracker.update(ego_actor)
            observation, debug_frame = self._build_observation()
            action, _ = self._model.predict(observation, deterministic=self._deterministic)
            predicted_control = self._action_adapter.to_control(action)
            self._cached_control = self._clone_control(predicted_control)

            self._last_debug_frame = debug_frame
            bev_image = debug_frame.get("bev_image")
            if self._save_video and bev_image is not None:
                self._video_frames.append(bev_image.copy())

        control = self._clone_control(self._cached_control)
        self._sim_step_index += 1
        return control

    def close(self) -> Optional[Path]:
        self._provider_compat.cleanup()
        replay_buffer = getattr(self._model, "replay_buffer", None)
        if replay_buffer is not None and hasattr(replay_buffer, "cleanup"):
            try:
                replay_buffer.cleanup()
            except Exception:
                pass

        if not self._save_video:
            return None

        video_path = self._output_root / "videos" / f"{self._route_name}.mp4"
        return export_bev_video(self._video_frames, video_path, fps=self._video_fps)
