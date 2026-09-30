"""RGB camera observation handler for CARLA.

The handler is intentionally yaml-driven and strict: enabling RGB requires
explicit camera specs under ``observation_space.rgb``. This keeps the runtime
observation shape aligned with ``b2d_rlinfra.environment.spaces`` and avoids silent sensor
fallbacks.
"""

import queue
import threading
import time
from collections import deque
from typing import Dict, Iterable, List, Optional

import numpy as np
from gymnasium import spaces
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.timer import GameTime

__layer__ = (2, "Environment")


def get_rgb_config(obs_config: Dict) -> Dict:
    return obs_config.get("rgb", {}) or {}


def ordered_rgb_specs(specs: Optional[Iterable[Dict]], order: Iterable[str]) -> List[Dict]:
    if not specs:
        raise ValueError("observation_space.rgb.sensors must be a non-empty list")
    raw_specs = [dict(spec) for spec in specs]
    spec_by_id = {}
    for spec in raw_specs:
        sensor_id = spec.get("id")
        if sensor_id is None:
            raise ValueError("Every RGB sensor spec must define an 'id'")
        sensor_id = str(sensor_id)
        if sensor_id in spec_by_id:
            raise ValueError(f"Duplicate RGB sensor id: {sensor_id}")
        spec_by_id[sensor_id] = spec

    ordered = []
    for camera_id in order:
        camera_id = str(camera_id)
        if camera_id not in spec_by_id:
            raise ValueError(f"Missing RGB camera spec for id={camera_id}")
        ordered.append(spec_by_id[camera_id])
    return ordered


def rgb_observation_shape(rgb_config: Dict) -> tuple:
    order = rgb_config.get("camera_order")
    if not order:
        raise ValueError("observation_space.rgb.camera_order must be a non-empty list")
    specs = ordered_rgb_specs(rgb_config.get("sensors"), order)
    heights = {int(spec["height"]) for spec in specs}
    widths = {int(spec["width"]) for spec in specs}
    if len(heights) != 1 or len(widths) != 1:
        raise ValueError("RGB observation requires all cameras to share width/height")
    return (len(specs), heights.pop(), widths.pop(), 3)


class RGBSensorObsHandler:
    """Spawn configured CARLA RGB cameras and expose stacked BGR frames."""

    def __init__(self, rgb_obs_config: Optional[Dict] = None):
        self.config = dict(rgb_obs_config or {})
        self.output_key = str(self.config.get("output_key", "rgb"))
        self.emit_output = bool(self.config.get("emit_output", True))
        self.enabled = bool(self.config.get("enable", True))
        self.camera_order = [str(camera_id) for camera_id in self.config.get("camera_order", [])]
        self.sensor_specs = ordered_rgb_specs(self.config.get("sensors"), self.camera_order)
        self.output_shape = rgb_observation_shape(self.config)
        self.frame_timeout = float(self.config.get("frame_timeout", 10.0))
        self.warmup_ticks = int(self.config.get("warmup_ticks", 10))
        self.tick_timeout = float(self.config.get("tick_timeout", 60.0))
        self.sensor_tick = self.config.get("sensor_tick", None)
        jpeg_quality = self.config.get("jpeg_quality", None)
        self.jpeg_quality = None if jpeg_quality is None else int(jpeg_quality)
        if self.jpeg_quality is not None and not (1 <= self.jpeg_quality <= 100):
            raise ValueError("observation_space.rgb.jpeg_quality must be in [1, 100]")
        self.frame_history_size = max(1, int(self.config.get("frame_history_size", 2)))
        self.temporal_output_key = self.config.get("temporal_output_key")
        self.temporal_history_index = [
            int(v) for v in self.config.get("temporal_history_index", [-2, -1])
        ]
        self.temporal_camera_id = str(
            self.config.get(
                "temporal_camera_id",
                self.camera_order[0] if self.camera_order else "",
            )
        )
        temporal_min_history = abs(min(self.temporal_history_index)) if self.temporal_history_index else 1
        self._temporal_frames = deque(maxlen=max(temporal_min_history, 1))

        self._sensors = []
        self._frames: Dict[str, tuple] = {}
        self._frame_history: Dict[str, Dict[int, np.ndarray]] = {}
        self._frame_queue: queue.Queue = queue.Queue()
        self._lock = threading.Lock()

    @property
    def observation_space(self) -> spaces.Box:
        return spaces.Box(low=0, high=255, shape=self.output_shape, dtype=np.uint8)

    def reset(self):
        self.destroy()
        if not self.enabled:
            return

        world = CarlaDataProvider.get_world()
        ego_actor = getattr(CarlaDataProvider, "_ego_actor", None)
        if world is None:
            raise RuntimeError("CARLA world is not available for RGB sensors")
        if ego_actor is None or not ego_actor.is_alive:
            raise RuntimeError("Ego actor is not available for RGB sensors")

        blueprint_library = world.get_blueprint_library()
        for spec in self.sensor_specs:
            sensor = self._spawn_camera(world, blueprint_library, ego_actor, spec)
            self._sensors.append(sensor)

        for _ in range(max(0, self.warmup_ticks)):
            world.tick(self.tick_timeout)

    def _spawn_camera(self, world, blueprint_library, ego_actor, spec: Dict):
        import carla

        type_id = str(spec.get("type", "sensor.camera.rgb"))
        if type_id != "sensor.camera.rgb":
            raise ValueError(f"Unsupported RGB sensor type: {type_id}")

        camera_id = str(spec["id"])
        blueprint = blueprint_library.find(type_id)
        blueprint.set_attribute("image_size_x", str(int(spec["width"])))
        blueprint.set_attribute("image_size_y", str(int(spec["height"])))
        blueprint.set_attribute("fov", str(spec["fov"]))
        blueprint.set_attribute("role_name", camera_id)
        sensor_tick = spec.get("sensor_tick", self.sensor_tick)
        if sensor_tick is not None:
            blueprint.set_attribute("sensor_tick", str(sensor_tick))

        transform = carla.Transform(
            carla.Location(
                x=float(spec["x"]),
                y=float(spec["y"]),
                z=float(spec["z"]),
            ),
            carla.Rotation(
                pitch=float(spec.get("pitch", 0.0)),
                roll=float(spec.get("roll", 0.0)),
                yaw=float(spec.get("yaw", 0.0)),
            ),
        )
        try:
            sensor = world.spawn_actor(
                blueprint,
                transform,
                attach_to=ego_actor,
                attachment_type=carla.AttachmentType.Rigid,
            )
        except TypeError:
            sensor = world.spawn_actor(blueprint, transform, attach_to=ego_actor)

        sensor.listen(lambda image, sensor_id=camera_id: self._on_image(sensor_id, image))
        return sensor

    def _on_image(self, sensor_id: str, image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        bgr = self._normalize_bgr_frame(array[:, :, :3])
        frame_id = int(image.frame)
        with self._lock:
            self._frames[sensor_id] = (frame_id, bgr)
            history = self._frame_history.setdefault(sensor_id, {})
            history[frame_id] = bgr
            if len(history) > self.frame_history_size:
                for old_frame in sorted(history)[: len(history) - self.frame_history_size]:
                    history.pop(old_frame, None)
        self._frame_queue.put((sensor_id, frame_id))

    def _normalize_bgr_frame(self, frame: np.ndarray) -> np.ndarray:
        bgr = np.asarray(frame, dtype=np.uint8)
        if self.jpeg_quality is None:
            return bgr.copy()
        return self._jpeg_round_trip_bgr(bgr, self.jpeg_quality)

    @staticmethod
    def _jpeg_round_trip_bgr(frame: np.ndarray, quality: int) -> np.ndarray:
        import cv2

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
        ok, encoded = cv2.imencode(".jpg", np.asarray(frame, dtype=np.uint8), encode_param)
        if not ok:
            raise RuntimeError("Failed to JPEG-encode RGB camera frame")
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if decoded is None:
            raise RuntimeError("Failed to JPEG-decode RGB camera frame")
        return decoded.astype(np.uint8, copy=False)

    def get_observation(self) -> Dict[str, np.ndarray]:
        if not self.enabled:
            return {}

        target_frame = self._target_frame()
        deadline = time.time() + self.frame_timeout
        frames = None
        while time.time() < deadline:
            with self._lock:
                frames = self._collect_exact_frames_locked(target_frame)
                if frames is not None:
                    break
            try:
                self._frame_queue.get(timeout=min(0.05, max(0.0, deadline - time.time())))
            except queue.Empty:
                pass

        if frames is None:
            with self._lock:
                available = {
                    camera_id: self._frames.get(camera_id, (None,))[0]
                    for camera_id in self.camera_order
                }
            raise TimeoutError(
                f"Timed out waiting for RGB camera frames for exact frame {target_frame}; "
                f"available={available}; {self._frame_debug_context()}"
            )

        obs = np.stack(frames, axis=0).astype(np.uint8, copy=False)
        result = {}
        if self.emit_output:
            result[self.output_key] = obs
        if self.temporal_output_key:
            camera_idx = self.camera_order.index(self.temporal_camera_id)
            current_frame = obs[camera_idx]
            self._temporal_frames.append(current_frame)
            temporal_frames = []
            for history_idx in self.temporal_history_index:
                try:
                    temporal_frames.append(self._temporal_frames[history_idx])
                except IndexError:
                    temporal_frames.append(self._temporal_frames[0])
            result[str(self.temporal_output_key)] = np.stack(temporal_frames, axis=0).astype(
                np.uint8,
                copy=False,
            )
        return result

    def _collect_exact_frames_locked(self, target_frame: int):
        if not all(camera_id in self._frame_history for camera_id in self.camera_order):
            return None

        frames = []
        for camera_id in self.camera_order:
            frame = self._frame_history[camera_id].get(target_frame)
            if frame is None:
                return None
            frames.append(frame)
        return frames

    def _target_frame(self) -> int:
        try:
            game_frame = int(GameTime.get_frame())
            if game_frame > 0:
                return game_frame
        except Exception:
            pass
        try:
            snapshot = CarlaDataProvider.get_world().get_snapshot()
            return int(snapshot.frame)
        except Exception:
            pass
        return 0

    @staticmethod
    def _frame_debug_context() -> str:
        game_frame = None
        snapshot_frame = None
        try:
            game_frame = int(GameTime.get_frame())
        except Exception:
            pass
        try:
            snapshot_frame = int(CarlaDataProvider.get_world().get_snapshot().frame)
        except Exception:
            pass
        return f"game_frame={game_frame}, snapshot_frame={snapshot_frame}"

    def destroy(self):
        for sensor in self._sensors:
            try:
                if sensor is not None and sensor.is_alive:
                    sensor.stop()
            except Exception:
                pass
            try:
                if sensor is not None and sensor.is_alive:
                    sensor.destroy()
            except Exception:
                pass
        self._sensors = []
        with self._lock:
            self._frames.clear()
            self._frame_history.clear()
        self._temporal_frames.clear()
        self._drain_frame_queue()

    def _drain_frame_queue(self):
        while True:
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                break

    def close(self):
        self.destroy()

    def __del__(self):
        try:
            self.destroy()
        except Exception:
            pass
