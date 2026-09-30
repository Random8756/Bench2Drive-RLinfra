"""GNSS / IMU / speedometer observation handlers for CARLA.

These handlers mirror the lifecycle of ``RGBSensorObsHandler`` for fixed-size
ego sensors that are commonly consumed by model-specific state builders.
"""

from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
import queue
import threading
import time
from typing import Any, Dict, Optional, Type

import numpy as np
import yaml
from gymnasium import spaces

from b2d_rlinfra.environment.handlers.sensor_context import clear_sensor_packet, publish_sensor_packet

try:
    from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
    from srunner.scenariomanager.timer import GameTime
except ImportError:  # pragma: no cover - optional outside CARLA runtime
    CarlaDataProvider = None
    GameTime = None

__layer__ = (2, "Environment")

PRESET_DIR = Path(__file__).resolve().parent / "presets"

GNSS_DEFAULT_SPEC = {
    "type": "sensor.other.gnss",
    "id": "GPS",
    "x": -1.4,
    "y": 0.0,
    "z": 0.0,
    "roll": 0.0,
    "pitch": 0.0,
    "yaw": 0.0,
}

IMU_DEFAULT_SPEC = {
    "type": "sensor.other.imu",
    "id": "IMU",
    "x": -1.4,
    "y": 0.0,
    "z": 0.0,
    "roll": 0.0,
    "pitch": 0.0,
    "yaw": 0.0,
}

SPEEDOMETER_DEFAULT_SPEC = {
    "type": "sensor.speedometer",
    "id": "SPEED",
    "reading_frequency": 20,
}

SENSOR_CONFIG_KEYS = {
    "type",
    "id",
    "x",
    "y",
    "z",
    "roll",
    "pitch",
    "yaw",
    "sensor_tick",
    "reading_frequency",
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _load_observation_preset(name: str) -> Dict[str, Any]:
    preset_path = PRESET_DIR / f"{name}.yaml"
    if not preset_path.exists():
        raise ValueError(f"Unknown observation_space preset: {name!r}")
    with preset_path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Observation preset {preset_path} must contain a mapping")
    return payload


def apply_observation_presets(obs_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Expand observation sensor presets while preserving explicit overrides."""
    raw_cfg = dict(obs_cfg or {})
    presets = raw_cfg.get("other_sensor_presets", []) or []
    if isinstance(presets, str):
        presets = [presets]
    if not isinstance(presets, (list, tuple)):
        raise ValueError("observation_space.other_sensor_presets must be a string or list of strings")

    expanded: Dict[str, Any] = {}
    for preset_name in presets:
        if not isinstance(preset_name, str):
            raise ValueError("observation_space.other_sensor_presets entries must be strings")
        expanded = _deep_merge(expanded, _load_observation_preset(str(preset_name)))
    raw_cfg.pop("other_sensor_presets", None)
    return _deep_merge(expanded, raw_cfg)


def _sensor_spec(config: Dict[str, Any], defaults: Dict[str, Any]) -> Dict[str, Any]:
    spec = _deep_merge(defaults, config.get("sensor", {}) or {})
    inline = {key: value for key, value in config.items() if key in SENSOR_CONFIG_KEYS}
    return _deep_merge(spec, inline)


def _target_frame() -> int:
    try:
        if GameTime is not None:
            game_frame = int(GameTime.get_frame())
            if game_frame > 0:
                return game_frame
    except Exception:
        pass
    try:
        if CarlaDataProvider is not None:
            snapshot = CarlaDataProvider.get_world().get_snapshot()
            return int(snapshot.frame)
    except Exception:
        pass
    return 0


def _frame_debug_context() -> str:
    game_frame = None
    snapshot_frame = None
    try:
        if GameTime is not None:
            game_frame = int(GameTime.get_frame())
    except Exception:
        pass
    try:
        if CarlaDataProvider is not None:
            snapshot_frame = int(CarlaDataProvider.get_world().get_snapshot().frame)
    except Exception:
        pass
    return f"game_frame={game_frame}, snapshot_frame={snapshot_frame}"


def _vector3_to_array(value: Any) -> np.ndarray:
    return np.asarray([float(value.x), float(value.y), float(value.z)], dtype=np.float64)


class _FixedFrameSensorObsHandler:
    sensor_type: str = ""
    default_output_key: str = ""
    default_spec: Dict[str, Any] = {}
    output_shape = (0,)
    output_dtype = np.float64

    def __init__(self, sensor_obs_config: Optional[Dict[str, Any]] = None):
        self.config = dict(sensor_obs_config or {})
        self.enabled = bool(self.config.get("enable", True))
        self.emit_output = bool(self.config.get("emit_output", True))
        self.sensor_spec = _sensor_spec(self.config, self.default_spec)
        self.sensor_id = str(self.sensor_spec["id"])
        self.output_key = str(self.config.get("output_key", self.default_output_key or self.sensor_id))
        self.frame_timeout = float(self.config.get("frame_timeout", 10.0))
        self.frame_history_size = max(1, int(self.config.get("frame_history_size", 4)))
        self._sensor = None
        self._frame_history: Dict[int, np.ndarray] = {}
        self._frame_queue: queue.Queue[int] = queue.Queue()
        self._lock = threading.Lock()

    @property
    def observation_space(self) -> spaces.Box:
        return spaces.Box(-np.inf, np.inf, shape=self.output_shape, dtype=self.output_dtype)

    def reset(self) -> None:
        self.destroy()
        clear_sensor_packet(CarlaDataProvider, self.sensor_id)
        if not self.enabled:
            return
        if CarlaDataProvider is None:
            raise RuntimeError(f"srunner CarlaDataProvider is not available for {self.sensor_id}")
        world = CarlaDataProvider.get_world()
        ego_actor = getattr(CarlaDataProvider, "_ego_actor", None)
        if world is None:
            raise RuntimeError(f"CARLA world is not available for {self.sensor_id}")
        if ego_actor is None or not ego_actor.is_alive:
            raise RuntimeError(f"Ego actor is not available for {self.sensor_id}")
        self._sensor = self._spawn_sensor(world, ego_actor)

    def _spawn_sensor(self, world: Any, ego_actor: Any) -> Any:
        import carla

        blueprint = world.get_blueprint_library().find(self.sensor_type)
        for key, value in self._blueprint_attributes().items():
            blueprint.set_attribute(str(key), str(value))
        transform = carla.Transform(
            carla.Location(
                x=float(self.sensor_spec.get("x", 0.0)),
                y=float(self.sensor_spec.get("y", 0.0)),
                z=float(self.sensor_spec.get("z", 0.0)),
            ),
            carla.Rotation(
                pitch=float(self.sensor_spec.get("pitch", 0.0)),
                roll=float(self.sensor_spec.get("roll", 0.0)),
                yaw=float(self.sensor_spec.get("yaw", 0.0)),
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
        sensor.listen(self._on_sensor_event)
        return sensor

    def _blueprint_attributes(self) -> Dict[str, Any]:
        attributes = {"role_name": self.sensor_id}
        sensor_tick = self.sensor_spec.get("sensor_tick")
        if sensor_tick is not None:
            attributes["sensor_tick"] = sensor_tick
        return attributes

    def _on_sensor_event(self, event: Any) -> None:
        data = self._parse_event(event)
        frame = int(event.frame)
        with self._lock:
            self._frame_history[frame] = data
            if len(self._frame_history) > self.frame_history_size:
                for old_frame in sorted(self._frame_history)[: len(self._frame_history) - self.frame_history_size]:
                    self._frame_history.pop(old_frame, None)
        self._frame_queue.put(frame)

    def _parse_event(self, event: Any) -> np.ndarray:
        raise NotImplementedError

    def get_observation(self) -> Dict[str, np.ndarray]:
        if not self.enabled:
            return {}
        target = _target_frame()
        data = self._wait_for_frame(target)
        publish_sensor_packet(CarlaDataProvider, self.sensor_id, self.sensor_type, target, data)
        if not self.emit_output:
            return {}
        return {self.output_key: data.astype(self.output_dtype, copy=False)}

    def _wait_for_frame(self, target_frame: int) -> np.ndarray:
        deadline = time.time() + self.frame_timeout
        while time.time() < deadline:
            with self._lock:
                data = self._frame_history.get(target_frame)
                if data is not None:
                    return data
            try:
                self._frame_queue.get(timeout=min(0.05, max(0.0, deadline - time.time())))
            except queue.Empty:
                pass

        with self._lock:
            available = sorted(self._frame_history)
        raise TimeoutError(
            f"Timed out waiting for {self.sensor_id} frame {target_frame}; "
            f"available={available[-8:]}; {_frame_debug_context()}"
        )

    def destroy(self) -> None:
        if self._sensor is not None:
            try:
                if getattr(self._sensor, "is_alive", True):
                    self._sensor.stop()
            except Exception:
                pass
            try:
                if getattr(self._sensor, "is_alive", True):
                    self._sensor.destroy()
            except Exception:
                pass
        self._sensor = None
        with self._lock:
            self._frame_history.clear()
        self._drain_frame_queue()

    def _drain_frame_queue(self) -> None:
        while True:
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                break

    def close(self) -> None:
        self.destroy()
        clear_sensor_packet(CarlaDataProvider, self.sensor_id)

    def __del__(self):
        try:
            self.destroy()
        except Exception:
            pass


class GnssSensorObsHandler(_FixedFrameSensorObsHandler):
    sensor_type = "sensor.other.gnss"
    default_output_key = "gnss"
    default_spec = GNSS_DEFAULT_SPEC
    output_shape = (3,)
    output_dtype = np.float64

    def _blueprint_attributes(self) -> Dict[str, Any]:
        attributes = super()._blueprint_attributes()
        attributes.update(
            {
                "noise_alt_stddev": 0.000005,
                "noise_lat_stddev": 0.000005,
                "noise_lon_stddev": 0.000005,
                "noise_alt_bias": 0.0,
                "noise_lat_bias": 0.0,
                "noise_lon_bias": 0.0,
            }
        )
        return attributes

    def _parse_event(self, event: Any) -> np.ndarray:
        return np.asarray([event.latitude, event.longitude, event.altitude], dtype=np.float64)


class ImuSensorObsHandler(_FixedFrameSensorObsHandler):
    sensor_type = "sensor.other.imu"
    default_output_key = "imu"
    default_spec = IMU_DEFAULT_SPEC
    output_shape = (7,)
    output_dtype = np.float64

    def _blueprint_attributes(self) -> Dict[str, Any]:
        attributes = super()._blueprint_attributes()
        attributes.update(
            {
                "noise_accel_stddev_x": 0.001,
                "noise_accel_stddev_y": 0.001,
                "noise_accel_stddev_z": 0.015,
                "noise_gyro_stddev_x": 0.001,
                "noise_gyro_stddev_y": 0.001,
                "noise_gyro_stddev_z": 0.001,
            }
        )
        return attributes

    def _parse_event(self, event: Any) -> np.ndarray:
        return np.asarray(
            [
                event.accelerometer.x,
                event.accelerometer.y,
                event.accelerometer.z,
                event.gyroscope.x,
                event.gyroscope.y,
                event.gyroscope.z,
                event.compass,
            ],
            dtype=np.float64,
        )


class SpeedometerSensorObsHandler:
    sensor_type = "sensor.speedometer"
    default_spec = SPEEDOMETER_DEFAULT_SPEC

    def __init__(self, sensor_obs_config: Optional[Dict[str, Any]] = None):
        self.config = dict(sensor_obs_config or {})
        self.enabled = bool(self.config.get("enable", True))
        self.emit_output = bool(self.config.get("emit_output", True))
        self.sensor_spec = _sensor_spec(self.config, self.default_spec)
        self.sensor_id = str(self.sensor_spec["id"])
        self.output_key = str(self.config.get("output_key", "speedometer"))

    @property
    def observation_space(self) -> spaces.Box:
        return spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32)

    def reset(self) -> None:
        clear_sensor_packet(CarlaDataProvider, self.sensor_id)

    def get_observation(self) -> Dict[str, np.ndarray]:
        if not self.enabled:
            return {}
        if CarlaDataProvider is None:
            raise RuntimeError("srunner CarlaDataProvider is not available for speedometer")
        speed = self._forward_speed()
        frame = _target_frame()
        publish_sensor_packet(CarlaDataProvider, self.sensor_id, self.sensor_type, frame, {"speed": speed})
        if not self.emit_output:
            return {}
        return {self.output_key: np.asarray([speed], dtype=np.float32)}

    @staticmethod
    def _forward_speed() -> float:
        if CarlaDataProvider is None:
            raise RuntimeError("srunner CarlaDataProvider is not available for speedometer")
        ego_actor = getattr(CarlaDataProvider, "_ego_actor", None)
        if ego_actor is None or not ego_actor.is_alive:
            raise RuntimeError("Ego actor is not available for speedometer")
        velocity = ego_actor.get_velocity()
        if hasattr(CarlaDataProvider, "get_transform"):
            transform = CarlaDataProvider.get_transform(ego_actor)
        else:
            transform = ego_actor.get_transform()
        vel_np = _vector3_to_array(velocity)
        pitch = math.radians(float(transform.rotation.pitch))
        yaw = math.radians(float(transform.rotation.yaw))
        orientation = np.asarray(
            [math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), math.sin(pitch)],
            dtype=np.float64,
        )
        return float(np.dot(vel_np, orientation))

    def close(self) -> None:
        clear_sensor_packet(CarlaDataProvider, self.sensor_id)


EGO_SENSOR_HANDLERS: Dict[str, Type[Any]] = {
    "gnss": GnssSensorObsHandler,
    "imu": ImuSensorObsHandler,
    "speedometer": SpeedometerSensorObsHandler,
}


def ego_sensor_branch_space(handler_cls: Type[Any], cfg: Dict[str, Any]) -> spaces.Space:
    return handler_cls(cfg).observation_space
