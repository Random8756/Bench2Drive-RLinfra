"""MindDrive geometry/navigation observation handler.

This branch carries the lightweight state needed by MindDrive's inference
pipeline. RGB remains owned by ``RGBSensorObsHandler`` and optional shared
memory transport.
"""

from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np
from gymnasium import spaces

from b2d_rlinfra.environment.model_integrations.minddrive_route import (
    MINDDRIVE_GEO_REFERENCE_ATTR,
    MINDDRIVE_POSITION_OFFSET_X,
    MINDDRIVE_POSITION_OFFSET_Y,
    _latlon_ref_from_world,
    minddrive_actor_position_xy,
    minddrive_gps_to_location_xy,
)
from b2d_rlinfra.environment.handlers.sensor_context import get_sensor_packet

try:
    from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
except ImportError:  # pragma: no cover - optional for no-CARLA unit tests
    CarlaDataProvider = None

__layer__ = (2, "Environment")

CAMERA_ORDER = (
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)

LIDAR2IMG_BY_CAMERA = {
    "CAM_FRONT": np.array([[1142.51841, 800.0, 0.0, -952.0],
                           [0.0, 450.0, -1142.51841, -809.704417],
                           [0.0, 1.0, 0.0, -1.19],
                           [0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
    "CAM_FRONT_LEFT": np.array([[0.0, 1394.75744, 0.0, -920.539908],
                                [-368.61842, 258.109396, -1142.51841, -647.29675],
                                [-0.819152044, 0.573576436, 0.0, -0.829094072],
                                [0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
    "CAM_FRONT_RIGHT": np.array([[1310.64327, -477.035138, 0.0, -406.010608],
                                 [368.61842, 258.109396, -1142.51841, -647.29675],
                                 [0.819152044, 0.573576436, 0.0, -0.829094072],
                                 [0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
    "CAM_BACK": np.array([[-560.166031, -800.0, 0.0, -1288.0],
                          [0.0, -450.0, -560.166031, -858.939847],
                          [0.0, -1.0, 0.0, -1.61],
                          [0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
    "CAM_BACK_LEFT": np.array([[-1142.51841, 800.0, 0.0, -684.385123],
                               [-422.861679, -153.909064, -1142.51841, -496.004706],
                               [-0.939692621, -0.342020143, 0.0, -0.492889531],
                               [0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
    "CAM_BACK_RIGHT": np.array([[360.989788, -1347.23223, 0.0, -104.238127],
                                [422.861679, -153.909064, -1142.51841, -496.004706],
                                [0.939692621, -0.342020143, 0.0, -0.492889531],
                                [0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
}

LIDAR2CAM_BY_CAMERA = {
    "CAM_FRONT": np.array([[1.0, 0.0, 0.0, 0.0],
                           [0.0, 0.0, -1.0, -0.24],
                           [0.0, 1.0, 0.0, -1.19],
                           [0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
    "CAM_FRONT_LEFT": np.array([[0.57357644, 0.81915204, 0.0, -0.22517331],
                                [0.0, 0.0, -1.0, -0.24],
                                [-0.81915204, 0.57357644, 0.0, -0.82909407],
                                [0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
    "CAM_FRONT_RIGHT": np.array([[0.57357644, -0.81915204, 0.0, 0.22517331],
                                 [0.0, 0.0, -1.0, -0.24],
                                 [0.81915204, 0.57357644, 0.0, -0.82909407],
                                 [0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
    "CAM_BACK": np.array([[-1.0, 0.0, 0.0, 0.0],
                          [0.0, 0.0, -1.0, -0.24],
                          [0.0, -1.0, 0.0, -1.61],
                          [0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
    "CAM_BACK_LEFT": np.array([[-0.34202014, 0.93969262, 0.0, -0.25388956],
                               [0.0, 0.0, -1.0, -0.24],
                               [-0.93969262, -0.34202014, 0.0, -0.49288953],
                               [0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
    "CAM_BACK_RIGHT": np.array([[-0.34202014, -0.93969262, 0.0, 0.25388956],
                                [0.0, 0.0, -1.0, -0.24],
                                [0.93969262, -0.34202014, 0.0, -0.49288953],
                                [0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
}

LIDAR2EGO = np.array([[0.0, 1.0, 0.0, -0.39],
                      [-1.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0, 1.84],
                      [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)

# Fallback target used only when the MindDrive route-context provider is not
# enabled. The MindDrive parity path should use _minddrive_route_context.
ROUTE_COMMAND_TARGET_INDEX = 5


def _invert_egopose(egopose: np.ndarray) -> np.ndarray:
    inverse = np.zeros((4, 4), dtype=np.float32)
    rotation = egopose[:3, :3]
    translation = egopose[:3, 3]
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -np.dot(rotation.T, translation)
    inverse[3, 3] = 1.0
    return inverse


def _command_to_int(command: Any) -> int:
    if hasattr(command, "value"):
        command = command.value
    return int(command)


def command2hot(command: Any, max_dim: int = 6) -> np.ndarray:
    command_int = _command_to_int(command)
    if command_int < 0:
        command_int = 4
    index = command_int - 1
    if index < 0 or index >= max_dim:
        raise ValueError(f"MindDrive route command out of range: {command_int}")
    one_hot = np.zeros(max_dim, dtype=np.float32)
    one_hot[index] = 1.0
    return one_hot


def command2nohot(command: Any, max_dim: int = 6) -> np.ndarray:
    command_int = _command_to_int(command)
    if command_int < 0:
        command_int = 4
    index = command_int - 1
    if index < 0 or index >= max_dim:
        raise ValueError(f"MindDrive route command out of range: {command_int}")
    return np.asarray([index], dtype=np.int64)


def minddrive_state_space() -> spaces.Dict:
    return spaces.Dict(
        {
            "can_bus": spaces.Box(-np.inf, np.inf, shape=(18,), dtype=np.float32),
            "ego_pose": spaces.Box(-np.inf, np.inf, shape=(4, 4), dtype=np.float32),
            "ego_pose_inv": spaces.Box(-np.inf, np.inf, shape=(4, 4), dtype=np.float32),
            "lidar2img": spaces.Box(-np.inf, np.inf, shape=(len(CAMERA_ORDER), 4, 4), dtype=np.float32),
            "lidar2cam": spaces.Box(-np.inf, np.inf, shape=(len(CAMERA_ORDER), 4, 4), dtype=np.float32),
            "cam_intrinsic": spaces.Box(-np.inf, np.inf, shape=(len(CAMERA_ORDER), 4, 4), dtype=np.float32),
            "command": spaces.Box(0, 5, shape=(1,), dtype=np.int64),
            "ego_fut_cmd": spaces.Box(0.0, 1.0, shape=(6,), dtype=np.float32),
            "local_command_xy": spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32),
            "frame_idx": spaces.Box(0, np.iinfo(np.int64).max, shape=(1,), dtype=np.int64),
            "timestamp": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
            "episode_id": spaces.Box(0, np.iinfo(np.int64).max, shape=(1,), dtype=np.int64),
        }
    )


class MindDriveStateObsHandler:
    def __init__(self, state_obs_config: Dict[str, Any] | None = None):
        self.config = dict(state_obs_config or {})
        self.output_key = str(self.config.get("output_key", "minddrive_state"))
        self.camera_order = tuple(self.config.get("camera_order", CAMERA_ORDER))
        self.route_source = str(self.config.get("route_source", "auto")).lower()
        self.timestamp_source = str(self.config.get("timestamp_source", "episode_step")).lower()
        self.timestamp_hz = float(self.config.get("timestamp_hz", 20.0))
        if self.timestamp_hz <= 0.0:
            raise ValueError("MindDrive timestamp_hz must be positive")
        self.position_offset_x = float(self.config.get("position_offset_x", MINDDRIVE_POSITION_OFFSET_X))
        self.position_offset_y = float(self.config.get("position_offset_y", MINDDRIVE_POSITION_OFFSET_Y))
        self.gps_sensor_id = str(self.config.get("gps_sensor_id", "GPS"))
        self.imu_sensor_id = str(self.config.get("imu_sensor_id", "IMU"))
        self.speed_sensor_id = str(self.config.get("speed_sensor_id", "SPEED"))
        self._lat_ref = None
        self._lon_ref = None
        self._episode_id = -1
        self._next_frame_idx = 0
        self.lidar2img = np.stack([LIDAR2IMG_BY_CAMERA[cam] for cam in self.camera_order], axis=0).astype(np.float32)
        self.lidar2cam = np.stack([LIDAR2CAM_BY_CAMERA[cam] for cam in self.camera_order], axis=0).astype(np.float32)
        self.cam_intrinsic = np.stack(
            [LIDAR2IMG_BY_CAMERA[cam] @ np.linalg.inv(LIDAR2CAM_BY_CAMERA[cam]) for cam in self.camera_order],
            axis=0,
        ).astype(np.float32)

    @property
    def observation_space(self) -> spaces.Dict:
        return minddrive_state_space()

    def reset(self) -> None:
        self._episode_id = 0 if self._episode_id < 0 else self._episode_id + 1
        self._next_frame_idx = 0
        self._lat_ref = None
        self._lon_ref = None

    def get_observation(self) -> Dict[str, Dict[str, np.ndarray]]:
        if CarlaDataProvider is None:
            raise RuntimeError("srunner CarlaDataProvider is not available for MindDrive state observation")
        ego_actor = getattr(CarlaDataProvider, "_ego_actor", None)
        if ego_actor is None or not getattr(ego_actor, "is_alive", True):
            raise RuntimeError("Ego actor is not available for MindDrive state observation")

        self._refresh_provider_contexts()
        transform = CarlaDataProvider.get_transform(ego_actor)
        location = transform.location
        velocity = ego_actor.get_velocity()
        legacy_sensor_context = getattr(CarlaDataProvider, "_minddrive_sensor_context", None)
        if not isinstance(legacy_sensor_context, dict):
            legacy_sensor_context = {}
        gps_packet = get_sensor_packet(self.gps_sensor_id, CarlaDataProvider)
        imu_packet = get_sensor_packet(self.imu_sensor_id, CarlaDataProvider)
        speed_packet = get_sensor_packet(self.speed_sensor_id, CarlaDataProvider)
        carla_yaw_rad = math.radians(float(transform.rotation.yaw))
        fallback_raw_theta = math.pi / 2.0 - carla_yaw_rad
        raw_theta = fallback_raw_theta
        acceleration_array = np.zeros(3, dtype=np.float32)
        angular_velocity_array = np.zeros(3, dtype=np.float32)

        if imu_packet is not None:
            imu_data = np.asarray(imu_packet.get("data"), dtype=np.float64).reshape(-1)
            if imu_data.size >= 7:
                raw_theta = float(imu_data[6])
                if math.isnan(raw_theta):
                    raw_theta = 0.0
                else:
                    acceleration_array = imu_data[:3].astype(np.float32)
                    angular_velocity_array = imu_data[3:6].astype(np.float32)
        elif "raw_theta" in legacy_sensor_context:
            raw_theta = float(legacy_sensor_context.get("raw_theta", fallback_raw_theta))
            acceleration = legacy_sensor_context.get("acceleration")
            angular_velocity = legacy_sensor_context.get("angular_velocity")
            if acceleration is not None:
                acceleration_array = np.asarray(acceleration, dtype=np.float32).reshape(-1)[:3]
            if angular_velocity is not None:
                angular_velocity_array = np.asarray(angular_velocity, dtype=np.float32).reshape(-1)[:3]
        else:
            if hasattr(ego_actor, "get_acceleration"):
                acceleration_obj = ego_actor.get_acceleration()
                acceleration_array = np.asarray(
                    [acceleration_obj.x, acceleration_obj.y, acceleration_obj.z],
                    dtype=np.float32,
                )
            if hasattr(ego_actor, "get_angular_velocity"):
                angular_velocity_obj = ego_actor.get_angular_velocity()
                angular_velocity_array = np.asarray(
                    [angular_velocity_obj.x, angular_velocity_obj.y, angular_velocity_obj.z],
                    dtype=np.float32,
                )

        ego_theta = -raw_theta + math.pi / 2.0
        if "ego_theta" in legacy_sensor_context and imu_packet is None:
            ego_theta = float(legacy_sensor_context["ego_theta"])

        reference_xy = minddrive_actor_position_xy(
            transform,
            offset_x=self.position_offset_x,
            offset_y=self.position_offset_y,
        )
        if gps_packet is not None:
            gps_data = np.asarray(gps_packet.get("data"), dtype=np.float64).reshape(-1)
            if gps_data.size >= 2:
                reference_xy = minddrive_gps_to_location_xy(gps_data, *self._latlon_ref())
        elif "position_xy" in legacy_sensor_context:
            reference_xy = np.asarray(legacy_sensor_context["position_xy"], dtype=np.float32).reshape(-1)[:2]

        fallback_speed = math.sqrt(float(velocity.x) ** 2 + float(velocity.y) ** 2 + float(velocity.z) ** 2)
        speed = float(legacy_sensor_context.get("speed", fallback_speed))
        if speed_packet is not None:
            speed_data = speed_packet.get("data")
            if isinstance(speed_data, dict) and "speed" in speed_data:
                speed = float(speed_data["speed"])

        can_bus = np.zeros(18, dtype=np.float32)
        can_bus[0] = float(reference_xy[0])
        can_bus[1] = -float(reference_xy[1])
        can_bus[3:7] = np.asarray(
            [math.cos(ego_theta / 2.0), 0.0, 0.0, math.sin(ego_theta / 2.0)],
            dtype=np.float32,
        )
        can_bus[7] = speed
        can_bus[10:13] = acceleration_array
        can_bus[11] *= -1.0
        can_bus[13:16] = -angular_velocity_array
        can_bus[16] = ego_theta
        can_bus[17] = ego_theta / np.pi * 180.0

        command, target_xy = self._route_command_target(location)
        command_near_xy = np.asarray([target_xy[0] - can_bus[0], -target_xy[1] - can_bus[1]], dtype=np.float32)
        rotation_matrix = np.asarray(
            [[math.cos(raw_theta), -math.sin(raw_theta)], [math.sin(raw_theta), math.cos(raw_theta)]],
            dtype=np.float32,
        )
        local_command_xy = rotation_matrix @ command_near_xy

        ego2world = np.eye(4, dtype=np.float32)
        c, s = math.cos(ego_theta), math.sin(ego_theta)
        ego2world[:3, :3] = np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        ego2world[:2, 3] = can_bus[:2]
        lidar2global = ego2world @ LIDAR2EGO

        frame_idx = self._frame_idx()
        episode_id = self._current_episode_id()
        state = {
            "can_bus": can_bus,
            "ego_pose": lidar2global.astype(np.float32),
            "ego_pose_inv": _invert_egopose(lidar2global),
            "lidar2img": self.lidar2img,
            "lidar2cam": self.lidar2cam,
            "cam_intrinsic": self.cam_intrinsic,
            "command": command2nohot(command),
            "ego_fut_cmd": command2hot(command),
            "local_command_xy": local_command_xy.astype(np.float32),
            "frame_idx": np.asarray([frame_idx], dtype=np.int64),
            "timestamp": np.asarray([frame_idx / self.timestamp_hz], dtype=np.float32),
            "episode_id": np.asarray([episode_id], dtype=np.int64),
        }
        if self.timestamp_source == "episode_step":
            self._next_frame_idx += 1
        return {self.output_key: state}

    @staticmethod
    def _refresh_provider_contexts() -> None:
        route_refresher = getattr(CarlaDataProvider, "_minddrive_route_context_refresher", None)
        if callable(route_refresher):
            route_refresher()

    def _latlon_ref(self) -> tuple[float, float]:
        context = getattr(CarlaDataProvider, "_minddrive_route_context", None)
        if isinstance(context, dict) and "lat_ref" in context and "lon_ref" in context:
            return float(context["lat_ref"]), float(context["lon_ref"])

        geo_reference = getattr(CarlaDataProvider, MINDDRIVE_GEO_REFERENCE_ATTR, None)
        if isinstance(geo_reference, dict) and "lat_ref" in geo_reference and "lon_ref" in geo_reference:
            return float(geo_reference["lat_ref"]), float(geo_reference["lon_ref"])

        if self._lat_ref is None or self._lon_ref is None:
            try:
                world = CarlaDataProvider.get_world()
                self._lat_ref, self._lon_ref = _latlon_ref_from_world(world)
            except Exception:
                self._lat_ref, self._lon_ref = 42.0, 2.0
        return float(self._lat_ref), float(self._lon_ref)

    def _route_command_target(self, ego_location: Any) -> tuple[Any, np.ndarray]:
        context = getattr(CarlaDataProvider, "_minddrive_route_context", None)
        if isinstance(context, dict) and "curr_command" in context and "near_node_xy" in context:
            return context["curr_command"], np.asarray(context["near_node_xy"], dtype=np.float32)

        if self.route_source in ("minddrive_route_planner", "minddrive_route", "context"):
            raise RuntimeError("MindDrive state observation requires CarlaDataProvider._minddrive_route_context")

        route = getattr(CarlaDataProvider, "_ego_vehicle_route", None) or []
        if not route:
            raise RuntimeError("MindDrive state observation requires CarlaDataProvider._ego_vehicle_route")
        command = route[0][1]
        target_index = min(ROUTE_COMMAND_TARGET_INDEX, len(route) - 1)
        target_waypoint = route[target_index][0]
        loc = target_waypoint.location if hasattr(target_waypoint, "location") else target_waypoint.transform.location
        return command, np.asarray([float(loc.x), float(loc.y)], dtype=np.float32)

    def _current_episode_id(self) -> int:
        if self._episode_id < 0:
            self._episode_id = 0
        return int(self._episode_id)

    def _frame_idx(self) -> int:
        if self.timestamp_source == "episode_step":
            return int(self._next_frame_idx)
        try:
            world = CarlaDataProvider.get_world()
            return int(world.get_snapshot().frame)
        except Exception as exc:
            raise RuntimeError("MindDrive state observation requires a valid CARLA frame") from exc
