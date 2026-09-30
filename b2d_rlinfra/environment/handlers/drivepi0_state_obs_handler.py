"""DrivePi0 proprioceptive state observation."""

from __future__ import annotations

import json
import math
import os
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from gymnasium import spaces
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from b2d_rlinfra.environment.handlers.sensor_context import get_sensor_packet
from b2d_rlinfra.environment.model_integrations.drivepi0_route import (
    DEFAULT_LAT_REF,
    DEFAULT_LON_REF,
    DRIVEPI0_GEO_REFERENCE_ATTR,
    DRIVEPI0_ROUTE_CONTEXT_ATTR,
    DRIVEPI0_ROUTE_REFRESHER_ATTR,
    drivepi0_gps_to_location_xy,
    drivepi0_latlon_ref_from_world,
)

__layer__ = (2, "Environment")


class DrivePi0StateObsHandler:
    """Emit DrivePi0's normalized proprio state under ``output_key``."""

    def __init__(self, config: Optional[Dict] = None):
        self.config = dict(config or {})
        self.output_key = str(self.config.get("output_key", "drivepi0_state"))
        self.temporal_mode = str(self.config.get("temporal_mode", "rlinf")).strip().lower()
        if self.temporal_mode in ("drivemoe_20hz", "official_20hz"):
            default_history_index = [-10, -8, -6, -4, -2]
        elif self.temporal_mode in ("drivemoe_10hz", "official_10hz"):
            default_history_index = [-6, -5, -4, -3, -2]
        else:
            default_history_index = [-5, -4, -3, -2, -1]
        self.history_index = [int(v) for v in self.config.get("history_index", default_history_index)]
        if not self.history_index:
            raise ValueError("drivepi0_state.history_index must be non-empty")
        if any(idx >= 0 for idx in self.history_index):
            raise ValueError("drivepi0_state.history_index must use negative history indices")
        self.history_len = max(1, abs(min(self.history_index)))
        self.route_lookahead_index = int(self.config.get("route_lookahead_index", 25))
        self.statistics_path = Path(str(self.config["statistics_path"])).expanduser()
        self.state_source = str(self.config.get("state_source", "sensor")).lower()
        self.route_source = str(self.config.get("route_source", "auto")).lower()
        self.fallback_to_actor = bool(self.config.get("fallback_to_actor", True))
        self.gps_sensor_id = str(self.config.get("gps_sensor_id", "GPS"))
        self.imu_sensor_id = str(self.config.get("imu_sensor_id", "IMU"))
        self.speed_sensor_id = str(self.config.get("speed_sensor_id", "SPEED"))
        self._stats = self._load_stats(self.statistics_path)
        self._history = deque(maxlen=self.history_len)
        self._lat_ref: Optional[float] = None
        self._lon_ref: Optional[float] = None
        self._diagnostics = dict(self.config.get("diagnostics", {}) or {})
        self._diagnostics_count = 0

    @property
    def observation_space(self) -> spaces.Box:
        return spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self.history_index), 10),
            dtype=np.float32,
        )

    def reset(self):
        self._history.clear()
        self._lat_ref = None
        self._lon_ref = None
        self._diagnostics_count = 0

    def get_observation(self) -> Dict[str, np.ndarray]:
        self._refresh_provider_contexts()
        record = self._read_current_record()
        self._history.append(record)
        rows = [self._select_history(idx, record) for idx in self.history_index]
        current_theta = record["theta"]

        state_rows = []
        for row in rows:
            rel_command = self._world_to_ego(
                row["command_x"] - record["x"],
                row["command_y"] - record["y"],
                current_theta,
            )
            state_rows.append(
                [
                    self._norm(row["speed"], "speed"),
                    self._norm(row["acceleration"][0], "acceleration", 0),
                    self._norm(row["acceleration"][1], "acceleration", 1),
                    self._norm(row["acceleration"][2], "acceleration", 2),
                    self._norm(row["angular_velocity"][0], "angular_velocity", 0),
                    self._norm(row["angular_velocity"][1], "angular_velocity", 1),
                    self._norm(row["angular_velocity"][2], "angular_velocity", 2),
                    self._norm(self._wrap_angle(row["theta"] - current_theta), "theta"),
                    self._norm(rel_command[0], "command_far_x"),
                    self._norm(rel_command[1], "command_far_y"),
                ]
            )

        state = np.asarray(state_rows, dtype=np.float32)
        self._write_diagnostics(record, state)
        return {self.output_key: state}

    def close(self):
        self._history.clear()

    def _read_current_record(self) -> Dict[str, Any]:
        if self.state_source in ("sensor", "official", "auto"):
            try:
                return self._read_sensor_record()
            except Exception:
                if not self.fallback_to_actor:
                    raise
        return self._read_actor_record()

    def _read_sensor_record(self) -> Dict[str, Any]:
        gps_packet = get_sensor_packet(self.gps_sensor_id, CarlaDataProvider)
        imu_packet = get_sensor_packet(self.imu_sensor_id, CarlaDataProvider)
        speed_packet = get_sensor_packet(self.speed_sensor_id, CarlaDataProvider)
        if gps_packet is None or imu_packet is None or speed_packet is None:
            missing = [
                sensor_id
                for sensor_id, packet in (
                    (self.gps_sensor_id, gps_packet),
                    (self.imu_sensor_id, imu_packet),
                    (self.speed_sensor_id, speed_packet),
                )
                if packet is None
            ]
            raise RuntimeError(f"DrivePi0 state obs missing sensor packets: {missing}")

        gps = np.asarray(gps_packet.get("data"), dtype=np.float64).reshape(-1)
        imu = np.asarray(imu_packet.get("data"), dtype=np.float64).reshape(-1)
        if gps.size < 2:
            raise RuntimeError(f"DrivePi0 GPS packet has invalid shape: {gps.shape}")
        if imu.size < 7:
            raise RuntimeError(f"DrivePi0 IMU packet has invalid shape: {imu.shape}")
        sensor_frames = {
            self.gps_sensor_id: gps_packet.get("frame"),
            self.imu_sensor_id: imu_packet.get("frame"),
            self.speed_sensor_id: speed_packet.get("frame"),
        }
        if any(frame is None for frame in sensor_frames.values()) or len(set(sensor_frames.values())) != 1:
            raise RuntimeError(f"DrivePi0 state obs requires exact-frame GPS/IMU/SPEED packets: {sensor_frames}")

        speed = self._speed_from_packet(speed_packet)
        compass = float(imu[6])
        if math.isnan(compass):
            compass = 0.0
        theta = compass - math.pi / 2.0
        lat_ref, lon_ref = self._latlon_ref()
        position_xy = drivepi0_gps_to_location_xy(gps, lat_ref, lon_ref)
        command_x, command_y = self._route_command(position_xy)

        return {
            "source": "sensor",
            "sensor_frames": sensor_frames,
            "x": float(position_xy[0]),
            "y": float(position_xy[1]),
            "theta": float(theta),
            "speed": float(speed),
            "acceleration": imu[:3].astype(np.float32),
            "angular_velocity": imu[3:6].astype(np.float32),
            "command_x": float(command_x),
            "command_y": float(command_y),
        }

    def _read_actor_record(self) -> Dict[str, Any]:
        ego_actor = getattr(CarlaDataProvider, "_ego_actor", None)
        if ego_actor is None or not ego_actor.is_alive:
            raise RuntimeError("Ego actor is not available for DrivePi0 state obs")

        transform = ego_actor.get_transform()
        location = transform.location
        velocity = ego_actor.get_velocity()
        acceleration = ego_actor.get_acceleration()
        angular_velocity = ego_actor.get_angular_velocity()
        command_x, command_y = self._legacy_route_command(np.asarray([location.x, location.y], dtype=np.float32))

        return {
            "source": "actor",
            "sensor_frames": {},
            "x": float(location.x),
            "y": float(location.y),
            "theta": math.radians(float(transform.rotation.yaw)),
            "speed": float(math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)),
            "acceleration": np.asarray(
                [acceleration.x, acceleration.y, acceleration.z],
                dtype=np.float32,
            ),
            "angular_velocity": np.radians(
                np.asarray([angular_velocity.x, angular_velocity.y, angular_velocity.z], dtype=np.float32)
            ),
            "command_x": float(command_x),
            "command_y": float(command_y),
        }

    def _route_command(self, position_xy: np.ndarray) -> tuple[float, float]:
        context = getattr(CarlaDataProvider, DRIVEPI0_ROUTE_CONTEXT_ATTR, None)
        if isinstance(context, dict) and "far_node_xy" in context:
            far_node_xy = np.asarray(context["far_node_xy"], dtype=np.float32).reshape(-1)
            if far_node_xy.size >= 2:
                return float(far_node_xy[0]), float(far_node_xy[1])

        if self.route_source in ("drivepi0_route", "context", "official"):
            raise RuntimeError("DrivePi0 state observation requires CarlaDataProvider._drivepi0_route_context")

        return self._legacy_route_command(position_xy)

    def _legacy_route_command(self, position_xy: np.ndarray) -> tuple[float, float]:
        route = getattr(CarlaDataProvider, "_ego_vehicle_route", None) or []
        if not route:
            return float(position_xy[0]), float(position_xy[1])
        idx = min(max(self.route_lookahead_index, 0), len(route) - 1)
        target = route[idx][0]
        loc = target.location if hasattr(target, "location") else target.transform.location
        return float(loc.x), float(loc.y)

    @staticmethod
    def _refresh_provider_contexts() -> None:
        route_refresher = getattr(CarlaDataProvider, DRIVEPI0_ROUTE_REFRESHER_ATTR, None)
        if callable(route_refresher):
            route_refresher()

    def _latlon_ref(self) -> tuple[float, float]:
        context = getattr(CarlaDataProvider, DRIVEPI0_ROUTE_CONTEXT_ATTR, None)
        if isinstance(context, dict) and "lat_ref" in context and "lon_ref" in context:
            return float(context["lat_ref"]), float(context["lon_ref"])

        geo_reference = getattr(CarlaDataProvider, DRIVEPI0_GEO_REFERENCE_ATTR, None)
        if isinstance(geo_reference, dict) and "lat_ref" in geo_reference and "lon_ref" in geo_reference:
            return float(geo_reference["lat_ref"]), float(geo_reference["lon_ref"])

        if self._lat_ref is None or self._lon_ref is None:
            try:
                world = CarlaDataProvider.get_world()
                self._lat_ref, self._lon_ref = drivepi0_latlon_ref_from_world(world)
            except Exception:
                self._lat_ref, self._lon_ref = DEFAULT_LAT_REF, DEFAULT_LON_REF
        return float(self._lat_ref), float(self._lon_ref)

    @staticmethod
    def _speed_from_packet(speed_packet: Dict[str, Any]) -> float:
        speed_data = speed_packet.get("data")
        if isinstance(speed_data, dict) and "speed" in speed_data:
            return float(speed_data["speed"])
        array = np.asarray(speed_data, dtype=np.float64).reshape(-1)
        if array.size < 1:
            raise RuntimeError("DrivePi0 SPEED packet has no speed value")
        return float(array[0])

    def _select_history(self, idx: int, fallback: Dict) -> Dict:
        if not self._history:
            return fallback
        try:
            return self._history[idx]
        except IndexError:
            return self._history[0]

    @staticmethod
    def _world_to_ego(dx: float, dy: float, theta: float) -> tuple[float, float]:
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        return (
            cos_theta * dx + sin_theta * dy,
            -sin_theta * dx + cos_theta * dy,
        )

    @staticmethod
    def _wrap_angle(value: float) -> float:
        return (value + math.pi) % (2.0 * math.pi) - math.pi

    @staticmethod
    def _load_stats(path: Path) -> Dict:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def _norm(self, value: float, key: str, index: Optional[int] = None) -> float:
        bounds = self._stats[key]
        lo = bounds[0][index] if index is not None and isinstance(bounds[0], list) else bounds[0]
        hi = bounds[1][index] if index is not None and isinstance(bounds[1], list) else bounds[1]
        denom = float(hi) - float(lo)
        if abs(denom) < 1e-8:
            return 0.0
        return float(2.0 * (float(value) - float(lo)) / denom - 1.0)

    def _write_diagnostics(self, record: Dict[str, Any], state: np.ndarray) -> None:
        if not bool(self._diagnostics.get("enable", False)):
            return
        max_records = int(self._diagnostics.get("max_records", 20))
        if self._diagnostics_count >= max_records:
            return
        log_every = max(1, int(self._diagnostics.get("log_every", 1)))
        if self._diagnostics_count % log_every != 0:
            self._diagnostics_count += 1
            return

        payload: Dict[str, Any] = {
            "time": time.time(),
            "pid": os.getpid(),
            "output_key": self.output_key,
            "source": record.get("source"),
            "temporal_mode": self.temporal_mode,
            "history_index": self.history_index,
            "selected_sensor_frames": [row.get("sensor_frames", {}) for row in self._selected_history_rows(record)],
            "current_sensor_frames": record.get("sensor_frames", {}),
            "x": record["x"],
            "y": record["y"],
            "theta": record["theta"],
            "speed": record["speed"],
            "acceleration": np.asarray(record["acceleration"]).astype(float).tolist(),
            "angular_velocity": np.asarray(record["angular_velocity"]).astype(float).tolist(),
            "command_x": record["command_x"],
            "command_y": record["command_y"],
            "normalized_state_last": state[-1].astype(float).tolist(),
        }
        if bool(self._diagnostics.get("compare_actor", True)) and record.get("source") == "sensor":
            try:
                actor_record = self._read_actor_record()
                payload["actor_delta"] = {
                    "dx": float(actor_record["x"] - record["x"]),
                    "dy": float(actor_record["y"] - record["y"]),
                    "dtheta": float(self._wrap_angle(actor_record["theta"] - record["theta"])),
                    "dspeed": float(actor_record["speed"] - record["speed"]),
                    "dcommand_x": float(actor_record["command_x"] - record["command_x"]),
                    "dcommand_y": float(actor_record["command_y"] - record["command_y"]),
                }
            except Exception as exc:
                payload["actor_delta_error"] = str(exc)

        template = str(
            self._diagnostics.get(
                "path_template",
                "logs/drivepi0_obs_diag/drivepi0_obs_{pid}.jsonl",
            )
        )
        path = Path(template.format(pid=os.getpid(), output_key=self.output_key))
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
        self._diagnostics_count += 1

    def _selected_history_rows(self, fallback: Dict[str, Any]) -> list[Dict[str, Any]]:
        return [self._select_history(idx, fallback) for idx in self.history_index]
