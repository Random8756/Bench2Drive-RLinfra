"""MindDrive-compatible route context provider.

This module intentionally lives in ``b2d_rlinfra.environment.model_integrations``: it exposes
environment-side context needed by a specific pretrained model while leaving
policy adapters and training code outside the env layer.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple

import numpy as np

from b2d_rlinfra.environment.handlers.sensor_context import get_sensor_packet
from b2d_rlinfra.environment.model_integrations.route_geo import (
    DEFAULT_LAT_REF,
    DEFAULT_LON_REF,
    EARTH_RADIUS_EQUA,
    actor_position_xy,
    gps_latlon,
    gps_to_location_xy,
    latlon_ref_from_world,
    location_xy,
    rollout_latlon_ref,
)

try:
    from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
except ImportError:  # pragma: no cover - optional for no-CARLA unit tests
    CarlaDataProvider = None

RouteEntry = Tuple[np.ndarray, Any]
MINDDRIVE_POSITION_OFFSET_X = -1.4
MINDDRIVE_POSITION_OFFSET_Y = 0.0
MINDDRIVE_GEO_REFERENCE_ATTR = "_minddrive_geo_reference"


def _location_xy(value: Any) -> np.ndarray:
    """Extract CARLA world x/y from route points, waypoints, or locations."""
    return location_xy(value, label="MindDrive route point")


def minddrive_actor_position_xy(
    transform: Any,
    *,
    offset_x: float = MINDDRIVE_POSITION_OFFSET_X,
    offset_y: float = MINDDRIVE_POSITION_OFFSET_Y,
) -> np.ndarray:
    """Return the official MindDrive GNSS-reference world x/y for an actor pose."""
    return actor_position_xy(transform, offset_x=offset_x, offset_y=offset_y)


def minddrive_gps_to_location_xy(gps: Any, lat_ref: float, lon_ref: float) -> np.ndarray:
    """Project GNSS latitude/longitude into the B2D/MindDrive world x/y frame."""
    return gps_to_location_xy(gps, lat_ref, lon_ref, label="MindDrive GPS value")


def _gps_latlon(value: Any) -> np.ndarray:
    return gps_latlon(value, label="MindDrive GPS route point")


def _latlon_ref_from_world(world: Any) -> Tuple[float, float]:
    """Read CARLA OpenDRIVE geoReference, matching leaderboard route projection."""
    return latlon_ref_from_world(world)


def _minddrive_rollout_latlon_ref(world_entry: Tuple[Any, Any], gps_entry: Tuple[Any, Any]) -> Tuple[float, float]:
    """Match MindDrive rollout's first waypoint GPS/world georef solve."""
    return rollout_latlon_ref(world_entry, gps_entry, label="MindDrive rollout")


def _route_option_name(option: Any) -> str:
    return str(getattr(option, "name", option)).upper()


def _is_lane_change_option(option: Any) -> bool:
    name = _route_option_name(option)
    return name.endswith("CHANGELANELEFT") or name.endswith("CHANGELANERIGHT")


def downsample_route_indices(route: Iterable[Tuple[Any, Any]], sample_factor: float) -> List[int]:
    """Port of leaderboard ``downsample_route`` used by MindDrive rollout/eval."""
    route_list = list(route)
    if not route_list:
        return []
    sample_factor = float(sample_factor)
    ids_to_sample: List[int] = []
    prev_option = None
    distance_since_sample = 0.0

    for index, point in enumerate(route_list):
        curr_option = point[1]
        is_lane_change = _is_lane_change_option(curr_option)
        prev_is_lane_change = _is_lane_change_option(prev_option)

        if prev_option is None:
            ids_to_sample.append(index)
            distance_since_sample = 0.0
        elif is_lane_change:
            ids_to_sample.append(index)
            distance_since_sample = 0.0
        elif prev_option != curr_option and not prev_is_lane_change:
            ids_to_sample.append(index)
            distance_since_sample = 0.0
        elif distance_since_sample > sample_factor:
            ids_to_sample.append(index)
            distance_since_sample = 0.0
        elif index == len(route_list) - 1:
            ids_to_sample.append(index)
            distance_since_sample = 0.0
        else:
            distance_since_sample += float(np.linalg.norm(_location_xy(point[0]) - _location_xy(route_list[index - 1][0])))

        prev_option = curr_option

    return ids_to_sample


def downsample_route(route: Iterable[Tuple[Any, Any]], sample_factor: float) -> List[Tuple[Any, Any]]:
    route_list = list(route)
    return [route_list[index] for index in downsample_route_indices(route_list, sample_factor)]


class MindDriveRoutePlanner:
    """Small local port of MindDrive's eval RoutePlanner route-deque logic."""

    def __init__(self, min_distance: float = 4.0, max_distance: float = 50.0):
        self.min_distance = float(min_distance)
        self.max_distance = float(max_distance)
        self.route: Deque[RouteEntry] = deque()

    def set_route(
        self,
        route: Iterable[Tuple[Any, Any]],
        *,
        gps: bool = False,
        lat_ref: float = DEFAULT_LAT_REF,
        lon_ref: float = DEFAULT_LON_REF,
    ) -> None:
        self.route.clear()
        for point, command in route:
            if gps:
                xy = minddrive_gps_to_location_xy(_gps_latlon(point), lat_ref, lon_ref)
            else:
                xy = _location_xy(point)
            self.route.append((xy, command))

    def reset(self) -> None:
        self.route.clear()

    def run_step(self, position_xy: Any) -> Tuple[RouteEntry, RouteEntry]:
        if not self.route:
            raise RuntimeError("MindDriveRoutePlanner requires a non-empty route")

        if len(self.route) == 1:
            current = self.route[0]
            return current, current

        gps = np.asarray(position_xy, dtype=np.float32).reshape(-1)[:2]
        to_pop = 0
        farthest_in_range = -np.inf
        cumulative_distance = 0.0

        for index in range(1, len(self.route)):
            if cumulative_distance > self.max_distance:
                break

            cumulative_distance += float(np.linalg.norm(self.route[index][0] - self.route[index - 1][0]))
            distance = float(np.linalg.norm(self.route[index][0] - gps))

            if distance <= self.min_distance and distance > farthest_in_range:
                farthest_in_range = distance
                to_pop = index

        for _ in range(to_pop):
            if len(self.route) > 2:
                self.route.popleft()

        return self.route[0], self.route[1]

    def context_for_position(self, position_xy: Any) -> Dict[str, Any]:
        current, near = self.run_step(position_xy)
        return {
            "curr_node_xy": current[0].astype(np.float32, copy=True),
            "curr_command": current[1],
            "near_node_xy": near[0].astype(np.float32, copy=True),
            "near_command": near[1],
            "remaining_route_points": int(len(self.route)),
        }


class MindDriveRouteContextWrapper:
    """Attach official-style MindDrive route context to CarlaDataProvider."""

    def __init__(self, env: Any, config: Optional[Dict[str, Any]] = None):
        self.env = env
        self.config = getattr(self.env, "config", {})
        self.route_config = dict(config or {})
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.planner = MindDriveRoutePlanner(
            min_distance=float(self.route_config.get("min_distance", 4.0)),
            max_distance=float(self.route_config.get("max_distance", 50.0)),
        )
        self.position_offset_x = float(self.route_config.get("position_offset_x", MINDDRIVE_POSITION_OFFSET_X))
        self.position_offset_y = float(self.route_config.get("position_offset_y", MINDDRIVE_POSITION_OFFSET_Y))
        self.downsample_factor = float(self.route_config.get("downsample_factor", 1.0))
        self.gps_sensor_id = str(self.route_config.get("gps_sensor_id", "GPS"))
        self._lat_ref = DEFAULT_LAT_REF
        self._lon_ref = DEFAULT_LON_REF

    def __getattr__(self, name: str) -> Any:
        if name == "env":
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        return getattr(self.env, name)

    def reset(self, seed=None, options=None):
        self._clear_context()
        observation, info = self.env.reset(seed=seed, options=options)
        self._register_context_refreshers()
        self._refresh_latlon_ref()
        self._install_route_from_current_scenario()
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return observation, reward, terminated, truncated, info

    def close(self) -> None:
        self._clear_context()
        close = getattr(self.env, "close", None)
        if callable(close):
            close()

    def _install_route_from_current_scenario(self) -> None:
        self.planner.reset()
        scenario = getattr(self.env, "scenario", None)
        route = getattr(scenario, "route", None)
        gps_route = getattr(scenario, "gps_route", None)
        if not route:
            route = getattr(CarlaDataProvider, "_vehicle_route", None) if CarlaDataProvider is not None else None
        if not route:
            raise RuntimeError("MindDrive route context requires a dense scenario route")

        route_list = list(route)
        sample_ids = downsample_route_indices(route_list, self.downsample_factor)
        sampled_world_route = [route_list[index] for index in sample_ids]

        if gps_route is not None:
            gps_route_list = list(gps_route)
            if len(gps_route_list) >= len(route_list):
                sampled_gps_route = [gps_route_list[index] for index in sample_ids]
                try:
                    self._lat_ref, self._lon_ref = _minddrive_rollout_latlon_ref(
                        sampled_world_route[0],
                        sampled_gps_route[0],
                    )
                    self.planner.set_route(
                        sampled_gps_route,
                        gps=True,
                        lat_ref=self._lat_ref,
                        lon_ref=self._lon_ref,
                    )
                    return
                except Exception:
                    self._refresh_latlon_ref()

        self.planner.set_route(sampled_world_route)

    def _reference_xy(self) -> np.ndarray:
        gps_packet = get_sensor_packet(self.gps_sensor_id, CarlaDataProvider) if CarlaDataProvider is not None else None
        if gps_packet is not None:
            gps = np.asarray(gps_packet.get("data"), dtype=np.float64).reshape(-1)
            if gps.size >= 2:
                return minddrive_gps_to_location_xy(gps, self._lat_ref, self._lon_ref)

        sensor_context = getattr(CarlaDataProvider, "_minddrive_sensor_context", None) if CarlaDataProvider is not None else None
        if isinstance(sensor_context, dict) and "position_xy" in sensor_context:
            position_xy = np.asarray(sensor_context["position_xy"], dtype=np.float32).reshape(-1)
            if position_xy.size >= 2:
                return position_xy[:2].astype(np.float32, copy=True)

        if CarlaDataProvider is None:
            raise RuntimeError("srunner CarlaDataProvider is not available for MindDrive route context")
        ego_actor = getattr(CarlaDataProvider, "_ego_actor", None)
        if ego_actor is None or not getattr(ego_actor, "is_alive", True):
            raise RuntimeError("Ego actor is not available for MindDrive route context")
        transform = CarlaDataProvider.get_transform(ego_actor)
        return minddrive_actor_position_xy(
            transform,
            offset_x=self.position_offset_x,
            offset_y=self.position_offset_y,
        )

    def _publish_context(self) -> None:
        if CarlaDataProvider is None:
            return
        context = self.planner.context_for_position(self._reference_xy())
        context["lat_ref"] = float(self._lat_ref)
        context["lon_ref"] = float(self._lon_ref)
        setattr(CarlaDataProvider, "_minddrive_route_context", context)
        setattr(
            CarlaDataProvider,
            MINDDRIVE_GEO_REFERENCE_ATTR,
            {"lat_ref": float(self._lat_ref), "lon_ref": float(self._lon_ref)},
        )

    def _register_context_refreshers(self) -> None:
        if CarlaDataProvider is not None:
            setattr(CarlaDataProvider, "_minddrive_route_context_refresher", self._publish_context)

    def _refresh_latlon_ref(self) -> None:
        if CarlaDataProvider is None:
            self._lat_ref = DEFAULT_LAT_REF
            self._lon_ref = DEFAULT_LON_REF
            return
        try:
            world = CarlaDataProvider.get_world()
            self._lat_ref, self._lon_ref = _latlon_ref_from_world(world) if world is not None else (
                DEFAULT_LAT_REF,
                DEFAULT_LON_REF,
            )
        except Exception:
            self._lat_ref = DEFAULT_LAT_REF
            self._lon_ref = DEFAULT_LON_REF

    @staticmethod
    def _clear_context() -> None:
        if CarlaDataProvider is not None:
            setattr(CarlaDataProvider, "_minddrive_route_context", None)
            setattr(CarlaDataProvider, "_minddrive_route_context_refresher", None)
            setattr(CarlaDataProvider, MINDDRIVE_GEO_REFERENCE_ATTR, None)
