"""DrivePi0-compatible route context provider."""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, Iterable, Optional, Tuple

import numpy as np

from b2d_rlinfra.environment.handlers.sensor_context import get_sensor_packet
from b2d_rlinfra.environment.model_integrations.route_geo import (
    DEFAULT_LAT_REF,
    DEFAULT_LON_REF,
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
DRIVEPI0_POSITION_OFFSET_X = -1.4
DRIVEPI0_POSITION_OFFSET_Y = 0.0
DRIVEPI0_ROUTE_CONTEXT_ATTR = "_drivepi0_route_context"
DRIVEPI0_ROUTE_REFRESHER_ATTR = "_drivepi0_route_context_refresher"
DRIVEPI0_GEO_REFERENCE_ATTR = "_drivepi0_geo_reference"


def drivepi0_gps_to_location_xy(gps: Any, lat_ref: float, lon_ref: float) -> np.ndarray:
    return gps_to_location_xy(gps, lat_ref, lon_ref, label="DrivePi0 GPS value")


def drivepi0_actor_position_xy(
    transform: Any,
    *,
    offset_x: float = DRIVEPI0_POSITION_OFFSET_X,
    offset_y: float = DRIVEPI0_POSITION_OFFSET_Y,
) -> np.ndarray:
    return actor_position_xy(transform, offset_x=offset_x, offset_y=offset_y)


def drivepi0_latlon_ref_from_world(world: Any) -> Tuple[float, float]:
    return latlon_ref_from_world(world)


def drivepi0_rollout_latlon_ref(world_entry: Tuple[Any, Any], gps_entry: Tuple[Any, Any]) -> Tuple[float, float]:
    return rollout_latlon_ref(world_entry, gps_entry, label="DrivePi0 rollout")


class DrivePi0RoutePlanner:
    """Small local port of DrivePi0's official RoutePlanner route-deque logic."""

    def __init__(self, min_distance: float = 7.5, max_distance: float = 25.0):
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
                xy = drivepi0_gps_to_location_xy(gps_latlon(point, label="DrivePi0 GPS route point"), lat_ref, lon_ref)
            else:
                xy = location_xy(point, label="DrivePi0 route point")
            self.route.append((xy, command))

    def reset(self) -> None:
        self.route.clear()

    def run_step(self, position_xy: Any) -> RouteEntry:
        if not self.route:
            raise RuntimeError("DrivePi0RoutePlanner requires a non-empty route")
        if len(self.route) == 1:
            return self.route[0]

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

        return self.route[1]

    def context_for_position(self, position_xy: Any) -> Dict[str, Any]:
        far_node = self.run_step(position_xy)
        return {
            "far_node_xy": far_node[0].astype(np.float32, copy=True),
            "far_command": far_node[1],
            "remaining_route_points": int(len(self.route)),
        }


class DrivePi0RouteContextWrapper:
    """Attach official-style DrivePi0 route context to CarlaDataProvider."""

    def __init__(self, env: Any, config: Optional[Dict[str, Any]] = None):
        self.env = env
        self.config = getattr(self.env, "config", {})
        self.route_config = dict(config or {})
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.planner = DrivePi0RoutePlanner(
            min_distance=float(self.route_config.get("min_distance", 7.5)),
            max_distance=float(self.route_config.get("max_distance", 25.0)),
        )
        self.position_offset_x = float(self.route_config.get("position_offset_x", DRIVEPI0_POSITION_OFFSET_X))
        self.position_offset_y = float(self.route_config.get("position_offset_y", DRIVEPI0_POSITION_OFFSET_Y))
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
        return self.env.step(action)

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
        if not route and CarlaDataProvider is not None:
            route = getattr(CarlaDataProvider, "_ego_vehicle_route", None) or getattr(CarlaDataProvider, "_vehicle_route", None)
        if not route:
            raise RuntimeError("DrivePi0 route context requires a dense scenario route")

        route_list = list(route)
        if gps_route is not None:
            gps_route_list = list(gps_route)
            if gps_route_list:
                try:
                    self._lat_ref, self._lon_ref = drivepi0_rollout_latlon_ref(route_list[0], gps_route_list[0])
                    self.planner.set_route(
                        gps_route_list,
                        gps=True,
                        lat_ref=self._lat_ref,
                        lon_ref=self._lon_ref,
                    )
                    return
                except Exception:
                    self._refresh_latlon_ref()

        self.planner.set_route(route_list)

    def _reference_xy(self) -> np.ndarray:
        gps_packet = get_sensor_packet(self.gps_sensor_id, CarlaDataProvider) if CarlaDataProvider is not None else None
        if gps_packet is not None:
            gps = np.asarray(gps_packet.get("data"), dtype=np.float64).reshape(-1)
            if gps.size >= 2:
                return drivepi0_gps_to_location_xy(gps, self._lat_ref, self._lon_ref)

        if CarlaDataProvider is None:
            raise RuntimeError("srunner CarlaDataProvider is not available for DrivePi0 route context")
        ego_actor = getattr(CarlaDataProvider, "_ego_actor", None)
        if ego_actor is None or not getattr(ego_actor, "is_alive", True):
            raise RuntimeError("Ego actor is not available for DrivePi0 route context")
        transform = CarlaDataProvider.get_transform(ego_actor)
        return drivepi0_actor_position_xy(
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
        setattr(CarlaDataProvider, DRIVEPI0_ROUTE_CONTEXT_ATTR, context)
        setattr(
            CarlaDataProvider,
            DRIVEPI0_GEO_REFERENCE_ATTR,
            {"lat_ref": float(self._lat_ref), "lon_ref": float(self._lon_ref)},
        )

    def _register_context_refreshers(self) -> None:
        if CarlaDataProvider is not None:
            setattr(CarlaDataProvider, DRIVEPI0_ROUTE_REFRESHER_ATTR, self._publish_context)

    def _refresh_latlon_ref(self) -> None:
        if CarlaDataProvider is None:
            self._lat_ref = DEFAULT_LAT_REF
            self._lon_ref = DEFAULT_LON_REF
            return
        try:
            world = CarlaDataProvider.get_world()
            self._lat_ref, self._lon_ref = drivepi0_latlon_ref_from_world(world) if world is not None else (
                DEFAULT_LAT_REF,
                DEFAULT_LON_REF,
            )
        except Exception:
            self._lat_ref = DEFAULT_LAT_REF
            self._lon_ref = DEFAULT_LON_REF

    @staticmethod
    def _clear_context() -> None:
        if CarlaDataProvider is not None:
            setattr(CarlaDataProvider, DRIVEPI0_ROUTE_CONTEXT_ATTR, None)
            setattr(CarlaDataProvider, DRIVEPI0_ROUTE_REFRESHER_ATTR, None)
            setattr(CarlaDataProvider, DRIVEPI0_GEO_REFERENCE_ATTR, None)
