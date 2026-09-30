from __future__ import annotations

from typing import Iterable, List, Optional

import carla

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider


class RouteTracker:
    def __init__(self, window_size: int = 5):
        self.window_size = int(window_size)

    def reset(self, route: Optional[Iterable]) -> None:
        route_list = list(route) if route else []
        CarlaDataProvider._ego_vehicle_route = route_list
        CarlaDataProvider._ego_vehicle_route_full = list(route_list)
        CarlaDataProvider._distance_traveled_on_tick = 0.0
        CarlaDataProvider._distance_traveled_on_tick_reward = 0.0

        ego_actor = getattr(CarlaDataProvider, "_ego_actor", None)
        if ego_actor is not None and ego_actor.is_alive:
            ego_actor.route_plan = CarlaDataProvider._ego_vehicle_route

    def _find_advance_index(self, vehicle_route: List, ego_location: carla.Location) -> int:
        closest_idx = 0
        if not vehicle_route:
            return closest_idx

        for index in range(len(vehicle_route) - 1):
            if index > self.window_size:
                break

            loc0 = vehicle_route[index][0].location
            loc1 = vehicle_route[index + 1][0].location
            waypoint_dir = carla.Vector3D(loc1.x - loc0.x, loc1.y - loc0.y)
            waypoint_to_ego = carla.Vector3D(ego_location.x - loc0.x, ego_location.y - loc0.y)
            dot_value = waypoint_to_ego.x * waypoint_dir.x + waypoint_to_ego.y * waypoint_dir.y
            if dot_value > 0:
                closest_idx = index + 1

        return closest_idx

    @staticmethod
    def _length_traveled(route_segment: List) -> float:
        route_length = 0.0
        for index in range(len(route_segment) - 1):
            route_length += route_segment[index][0].location.distance(route_segment[index + 1][0].location)
        return route_length

    def update(self, ego_actor) -> float:
        vehicle_route = getattr(CarlaDataProvider, "_ego_vehicle_route", None)
        if ego_actor is None or not ego_actor.is_alive or not vehicle_route:
            CarlaDataProvider._distance_traveled_on_tick = 0.0
            CarlaDataProvider._distance_traveled_on_tick_reward = 0.0
            return 0.0

        ego_location = ego_actor.get_location()
        closest_idx = self._find_advance_index(vehicle_route, ego_location)
        distance_traveled = self._length_traveled(vehicle_route[: closest_idx + 1])

        CarlaDataProvider._ego_vehicle_route = vehicle_route[closest_idx:]
        CarlaDataProvider._distance_traveled_on_tick = distance_traveled
        CarlaDataProvider._distance_traveled_on_tick_reward = distance_traveled
        ego_actor.route_plan = CarlaDataProvider._ego_vehicle_route
        return distance_traveled
