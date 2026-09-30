from __future__ import annotations

from typing import Dict

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider


class ProviderCompat:
    _SPECIAL_MAP_NAMES = (
        "_actor_obstacle_map",
        "_actor_emergency_map",
        "_actor_dooropen_map",
        "_actor_walker_blocker_map",
        "_actor_dirty_map",
    )

    def bootstrap(self) -> None:
        if not hasattr(CarlaDataProvider, "_ego_actor"):
            CarlaDataProvider._ego_actor = None
        if not hasattr(CarlaDataProvider, "_ego_vehicle_route"):
            CarlaDataProvider._ego_vehicle_route = None
        if not hasattr(CarlaDataProvider, "_ego_vehicle_route_full"):
            CarlaDataProvider._ego_vehicle_route_full = None
        if not hasattr(CarlaDataProvider, "_distance_traveled_on_tick"):
            CarlaDataProvider._distance_traveled_on_tick = 0.0
        if not hasattr(CarlaDataProvider, "_distance_traveled_on_tick_reward"):
            CarlaDataProvider._distance_traveled_on_tick_reward = 0.0

        for map_name in self._SPECIAL_MAP_NAMES:
            current = getattr(CarlaDataProvider, map_name, None)
            if current is None:
                setattr(CarlaDataProvider, map_name, {})

    def _prune_actor_maps(self) -> None:
        for map_name in (
            "_actor_obstacle_map",
            "_actor_emergency_map",
            "_actor_dooropen_map",
            "_actor_dirty_map",
        ):
            actor_map: Dict = getattr(CarlaDataProvider, map_name, {})
            for actor in list(actor_map.keys()):
                if actor is None or not actor.is_alive:
                    actor_map.pop(actor, None)

        walker_map: Dict = getattr(CarlaDataProvider, "_actor_walker_blocker_map", {})
        for walker, blocker in list(walker_map.items()):
            if walker is None or blocker is None or not walker.is_alive or not blocker.is_alive:
                walker_map.pop(walker, None)

    def refresh(self):
        self.bootstrap()
        self._prune_actor_maps()

        hero_actor = CarlaDataProvider.get_hero_actor()
        if hero_actor is not None and hero_actor.is_alive:
            CarlaDataProvider._ego_actor = hero_actor
        elif getattr(CarlaDataProvider, "_ego_actor", None) is not None:
            ego_actor = CarlaDataProvider._ego_actor
            if ego_actor is None or not ego_actor.is_alive:
                CarlaDataProvider._ego_actor = None

        return getattr(CarlaDataProvider, "_ego_actor", None)

    def cleanup(self) -> None:
        self.bootstrap()
        CarlaDataProvider._ego_actor = None
        CarlaDataProvider._ego_vehicle_route = None
        CarlaDataProvider._ego_vehicle_route_full = None
        CarlaDataProvider._distance_traveled_on_tick = 0.0
        CarlaDataProvider._distance_traveled_on_tick_reward = 0.0
        for map_name in self._SPECIAL_MAP_NAMES:
            getattr(CarlaDataProvider, map_name, {}).clear()
