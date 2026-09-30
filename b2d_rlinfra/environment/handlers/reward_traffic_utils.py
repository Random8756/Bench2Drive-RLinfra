import math

import carla
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from b2d_rlinfra.environment.handlers.criteria.rl_utils import get_traffic_light_waypoints as rl_get_traffic_light_waypoints

TL_LOOKAHEAD_M = 20.0
TL_SECOND_RED_DISTANCE_M = 20.0
TL_CLEAR_DISTANCE_M = 8.0
TL_STOP_SPEED_THRESHOLD_MPS = 0.2
TL_RED_LIGHT_DECEL_DISTANCE_M = 8.0


def compute_2d_distance(loc1, loc2):
    return math.sqrt((loc1.x - loc2.x) ** 2 + (loc1.y - loc2.y) ** 2)


def get_light_reference_location(traffic_light):
    try:
        return traffic_light.get_transform().transform(traffic_light.trigger_volume.location)
    except Exception:
        return traffic_light.get_transform().location


def point_inside_boundingbox(point, bb_center, bb_extent, multiplier=1.2):
    A = carla.Vector2D(bb_center.x - multiplier * bb_extent.x, bb_center.y - multiplier * bb_extent.y)
    B = carla.Vector2D(bb_center.x + multiplier * bb_extent.x, bb_center.y - multiplier * bb_extent.y)
    D = carla.Vector2D(bb_center.x - multiplier * bb_extent.x, bb_center.y + multiplier * bb_extent.y)
    M = carla.Vector2D(point.x, point.y)

    AB = B - A
    AD = D - A
    AM = M - A
    am_ab = AM.x * AB.x + AM.y * AB.y
    ab_ab = AB.x * AB.x + AB.y * AB.y
    am_ad = AM.x * AD.x + AM.y * AD.y
    ad_ad = AD.x * AD.x + AD.y * AD.y
    return am_ab > 0 and am_ab < ab_ab and am_ad > 0 and am_ad < ad_ad


def build_route_lookahead_locations(vehicle_route, lookahead_m):
    if not vehicle_route:
        return []

    points = [vehicle_route[0][0].location]
    if lookahead_m <= 0:
        return points

    cumulative = 0.0
    for idx in range(1, len(vehicle_route)):
        prev_loc = vehicle_route[idx - 1][0].location
        curr_loc = vehicle_route[idx][0].location
        cumulative += compute_2d_distance(prev_loc, curr_loc)
        points.append(curr_loc)
        if cumulative >= lookahead_m:
            break
    return points


def is_effectively_active(actor):
    try:
        return bool(actor.is_active)
    except Exception:
        return bool(getattr(actor, 'is_alive', False))


def refresh_traffic_lights(world, carla_map, cached_traffic_lights,
                           ego_loc=None, max_radius=250.0, actors_candidates=None):
    """Incrementally refresh traffic-light cache.

    When *ego_loc* is provided, cached entries beyond *max_radius* are pruned
    and only active actors within *max_radius* are scanned for new additions.
    """
    if world is None or carla_map is None:
        return []

    r2 = max_radius ** 2 if ego_loc is not None else None

    # 1. keep alive entries; optionally prune by distance
    refreshed = []
    for actor, center, waypoints in cached_traffic_lights:
        if not getattr(actor, 'is_alive', False):
            continue
        if r2 is not None and center is not None:
            dx = center.x - ego_loc.x
            dy = center.y - ego_loc.y
            if dx * dx + dy * dy > r2:
                continue
        refreshed.append((actor, center, waypoints))

    known_light_ids = {int(actor.id) for actor, _, _ in refreshed}

    # 2. scan actors — active only, within radius
    if actors_candidates is not None:
        source_actors = actors_candidates
    elif world is CarlaDataProvider.get_world():
        source_actors = CarlaDataProvider.get_all_actors()
    else:
        source_actors = world.get_actors()
    for actor in source_actors:
        if not is_effectively_active(actor):
            continue
        if 'traffic_light' not in actor.type_id:
            continue
        actor_id = int(actor.id)
        if actor_id in known_light_ids:
            continue
        if r2 is not None:
            loc = actor.get_location()
            dx = loc.x - ego_loc.x
            dy = loc.y - ego_loc.y
            if dx * dx + dy * dy > r2:
                continue
        center, waypoints = rl_get_traffic_light_waypoints(actor, carla_map)
        refreshed.append((actor, center, waypoints))
        known_light_ids.add(actor_id)

    return refreshed


def find_traffic_light_entry_by_id(traffic_lights, light_id):
    if light_id is None:
        return None
    for traffic_light, center, waypoints in traffic_lights:
        if not getattr(traffic_light, 'is_alive', False):
            continue
        if int(traffic_light.id) == int(light_id):
            return traffic_light, center, waypoints
    return None


def is_light_affecting_route(traffic_light, center, lookahead_locations, extra_locations=None):
    points = []
    if lookahead_locations:
        points.extend(lookahead_locations)
    if extra_locations:
        points.extend([loc for loc in extra_locations if loc is not None])
    if not points:
        return False

    for loc in points:
        if point_inside_boundingbox(loc, center, traffic_light.trigger_volume.extent):
            return True
    return False


def get_relevant_traffic_lights(
    traffic_lights,
    lookahead_locations,
    ego_location=None,
    max_distance=None,
    extra_locations=None,
):
    relevant = []
    for traffic_light, center, waypoints in traffic_lights:
        if not getattr(traffic_light, 'is_alive', False):
            continue
        if not is_light_affecting_route(
            traffic_light=traffic_light,
            center=center,
            lookahead_locations=lookahead_locations,
            extra_locations=extra_locations,
        ):
            continue
        distance = compute_2d_distance(ego_location, center) if ego_location is not None else 0.0
        if max_distance is not None and distance > max_distance:
            continue
        relevant.append((traffic_light, center, waypoints, distance))

    if ego_location is not None:
        relevant.sort(key=lambda item: item[3])
    return relevant


def get_first_route_light(
    traffic_lights,
    lookahead_locations,
    ego_location=None,
    max_distance=None,
):
    if not lookahead_locations:
        return None

    best_item = None
    best_index = None
    for traffic_light, center, waypoints in traffic_lights:
        if not getattr(traffic_light, 'is_alive', False):
            continue

        first_hit_idx = None
        for idx, loc in enumerate(lookahead_locations):
            if point_inside_boundingbox(loc, center, traffic_light.trigger_volume.extent):
                first_hit_idx = idx
                break
        if first_hit_idx is None:
            continue

        distance = compute_2d_distance(ego_location, center) if ego_location is not None else 0.0
        if max_distance is not None and distance > max_distance:
            continue

        if best_index is None or first_hit_idx < best_index:
            best_index = first_hit_idx
            best_item = (traffic_light, center, waypoints, first_hit_idx, distance)

    return best_item
