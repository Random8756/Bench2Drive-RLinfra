"""Termination handler - PPO-tuned variant.

Detects custom termination conditions beyond the Leaderboard native py-tree events.

Detected conditions:
1. ``not_in_carlane`` - left the drivable road.
2. ``useless_move_timeout`` - meaningless looping / spinning in place.
3. ``second_red_same_light`` - ran the same red light twice.

Note: ``route_deviation`` (``deviation_threshold``) is not a termination
condition, it is only used by reward computation.
"""

import carla
import math
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.timer import GameTime
from b2d_rlinfra.environment.handlers.reward_traffic_utils import (
    TL_CLEAR_DISTANCE_M,
    TL_LOOKAHEAD_M,
    TL_SECOND_RED_DISTANCE_M,
    TL_STOP_SPEED_THRESHOLD_MPS,
    build_route_lookahead_locations as tl_utils_build_route_lookahead_locations,
    get_first_route_light as tl_utils_get_first_route_light,
    refresh_traffic_lights as tl_utils_refresh_traffic_lights,
)
from b2d_rlinfra.environment.info_keys import WRAPPER_ROUTE_DISTANCE_ON_TICK

__layer__ = (2, "Environment")


def compute_2d_distance(loc1, loc2):
    """Compute 2D distance between two CARLA locations."""
    return math.sqrt((loc1.x - loc2.x)**2 + (loc1.y - loc2.y)**2)


class TerminationHandler:
    """
    PPO-tuned custom termination detector.

    Attributes:
        off_road_threshold (float): off-road tolerance in meters.
        blocked_time_out_limit (float): timeout for no useful movement.
    """

    REASON_NOT_IN_CARLANE = 'not_in_carlane'
    REASON_USELESS_MOVE_TIMEOUT = 'useless_move_timeout'
    REASON_SECOND_RED_SAME_LIGHT = 'second_red_same_light'
    REASON_SUCCESS = 'success'
    
    def __init__(self, config):
        """
        Args:
            config: project config with optional ``environment.termination``
                thresholds.
        """
        self.config = config

        env_config = config.get('environment', {})
        termination_config = env_config.get('termination', {})
        self.off_road_threshold = termination_config.get('off_road_threshold', 1.5)
        self.blocked_time_out_limit = termination_config.get('blocked_time_out_limit', 30.0)
        self.success_percent_threshold = termination_config.get('success_percent_threshold', 99.9)
        self.second_red_light_distance_threshold = termination_config.get(
            'second_red_light_distance_threshold', TL_SECOND_RED_DISTANCE_M
        )
        self.second_red_light_clear_distance = termination_config.get(
            'second_red_light_clear_distance', TL_CLEAR_DISTANCE_M
        )
        self.second_red_stop_speed_threshold = termination_config.get(
            'second_red_stop_speed_threshold', TL_STOP_SPEED_THRESHOLD_MPS
        )
        self.second_red_lookahead_m = termination_config.get(
            'second_red_lookahead_m', TL_LOOKAHEAD_M
        )
        
        self._record_useful_move_time = 0.0
        self._record_len = 0
        self._tracked_traffic_light_id = None
        self._tracked_traffic_light_location = None
        self._tracked_traffic_light_min_distance = None
        self._stopped_on_red = False
        self._seen_green_after_stop = False
        self._list_traffic_lights = []
        
        self._world = None
        self._map = None
        
    def reset(self):
        """Reset state at the start of each episode."""
        self._record_useful_move_time = GameTime.get_time()
        self._record_len = 0
        self._tracked_traffic_light_id = None
        self._tracked_traffic_light_location = None
        self._tracked_traffic_light_min_distance = None
        self._stopped_on_red = False
        self._seen_green_after_stop = False
        self._list_traffic_lights = []
        self._world = None
        self._map = None
        
    def _ensure_world_map(self):
        """Initialize cached world and map references."""
        if self._world is None:
            self._world = CarlaDataProvider.get_world()
        if self._map is None:
            self._map = CarlaDataProvider.get_map()
    
    def check_all(self, info: dict) -> dict:
        """
        Check all configured custom termination conditions.

        Args:
            info: current step state. Must contain:
                - WRAPPER_ROUTE_DISTANCE_ON_TICK ('wrapper/route/distance_on_tick'):
                  distance traveled this frame, written by RoutePlanWrapper.

        Returns:
            dict: {
                'should_terminate': bool,
                'triggered_events': list[dict]
            }

            Each triggered event has:
            {
                'event_type': str,
                'source': 'custom',
                'reason': str,
                'details': dict
            }
        """
        self._ensure_world_map()
        
        result = {
            'should_terminate': False,
            'triggered_events': []
        }
        
        # ``route_deviation`` is reward-only, not a termination condition.
        checks = [
            (self.check_not_in_carlane, 'NOT_IN_CARLANE'),
            (self.check_useless_move_timeout, 'USELESS_MOVE_TIMEOUT'),
            (self.check_second_red_same_light, 'SECOND_RED_SAME_LIGHT'),
            (self.check_success, 'SUCCESS'),
        ]
        
        for check_func, event_type in checks:
            check_result = check_func(info)
            if check_result['triggered']:
                result['should_terminate'] = True
                result['triggered_events'].append({
                    'event_type': event_type,
                    'source': 'custom',
                    'reason': check_result['reason'],
                    'details': check_result['details']
                })
        
        return result
    
    def check_not_in_carlane(self, info: dict) -> dict:
        """Check whether the ego vehicle has left the drivable road."""
        result = {
            'triggered': False,
            'reason': self.REASON_NOT_IN_CARLANE,
            'details': {}
        }
        
        ego_actor = CarlaDataProvider._ego_actor
        if ego_actor is None or not ego_actor.is_alive:
            return result
            
        self._ensure_world_map()
        
        ev_transform = ego_actor.get_transform()
        ego_wp = self._map.get_waypoint(ev_transform.location)
        if ego_wp is None:
            result['triggered'] = True
            result['details']['waypoint_found'] = False
            result['details']['reason'] = 'no_nearest_waypoint'
            return result
        
        # Junction lane geometry is ambiguous; skip this check there.
        if ego_wp.is_junction:
            result['details']['in_junction'] = True
            return result
            
        location = ev_transform.location
        driving_wp = self._map.get_waypoint(location, lane_type=carla.LaneType.Driving)
        parking_wp = self._map.get_waypoint(location, lane_type=carla.LaneType.Parking)
        
        lane_candidates = []
        if driving_wp is not None:
            lane_candidates.append((
                'driving',
                compute_2d_distance(location, driving_wp.transform.location),
                driving_wp.lane_width,
            ))
        if parking_wp is not None:
            lane_candidates.append((
                'parking',
                compute_2d_distance(location, parking_wp.transform.location),
                parking_wp.lane_width,
            ))

        if not lane_candidates:
            result['triggered'] = True
            result['details']['lane_reference_found'] = False
            result['details']['reason'] = 'no_driving_or_parking_waypoint'
            return result

        lane_type, distance, lane_width = min(lane_candidates, key=lambda item: item[1])
        
        result['details']['distance_to_lane'] = distance
        result['details']['lane_width'] = lane_width
        result['details']['lane_type'] = lane_type
        result['details']['threshold'] = lane_width / 2 + self.off_road_threshold
        
        if distance > (lane_width / 2 + self.off_road_threshold):
            result['triggered'] = True

        return result
    
    def check_useless_move_timeout(self, info: dict) -> dict:
        """Check whether the ego vehicle made no useful route progress for too long."""
        result = {
            'triggered': False,
            'reason': self.REASON_USELESS_MOVE_TIMEOUT,
            'details': {}
        }
        
        # Strict read exposes wrapper-ordering errors immediately.
        distance_traveled = info[WRAPPER_ROUTE_DISTANCE_ON_TICK]
        timenow = GameTime.get_time()
        
        if self._record_len < 1:
            self._record_useful_move_time = timenow
        elif distance_traveled < 1:
            time_since_useful_move = timenow - self._record_useful_move_time
            result['details']['time_since_useful_move'] = time_since_useful_move
            result['details']['threshold'] = self.blocked_time_out_limit
            
            if time_since_useful_move >= self.blocked_time_out_limit:
                result['triggered'] = True
        else:
            self._record_useful_move_time = timenow
        
        self._record_len += 1
        return result

    def _clear_second_red_tracking(self):
        """Clear tracking state for second-red-on-same-light detection."""
        self._tracked_traffic_light_id = None
        self._tracked_traffic_light_location = None
        self._tracked_traffic_light_min_distance = None
        self._stopped_on_red = False
        self._seen_green_after_stop = False

    @staticmethod
    def _get_ego_speed(ego_actor) -> float:
        """Compute ego speed magnitude in m/s."""
        velocity = ego_actor.get_velocity()
        return math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)

    def _refresh_traffic_lights(self):
        self._ensure_world_map()
        self._list_traffic_lights = tl_utils_refresh_traffic_lights(
            world=self._world,
            carla_map=self._map,
            cached_traffic_lights=self._list_traffic_lights,
        )

    def _build_route_lookahead_locations(self, lookahead_m):
        vehicle_route = getattr(CarlaDataProvider, '_ego_vehicle_route', None)
        if not vehicle_route:
            vehicle_route = getattr(CarlaDataProvider, '_vehicle_route', None)
        return tl_utils_build_route_lookahead_locations(
            vehicle_route=vehicle_route,
            lookahead_m=lookahead_m,
        )

    def check_second_red_same_light(self, info: dict) -> dict:
        """
        Detect reward-hacking pattern:
        stop on red -> see green -> meet red again for the same nearby light.
        """
        result = {
            'triggered': False,
            'reason': self.REASON_SECOND_RED_SAME_LIGHT,
            'details': {}
        }
        debug_stage = 'init'
        current_light_id = -1
        current_light_state = 'None'
        current_light_distance = -1.0

        def _update_debug_info():
            if not isinstance(info, dict):
                return
            info['debug_second_red_stage'] = debug_stage
            info['debug_second_red_tracked_id'] = int(self._tracked_traffic_light_id) if self._tracked_traffic_light_id is not None else -1
            info['debug_second_red_current_id'] = int(current_light_id)
            info['debug_second_red_current_state'] = str(current_light_state)
            info['debug_second_red_current_distance'] = float(current_light_distance)
            info['debug_second_red_stopped'] = bool(self._stopped_on_red)
            info['debug_second_red_seen_green'] = bool(self._seen_green_after_stop)

        ego_actor = CarlaDataProvider._ego_actor
        if ego_actor is None or not ego_actor.is_alive:
            self._clear_second_red_tracking()
            debug_stage = 'no_ego'
            _update_debug_info()
            return result

        ego_location = ego_actor.get_location()
        if ego_location is None:
            debug_stage = 'no_ego_location'
            _update_debug_info()
            return result

        ego_speed = self._get_ego_speed(ego_actor)
        self._refresh_traffic_lights()
        lookahead_locations = self._build_route_lookahead_locations(self.second_red_lookahead_m)

        # Clear tracking only after the ego has approached and moved away.
        if (
            self._tracked_traffic_light_location is not None
            and self._tracked_traffic_light_min_distance is not None
        ):
            tracked_distance = compute_2d_distance(ego_location, self._tracked_traffic_light_location)
            if tracked_distance > self._tracked_traffic_light_min_distance + self.second_red_light_clear_distance:
                self._clear_second_red_tracking()
                debug_stage = 'tracked_passed_clear_distance'

        traffic_light = None
        light_location = None
        light_distance = -1.0

        # Prefer the tracked light to avoid stop-line route scan jitter.
        if self._tracked_traffic_light_id is not None:
            tracked_id = int(self._tracked_traffic_light_id)
            for actor, center, _ in self._list_traffic_lights:
                if not getattr(actor, 'is_alive', False):
                    continue
                if int(actor.id) == tracked_id:
                    traffic_light = actor
                    light_location = center
                    light_distance = compute_2d_distance(ego_location, center)
                    break
            if traffic_light is None:
                self._clear_second_red_tracking()
                debug_stage = 'tracked_light_missing'
                _update_debug_info()
                return result
        else:
            first_light = tl_utils_get_first_route_light(
                traffic_lights=self._list_traffic_lights,
                lookahead_locations=lookahead_locations,
                ego_location=ego_location,
            )
            if first_light is None:
                debug_stage = 'no_current_light'
                _update_debug_info()
                return result
            traffic_light, light_location, _, _, light_distance = first_light
            if traffic_light is None or (hasattr(traffic_light, 'is_alive') and not traffic_light.is_alive):
                debug_stage = 'no_current_light'
                _update_debug_info()
                return result

        light_id = int(traffic_light.id)
        light_state = traffic_light.state
        current_light_id = light_id
        current_light_state = light_state.name if hasattr(light_state, 'name') else str(light_state)
        current_light_distance = light_distance

        result['details'].update({
            'traffic_light_id': light_id,
            'distance_to_light': light_distance,
            'distance_reference': 'trigger_volume_center',
            'distance_threshold': self.second_red_light_distance_threshold,
            'clear_distance': self.second_red_light_clear_distance,
            'ego_speed': ego_speed,
            'stop_speed_threshold': self.second_red_stop_speed_threshold,
            'stopped_on_red': self._stopped_on_red,
            'seen_green_after_stop': self._seen_green_after_stop,
            'tracked_min_distance': self._tracked_traffic_light_min_distance if self._tracked_traffic_light_min_distance is not None else -1.0,
        })

        # Start tracking the first nearby red light.
        if self._tracked_traffic_light_id is None:
            if (
                light_state == carla.TrafficLightState.Red
                and light_distance <= self.second_red_light_distance_threshold
            ):
                self._tracked_traffic_light_id = light_id
                self._tracked_traffic_light_location = light_location
                self._tracked_traffic_light_min_distance = float(light_distance)
                self._stopped_on_red = ego_speed <= self.second_red_stop_speed_threshold
                self._seen_green_after_stop = False
                debug_stage = 'track_first_red_stop' if self._stopped_on_red else 'track_first_red_approach'
            else:
                debug_stage = 'waiting_first_red'
            _update_debug_info()
            return result

        self._tracked_traffic_light_location = light_location
        if (
            self._tracked_traffic_light_min_distance is None
            or light_distance < self._tracked_traffic_light_min_distance
        ):
            self._tracked_traffic_light_min_distance = float(light_distance)
        if light_distance > self._tracked_traffic_light_min_distance + self.second_red_light_clear_distance:
            self._clear_second_red_tracking()
            debug_stage = 'tracked_passed_clear_distance'
            _update_debug_info()
            return result

        if light_state == carla.TrafficLightState.Red:
            if (
                not self._stopped_on_red
                and light_distance <= self.second_red_light_distance_threshold
                and ego_speed <= self.second_red_stop_speed_threshold
            ):
                self._stopped_on_red = True
                debug_stage = 'first_red_stop'
            elif (
                self._stopped_on_red
                and self._seen_green_after_stop
                and light_distance <= self.second_red_light_distance_threshold
                and ego_speed <= self.second_red_stop_speed_threshold
            ):
                result['triggered'] = True
                result['details']['event'] = 'second_red_same_light'
                result['details']['debug_stage'] = 'trigger_second_red'
                debug_stage = 'trigger_second_red'
                self._clear_second_red_tracking()
            else:
                debug_stage = 'red_waiting_green_or_stop'
        elif light_state == carla.TrafficLightState.Green and self._stopped_on_red:
            self._seen_green_after_stop = True
            debug_stage = 'seen_green_after_stop'
        else:
            debug_stage = 'tracked_non_red_green'

        _update_debug_info()

        return result
    
    def check_success(self, info: dict) -> dict:
        """Check whether route completion passes the success threshold."""
        result = {
            'triggered': False,
            'reason': self.REASON_SUCCESS,
            'details': {}
        }
        
        all_events = info.get('all_events', [])
        if not all_events:
            return result
        
        try:
            latest_event = all_events[-1]
            if isinstance(latest_event, dict):
                details = latest_event.get('details', {}) or {}
                route_completed = details.get('route_completed', 0.0)
            else:
                event_dict = latest_event.get_dict()
                if event_dict is None:
                    return result
                route_completed = event_dict.get('route_completed', 0.0)
        except (IndexError, AttributeError, TypeError):
            return result
        
        result['details']['route_completed'] = route_completed
        result['details']['threshold'] = self.success_percent_threshold
        
        if route_completed >= self.success_percent_threshold:
            result['triggered'] = True
        
        return result
    
    def _compute_route_deviation(self) -> float:
        """
        Compute route-center deviation in lane-width units.
        
        Returns:
            float: deviation in lane-width units.
        """
        if len(CarlaDataProvider._vehicle_route) < 1:
            return 0.0
            
        ego_actor = CarlaDataProvider._ego_actor
        if ego_actor is None or not ego_actor.is_alive:
            return 0.0
            
        self._ensure_world_map()
        
        ev_transform = ego_actor.get_transform()
        route_location = CarlaDataProvider._vehicle_route[0][0].location
        route_wp = self._map.get_waypoint(route_location)
        
        ref_vec = carla.Vector3D(
            ev_transform.location.x - route_location.x,
            ev_transform.location.y - route_location.y
        )
        right_vec = route_wp.transform.get_right_vector()
        lateral_offset = abs(ref_vec.x * right_vec.x + ref_vec.y * right_vec.y)
        
        deviation = lateral_offset / route_wp.lane_width
        return deviation
    
    def get_deviation(self) -> float:
        """Return current route-center deviation in lane-width units."""
        return self._compute_route_deviation()


TERMINATION_CASE_MAP = {
    TerminationHandler.REASON_NOT_IN_CARLANE: 18,        # 'not_in_carlane'
    TerminationHandler.REASON_USELESS_MOVE_TIMEOUT: 17,  # 'useless_move_timeout'
    TerminationHandler.REASON_SECOND_RED_SAME_LIGHT: 20, # 'second_red_same_light'
    TerminationHandler.REASON_SUCCESS: 19,               # 'success'
}
