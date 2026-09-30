"""Reward handler - PPO-tuned variant."""
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.traffic_events import TrafficEventType
from srunner.scenariomanager.timer import GameTime
from collections import deque
import functools
import carla
import math
import py_trees
import copy
import logging

__layer__ = (2, "Environment")

import numpy as np

logger = logging.getLogger("Interface Wrapper")
from b2d_rlinfra.environment.handlers.reward_traffic_utils import (
    TL_CLEAR_DISTANCE_M,
    TL_LOOKAHEAD_M,
    TL_RED_LIGHT_DECEL_DISTANCE_M,
    TL_SECOND_RED_DISTANCE_M,
    TL_STOP_SPEED_THRESHOLD_MPS,
    build_route_lookahead_locations as tl_utils_build_route_lookahead_locations,
    find_traffic_light_entry_by_id,
    get_first_route_light as tl_utils_get_first_route_light,
    is_light_affecting_route as tl_utils_is_light_affecting_route,
    refresh_traffic_lights as tl_utils_refresh_traffic_lights,
)
from b2d_rlinfra.environment.info_keys import (
    WRAPPER_OBS_EMERGENCY_IN_VISION,
    WRAPPER_REWARD_PARKING_EXIT_DEVIATION_EXEMPT,
    WRAPPER_TERM_TRIGGERED,
)

def compute_2d_distance(loc1, loc2):
    return math.sqrt((loc1.x-loc2.x)**2+(loc1.y-loc2.y)**2)

EVENT_NAME_TO_KEY = {
    'SUCCESS': 'SUCCESS',
    TrafficEventType.ROUTE_COMPLETION.name: TrafficEventType.ROUTE_COMPLETION,
    TrafficEventType.COLLISION_PEDESTRIAN.name: TrafficEventType.COLLISION_PEDESTRIAN,
    TrafficEventType.COLLISION_VEHICLE.name: TrafficEventType.COLLISION_VEHICLE,
    TrafficEventType.COLLISION_STATIC.name: TrafficEventType.COLLISION_STATIC,
    TrafficEventType.TRAFFIC_LIGHT_INFRACTION.name: TrafficEventType.TRAFFIC_LIGHT_INFRACTION,
    TrafficEventType.STOP_INFRACTION.name: TrafficEventType.STOP_INFRACTION,
    TrafficEventType.SCENARIO_TIMEOUT.name: TrafficEventType.SCENARIO_TIMEOUT,
    TrafficEventType.YIELD_TO_EMERGENCY_VEHICLE.name: TrafficEventType.YIELD_TO_EMERGENCY_VEHICLE,
    TrafficEventType.ROUTE_DEVIATION.name: TrafficEventType.ROUTE_DEVIATION,
    TrafficEventType.VEHICLE_BLOCKED.name: TrafficEventType.VEHICLE_BLOCKED,
    'USELESS_MOVE_TIMEOUT': 'USELESS_MOVE_TIMEOUT',
    'NOT_IN_CARLANE': 'NOT_IN_CARLANE',
    'SECOND_RED_SAME_LIGHT': 'SECOND_RED_SAME_LIGHT',
}


REWARD_CONFIG = {
    'event_rewards': {
        'ROUTE_COMPLETION': 200.0,
        'SUCCESS': 200.0,
        'COLLISION_PEDESTRIAN': -70.0,
        'COLLISION_VEHICLE': -70.0,
        'COLLISION_STATIC': -70.0,
        'TRAFFIC_LIGHT_INFRACTION': -70.0,
        'STOP_INFRACTION': -70.0,
        'SCENARIO_TIMEOUT': -70.0,
        'YIELD_TO_EMERGENCY_VEHICLE': -70.0,
        'ROUTE_DEVIATION': -70.0,
        'VEHICLE_BLOCKED': -70.0,
        'USELESS_MOVE_TIMEOUT': -70.0,
        'NOT_IN_CARLANE': -70.0,
        'SECOND_RED_SAME_LIGHT': -70.0,
    },
    'step_rewards': {
        'progress': {
            'reward_per_meter': 2.0,
        },
        'speed': {
            'weight': 2.0,
            'front_actor_distance_threshold': 15.0,
            'front_actor_safety_distance': 10.0,
            'obstacle_blocked_distance': 10.0,
            'overspeed_penalty_scale': 3.0,
            'underspeed_penalty_scale': 1.5,
            'must_wait_idle_reward': 0.2,
        },
        'deviation': {
            'micro_deviation_penalty': 1.0,
        },
        'steering': {
            'consistency_penalty': 0.2,
        },
        'reverse': {
            'penalty': 1.0,
        },
        'borrow': {
            'deviation_progress_reward': 2.0,
            'turn_alignment_reward': 12.0,
        },
        'merge': {
            'bonus': 20.0,
        },
        'stop_sign': {
            'detect_distance': 20.0,
            'completion_distance': 4.0,
            'decel_distance': 12.0,
            'early_wait_penalty': 1.0,
            'clear_distance': TL_CLEAR_DISTANCE_M,
            'completed_bonus': 10.0,
            'passed_without_stop_penalty': -20.0,
            'zero_speed_reward_max_steps': 3,
        },
        'traffic_light': {
            'red_light_reward_clear_distance': TL_CLEAR_DISTANCE_M,
        },
    },
}


def _build_reward_dict(event_rewards):
    return {
        EVENT_NAME_TO_KEY.get(name, name): float(value)
        for name, value in event_rewards.items()
    }


REWARD_DICT = _build_reward_dict(REWARD_CONFIG['event_rewards'])

NAME_DICT = {
    TrafficEventType.COLLISION_STATIC : 1,
    TrafficEventType.COLLISION_VEHICLE : 2,
    TrafficEventType.COLLISION_PEDESTRIAN : 3,
    TrafficEventType.ROUTE_DEVIATION : 4,
    TrafficEventType.ROUTE_COMPLETION : 5,
    TrafficEventType.TRAFFIC_LIGHT_INFRACTION : 7,
    TrafficEventType.STOP_INFRACTION : 10,
    TrafficEventType.VEHICLE_BLOCKED: 13,
    TrafficEventType.YIELD_TO_EMERGENCY_VEHICLE: 15,
    TrafficEventType.SCENARIO_TIMEOUT: 16,
    'USELESS_MOVE_TIMEOUT': 17,
    'NOT_IN_CARLANE': 18,
    'SUCCESS': 19,
    'SECOND_RED_SAME_LIGHT': 20
}

def _build_reward_name_dict(step_rewards):
    progress_cfg = step_rewards.get('progress', {})
    speed_cfg = step_rewards.get('speed', {})
    deviation_cfg = step_rewards.get('deviation', {})
    steering_cfg = step_rewards.get('steering', {})
    merge_cfg = step_rewards.get('merge', {})
    return {
        'reward_per_meter': float(progress_cfg.get('reward_per_meter', 2.0)),
        'micro_deviation_penalty': float(deviation_cfg.get('micro_deviation_penalty', 1.0)),
        'steer_consistency_penalty': float(steering_cfg.get('consistency_penalty', 0.2)),
        'reward_speed': float(speed_cfg.get('weight', 2.0)),
        'merge': float(merge_cfg.get('bonus', 20.0)),
    }


REWARD_NAME_DICT = _build_reward_name_dict(REWARD_CONFIG['step_rewards'])


class RewardHandler():
    """
    Reward generator for CARLA autonomous driving environment.
    
    Key Instance Variables:
    -----------------------
    direction : int
        Lane change direction indicator for obstacle avoidance or emergency vehicle yielding.
        Values:
            0  : No lane change needed, normal driving
            -1 : Change to LEFT lane (multiply with right_vector to get left direction)
            1  : Change to RIGHT lane (use right_vector directly)
        
        Set by: detect_blocked_by_obstacles(), detect_followed_by_emergency()
        Used by: get_onlyblocker_enabled_all_lane_deviation(), speed reward calculation
    
    have_merged : bool
        Whether the ego vehicle has successfully merged into the target lane
        after initiating a lane change maneuver.
    
    twoway_blocked : bool
        Whether the ego is blocked on a two-way road (needs to use opposite lane).
    
    passable : bool
        Whether there is a safe gap to pass through when twoway_blocked is True.
    
    _blocked_by_obstacles_status : bool
        Current frame status: ego is blocked by static obstacles on the route.
    
    _followed_by_emergency_status : bool
        Current frame status: ego is being followed by an emergency vehicle.
    
    _blocked_by_emergency_status : bool
        Current frame status: ego is blocked by an emergency vehicle at junction.
    """
    
    PROXIMITY_THRESHOLD = 2.0
    SPEED_THRESHOLD = 0.1
    WAYPOINT_STEP = 0.5
    DIST_THRESHOLD = 3.25
    OFFSET = 1.0
    # used in 'is_emergency_close_enough' func
    EMERGENCY_VEHICLES_CLOSE_THRESHOLD = 20
    JUNCTION_FIXED_DESIRED_SPEED_BY_SCENARIO = {
        'InterurbanAdvancedActorFlow': 12.0,
        'EnterActorFlow': 16.0,
        'CrossingBicycleFlow': 12.0,
    }
    
    # Map event-name strings to REWARD_DICT/NAME_DICT keys.
    EVENT_TYPE_TO_KEY = EVENT_NAME_TO_KEY
    
    # Success events are ignored while an emergency vehicle is following.
    SUCCESS_EVENTS = {'SUCCESS', TrafficEventType.ROUTE_COMPLETION.name}
    
    def __init__(self, config, scenario_name):
        self.config = config
        self.scenario_name = scenario_name or "leaderboardv2"

        self.record_len = 0
        self.record_useful_move_time = 0.
        
        self.reward_config = REWARD_CONFIG
        step_rewards_cfg = self.reward_config.get('step_rewards', {})
        speed_cfg = step_rewards_cfg.get('speed', {})
        stop_sign_cfg = step_rewards_cfg.get('stop_sign', {})
        traffic_light_cfg = step_rewards_cfg.get('traffic_light', {})
        borrow_cfg = step_rewards_cfg.get('borrow', {})
        reverse_cfg = step_rewards_cfg.get('reverse', {})
        self.reward_dict = dict(REWARD_DICT)
        self.borrow_deviation_progress_reward = float(
            borrow_cfg.get('deviation_progress_reward', 2.0)
        )
        self.borrow_turn_alignment_reward = float(
            borrow_cfg.get('turn_alignment_reward', 12.0)
        )
        self.reverse_penalty_magnitude = float(reverse_cfg.get('penalty', 1.0))

        self.window_size = 5
        self._distance_threshold = float(speed_cfg.get('front_actor_distance_threshold', 15.0))
        self.safety_distance = float(speed_cfg.get('front_actor_safety_distance', 10.0))
        self.obstacle_blocked_distance = float(
            speed_cfg.get('obstacle_blocked_distance', 10.0)
        )
        if self._distance_threshold <= self.safety_distance:
            self._distance_threshold = self.safety_distance + 1.0
        self.pedestrian_distance_threshold = 10.0
        self.overspeed_penalty_scale = float(speed_cfg.get('overspeed_penalty_scale', 3.0))
        self.underspeed_penalty_scale = float(speed_cfg.get('underspeed_penalty_scale', 1.5))
        self.stop_target_reward = 0.15
        self.stop_target_penalty_scale = self.overspeed_penalty_scale
        if self.stop_target_penalty_scale <= 0:
            self.stop_target_penalty_scale = 3.0
        self.stop_sign_completed_bonus = float(stop_sign_cfg.get('completed_bonus', 10.0))
        self.stop_sign_passed_without_stop_penalty = float(
            stop_sign_cfg.get('passed_without_stop_penalty', -20.0)
        )
        self.stop_sign_zero_speed_reward_max_steps = int(
            stop_sign_cfg.get('zero_speed_reward_max_steps', 3)
        )
        self._stop_sign_completed_bonus_steps = 0
        self._stop_sign_passed_without_stop_penalty_pending = 0.0
        self.stop_sign_record = None
        self.controloss_zone_record = None
        self.stop_sign_last = 0
        self.controloss_zone_last = 0

        self._list_stop_signs = []
        self._list_traffic_lights = []
        self._list_controloss_zones = []
        self.distance_light = TL_RED_LIGHT_DECEL_DISTANCE_M
        self.traffic_light_lookahead_m = TL_LOOKAHEAD_M
        self.second_red_light_distance_threshold = TL_SECOND_RED_DISTANCE_M
        self.red_light_reward_clear_distance = float(
            traffic_light_cfg.get('red_light_reward_clear_distance', TL_CLEAR_DISTANCE_M)
        )
        self.stop_speed_threshold = TL_STOP_SPEED_THRESHOLD_MPS
        self.must_wait_stop_speed_threshold = 0.05
        self.must_wait_idle_reward = float(
            speed_cfg.get('must_wait_idle_reward', 0.2)
        )
        self.stop_sign_detect_distance = float(stop_sign_cfg.get('detect_distance', 20.0))
        self.stop_sign_completion_distance = float(
            stop_sign_cfg.get('completion_distance', 4.0)
        )
        self.stop_sign_decel_distance = float(stop_sign_cfg.get('decel_distance', 12.0))
        self.stop_sign_early_wait_penalty = float(
            stop_sign_cfg.get('early_wait_penalty', 1.0)
        )
        self.stop_sign_reward_clear_distance = float(
            stop_sign_cfg.get('clear_distance', TL_CLEAR_DISTANCE_M)
        )
        self._latched_red_light_id = None
        self._latched_red_light_center = None
        self._latched_red_light_min_distance = None
        self._latched_stop_sign_id = None
        self._latched_stop_sign_center = None
        self._latched_stop_sign_min_distance = None
        self._latched_stop_sign_completed = False
        self._stop_sign_completed_bonus_steps = 0
        self._stop_sign_passed_without_stop_penalty_pending = 0.0
        self._debug_red_lock_source = 'none'
        self._debug_red_release_reason = '-'
        self._debug_red_recent_distance = -1.0
        self._debug_red_has_relevant_green = False
        self._debug_stop_lock_source = 'none'
        self._debug_stop_release_reason = '-'
        self._debug_stop_recent_distance = -1.0
        self._debug_stop_completion_distance = -1.0
                 
        self.history_yaw = deque(maxlen=11)
        self.wps = deque(maxlen=3)
        
        self._have_blocked_by_obstacles_status = False
        self._blocked_by_obstacles_status = False
        self._blocked_by_emergency_status = False
        self._followed_by_emergency_status = False
        self.have_merged = False
        self.direction = 0
        self.twoway_blocked_step = 0
        self.twoway_blocked = False
        self.passable = False
        self.have_turn2green = False
        self.red = False
        self.cumulative_length = 0.

        self.last_steer = None
        self.last_yaw = None
        self.last_merge_status = False
        self.last_angular_velocity = None
        self.last_borrow_micro_deviation = None
        self._step_active_actors = None
    
    _SURR_CACHE_RADIUS = 200.0
    _SURR_REFRESH_INTERVAL = 5

    @staticmethod
    def _is_effectively_active(actor) -> bool:
        """Prefer actor.is_active; fallback to is_alive for compatibility."""
        try:
            return bool(actor.is_active)
        except Exception:
            return bool(getattr(actor, 'is_alive', False))

    def _prime_step_actor_cache(self) -> None:
        """Fetch actor list once per step and cache active candidates."""
        self._step_active_actors = []
        if self.world is None:
            return
        all_actors = CarlaDataProvider.get_all_actors()
        self._step_active_actors = [
            a for a in all_actors
            if self._is_effectively_active(a)
        ]

    def get_surrounding_actors(self):
        self._surr_step_counter = getattr(self, '_surr_step_counter', 0) + 1
        need_refresh = (self._surr_step_counter % self._SURR_REFRESH_INTERVAL == 1
                        or not self._list_stop_signs and not self._list_traffic_lights)

        ego_loc = self.ego_actor.get_transform().location if self.ego_actor else None

        if need_refresh:
            active_actors = getattr(self, '_step_active_actors', None)
            if active_actors is None:
                _all_actors = CarlaDataProvider.get_all_actors()
                active_actors = [a for a in _all_actors if self._is_effectively_active(a)]

            known_stop_ids = {int(a.id) for a in self._list_stop_signs
                              if getattr(a, 'is_alive', False)}

            for actor in active_actors:
                actor_id = int(actor.id)
                if 'traffic.stop' in actor.type_id and actor_id not in known_stop_ids:
                    self._list_stop_signs.append(actor)
                    known_stop_ids.add(actor_id)

            self._list_traffic_lights = tl_utils_refresh_traffic_lights(
                world=self.world,
                carla_map=self.map,
                cached_traffic_lights=self._list_traffic_lights,
                ego_loc=ego_loc,
                max_radius=self._SURR_CACHE_RADIUS,
                actors_candidates=active_actors,
            )

            # Drop cached actors that are far from ego.
            if ego_loc is not None:
                r2 = self._SURR_CACHE_RADIUS ** 2
                self._list_stop_signs = [
                    a for a in self._list_stop_signs
                    if getattr(a, 'is_alive', False)
                    and (a.get_location().x - ego_loc.x) ** 2
                       + (a.get_location().y - ego_loc.y) ** 2 < r2
                ]
    
    def reset(self):
        self.stop_sign_record = None
        self.controloss_zone_record = None
        self.stop_sign_last = 0
        self.controloss_zone_last = 0
        self._have_blocked_by_obstacles_status = False
        self._blocked_by_obstacles_status = False
        self._blocked_by_emergency_status = False
        self._followed_by_emergency_status = False
        self.have_merged = False
        self.direction = 0
        self.twoway_blocked_step = 0
        self.twoway_blocked = False
        self.passable = False
        self.have_turn2green = False
        self.red = False
        self._latched_red_light_id = None
        self._latched_red_light_center = None
        self._latched_red_light_min_distance = None
        self._latched_stop_sign_id = None
        self._latched_stop_sign_center = None
        self._latched_stop_sign_min_distance = None
        self._latched_stop_sign_completed = False
        self._stop_sign_completed_bonus_steps = 0
        self._stop_sign_passed_without_stop_penalty_pending = 0.0
        self._debug_red_lock_source = 'none'
        self._debug_red_release_reason = '-'
        self._debug_red_recent_distance = -1.0
        self._debug_red_has_relevant_green = False
        self._debug_stop_lock_source = 'none'
        self._debug_stop_release_reason = '-'
        self._debug_stop_recent_distance = -1.0
        self._debug_stop_completion_distance = -1.0
        self.cumulative_length = 0.

        self.last_steer = None
        self.last_yaw = None
        self.last_merge_status = False
        self.last_angular_velocity = None
        self.last_borrow_micro_deviation = None
        self._list_stop_signs = []
        self._list_traffic_lights = []
        self._list_controloss_zones = []
        self._surr_step_counter = 0
        self._step_active_actors = None

    def detect_blocked_status(self, info):
        if len(CarlaDataProvider._ego_vehicle_route) <= 1:
            self._blocked_by_obstacles_status = False
            self._blocked_by_emergency_status = False
            self._followed_by_emergency_status = False
        else:
            self._blocked_by_emergency_status = self.blocked_by_emergency()
            self._blocked_by_obstacles_status = self.detect_blocked_by_obstacles()
            self._followed_by_emergency_status = self.detect_followed_by_emergency(info)
            if self._blocked_by_obstacles_status:
                self._have_blocked_by_obstacles_status = True
        if (not self._blocked_by_obstacles_status) and (not self._blocked_by_emergency_status) and\
                (not self._followed_by_emergency_status):
                    self.have_merged = False
                    self.twoway_blocked_step = 0
                    self.twoway_blocked = False
                    self.passable = False
                    self.last_borrow_micro_deviation = None

    def generate_reward(self, observation, terminated, info, action, name= "leaderboardv2", crash_message = ""):
        reward = 0.0
        if crash_message == "Simulation crashed":
            logger.warning('Simulation Crashed, Reward Is Set As 0.0')
            return np.array(reward, dtype=np.float32), info, False
        
        self.info = info
        self.action = action
        self.observation = observation
        
        self.ego_actor = CarlaDataProvider._ego_actor
        self.world = CarlaDataProvider.get_world()
        self.map = CarlaDataProvider.get_map()
        self._prime_step_actor_cache()
        self.get_surrounding_actors()
        
        self.ev_transform = self.ego_actor.get_transform()    
        self.ego_wp = self.map.get_waypoint(self.ev_transform.location)
        self.history_yaw.append(self.ev_transform.rotation.yaw)
        self.route_transform = self.get_route_transform()
        self.current_route_waypoint = self.map.get_waypoint(self.route_transform.location)
        
        self.detect_blocked_status(info)
        if self._blocked_by_obstacles_status or self._followed_by_emergency_status or self._blocked_by_emergency_status:
            info['blocked'] = True
        
        if (self.current_route_waypoint and self.current_route_waypoint.is_junction):
            self.max_speed = self.ego_actor.get_speed_limit() / 3.6
        elif self._blocked_by_obstacles_status:
            self.max_speed = self.ego_actor.get_speed_limit() / 3.6
        else:
            self.max_speed = self.ego_actor.get_speed_limit() / 3.6
        if self.check_scenario_name('EnterActorFlow'):
            self.max_speed = 16.0
        
        # Compute desired speed early so deviation penalty can be conditionally skipped near red-light stop.
        desired_speed = self.get_desired_speed(info)
        base_desired_speed = desired_speed
        info['debug_red_latched_id'] = int(self._latched_red_light_id) if self._latched_red_light_id is not None else -1
        info['debug_red_recent_id'] = -1
        info['debug_red_lock_source'] = self._debug_red_lock_source
        info['debug_red_release_reason'] = self._debug_red_release_reason
        info['debug_red_recent_distance'] = float(self._debug_red_recent_distance)
        info['debug_red_min_distance'] = float(self._latched_red_light_min_distance) if self._latched_red_light_min_distance is not None else -1.0
        info['debug_red_has_relevant_green'] = bool(self._debug_red_has_relevant_green)
        info['debug_red_is_locked'] = bool(self._latched_red_light_id is not None)
        info['debug_stop_latched_id'] = int(self._latched_stop_sign_id) if self._latched_stop_sign_id is not None else -1
        info['debug_stop_lock_source'] = self._debug_stop_lock_source
        info['debug_stop_release_reason'] = self._debug_stop_release_reason
        info['debug_stop_recent_distance'] = float(self._debug_stop_recent_distance)
        info['debug_stop_completion_distance'] = float(self._debug_stop_completion_distance)
        info['debug_stop_min_distance'] = float(self._latched_stop_sign_min_distance) if self._latched_stop_sign_min_distance is not None else -1.0
        info['debug_stop_is_locked'] = bool(self._latched_stop_sign_id is not None)
        info['debug_stop_completed'] = bool(self._latched_stop_sign_completed)
        skip_deviation_penalty = (
            (self.red or (self._latched_stop_sign_id is not None and not self._latched_stop_sign_completed))
            and desired_speed < 1.0
        )
        info['skip_deviation_penalty_red_light'] = bool(skip_deviation_penalty)
        parking_exit_deviation_exempt = self.is_parking_exit_deviation_exempt()
        info[WRAPPER_REWARD_PARKING_EXIT_DEVIATION_EXEMPT] = bool(parking_exit_deviation_exempt)
        yield_emergency_route_deviation_exempt = self.check_scenario_name('YieldToEmergencyVehicle')
        info['yield_emergency_route_deviation_exempt'] = bool(yield_emergency_route_deviation_exempt)

        # Update borrow-lane state first, then decide which lane centerline should
        # be used as the deviation target.
        self._refresh_borrow_lane_flags()
        borrow_blocking_distance = self.get_borrow_lane_blocking_distance()
        passable = bool(self.twoway_blocked and borrow_blocking_distance > 50.0)
        self.passable = passable
        borrow_target_deviation = self.get_onlyblocker_enabled_all_lane_deviation(
            self.route_transform,
            passable=passable,
        )
        twoway_borrow_merged = self.twoway_blocked and self.have_merged
        must_wait = (
            self.twoway_blocked
            and not self.have_merged
            and not passable
        )
        can_borrow = (
            self._blocked_by_obstacles_status
            and (self.have_merged or passable or not self.twoway_blocked)
        )
        use_borrow_deviation_target = can_borrow or self._followed_by_emergency_status
        if use_borrow_deviation_target:
            micro_deviation = borrow_target_deviation
        else:
            micro_deviation = self.get_route_lane_deviation(self.route_transform)
        info["borrow_must_wait"] = bool(must_wait)
        info["borrow_can_go"] = bool(can_borrow)
        info["borrow_direction"] = int(self.direction)
        info["borrow_twoway"] = bool(self.twoway_blocked)
        info["borrow_passable"] = bool(passable)
        info["borrow_twoway_merged"] = bool(twoway_borrow_merged)
        info["borrow_blocking_distance"] = (
            float(borrow_blocking_distance) if borrow_blocking_distance < float('inf') else -1.0
        )
        info["micro_deviation_raw"] = float(micro_deviation)
        info["borrow_deviation_target_active"] = bool(use_borrow_deviation_target)
        borrow_deviation_progress = 0.0
        if can_borrow:
            if self.last_borrow_micro_deviation is not None:
                borrow_deviation_progress = self.last_borrow_micro_deviation - micro_deviation
            borrow_deviation_progress = float(np.clip(borrow_deviation_progress, -0.2, 0.2))
            self.last_borrow_micro_deviation = micro_deviation
            reward += self.borrow_deviation_progress_reward * borrow_deviation_progress
        else:
            self.last_borrow_micro_deviation = None
        info["borrow_deviation_progress"] = float(borrow_deviation_progress)
        skip_borrow_wait_deviation_penalty = bool(must_wait)
        if skip_deviation_penalty or skip_borrow_wait_deviation_penalty:
            info['micro_deviation_penalty'] = 0.0
            info['skip_deviation_penalty_borrow_wait'] = bool(skip_borrow_wait_deviation_penalty)
        else:
            info['skip_deviation_penalty_borrow_wait'] = False
            if (
                micro_deviation > 1.3
                and (not parking_exit_deviation_exempt)
                and (not yield_emergency_route_deviation_exempt)
            ):
                reward += self.reward_dict[TrafficEventType.ROUTE_DEVIATION]
                info['micro_deviation_penalty'] = self.reward_dict[TrafficEventType.ROUTE_DEVIATION]
            else:
                rew_micro_deviation = -micro_deviation
                if can_borrow:
                    rew_micro_deviation = -micro_deviation
                elif self._blocked_by_emergency_status:
                    rew_micro_deviation = 0
                elif (self._followed_by_emergency_status and (not self.have_merged)):
                    rew_micro_deviation = 2*rew_micro_deviation
                else:
                    rew_micro_deviation = 2*rew_micro_deviation
                rew_micro_deviation = rew_micro_deviation * REWARD_NAME_DICT['micro_deviation_penalty']
                # Ignore tiny deviations inside the dead zone.
                if rew_micro_deviation > -0.15:
                    rew_micro_deviation = 0
                reward += rew_micro_deviation
                info['micro_deviation_penalty'] = rew_micro_deviation
        
        # RoutePlanWrapper must update this before RewardWrapper consumes it.
        rew_travel = CarlaDataProvider._distance_traveled_on_tick
        self.cumulative_length += rew_travel
        reward_per_meter = REWARD_NAME_DICT['reward_per_meter'] * rew_travel
        if must_wait:
            reward_per_meter = 0
        elif can_borrow:
            if not self.have_merged:
                reward_per_meter = 0
            else:
                reward_per_meter = 0.5 * reward_per_meter
                if borrow_deviation_progress <= 0:
                    reward_per_meter = 0
        elif ((self._blocked_by_obstacles_status or self._followed_by_emergency_status) and (not self.have_merged)):
            reward_per_meter = 0
        elif self.current_route_waypoint.is_junction:
            # to prevent ego fearing to enter the junction
            reward_per_meter = 1 * reward_per_meter
        reward += reward_per_meter
        info['reward_per_meter'] = reward_per_meter
        info['cumulative_length'] = self.cumulative_length
        
        ego_velocity = self.ego_actor.get_velocity()
        ego_speed = np.linalg.norm(np.array([ego_velocity.x, ego_velocity.y, ego_velocity.z]))
        info['speed'] = ego_speed
        info['turn'] = 0.0
        info["borrow_start_penalty"] = 0.0
        early_stop_wait = (
            self._latched_stop_sign_id is not None
            and not self._latched_stop_sign_completed
            and self._debug_stop_recent_distance > self.stop_sign_completion_distance
            and ego_speed < self.stop_speed_threshold * 2
            and not self.red
            and not self._blocked_by_obstacles_status
            and not self._blocked_by_emergency_status
            and not self._followed_by_emergency_status
        )
        
        if twoway_borrow_merged:
            desired_speed = 12.0
            reward_speed = self.get_speed_reward(ego_speed, desired_speed)
        elif self._blocked_by_emergency_status:
            desired_speed = 0
            reward_speed = self.get_speed_reward(ego_speed, desired_speed)
        elif must_wait:
            desired_speed = 0.0
            reward_speed = self.get_speed_reward(ego_speed, desired_speed)
            if ego_speed > self.must_wait_stop_speed_threshold:
                reward_speed = min(reward_speed, -0.5)
        elif self._blocked_by_obstacles_status and not self.have_merged and not can_borrow:
            desired_speed = 0.0
            reward_speed = self.get_speed_reward(ego_speed, desired_speed)
        elif can_borrow:
            if self.twoway_blocked:
                desired_speed = 12.0 if self.have_merged else 8.33
            else:
                desired_speed = base_desired_speed
            borrow_start_penalty = 0.0
            if (not self.have_merged) and ego_speed < 0.5 and borrow_deviation_progress <= 0:
                borrow_start_penalty = 0.3
            reward_speed = self.get_speed_reward(ego_speed, desired_speed) - borrow_start_penalty
            info["borrow_start_penalty"] = -borrow_start_penalty * REWARD_NAME_DICT['reward_speed']
        elif self._followed_by_emergency_status and self.current_route_waypoint.is_junction:
            reward_speed = self.get_speed_reward(ego_speed, desired_speed)
        elif self._followed_by_emergency_status and (not self.have_merged):

            ego_speed = ego_speed * np.clip(self.compute_dot(self.direction * self.current_route_waypoint.transform.get_right_vector(), 
                                                            self.ego_actor.get_transform().get_forward_vector()),
                                            -1, 1) 
            desired_speed = 0.5 * self.max_speed

            reward_speed = (ego_speed / desired_speed)

            info['turn'] = reward_speed
        elif self.have_turn2green and ego_speed<=0.1:
            info['pass_green'] = -0.1
            reward_speed = -0.1
        else:
            reward_speed = self.get_speed_reward(ego_speed, desired_speed)

        if can_borrow and not self.have_merged:
            turn_align = float(np.clip(
                self.compute_dot(
                    self.direction * self.current_route_waypoint.transform.get_right_vector(),
                    self.ego_actor.get_transform().get_forward_vector()
                ),
                0.0,
                1.0,
            ))
            turn_speed_gate = float(np.clip(ego_speed / 3.0, 0.0, 1.0))
            turn_progress_gate = float(np.clip(borrow_deviation_progress / 0.03, -1.0, 1.0))
            turn_reward = (
                self.borrow_turn_alignment_reward
                * turn_align
                * turn_speed_gate
                * turn_progress_gate
            )
            reward += turn_reward
            info["turn"] = float(turn_reward)
            info["borrow_turn_align"] = turn_align
            info["borrow_turn_speed_gate"] = turn_speed_gate
            info["borrow_turn_progress_gate"] = turn_progress_gate

        if early_stop_wait:
            reward_speed = min(reward_speed, -self.stop_sign_early_wait_penalty)
            info['stop_wait_penalty'] = -self.stop_sign_early_wait_penalty * REWARD_NAME_DICT['reward_speed']
        else:
            info['stop_wait_penalty'] = 0.0

        pending_stop_zero_speed_reward = (
            self._latched_stop_sign_id is not None
            and not self._latched_stop_sign_completed
            and desired_speed <= self.stop_speed_threshold
            and ego_speed <= self.stop_speed_threshold
            and reward_speed > 0
        )
        if pending_stop_zero_speed_reward:
            reward_speed = 0.0

        info['desired_speed'] = desired_speed
        info['reward_speed_raw'] = reward_speed
        reward_speed_scaled = reward_speed * REWARD_NAME_DICT['reward_speed']
        info['reward_speed'] = reward_speed_scaled
        reward += reward_speed_scaled

        must_wait_idle_reward = 0.0
        if must_wait and ego_speed <= self.must_wait_stop_speed_threshold:
            must_wait_idle_reward = self.must_wait_idle_reward
            reward += must_wait_idle_reward
        info["borrow_must_wait_idle_reward"] = must_wait_idle_reward

        stop_completed_zero_speed_active = (
            self._latched_stop_sign_id is not None
            and self._latched_stop_sign_completed
            and ego_speed < self.SPEED_THRESHOLD
        )
        if (
            stop_completed_zero_speed_active
            and self._stop_sign_completed_bonus_steps < self.stop_sign_zero_speed_reward_max_steps
        ):
            self._stop_sign_completed_bonus_steps += 1
            stop_completed_reward = float(self.stop_sign_completed_bonus)
        else:
            stop_completed_reward = 0.0

        stop_passed_without_stop_penalty = float(self._stop_sign_passed_without_stop_penalty_pending)
        reward += stop_completed_reward + stop_passed_without_stop_penalty
        info['stop_completed_reward'] = stop_completed_reward
        info['stop_passed_without_stop_penalty'] = stop_passed_without_stop_penalty
        info['stop_pending_zero_speed_reward_suppressed'] = bool(pending_stop_zero_speed_reward)
        info['stop_completed_bonus_steps'] = int(self._stop_sign_completed_bonus_steps)
        
        rew_steer_consistency = self.get_steer_consistency(action.steer)
        reward = reward - rew_steer_consistency * (0 if REWARD_NAME_DICT['steer_consistency_penalty'] == -1 else REWARD_NAME_DICT['steer_consistency_penalty'])
        info['steer_consistency_penaly'] = -rew_steer_consistency * (0 if REWARD_NAME_DICT['steer_consistency_penalty'] == -1 else REWARD_NAME_DICT['steer_consistency_penalty'])
        
        
        # WRAPPER_TERM_TRIGGERED is filled by EventTerminationWrapper on every
        # step (and may be re-asserted by RewardWrapper's fallback). Reading
        # strictly so wrapper misordering surfaces as KeyError.
        is_terminated = terminated or info[WRAPPER_TERM_TRIGGERED]
        terminate_events = info.get('terminate_events', [])
        
        terminate_reward = 0.0
        if is_terminated and terminate_events:
            for event in terminate_events:
                event_type = event['event_type']
                
                dict_key = self.EVENT_TYPE_TO_KEY.get(event_type)
                if dict_key is None:
                    continue
                
                if event_type in self.SUCCESS_EVENTS and self._followed_by_emergency_status:
                    continue
                
                if dict_key in self.reward_dict and dict_key in NAME_DICT:
                    event_reward = self.reward_dict[dict_key]
                    reward += event_reward
                    terminate_reward += event_reward
                    info['case'] = NAME_DICT[dict_key]
        # Route completion may terminate without a matching terminate event.
        if is_terminated and terminate_reward == 0.0:
            route_completed = None
            for event in info.get('all_events', []):
                if event.get('type') != 'ROUTE_COMPLETION':
                    continue
                details = event.get('details', {}) or {}
                value = details.get('route_completed')
                if value is None:
                    continue
                route_completed = value if route_completed is None else max(route_completed, value)
            if (
                route_completed is not None
                and route_completed >= 99.9
                and not self._followed_by_emergency_status
            ):
                event_reward = self.reward_dict.get(TrafficEventType.ROUTE_COMPLETION, 0.0)
                reward += event_reward
                terminate_reward += event_reward
                if TrafficEventType.ROUTE_COMPLETION in NAME_DICT:
                    info['case'] = NAME_DICT[TrafficEventType.ROUTE_COMPLETION]
        
        if terminate_reward != 0:
            info['terminate_reward'] = terminate_reward
        merge_bonus = 0.0
        if self.have_merged and not self.last_merge_status:
            merge_bonus = float(REWARD_NAME_DICT['merge'])
            reward += merge_bonus
        info['merge_bonus'] = merge_bonus

        self.last_merge_status = self.have_merged
        if self.have_merged:
            info['merged'] = True
        if self.red:
            info['red'] = True
        
        if action.reverse:
            reverse_penalty = -self.reverse_penalty_magnitude
            info['reverse_penalty'] = float(reverse_penalty)
            reward = reward + reverse_penalty
        else:
            info['reverse_penalty'] = 0.0
        
        self.append_wp()
        # WRAPPER_OBS_EMERGENCY_IN_VISION is filled by ObservationWrapper.step
        # on every step. Strict read so wrapper misordering fails loudly.
        if info[WRAPPER_OBS_EMERGENCY_IN_VISION] and self.is_emergency_close_enough():
            info['emergency'] = True
        return np.array(reward, dtype=np.float32), info, (self._blocked_by_obstacles_status and not self.have_merged)
    

    def get_route_transform(self):
        """
        Get the transform of the current route point with yaw pointing to next route point.
        Uses current->next direction for accurate heading even during turns.
        
        Returns:
            carla.Transform: Transform at current route point with forward direction
        """
        route_plan = CarlaDataProvider._ego_vehicle_route[:]
        
        # Handle edge cases
        if not route_plan or len(route_plan) < 1:
            return self.ev_transform
        
        loc0 = route_plan[0][0].location
        
        # If only one point left, use its built-in rotation
        if len(route_plan) < 2:
            return carla.Transform(
                location=carla.Location(loc0.x, loc0.y, loc0.z),
                rotation=carla.Rotation(yaw=route_plan[0][0].rotation.yaw)
            )
        
        loc1 = route_plan[1][0].location
        
        # Calculate yaw from current point to next point
        dx = loc1.x - loc0.x
        dy = loc1.y - loc0.y
        
        # If points are too close, use built-in rotation
        if (dx * dx + dy * dy) < 0.01:  # < 0.1m
            yaw = route_plan[0][0].rotation.yaw
        else:
            yaw = np.rad2deg(np.arctan2(dy, dx))
        
        return carla.Transform(
            location=carla.Location(loc0.x, loc0.y, loc0.z),
            rotation=carla.Rotation(yaw=yaw)
        )


    def append_wp(self):
        temp = carla.Location(self.current_route_waypoint.transform.location.x,
                              self.current_route_waypoint.transform.location.y,
                              self.current_route_waypoint.transform.location.z)
        self.wps.append(temp)

    def get_steer_consistency(self, steer):
        angular_velocity = self.ego_actor.get_angular_velocity()
        current_angular_velocity = angular_velocity.z * np.pi / 180.0 # rad/s
        
        if self.last_angular_velocity is None:
            self.last_angular_velocity = current_angular_velocity
            return 0
        
        if len(CarlaDataProvider._ego_vehicle_route) < self.window_size:
            self.last_angular_velocity = current_angular_velocity
            return 0
        
        self.current_route_waypoint = self.map.get_waypoint(CarlaDataProvider._ego_vehicle_route[0][0].location)
        next_plan_wp = self.map.get_waypoint(CarlaDataProvider._ego_vehicle_route[1][0].location)

        if self.current_route_waypoint.lane_id != next_plan_wp.lane_id or self.current_route_waypoint.road_id != next_plan_wp.road_id:
            self.last_angular_velocity = current_angular_velocity
            return 0
        
        if ((self._blocked_by_obstacles_status or self._followed_by_emergency_status)) or\
            (self.twoway_blocked and (not self.passable)):
            self.last_angular_velocity = current_angular_velocity
            return 0

        for i in range(self.window_size):
            next_plan_wp = self.map.get_waypoint(CarlaDataProvider._ego_vehicle_route[i][0].location)
            if next_plan_wp.is_junction:
                self.last_angular_velocity = current_angular_velocity
                return 0
        
        dt = 0.1
        angular_acceleration = np.abs(current_angular_velocity - self.last_angular_velocity) / dt
        self.last_angular_velocity = current_angular_velocity
        
        # Cap angular-acceleration penalty to avoid collision spikes.
        diff = np.clip(angular_acceleration, 0, 2.0)
        return diff
    
    def is_within_distance(self, ego_loc, loc, distance):
        c_distance = abs(ego_loc.x - loc.x) < distance \
                and abs(ego_loc.y - loc.y) < distance \
                and abs(ego_loc.z - loc.z) < 8.0
        c_ev = abs(ego_loc.x - loc.x) < 1.0 and abs(ego_loc.y - loc.y) < 1.0
        return c_distance and (not c_ev)

    def get_speed_reward(self, ego_speed, desired_speed):
        if self.max_speed <= 0:
            return 0
        ego_speed = max(float(ego_speed), 0.0)
        desired_speed = max(float(desired_speed), 0.0)

        if desired_speed <= self.stop_speed_threshold:
            if ego_speed <= self.stop_speed_threshold:
                return 0.0
            stop_error = ego_speed - self.stop_speed_threshold
            stop_penalty = (stop_error / self.max_speed) * self.stop_target_penalty_scale
            return -stop_penalty

        speed_diff = ego_speed - desired_speed
        if speed_diff > 0:
            penalty = (speed_diff / self.max_speed) * self.overspeed_penalty_scale
        else:
            penalty = (-speed_diff / self.max_speed) * self.underspeed_penalty_scale
        return 1 - penalty

    def _clear_latched_red_light(self):
        self._latched_red_light_id = None
        self._latched_red_light_center = None
        self._latched_red_light_min_distance = None

    def _clear_latched_stop_sign(self):
        self._latched_stop_sign_id = None
        self._latched_stop_sign_center = None
        self._latched_stop_sign_min_distance = None
        self._latched_stop_sign_completed = False
        self._stop_sign_completed_bonus_steps = 0

    def _find_traffic_light_by_id(self, light_id):
        entry = find_traffic_light_entry_by_id(self._list_traffic_lights, light_id)
        if entry is None:
            return None, None
        traffic_light, center, _ = entry
        return traffic_light, center

    @staticmethod
    def _get_stop_sign_reference_location(stop_sign):
        try:
            return stop_sign.get_transform().transform(stop_sign.trigger_volume.location)
        except Exception:
            return stop_sign.get_transform().location

    def _find_stop_sign_by_id(self, stop_sign_id):
        if stop_sign_id is None:
            return None, None
        for stop_sign in self._list_stop_signs:
            if not getattr(stop_sign, 'is_alive', False):
                continue
            if int(stop_sign.id) == int(stop_sign_id):
                return stop_sign, self._get_stop_sign_reference_location(stop_sign)
        return None, None

    def _build_route_lookahead_locations(self, lookahead_m):
        return tl_utils_build_route_lookahead_locations(
            vehicle_route=CarlaDataProvider._ego_vehicle_route,
            lookahead_m=lookahead_m,
        )

    def _compute_red_light_target_speed(self):
        """
        Smoothly decelerate before junction entry when a red-light lock is active.
        Keep zero speed inside junction to prevent speed bouncing.
        """
        if self.current_route_waypoint is None:
            return 0.0
        if self.current_route_waypoint.is_junction:
            return 0.0

        junction_entry_wp = self.current_route_waypoint
        for _ in range(400):
            next_wps = junction_entry_wp.next(0.5)
            if not next_wps:
                break
            next_wp = next_wps[0]
            if next_wp.is_junction:
                break
            junction_entry_wp = next_wp

        dis = compute_2d_distance(self.ev_transform.location, junction_entry_wp.transform.location)
        if self.distance_light <= 0:
            return 0.0
        junction_target_speed = ((dis - 5.0) / self.distance_light) * self.max_speed
        max_speed_cap = max(self.max_speed, 0.0)
        return float(np.clip(junction_target_speed, 0.0, max_speed_cap))

    def _compute_stop_sign_target_speed(self, stop_distance):
        if self.stop_sign_decel_distance <= 0:
            return 0.0
        zero_speed_distance = max(float(self.stop_sign_completion_distance), 0.0)
        if stop_distance <= zero_speed_distance:
            return 0.0

        if self.stop_sign_decel_distance <= zero_speed_distance:
            return 0.0

        distance_span = max(self.stop_sign_decel_distance - zero_speed_distance, 1e-3)
        target_speed = ((stop_distance - zero_speed_distance) / distance_span) * self.max_speed
        max_speed_cap = max(self.max_speed, 0.0)
        return float(np.clip(target_speed, 0.0, max_speed_cap))
    
    def is_pedestrian_at_route(self, pedestrian):
        if len(CarlaDataProvider._ego_vehicle_route) < 1:
            return False
        if not pedestrian.is_alive:
            return False
        ped_loc = pedestrian.get_location()
        ped_wp = self.judge_lanetype(ped_loc, pedestrian.bounding_box.extent)
        if ped_wp.lane_type != carla.LaneType.Driving:
            return False
        lens = min(15, len(CarlaDataProvider._ego_vehicle_route))
        for i in range(lens):
            if compute_2d_distance(ped_wp.transform.location, CarlaDataProvider._ego_vehicle_route[i][0].location) < 2.0:
                temp_wp = self.map.get_waypoint(CarlaDataProvider._ego_vehicle_route[i][0].location)
                if ped_wp.lane_id == temp_wp.lane_id:
                    return True
        return False
        
    def get_desired_speed(self, info):
        self._stop_sign_passed_without_stop_penalty_pending = 0.0
        if len(CarlaDataProvider._ego_vehicle_route) < 1:
            return 0
        self.red = False

        has_relevant_green = False
        self._debug_red_lock_source = 'none'
        self._debug_red_release_reason = '-'
        self._debug_red_recent_distance = -1.0
        self._debug_red_has_relevant_green = False
        self._debug_stop_lock_source = 'none'
        self._debug_stop_release_reason = '-'
        self._debug_stop_recent_distance = -1.0
        self._debug_stop_completion_distance = -1.0
        stop_detect_wps = self._get_waypoints(
            self.ego_actor,
            proximity_threshold=self.stop_sign_detect_distance,
        )
        stop_completion_wps = self._get_waypoints(
            self.ego_actor,
            proximity_threshold=self.stop_sign_completion_distance,
        )
        if self._latched_red_light_id is not None:
            latched_light, latched_center = self._find_traffic_light_by_id(self._latched_red_light_id)
            if latched_light is None or not getattr(latched_light, 'is_alive', False):
                self._clear_latched_red_light()
                self._debug_red_release_reason = 'latched_missing'
            else:
                self._latched_red_light_center = latched_center
                latched_distance = compute_2d_distance(self.ev_transform.location, latched_center)
                self._debug_red_recent_distance = float(latched_distance)
                if (
                    self._latched_red_light_min_distance is None
                    or latched_distance < self._latched_red_light_min_distance
                ):
                    self._latched_red_light_min_distance = float(latched_distance)
                if latched_light.state == carla.TrafficLightState.Green:
                    self._clear_latched_red_light()
                    has_relevant_green = True
                    self._debug_red_release_reason = 'latched_green'
                elif (
                    self._latched_red_light_min_distance is not None
                    and latched_distance > self._latched_red_light_min_distance + self.red_light_reward_clear_distance
                ):
                    self._clear_latched_red_light()
                    self._debug_red_release_reason = 'latched_passed_clear_distance'

        lookahead_locations = self._build_route_lookahead_locations(self.traffic_light_lookahead_m)
        first_light = tl_utils_get_first_route_light(
            traffic_lights=self._list_traffic_lights,
            lookahead_locations=lookahead_locations,
            ego_location=self.ev_transform.location,
        )

        if first_light is not None:
            first_light_actor, first_light_center, _, _, first_light_distance = first_light
            if first_light_actor.state == carla.TrafficLightState.Green:
                has_relevant_green = True
            if (
                self._latched_red_light_id is None
                and first_light_actor.state == carla.TrafficLightState.Red
                and first_light_distance <= self.second_red_light_distance_threshold
            ):
                self._latched_red_light_id = int(first_light_actor.id)
                self._latched_red_light_center = first_light_center
                self._latched_red_light_min_distance = float(first_light_distance)
                self._debug_red_recent_distance = float(first_light_distance)
                self._debug_red_lock_source = 'route_first_red'

        if self._latched_red_light_id is None:
            self.have_turn2green = bool(has_relevant_green)
        self._debug_red_has_relevant_green = bool(has_relevant_green)

        red_speed_limit = None
        if self._latched_red_light_id is not None:
            self.red = True
            red_speed_limit = self._compute_red_light_target_speed()

        stop_speed_limit = None
        if self._latched_stop_sign_id is not None:
            latched_stop_sign, latched_stop_center = self._find_stop_sign_by_id(self._latched_stop_sign_id)
            if latched_stop_sign is None:
                self._clear_latched_stop_sign()
                self._debug_stop_release_reason = 'latched_missing'
            else:
                self._latched_stop_sign_center = latched_stop_center
                latched_stop_distance = compute_2d_distance(self.ev_transform.location, latched_stop_center)
                self._debug_stop_recent_distance = float(latched_stop_distance)
                if (
                    self._latched_stop_sign_min_distance is None
                    or latched_stop_distance < self._latched_stop_sign_min_distance
                ):
                    self._latched_stop_sign_min_distance = float(latched_stop_distance)

                affected_by_stop = self.is_actor_affected_by_stop(
                    stop_completion_wps,
                    latched_stop_sign,
                    proximity_threshold=self.stop_sign_completion_distance,
                )
                self._debug_stop_completion_distance = float(self.stop_sign_completion_distance)
                if (
                    not self._latched_stop_sign_completed
                    and affected_by_stop
                    and CarlaDataProvider.get_velocity(self.ego_actor) < self.SPEED_THRESHOLD
                ):
                    self._latched_stop_sign_completed = True
                    self.stop_sign_record = int(latched_stop_sign.id)

                if self._latched_stop_sign_completed:
                    if (
                        not affected_by_stop
                        or (
                            self._latched_stop_sign_min_distance is not None
                            and latched_stop_distance > self._latched_stop_sign_min_distance + self.stop_sign_reward_clear_distance
                        )
                    ):
                        self._clear_latched_stop_sign()
                        self._debug_stop_release_reason = 'completed_passed_clear_distance'
                else:
                    stop_speed_limit = self._compute_stop_sign_target_speed(latched_stop_distance)
                    if (
                        not affected_by_stop
                        and self._latched_stop_sign_min_distance is not None
                        and latched_stop_distance > self._latched_stop_sign_min_distance + self.stop_sign_reward_clear_distance
                    ):
                        self._stop_sign_passed_without_stop_penalty_pending = (
                            self.stop_sign_passed_without_stop_penalty
                        )
                        self._clear_latched_stop_sign()
                        self._debug_stop_release_reason = 'passed_without_stop'
                        stop_speed_limit = None

        if self._latched_stop_sign_id is None:
            stop_sign = self._scan_for_stop_sign(
                self.ego_wp.transform,
                stop_detect_wps,
                proximity_threshold=self.stop_sign_detect_distance,
            )
            if stop_sign is not None and int(stop_sign.id) != self.stop_sign_record:
                stop_sign_center = self._get_stop_sign_reference_location(stop_sign)
                stop_sign_distance = compute_2d_distance(self.ev_transform.location, stop_sign_center)
                self._latched_stop_sign_id = int(stop_sign.id)
                self._latched_stop_sign_center = stop_sign_center
                self._latched_stop_sign_min_distance = float(stop_sign_distance)
                self._latched_stop_sign_completed = False
                self._stop_sign_completed_bonus_steps = 0
                self._debug_stop_recent_distance = float(stop_sign_distance)
                self._debug_stop_lock_source = 'route_stop_ahead'
                stop_speed_limit = self._compute_stop_sign_target_speed(stop_sign_distance)

        def apply_speed_limits(target_speed):
            if red_speed_limit is not None:
                target_speed = min(target_speed, red_speed_limit)
            if stop_speed_limit is not None:
                target_speed = min(target_speed, stop_speed_limit)
            return target_speed

        if self._blocked_by_emergency_status:
            return 0
        if self.ego_wp.is_junction or self.current_route_waypoint.is_junction:
            if self._blocked_by_obstacles_status or self._blocked_by_emergency_status:
                return 0
            # emergency vehicles approaching with higher priority in junction
            elif info[WRAPPER_OBS_EMERGENCY_IN_VISION] and self.is_emergency_close_enough() and (not self.is_emergency_at_route()) and (not self.check_scenario_name('BlockedIntersection')):
                return 0
            elif self.scenario_name in self.JUNCTION_FIXED_DESIRED_SPEED_BY_SCENARIO:
                return apply_speed_limits(self.JUNCTION_FIXED_DESIRED_SPEED_BY_SCENARIO[self.scenario_name])
            else:
                return apply_speed_limits(self.max_speed)
        if self.ego_wp.lane_id != self.current_route_waypoint.lane_id:
            return apply_speed_limits(self.max_speed)
        if self._followed_by_emergency_status or (
            self._blocked_by_obstacles_status and self.twoway_blocked
        ):
            return apply_speed_limits(self.max_speed)

        if info[WRAPPER_OBS_EMERGENCY_IN_VISION] and self.is_emergency_close_enough() and (not self.is_emergency_at_route()) and (not self.check_scenario_name('BlockedIntersection')):
            return 0
        actor_list = getattr(self, '_step_active_actors', None)
        if actor_list is None:
            actor_list = CarlaDataProvider.get_all_actors()
        range_actors = []
        for actor in actor_list:
            tid = actor.type_id
            if (tid.startswith("vehicle") or tid.startswith("walker")
                    or 'construction' in tid):
                detect_distance = self._distance_threshold
                if tid.startswith("walker"):
                    detect_distance = self.pedestrian_distance_threshold
                if self.is_within_distance(self.ev_transform.location, actor.get_location(), detect_distance):
                    is_obstacle = False
                    for obstacle in CarlaDataProvider._actor_obstacle_map.copy():
                        if not getattr(obstacle, 'is_alive', False):
                            continue
                        if self.close_enough(actor.get_location(), obstacle.get_location()):
                            is_obstacle = True
                            break
                    if not is_obstacle:
                        range_actors.append(actor)
        
        # Pedestrians on route require a stop; nearby off-route pedestrians cap speed.
        has_pedestrian = False
        for actor in range_actors:
            if actor.is_alive and actor.type_id.startswith("walker"):
                has_pedestrian = True
                if self.is_pedestrian_at_route(actor):
                    return 0
        if has_pedestrian:
            return apply_speed_limits(2)
        
        check_wps = self._get_waypoints(self.ego_actor)
        controloss_zone = self._scan_for_contrloss_zone(self.ego_wp.transform, check_wps)
        if controloss_zone:
            if (self.controloss_zone_record and self.controloss_zone_record == controloss_zone.id):
                pass
            else:
                controloss_location = controloss_zone.get_location()
                controloss_zone_wp = self.map.get_waypoint(controloss_location)
                if self.is_actor_affected_by_controloss_zone(check_wps, controloss_zone):
                    self.controloss_zone_last += 1
                    if self.controloss_zone_last > 3:
                        self.controloss_zone_last = 0
                        self.controloss_zone_record = controloss_zone.id
                    return 0
        ego_vec = self.ego_wp.transform.get_forward_vector()
        same_lane_front_actors_distance = []
        for actor in range_actors:
            if not actor.is_alive:
                continue
            actor_location = actor.get_location()
            rec_vec = carla.Vector3D(actor_location.x-self.ego_wp.transform.location.x,
                                     actor_location.y-self.ego_wp.transform.location.y)

            res = self.compute_dot(ego_vec, rec_vec)
            actor_wp = self.judge_lanetype(actor_location, actor.bounding_box.extent)
            # detect about +- 30 degree
            if res > 0 and actor_wp.road_id == self.ego_wp.road_id and actor_wp.lane_id == self.ego_wp.lane_id and\
                self.ego_wp.lane_type != carla.LaneType.Parking:
                same_lane_front_actors_distance.append(compute_2d_distance(self.ev_transform.location, actor_location))
        if not same_lane_front_actors_distance:
            min_dis = self._distance_threshold
        else:
            min_dis = min(same_lane_front_actors_distance)
        distance_span = max(self._distance_threshold - self.safety_distance, 1e-3)
        scale = np.clip((min_dis - self.safety_distance), a_min=0.0, a_max=distance_span)
        desired_speed = (scale / distance_span) * self.max_speed
        return apply_speed_limits(desired_speed)

    def get_route_lane_deviation(self, route_transform):
        if len(CarlaDataProvider._ego_vehicle_route) < 1:
            return 0
        ego_wp = self.current_route_waypoint
        if ego_wp is None:
            ego_wp = self.map.get_waypoint(CarlaDataProvider._ego_vehicle_route[0][0].location)
        ref_vec = carla.Vector3D(
            self.ev_transform.location.x - route_transform.location.x,
            self.ev_transform.location.y - route_transform.location.y
        )
        deviation = self.compute_dot(route_transform.get_right_vector(), ref_vec) / ego_wp.lane_width
        return np.abs(deviation)
    
    
    def _refresh_borrow_lane_flags(self):
        if not self._blocked_by_obstacles_status or self.current_route_waypoint is None:
            return
        if self.have_left_samedir_lane(self.current_route_waypoint) or self.have_left_opposite_lane(self.current_route_waypoint):
            self.direction = -1
            if self.have_left_opposite_lane(self.current_route_waypoint):
                self.twoway_blocked_step += 1
                self.twoway_blocked = True
        elif self.have_right_samedir_lane(self.current_route_waypoint):
            self.direction = 1
        else:
            self.direction = 0

    def get_onlyblocker_enabled_all_lane_deviation(self, route_transform, passable=None):
        if len(CarlaDataProvider._ego_vehicle_route) < 1:
            return 0
        if passable is None:
            passable = self.passable
        self.current_route_waypoint = self.map.get_waypoint(CarlaDataProvider._ego_vehicle_route[0][0].location)
        ego_wp = self.current_route_waypoint
        
        if self._blocked_by_obstacles_status:
            if self.have_left_samedir_lane(self.current_route_waypoint) or self.have_left_opposite_lane(self.current_route_waypoint):
                self.direction = -1
                if self.have_left_opposite_lane(self.current_route_waypoint):
                    self.twoway_blocked = True
                if self.twoway_blocked and (not passable):
                    ref_vec = carla.Vector3D(self.ev_transform.location.x-route_transform.location.x,
                                                self.ev_transform.location.y-route_transform.location.y)
                    deviation = self.compute_dot(route_transform.get_right_vector(), ref_vec)/ego_wp.lane_width
                    return np.abs(deviation)
                le_trans = self.current_route_waypoint.transform
                le_vec = -1 * self.current_route_waypoint.transform.get_right_vector()
                displacement = carla.Location((ego_wp.lane_width)*le_vec.x, (ego_wp.lane_width)*le_vec.y)
                le_trans_location = carla.Location(
                                                    le_trans.location.x + displacement.x,
                                                    le_trans.location.y + displacement.y,
                                                    le_trans.location.z + displacement.z
                                                )
                ref_vec = carla.Vector3D(self.ev_transform.location.x-le_trans_location.x,
                                         self.ev_transform.location.y-le_trans_location.y)
                # opposite directions
                deviation = self.compute_dot(le_vec, ref_vec)/ego_wp.lane_width
                if np.abs(deviation) < 0.2:
                    self.have_merged = True
                return np.abs(deviation)
            elif self.have_right_samedir_lane(self.current_route_waypoint):
                self.direction = 1
                ri_trans = self.current_route_waypoint.transform
                ri_vec = self.current_route_waypoint.transform.get_right_vector()
                displacement = carla.Location((ego_wp.lane_width)*ri_vec.x, (ego_wp.lane_width)*ri_vec.y)
                ri_trans.location = ri_trans.location + displacement
                ref_vec = carla.Vector3D(self.ev_transform.location.x-ri_trans.location.x,
                                         self.ev_transform.location.y-ri_trans.location.y)
                deviation = self.compute_dot(ri_vec, ref_vec)/ego_wp.lane_width
                if np.abs(deviation) < 0.2:
                    self.have_merged = True
                return np.abs(deviation)
            else:
                ref_vec = carla.Vector3D(self.ev_transform.location.x-route_transform.location.x,
                                     self.ev_transform.location.y-route_transform.location.y)
                deviation = self.compute_dot(route_transform.get_right_vector(), ref_vec)/ego_wp.lane_width
                return np.abs(deviation)  
        elif self._followed_by_emergency_status:
            if self.have_left_samedir_lane(self.current_route_waypoint):
                self.direction = -1
                le_trans = self.current_route_waypoint.transform
                le_vec = -1 * self.current_route_waypoint.transform.get_right_vector()
                displacement = carla.Location((ego_wp.lane_width)*le_vec.x, (ego_wp.lane_width)*le_vec.y)
                le_trans.location = le_trans.location + displacement
                ref_vec = carla.Vector3D(self.ev_transform.location.x-le_trans.location.x,
                                         self.ev_transform.location.y-le_trans.location.y)
                # opposite directions
                deviation = self.compute_dot(le_vec, ref_vec)/ego_wp.lane_width
                if np.abs(deviation) < 0.2:
                    self.have_merged = True
                return np.abs(deviation)
            elif self.have_right_samedir_lane(self.current_route_waypoint):
                self.direction = 1
                ri_trans = self.current_route_waypoint.transform
                ri_vec = self.current_route_waypoint.transform.get_right_vector()
                displacement = carla.Location((ego_wp.lane_width)*ri_vec.x, (ego_wp.lane_width)*ri_vec.y)
                ri_trans.location = ri_trans.location + displacement
                ref_vec = carla.Vector3D(self.ev_transform.location.x-ri_trans.location.x,
                                         self.ev_transform.location.y-ri_trans.location.y)
                deviation = self.compute_dot(ri_vec, ref_vec)/ego_wp.lane_width
                if np.abs(deviation) < 0.2:
                    self.have_merged = True
                return np.abs(deviation)
            else:
                ref_vec = carla.Vector3D(self.ev_transform.location.x-route_transform.location.x,
                                     self.ev_transform.location.y-route_transform.location.y)
                deviation = self.compute_dot(route_transform.get_right_vector(), ref_vec)/ego_wp.lane_width
                return np.abs(deviation)   
        ref_vec = carla.Vector3D(self.ev_transform.location.x-route_transform.location.x,
                                    self.ev_transform.location.y-route_transform.location.y)
        deviation = self.compute_dot(route_transform.get_right_vector(), ref_vec)/ego_wp.lane_width
        return np.abs(deviation)            
    
    def judge_lanetype(self, location, extent):
        """Return the nearest waypoint for the lane type containing a location."""
        driving_wp = self.map.get_waypoint(location, lane_type=carla.LaneType.Driving)
        
        ref_vec = carla.Vector3D(
            location.x - driving_wp.transform.location.x,
            location.y - driving_wp.transform.location.y,
            location.z - driving_wp.transform.location.z
        )
        distance_to_road = np.abs(self.compute_dot(ref_vec, driving_wp.transform.get_right_vector()))
        # half of the lane width + the extent of the bounding box
        affect_road = distance_to_road < (0.5 * driving_wp.lane_width + extent.y)
        
        # get the waypoint of the parking and sidewalk
        parking_wp = self.map.get_waypoint(location, lane_type=carla.LaneType.Parking)
        sidewalk_wp = self.map.get_waypoint(location, lane_type=carla.LaneType.Sidewalk)
        
        # build the candidate list
        candidates = [driving_wp] if affect_road else []
        candidates.extend([wp for wp in [parking_wp, sidewalk_wp] if wp])
        
        if not candidates:
            return self.map.get_waypoint(location, lane_type=carla.LaneType.Shoulder)
        
        return min(candidates, key=lambda wp: compute_2d_distance(location, wp.transform.location))

    def is_parking_exit_deviation_exempt(self):
        if not self.check_scenario_name('ParkingExit'):
            return False
        if self.ego_actor is None or self.map is None:
            return False

        ego_location = self.ego_actor.get_transform().location
        driving_wp = self.map.get_waypoint(ego_location, lane_type=carla.LaneType.Driving)
        parking_wp = self.map.get_waypoint(ego_location, lane_type=carla.LaneType.Parking)
        if parking_wp is None:
            return False

        driving_distance = (
            compute_2d_distance(ego_location, driving_wp.transform.location)
            if driving_wp is not None else float('inf')
        )
        parking_distance = compute_2d_distance(ego_location, parking_wp.transform.location)
        return parking_distance + 0.1 < driving_distance
    
    
    def detect_blocked_by_obstacles(self):
        inx = min(self.window_size, len(CarlaDataProvider._ego_vehicle_route))-1
        if inx < 0:
            self.direction = 0
            return False
        at_routeplan_obstacles_wps = []
        at_routeplan_obstacles = []
        for obstacle in CarlaDataProvider._actor_obstacle_map.copy():
            if not obstacle.is_alive:
                continue
            obstacle_location = obstacle.get_location()
            if compute_2d_distance(self.current_route_waypoint.transform.location, obstacle_location) < self.obstacle_blocked_distance:
                obstacle_wp = self.judge_lanetype(obstacle_location, obstacle.bounding_box.extent)
                if obstacle_wp.lane_type != carla.LaneType.Driving:
                    continue
                if obstacle_wp.is_junction:
                    continue
                if self.is_obstacle_at_route(obstacle_wp):
                    at_routeplan_obstacles_wps.append(obstacle_wp)
                    at_routeplan_obstacles.append(obstacle)
        if not at_routeplan_obstacles_wps:
            self.direction = 0
            return False
        return True

    def detect_followed_by_emergency(self, info):
        """Detect if the ego vehicle is being followed by an emergency vehicle"""
        if len(CarlaDataProvider._ego_vehicle_route) < self.window_size:
            self.direction = 0
            return False
        
        route_wp = self.current_route_waypoint
        route_loc = route_wp.transform.location
        detect_range = 3.0 * self.safety_distance
        has_emergency_in_birdview = info[WRAPPER_OBS_EMERGENCY_IN_VISION]
        
        for emergency in CarlaDataProvider._actor_emergency_map.copy():
            if not emergency.is_alive:
                continue
            
            emg_loc = emergency.get_location()
            if compute_2d_distance(emg_loc, route_loc) >= detect_range:
                continue
            
            emg_wp = self.judge_lanetype(emg_loc, emergency.bounding_box.extent)
            if emg_wp.lane_type != carla.LaneType.Driving or emg_wp.is_junction:
                continue
            
            # check if in the same lane and road
            if emg_wp.lane_id != route_wp.lane_id or emg_wp.road_id != route_wp.road_id:
                continue
            
            # check if behind or within 10m ahead
            ref_vec = carla.Vector3D(emg_loc.x - route_loc.x, emg_loc.y - route_loc.y)
            forward_dist = self.compute_dot(ref_vec, route_wp.transform.get_forward_vector())
            
            if forward_dist < self.safety_distance and has_emergency_in_birdview:
                logger.debug("Followed By An Emergency Car")
                return True
        
        self.direction = 0
        return False

    
    def blocked_by_emergency(self):
        inx = min(self.window_size, len(CarlaDataProvider._ego_vehicle_route))-1
        if inx < 0:
            return False
        if not self.current_route_waypoint.is_junction:
            return False
        next_windowsize_wp = self.map.get_waypoint(CarlaDataProvider._ego_vehicle_route[inx][0].location)
        for emergency in CarlaDataProvider._actor_emergency_map.copy():
            if not emergency.is_alive:
                continue
            emergency_location = emergency.get_location()
            if (compute_2d_distance(self.current_route_waypoint.transform.location, emergency_location) < 1.1 * self.safety_distance):
                emergency_wp = self.judge_lanetype(emergency_location, emergency.bounding_box.extent)
                if emergency_wp.lane_type != carla.LaneType.Driving:
                    return False
                if emergency_wp.is_junction:
                        return False
                ref_vec = carla.Vector3D(self.current_route_waypoint.transform.location.x - emergency_location.x, self.current_route_waypoint.transform.location.y-emergency_location.y)
                if self.current_route_waypoint.is_junction:
                    if emergency_wp.lane_id == next_windowsize_wp.lane_id and\
                        (self.compute_dot(ref_vec, emergency_wp.transform.get_forward_vector()) < 0):
                        self.record_emergency_location = emergency_location
                        logger.debug("blocked by emergency car")
                        return True
                else:
                    if emergency_wp.road_id == self.current_route_waypoint.road_id and emergency_wp.lane_id == self.current_route_waypoint.lane_id and\
                        (self.compute_dot(ref_vec, emergency_wp.transform.get_forward_vector()) < 0):
                        self.record_emergency_location = emergency_location
                        logger.debug("blocked by emergency car")
                        return True
        return False
    
                
    def is_single_lane(self, wp):
        if (not wp.get_left_lane() or wp.get_left_lane().lane_type!=carla.libcarla.LaneType.Driving) \
            and (not wp.get_right_lane() or wp.get_right_lane().lane_type!=carla.libcarla.LaneType.Driving):
            return True
        return False

    def have_left_opposite_lane(self, wp):
        if (wp.get_left_lane() and wp.get_left_lane().lane_type==carla.libcarla.LaneType.Driving) \
            and (wp.lane_change == carla.libcarla.LaneChange.NONE or wp.lane_change == carla.libcarla.LaneChange.Right):
            return True
        return False

    def have_left_samedir_lane(self, wp):
        if (wp.get_left_lane() and wp.get_left_lane().lane_type==carla.libcarla.LaneType.Driving) and (wp.lane_change == carla.libcarla.LaneChange.Both or wp.lane_change == carla.libcarla.LaneChange.Left):
            return True
        return False
            
    def have_right_samedir_lane(self, wp):
        if (wp.get_right_lane() and wp.get_right_lane().lane_type==carla.libcarla.LaneType.Driving) \
            and (wp.lane_change == carla.libcarla.LaneChange.Both or wp.lane_change == carla.libcarla.LaneChange.Right):
            return True
        return False
    
    def compute_dot(self, a_rot, b_rot):
        return a_rot.x * b_rot.x + a_rot.y * b_rot.y
    
    def point_inside_boundingbox(self, point, bb_center, bb_extent, multiplier=1.2):

        # pylint: disable=invalid-name
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

    def is_actor_affected_by_stop(self, wp_list, stop, proximity_threshold=None):
        """
        Check if the given actor is affected by the stop.
        Without using waypoints, a stop might not be detected if the actor is moving at the lane edge.
        """
        # Quick distance test
        stop_location = stop.get_transform().transform(stop.trigger_volume.location)
        actor_location = wp_list[0].transform.location
        threshold = self.PROXIMITY_THRESHOLD if proximity_threshold is None else float(proximity_threshold)
        if compute_2d_distance(stop_location, actor_location) > threshold:
            return False

        # Check if the any of the actor wps is inside the stop's bounding box.
        # Using more than one waypoint removes issues with small trigger volumes and backwards movement
        stop_extent = stop.trigger_volume.extent
        for actor_wp in wp_list:
            if self.point_inside_boundingbox(actor_wp.transform.location, stop_location, stop_extent):
                return True

        return False
       
    def is_actor_affected_by_controloss_zone(self, wp_list, zone):
        """
        Check if the given actor is affected by the zone.
        Without using waypoints, a zone might not be detected if the actor is moving at the lane edge.
        """
        # Quick distance test
        zone_location = zone.get_transform().transform(zone.bounding_box.location)
        actor_location = wp_list[0].transform.location
        if compute_2d_distance(zone_location, actor_location)> self.PROXIMITY_THRESHOLD:
            return False

        # Check if the any of the actor wps is inside the zone's bounding box.
        # Using more than one waypoint removes issues with small trigger volumes and backwards movement
        zone_extent = 1.3 * zone.bounding_box.extent
        for actor_wp in wp_list:
            if self.point_inside_boundingbox(actor_wp.transform.location, zone_location, zone_extent):
                return True

        return False
    
    def _get_waypoints(self, actor, proximity_threshold=None):
        """Returns a list of waypoints starting from the ego location and a set amount forward"""
        wp_list = []
        threshold = self.PROXIMITY_THRESHOLD if proximity_threshold is None else float(proximity_threshold)
        steps = max(1, int(threshold / self.WAYPOINT_STEP))

        # Add the actor location
        wp = self.map.get_waypoint(actor.get_location())
        wp_list.append(wp)

        # And its forward waypoints
        next_wp = wp
        for _ in range(steps):
            next_wps = next_wp.next(self.WAYPOINT_STEP)
            if not next_wps:
                break
            next_wp = next_wps[0]
            wp_list.append(next_wp)

        return wp_list
    
    def _scan_for_stop_sign(self, actor_transform, wp_list, proximity_threshold=None):
        """
        Check the stop signs to see if any of them affect the actor.
        Ignore all checks when going backwards or through an opposite direction"""

        actor_direction = actor_transform.get_forward_vector()

        # Ignore all when going backwards
        actor_velocity = self.ego_actor.get_velocity()
        if actor_velocity.dot(actor_direction) < -0.17:  # 100º, just in case
            return None

        # Ignore all when going in the opposite direction
        lane_direction = wp_list[0].transform.get_forward_vector()
        if actor_direction.dot(lane_direction) < -0.17:  # 100º, just in case
            return None

        relevant_stop_signs = []
        for stop in self._list_stop_signs:
            if self.is_actor_affected_by_stop(wp_list, stop, proximity_threshold=proximity_threshold):
                stop_center = self._get_stop_sign_reference_location(stop)
                stop_distance = compute_2d_distance(self.ev_transform.location, stop_center)
                relevant_stop_signs.append((stop_distance, stop))
        if relevant_stop_signs:
            relevant_stop_signs.sort(key=lambda item: item[0])
            return relevant_stop_signs[0][1]
            
    def _scan_for_contrloss_zone(self, actor_transform, wp_list):
        """
        Check the stop signs to see if any of them affect the actor.
        Ignore all checks when going backwards or through an opposite direction"""

        actor_direction = actor_transform.get_forward_vector()

        # Ignore all when going backwards
        actor_velocity = self.ego_actor.get_velocity()
        if actor_velocity.dot(actor_direction) < -0.17:  # 100º, just in case
            return None

        # Ignore all when going in the opposite direction
        lane_direction = wp_list[0].transform.get_forward_vector()
        if actor_direction.dot(lane_direction) < -0.17:  # 100º, just in case
            return None

        for zone in self._list_controloss_zones:
            if self.is_actor_affected_by_controloss_zone(wp_list, zone):
                return zone
            
    def affected_by_traffic_light(self, traffic_light, center):
        lookahead_locations = self._build_route_lookahead_locations(self.traffic_light_lookahead_m)
        return tl_utils_is_light_affecting_route(
            traffic_light=traffic_light,
            center=center,
            lookahead_locations=lookahead_locations,
            extra_locations=self.wps,
        )
                
    def is_obstacle_at_route(self, obstacle_wp):
        """Check if the obstacle is on the route (same lane and ahead or within one car length behind)"""
        if not CarlaDataProvider._ego_vehicle_route:
            return False
        
        route_wp = self.current_route_waypoint
        obs_loc = obstacle_wp.transform.location
        route_loc = route_wp.transform.location
        
        # check if the obstacle is in the same lane
        if obstacle_wp.lane_id != route_wp.lane_id:
            return False
        
        # check if the obstacle is ahead or within one car length behind
        ref_vec = carla.Vector3D(obs_loc.x - route_loc.x, obs_loc.y - route_loc.y)
        forward_dist = self.compute_dot(ref_vec, route_wp.transform.get_forward_vector())
        
        return forward_dist > -self.ego_actor.bounding_box.extent.x
    
    def is_emergency_close_enough(self):
        if len(CarlaDataProvider._ego_vehicle_route) < 1 or not self.current_route_waypoint:
            return False
        dir_vec = self.current_route_waypoint.transform.get_right_vector()
        for emergency in CarlaDataProvider._actor_emergency_map.copy():
            if not emergency.is_alive:
                continue
            emergency_location = emergency.get_location()
            emergency_wp = self.judge_lanetype(emergency_location, emergency.bounding_box.extent)
            ref_vec = carla.Vector3D(emergency_wp.transform.location.x-self.current_route_waypoint.transform.location.x,
                                     emergency_wp.transform.location.y-self.current_route_waypoint.transform.location.y)
            if np.abs(self.compute_dot(ref_vec, dir_vec)) < self.EMERGENCY_VEHICLES_CLOSE_THRESHOLD:
                return True
        return False
        
    def is_emergency_at_route(self):
        if len(CarlaDataProvider._ego_vehicle_route) < 1:
            return False
        lens = min(50, len(CarlaDataProvider._ego_vehicle_route))
        for i in range(lens):
            for emergency in CarlaDataProvider._actor_emergency_map.copy():
                if not emergency.is_alive:
                    continue
                emergency_location = emergency.get_location()
                emergency_wp = self.judge_lanetype(emergency_location, emergency.bounding_box.extent)
                if compute_2d_distance(emergency_wp.transform.location, CarlaDataProvider._ego_vehicle_route[i][0].location) < 1.0:
                    temp_wp = self.map.get_waypoint(CarlaDataProvider._ego_vehicle_route[i][0].location)
                    if emergency_wp.lane_id == temp_wp.lane_id:
                        return True
        return False
    
    def get_borrow_lane_blocking_distance(self):
        if not self.twoway_blocked or self.direction == 0:
            return float('inf')
        if self.current_route_waypoint is None or self.ev_transform is None:
            return float('inf')

        if self.direction < 0:
            target_wp = self.current_route_waypoint.get_left_lane()
        else:
            target_wp = self.current_route_waypoint.get_right_lane()
        if target_wp is None:
            return float('inf')

        lookahead = 50.0
        ego_loc = self.ev_transform.location
        route_forward = self.current_route_waypoint.transform.get_forward_vector()
        actor_list = getattr(self, '_step_active_actors', None)
        if actor_list is None:
            actor_list = CarlaDataProvider.get_all_actors()

        min_distance = float('inf')
        for actor in actor_list:
            if actor.id == self.ego_actor.id:
                continue
            if not getattr(actor, 'is_alive', False):
                continue
            if not (actor.type_id.startswith("vehicle") or actor.type_id.startswith("walker")):
                continue
            actor_location = actor.get_location()
            if not self.is_within_distance(ego_loc, actor_location, lookahead):
                continue
            actor_wp = self.judge_lanetype(actor_location, actor.bounding_box.extent)
            if actor_wp is None:
                continue
            if actor_wp.road_id != target_wp.road_id or actor_wp.lane_id != target_wp.lane_id:
                continue

            rel_vec = carla.Vector3D(actor_location.x - ego_loc.x, actor_location.y - ego_loc.y)
            longitudinal = self.compute_dot(rel_vec, route_forward)
            if longitudinal < 5.0 or longitudinal > lookahead:
                continue
            min_distance = min(min_distance, compute_2d_distance(ego_loc, actor_location))

        return min_distance
    
    def close_enough(self, a, b):
        c_distance = abs(a.x - b.x) < 1.0 \
            and abs(a.y - b.y) < 1.0 
        return c_distance
        
    
    def check_scenario_name(self, name):
        if self.scenario_name == name or self.scenario_name == 'Shadow' + name:
            return True
        return False
