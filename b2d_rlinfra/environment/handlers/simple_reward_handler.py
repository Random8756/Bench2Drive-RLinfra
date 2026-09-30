"""Simple reward handler.

A lighter alternative to ``reward_handler.py``: computes a single dense
reward per step combining progress, smoothness, infraction penalties and
termination shaping.
"""
import logging
import math
from collections import deque
from dataclasses import dataclass, field

import carla
import numpy as np

logger = logging.getLogger("Interface Wrapper")
from scipy.signal import savgol_filter

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.timer import GameTime

from b2d_rlinfra.environment.handlers.criteria import run_red_light, run_stop_sign, run_stop_sign2, collision, blocked, route_completion, in_route_test, outside_route_lanes
from b2d_rlinfra.environment.handlers.criteria import rl_utils as rl_u
from b2d_rlinfra.environment.info_keys import (
    WRAPPER_TERM_TRIGGERED,
    WRAPPER_REWARD_SIMPLE_TERMINATED,
    WRAPPER_REWARD_SIMPLE_TRUNCATED,
)

__layer__ = (2, "Environment")


REWARD_CONFIG = {
  # Basic runtime parameters.
  'time_interval': 0.1,
  'action_repeat': 1,
  'eval_time': 900.0,
  'ego_extent_x': 2.5,

  # Termination checks.
  'use_leave_route_done': False,
  'min_thresh_lat_dist': 3.0,
  'off_road_term_perc': 0.95,
  'use_off_road_term': True,

  # Traffic lights and stop signs.
  'consider_tl': True,
  'penalize_yellow_light': False,
  'use_new_stop_sign_detector': True,

  # Terminal reward.
  'terminal_reward': -10.0,
  'use_termination_hint': True,
  'terminal_hint': 5.0,
  'use_rl_termination_hint': False,

  # Soft lane constraints.
  'use_outside_route_lanes': True,
  'use_perc_progress': True,
  'lane_distance_violation_threshold': 0.5,
  'lane_dist_penalty_softener': 0.5,

  # Speeding.
  'speeding_infraction': True,
  'rr_maximum_speed': 10.0,
  'max_overspeed_value_threshold': 5.0,

  # Comfort.
  'use_comfort_infraction': True,
  'comfort_penalty_ticks': 10,
  'comfort_penalty_factor': 0.5,
  'max_lon_accel': 10.0,
  'min_lon_accel': -20.0,
  'max_abs_lat_accel': 9.0,
  'max_abs_mag_jerk': 30.0,
  'max_abs_lon_jerk': 30.0,
  'max_abs_yaw_rate': 1.0,
  'max_abs_yaw_accel': 3.0,

  # Time-to-collision.
  'use_ttc': True,
  'ttc_resolution': 1,
  'ttc_penalty_ticks': 10,

  # Vehicle-distance forecast.
  'use_vehicle_close_penalty': True,
  'ego_forecast_time': 2.0,
  'ego_forecast_min_speed': 1.0,

  # Other soft constraints.
  'use_min_speed_infraction': False,
  'use_max_change_penalty': False,
  'max_change': 0.3,

  # Reward form.
  'use_single_reward': True,
  'use_survival_reward': False,
  'survival_reward_magnitude': 0.01,
  'positive_reward_scale': 1.0,
}


def make_simple_terminate_event(event_type, reason, details=None):
  """Build a serializable termination event emitted by SimpleReward."""
  clean_details = {}
  if isinstance(details, dict):
    for key, value in details.items():
      if isinstance(value, np.integer):
        clean_details[key] = int(value)
      elif isinstance(value, np.floating):
        clean_details[key] = float(value)
      elif isinstance(value, np.bool_):
        clean_details[key] = bool(value)
      elif isinstance(value, (int, float, str, bool, type(None))):
        clean_details[key] = value
      else:
        clean_details[key] = str(value)

  return {
    'event_type': event_type,
    'source': 'simple_reward',
    'reason': reason,
    'details': clean_details,
  }


def merge_simple_terminate_events(info):
  """Append SimpleReward events to ``info['terminate_events']`` in place."""
  if not isinstance(info, dict):
    return info

  simple_events = info.get('simple_terminate_events')
  if not isinstance(simple_events, list) or not simple_events:
    return info

  terminate_events = info.get('terminate_events')
  if not isinstance(terminate_events, list):
    terminate_events = []
    info['terminate_events'] = terminate_events

  hard_events = [
    event for event in simple_events
    if isinstance(event, dict) and event.get('event_type') != 'ROUTE_COMPLETION'
  ]
  route_completion_events = [
    event for event in simple_events
    if isinstance(event, dict) and event.get('event_type') == 'ROUTE_COMPLETION'
  ]

  if hard_events:
    first_route_completion_idx = len(terminate_events)
    for idx, event in enumerate(terminate_events):
      if isinstance(event, dict) and event.get('event_type') == 'ROUTE_COMPLETION':
        first_route_completion_idx = idx
        break
    terminate_events[first_route_completion_idx:first_route_completion_idx] = hard_events

  terminate_events.extend(route_completion_events)

  return info


def resolve_simple_termination(
    collision_detected=False,
    ran_red_light=False,
    ran_stop_sign=False,
    ego_blocked=False,
    route_deviation=False,
    timeout=False,
    finished_route=False,
    current_route_completion=0.0,
    terminal_reward_config=0.0,
    use_termination_hint=False,
    terminal_hint=0.0,
    use_rl_termination_hint=False,
    simple_route_lat_dist=None,
    route_deviation_threshold=None,
    timestamp=None,
    eval_time=None):
  """Resolve SimpleReward terminal state, reward and event metadata."""
  hard_termination = (
      collision_detected or ran_red_light or ran_stop_sign or
      ego_blocked or route_deviation or timeout
  )

  terminal_reward = 0.0
  if hard_termination:
    terminal_reward = terminal_reward_config
    if use_termination_hint:
      if use_rl_termination_hint:
        hint_condition = (collision_detected or ran_red_light)
      else:
        hint_condition = collision_detected
      if hint_condition:
        terminal_reward -= terminal_hint

  simple_terminate_events = []
  infraction_types = []

  if collision_detected:
    infraction_types.append('collision')
    simple_terminate_events.append(make_simple_terminate_event(
      'COLLISION_UNKNOWN',
      'collision',
    ))
  if ran_red_light:
    infraction_types.append('ran_red_light')
    simple_terminate_events.append(make_simple_terminate_event(
      'TRAFFIC_LIGHT_INFRACTION',
      'ran_red_light',
    ))
  if ran_stop_sign:
    infraction_types.append('ran_stop_sign')
    simple_terminate_events.append(make_simple_terminate_event(
      'STOP_INFRACTION',
      'ran_stop_sign',
    ))
  if ego_blocked:
    infraction_types.append('ego_blocked')
    simple_terminate_events.append(make_simple_terminate_event(
      'VEHICLE_BLOCKED',
      'ego_blocked',
    ))
  if route_deviation:
    infraction_types.append('route_deviation')
    simple_terminate_events.append(make_simple_terminate_event(
      'ROUTE_DEVIATION',
      'route_deviation',
      {
        'simple_route_lat_dist': simple_route_lat_dist,
        'threshold': route_deviation_threshold,
      },
    ))
  if timeout:
    infraction_types.append('timeout')
    simple_terminate_events.append(make_simple_terminate_event(
      'SCENARIO_TIMEOUT',
      'timeout',
      {
        'timestamp': timestamp,
        'eval_time': eval_time,
      },
    ))
  if finished_route:
    simple_terminate_events.append(make_simple_terminate_event(
      'ROUTE_COMPLETION',
      'route_completed',
      {'route_completed': float(current_route_completion)},
    ))

  return {
    'hard_termination': bool(hard_termination),
    'termination': bool(hard_termination or finished_route),
    'truncation': False,
    'terminal_reward': float(terminal_reward),
    'infraction_types': infraction_types,
    'simple_terminate_events': simple_terminate_events,
  }


@dataclass
class EgoState:
  longitudinal_acceleration: deque = field(default_factory=lambda: deque(maxlen=15))
  lateral_acceleration: deque = field(default_factory=lambda: deque(maxlen=15))
  acceleration_magnitude: deque = field(default_factory=lambda: deque(maxlen=15))
  yaw: deque = field(default_factory=lambda: deque(maxlen=15))


class SimpleReward(object):
  '''
    A simple reward, that only tells the agent what to do not how to do it. E.g. it does not compute any optimal speed
    like the roach reward. It's designed to mimic the DS (with the same global optimum) but be easier to optimize.
    '''

  def __init__(self, vehicle, world_map, world, config, route):
    self.vehicle = vehicle
    self.config = config
    self.route = route if route else []
    self.has_route = len(self.route) >= 2
    
    from b2d_rlinfra.environment.handlers.criteria.traffic_light import TrafficLightHandler
    waypoint_list = []
    if self.route:
      for item in self.route:
        if isinstance(item, tuple) and len(item) >= 2:
          transform = item[0]
          waypoint = world_map.get_waypoint(transform.location, project_to_road=True)
          if waypoint:
            waypoint_list.append(waypoint)
    TrafficLightHandler.reset(world, world_map, waypoint_list if waypoint_list else None, config)
    
    self.red_light_infraction_detector = None
    if self.config.consider_tl:
      if TrafficLightHandler.num_tl > 0:
        self.red_light_infraction_detector = run_red_light.RunRedLight(world_map, self.config.penalize_yellow_light)
      else:
        logger.warning("Traffic light detection disabled: no traffic lights found in current world")
    if self.config.use_new_stop_sign_detector:
      self.stop_infraction_detector = run_stop_sign2.RunStopSign2(world, world_map)
    else:
      self.stop_infraction_detector = run_stop_sign.RunStopSign(world, world_map)
    self.collision_detector = collision.Collision(vehicle, world)
    self.block_detector = blocked.Blocked()
    self.route_completion = (
      route_completion.RouteCompletionTest(self.route, world_map) if self.has_route else None
    )
    self.outside_route_lanes = outside_route_lanes.OutsideRouteLanesTest(vehicle, world_map)
    self.in_route = in_route_test.InRouteTest(vehicle, self.route) if self.has_route else None
    self.world = world
    self.world_map = world_map
    self.last_route_completion = 0.0
    self.last_acceleration = np.array([0.0, 0.0])
    self.last_rotation = None
    self.last_abs_yaw_diff_rad = 0.0
    self.past_mag_jerk = deque([0.0, 0.0, 0.0, 0.0, 0.0], maxlen=5)
    self.past_lon_jerk = deque([0.0, 0.0, 0.0, 0.0, 0.0], maxlen=5)
    self.ego_model = EgoModel(dt=self.config.time_interval)
    self.first_frame = True
    self.last_action = carla.VehicleControl(steer=0.0, throttle=0.0, brake=0.0)

    # For TTC, only need to compute once
    self.ttc_future_time_deltas = np.arange(1.0, 10, self.config.ttc_resolution, dtype=int) * self.config.time_interval
    self.remaining_ttc_penalty_ticks = 0  # How many ticks the agent is still punished for violating TTC.

    # For comfort
    self.ego_state_history = EgoState()
    self.dx = self.config.time_interval * self.config.action_repeat
    self.remaining_comfort_penalty_ticks = np.zeros(6)  # For each of the 6 individual comfort metrics.
    # self.comfort_histogram = {'acc_lon': [],
    #                           'acc_lat': [],
    #                           'jerk': [],
    #                           'jerk_lon': [],
    #                           'yaw_rate': [],
    #                           'yaw_acceleration': []}

  # We keep collision_with_pedestrian to have a common interface but it is not used.
  def get(
      self,
      timestamp,
      waypoint_route,
      collision_with_pedestrian=None,  # pylint: disable=locally-disabled, unused-argument
      vehicles_all=(),
      walkers_all=(),
      static_all=(),
      perc_off_road=None):  # pylint: disable=locally-disabled, unused-argument

    waypoint_route = waypoint_route if waypoint_route else []
    if perc_off_road is None:
      perc_off_road = 0.0

    #########################################################################
    # Compute termination conditions and terminal reward.
    #########################################################################
    ego_vehicle_location = self.vehicle.get_location()
    ego_vehicle_transform = self.vehicle.get_transform()
    ev_vel = self.vehicle.get_velocity()  # in m/s
    ev_speed = np.linalg.norm(np.array([ev_vel.x, ev_vel.y]))

    # Done condition 1: vehicle blocked
    ego_blocked = self.block_detector.tick(self.vehicle, timestamp) is not None

    # Done condition 2: lateral distance too large
    # ego point is in route coordinate frame. x front y right. so y contains the lateral distance
    simple_route_lat_dist = None
    simple_route_deviation = False
    route_deviation = False
    if waypoint_route:
      closest_route_point = self.get_closest_route_point(waypoint_route)
      simple_route_lat_dist = float(abs(closest_route_point[1]))
      simple_route_deviation = simple_route_lat_dist > self.config.min_thresh_lat_dist
      route_deviation = bool(self.config.use_leave_route_done and simple_route_deviation)
    # else:
    #   # Ego agent left the road altogether terminate episode.
    #   if self.world_map.get_waypoint(ego_vehicle_location, project_to_road=False) is None:
    #     route_deviation = True

    if self.config.consider_tl and self.red_light_infraction_detector is not None:
      # Done condition 3: running red light
      ran_red_light = self.red_light_infraction_detector.tick(self.vehicle) is not None
    else:
      ran_red_light = False

    # # Done condition 4: collision
    collision_detected = self.collision_detector.tick(self.vehicle, timestamp) is not None

    # Done condition 5: run stop sign
    stop_criteria = self.stop_infraction_detector.tick(self.vehicle)
    ran_stop_sign = (stop_criteria is not None) and (stop_criteria['event'] == 'run')

    # Done condition 6: Agent is too close to other actor
    is_vehicle_too_close = False
    if self.config.use_vehicle_close_penalty:
      is_vehicle_too_close = self.vehicle_too_close(ego_vehicle_transform, ev_speed, self.vehicle.get_control(),
                                                    vehicles_all, walkers_all)

    # Done condition 7: Route deviation
    route_deviation_2 = False
    if self.in_route is not None:
      route_deviation_2 = not self.in_route.update()

    # Done condition 8: Driving off the drivable area
    off_road_term = False
    if perc_off_road > self.config.off_road_term_perc and self.config.use_off_road_term:
      off_road_term = True

    # Soft penalty 0: Outside route lanes
    outside_lanes = False
    if self.config.use_outside_route_lanes:
      outside_lanes = self.outside_route_lanes.update()

    # Soft penalty 1: check if agent drives on lane markings. Around 0.2ms
    current_wp = self.world_map.get_waypoint(ego_vehicle_location, project_to_road=True)
    perc_dist_to_centerline = 1.0

    if current_wp is not None and not current_wp.is_junction:
      if self.config.use_perc_progress:
        close_point_global = np.array([current_wp.transform.location.x, current_wp.transform.location.y])
        # No next point, so we use the orientation of the route waypoint as direction of the route.
        yaw_route = current_wp.transform.rotation.yaw
        ego_pos = np.array([ego_vehicle_location.x, ego_vehicle_location.y])

        ego_in_lane_coordinate = rl_u.inverse_conversion_2d(ego_pos, close_point_global, np.deg2rad(yaw_route))
        lat_dist_to_closest_lane_center = abs(ego_in_lane_coordinate[1])
        violation_length = np.clip(current_wp.lane_width * 0.5 - self.config.lane_distance_violation_threshold,
                                   a_min=0.000001,
                                   a_max=None)
        violtion_percent = (lat_dist_to_closest_lane_center -
                            self.config.lane_distance_violation_threshold) / violation_length
        perc_dist_to_centerline = (
            1.0 - np.clip(violtion_percent, a_min=0.0, a_max=1.0) * self.config.lane_dist_penalty_softener)

    # Soft penalty 2: Is the agent speed within speed limit?
    agent_too_fast = False
    speed_penalty = 1.0
    if self.config.speeding_infraction:
      speed_limit = self.vehicle.get_speed_limit()
      if isinstance(speed_limit, float):
        # Speed limit is in km/h we compute with m/s, so we convert it by / 3.6
        speed_limit = speed_limit / 3.6
      else:
        #  Car can have no speed limit right after spawning
        speed_limit = self.config.rr_maximum_speed

      exceeding_speed = ev_speed - speed_limit
      if exceeding_speed > 0.0:
        violation_loss = exceeding_speed / self.config.max_overspeed_value_threshold
        speed_penalty = max(0.0, 1.0 - violation_loss)
        agent_too_fast = True

    # Soft penalty 3: Is the agent speed within comfort limit?
    comfort_penalty = 1.0
    if self.config.use_comfort_infraction:
      comfort_penalty = self.compute_comfort_penalty()

    # Soft penalty 4: Is actor too slow?
    fraction_of_speed = 1.0
    if self.config.use_min_speed_infraction:
      fraction_of_speed = self.is_ego_too_slow(ev_speed, vehicles_all)

    # Soft penalty 5: Did action change too much
    action_changed_too_much = False
    if self.config.use_max_change_penalty:
      action = self.vehicle.get_control()
      steer_diff = abs(action.steer - self.last_action.steer)
      throt_diff = abs(action.throttle - self.last_action.throttle)
      brake_diff = abs(action.brake - self.last_action.brake)
      if (steer_diff > self.config.max_change or throt_diff > self.config.max_change or
          brake_diff > self.config.max_change):
        action_changed_too_much = True
      self.last_action = action

    # Soft penalty 6: TTC violation
    ttc_penalty = False
    if self.config.use_ttc:
      if self.remaining_ttc_penalty_ticks > 0:
        self.remaining_ttc_penalty_ticks -= 1

      ttc_penalty = self.does_agent_violate_ttc(ego_vehicle_transform, ev_speed, vehicles_all, walkers_all, static_all)

      if ttc_penalty:
        self.remaining_ttc_penalty_ticks = self.config.ttc_penalty_ticks

      if self.remaining_ttc_penalty_ticks > 0:
        ttc_penalty = True

    timeout = timestamp > self.config.eval_time

    finished_route = False
    if len(waypoint_route) < 2:
      finished_route = True

    # if route_deviation:
    #   print('route_deviation')
    # if route_deviation_2:
    #   print('route_deviation 2')
    # if ran_red_light:
    #   print('Run Red Light')
    # if ran_stop_sign:
    #   print('Run Stop Sign')
    # if collision_detected:
    #   print('Collision detected')
    # if ego_blocked:
    #   print('Vehicle is stuck')
    # if timeout:
    #   print('Agent timed out.')
    # if finished_route:
    #   print('Finished route')
    # if is_vehicle_too_close:
    #   print('Agent is too close to the leading vehicle.')
    # if off_road_term:
    #   print('Agent drove off the drivable area.')
    # if agent_too_fast:
    #   print('Agent is driving too fast.')
    # if agent_drives_uncomfortable:
    #   print('Agent is driving unconformable.')
    # if outside_lanes:
    #   print('outside_lanes')
    # if action_changed_too_much:
    #   print('Action changed too much')

    soft_penalty_conditions = {
        'vehicle_too_close': is_vehicle_too_close,
        'route_deviation_2': route_deviation_2,
        'off_road': off_road_term,
        'outside_lanes': outside_lanes
    }

    if self.route_completion is not None:
      current_route_completion = self.route_completion.update(self.vehicle)
    else:
      current_route_completion = self.last_route_completion
    progress_reward = current_route_completion - self.last_route_completion
    self.last_route_completion = current_route_completion

    termination_info = resolve_simple_termination(
      collision_detected=collision_detected,
      ran_red_light=ran_red_light,
      ran_stop_sign=ran_stop_sign,
      ego_blocked=ego_blocked,
      route_deviation=route_deviation,
      timeout=timeout,
      finished_route=finished_route,
      current_route_completion=current_route_completion,
      terminal_reward_config=self.config.terminal_reward,
      use_termination_hint=self.config.use_termination_hint,
      terminal_hint=self.config.terminal_hint,
      use_rl_termination_hint=self.config.use_rl_termination_hint,
      simple_route_lat_dist=simple_route_lat_dist,
      route_deviation_threshold=self.config.min_thresh_lat_dist,
      timestamp=timestamp,
      eval_time=self.config.eval_time,
    )
    hard_termination = termination_info['hard_termination']
    termination = termination_info['termination']
    truncation = termination_info['truncation']
    terminal_reward = termination_info['terminal_reward']
    
    penalty_info = {
      'outside_lanes': False,
      'agent_too_fast': False,
      'ttc_penalty': False,
      'comfort_penalty': False,
      'lane_deviation': False,
      'min_speed': False,
      'action_change': False,
      'vehicle_too_close': False,
      'route_deviation_2': False,
      'off_road': False,
      'speed_penalty_factor': 1.0,
      'comfort_penalty_factor': 1.0,
      'ttc_penalty_factor': 1.0,
      'lane_penalty_factor': 1.0,
      'min_speed_factor': 1.0,
      'action_change_factor': 1.0,
      'vehicle_too_close_factor': 1.0,
      'route_deviation_2_factor': 1.0,
      'off_road_factor': 1.0,
      'outside_lanes_factor': 1.0
    }

    if self.config.use_single_reward:
      if agent_too_fast:
        penalty_info['agent_too_fast'] = True
        penalty_info['speed_penalty_factor'] = speed_penalty
        progress_reward = speed_penalty * progress_reward

      if ttc_penalty:
        penalty_info['ttc_penalty'] = True
        penalty_info['ttc_penalty_factor'] = 0.5
        progress_reward = 0.5 * progress_reward

      if comfort_penalty < 1.0:
        penalty_info['comfort_penalty'] = True
        penalty_info['comfort_penalty_factor'] = comfort_penalty
        progress_reward = comfort_penalty * progress_reward
    else:  # nuPlan style reward
      length_factor = 2000
      scale_factor = 100
      r_ttc = 0.0 if ttc_penalty else scale_factor
      r_speed = 0.0 if agent_too_fast else scale_factor
      r_comfort = 0.0 if comfort_penalty < 1.0 else scale_factor

      r_ttc /= length_factor
      r_speed /= length_factor
      r_comfort /= length_factor

      progress_reward = (5 * progress_reward + 5 * r_ttc + 4 * r_speed + 2 * r_comfort) / 16
      
      if ttc_penalty:
        penalty_info['ttc_penalty'] = True
      if agent_too_fast:
        penalty_info['agent_too_fast'] = True
      if comfort_penalty < 1.0:
        penalty_info['comfort_penalty'] = True

    if self.config.use_perc_progress:
      if perc_dist_to_centerline < 1.0:
        penalty_info['lane_deviation'] = True
        penalty_info['lane_penalty_factor'] = perc_dist_to_centerline
      progress_reward = perc_dist_to_centerline * progress_reward

    if self.config.use_min_speed_infraction:
      if fraction_of_speed < 1.0:
        penalty_info['min_speed'] = True
        penalty_info['min_speed_factor'] = fraction_of_speed
      progress_reward = fraction_of_speed * progress_reward

    if action_changed_too_much:
      penalty_info['action_change'] = True
      penalty_info['action_change_factor'] = 0.5
      progress_reward = 0.5 * progress_reward
    
    # ====================================================================
    # ====================================================================
    if soft_penalty_conditions['vehicle_too_close']:
      penalty_info['vehicle_too_close'] = True
      penalty_info['vehicle_too_close_factor'] = 0.5
      progress_reward = 0.5 * progress_reward
    
    if soft_penalty_conditions['route_deviation_2']:
      penalty_info['route_deviation_2'] = True
      penalty_info['route_deviation_2_factor'] = 0.3
      progress_reward = 0.3 * progress_reward
    
    if soft_penalty_conditions['off_road']:
      penalty_info['off_road'] = True
      penalty_info['off_road_factor'] = 0.2
      progress_reward = 0.2 * progress_reward
    
    if soft_penalty_conditions['outside_lanes']:
      penalty_info['outside_lanes'] = True
      penalty_info['outside_lanes_factor'] = 0.5
      progress_reward = 0.5 * progress_reward

    positive_reward_scale = getattr(self.config, 'positive_reward_scale', 1.0)
    progress_reward = progress_reward * positive_reward_scale
    reward = progress_reward + terminal_reward

    if self.config.use_survival_reward:
      reward += positive_reward_scale * self.config.survival_reward_magnitude

    # ====================================================================
    # ====================================================================
    # if -10.0 < reward < -9.9:
    if False:
      print(f"\n{'='*70}")
      print(f"DEBUG: Negative reward detected!")
      print(f"  Final reward: {reward:.4f}")
      print(f"  - progress_reward (before penalties): {current_route_completion - self.last_route_completion:.4f}")
      print(f"  - progress_reward (after all penalties): {progress_reward:.4f}")
      print(f"  - terminal_reward: {terminal_reward:.4f}")
      print(f"  - survival_reward: {self.config.survival_reward_magnitude if self.config.use_survival_reward else 0:.4f}")
      print(f"  Route completion: {self.last_route_completion:.2f}% -> {current_route_completion:.2f}%")
      print(f"  Vehicle speed: {ev_speed:.2f} m/s")
      print(f"  Hard termination: {hard_termination}")
      print(f"  Soft penalties active:")
      print(f"    - outside_lanes: {outside_lanes}")
      print(f"    - vehicle_too_close: {soft_penalty_conditions['vehicle_too_close']}")
      print(f"    - route_deviation_2: {soft_penalty_conditions['route_deviation_2']}")
      print(f"    - off_road: {soft_penalty_conditions['off_road']}")
      print(f"  Penalty factors:")
      for key, value in penalty_info.items():
        if '_factor' in key:
          print(f"    - {key}: {value}")
      print(f"{'='*70}\n")

    wrong_start = False
    if self.first_frame:
      # Unit test to check if the route is faulty
      if self.world_map.get_waypoint(ego_vehicle_location, project_to_road=False) is None:
        wrong_start = True

    if self.first_frame and (termination or truncation or wrong_start):
      # There is a bug in one of the routes where the agent is spawned at a bad position.
      logger.warning('Faulty route file: Map: %s', self.world_map.name)

    self.first_frame = False

    infraction_types = termination_info['infraction_types']
    info = {
      'n_steps': 0,
      'suggest': 0,
      'timeout': timeout,
      'infraction_type': infraction_types[0] if infraction_types else '',
      'infraction_types': infraction_types,
      'soft_penalty_type': '',
      'simple_route_deviation': bool(simple_route_deviation),
      'simple_route_lat_dist': simple_route_lat_dist,
      'simple_terminate_events': termination_info['simple_terminate_events'],
    }
    
    soft_penalty_types = []
    if is_vehicle_too_close:
      soft_penalty_types.append('vehicle_too_close')
    if route_deviation_2:
      soft_penalty_types.append('route_deviation_2')
    if off_road_term:
      soft_penalty_types.append('off_road_term')
    if outside_lanes:
      soft_penalty_types.append('outside_lanes')
    
    if soft_penalty_types:
      info['soft_penalty_type'] = ','.join(soft_penalty_types)
    else:
      info['soft_penalty_type'] = ''
    
    info['simple_reward_RC'] = current_route_completion
    info['simple_reward_penal'] = penalty_info

    return reward, termination, truncation, info

  def compute_comfort_penalty(self):
    '''
    Computes the comfort penalty factor
    '''
    self.remaining_comfort_penalty_ticks = np.clip(self.remaining_comfort_penalty_ticks - 1, a_min=0, a_max=None)

    transform = self.vehicle.get_transform()
    forward_vector = transform.get_forward_vector()
    right_vector = transform.get_right_vector()

    acceleration = self.vehicle.get_acceleration()  # In world coordinates
    acceleration_magnitude = acceleration.length()

    # Project to local coordinates
    longitudinal_acceleration = (acceleration.x * forward_vector.x + acceleration.y * forward_vector.y +
                                 acceleration.z * forward_vector.z)
    lateral_acceleration = (acceleration.x * right_vector.x + acceleration.y * right_vector.y +
                            acceleration.z * right_vector.z)

    yaw = math.radians(transform.rotation.yaw)

    self.ego_state_history.longitudinal_acceleration.append(longitudinal_acceleration)
    self.ego_state_history.lateral_acceleration.append(lateral_acceleration)
    self.ego_state_history.acceleration_magnitude.append(acceleration_magnitude)
    self.ego_state_history.yaw.append(yaw)

    comfort_penalty = 1.0
    
    if len(self.ego_state_history.longitudinal_acceleration) >= 8:
      def make_odd(n):
        return n if n % 2 == 1 else n - 1
      
      wl_accel = make_odd(min(8, len(self.ego_state_history.longitudinal_acceleration)))
      longitudinal_acceleration = savgol_filter(self.ego_state_history.longitudinal_acceleration,
                                                polyorder=2,
                                                window_length=wl_accel,
                                                axis=-1)
      lateral_acceleration = savgol_filter(self.ego_state_history.lateral_acceleration,
                                           polyorder=2,
                                           window_length=wl_accel,
                                           axis=-1)
      acceleration_magnitude = savgol_filter(self.ego_state_history.acceleration_magnitude,
                                             polyorder=2,
                                             window_length=wl_accel,
                                             axis=-1)

      wl_jerk = make_odd(min(15, len(acceleration_magnitude)))
      jerk = savgol_filter(acceleration_magnitude,
                           polyorder=2,
                           deriv=1,
                           delta=self.dx,
                           window_length=wl_jerk,
                           axis=-1)
      longitudinal_jerk = savgol_filter(longitudinal_acceleration,
                                        polyorder=2,
                                        deriv=1,
                                        delta=self.dx,
                                        window_length=wl_jerk,
                                        axis=-1)

      # https://github.com/DanielDauner/tuplan_garage_rl/blob/c6e2a477187c3c124c38bb09d1754e840b8a3222/tuplan_garage/planning/simulation/planner/pdm_planner/scoring/pdm_comfort_metrics_debug.py#L169
      two_pi = 2.0 * np.pi
      yaw_np = np.array(self.ego_state_history.yaw)
      adjustments = np.zeros_like(yaw_np)
      adjustments[1:] = np.cumsum(np.round(np.diff(yaw_np, axis=-1) / two_pi), axis=-1)
      unwrapped_yaw = yaw_np - two_pi * adjustments
      wl_yaw = make_odd(min(5, len(unwrapped_yaw)))
      yaw_rate = savgol_filter(unwrapped_yaw,
                               polyorder=2,
                               deriv=1,
                               delta=self.dx,
                               window_length=wl_yaw,
                               axis=-1)
      yaw_acceleration = savgol_filter(unwrapped_yaw,
                                       polyorder=3,
                                       deriv=2,
                                       delta=self.dx,
                                       window_length=wl_yaw,
                                       axis=-1)

      uncomfortable_acc_lon = ((longitudinal_acceleration > self.config.max_lon_accel)[-1] or
                               (longitudinal_acceleration < self.config.min_lon_accel)[-1])
      uncomfortable_acc_lat = (np.abs(lateral_acceleration) > self.config.max_abs_lat_accel)[-1]
      uncomfortable_jerk = (np.abs(jerk) > self.config.max_abs_mag_jerk)[-1]
      uncomfortable_jerk_lon = (np.abs(longitudinal_jerk) > self.config.max_abs_lon_jerk)[-1]
      uncomfortable_yaw_rate = (np.abs(yaw_rate) > self.config.max_abs_yaw_rate)[-1]
      uncomfortable_yaw_acceleration = (np.abs(yaw_acceleration) > self.config.max_abs_yaw_accel)[-1]

      if uncomfortable_acc_lon:
        self.remaining_comfort_penalty_ticks[0] = self.config.comfort_penalty_ticks
      if uncomfortable_acc_lat:
        self.remaining_comfort_penalty_ticks[1] = self.config.comfort_penalty_ticks
      if uncomfortable_jerk:
        self.remaining_comfort_penalty_ticks[2] = self.config.comfort_penalty_ticks
      if uncomfortable_jerk_lon:
        self.remaining_comfort_penalty_ticks[3] = self.config.comfort_penalty_ticks
      if uncomfortable_yaw_rate:
        self.remaining_comfort_penalty_ticks[4] = self.config.comfort_penalty_ticks
      if uncomfortable_yaw_acceleration:
        self.remaining_comfort_penalty_ticks[5] = self.config.comfort_penalty_ticks

      num_infractions = np.sum(self.remaining_comfort_penalty_ticks.astype(bool))
      comfort_penalty = 1.0 - self.config.comfort_penalty_factor * (num_infractions / 6.0)

    return comfort_penalty

  def destroy(self):
    self.collision_detector.clean()

  def get_closest_route_point(self, waypoint_route):
    '''
    :return: The ego agents position in the coordinate system of the closest route point.
    '''
    ego_vehicle_transform = self.vehicle.get_transform()
    pos = ego_vehicle_transform.location
    pos = np.array([pos.x, pos.y])

    if not waypoint_route:
      return np.array([0.0, 0.0, 0.0], dtype=np.float32)

    if len(waypoint_route) > 1:
      close_point_global = np.array([waypoint_route[0][0].location.x, waypoint_route[0][0].location.y])
      next_point_global = np.array([waypoint_route[1][0].location.x, waypoint_route[1][0].location.y])
      distance = next_point_global - close_point_global

      # Compute orientation of route.
      if np.linalg.norm(distance) < 0.1:
        # For cases where the points are too close to each other the orientation vector may be too random.
        # We use the orientation of the waypoint itself instead which usually also points in the direction of the route.
        yaw_route = waypoint_route[0][0].rotation.yaw
      else:
        route_vector = distance
        yaw_route = np.rad2deg(np.arctan2(route_vector[1], route_vector[0]))
    else:
      close_point_global = np.array([waypoint_route[0][0].location.x, waypoint_route[0][0].location.y])
      # No next point, so we use the orientation of the route waypoint as direction of the route.
      yaw_route = waypoint_route[0][0].rotation.yaw

    ego_in_route_coordinate = rl_u.inverse_conversion_2d(pos, close_point_global, np.deg2rad(yaw_route))
    ego_in_route_yaw = rl_u.normalize_angle_degree(ego_vehicle_transform.rotation.yaw - yaw_route)
    ego_in_route_coordinate = np.append(ego_in_route_coordinate, ego_in_route_yaw)

    return ego_in_route_coordinate

  def is_ego_too_slow(self, ego_speed, other_vehicles):
    '''
     Compares the speed of the ego vehicle with the avg speed of surrounding background traffic.
    '''
    background_vehicles = [v for v in other_vehicles if v.attributes['role_name'] == 'background']

    frame_mean_speed = 0.000000000000001
    if len(background_vehicles) > 0:
      for vehicle in background_vehicles:
        frame_mean_speed += CarlaDataProvider.get_velocity(vehicle)
      frame_mean_speed /= len(background_vehicles)

    fraction_of_speed = np.clip(ego_speed / frame_mean_speed, a_min=0.0, a_max=1.0)
    return fraction_of_speed

  def vehicle_too_close(self, ego_vehicle_transform, speed, ego_control, vehicles_all, walkers_all):
    '''
    Forecasts ego vehicle with WOR kinematic bicycle model. If forcast overlaps with another vehicles current state
    the ego vehicle is considered to be too close to the preceding vehicle. During the forecast brake and throttle
    are considered to be 0 whereas steering is repeated. The speed of the vehicle is clipped to a minimum amount.
    :return: True or False depending on whether the ego vehicle is too close to another traffic participant.
    '''
    next_loc = np.array([ego_vehicle_transform.location.x, ego_vehicle_transform.location.y])

    next_yaw = np.array([np.deg2rad(ego_vehicle_transform.rotation.yaw)])
    ego_action = np.array([ego_control.steer, 0.0, 0.0])

    next_speed = np.array([speed if speed > self.config.ego_forecast_min_speed else self.config.ego_forecast_min_speed])

    ego_bounding_boxes = []

    for _ in range(int(self.config.ego_forecast_time / self.config.time_interval)):
      next_loc, next_yaw, next_speed = self.ego_model.forward(next_loc, next_yaw, next_speed, ego_action)

      delta_yaws = np.rad2deg(next_yaw).item()

      transform = carla.Transform(
          carla.Location(x=next_loc[0].item(), y=next_loc[1].item(), z=ego_vehicle_transform.location.z),
          carla.Rotation(pitch=ego_vehicle_transform.rotation.pitch,
                         yaw=delta_yaws,
                         roll=ego_vehicle_transform.rotation.roll))

      bounding_box = carla.BoundingBox(transform.location, self.vehicle.bounding_box.extent)
      bounding_box.rotation = transform.rotation

      ego_bounding_boxes.append(bounding_box)

    for ego_bounding_box in ego_bounding_boxes:
      for actor in [*vehicles_all, *walkers_all]:
        if actor.id == self.vehicle.id:
          continue

        traffic_transform = actor.get_transform()
        traffic_bb_center = traffic_transform.transform(actor.bounding_box.location)

        if ego_vehicle_transform.location.distance(traffic_bb_center) < 10.0:
          traffic_bounding_box = carla.BoundingBox(traffic_bb_center, actor.bounding_box.extent)
          traffic_bounding_box.rotation = carla.Rotation(
              pitch=rl_u.normalize_angle_degree(actor.bounding_box.rotation.pitch + traffic_transform.rotation.pitch),
              yaw=rl_u.normalize_angle_degree(actor.bounding_box.rotation.yaw + traffic_transform.rotation.yaw),
              roll=rl_u.normalize_angle_degree(actor.bounding_box.rotation.roll + traffic_transform.rotation.roll))

          # check the first BB of the traffic participant. We don't extrapolate into the future here.
          if rl_u.check_obb_intersection(ego_bounding_box, traffic_bounding_box):
            #                           rotation=ego_bounding_box.rotation,
            #                           thickness=0.3,
            #                           color=carla.Color(255, 0, 0, 255),
            #                           life_time=self.config.time_interval + 0.01)
            return True

      # color = carla.Color(0, 255, 0, 255)
      #                           rotation=ego_bounding_box.rotation,
      #                           thickness=0.3,
      #                           color=color,
      #                           life_time=self.config.time_interval + 0.01)

    return False

  def does_agent_violate_ttc(self, ego_vehicle_transform, ev_speed, vehicles_all, walkers_all, static_all):
    stopped_speed_threshold: float = 0.1  # [m/s] (ttc)

    if (len(vehicles_all) <= 0 and len(walkers_all) <= 0) or ev_speed < stopped_speed_threshold:
      return False

    distance = carla.Vector3D(self.vehicle.bounding_box.extent.x, self.vehicle.bounding_box.extent.y).length()
    ego_transforms = []
    ego_bbs = []

    for delta in self.ttc_future_time_deltas:
      next_loc = ego_vehicle_transform.transform(carla.Location(x=1.0 * delta * ev_speed, y=0.0, z=0.0))

      next_transform = carla.Transform(next_loc, ego_vehicle_transform.rotation)

      ego_bounding_box = carla.BoundingBox(next_transform.location, self.vehicle.bounding_box.extent)
      ego_bounding_box.rotation = next_transform.rotation
      ego_transforms.append(next_transform)
      ego_bbs.append(ego_bounding_box)

      #                           color=carla.Color(255, 0, 0, 255),
      #                           life_time=self.config.time_interval + 0.01)

    for actor in [*vehicles_all, *walkers_all, *static_all]:
      if actor.id == self.vehicle.id:
        continue

      actor_distance = carla.Vector3D(actor.bounding_box.extent.x, actor.bounding_box.extent.y).length()
      actor_transform = actor.get_transform()
      actor_speed = actor.get_velocity().length()
      no_collision_distance = distance + actor_distance

      for idx, delta in enumerate(self.ttc_future_time_deltas):
        actor_next_loc = actor_transform.transform(carla.Location(x=delta * actor_speed, y=0.0, z=0.0))

        # Filter cases that are clearly no collision, using enclosing circles.
        if actor_next_loc.distance(ego_transforms[idx].location) > no_collision_distance:
          continue

        bounding_box = carla.BoundingBox(actor_next_loc, actor.bounding_box.extent)
        bounding_box.rotation = actor_transform.rotation
        #                           color=carla.Color(255, 255, 0, 255),
        #                           life_time=self.config.time_interval + 0.01)

        if rl_u.check_obb_intersection(ego_bbs[idx], bounding_box):
          return True

    return False


class EgoModel():
  """
    Kinematic bicycle model describing the motion of a car given it's state and
    action. Tuned parameters are taken from World on Rails.
    """

  def __init__(self, dt=1. / 4):
    self.dt = dt

    # Kinematic bicycle model. Numbers are the tuned parameters from World
    # on Rails
    self.front_wb = -0.090769015
    self.rear_wb = 1.4178275

    self.steer_gain = 0.36848336
    self.brake_accel = -4.952399
    self.throt_accel = 0.5633837

  def forward(self, locs, yaws, spds, acts):
    # Kinematic bicycle model. Numbers are the tuned parameters from World on Rails
    steer = acts[..., 0:1].item()
    throt = acts[..., 1:2].item()
    brake = acts[..., 2:3].astype(np.uint8)

    if brake:
      accel = self.brake_accel
    else:
      accel = self.throt_accel * throt

    wheel = self.steer_gain * steer

    beta = math.atan(self.rear_wb / (self.front_wb + self.rear_wb) * math.tan(wheel))
    yaws = yaws.item()
    spds = spds.item()
    next_locs_0 = locs[0].item() + spds * math.cos(yaws + beta) * self.dt
    next_locs_1 = locs[1].item() + spds * math.sin(yaws + beta) * self.dt
    next_yaws = yaws + spds / self.rear_wb * math.sin(beta) * self.dt
    next_spds = spds + accel * self.dt
    next_spds = next_spds * (next_spds > 0.0)  # Fast ReLU

    next_locs = np.array([next_locs_0, next_locs_1])
    next_yaws = np.array(next_yaws)
    next_spds = np.array(next_spds)

    return next_locs, next_yaws, next_spds


class SimpleRewardConfig:
  def __init__(self):
    for key, value in REWARD_CONFIG.items():
      setattr(self, key, value)


class SimpleRewardGenerator:
  
  def __init__(self, config: dict, scenario_name: str = "leaderboardv2"):
    self.full_config = config
    self.scenario_name = scenario_name
    
    reward_config = config.get('reward', {})
    extra_reward_keys = sorted(set(reward_config.keys()) - {'type'})
    if extra_reward_keys:
      logger.warning(
        "Ignoring simple reward YAML parameters; edit REWARD_CONFIG in "
        "simple_reward_handler.py instead. Unsupported keys: %s",
        extra_reward_keys
      )
    self.reward_config = SimpleRewardConfig()
    
    self._simple_reward = None
    self._initialized = False
    self._perc_off_road_warned = False
    
    self._map_loader = None
    self._init_map_loader()
  
  def _init_map_loader(self):
    self._map_data = {
      'road': None,
      'world_offset': None,
      'pixels_per_meter': None,
      'town_name': None,
      'loaded': False
    }
    
    obs_config = self.full_config.get('observation_space', {})
    bev_config = obs_config.get('vector', obs_config.get('birdview', {}))
    self._map_dir = bev_config.get('map_dir', 'resources/maps')
    self._pixels_per_meter = bev_config.get('pixels_per_meter', 5.0)
    self._bev_width_meters = bev_config.get('mask_width', 32)
    self._bev_width_pixels = int(self._bev_width_meters * self._pixels_per_meter)
    self._ego_to_bottom_meters = bev_config.get('ego_to_bottom', self._bev_width_meters / 2)
    self._pixels_ev_to_bottom = int(self._ego_to_bottom_meters * self._pixels_per_meter)
  
  def _load_map_data(self):
    import os
    
    world = CarlaDataProvider.get_world()
    if world is None:
      return
    
    town_name = world.get_map().name.split('/')[-1]
    if self._map_data['loaded'] and self._map_data.get('town_name') == town_name:
      return

    self._map_data.update({
      'road': None,
      'world_offset': None,
      'pixels_per_meter': None,
      'town_name': town_name,
      'loaded': False
    })

    maps_h5_path = os.path.join(self._map_dir, f'{town_name}.h5')
    
    if not os.path.exists(maps_h5_path):
      if not self._perc_off_road_warned:
        logger.warning("Map file not found: %s, perc_off_road disabled", maps_h5_path)
        self._perc_off_road_warned = True
      return
    
    try:
      import h5py

      with h5py.File(maps_h5_path, 'r', libver='latest', swmr=True) as hf:
        self._map_data['road'] = np.array(hf['road'], dtype=np.uint8).swapaxes(0, 1)
        self._map_data['world_offset'] = np.array(hf.attrs['world_offset_in_meters'], dtype=np.float32)
        self._map_data['pixels_per_meter'] = float(hf.attrs['pixels_per_meter'])
        self._map_data['town_name'] = town_name
        
        if not np.isclose(self._pixels_per_meter, self._map_data['pixels_per_meter']):
          logger.warning("pixels_per_meter mismatch: config=%s, map=%s", self._pixels_per_meter, self._map_data['pixels_per_meter'])
      
      self._map_data['loaded'] = True
    except Exception as e:
      if not self._perc_off_road_warned:
        logger.warning("Failed to load map data: %s, perc_off_road disabled", e)
        self._perc_off_road_warned = True
  
  def reset(self):
    
    vehicle = CarlaDataProvider._ego_actor
    world = CarlaDataProvider.get_world()
    world_map = CarlaDataProvider.get_map()
    route = CarlaDataProvider._ego_vehicle_route
    
    if vehicle is None or world is None or world_map is None:
      raise RuntimeError(
        "SimpleRewardGenerator.reset() called before environment is ready. "
        "Ensure CarlaDataProvider has valid ego_actor, world, and map."
      )
    
    self._load_map_data()
    
    if self._simple_reward is not None:
      try:
        self._simple_reward.destroy()
      except Exception:
        pass
    
    waypoint_route = route if route else []
    
    self._simple_reward = SimpleReward(
      vehicle=vehicle,
      world_map=world_map,
      world=world,
      config=self.reward_config,
      route=waypoint_route
    )
    self._initialized = True
  
  def generate_reward(
      self,
      observation,
      done: bool,
      info: dict,
      action,
      name: str = "leaderboardv2",
      crash_message: str = ""
  ):
    if info is None:
      info = {}

    if crash_message and str(crash_message).startswith("Simulation crashed"):
      logger.warning('Simulation Crashed, Reward Is Set As 0.0')
      return np.array(0.0, dtype=np.float32), info, False
    
    if not self._initialized or self._simple_reward is None:
      logger.warning("SimpleRewardGenerator not initialized, returning 0 reward")
      return np.array(0.0, dtype=np.float32), info, False
    
    world = CarlaDataProvider.get_world()
    timestamp = GameTime.get_time()
    route = CarlaDataProvider._ego_vehicle_route
    
    waypoint_route = route if route else []
    
    if world is not None:
      actors = world.get_actors()
      vehicles_all = actors.filter('*vehicle*')
      walkers_all = actors.filter('*walker*')
      static_all = actors.filter('*static*')
    else:
      vehicles_all = []
      walkers_all = []
      static_all = []
    
    perc_off_road = self._compute_perc_off_road(observation)
    
    reward, termination, truncation, reward_info = self._simple_reward.get(
      timestamp=timestamp,
      waypoint_route=waypoint_route,
      collision_with_pedestrian=None,
      vehicles_all=vehicles_all,
      walkers_all=walkers_all,
      static_all=static_all,
      perc_off_road=perc_off_road
    )
    
    info.update(reward_info)
    merge_simple_terminate_events(info)
    
    # These keys are consumed only by RewardWrapper's fallback override
    # (analysis section 2.2). They live under the wrapper/ namespace to make
    # their inter-wrapper-only nature explicit.
    if termination:
      info[WRAPPER_REWARD_SIMPLE_TERMINATED] = True
      info[WRAPPER_TERM_TRIGGERED] = True
    if truncation:
      info[WRAPPER_REWARD_SIMPLE_TRUNCATED] = True
    
    return np.array(reward, dtype=np.float32), info, False
  
  def destroy(self):
    if self._simple_reward is not None:
      try:
        self._simple_reward.destroy()
      except Exception:
        pass
      self._simple_reward = None
    self._initialized = False
  
  def _world_to_pixel(self, location):
    x = self._map_data['pixels_per_meter'] * (location.x - self._map_data['world_offset'][0])
    y = self._map_data['pixels_per_meter'] * (location.y - self._map_data['world_offset'][1])
    return np.array([x, y], dtype=np.float32)
  
  def _get_warp_transform(self, ev_loc, ev_rot):
    import cv2 as cv
    
    ev_loc_in_px = self._world_to_pixel(ev_loc)
    yaw = np.deg2rad(ev_rot.yaw)
    
    forward_vec = np.array([np.cos(yaw), np.sin(yaw)])
    right_vec = np.array([np.cos(yaw + 0.5*np.pi), np.sin(yaw + 0.5*np.pi)])
    
    bottom_left = ev_loc_in_px - self._pixels_ev_to_bottom * forward_vec - (0.5*self._bev_width_pixels) * right_vec
    top_left = ev_loc_in_px + (self._bev_width_pixels - self._pixels_ev_to_bottom) * forward_vec - (0.5*self._bev_width_pixels) * right_vec
    top_right = ev_loc_in_px + (self._bev_width_pixels - self._pixels_ev_to_bottom) * forward_vec + (0.5*self._bev_width_pixels) * right_vec
    
    src_pts = np.stack((bottom_left, top_left, top_right), axis=0).astype(np.float32)
    dst_pts = np.array([[0, self._bev_width_pixels-1],
                        [0, 0],
                        [self._bev_width_pixels-1, 0]], dtype=np.float32)
    return cv.getAffineTransform(src_pts, dst_pts)
  
  def _get_ego_mask(self, ev_transform, ev_bbox, M_warp):
    import cv2 as cv
    import carla
    
    mask = np.zeros([self._bev_width_pixels, self._bev_width_pixels], dtype=np.uint8)
    
    bb_loc = ev_bbox.location
    bb_ext = ev_bbox.extent
    
    corners = [carla.Location(x=-bb_ext.x, y=-bb_ext.y),
               carla.Location(x=bb_ext.x, y=-bb_ext.y),
               carla.Location(x=bb_ext.x, y=0),
               carla.Location(x=bb_ext.x, y=bb_ext.y),
               carla.Location(x=-bb_ext.x, y=bb_ext.y)]
    corners = [bb_loc + corner for corner in corners]
    corners = [ev_transform.transform(corner) for corner in corners]
    corners_in_pixel = np.array([[self._world_to_pixel(corner)] for corner in corners])
    corners_warped = cv.transform(corners_in_pixel, M_warp)
    
    cv.fillConvexPoly(mask, np.round(corners_warped).astype(np.int32), 1)
    return mask.astype(bool)
  
  def _compute_perc_off_road(self, observation):
    """
    """
    if not self._map_data['loaded']:
      if not self._perc_off_road_warned:
        logger.warning("Map data not loaded, perc_off_road disabled (set to 0.0)")
        self._perc_off_road_warned = True
      return 0.0
    
    try:
      import cv2 as cv

      ego_actor = CarlaDataProvider._ego_actor
      if ego_actor is None:
        return 0.0
      
      ev_transform = ego_actor.get_transform()
      ev_loc = ev_transform.location
      ev_rot = ev_transform.rotation
      ev_bbox = ego_actor.bounding_box
      
      M_warp = self._get_warp_transform(ev_loc, ev_rot)
      
      road_map = self._map_data['road']
      road_mask = cv.warpAffine(
        road_map, M_warp, (self._bev_width_pixels, self._bev_width_pixels)
      ).astype(bool)
      
      ego_mask = self._get_ego_mask(ev_transform, ev_bbox, M_warp)
      
      ego_sum = np.sum(ego_mask)
      if ego_sum == 0:
        return 0.0
      
      off_road_area = np.sum(ego_mask & np.logical_not(road_mask))
      perc_off_road = off_road_area / ego_sum
      
      return float(perc_off_road)
    
    except Exception as e:
      if not self._perc_off_road_warned:
        logger.warning("failed to compute perc_off_road (%s), disabled (set to 0.0)", e)
        self._perc_off_road_warned = True
      return 0.0
  
  def get_off_road_visualization(self):
    """
    
    Returns:
        dict: {
          'road_mask': np.ndarray (H, W) bool,
          'ego_mask': np.ndarray (H, W) bool,
          'ego_on_road': np.ndarray (H, W) bool,
          'ego_off_road': np.ndarray (H, W) bool,
          'perc_off_road': float
        }
    """
    if not self._map_data['loaded']:
      return None
    
    try:
      import cv2 as cv

      ego_actor = CarlaDataProvider._ego_actor
      if ego_actor is None:
        return None
      
      ev_transform = ego_actor.get_transform()
      ev_loc = ev_transform.location
      ev_rot = ev_transform.rotation
      ev_bbox = ego_actor.bounding_box
      
      M_warp = self._get_warp_transform(ev_loc, ev_rot)
      
      road_map = self._map_data['road']
      road_mask = cv.warpAffine(
        road_map, M_warp, (self._bev_width_pixels, self._bev_width_pixels)
      ).astype(bool)
      
      ego_mask = self._get_ego_mask(ev_transform, ev_bbox, M_warp)
      
      ego_sum = np.sum(ego_mask)
      if ego_sum == 0:
        perc_off_road = 0.0
      else:
        off_road_area = np.sum(ego_mask & np.logical_not(road_mask))
        perc_off_road = off_road_area / ego_sum
      
      return {
        'road_mask': road_mask,
        'ego_mask': ego_mask,
        'ego_on_road': ego_mask & road_mask,
        'ego_off_road': ego_mask & np.logical_not(road_mask),
        'perc_off_road': perc_off_road
      }
    
    except Exception as e:
      return None
