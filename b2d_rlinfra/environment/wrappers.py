"""Gym-style wrapper stack composed around ``CARLAEnv``.

    RoutePlanWrapper        - wires the dense global route into the env.
    ObservationWrapper      - builds the configurable obs (BEV + scalars).
    EventTerminationWrapper - maps Leaderboard traffic events to ``terminated``.
    RewardWrapper           - reward shaping (simple / PPO/A2C/SAC/TD3 variants).
    ActionWrapper           - discrete / continuous / trajectory actions.
    LQRExpertWrapper        - adds a rule-based expert action for BC warmup.
"""
import logging
import math
import carla
import numpy as np
from srunner.scenariomanager.traffic_events import TrafficEventType
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.timer import GameTime
from b2d_rlinfra.environment.handlers.birdview_obs_handler import BirdViewObsManager as BirdviewObsHandler
from b2d_rlinfra.environment.handlers.ego_sensor_obs_handler import EGO_SENSOR_HANDLERS, apply_observation_presets
from b2d_rlinfra.environment.handlers.minddrive_state_obs_handler import MindDriveStateObsHandler
from b2d_rlinfra.environment.handlers.rgb_sensor_obs_handler import RGBSensorObsHandler, get_rgb_config
from b2d_rlinfra.environment.handlers.scalars_obs_handler import ScalarObsHandler
from b2d_rlinfra.environment.handlers.drivepi0_state_obs_handler import DrivePi0StateObsHandler
from b2d_rlinfra.environment.handlers.reward_handler_ppo import RewardHandler as PPORewardGenerator
from b2d_rlinfra.environment.handlers.reward_handler_a2c import RewardHandler as A2CRewardGenerator
from b2d_rlinfra.environment.handlers.reward_handler_sac import RewardHandler as SACRewardGenerator
from b2d_rlinfra.environment.handlers.reward_handler_td3 import RewardHandler as TD3RewardGenerator
from b2d_rlinfra.environment.handlers.reward_handler_minddrive import RewardHandler as MindDriveSparseRewardGenerator
from b2d_rlinfra.environment.handlers.simple_reward_handler import SimpleRewardGenerator
from b2d_rlinfra.environment.handlers.termination_handler_ppo import TerminationHandler as PPOTerminationHandler
from b2d_rlinfra.environment.handlers.termination_handler_a2c import TerminationHandler as A2CTerminationHandler
from b2d_rlinfra.environment.handlers.termination_handler_sac import TerminationHandler as SACTerminationHandler
from b2d_rlinfra.environment.handlers.termination_handler_td3 import TerminationHandler as TD3TerminationHandler
from b2d_rlinfra.environment.spaces import (
    build_action_space as _build_action_space_from_yaml,
    build_observation_space_dict as _build_observation_space_dict_from_yaml,
    signed_pedal_to_throttle_brake,
)
from b2d_rlinfra.environment.info_keys import (
    WRAPPER_OBS_EMERGENCY_IN_VISION,
    WRAPPER_REWARD_BLOCKED_BY_OBSTACLES,
    WRAPPER_REWARD_PARKING_EXIT_DEVIATION_EXEMPT,
    WRAPPER_REWARD_SIMPLE_TERMINATED,
    WRAPPER_REWARD_SIMPLE_TRUNCATED,
    WRAPPER_ROUTE_DISTANCE_ON_TICK,
    WRAPPER_ROUTE_DISTANCE_ON_TICK_REWARD,
    WRAPPER_TERM_TRIGGERED,
)

REWARD_HANDLER_REGISTRY = {
    'simple': SimpleRewardGenerator,
    'ppo': PPORewardGenerator,
    'a2c': A2CRewardGenerator,
    'sac': SACRewardGenerator,
    'td3': TD3RewardGenerator,
    'minddrive_sparse': MindDriveSparseRewardGenerator,
}

TERMINATION_HANDLER_REGISTRY = {
    'ppo': PPOTerminationHandler,
    'a2c': A2CTerminationHandler,
    'sac': SACTerminationHandler,
    'td3': TD3TerminationHandler,
}

logger = logging.getLogger("Interface Wrapper")

__layer__ = (2, "Environment")

# Traffic events that may trigger early termination.
TERMINATE_LIST = [
    TrafficEventType.COLLISION_PEDESTRIAN,
    TrafficEventType.COLLISION_VEHICLE,
    TrafficEventType.COLLISION_STATIC,
    TrafficEventType.TRAFFIC_LIGHT_INFRACTION,
    TrafficEventType.STOP_INFRACTION,
    TrafficEventType.VEHICLE_BLOCKED,
    TrafficEventType.ROUTE_DEVIATION,
    TrafficEventType.ROUTE_COMPLETION,
]

# Mapping from Leaderboard event names to TrafficEventType enums.
LEADERBOARD_EVENT_MAP = {
    'COLLISION_PEDESTRIAN': TrafficEventType.COLLISION_PEDESTRIAN,
    'COLLISION_VEHICLE': TrafficEventType.COLLISION_VEHICLE,
    'COLLISION_STATIC': TrafficEventType.COLLISION_STATIC,
    'TRAFFIC_LIGHT_INFRACTION': TrafficEventType.TRAFFIC_LIGHT_INFRACTION,
    'STOP_INFRACTION': TrafficEventType.STOP_INFRACTION,
    'VEHICLE_BLOCKED': TrafficEventType.VEHICLE_BLOCKED,
    'ROUTE_DEVIATION': TrafficEventType.ROUTE_DEVIATION,
    'ROUTE_COMPLETION': TrafficEventType.ROUTE_COMPLETION,
}

# Custom termination events emitted by algorithm-specific termination handlers
# and consumed by algorithm-specific reward handlers.
CUSTOM_TERMINATE_EVENTS = {
    'USELESS_MOVE_TIMEOUT',   # ineffective movement (e.g. driving in circles)
    'NOT_IN_CARLANE',         # left the drivable road
    'SECOND_RED_SAME_LIGHT',
    'SUCCESS',                # reached the destination
}

# Merged event map used for config validation.
TERMINATE_EVENT_MAP = {**LEADERBOARD_EVENT_MAP}
TERMINATE_EVENT_MAP.update({name: name for name in CUSTOM_TERMINATE_EVENTS})

class ActionWrapper():
    def __init__(self, env):
        self.env = env
        self.config = self.env.config
        # Share the canonical space with the inner environment.
        self.action_space = self.make_action_space()
        self.observation_space = self.env.observation_space
        self.env.action_space = self.action_space

        action_cfg = self.env.config["action_space"]
        self.action_repeat = max(1, int(action_cfg.get("action_repeat", 1)))
        if action_cfg["type"] == "trajectory":
            from b2d_rlinfra.environment.controllers import TrajectoryController
            self.controller = TrajectoryController(action_cfg)
    
    def __getattr__(self, name):
        """Forward unknown attributes to the underlying env."""
        if name == 'env':
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        return getattr(self.env, name)

    def make_action_space(self):
        return _build_action_space_from_yaml(self.env.config)

    def step(self, action):
        """Apply ``action`` and forward the env step result.

        Returns:
            ``(observation, reward, terminated, truncated, info)``.
        """
        action_type = self.env.config['action_space']['type']
        if action_type == 'discrete':
            actions_list = self.env.config['action_space']['discrete_actions_list']
            action_idx = int(action) if hasattr(action, 'item') else action
            action = actions_list[action_idx]
            carla_action = carla.VehicleControl(
                throttle=float(action[0]), 
                steer=float(action[1]), 
                brake=float(action[2])
            )
        elif action_type in (
            'continuous_signed_pedal_steer',
            'continuous_accelerate_steering_rate',
        ):
            signed_pedal = float(action[0])
            steer = float(action[1])
            throttle, brake = signed_pedal_to_throttle_brake(signed_pedal)
            carla_action = carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)
        elif action_type in ('continuous_accelerate_steering_brake', 'continuous_throttle_steer_brake'):
            carla_action = carla.VehicleControl(
                throttle=float(action[0]), 
                steer=float(action[1]), 
                brake=float(action[2])
            )
        elif action_type == 'trajectory':
            traj_cfg = self.env.config['action_space'].get('trajectory', {})
            num_points = int(traj_cfg.get('num_points', 4))
            state_dim = int(traj_cfg.get('state_dim', 3))
            trajectory = np.array(action, dtype=np.float64).reshape(num_points, state_dim)

            ego_actor = CarlaDataProvider._ego_actor
            ego_speed = 0.0
            if ego_actor is not None and ego_actor.is_alive:
                velocity = ego_actor.get_velocity()
                ego_speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)

            throttle, steer, brake = self.controller.step(trajectory, ego_speed)
            carla_action = carla.VehicleControl(
                throttle=float(throttle),
                steer=float(steer),
                brake=float(brake),
            )
        else:
            raise ValueError(f"Invalid action space type: {action_type}")
        obs, reward, terminated, truncated, info = {}, 0.0, False, False, {}
        total_reward = 0.0
        for _ in range(self.action_repeat):
            obs, reward, terminated, truncated, info = self.env.step(carla_action)
            total_reward += float(reward)
            if terminated or truncated:
                break
        reward = total_reward

        if info is None:
            info = {}
        elif not isinstance(info, dict):
            return obs, reward, terminated, truncated, info
        else:
            info = dict(info)

        info['action_type'] = action_type
        info['action_repeat'] = self.action_repeat
        info['executed_control'] = np.array(
            [carla_action.throttle, carla_action.steer, carla_action.brake],
            dtype=np.float32,
        )
        if action_type == 'trajectory':
            info['trajectory_num_points'] = num_points
            info['trajectory_state_dim'] = state_dim

        return obs, reward, terminated, truncated, info
    
    def reset(self, seed=None, options=None):
        if hasattr(self, 'controller'):
            self.controller.reset()
        return self.env.reset(seed=seed, options=options)
    
    def render(self, mode='human'):
        return self.env.render(mode=mode)

class ObservationWrapper():
    def __init__(self, env):
        self.env = env
        self.config = self.env.config
        # Build handlers and share their canonical space with the inner env.
        self.observation_space = self.make_observation_space()
        self.action_space = self.env.action_space
        self.env.observation_space = self.observation_space
    
    def __getattr__(self, name):
        """Forward unknown attributes to the wrapped environment."""
        if name == 'env':
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        return getattr(self.env, name)
    
    def make_observation_space(self):
        observation_space = _build_observation_space_dict_from_yaml(self.env.config)

        obs_cfg = apply_observation_presets(self.env.config["observation_space"])
        self.obs_handlers = {}

        vector_cfg = obs_cfg.get("vector", {})
        if vector_cfg.get("enable", False):
            self.obs_handlers[vector_cfg["type"]] = BirdviewObsHandler(vector_cfg)

        # Visualization-only BEV is returned in info, not observation_space.
        vis_bev_cfg = obs_cfg.get("vis_bev", {})
        if (
            vis_bev_cfg.get("enable", False)
            and vis_bev_cfg.get("type") == "bev_image"
        ):
            self.obs_handlers["vis_bev"] = BirdviewObsHandler(vis_bev_cfg)

        scalars_cfg = obs_cfg.get("scalars", {})
        if scalars_cfg:
            self.obs_handlers["scalars"] = ScalarObsHandler(scalars_cfg)

        for sensor_key, handler_cls in EGO_SENSOR_HANDLERS.items():
            sensor_cfg = obs_cfg.get(sensor_key, {}) or {}
            if sensor_cfg.get("enable", False):
                handler = handler_cls(sensor_cfg)
                handler_key = (
                    handler.output_key
                    if getattr(handler, "emit_output", True)
                    else f"__{sensor_key}_{handler.sensor_id}"
                )
                self.obs_handlers[handler_key] = handler

        rgb_cfg = get_rgb_config(obs_cfg)
        if rgb_cfg.get("enable", False):
            rgb_cfg = dict(rgb_cfg)
            env_index = getattr(self.env, "env_index", None)
            if env_index is not None:
                rgb_cfg.setdefault("log_prefix", f"Worker {env_index} RGB")
            handler = RGBSensorObsHandler(rgb_cfg)
            if handler.emit_output:
                self.obs_handlers[handler.output_key] = handler
            if handler.temporal_output_key:
                self.obs_handlers[str(handler.temporal_output_key)] = handler

        drivepi0_state_cfg = obs_cfg.get("drivepi0_state", {})
        if drivepi0_state_cfg.get("enable", False):
            handler = DrivePi0StateObsHandler(drivepi0_state_cfg)
            self.obs_handlers[handler.output_key] = handler

        minddrive_state_cfg = obs_cfg.get("minddrive_state", {})
        if minddrive_state_cfg.get("enable", False):
            handler = MindDriveStateObsHandler(minddrive_state_cfg)
            self.obs_handlers[handler.output_key] = handler

        return observation_space

    def step(self, action):
        """
        Execute one action and return the configured observation.

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        _, reward, terminated, truncated, info = self.env.step(action)
        observation = {}

        seen_handlers = set()
        for key, handler in self.obs_handlers.items():
            if key == "vis_bev":
                continue
            handler_id = id(handler)
            if handler_id in seen_handlers:
                continue
            seen_handlers.add(handler_id)
            observation.update(handler.get_observation())

        if 'vector' not in observation:
            if 'bev_mask' in observation:
                observation['vector'] = observation['bev_mask']
            elif 'bev_image' in observation:
                observation['vector'] = observation['bev_image']

        if "vis_bev" in self.obs_handlers:
            vis_result = self.obs_handlers["vis_bev"].get_observation()
            bev_img = vis_result.get("bev_image")
            if bev_img is not None:
                info["bev_image"] = bev_img
        
        ego_actor = CarlaDataProvider._ego_actor
        if ego_actor is not None and ego_actor.is_alive:
            velocity = ego_actor.get_velocity()
            speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
            control = ego_actor.get_control()
            info['speed'] = float(speed)
            info['steer'] = float(control.steer)
            info['throttle'] = float(control.throttle)
            info['brake'] = float(control.brake)
        
        if 'emergency_vehicles_in_vision' in observation:
            info[WRAPPER_OBS_EMERGENCY_IN_VISION] = observation['emergency_vehicles_in_vision']
        else:
            info[WRAPPER_OBS_EMERGENCY_IN_VISION] = False
        
        return observation, reward, terminated, truncated, info
    
    def reset(self, seed=None, options=None):
        _, info = self.env.reset(seed=seed, options=options)

        seen_handlers = set()
        for handler in self.obs_handlers.values():
            handler_id = id(handler)
            if handler_id in seen_handlers:
                continue
            seen_handlers.add(handler_id)
            handler.reset()

        warmup_observation = self._reset_warmup_observation()
        if warmup_observation is None and self._needs_observation_frame_tick():
            self._tick_observation_frame()

        observation = warmup_observation if warmup_observation is not None else self._collect_observation_handlers()

        if "vis_bev" in self.obs_handlers:
            vis_result = self.obs_handlers["vis_bev"].get_observation()
            bev_img = vis_result.get("bev_image")
            if bev_img is not None:
                info["bev_image"] = bev_img

        # Keep reset() info shape consistent with step() info shape so that
        # any downstream consumer reading WRAPPER_OBS_EMERGENCY_IN_VISION
        # never sees a silent KeyError.
        info[WRAPPER_OBS_EMERGENCY_IN_VISION] = bool(
            observation.get('emergency_vehicles_in_vision', False)
        )

        return observation, info

    def _collect_observation_handlers(self):
        observation = {}
        seen_handlers = set()
        for key, handler in self.obs_handlers.items():
            if key == "vis_bev":
                continue
            handler_id = id(handler)
            if handler_id in seen_handlers:
                continue
            seen_handlers.add(handler_id)
            observation.update(handler.get_observation())

        if 'vector' not in observation:
            if 'bev_mask' in observation:
                observation['vector'] = observation['bev_mask']
            elif 'bev_image' in observation:
                observation['vector'] = observation['bev_image']
        return observation

    def _reset_warmup_observation(self):
        obs_cfg = self.config.get("observation_space", {}) or {}
        warmup_ticks = int(obs_cfg.get("reset_warmup_ticks", 0) or 0)
        if warmup_ticks <= 0:
            return None

        observation = None
        for _ in range(warmup_ticks):
            if self._needs_observation_frame_tick():
                self._tick_observation_frame()
            observation = self._collect_observation_handlers()
        return observation

    def _needs_observation_frame_tick(self):
        for handler in getattr(self, "obs_handlers", {}).values():
            if isinstance(handler, RGBSensorObsHandler):
                return True
            if any(isinstance(handler, cls) for cls in EGO_SENSOR_HANDLERS.values()):
                return True
        return False

    def _tick_observation_frame(self):
        world = CarlaDataProvider.get_world()
        timeout = float((self.config.get("carla", {}) or {}).get("timeout", 60.0))
        world.tick(timeout)
        timestamp = world.get_snapshot().timestamp
        GameTime.on_carla_tick(timestamp)
        CarlaDataProvider.on_carla_tick()
    
    def render(self, mode='human'):
        return self.env.render(mode=mode)

    def close(self):
        seen_handlers = set()
        for handler in getattr(self, "obs_handlers", {}).values():
            handler_id = id(handler)
            if handler_id in seen_handlers:
                continue
            seen_handlers.add(handler_id)
            if hasattr(handler, "close"):
                handler.close()
        if hasattr(self.env, "close"):
            self.env.close()

class EventTerminationWrapper():
    """
    Terminates episodes when configured Leaderboard or custom events fire.

    Example ``config['environment']``:
        terminate_events:
            # Leaderboard events
            - COLLISION_PEDESTRIAN
            - COLLISION_VEHICLE
            - COLLISION_STATIC
            - TRAFFIC_LIGHT_INFRACTION
            - STOP_INFRACTION
            - VEHICLE_BLOCKED
            - ROUTE_DEVIATION

            # Custom events
            - USELESS_MOVE_TIMEOUT
            - OFF_ROAD
            - CUSTOM_ROUTE_DEVIATION

    Example ``config['termination']``:
        type: ppo  # ppo / a2c / sac / td3

    Example ``config['environment']['termination']``:
        deviation_threshold: 1.3
        off_road_threshold: 1.5
        blocked_time_out_limit: 30.0
    """
    def __init__(self, env):
        self.env = env
        self.config = self.env.config
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

        termination_config = self.config.get('termination', {})
        if not isinstance(termination_config, dict):
            raise ValueError("termination must be a dict, e.g. termination: {type: ppo}")
        self.termination_type = str(termination_config.get('type', '')).lower()
        if self.termination_type not in TERMINATION_HANDLER_REGISTRY:
            raise ValueError(
                f"Unknown termination type: '{self.termination_type}'. "
                f"Supported types: {sorted(TERMINATION_HANDLER_REGISTRY)}"
            )
        
        env_config = self.env.config.get('environment', {})
        terminate_event_names = env_config.get('terminate_events', [])

        self.leaderboard_events = []
        self.custom_events = set()
        
        for event_name in terminate_event_names:
            if event_name in LEADERBOARD_EVENT_MAP:
                self.leaderboard_events.append(LEADERBOARD_EVENT_MAP[event_name])
            elif event_name in CUSTOM_TERMINATE_EVENTS:
                self.custom_events.add(event_name)
            else:
                all_events = list(LEADERBOARD_EVENT_MAP.keys()) + list(CUSTOM_TERMINATE_EVENTS)
                raise ValueError(
                    f"Unknown Termination Event Type: {event_name}\n"
                    f"Allowed Types: {all_events}"
                )
        
        if self.custom_events:
            self.termination_handler = self._create_termination_handler()
        else:
            self.termination_handler = None

    def _create_termination_handler(self):
        return TERMINATION_HANDLER_REGISTRY[self.termination_type](self.config)
    
    def __getattr__(self, name):
        """Forward unknown attributes to the wrapped environment."""
        if name == 'env':
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        return getattr(self.env, name)
        
    def step(self, action):
        """
        Execute one action and add configured termination events.

        Returns:
            tuple: (observation, reward, terminated, truncated, info)

        Note:
            Triggered events set ``terminated=True`` without changing
            ``truncated``.

        Info fields:
            - 'wrapper/term/triggered' (WRAPPER_TERM_TRIGGERED): bool,
              Internal signal from this wrapper or RewardWrapper fallback.
            - 'terminate_events': list of triggered events:
              {
                  'event_type': str,
                  'source': str,
                  'reason': str,
                  'details': dict
              }
            - 'all_events': list of serialized Leaderboard events.
        """
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        info[WRAPPER_TERM_TRIGGERED] = False
        info['terminate_events'] = []

        all_events = []
        if hasattr(self.env, 'scenario') and self.env.scenario is not None:
            for node in self.env.scenario.get_criteria():
                all_events.extend(node.events)
        
        all_events.sort(key=lambda e: e.get_frame(), reverse=True)

        # Keep event info serializable by stripping CARLA actor references.
        serializable_events = []
        for event in all_events:
            event_dict = {
                'type': event.get_type().name if hasattr(event.get_type(), 'name') else str(event.get_type()),
                'frame': event.get_frame(),
                'message': event.get_message(),
            }
            
            if hasattr(event, 'get_dict') and event.get_dict() is not None:
                original_dict = event.get_dict()
                safe_dict = {}
                for key, value in original_dict.items():
                    if hasattr(value, 'type_id'):
                        safe_dict[key] = {
                            'type_id': str(value.type_id),
                            'id': int(value.id) if hasattr(value, 'id') else None,
                        }
                    elif hasattr(value, 'x') and hasattr(value, 'y') and hasattr(value, 'z'):
                        safe_dict[key] = {
                            'x': float(value.x),
                            'y': float(value.y),
                            'z': float(value.z),
                        }
                    elif isinstance(value, (int, float, str, bool, type(None))):
                        safe_dict[key] = value
                    else:
                        safe_dict[key] = str(value)
                
                event_dict['details'] = safe_dict
            
            serializable_events.append(event_dict)
        
        info['all_events'] = serializable_events
        
        if self.termination_handler is not None:
            custom_result = self.termination_handler.check_all(info)

            for event in custom_result['triggered_events']:
                event_type = event['event_type']

                if event_type in self.custom_events:
                    terminated = True
                    info[WRAPPER_TERM_TRIGGERED] = True
                    info['terminate_events'].append({
                        'event_type': event_type,
                        'source': 'custom',
                        'reason': event['reason'],
                        'details': event['details']
                    })
        
        if self.leaderboard_events:
            for event in all_events:
                if event.get_type() in self.leaderboard_events:
                    terminated = True
                    info[WRAPPER_TERM_TRIGGERED] = True
                    
                    # Keep details serializable by stripping CARLA actor references.
                    safe_details = {
                        'frame': event.get_frame(),
                    }
                    
                    if hasattr(event, 'get_dict') and event.get_dict() is not None:
                        original_dict = event.get_dict()
                        safe_dict = {}
                        for key, value in original_dict.items():
                            if hasattr(value, 'type_id'):
                                safe_dict[key] = {
                                    'type_id': str(value.type_id),
                                    'id': int(value.id) if hasattr(value, 'id') else None,
                                }
                            elif hasattr(value, 'x') and hasattr(value, 'y') and hasattr(value, 'z'):
                                safe_dict[key] = {
                                    'x': float(value.x),
                                    'y': float(value.y),
                                    'z': float(value.z),
                                }
                            elif isinstance(value, (int, float, str, bool, type(None))):
                                safe_dict[key] = value
                            else:
                                safe_dict[key] = str(value)
                        
                        safe_details['dict'] = safe_dict
                    
                    info['terminate_events'].append({
                        'event_type': event.get_type().name,
                        'source': 'leaderboard',
                        'reason': event.get_message(),
                        'details': safe_details
                    })
        
        return observation, reward, terminated, truncated, info
    
    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        
        if self.termination_handler is not None:
            self.termination_handler.reset()

        info[WRAPPER_TERM_TRIGGERED] = False
        info['terminate_events'] = []
        info['all_events'] = []
        
        return obs, info

class RoutePlanWrapper():
    PROGRESS_CORRIDOR_M = 12.0

    def __init__(self, env):
        self.env = env
        self.window_size = 5
        self.config = self.env.config
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self._last_route_location = None
    
    def __getattr__(self, name):
        """Forward unknown attributes to the wrapped environment."""
        if name == 'env':
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        return getattr(self.env, name)

    @staticmethod
    def _segment_progress_and_lateral_distance(loc0, loc1, ego_location):
        seg_x = loc1.x - loc0.x
        seg_y = loc1.y - loc0.y
        seg_len_sq = seg_x * seg_x + seg_y * seg_y
        if seg_len_sq <= 1e-6:
            return 0.0, math.hypot(ego_location.x - loc0.x, ego_location.y - loc0.y)

        rel_x = ego_location.x - loc0.x
        rel_y = ego_location.y - loc0.y
        progress_ratio = (rel_x * seg_x + rel_y * seg_y) / seg_len_sq
        clamped_ratio = min(max(progress_ratio, 0.0), 1.0)
        proj_x = loc0.x + clamped_ratio * seg_x
        proj_y = loc0.y + clamped_ratio * seg_y
        lateral_distance = math.hypot(ego_location.x - proj_x, ego_location.y - proj_y)
        return progress_ratio, lateral_distance

    def _find_advance_index(self, vehicle_route, ego_location):
        closest_idx = 0
        if not vehicle_route:
            return closest_idx

        for i in range(len(vehicle_route) - 1):
            if i > self.window_size:
                break

            loc0 = vehicle_route[i][0].location
            loc1 = vehicle_route[i + 1][0].location
            progress_ratio, lateral_distance = self._segment_progress_and_lateral_distance(
                loc0,
                loc1,
                ego_location,
            )

            # require the ego to remain close to the route corridor.
            if lateral_distance <= self.PROGRESS_CORRIDOR_M and progress_ratio > 0.0:
                closest_idx = i + 1

        return closest_idx
        
    def step(self, action):
        """
        Execute one action and update route progress.

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        observation, reward, terminated, truncated, info = self.env.step(action)

        ev_location = CarlaDataProvider._ego_actor.get_location()
        distance_traveled_on_tick = 0.
        reward_distance_traveled_on_tick = 0.
        
        vehicle_route = CarlaDataProvider._ego_vehicle_route
        if not vehicle_route:
            CarlaDataProvider._distance_traveled_on_tick = 0.
            CarlaDataProvider._distance_traveled_on_tick_reward = 0.
            info[WRAPPER_ROUTE_DISTANCE_ON_TICK] = 0.
            info[WRAPPER_ROUTE_DISTANCE_ON_TICK_REWARD] = 0.
            return observation, reward, terminated, truncated, info

        closest_idx = self._find_advance_index(vehicle_route, ev_location)
                
        if closest_idx > 0:
            self._last_route_location = carla.Location(vehicle_route[0][0].location.x,
                                                       vehicle_route[0][0].location.y,
                                                       vehicle_route[0][0].location.z)
            
        distance_traveled_on_tick = self.get_length_traveled(vehicle_route[:closest_idx+1])
        CarlaDataProvider._ego_vehicle_route = vehicle_route[closest_idx:]
        reward_distance_traveled_on_tick = distance_traveled_on_tick
        CarlaDataProvider._distance_traveled_on_tick = distance_traveled_on_tick
        CarlaDataProvider._distance_traveled_on_tick_reward = reward_distance_traveled_on_tick
        info[WRAPPER_ROUTE_DISTANCE_ON_TICK] = distance_traveled_on_tick
        info[WRAPPER_ROUTE_DISTANCE_ON_TICK_REWARD] = reward_distance_traveled_on_tick
        return observation, reward, terminated, truncated, info
    
    def reset(self, seed=None, options=None):
        observation, info = self.env.reset(seed=seed, options=options)
        self._last_route_location = CarlaDataProvider._ego_actor.get_location()
        CarlaDataProvider._distance_traveled_on_tick = 0.
        CarlaDataProvider._distance_traveled_on_tick_reward = 0.
        info[WRAPPER_ROUTE_DISTANCE_ON_TICK] = 0.
        info[WRAPPER_ROUTE_DISTANCE_ON_TICK_REWARD] = 0.
        return observation, info

    def get_length_traveled(self, route):
        length_in_m = 0.0
        for i in range(len(route)-1):
            d = route[i][0].location.distance(route[i+1][0].location)
            length_in_m += d
        return length_in_m

class RewardWrapper():
    """
    Reward wrapper supporting ``simple`` and algorithm-specific reward handlers.
    """
    
    def __init__(self, env):
        self.env = env
        self.config = self.env.config
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        
        reward_config = self.config.get('reward', {})
        if not isinstance(reward_config, dict):
            raise ValueError("reward must be a dict, e.g. reward: {type: ppo}")
        self.reward_type = str(reward_config.get('type', '')).lower()
        if not self.reward_type:
            raise ValueError(f"reward.type is required. Supported types: {sorted(REWARD_HANDLER_REGISTRY)}")
        if self.reward_type in REWARD_HANDLER_REGISTRY:
            extra_reward_keys = sorted(set(reward_config.keys()) - {'type'})
            if extra_reward_keys:
                raise ValueError(
                    "reward config for type '%s' only supports the 'type' key. "
                    "Move reward parameters into the handler REWARD_CONFIG. "
                    "Unsupported keys: %s" % (self.reward_type, extra_reward_keys)
                )
        
        # Disabled by default; enable explicitly through environment.truncation.penalty.
        env_config = self.config.get('environment', {})
        self._max_episode_steps = int(env_config.get('max_episode_steps', 10000))
        truncation_config = env_config.get('truncation', {})
        if truncation_config is None:
            truncation_config = {}
        if not isinstance(truncation_config, dict):
            raise ValueError("environment.truncation must be a dict, e.g. truncation: {penalty: -200.0}")
        self._truncation_base_penalty = float(truncation_config.get('penalty', 0))
        self._episode_step = 0
        
        self.reward_handler = None
    
    def __getattr__(self, name):
        if name == 'env':
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        return getattr(self.env, name)
    
    def _get_route_completed_ratio_from_info(self, info):
        """Read route completion ratio in [0, 1] from normalized info fields."""
        if not isinstance(info, dict):
            return 0.0

        def _as_ratio(value, *, ratio_hint=False):
            if not isinstance(value, (int, float, np.number)):
                return None
            value = float(value)
            if not ratio_hint:
                value = value / 100.0
            return min(max(value, 0.0), 1.0)

        for key in ('route_completed_ratio', 'truncation_route_completed_ratio'):
            ratio = _as_ratio(info.get(key), ratio_hint=True)
            if ratio is not None:
                return ratio

        ratio = _as_ratio(info.get('simple_reward_RC'), ratio_hint=False)
        if ratio is not None:
            return ratio

        best_ratio = None
        for event in info.get('all_events', []):
            if not isinstance(event, dict):
                continue
            event_type = event.get('type')
            event_type = getattr(event_type, 'name', str(event_type))
            if event_type == 'ROUTE_COMPLETION':
                rc = event.get('route_completed')
                if rc is None:
                    rc = (event.get('details') or {}).get('route_completed')
                ratio = _as_ratio(rc, ratio_hint=False)
                if ratio is not None:
                    best_ratio = ratio if best_ratio is None else max(best_ratio, ratio)
        return best_ratio if best_ratio is not None else 0.0
    
    def _create_reward_handler(self):
        scenario_name = getattr(self.env, 'scenario_name', 'leaderboardv2')
        handler_cls = REWARD_HANDLER_REGISTRY.get(self.reward_type)
        if handler_cls is None:
            raise ValueError(
                f"Unknown reward type: '{self.reward_type}'. "
                f"Supported types: {sorted(REWARD_HANDLER_REGISTRY)}"
            )
        return handler_cls(self.config, scenario_name)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        if self.reward_handler is not None:
            crash_message = info.get('crash_message', '')
            reward_result = self.reward_handler.generate_reward(
                observation,
                terminated,
                info,
                action,
                crash_message=crash_message
            )

            # Accept historical reward-handler return shapes.
            blocked_status = False
            if isinstance(reward_result, tuple):
                if len(reward_result) == 3:
                    reward, new_info, blocked_status = reward_result
                    if isinstance(new_info, dict):
                        info = new_info
                elif len(reward_result) == 2:
                    reward, new_info = reward_result
                    if isinstance(new_info, dict):
                        info = new_info
                else:
                    raise ValueError(
                        f"Unexpected reward result tuple length: {len(reward_result)} "
                        f"for reward_type={self.reward_type}"
                    )
            else:
                reward = reward_result

            info[WRAPPER_REWARD_BLOCKED_BY_OBSTACLES] = blocked_status
            
            # SimpleReward can mark extra termination/truncation conditions.
            if info.get(WRAPPER_REWARD_SIMPLE_TERMINATED, False):
                terminated = True
                info[WRAPPER_TERM_TRIGGERED] = True
                simple_events = info.get('simple_terminate_events')
                if isinstance(simple_events, list):
                    terminate_events = info.get('terminate_events')
                    if not isinstance(terminate_events, list):
                        terminate_events = []
                        info['terminate_events'] = terminate_events
                    for event in simple_events:
                        if isinstance(event, dict) and event not in terminate_events:
                            terminate_events.append(event)
            if info.get(WRAPPER_REWARD_SIMPLE_TRUNCATED, False):
                truncated = True
            
            # Force termination for severe route-deviation penalties.
            if not terminated:
                # Visualization consumes this as a flat reward key.
                micro_deviation_penalty = info.get('micro_deviation_penalty', 0)
                parking_exit_deviation_exempt = bool(
                    info.get(WRAPPER_REWARD_PARKING_EXIT_DEVIATION_EXEMPT, False)
                )
                if (micro_deviation_penalty <= -40) and (not parking_exit_deviation_exempt):
                    terminated = True
                    info[WRAPPER_TERM_TRIGGERED] = True
                    if 'terminate_events' not in info:
                        info['terminate_events'] = []
                    has_route_deviation = any(
                        event.get('event_type') == 'ROUTE_DEVIATION' 
                        for event in info['terminate_events']
                    )
                    if not has_route_deviation:
                        info['terminate_events'].append({
                            'event_type': 'ROUTE_DEVIATION',
                            'source': 'reward_wrapper',
                            'reason': f'Severe route deviation detected (penalty={micro_deviation_penalty:.2f})',
                            'details': {'micro_deviation_penalty': float(micro_deviation_penalty)}
                        })

        info['route_completed_ratio'] = float(self._get_route_completed_ratio_from_info(info))

        # Apply a route-completion-scaled penalty on max-step truncation.
        self._episode_step += 1
        if (
            not terminated
            and self._truncation_base_penalty != 0
            and self._episode_step >= self._max_episode_steps
        ):
            ratio = info['route_completed_ratio']
            scale = 1.0 - ratio
            penalty = self._truncation_base_penalty * scale
            reward = float(reward) + penalty
            reward = np.array(reward, dtype=np.float32)
            info['truncation_penalty'] = float(penalty)
            info['truncation_route_completed_ratio'] = float(ratio)
            logger.debug(
                f"Truncation penalty: step={self._episode_step}, "
                f"route={ratio*100:.1f}%, penalty={penalty:.1f}"
            )

        return observation, reward, terminated, truncated, info
    
    def reset(self, seed=None, options=None):
        observation, info = self.env.reset(seed=seed, options=options)
        
        if self.reward_handler is None:
            self.reward_handler = self._create_reward_handler()
        self.reward_handler.scenario_name = getattr(self.env, 'scenario_name', 'leaderboardv2')
        self.reward_handler.reset()
        self._episode_step = 0
        
        return observation, info
    
    def render(self, mode='human'):
        return self.env.render(mode=mode)
    
    def close(self):
        if self.reward_handler is not None:
            if hasattr(self.reward_handler, 'destroy'):
                self.reward_handler.destroy()
            self.reward_handler = None
        
        if hasattr(self.env, 'close'):
            self.env.close()
    
class LQRExpertWrapper:
    """
    Environment wrapper that computes LQR expert actions at each step.

    Runs the rule-based LQR controller (from simple_control) inside the
    environment process (where CarlaDataProvider is available) and stores
    the resulting expert action in ``info['expert_action']`` as a numpy
    array in **env-action-space** format.

    The expert action is intended to serve as a per-sample BC target
    when ``bc_source='lqr'`` is configured in the RL algorithm.

    This wrapper is transparent: it does not modify observations, rewards,
    or termination signals.
    """

    def __init__(self, env, lqr_config: dict = None):
        self.env = env
        self.config = self.env.config
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        self._lqr_config = lqr_config or {}
        self._lqr_controller = None
        self._action_type = self.config.get('action_space', {}).get('type', '')
        self._last_obs = None

    def __getattr__(self, name):
        if name == 'env':
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )
        return getattr(self.env, name)

    def _ensure_controller(self):
        """Lazily create the LQR controller (requires CARLA world)."""
        if self._lqr_controller is not None:
            return
        import importlib
        try:
            _lqr_mod = importlib.import_module(
                'b2d_rlinfra.environment.rule_agents.simple_control.lqr_controller'
            )
        except Exception as exc:
            raise ImportError(
                "Failed to import "
                "b2d_rlinfra.environment.rule_agents.simple_control.lqr_controller."
            ) from exc
        LQRController = _lqr_mod.LQRController
        self._lqr_controller = LQRController(
            action_space=self.action_space,
            config=self._lqr_config,
        )

    def _control_to_env_action(self, tsb: np.ndarray) -> np.ndarray:
        """Convert [throttle, steer, brake] to the RL env-action-space format.

        Args:
            tsb: np.ndarray of shape (3,) — [throttle, steer, brake].

        Returns:
            np.ndarray matching self.action_space.
        """
        tsb = np.asarray(tsb, dtype=np.float32).reshape(-1)
        if tsb.shape[0] != 3:
            raise ValueError(
                f"LQR controller returned invalid control shape {tsb.shape}, expected (3,) "
                "for [throttle, steer, brake]."
            )
        if not np.all(np.isfinite(tsb)):
            raise ValueError(f"LQR controller returned non-finite control: {tsb}")

        throttle, steer, brake = float(tsb[0]), float(tsb[1]), float(tsb[2])

        if self._action_type in (
            'continuous_signed_pedal_steer',
            'continuous_accelerate_steering_rate',
        ):
            # action = [signed_pedal, steer]; signed_pedal = throttle - brake
            signed_pedal = throttle - brake
            env_action = np.array([signed_pedal, steer], dtype=np.float32)
        elif self._action_type in (
            'continuous_throttle_steer_brake',
            'continuous_accelerate_steering_brake',
        ):
            env_action = np.array([throttle, steer, brake], dtype=np.float32)
        else:
            raise ValueError(
                f"Unsupported action_space type for LQR expert: '{self._action_type}'"
            )

        # Clip to action space bounds
        if hasattr(self.action_space, 'low') and hasattr(self.action_space, 'high'):
            env_action = np.clip(env_action, self.action_space.low, self.action_space.high)

        return env_action

    def _compute_expert_action(self, obs) -> np.ndarray:
        if obs is None:
            raise RuntimeError("LQR expert action requested with empty observation.")

        self._ensure_controller()
        tsb = self._lqr_controller(obs)  # [throttle, steer, brake]
        expert_action = self._control_to_env_action(tsb)
        if not np.all(np.isfinite(expert_action)):
            raise ValueError(
                f"LQR wrapper produced invalid expert action: {expert_action}"
            )
        return expert_action

    def step(self, action):
        expert_action = self._compute_expert_action(self._last_obs)
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._last_obs = obs

        if info is None:
            info = {}
        elif not isinstance(info, dict):
            return obs, reward, terminated, truncated, info
        else:
            info = dict(info)

        info['expert_action'] = expert_action

        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)

        # Reset LQR controller for the new episode
        if self._lqr_controller is not None:
            try:
                self._lqr_controller.reset()
            except Exception as exc:
                raise RuntimeError("Failed to reset LQR controller.") from exc
        self._last_obs = obs

        if info is None:
            info = {}
        elif not isinstance(info, dict):
            return obs, info
        else:
            info = dict(info)

        info['expert_action'] = self._compute_expert_action(self._last_obs)

        return obs, info

    def render(self, mode='human'):
        return self.env.render(mode=mode)

    def close(self):
        self._lqr_controller = None
        self._last_obs = None
        if hasattr(self.env, 'close'):
            self.env.close()
