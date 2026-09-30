"""Layer 2 — Environment.

Environment abstraction: wraps CARLA in a Gym-style interface with
configurable observation, action, reward shaping, and termination logic.

Layer entries and implementation modules
----------------------------------------
    L2_CARLAEnv                  ← b2d_rlinfra.environment.carla_env
    L2_make_env                  ← b2d_rlinfra.learning.training.runner_utils
    L2_build_observation_space   ← b2d_rlinfra.environment.spaces
    L2_build_action_space        ← b2d_rlinfra.environment.spaces
    L2_*Wrapper                  ← b2d_rlinfra.environment.wrappers
    L2_BirdViewObsManager        ← b2d_rlinfra.environment.handlers.birdview_obs_handler
    L2_RGBSensorObsHandler       ← b2d_rlinfra.environment.handlers.rgb_sensor_obs_handler
    L2_ScalarObsHandler          ← b2d_rlinfra.environment.handlers.scalars_obs_handler
    Reward handler               ← b2d_rlinfra.environment.handlers.reward_handler_ppo
"""

from __future__ import annotations

__layer__ = (2, "Environment")

from b2d_rlinfra.environment.carla_env import CARLAEnv as L2_CARLAEnv

from b2d_rlinfra.environment.spaces import (
    build_observation_space as L2_build_observation_space,
    build_observation_space_dict as L2_build_observation_space_dict,
    build_action_space as L2_build_action_space,
)

from b2d_rlinfra.environment.wrappers import (
    ActionWrapper as L2_ActionWrapper,
    ObservationWrapper as L2_ObservationWrapper,
    RewardWrapper as L2_RewardWrapper,
    EventTerminationWrapper as L2_EventTerminationWrapper,
    RoutePlanWrapper as L2_RoutePlanWrapper,
    LQRExpertWrapper as L2_LQRExpertWrapper,
)

from b2d_rlinfra.environment.handlers.birdview_obs_handler import BirdViewObsManager as L2_BirdViewObsManager
from b2d_rlinfra.environment.handlers.rgb_sensor_obs_handler import RGBSensorObsHandler as L2_RGBSensorObsHandler
from b2d_rlinfra.environment.handlers.scalars_obs_handler import ScalarObsHandler as L2_ScalarObsHandler

from b2d_rlinfra.environment.handlers.reward_handler_ppo import RewardHandler as L2_PPORewardHandler
from b2d_rlinfra.environment.handlers.simple_reward_handler import SimpleRewardGenerator as L2_SimpleRewardGenerator
from b2d_rlinfra.environment.handlers.termination_handler_ppo import TerminationHandler as L2_PPOTerminationHandler

from b2d_rlinfra.environment.controllers.trajectory_controller import (
    TrajectoryController as L2_TrajectoryController,
)

# Wrapper-stack factory used by training and evaluation launchers.
from b2d_rlinfra.learning.training.runner_utils import make_env as L2_make_env


__all__ = [
    "L2_CARLAEnv",
    "L2_make_env",
    "L2_build_observation_space",
    "L2_build_observation_space_dict",
    "L2_build_action_space",
    "L2_ActionWrapper",
    "L2_ObservationWrapper",
    "L2_RewardWrapper",
    "L2_EventTerminationWrapper",
    "L2_RoutePlanWrapper",
    "L2_LQRExpertWrapper",
    "L2_BirdViewObsManager",
    "L2_RGBSensorObsHandler",
    "L2_ScalarObsHandler",
    "L2_PPORewardHandler",
    "L2_SimpleRewardGenerator",
    "L2_PPOTerminationHandler",
    "L2_TrajectoryController",
]
