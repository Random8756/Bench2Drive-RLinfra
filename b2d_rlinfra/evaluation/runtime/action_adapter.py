from __future__ import annotations

import numpy as np

import carla

from b2d_rlinfra.environment.spaces import signed_pedal_to_throttle_brake


class ActionAdapter:
    def __init__(self, action_config):
        self._config = action_config
        self._action_type = action_config.get("type")
        if self._action_type == "trajectory":
            raise ValueError("trajectory action space is not supported by leaderboard eval")

    @staticmethod
    def _flatten_action(action) -> np.ndarray:
        return np.asarray(action).reshape(-1)

    def to_control(self, action) -> carla.VehicleControl:
        action_type = self._action_type

        if action_type == "discrete":
            actions_list = self._config["discrete_actions_list"]
            action_index = int(self._flatten_action(action)[0])
            throttle, steer, brake = actions_list[action_index]
            return carla.VehicleControl(
                throttle=float(throttle),
                steer=float(steer),
                brake=float(brake),
            )

        flat_action = self._flatten_action(action)
        if action_type in (
            "continuous_signed_pedal_steer",
            "continuous_accelerate_steering_rate",
        ):
            signed_pedal = float(flat_action[0])
            steer = float(flat_action[1])
            throttle, brake = signed_pedal_to_throttle_brake(signed_pedal)
            return carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)

        if action_type in (
            "continuous_throttle_steer_brake",
            "continuous_accelerate_steering_brake",
        ):
            return carla.VehicleControl(
                throttle=float(flat_action[0]),
                steer=float(flat_action[1]),
                brake=float(flat_action[2]),
            )

        raise ValueError(f"Unsupported action space type: {action_type}")
