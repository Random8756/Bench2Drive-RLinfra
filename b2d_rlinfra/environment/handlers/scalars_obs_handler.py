"""Scalar observation handler.

Extracts scalar features (speed, throttle, brake, steer, ...), optionally
with history windows, used alongside the BEV mask as policy input.
"""
import carla
import numpy as np
from collections import deque
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

__layer__ = (2, "Environment")


class ScalarObsHandler():
    def __init__(self, scalar_obs_config):
        if not scalar_obs_config.get("items", False):
            raise ValueError("Non-scalar items are not supported/enabled!")
        self.config = scalar_obs_config

        self._normalize_by = {}
        for item in self.config["items"]:
            item_type = item["type"]
            try:
                normalize_by = float(item.get("normalize_by", 1.0))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Scalar item {item_type!r} normalize_by must be a positive number"
                ) from exc
            if not np.isfinite(normalize_by) or normalize_by <= 0.0:
                raise ValueError(
                    f"Scalar item {item_type!r} normalize_by must be a positive number"
                )
            self._normalize_by[item_type] = normalize_by
        
        self.use_history = "history_index" in self.config
        
        if self.use_history:
            self.history_indices = self.config["history_index"]
            self.history_len = abs(min(self.history_indices)) + 1
            
            self._history_buffers = {
                item["type"]: deque(maxlen=self.history_len)
                for item in self.config["items"]
                if item.get("use_history", False)
            }

    def reset(self):
        """Clear history buffers after environment reset."""
        if self.use_history:
            for buffer in self._history_buffers.values():
                buffer.clear()

    def _get_current_values(self):
        """Read raw scalar values for the current frame."""
        ego_actor = getattr(CarlaDataProvider, "_ego_actor", None)
        if ego_actor is None:
            raise ValueError("Ego vehicle not available in CarlaDataProvider!")
        control = ego_actor.get_control()
        location = CarlaDataProvider.get_transform(ego_actor).location
        ego_velocity = ego_actor.get_velocity()
        
        getters = {
            'speed': lambda: np.linalg.norm(np.array([ego_velocity.x, ego_velocity.y, ego_velocity.z])),
            'location': lambda: np.array([location.x, location.y, location.z, 1.0]),
            'throttle': lambda: control.throttle,
            'brake': lambda: control.brake,
            'steer': lambda: control.steer,
            'reverse': lambda: control.reverse,
            'gear': lambda: control.gear,
        }
        
        current = {}
        for item in self.config["items"]:
            item_type = item["type"]
            if item_type not in getters:
                raise ValueError(f"Unknown scalar type: {item_type}")
            current[item_type] = getters[item_type]() / self._normalize_by[item_type]
        return current

    def _batch_world_to_ego(self, locations, ego_transform):
        """Convert world locations to the current ego-local frame in batch.

        Args:
            locations: list of homogeneous [x, y, z, 1.0] arrays.
            ego_transform: current ego transform.
        """
        inv_matrix = np.array(ego_transform.get_inverse_matrix())

        world_points = np.stack(locations)
        local_points = world_points @ inv_matrix.T
        return local_points[:, :2]

    def _flatten_value(self, value):
        if isinstance(value, np.ndarray):
            return value.astype(np.float32).reshape(-1).tolist()
        if isinstance(value, (list, tuple)):
            return np.asarray(value, dtype=np.float32).reshape(-1).tolist()
        return [float(value)]

    def get_observation(self):
        if CarlaDataProvider.get_hero_actor() is None:
            raise ValueError("Ego vehicle not available in CarlaDataProvider!")

        current_values = self._get_current_values()
        
        if not self.use_history:
            values = []
            for item in self.config["items"]:
                item_type = item["type"]
                values.extend(self._flatten_value(current_values[item_type]))
            return {"scalars": np.asarray(values, dtype=np.float32)}
        
        for item_type, buffer in self._history_buffers.items():
            buffer.append(current_values[item_type])
        
        values = []
        for item in self.config["items"]:
            item_type = item["type"]
            
            if item.get("use_history", False):
                buffer = self._history_buffers[item_type]
                
                history_values = []
                for idx in self.history_indices:
                    if len(buffer) >= abs(idx):
                        history_values.append(buffer[idx])
                    else:
                        history_values.append(buffer[0] if buffer else current_values[item_type])
                
                if item_type == "location":
                    ego_transform = CarlaDataProvider._ego_actor.get_transform()
                    history_values = self._batch_world_to_ego(history_values, ego_transform)
                
                for value in history_values:
                    values.extend(self._flatten_value(value))
            else:
                values.extend(self._flatten_value(current_values[item_type]))
        
        return {"scalars": np.asarray(values, dtype=np.float32)}
