"""Canonical observation and action space builders from environment config."""

from typing import Any, Dict, Tuple

import numpy as np
from gymnasium import spaces

from b2d_rlinfra.environment.handlers.ego_sensor_obs_handler import (
    EGO_SENSOR_HANDLERS,
    apply_observation_presets,
    ego_sensor_branch_space,
)

__layer__ = (2, "Environment")

_VECTOR_TYPES = ("bev_mask", "bev_image")


def resolve_rgb_obs_key(rgb_cfg: Dict[str, Any]) -> str:
    """Return the observation dict key used for RGB SHM IPC."""
    output_key = str(rgb_cfg.get("output_key", "rgb"))
    if not bool(rgb_cfg.get("emit_output", True)):
        temporal_key = rgb_cfg.get("temporal_output_key")
        if temporal_key:
            return str(temporal_key)
    return output_key


_CONTINUOUS_ACTION_TYPES = (
    "continuous_signed_pedal_steer",
    "continuous_accelerate_steering_rate",
    "continuous_throttle_steer_brake",
    "continuous_accelerate_steering_brake",
)
_SCALAR_ITEM_DIMS = {
    "speed": 1,
    "throttle": 1,
    "brake": 1,
    "steer": 1,
    "reverse": 1,
    "gear": 1,
}
_RGB_SENSOR_TYPE = "sensor.camera.rgb"


# ---------------------------------------------------------------------------
# Observation space
# ---------------------------------------------------------------------------

def _vector_branch_space(vector_cfg: Dict[str, Any]) -> spaces.Box:
    vtype = vector_cfg.get("type")
    if vtype not in _VECTOR_TYPES:
        raise ValueError(
            f"observation_space.vector.type must be one of {_VECTOR_TYPES}, "
            f"got {vtype!r}."
        )

    mask_width_m = float(vector_cfg.get("mask_width", 64))
    pixels_per_meter = float(vector_cfg.get("pixels_per_meter", 1.0))
    img_size = int(mask_width_m * pixels_per_meter)

    elements = vector_cfg.get("elements", {})
    history_index = vector_cfg.get("history_index", [-1])
    num_hist = len(history_index)

    channels = 0
    for _name, elem_cfg in elements.items():
        channels += num_hist if bool(elem_cfg.get("use_history", False)) else 1
    if channels == 0:
        raise ValueError(
            "observation_space.vector.elements must enable at least one BEV "
            "channel; refusing to infer a default channel count."
        )

    if vtype == "bev_mask":
        return spaces.Box(
            low=0, high=1, shape=(channels, img_size, img_size), dtype=np.uint8,
        )
    return spaces.Box(
        low=0, high=255, shape=(img_size, img_size, 3), dtype=np.uint8,
    )


def _scalars_branch_space(scalars_cfg: Dict[str, Any]) -> spaces.Box:
    items = scalars_cfg.get("items", [])
    history_index = scalars_cfg.get("history_index", [-1])
    num_hist = len(history_index)

    dim = 0
    for item in items:
        item_type = item.get("type")
        if item_type == "location":
            per_step_dim = 2 if item.get("use_history", False) else 4
        elif item_type in _SCALAR_ITEM_DIMS:
            per_step_dim = _SCALAR_ITEM_DIMS[item_type]
        else:
            raise ValueError(f"Unsupported scalar item type: {item_type!r}")

        dim += per_step_dim * (num_hist if item.get("use_history", False) else 1)

    return spaces.Box(
        low=-np.inf, high=np.inf, shape=(dim,), dtype=np.float32,
    )


def _rgb_branch_space(rgb_cfg: Dict[str, Any]) -> spaces.Box:
    camera_order = rgb_cfg.get("camera_order")
    if not camera_order:
        raise ValueError("observation_space.rgb.camera_order must be a non-empty list")

    raw_specs = rgb_cfg.get("sensors")
    if not raw_specs:
        raise ValueError("observation_space.rgb.sensors must be a non-empty list")

    spec_by_id = {}
    for spec in raw_specs:
        sensor_id = spec.get("id")
        if sensor_id is None:
            raise ValueError("Every RGB sensor spec must define an 'id'")
        sensor_id = str(sensor_id)
        if sensor_id in spec_by_id:
            raise ValueError(f"Duplicate RGB sensor id: {sensor_id}")
        sensor_type = str(spec.get("type", _RGB_SENSOR_TYPE))
        if sensor_type != _RGB_SENSOR_TYPE:
            raise ValueError(f"Unsupported RGB sensor type: {sensor_type!r}")
        spec_by_id[sensor_id] = spec

    ordered_specs = []
    for camera_id in camera_order:
        camera_id = str(camera_id)
        if camera_id not in spec_by_id:
            raise ValueError(f"Missing RGB camera spec for id={camera_id}")
        ordered_specs.append(spec_by_id[camera_id])

    heights = {int(spec["height"]) for spec in ordered_specs}
    widths = {int(spec["width"]) for spec in ordered_specs}
    if len(heights) != 1 or len(widths) != 1:
        raise ValueError("RGB observation requires all cameras to share width/height")

    return spaces.Box(
        low=0,
        high=255,
        shape=(len(ordered_specs), heights.pop(), widths.pop(), 3),
        dtype=np.uint8,
    )


def _minddrive_state_branch_space(_state_cfg: Dict[str, Any]) -> spaces.Dict:
    from b2d_rlinfra.environment.handlers.minddrive_state_obs_handler import minddrive_state_space

    return minddrive_state_space()


def build_observation_space_dict(env_config: Dict[str, Any]) -> spaces.Dict:
    """Build the canonical ``spaces.Dict`` corresponding to ``env_config``.

    Always returns a ``spaces.Dict`` with one or more of the keys
    ``"vector"``, ``"scalars"``, ``"rgb"``, ``"minddrive_state"``. This is
    the shape ObservationWrapper actually presents to downstream wrappers, so
    everything else should match it.
    """
    obs_cfg = apply_observation_presets(env_config.get("observation_space", {}))
    space_dict: Dict[str, spaces.Space] = {}

    vector_cfg = obs_cfg.get("vector", {})
    if vector_cfg.get("enable", False):
        space_dict["vector"] = _vector_branch_space(vector_cfg)

    scalars_cfg = obs_cfg.get("scalars", {})
    if scalars_cfg:
        space_dict["scalars"] = _scalars_branch_space(scalars_cfg)

    rgb_cfg = obs_cfg.get("rgb", {})
    if rgb_cfg.get("enable", False):
        if bool(rgb_cfg.get("emit_output", True)):
            space_dict[str(rgb_cfg.get("output_key", "rgb"))] = _rgb_branch_space(rgb_cfg)
        temporal_key = rgb_cfg.get("temporal_output_key")
        if temporal_key:
            camera_order = rgb_cfg.get("camera_order") or []
            temporal_camera_id = str(rgb_cfg.get("temporal_camera_id", camera_order[0] if camera_order else ""))
            if temporal_camera_id not in [str(camera_id) for camera_id in camera_order]:
                raise ValueError(
                    f"RGB temporal_camera_id={temporal_camera_id!r} is not in camera_order"
                )
            base_space = _rgb_branch_space(rgb_cfg)
            history_indices = rgb_cfg.get("temporal_history_index", [-2, -1])
            space_dict[str(temporal_key)] = spaces.Box(
                low=0,
                high=255,
                shape=(len(history_indices), base_space.shape[1], base_space.shape[2], 3),
                dtype=np.uint8,
            )

    for sensor_key, handler_cls in EGO_SENSOR_HANDLERS.items():
        sensor_cfg = obs_cfg.get(sensor_key, {}) or {}
        if sensor_cfg.get("enable", False) and bool(sensor_cfg.get("emit_output", True)):
            space_dict[str(sensor_cfg.get("output_key", sensor_key))] = ego_sensor_branch_space(
                handler_cls,
                sensor_cfg,
            )

    drivepi0_state_cfg = obs_cfg.get("drivepi0_state", {})
    if drivepi0_state_cfg.get("enable", False):
        history_index = drivepi0_state_cfg.get("history_index", [-5, -4, -3, -2, -1])
        output_key = str(drivepi0_state_cfg.get("output_key", "drivepi0_state"))
        space_dict[output_key] = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(history_index), 10),
            dtype=np.float32,
        )

    minddrive_state_cfg = obs_cfg.get("minddrive_state", {})
    if minddrive_state_cfg.get("enable", False):
        space_dict[str(minddrive_state_cfg.get("output_key", "minddrive_state"))] = (
            _minddrive_state_branch_space(minddrive_state_cfg)
        )

    if not space_dict:
        raise ValueError(
            "observation_space yaml does not enable any branch "
            "(neither 'vector', 'scalars', 'rgb', ego sensors, nor 'minddrive_state'). "
            "Refusing to build an empty observation space silently."
        )

    return spaces.Dict(space_dict)


def build_observation_space(env_config: Dict[str, Any]) -> spaces.Space:
    """Backwards-compatible entry point used by the training runners.

    Mirrors the legacy ``runner_utils.build_observation_space`` behaviour:
    when only the ``vector`` branch is present, return the inner ``Box`` so
    that policies expecting a flat tensor input keep working untouched.
    """
    space_dict = build_observation_space_dict(env_config)
    if list(space_dict.spaces.keys()) == ["vector"]:
        return space_dict.spaces["vector"]
    return space_dict


# ---------------------------------------------------------------------------
# Action space
# ---------------------------------------------------------------------------

def signed_pedal_to_throttle_brake(signed_pedal: float) -> Tuple[float, float]:
    """Map one signed longitudinal pedal to mutually exclusive CARLA pedals."""
    signed_pedal = float(signed_pedal)
    if signed_pedal >= 0.0:
        return signed_pedal, 0.0
    return 0.0, -signed_pedal

def build_action_space(env_config: Dict[str, Any]) -> spaces.Space:
    """Build the action space from yaml.

    Supported ``action_space.type`` values:
        * ``discrete``                              (yaml: ``discrete_actions_list``)
        * ``continuous_signed_pedal_steer``         (yaml: ``continuous_actions_list``)
        * ``continuous_accelerate_steering_rate``   (yaml: ``continuous_actions_list``)
        * ``continuous_throttle_steer_brake``       (yaml: ``continuous_actions_list``)
        * ``continuous_accelerate_steering_brake``  (yaml: ``continuous_actions_list``)
        * ``trajectory``                            (yaml: ``trajectory.{num_points, state_dim}``)
    """
    action_cfg = env_config.get("action_space", {})
    space_type = action_cfg.get("type")

    if space_type == "discrete":
        actions_list = action_cfg.get("discrete_actions_list")
        if actions_list is None:
            # Fallback for configs that only provide ``discrete_action_num``.
            n = int(action_cfg.get("discrete_action_num", 0))
            if n <= 0:
                raise ValueError(
                    "action_space.type='discrete' requires either "
                    "'discrete_actions_list' or a positive 'discrete_action_num'."
                )
            return spaces.Discrete(n)
        return spaces.Discrete(len(actions_list))

    if space_type in _CONTINUOUS_ACTION_TYPES:
        ranges = action_cfg.get("continuous_actions_list")
        if not ranges:
            raise ValueError(
                f"action_space.type={space_type!r} requires "
                "'continuous_actions_list' in yaml; refusing to silently "
                "default to a [-1, 1] box."
            )
        low = np.array([r[0] for r in ranges], dtype=np.float32)
        high = np.array([r[1] for r in ranges], dtype=np.float32)
        action_dim = int(action_cfg.get("action_dim", len(ranges)))
        if len(ranges) != action_dim:
            raise ValueError(
                f"Action config mismatch: action_dim={action_dim}, "
                f"len(continuous_actions_list)={len(ranges)}."
            )
        if space_type == "continuous_signed_pedal_steer" and action_dim != 2:
            raise ValueError(
                "action_space.type='continuous_signed_pedal_steer' requires "
                f"exactly two dimensions [signed_pedal, steer], got action_dim={action_dim}."
            )
        if (
            space_type == "continuous_signed_pedal_steer"
            and (not np.all(np.isfinite(low)) or not np.all(np.isfinite(high)))
        ):
            raise ValueError(
                f"continuous_signed_pedal_steer bounds must be finite, got "
                f"low={low.tolist()}, high={high.tolist()}."
            )
        if not np.all(low < high):
            raise ValueError(
                f"continuous_actions_list ranges must satisfy low < high, "
                f"got low={low.tolist()}, high={high.tolist()}."
            )
        return spaces.Box(low=low, high=high, dtype=np.float32)

    if space_type == "trajectory":
        traj_cfg = action_cfg.get("trajectory", {}) or {}
        num_points = int(traj_cfg.get("num_points", 4))
        state_dim = int(traj_cfg.get("state_dim", 3))
        flat_dim = num_points * state_dim
        return spaces.Box(
            low=-np.inf, high=np.inf, shape=(flat_dim,), dtype=np.float32,
        )

    raise ValueError(
        f"Unsupported action_space.type: {space_type!r}. "
        f"Allowed: ['discrete', {', '.join(repr(t) for t in _CONTINUOUS_ACTION_TYPES)}, 'trajectory']."
    )


__all__ = [
    "build_observation_space",
    "build_observation_space_dict",
    "build_action_space",
    "signed_pedal_to_throttle_brake",
]
