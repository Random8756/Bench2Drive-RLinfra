"""Configuration normalization for rl_finetune runtime paths."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Mapping, Set

DISTRIBUTED_KEYS = {
    "enabled",
    "num_nodes",
    "collectors_per_node",
    "learner_node_collectors",
    "min_active_collectors",
    "startup_timeout",
    "heartbeat_interval",
    "heartbeat_timeout",
    "stagger_seconds",
    "cuda_vulkan_identity_fallback",
}

LEARNER_KEYS = {
    "strategy",
    "devices",
    "backend",
    "timeout_seconds",
}


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _validate_known_keys(section: str, config: Mapping[str, Any], allowed: Set[str]) -> None:
    unknown = sorted(set(config) - allowed)
    if unknown:
        formatted = ", ".join(repr(key) for key in unknown)
        raise ValueError(f"Invalid {section} config: unknown field(s): {formatted}")


def normalize_rl_finetune_config(raw_config: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a normalized config dict and validate distributed control fields.

    The public YAML remains intentionally plain dictionaries.  This function is
    the single place where rl_finetune runtime defaults and distributed schema
    checks are applied before the coordinator, collector, or node agent consume
    the config. Top-level ``rl_finetune`` remains open so model- or experiment-
    specific fields can be added without changing this central normalizer.
    """

    config = deepcopy(dict(raw_config or {}))
    rl_cfg = dict(config.get("rl_finetune", {}) or {})
    env_cfg = dict(config.get("env", {}) or {})
    carla_cfg = dict(env_cfg.get("carla", {}) or {})
    dist_cfg = dict(rl_cfg.get("distributed", {}) or {})
    learner_cfg = dict(rl_cfg.get("learner", {}) or {})

    _validate_known_keys("rl_finetune.distributed", dist_cfg, DISTRIBUTED_KEYS)
    _validate_known_keys("rl_finetune.learner", learner_cfg, LEARNER_KEYS)

    dist_cfg["enabled"] = _as_bool(dist_cfg.get("enabled", False))
    rl_cfg["distributed"] = dist_cfg

    learner_cfg["strategy"] = str(learner_cfg.get("strategy", "single")).strip().lower()
    if learner_cfg["strategy"] not in {"single", "ddp"}:
        raise ValueError("rl_finetune.learner.strategy must be 'single' or 'ddp'")
    devices = learner_cfg.get("devices", [rl_cfg.get("device", "cuda:0")])
    if isinstance(devices, str):
        devices = [devices]
    learner_cfg["devices"] = [str(device) for device in devices]
    if not learner_cfg["devices"]:
        raise ValueError("rl_finetune.learner.devices must contain at least one device")
    if len(set(learner_cfg["devices"])) != len(learner_cfg["devices"]):
        raise ValueError("rl_finetune.learner.devices must not contain duplicates")
    learner_cfg["backend"] = str(learner_cfg.get("backend", "nccl")).strip().lower()
    if learner_cfg["backend"] not in {"nccl", "gloo"}:
        raise ValueError("rl_finetune.learner.backend must be 'nccl' or 'gloo'")
    learner_cfg["timeout_seconds"] = float(learner_cfg.get("timeout_seconds", 1800.0))
    if learner_cfg["timeout_seconds"] <= 0:
        raise ValueError("rl_finetune.learner.timeout_seconds must be positive")
    if learner_cfg["strategy"] == "single" and len(learner_cfg["devices"]) != 1:
        raise ValueError("single learner strategy requires exactly one device")
    if learner_cfg["strategy"] == "ddp":
        if len(learner_cfg["devices"]) < 2:
            raise ValueError("ddp learner strategy requires at least two devices")
        if not dist_cfg["enabled"]:
            raise ValueError("ddp learner strategy currently requires distributed.enabled=true")
        if any(
            not device.lower().startswith("cuda:")
            or not device.rsplit(":", 1)[1].isdigit()
            for device in learner_cfg["devices"]
        ):
            raise ValueError("ddp learner devices must be explicit CUDA devices such as 'cuda:0'")
    configured_device = str(rl_cfg.get("device", learner_cfg["devices"][0]))
    if configured_device != learner_cfg["devices"][0]:
        raise ValueError(
            "rl_finetune.device must equal the first rl_finetune.learner.devices entry"
        )
    rl_cfg["device"] = learner_cfg["devices"][0]
    rl_cfg["learner"] = learner_cfg

    if "rollout_compression" in rl_cfg:
        raise ValueError(
            "rl_finetune.rollout_compression was removed; RolloutPack v1 is always uncompressed"
        )

    rl_cfg.setdefault("execution_mode", "process")
    rl_cfg.setdefault("rollouts_per_update", 1)
    rl_cfg.setdefault("collect_timeout", None)
    rl_cfg.setdefault("control_poll_interval", 1.0)
    rl_cfg.setdefault("shutdown_timeout", 120.0)
    rl_cfg.setdefault("checkpoint_interval", 1)
    rl_cfg.setdefault("max_policy_lag", 2)
    rl_cfg.setdefault("stale_rollout_action", "drop")
    rl_cfg.setdefault("cleanup_stale_rollouts", True)

    stale_action = str(rl_cfg.get("stale_rollout_action", "drop")).lower()
    if stale_action not in {"drop", "warn"}:
        raise ValueError("rl_finetune.stale_rollout_action must be 'drop' or 'warn'")
    rl_cfg["stale_rollout_action"] = stale_action
    if rl_cfg.get("max_policy_lag") is None and stale_action == "drop":
        raise ValueError(
            "rl_finetune.max_policy_lag cannot be null when stale_rollout_action='drop'"
        )

    if dist_cfg["enabled"]:
        if "num_collectors" in rl_cfg:
            raise ValueError(
                "Invalid distributed rl_finetune config: 'rl_finetune.num_collectors' "
                "must be omitted because collector count is derived from 'rl_finetune.distributed'"
            )
        if "num_envs" in carla_cfg:
            raise ValueError(
                "Invalid distributed rl_finetune config: 'env.carla.num_envs' "
                "must be omitted because collector count is derived from 'rl_finetune.distributed'"
            )
        for path_key in ("rollout_dir", "weight_dir"):
            if path_key in rl_cfg:
                raise ValueError(
                    f"Invalid distributed rl_finetune config: 'rl_finetune.{path_key}' "
                    "must be omitted because distributed runs store rollouts and weights under RUN_DIR"
                )
        dist_cfg.setdefault("num_nodes", 1)
        dist_cfg.setdefault("collectors_per_node", 1)
        dist_cfg.setdefault("startup_timeout", 600.0)
        dist_cfg.setdefault("heartbeat_interval", 10.0)
        dist_cfg.setdefault("heartbeat_timeout", 900.0)
        dist_cfg.setdefault("stagger_seconds", 0.0)
        dist_cfg.setdefault("cuda_vulkan_identity_fallback", True)

    if learner_cfg["strategy"] == "ddp":
        algorithm_cfg = dict(config.get("algorithm", {}) or {})
        global_batch_size = int(algorithm_cfg.get("batch_size", 64))
        world_size = len(learner_cfg["devices"])
        if global_batch_size <= 0 or global_batch_size % world_size != 0:
            raise ValueError(
                "algorithm.batch_size is the global learner batch size and must be positive "
                f"and divisible by learner world_size={world_size}; got {global_batch_size}"
            )
        if "collector_devices" not in rl_cfg:
            raise ValueError(
                "distributed DDP requires explicit rl_finetune.collector_devices so learner "
                "and collector GPU placement cannot overlap accidentally"
            )

    config["rl_finetune"] = rl_cfg
    config["env"] = env_cfg
    return config


def distributed_enabled(raw_config: Mapping[str, Any]) -> bool:
    rl_cfg = dict((raw_config or {}).get("rl_finetune", {}) or {})
    dist_cfg = dict(rl_cfg.get("distributed", {}) or {})
    return _as_bool(dist_cfg.get("enabled", False))


def bind_environment_result_dir(env_config: Mapping[str, Any], run_dir: str) -> Dict[str, Any]:
    """Return an env config whose result artifacts belong to one finetune run."""

    rebound = deepcopy(dict(env_config or {}))
    environment = dict(rebound.get("environment", {}) or {})
    environment["result_dir"] = str(run_dir)
    rebound["environment"] = environment
    return rebound
