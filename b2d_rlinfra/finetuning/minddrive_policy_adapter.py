"""MindDrive policy adapter for rl finetune."""

from __future__ import annotations

import importlib
import os
from collections import deque
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from b2d_rlinfra.finetuning.minddrive_obs_adapter import MindDriveObsAdapter, ensure_minddrive_on_path
from b2d_rlinfra.finetuning.policy_adapter import (
    DDPOptions,
    LearnerOutput,
    LearnerSpec,
    PolicyAdapter,
    PolicyStep,
)


_PLACEHOLDER_MARKERS = ("<", ">", "TODO", "todo", "placeholder", "/path/to")
_PID_CONTROLLER_ALIASES = {
    "rollout_decouple": "rollout_decouple",
    "eval_de": "eval_de",
}
_COLLECTION_ACTION_MODES = {"sample", "argmax"}
_FP16_PRECISIONS = {"fp16", "float16"}


def _validate_required_path(value: Optional[str], *, field_name: str) -> Path:
    if value is None or str(value).strip() == "":
        raise ValueError(f"{field_name} is required for policy_adapter.type=minddrive")
    text = str(value)
    if any(marker in text for marker in _PLACEHOLDER_MARKERS):
        raise ValueError(f"{field_name} must point to a real file/directory, got placeholder {text!r}")
    path = Path(text).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{field_name} does not exist: {path}")
    return path


def _to_device_tensor(value: Any, *, device: torch.device, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor.to(device)


def _to_scalar_int(value: Any) -> int:
    if torch.is_tensor(value):
        return int(value.detach().reshape(-1)[0].cpu().item())
    return int(np.asarray(value).reshape(-1)[0])


def _to_scalar_float(value: Any) -> float:
    if torch.is_tensor(value):
        return float(value.detach().reshape(-1)[0].cpu().item())
    return float(np.asarray(value).reshape(-1)[0])


def _metadata_array(metadata: Mapping[str, Any], key: str, default: Any = np.nan) -> np.ndarray:
    value = metadata.get(key, default)
    return np.asarray(value, dtype=np.float32)


def _first_log_prob_row(log_probs: torch.Tensor) -> torch.Tensor:
    if log_probs.ndim == 1:
        return log_probs
    return log_probs.reshape(-1, log_probs.shape[-1])[0]


def _select_action_log_prob(log_probs: torch.Tensor, action: int) -> torch.Tensor:
    return _first_log_prob_row(log_probs)[int(action)]


def _normalize_pid_controller_type(value: Any) -> str:
    normalized = str(value or "rollout_decouple").strip().lower()
    try:
        return _PID_CONTROLLER_ALIASES[normalized]
    except KeyError as exc:
        choices = ", ".join(sorted(_PID_CONTROLLER_ALIASES))
        raise ValueError(f"Unsupported MindDrive pid_controller {value!r}; expected one of: {choices}") from exc


def _is_fp16_precision(value: Any) -> bool:
    return str(value or "").strip().lower() in _FP16_PRECISIONS


class _MindDriveLearnerModule(torch.nn.Module):
    """MindDrive-owned training graph; third-party adapters provide their own."""

    def __init__(self, adapter: "MindDrivePolicyAdapter"):
        super().__init__()
        self.model = adapter.model
        object.__setattr__(self, "_adapter", adapter)

    def forward(self, batch: Dict[str, Any]) -> LearnerOutput:
        return self._adapter._learner_forward(batch)


class MindDrivePolicyAdapter(PolicyAdapter):
    def __init__(
        self,
        *,
        model: torch.nn.Module,
        obs_adapter: MindDriveObsAdapter,
        device: torch.device,
        optimizer: torch.optim.Optimizer,
        trainable_names: List[str],
        config: Mapping[str, Any],
        pid_controller: Optional[Any] = None,
    ):
        self.model = model
        self.obs_adapter = obs_adapter
        self.device = torch.device(device)
        self._trainable_names = list(trainable_names)
        self.config = dict(config or {})
        self.precision = str(self.config.get("precision", "fp16")).lower()
        self.kl_coef = float((self.config.get("ppo", {}) or {}).get("kl_coef", 0.0))
        self.use_kl = bool((self.config.get("ppo", {}) or {}).get("use_kl", self.kl_coef != 0.0))
        self.pid_controller_type = _normalize_pid_controller_type(self.config.get("pid_controller", "rollout_decouple"))
        self.collection_action_mode = str(self.config.get("collection_action_mode", "sample") or "sample").strip().lower()
        if self.collection_action_mode not in _COLLECTION_ACTION_MODES:
            choices = ", ".join(sorted(_COLLECTION_ACTION_MODES))
            raise ValueError(
                f"Unsupported MindDrive collection_action_mode {self.collection_action_mode!r}; "
                f"expected one of: {choices}"
            )
        self.pidcontroller = pid_controller
        self.policy_version = 0
        self._learner_module = _MindDriveLearnerModule(self)
        self._learner_spec = LearnerSpec(
            module=self._learner_module,
            optimizer=optimizer,
            precision=self.precision,
            ddp_options=DDPOptions(broadcast_buffers=False),
            aux_loss_weights={"kl_loss": self.kl_coef} if self.use_kl else {},
        )
        self._learner_spec.validate()

    @classmethod
    def load_initial(
        cls,
        config: Mapping[str, Any],
        checkpoint: Optional[str],
        device: torch.device,
    ) -> "MindDrivePolicyAdapter":
        cfg = dict(config or {})
        minddrive_root = _validate_required_path(cfg.get("minddrive_root"), field_name="policy_adapter.config.minddrive_root")
        minddrive_config = _validate_required_path(cfg.get("minddrive_config"), field_name="policy_adapter.config.minddrive_config")
        checkpoint_path = _validate_required_path(checkpoint, field_name="policy_adapter.checkpoint")
        ensure_minddrive_on_path(str(minddrive_root))

        # MindDrive configs use relative paths (e.g. 'ckpts/llava-qwen2-0.5b')
        # that resolve from the MindDrive repo root.
        prev_cwd = os.getcwd()
        os.chdir(str(minddrive_root))
        try:
            from mmcv import Config
            from mmcv.models import build_model
            from mmcv.utils import load_checkpoint, wrap_fp16_model

            md_cfg = Config.fromfile(str(minddrive_config))
            cls._import_config_plugin(md_cfg)
            model = build_model(md_cfg.model, train_cfg=md_cfg.get("train_cfg"), test_cfg=md_cfg.get("test_cfg"))
            load_checkpoint(model, str(checkpoint_path), map_location="cpu")
            if bool(cfg.get("reset_value_head", True)):
                cls._reset_value_net_pro(model, cfg.get("seed"))

            obs_adapter = MindDriveObsAdapter(
                minddrive_root=str(minddrive_root),
                minddrive_config=str(minddrive_config),
                device=torch.device(device),
                rgb_key=str(cfg.get("rgb_key", "rgb")),
                state_key=str(cfg.get("state_key", "minddrive_state")),
            )
        finally:
            os.chdir(prev_cwd)
        if hasattr(model, "rl_training"):
            model.rl_training = True
        model.to(device)
        cls._wrap_fp16_model_for_training(model, cfg.get("precision", "fp16"), wrap_fp16_model)

        trainable_names = cls._configure_trainable_parameters(model, cfg)
        if not trainable_names:
            raise ValueError(
                "MindDrive trainable parameter filter selected no parameters; expected decision_expert LoRA or value_net_pro"
            )
        learning_rate = float(cfg.get("learning_rate", 1.0e-4))
        optimizer = torch.optim.AdamW(
            [param for name, param in model.named_parameters() if name in set(trainable_names)],
            lr=learning_rate,
            weight_decay=float(cfg.get("weight_decay", 0.0)),
        )
        pid_controller = cls._make_pid_controller(str(minddrive_root), cfg.get("pid_controller", "rollout_decouple"))
        return cls(
            model=model,
            obs_adapter=obs_adapter,
            device=torch.device(device),
            optimizer=optimizer,
            trainable_names=trainable_names,
            config=cfg,
            pid_controller=pid_controller,
        )

    @staticmethod
    def _import_config_plugin(cfg: Any) -> None:
        if not bool(getattr(cfg, "plugin", False)):
            return
        plugin_dir = getattr(cfg, "plugin_dir", None)
        if not plugin_dir:
            return
        module_name = str(plugin_dir).strip("/").replace("/", ".")
        if module_name:
            importlib.import_module(module_name)

    @staticmethod
    def _wrap_fp16_model_for_training(model: torch.nn.Module, precision: Any, wrap_fp16_model: Any) -> None:
        if _is_fp16_precision(precision):
            wrap_fp16_model(model)

    @staticmethod
    def _reset_value_net_pro(model: torch.nn.Module, seed: Any = None) -> None:
        value_head = getattr(model, "value_net_pro", None)
        if value_head is None:
            raise AttributeError("MindDrive model does not expose value_net_pro; cannot reset value head")

        def reset() -> None:
            if isinstance(value_head, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(value_head.weight)
                if value_head.bias is not None:
                    torch.nn.init.zeros_(value_head.bias)
                return
            reset_parameters = getattr(value_head, "reset_parameters", None)
            if callable(reset_parameters):
                reset_parameters()
                return
            raise TypeError(
                "MindDrive value_net_pro must be torch.nn.Linear or expose reset_parameters() "
                f"to reset for new reward finetune, got {type(value_head).__name__}"
            )

        if seed is None:
            reset()
            return

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            reset()

    @staticmethod
    def _configure_trainable_parameters(model: torch.nn.Module, config: Mapping[str, Any]) -> List[str]:
        explicit_keywords = list(config.get("trainable", []) or config.get("trainable_modules", []) or [])
        trainable_names: List[str] = []
        for name, param in model.named_parameters():
            param.requires_grad_(False)
            selected = ("lora_" in name and "decision_expert" in name) or ("value_net_pro" in name)
            if explicit_keywords:
                selected = selected and any(str(keyword) in name for keyword in explicit_keywords)
            if selected:
                param.requires_grad_(True)
                trainable_names.append(name)
        return trainable_names

    @staticmethod
    def _make_pid_controller(minddrive_root: str, controller_type: Any = "rollout_decouple") -> Any:
        ensure_minddrive_on_path(minddrive_root)
        normalized = _normalize_pid_controller_type(controller_type)
        module_name = (
            "rl_projects.utils.pid_controller_decouple"
            if normalized == "rollout_decouple"
            else "team_code.pid_controller_de"
        )
        module = importlib.import_module(module_name)
        return module.PIDController()

    def _autocast(self):
        if self.device.type != "cuda":
            return nullcontext()
        if self.precision == "bf16":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    @contextmanager
    def _collection_action_selection(self):
        if self.collection_action_mode != "argmax":
            yield
            return

        distribution = getattr(self.model, "action_distribution", None)
        sample = getattr(distribution, "sample", None)
        if distribution is None or not callable(sample):
            yield
            return

        def _argmax_sample():
            mode = getattr(distribution, "mode", None)
            if callable(mode):
                return mode()
            torch_distribution = getattr(distribution, "distribution", None)
            if torch_distribution is None:
                raise RuntimeError("MindDrive action_distribution has no torch distribution for argmax sampling")
            probs = getattr(torch_distribution, "probs", None)
            if probs is not None:
                return torch.argmax(probs, dim=1)
            logits = getattr(torch_distribution, "logits", None)
            if logits is not None:
                return torch.argmax(logits, dim=1)
            raise RuntimeError("MindDrive action_distribution cannot expose probs/logits for argmax sampling")

        distribution.sample = _argmax_sample
        try:
            yield
        finally:
            distribution.sample = sample

    def on_episode_start(self, info: Mapping[str, Any]) -> None:
        seen = set()
        for target in (
            self.model,
            getattr(self.model, "pts_bbox_head", None),
            getattr(self.model, "map_head", None),
        ):
            if target is None or id(target) in seen:
                continue
            seen.add(id(target))
            reset_memory = getattr(target, "reset_memory", None)
            if callable(reset_memory):
                reset_memory()
        self._reset_pid_controller()

    def _reset_pid_controller(self) -> None:
        if self.pidcontroller is None:
            return

        reset = getattr(self.pidcontroller, "reset", None)
        if callable(reset):
            reset()
            return

        minddrive_root = self.config.get("minddrive_root")
        if minddrive_root:
            self.pidcontroller = self._make_pid_controller(str(minddrive_root), self.pid_controller_type)
            return

        for controller_name in ("turn_controller", "speed_controller"):
            controller = getattr(self.pidcontroller, controller_name, None)
            if controller is None:
                continue
            controller_reset = getattr(controller, "reset", None)
            if callable(controller_reset):
                controller_reset()
                continue
            window = getattr(controller, "_window", None)
            maxlen = getattr(window, "maxlen", None)
            if maxlen is not None:
                controller._window = deque([0 for _ in range(maxlen)], maxlen=maxlen)
            if hasattr(controller, "_max"):
                controller._max = 0.0
            if hasattr(controller, "_min"):
                controller._min = 0.0

    @torch.no_grad()
    def collect_step(self, obs: Any) -> PolicyStep:
        self.model.eval()
        batch = self.obs_adapter.to_batch(obs)
        with self._collection_action_selection():
            with self._autocast():
                outputs = self.model(batch, return_loss=False)
        pts_bbox = outputs[0]["pts_bbox"]
        meta_action_info = dict(pts_bbox["meta_action_info"])
        ppo_info = dict(pts_bbox["ppo_info"])

        stored_inputs_embeds = (
            meta_action_info["inputs_embeds"]
            .detach()
            .squeeze(0)
            .to(device="cpu", dtype=torch.float16)
            .numpy()
        )
        policy_input_state = {
            "inputs_embeds": stored_inputs_embeds,
            "new_input_ids": meta_action_info["new_input_ids"].detach().cpu().squeeze(0),
        }
        train_action = _to_scalar_int(pts_bbox["speed_value"])
        ref_log_probs = _to_device_tensor(ppo_info["reference_action_log_prob"], device=self.device, dtype=torch.float32)
        selected_old_log_prob = _select_action_log_prob(ref_log_probs, train_action)
        value = _to_scalar_float(ppo_info["values"])
        env_action, pid_info = self._control_from_pts_bbox(pts_bbox, obs)
        return PolicyStep(
            policy_input_state=policy_input_state,
            train_action=np.asarray(train_action, dtype=np.int64),
            env_action=env_action,
            value=value,
            old_action_log_prob=float(selected_old_log_prob.detach().cpu().item()),
            action_logprob_info={
                "ref_log_probs": _first_log_prob_row(ref_log_probs).detach().float().cpu().numpy(),
                "path_value": np.asarray(_to_scalar_int(pts_bbox["path_value"]), dtype=np.int64),
                **pid_info,
            },
        )

    @torch.no_grad()
    def value_from_obs(self, obs: Any) -> float:
        # MindDrive does not expose a value-only inference path yet; bootstrap
        # value estimation reuses collect_step under no_grad and pays one
        # extra forward/control pass on truncated episodes.
        return float(self.collect_step(obs).value)

    def _learner_forward(self, batch: Dict[str, Any]) -> LearnerOutput:
        policy_state = batch["policy_input_state"]
        model_input = {
            "inputs_embeds": _to_device_tensor(policy_state["inputs_embeds"], device=self.device, dtype=torch.float32),
            "new_input_ids": _to_device_tensor(policy_state["new_input_ids"], device=self.device, dtype=torch.long),
        }
        with self._autocast():
            action_log_probs, _action_language_log_probs, values = self.model(
                model_input,
                return_loss=False,
                is_rl_training=True,
            )
        action_log_probs = action_log_probs.float()
        actions = _to_device_tensor(batch["actions"], device=self.device, dtype=torch.long).view(-1, 1)
        selected_log_probs = action_log_probs.gather(1, actions).squeeze(1)
        values = values.float().view(-1)
        entropy = torch.distributions.Categorical(probs=torch.exp(action_log_probs)).entropy()

        aux_losses: Dict[str, torch.Tensor] = {}
        ref_log_probs = batch.get("ref_log_probs", batch.get("action_logprob_info_ref_log_probs"))
        if self.use_kl and ref_log_probs is not None:
            ref_log_probs = _to_device_tensor(ref_log_probs, device=self.device, dtype=torch.float32)
            aux_losses["kl_loss"] = F.kl_div(action_log_probs, ref_log_probs, log_target=True, reduction="batchmean")
        return {
            "log_probs": selected_log_probs,
            "values": values,
            "entropy": entropy,
            "aux_losses": aux_losses,
            "aux_logs": {"full_log_probs": action_log_probs.detach()},
        }

    def learner_spec(self) -> LearnerSpec:
        return self._learner_spec

    def _control_from_pts_bbox(self, pts_bbox: Mapping[str, Any], obs: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
        if "observation" in obs and isinstance(obs["observation"], Mapping):
            obs = obs["observation"]
        if self.pidcontroller is None:
            raise RuntimeError("MindDrive PID controller is not initialized")
        state = obs[str(self.config.get("state_key", "minddrive_state"))]
        speed = _to_scalar_float(np.asarray(state["can_bus"])[7])
        local_command_xy = np.asarray(state["local_command_xy"], dtype=np.float32)
        speed_waypoints = pts_bbox["ego_fut_preds"].detach().float().cpu().numpy()
        path_waypoints = pts_bbox["pw_ego_fut_pred"].detach().float().cpu().numpy()
        if self.pid_controller_type == "rollout_decouple":
            steer, throttle, brake, metadata = self.pidcontroller.control_pid(
                speed_waypoints,
                path_waypoints,
                speed,
                None,
            )
        else:
            steer, throttle, brake, metadata = self.pidcontroller.control_pid(
                path_waypoints,
                speed_waypoints,
                speed,
                local_command_xy,
            )
        if float(brake) < 0.05:
            brake = 0.0
        if float(throttle) > float(brake):
            brake = 0.0
        if speed > 5.0:
            throttle = 0.0
        action = np.asarray(
            [
                np.clip(float(throttle), 0.0, 0.75),
                np.clip(float(steer), -1.0, 1.0),
                np.clip(float(brake), 0.0, 1.0),
            ],
            dtype=np.float32,
        )
        return action, {
            "speed_waypoints": speed_waypoints,
            "path_waypoints": path_waypoints,
            "local_command_xy": np.asarray(local_command_xy, dtype=np.float32),
            "pid_steer_raw": np.asarray(float(steer), dtype=np.float32),
            "pid_throttle_raw": np.asarray(float(throttle), dtype=np.float32),
            "pid_brake_raw": np.asarray(float(brake), dtype=np.float32),
            "pid_desired_speed": np.asarray(float(metadata["desired_speed"]), dtype=np.float32),
            "pid_angle": _metadata_array(metadata, "angle"),
            "pid_angle_last": _metadata_array(metadata, "angle_last"),
            "pid_angle_target": _metadata_array(metadata, "angle_target"),
            "pid_angle_final": _metadata_array(metadata, "angle_final"),
            "pid_delta": _metadata_array(metadata, "delta"),
            "pid_aim": _metadata_array(metadata, "aim", [np.nan, np.nan]),
            "pid_target": _metadata_array(metadata, "target", [np.nan, np.nan]),
            "pid_lookahead": _metadata_array(metadata, "lookahead"),
        }

    def trainable_state_dict(self) -> Dict[str, Any]:
        params = {}
        for name, param in self.model.named_parameters():
            if name in set(self._trainable_names):
                params[name] = param.detach().cpu()
        return {
            "params": params,
            "metadata": {
                "format": "minddrive_trainable_v1",
                "trainable_names": list(self._trainable_names),
            },
        }

    def load_trainable_state_dict(self, state_dict: Dict[str, Any]) -> None:
        params = state_dict.get("params", state_dict)
        own_params = dict(self.model.named_parameters())
        missing = [name for name in self._trainable_names if name not in params]
        if missing:
            raise KeyError(f"MindDrive trainable state missing parameters: {missing[:5]}")
        with torch.no_grad():
            for name in self._trainable_names:
                own_params[name].copy_(params[name].to(self.device))

    def trainable_components(self) -> List[str]:
        components = []
        if any("lora_" in name and "decision_expert" in name for name in self._trainable_names):
            components.append("decision_expert_lora")
        if any("value_net_pro" in name for name in self._trainable_names):
            components.append("value_net_pro")
        return components

    def set_train(self, mode: bool = True) -> None:
        self.model.train(mode)
