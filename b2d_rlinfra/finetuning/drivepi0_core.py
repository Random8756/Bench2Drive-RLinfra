"""DrivePi0 model runtime shared by rl_finetune adapters."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from b2d_rlinfra.learning.policies.drivepi0_kv_cache import kv_obs_keys, pack_kv_caches, unpack_kv_caches
from b2d_rlinfra.learning.utils.distributions import DiagGaussianDistribution


_PLACEHOLDER_MARKERS = ("<", ">", "TODO", "todo", "placeholder", "/path/to")
_IMAGE_PREPROCESS_MODES = {"rlinf", "default", "drivemoe", "official"}
_ACTION_INIT_MODES = {"zeros", "randn"}
_DRIVEPI0_RL_TRAINABLE_PREFIXES = ("action_encoder.", "action_decoder.")
_DRIVEMOE_IMAGE_AUGMENT_KWARGS = dict(
    random_resized_crop=dict(scale=[0.8, 1.0], ratio=[0.9, 1.1]),
    random_brightness=[0.1],
    random_contrast=[0.9, 1.1],
    random_saturation=[0.9, 1.1],
    random_hue=[0.05],
    augment_order=[
        "random_resized_crop",
        "random_brightness",
        "random_contrast",
        "random_saturation",
        "random_hue",
    ],
)


def _validate_required_path(value: Optional[str], *, field_name: str) -> Path:
    if value is None or str(value).strip() == "":
        raise ValueError(f"{field_name} is required for DrivePi0 rl_finetune")
    text = str(value)
    if any(marker in text for marker in _PLACEHOLDER_MARKERS):
        raise ValueError(f"{field_name} must point to a real file/directory, got placeholder {text!r}")
    path = Path(text).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{field_name} does not exist: {path}")
    return path


def ensure_drivemoe_on_path(repo_dir: Path) -> None:
    os.environ.setdefault("DRIVEMOE_REPO_DIR", str(repo_dir))
    os.environ.setdefault("REPO_DIR", str(repo_dir))
    for path in (repo_dir, repo_dir / "src" / "agent"):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def _is_drivepi0_rl_trainable_name(name: str) -> bool:
    return any(str(name).startswith(prefix) for prefix in _DRIVEPI0_RL_TRAINABLE_PREFIXES)


class DrivePi0Runtime:
    """Load DrivePi0 once and expose encode / act / evaluate helpers."""

    def __init__(
        self,
        *,
        config: Mapping[str, Any],
        device: torch.device,
        action_dim: int,
        value_hidden_dim: int = 512,
        log_std_init: float = -2.0,
        learning_rate: float = 1.0e-5,
        project_root: Optional[Path] = None,
    ):
        self.config = dict(config or {})
        self.device = torch.device(device)
        self.rgb_key = str(self.config.get("rgb_key", "drivepi0_rgb"))
        self.state_key = str(self.config.get("state_key", "drivepi0_state"))
        self.text_prompt = str(self.config.get("text_prompt", "predict trajectory"))
        self.kv_prefix = str(self.config.get("kv_prefix", "drivepi0_kv"))
        self.action_dim = int(action_dim)
        self.image_preprocess = str(self.config.get("image_preprocess", "rlinf")).strip().lower()
        if self.image_preprocess not in _IMAGE_PREPROCESS_MODES:
            choices = ", ".join(sorted(_IMAGE_PREPROCESS_MODES))
            raise ValueError(f"Unsupported DrivePi0 image_preprocess={self.image_preprocess!r}; expected one of: {choices}")
        if self.image_preprocess == "default":
            self.image_preprocess = "rlinf"
        if self.image_preprocess == "official":
            self.image_preprocess = "drivemoe"
        jpeg_quality = self.config.get("image_jpeg_quality", None)
        self.image_jpeg_quality = None if jpeg_quality is None else int(jpeg_quality)
        if self.image_jpeg_quality is not None and not (1 <= self.image_jpeg_quality <= 100):
            raise ValueError("policy_adapter.config.image_jpeg_quality must be in [1, 100]")
        self.image_augment = bool(self.config.get("image_augment", False))
        self.action_init_mode = str(self.config.get("action_init_mode", "zeros")).strip().lower()
        if self.action_init_mode not in _ACTION_INIT_MODES:
            choices = ", ".join(sorted(_ACTION_INIT_MODES))
            raise ValueError(f"Unsupported DrivePi0 action_init_mode={self.action_init_mode!r}; expected one of: {choices}")
        self.action_init_key = str(self.config.get("action_init_key", f"{self.kv_prefix}_action_init"))
        self._tf_module = None
        self._augment_image_fn = None
        self._project_root = project_root or Path(__file__).resolve().parents[2]
        self._kv_obs_keys = kv_obs_keys(self.kv_prefix)

        repo_dir = _validate_required_path(
            self.config.get("repo_dir", self.config.get("drivemoe_root", "../DriveMoE")),
            field_name="policy_adapter.config.repo_dir",
        )
        ensure_drivemoe_on_path(repo_dir)
        self.repo_dir = repo_dir
        self._load_model()
        state_dim = int(np.prod(self._state_shape()))
        self.value_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(state_dim, int(value_hidden_dim)),
            nn.LayerNorm(int(value_hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(value_hidden_dim), int(value_hidden_dim)),
            nn.LayerNorm(int(value_hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(value_hidden_dim), 1),
        ).to(self.device)
        self.action_dist = DiagGaussianDistribution(self.action_dim)
        self.log_std = nn.Parameter(torch.full((self.action_dim,), float(log_std_init), device=self.device))
        self.optimizer = torch.optim.AdamW(
            list(self.trainable_parameters()),
            lr=float(learning_rate),
            eps=1e-5,
        )

    def _state_shape(self) -> Tuple[int, ...]:
        shape = self.config.get("state_shape")
        if shape is not None:
            return tuple(int(v) for v in shape)
        return (5, 10)

    def _resolve_path(self, raw: Any, *, base: Optional[Path] = None, must_exist: bool = True) -> Path:
        raw_str = os.path.expandvars(str(raw))
        path = Path(raw_str).expanduser()
        candidates = []
        if path.is_absolute():
            candidates.append(path)
        else:
            if base is not None:
                candidates.append(base / path)
            candidates.append(self._project_root / path)
            candidates.append(self._project_root.parent / path)
        for candidate in candidates:
            if candidate.exists() or not must_exist:
                return candidate.resolve()
        raise FileNotFoundError(f"DrivePi0 path not found: {raw}")

    def _load_model(self) -> None:
        from omegaconf import OmegaConf
        from transformers import AutoTokenizer

        from src.model.DrivePi0.drivepi0 import DrivePiZero
        from src.model.DrivePi0.processing import VLAProcessor

        config_path = self._resolve_path(
            self.config.get("config_path", "config/eval/DrivePi0/closed_loop.yaml"),
            base=self.repo_dir,
        )
        checkpoint_path = _validate_required_path(
            str(self.config.get("checkpoint_path", self.config.get("checkpoint", ""))),
            field_name="policy_adapter.checkpoint",
        )
        pretrained_model_path = self._resolve_path(
            self.config.get("pretrained_model_path", "ckpts/paligemma-3b-pt-224"),
            base=self.repo_dir,
        )
        statistics_path = self._resolve_path(
            self.config.get("statistics_path", "config/statistics/b2d_statistics.json"),
            base=self.repo_dir,
        )

        cfg = OmegaConf.load(config_path)
        cfg.checkpoint_path = str(checkpoint_path)
        cfg.pretrained_model_path = str(pretrained_model_path)
        cfg.data.statistics_path = str(statistics_path)
        cfg.gpu_id = int(self.config.get("gpu_id", 0))
        cfg.use_bf16 = bool(self.config.get("use_bf16", True))
        cfg.num_inference_steps = int(
            self.config.get("num_inference_steps", cfg.num_inference_steps)
        )
        OmegaConf.resolve(cfg)

        self.drivepi0_config = cfg
        self.drivepi0_model = DrivePiZero(cfg, use_ddp=False)
        self._load_checkpoint(checkpoint_path)
        self.drivepi0_model.tie_action_proprio_weights()
        self.drivepi0_model.freeze_all_weights()
        self._set_drivepi0_rl_trainable(True)
        dtype = torch.bfloat16 if cfg.get("use_bf16", True) else torch.float32
        self.dtype = dtype
        self.drivepi0_model.to(device=self.device, dtype=dtype)
        self.drivepi0_model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(str(pretrained_model_path), padding_side="right")
        self.processor = VLAProcessor(
            self.tokenizer,
            num_image_tokens=cfg.vision.config.num_image_tokens,
            max_seq_len=cfg.max_seq_len,
            tokenizer_padding=cfg.tokenizer_padding,
        )
        with open(statistics_path, "r", encoding="utf-8") as handle:
            self._stats = json.load(handle)

    def _to_numpy(self, value: Any) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def _kv_packed_from_policy_state(self, policy_input_state: Mapping[str, Any]) -> Dict[str, np.ndarray]:
        return {key: self._to_numpy(policy_input_state[key]) for key in self._kv_obs_keys}

    def _load_checkpoint(self, path: Path) -> None:
        try:
            data = torch.load(path, weights_only=True, map_location="cpu")
        except Exception:
            data = torch.load(path, weights_only=False, map_location="cpu")
        state_dict = data.get("model", data) if isinstance(data, dict) else data
        state_dict = {str(k).replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        self.drivepi0_model.load_state_dict(state_dict, strict=True)

    def set_train(self, mode: bool) -> None:
        self.value_net.train(mode)
        if mode:
            # Rollouts are collected with DrivePi0 in eval mode. Keep the
            # policy forward path in the same mode during PPO logprob replay
            # so pre-step KL is not inflated by train/eval behavior drift.
            self.drivepi0_model.eval()
        else:
            self.drivepi0_model.eval()

    def _drivepi0_rl_trainable_named_parameters(self):
        for name, param in self.drivepi0_model.named_parameters():
            if _is_drivepi0_rl_trainable_name(name):
                yield name, param

    def _set_drivepi0_rl_trainable(self, enabled: bool) -> None:
        for name, param in self.drivepi0_model.named_parameters():
            param.requires_grad_(_is_drivepi0_rl_trainable_name(name) and enabled)

    def trainable_parameters(self):
        for _, param in self._drivepi0_rl_trainable_named_parameters():
            yield param
        yield from self.value_net.parameters()
        yield self.log_std

    def trainable_components(self) -> list[str]:
        return ["drivepi0_action_encoder", "drivepi0_action_decoder", "value_net", "log_std"]

    def trainable_state_dict(self) -> Dict[str, Any]:
        action_params = {
            name: param.detach().cpu()
            for name, param in self._drivepi0_rl_trainable_named_parameters()
        }
        return {
            "drivepi0_action_expert": action_params,
            "value_net": self.value_net.state_dict(),
            "log_std": self.log_std.detach().cpu(),
            "metadata": {
                "format": "drivepi0_trainable_v2",
                "trainable_components": self.trainable_components(),
                "trainable_scope": "drivepi0_action_io_only",
                "drivepi0_trainable_prefixes": list(_DRIVEPI0_RL_TRAINABLE_PREFIXES),
            },
        }

    def load_trainable_state_dict(self, state_dict: Dict[str, Any]) -> None:
        action_params = state_dict.get("drivepi0_action_expert")
        if not isinstance(action_params, Mapping) or not action_params:
            raise KeyError("DrivePi0 trainable state must include non-empty drivepi0_action_expert")
        own = dict(self.drivepi0_model.named_parameters())
        expected_names = {name for name, _ in self._drivepi0_rl_trainable_named_parameters()}
        received_names = {str(name) for name in action_params.keys()}
        unexpected = sorted(received_names - expected_names)
        missing = sorted(expected_names - received_names)
        if unexpected:
            preview = ", ".join(unexpected[:5])
            raise KeyError(f"DrivePi0 trainable state has unexpected frozen parameter(s): {preview}")
        if missing:
            preview = ", ".join(missing[:5])
            raise KeyError(f"DrivePi0 trainable state is missing action I/O parameter(s): {preview}")
        for name, value in action_params.items():
            name = str(name)
            if name not in own:
                raise KeyError(f"DrivePi0 trainable parameter missing in model: {name}")
            with torch.no_grad():
                own[name].copy_(value.to(self.device))
        if "value_net" in state_dict:
            self.value_net.load_state_dict(state_dict["value_net"])
        if "log_std" in state_dict:
            with torch.no_grad():
                self.log_std.copy_(state_dict["log_std"].to(self.device))

    def _prepare_images(self, rgb: torch.Tensor) -> torch.Tensor:
        if self.image_preprocess == "drivemoe":
            return self._prepare_images_drivemoe(rgb)
        return self._prepare_images_rlinf(rgb)

    def _prepare_images_rlinf(self, rgb: torch.Tensor) -> torch.Tensor:
        x = rgb.to(device=self.device)
        if x.ndim == 4:
            x = x.unsqueeze(0)
        if x.ndim != 5:
            raise ValueError(f"{self.rgb_key} must have shape [T,H,W,3] or [B,T,H,W,3]; got {tuple(x.shape)}")
        x = x[..., [2, 1, 0]]
        bsz, num_images, height, width, channels = x.shape
        x = x.permute(0, 1, 4, 2, 3).reshape(bsz * num_images, channels, height, width)
        x = F.interpolate(x.float(), size=(224, 224), mode="bilinear", align_corners=False)
        x = x.round().clamp(0, 255).to(torch.uint8)
        return x.reshape(bsz, num_images, channels, 224, 224)

    def _prepare_images_drivemoe(self, rgb: torch.Tensor) -> torch.Tensor:
        x = rgb.detach().cpu()
        if x.ndim == 4:
            x = x.unsqueeze(0)
        if x.ndim != 5:
            raise ValueError(f"{self.rgb_key} must have shape [T,H,W,3] or [B,T,H,W,3]; got {tuple(x.shape)}")
        if int(x.shape[-1]) != 3:
            raise ValueError(f"{self.rgb_key} last dimension must be 3 channels; got {tuple(x.shape)}")

        array = x.round().clamp(0, 255).to(torch.uint8).numpy()
        bsz, num_images = int(array.shape[0]), int(array.shape[1])
        frames = []
        for frame_bgr in array.reshape(-1, array.shape[-3], array.shape[-2], 3):
            frames.append(self._prepare_one_drivemoe_image(frame_bgr))
        stacked = np.stack(frames, axis=0).reshape(bsz, num_images, 224, 224, 3)
        images = torch.as_tensor(stacked, dtype=torch.uint8).permute(0, 1, 4, 2, 3).contiguous()
        return images

    def _prepare_one_drivemoe_image(self, frame_bgr: np.ndarray) -> np.ndarray:
        import cv2
        from PIL import Image

        image = cv2.cvtColor(np.asarray(frame_bgr, dtype=np.uint8), cv2.COLOR_BGR2RGB)
        if self.image_jpeg_quality is not None:
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(self.image_jpeg_quality)]
            ok, encoded = cv2.imencode(".jpg", image, encode_param)
            if not ok:
                raise RuntimeError("Failed to JPEG-encode DrivePi0 image")
            decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if decoded is None:
                raise RuntimeError("Failed to JPEG-decode DrivePi0 image")
            image = decoded

        pil_image = Image.fromarray(image)
        try:
            resample = Image.Resampling.BILINEAR
        except AttributeError:
            resample = Image.BILINEAR
        image = np.asarray(pil_image.resize((224, 224), resample=resample).convert("RGB"), dtype=np.uint8)
        if self.image_augment:
            image = self._augment_drivemoe_image(image)
        return image.astype(np.uint8, copy=False)

    def _augment_drivemoe_image(self, image: np.ndarray) -> np.ndarray:
        if self._tf_module is None or self._augment_image_fn is None:
            import tensorflow as tf

            try:
                tf.config.set_visible_devices([], "GPU")
            except Exception:
                pass
            from src.data.utils.augmentations import augment_image

            self._tf_module = tf
            self._augment_image_fn = augment_image
        tf_tensor = self._tf_module.convert_to_tensor(np.asarray(image, dtype=np.uint8))
        output = self._augment_image_fn(image=tf_tensor, **_DRIVEMOE_IMAGE_AUGMENT_KWARGS)
        return output.numpy().astype(np.uint8)

    def denormalize_trajectory(self, action: torch.Tensor) -> torch.Tensor:
        fur_x = self._stats["fur_x"]
        fur_y = self._stats["fur_y"]
        out = action.clone()
        out[..., 0] = (out[..., 0] + 1.0) * (float(fur_x[1]) - float(fur_x[0])) / 2.0 + float(fur_x[0])
        out[..., 1] = (out[..., 1] + 1.0) * (float(fur_y[1]) - float(fur_y[0])) / 2.0 + float(fur_y[0])
        return out.float()

    def _encode_vlm_proprio_kv(
        self,
        *,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        image_text_proprio_mask: torch.FloatTensor,
        vlm_position_ids: torch.LongTensor,
        proprio_position_ids: torch.LongTensor,
        proprios: torch.FloatTensor,
    ) -> Dict[str, object]:
        model = self.drivepi0_model
        kv_caches = model.joint_model.build_mixture_caches()
        inputs_embeds = model._forward_siglip_and_text_embedding(input_ids, pixel_values)
        proprio_embeds = model.proprio_encoder(proprios)
        _, kv_caches = model.joint_model(
            attention_mask=image_text_proprio_mask,
            position_ids_all={"vlm": vlm_position_ids, "proprio": proprio_position_ids},
            embeds_all={"vlm": inputs_embeds, "proprio": proprio_embeds},
            kv_caches=kv_caches,
            return_caches=True,
        )
        return kv_caches

    def _rollout_action_from_kv(
        self,
        *,
        kv_caches: Dict[str, object],
        action_mask: torch.FloatTensor,
        action_position_ids: torch.LongTensor,
        bsz: int,
        initial_action: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        model = self.drivepi0_model
        if initial_action is None:
            if self.action_init_mode == "randn":
                action = torch.randn((bsz, model.horizon_steps, model.action_dim), device=self.device, dtype=self.dtype)
            else:
                action = torch.zeros((bsz, model.horizon_steps, model.action_dim), device=self.device, dtype=self.dtype)
        else:
            action = initial_action.to(device=self.device, dtype=self.dtype)
            if action.ndim == 2:
                action = action.unsqueeze(0)
            expected = (bsz, model.horizon_steps, model.action_dim)
            if tuple(action.shape) != expected:
                raise ValueError(f"DrivePi0 initial_action must have shape {expected}, got {tuple(action.shape)}")
        delta_t = 1.0 / model.num_inference_steps
        t = torch.zeros(bsz, device=self.device, dtype=self.dtype)
        for _ in range(model.num_inference_steps):
            time_cond = model.time_embedding(t)
            if model.action_expert_adaptive_mode:
                action_embeds = model.action_encoder(action)
            else:
                action_embeds = model.action_encoder(action, time_cond)
            action_embeds = model.joint_model(
                attention_mask=action_mask,
                position_ids_all={"action": action_position_ids},
                embeds_all={"action": action_embeds},
                time_cond=time_cond,
                kv_caches=kv_caches,
                cache_mode="append_non_active",
            )["action"]
            action = action + delta_t * model.action_decoder(action_embeds)
            t = t + delta_t
        if model.final_action_clip_value is not None:
            action = torch.clamp(action, -model.final_action_clip_value, model.final_action_clip_value)
        return action

    def _build_action_masks(self, bsz: int) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        dummy_images = torch.zeros(bsz, 2, 3, 224, 224, dtype=torch.uint8)
        model_inputs = self.processor(text=[self.text_prompt] * bsz, images=dummy_images)
        model_inputs = {k: v.to(self.device) for k, v in model_inputs.items()}
        causal_mask, _, _, action_position_ids = self.drivepi0_model.build_causal_mask_and_position_ids(
            model_inputs["attention_mask"], self.dtype
        )
        _, action_mask = self.drivepi0_model.split_full_mask_into_submasks(causal_mask)
        return action_mask.to(device=self.device), action_position_ids.to(device=self.device)

    def encode_observation(self, obs_mapping: Mapping[str, Any]) -> Dict[str, np.ndarray]:
        rgb_device = "cpu" if self.image_preprocess == "drivemoe" else self.device
        rgb = torch.as_tensor(np.asarray(obs_mapping[self.rgb_key]), device=rgb_device)
        if self.image_preprocess != "drivemoe":
            rgb = rgb.float()
        state = torch.as_tensor(np.asarray(obs_mapping[self.state_key]), device=self.device).float()
        if rgb.ndim == 4:
            rgb = rgb.unsqueeze(0)
        if state.ndim == 2:
            state = state.unsqueeze(0)
        images = self._prepare_images(rgb).cpu()
        proprios = state.to(dtype=self.dtype)
        bsz = int(proprios.shape[0])
        model_inputs = self.processor(text=[self.text_prompt] * bsz, images=images)
        model_inputs = {k: v.to(self.device) for k, v in model_inputs.items()}
        causal_mask, vlm_position_ids, state_position_ids, _ = (
            self.drivepi0_model.build_causal_mask_and_position_ids(model_inputs["attention_mask"], self.dtype)
        )
        image_text_proprio_mask, _ = self.drivepi0_model.split_full_mask_into_submasks(causal_mask)
        kv_caches = self._encode_vlm_proprio_kv(
            input_ids=model_inputs["input_ids"],
            pixel_values=model_inputs["pixel_values"].to(self.dtype),
            image_text_proprio_mask=image_text_proprio_mask.to(self.device),
            vlm_position_ids=vlm_position_ids.to(self.device),
            proprio_position_ids=state_position_ids.to(self.device),
            proprios=proprios,
        )
        packed = pack_kv_caches(kv_caches, prefix=self.kv_prefix, squeeze_batch=True)
        policy_state = {self.state_key: proprios.detach().float().cpu().numpy().squeeze(0)}
        policy_state.update({k: np.asarray(v) for k, v in packed.items()})
        if self.action_init_mode == "randn":
            action_init = torch.randn(
                (bsz, self.drivepi0_model.horizon_steps, self.drivepi0_model.action_dim),
                device=self.device,
                dtype=self.dtype,
            )
            policy_state[self.action_init_key] = action_init.detach().float().cpu().numpy().squeeze(0)
        return policy_state

    def _policy_state_tensors(self, policy_input_state: Mapping[str, Any], *, batch: bool) -> Dict[str, torch.Tensor]:
        state = torch.as_tensor(policy_input_state[self.state_key], device=self.device, dtype=torch.float32)
        if not batch and state.ndim == 2:
            state = state.unsqueeze(0)
        packed = {
            key: self._to_numpy(policy_input_state[key])
            for key in self._kv_obs_keys
            if key in policy_input_state
        }
        kv_caches = unpack_kv_caches(packed, device=self.device, dtype=self.dtype, prefix=self.kv_prefix)
        return {"state": state, "kv_caches": kv_caches}

    def _action_init_from_policy_state(self, policy_input_state: Mapping[str, Any]) -> Optional[torch.Tensor]:
        if self.action_init_key in policy_input_state:
            return torch.as_tensor(policy_input_state[self.action_init_key], device=self.device)
        if self.action_init_mode == "randn":
            raise KeyError(
                f"DrivePi0 policy_input_state is missing {self.action_init_key!r}; "
                "required when action_init_mode='randn'"
            )
        return None

    def predict_normalized_mean_from_state(self, policy_input_state: Mapping[str, Any]) -> torch.Tensor:
        items = self._policy_state_tensors(policy_input_state, batch=False)
        bsz = int(items["state"].shape[0])
        action_mask, action_position_ids = self._build_action_masks(bsz)
        action_init = self._action_init_from_policy_state(policy_input_state)
        pred_norm = self._rollout_action_from_kv(
            kv_caches=items["kv_caches"],
            action_mask=action_mask,
            action_position_ids=action_position_ids,
            bsz=bsz,
            initial_action=action_init,
        )
        return pred_norm.float().reshape(bsz, -1)

    def predict_normalized_mean_batch(self, policy_input_state: Mapping[str, Any]) -> torch.Tensor:
        state = torch.as_tensor(policy_input_state[self.state_key], device=self.device, dtype=torch.float32)
        bsz = int(state.shape[0])
        packed = self._kv_packed_from_policy_state(policy_input_state)
        kv_caches = unpack_kv_caches(packed, device=self.device, dtype=self.dtype, prefix=self.kv_prefix)
        action_mask, action_position_ids = self._build_action_masks(bsz)
        action_init = self._action_init_from_policy_state(policy_input_state)
        pred_norm = self._rollout_action_from_kv(
            kv_caches=kv_caches,
            action_mask=action_mask,
            action_position_ids=action_position_ids,
            bsz=bsz,
            initial_action=action_init,
        )
        return pred_norm.float().reshape(bsz, -1)

    def value_from_state(self, policy_input_state: Mapping[str, Any]) -> torch.Tensor:
        state = torch.as_tensor(policy_input_state[self.state_key], device=self.device, dtype=torch.float32)
        if state.ndim == 2:
            state = state.unsqueeze(0)
        return self.value_net(state).squeeze(-1)

    def distribution_from_state(self, policy_input_state: Mapping[str, Any]):
        mean = self.predict_normalized_mean_from_state(policy_input_state)
        return self.action_dist.proba_distribution(mean, self.log_std)

    def distribution_from_batch(self, policy_input_state: Mapping[str, Any]):
        mean = self.predict_normalized_mean_batch(policy_input_state)
        return self.action_dist.proba_distribution(mean, self.log_std)
