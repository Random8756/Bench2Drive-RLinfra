"""Convert Bench2Drive observations into MindDrive inference batches."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
import torch


def ensure_minddrive_on_path(minddrive_root: str) -> Path:
    root = Path(str(minddrive_root)).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"policy_adapter.config.minddrive_root does not exist: {root}")
    extra_dirs = [
        root,
        root / "rl_projects",
        root / "rl_projects" / "scenario_runner",
    ]
    for d in extra_dirs:
        d_str = str(d)
        if d_str not in sys.path:
            sys.path.insert(0, d_str)
    return root


def _first_scalar(value: Any) -> float:
    array = np.asarray(value)
    if array.size == 0:
        raise ValueError("expected non-empty scalar field")
    return float(array.reshape(-1)[0])


def _first_int(value: Any) -> int:
    array = np.asarray(value)
    if array.size == 0:
        raise ValueError("expected non-empty integer field")
    return int(array.reshape(-1)[0])


def _move_to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, Mapping):
        return {key: _move_to_device(v, device) for key, v in value.items()}
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    return value


def _collection_inference_pipeline_config(cfg: Any) -> list[Dict[str, Any]]:
    _skip_types = {"LoadMultiViewImageFromFilesInCeph"}
    _strip_keys = {"rl_training"}
    inference_pipeline = []
    for item in getattr(cfg, "inference_only_pipeline", []):
        cleaned = {k: v for k, v in dict(item).items() if k not in _strip_keys}
        if cleaned.get("type") in _skip_types:
            continue
        if cleaned.get("type") == "LoadAnnoatationMixCriticalVQATest":
            cleaned["single"] = True
        inference_pipeline.append(cleaned)
    return inference_pipeline


class MindDriveObsAdapter:
    def __init__(
        self,
        *,
        minddrive_root: str,
        minddrive_config: str,
        device: torch.device,
        rgb_key: str = "rgb",
        state_key: str = "minddrive_state",
        pipeline: Optional[Any] = None,
    ):
        ensure_minddrive_on_path(minddrive_root)
        config_path = Path(str(minddrive_config)).expanduser().resolve()
        if not config_path.exists():
            raise FileNotFoundError(f"policy_adapter.config.minddrive_config does not exist: {config_path}")

        from mmcv import Config
        from mmcv.core.bbox import get_box_type
        from mmcv.datasets.pipelines import Compose
        from mmcv.parallel.collate import collate as mm_collate_to_batch_form

        self.cfg = Config.fromfile(str(config_path))
        self.device = torch.device(device)
        self.rgb_key = str(rgb_key)
        self.state_key = str(state_key)
        self.get_box_type = get_box_type
        self.collate = mm_collate_to_batch_form
        if pipeline is None:
            pipeline = Compose(_collection_inference_pipeline_config(self.cfg))
        self.pipeline = pipeline

    def build_results(self, obs: Mapping[str, Any]) -> Dict[str, Any]:
        if "observation" in obs and isinstance(obs["observation"], Mapping):
            obs = obs["observation"]
        if self.rgb_key not in obs:
            raise KeyError(f"MindDrive observation requires RGB key {self.rgb_key!r}")
        if self.state_key not in obs:
            raise KeyError(f"MindDrive observation requires state key {self.state_key!r}")

        rgb = np.asarray(obs[self.rgb_key])
        if rgb.ndim != 4:
            raise ValueError(f"MindDrive RGB must be [num_cameras,H,W,C], got shape {rgb.shape}")
        state = obs[self.state_key]
        if not isinstance(state, Mapping):
            raise TypeError("MindDrive state observation must be a mapping")
        episode_id = _first_int(state.get("episode_id", np.asarray([0], dtype=np.int64)))

        results: Dict[str, Any] = {
            "lidar2img": np.asarray(state["lidar2img"], dtype=np.float32),
            "lidar2cam": np.asarray(state["lidar2cam"], dtype=np.float32),
            "cam_intrinsic": np.asarray(state["cam_intrinsic"], dtype=np.float32),
            "img": [np.asarray(rgb[idx], dtype=np.uint8) for idx in range(rgb.shape[0])],
            "folder": "rl_finetune",
            "scene_token": f"rl_finetune_ep_{episode_id}",
            "frame_idx": _first_int(state["frame_idx"]),
            "timestamp": _first_scalar(state["timestamp"]),
            "can_bus": np.asarray(state["can_bus"], dtype=np.float32),
            "ego_pose": np.asarray(state["ego_pose"], dtype=np.float32),
            "ego_pose_inv": np.asarray(state["ego_pose_inv"], dtype=np.float32),
            "command": _first_int(state["command"]),
            "ego_fut_cmd": np.asarray(state["ego_fut_cmd"], dtype=np.float32),
            "local_command_xy": np.asarray(state["local_command_xy"], dtype=np.float32),
        }
        results["lidar2ego"] = np.asarray(
            [[0.0, 1.0, 0.0, -0.39],
             [-1.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 1.0, 1.84],
             [0.0, 0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        results["l2g_r_mat"] = results["ego_pose"][:3, :3]
        results["l2g_t"] = results["ego_pose"][:3, 3]
        results["box_type_3d"], _ = self.get_box_type("LiDAR")

        stacked_imgs = np.stack(results["img"], axis=-1)
        results["img_shape"] = stacked_imgs.shape
        results["ori_shape"] = stacked_imgs.shape
        results["pad_shape"] = stacked_imgs.shape
        return results

    def to_batch(self, obs: Mapping[str, Any]) -> Dict[str, Any]:
        results = self.pipeline(self.build_results(obs))
        batch = self.collate([results], samples_per_gpu=1)
        return self._move_minddrive_batch(batch)

    def _move_minddrive_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        for key, data in list(batch.items()):
            if key == "img_metas":
                continue
            if isinstance(data, list) and data and torch.is_tensor(data[0]):
                data[0] = data[0].to(self.device)
            elif key == "input_ids" and isinstance(data, list):
                batch[key] = _move_to_device(data, self.device)
            else:
                batch[key] = _move_to_device(data, self.device)
        return batch
