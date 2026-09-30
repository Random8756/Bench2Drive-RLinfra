"""Latest-only trainable weight publication for rl finetune."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

from b2d_rlinfra.finetuning.coordination import atomic_torch_save, atomic_write_json


class WeightStore:
    def __init__(self, weight_dir: str, filename: str = "policy_latest.pt"):
        self.weight_dir = Path(weight_dir)
        self.weight_dir.mkdir(parents=True, exist_ok=True)
        self.filename = filename
        self.latest_path = self.weight_dir / filename
        self.latest_json = self.weight_dir / "latest.json"

    def publish(
        self,
        *,
        policy_version: int,
        base_checkpoint: Optional[str],
        trainable_state_dict: Dict[str, Any],
        trainable_components: Any,
        dtype: str,
    ) -> Path:
        payload = {
            "policy_version": int(policy_version),
            "base_checkpoint": base_checkpoint,
            "trainable_state_dict": trainable_state_dict,
            "trainable_components": list(trainable_components),
            "dtype": str(dtype),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        atomic_torch_save(self.latest_path, payload)
        self._write_latest_json(payload)
        return self.latest_path

    def _write_latest_json(self, payload: Dict[str, Any]) -> None:
        data = {
            "policy_version": payload["policy_version"],
            "base_checkpoint": payload.get("base_checkpoint"),
            "trainable_components": payload.get("trainable_components", []),
            "dtype": payload.get("dtype"),
            "created_at": payload.get("created_at"),
            "path": str(self.latest_path),
        }
        atomic_write_json(self.latest_json, data)

    def load_latest(self, map_location: Optional[Any] = None) -> Optional[Dict[str, Any]]:
        if not self.latest_path.exists():
            return None
        return torch.load(self.latest_path, map_location=map_location, weights_only=True)

    def latest_version(self) -> Optional[int]:
        """Read the published policy_version from latest.json, or None if unavailable."""
        if not self.latest_json.exists():
            return None
        try:
            with open(self.latest_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            return int(data["policy_version"])
        except (OSError, ValueError, TypeError, KeyError):
            return None

    def maybe_load_into(
        self,
        adapter: Any,
        current_version: int,
        *,
        map_location: Optional[Any] = None,
        force: bool = False,
    ) -> Tuple[int, bool]:
        if not force:
            # publish() replaces the .pt before latest.json, so this version never
            # runs ahead of the payload; missing/corrupt json falls through to torch.load.
            known_version = self.latest_version()
            if known_version is not None and known_version <= int(current_version):
                return current_version, False
        payload = self.load_latest(map_location=map_location)
        if not payload:
            return current_version, False
        version = int(payload.get("policy_version", -1))
        if not force and version <= int(current_version):
            return current_version, False
        adapter.load_trainable_state_dict(payload["trainable_state_dict"])
        adapter.policy_version = version
        return version, True
