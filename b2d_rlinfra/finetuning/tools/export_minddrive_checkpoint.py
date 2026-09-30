#!/usr/bin/env python3
"""Export a MindDrive RL-finetuned delta checkpoint for official evaluation."""

from __future__ import annotations

import argparse
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional

import torch


_TRAINING_ONLY_MARKERS = (
    "value_net",
    "critic",
    "optimizer",
    "grad_scaler",
    "log_std",
)


class MindDriveCheckpointExportError(ValueError):
    """Raised when a MindDrive RL checkpoint cannot be exported safely."""


def _load_torch(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)


def _extract_trainable_params(payload: Any) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise MindDriveCheckpointExportError("RL checkpoint payload must be a mapping")

    trainable_state = payload.get("trainable_state_dict")
    if not isinstance(trainable_state, Mapping):
        raise MindDriveCheckpointExportError("RL checkpoint is missing trainable_state_dict")

    params = trainable_state.get("params")
    if not isinstance(params, Mapping) or not params:
        raise MindDriveCheckpointExportError("RL checkpoint is missing trainable_state_dict['params']")
    return params


def _extract_base_state_dict(payload: Any) -> MutableMapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise MindDriveCheckpointExportError("Base checkpoint payload must be a mapping")

    state_dict = payload.get("state_dict", payload)
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise MindDriveCheckpointExportError("Base checkpoint state_dict is empty or invalid")

    merged = OrderedDict((str(key), value) for key, value in state_dict.items())
    metadata = getattr(state_dict, "_metadata", None)
    if metadata is not None:
        merged._metadata = metadata.copy() if hasattr(metadata, "copy") else metadata  # type: ignore[attr-defined]
    return merged


def _is_exportable_param(name: str) -> bool:
    lowered = name.lower()
    if "value_net_pro" in lowered:
        return True
    if any(marker in lowered for marker in _TRAINING_ONLY_MARKERS):
        return False
    return "decision_expert" in lowered and "lora" in lowered


def _match_base_key(name: str, state_dict: Mapping[str, Any]) -> Optional[str]:
    candidates = [name]
    if name.startswith("module."):
        candidates.append(name[len("module.") :])
    else:
        candidates.append(f"module.{name}")
    if name.startswith("_orig_mod."):
        stripped = name[len("_orig_mod.") :]
        candidates.extend([stripped, f"module.{stripped}"])

    for candidate in candidates:
        if candidate in state_dict:
            return candidate
    return None


def export_minddrive_checkpoint(
    *,
    rl_checkpoint: Path,
    base_checkpoint: Path,
    output: Path,
) -> dict[str, Any]:
    rl_payload = _load_torch(rl_checkpoint)
    base_payload = _load_torch(base_checkpoint)
    trainable_params = _extract_trainable_params(rl_payload)
    merged_state_dict = _extract_base_state_dict(base_payload)

    exported: list[str] = []
    ignored: list[str] = []
    missing: list[str] = []
    shape_mismatches: list[str] = []

    for raw_name, value in trainable_params.items():
        name = str(raw_name)
        if not _is_exportable_param(name):
            ignored.append(name)
            continue

        base_key = _match_base_key(name, merged_state_dict)
        if base_key is None:
            missing.append(name)
            continue
        if torch.is_tensor(value) and torch.is_tensor(merged_state_dict[base_key]):
            if tuple(value.shape) != tuple(merged_state_dict[base_key].shape):
                shape_mismatches.append(
                    f"{name}: rl={tuple(value.shape)} base={tuple(merged_state_dict[base_key].shape)}"
                )
                continue
        merged_state_dict[base_key] = value.detach().cpu() if torch.is_tensor(value) else value
        exported.append(name)

    if missing:
        preview = ", ".join(missing[:5])
        raise MindDriveCheckpointExportError(
            f"{len(missing)} MindDrive exportable parameter(s) were not found in base checkpoint: {preview}"
        )
    if shape_mismatches:
        preview = ", ".join(shape_mismatches[:5])
        raise MindDriveCheckpointExportError(
            f"{len(shape_mismatches)} MindDrive exportable parameter shape mismatch(es): {preview}"
        )
    if not exported:
        raise MindDriveCheckpointExportError(
            "No MindDrive parameters were exported; expected decision_expert LoRA or value_net_pro parameters"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = getattr(merged_state_dict, "_metadata", None)
    result = {
        "meta": {
            "format": "minddrive_eval_export_v1",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "rl_checkpoint": str(rl_checkpoint),
            "base_checkpoint": str(base_checkpoint),
            "exported_params": exported,
            "exported_param_policy": "decision_expert_lora_and_value_net_pro",
            "ignored_trainable_params": ignored,
            "state_dict_metadata_preserved": metadata is not None,
            "state_dict_metadata_len": len(metadata) if metadata is not None else 0,
        },
        "state_dict": merged_state_dict,
    }
    torch.save(result, output)
    return result["meta"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge MindDrive RL finetune trainable weights into a base MindDrive checkpoint."
    )
    parser.add_argument("--rl-checkpoint", required=True, type=Path, help="Path to policy_latest.pt or rl_finetune_update_*.pt")
    parser.add_argument("--base-checkpoint", required=True, type=Path, help="Path to the base MindDrive checkpoint")
    parser.add_argument("--output", required=True, type=Path, help="Path to write the exported eval checkpoint")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    meta = export_minddrive_checkpoint(
        rl_checkpoint=args.rl_checkpoint,
        base_checkpoint=args.base_checkpoint,
        output=args.output,
    )
    print(
        "Exported MindDrive eval checkpoint "
        f"to {args.output} with {len(meta['exported_params'])} parameter(s); "
        f"ignored {len(meta['ignored_trainable_params'])} training-only parameter(s)."
    )


if __name__ == "__main__":
    main()
