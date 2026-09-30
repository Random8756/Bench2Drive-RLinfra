#!/usr/bin/env python3
"""Export a DrivePi0 RL-finetuned delta checkpoint for official DriveMoE evaluation."""

from __future__ import annotations

import argparse
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional

import torch


class DrivePi0CheckpointExportError(ValueError):
    """Raised when a DrivePi0 RL checkpoint cannot be exported safely."""


_DRIVEPI0_EXPORTABLE_PREFIXES = ("action_encoder.", "action_decoder.")


def _is_exportable_action_param(name: str) -> bool:
    return any(str(name).startswith(prefix) for prefix in _DRIVEPI0_EXPORTABLE_PREFIXES)


def _load_torch(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)


def _extract_trainable_state(payload: Any) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise DrivePi0CheckpointExportError("RL checkpoint payload must be a mapping")

    trainable_state = payload.get("trainable_state_dict")
    if isinstance(trainable_state, Mapping):
        return trainable_state

    metadata = payload.get("metadata")
    if isinstance(metadata, Mapping) and metadata.get("format") in {
        "drivepi0_trainable_v1",
        "drivepi0_trainable_v2",
    }:
        return payload

    raise DrivePi0CheckpointExportError("RL checkpoint is missing DrivePi0 trainable_state_dict")


def _extract_action_params(payload: Any) -> tuple[Mapping[str, Any], list[str]]:
    trainable_state = _extract_trainable_state(payload)
    action_params = trainable_state.get("drivepi0_action_expert")
    if not isinstance(action_params, Mapping) or not action_params:
        raise DrivePi0CheckpointExportError(
            "RL checkpoint is missing trainable_state_dict['drivepi0_action_expert']"
        )
    invalid = sorted(str(key) for key in action_params.keys() if not _is_exportable_action_param(str(key)))
    if invalid:
        preview = ", ".join(invalid[:5])
        raise DrivePi0CheckpointExportError(
            "RL checkpoint contains non-exportable DrivePi0 parameter(s); "
            f"expected only action_encoder/action_decoder: {preview}"
        )

    ignored = [
        str(key)
        for key in trainable_state.keys()
        if str(key) not in {"drivepi0_action_expert", "metadata"}
    ]
    return action_params, ignored


def _copy_state_dict(state_dict: Mapping[str, Any]) -> MutableMapping[str, Any]:
    copied = OrderedDict((str(key), value) for key, value in state_dict.items())
    metadata = getattr(state_dict, "_metadata", None)
    if metadata is not None:
        copied._metadata = metadata.copy() if hasattr(metadata, "copy") else metadata  # type: ignore[attr-defined]
    return copied


def _extract_base_model(payload: Any) -> tuple[dict[str, Any], MutableMapping[str, Any]]:
    if not isinstance(payload, Mapping):
        raise DrivePi0CheckpointExportError("Base checkpoint payload must be a mapping")

    if isinstance(payload.get("model"), Mapping):
        output_payload = dict(payload)
        model_state = _copy_state_dict(payload["model"])
        output_payload["model"] = model_state
        return output_payload, model_state

    if isinstance(payload.get("state_dict"), Mapping):
        model_state = _copy_state_dict(payload["state_dict"])
        return {"model": model_state}, model_state

    if payload:
        model_state = _copy_state_dict(payload)
        return {"model": model_state}, model_state

    raise DrivePi0CheckpointExportError("Base checkpoint model state is empty or invalid")


def _match_base_key(name: str, state_dict: Mapping[str, Any]) -> Optional[str]:
    candidates = [name]
    if name.startswith("_orig_mod."):
        stripped = name[len("_orig_mod.") :]
        candidates.extend([stripped, f"module.{stripped}"])
    else:
        candidates.append(f"_orig_mod.{name}")

    if name.startswith("module."):
        stripped = name[len("module.") :]
        candidates.extend([stripped, f"_orig_mod.{stripped}"])
    else:
        candidates.append(f"module.{name}")

    for candidate in candidates:
        if candidate in state_dict:
            return candidate
    return None


def _tied_action_proprio_counterpart(key: str) -> Optional[str]:
    if ".mixtures.proprio." in key:
        return key.replace(".mixtures.proprio.", ".mixtures.action.", 1)
    if ".mixtures.action." in key:
        return key.replace(".mixtures.action.", ".mixtures.proprio.", 1)
    return None


def export_drivepi0_checkpoint(
    *,
    rl_checkpoint: Path,
    base_checkpoint: Path,
    output: Path,
) -> dict[str, Any]:
    rl_payload = _load_torch(rl_checkpoint)
    base_payload = _load_torch(base_checkpoint)
    action_params, ignored = _extract_action_params(rl_payload)
    output_payload, merged_model = _extract_base_model(base_payload)

    exported: list[str] = []
    exported_base_keys: list[str] = []
    missing: list[str] = []
    shape_mismatches: list[str] = []

    for raw_name, value in action_params.items():
        name = str(raw_name)
        base_key = _match_base_key(name, merged_model)
        if base_key is None:
            missing.append(name)
            continue

        target_keys = [base_key]
        counterpart = _tied_action_proprio_counterpart(base_key)
        if counterpart is not None:
            if counterpart not in merged_model:
                missing.append(f"{name} tied counterpart: {counterpart}")
                continue
            target_keys.append(counterpart)

        target_keys = list(dict.fromkeys(target_keys))
        has_shape_mismatch = False
        for target_key in target_keys:
            if torch.is_tensor(value) and torch.is_tensor(merged_model[target_key]):
                if tuple(value.shape) != tuple(merged_model[target_key].shape):
                    shape_mismatches.append(
                        f"{name} -> {target_key}: rl={tuple(value.shape)} base={tuple(merged_model[target_key].shape)}"
                    )
                    has_shape_mismatch = True
        if has_shape_mismatch:
            continue

        exported_value = value.detach().cpu() if torch.is_tensor(value) else value
        for target_key in target_keys:
            merged_model[target_key] = exported_value
            exported_base_keys.append(target_key)
        exported.append(name)

    if missing:
        preview = ", ".join(missing[:5])
        raise DrivePi0CheckpointExportError(
            f"{len(missing)} DrivePi0 action-expert parameter(s) were not found in base checkpoint: {preview}"
        )
    if shape_mismatches:
        preview = ", ".join(shape_mismatches[:5])
        raise DrivePi0CheckpointExportError(
            f"{len(shape_mismatches)} DrivePi0 action-expert parameter shape mismatch(es): {preview}"
        )
    if not exported:
        raise DrivePi0CheckpointExportError(
            "No DrivePi0 parameters were exported; expected drivepi0_action_expert parameters"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = getattr(merged_model, "_metadata", None)
    meta = {
        "format": "drivepi0_eval_export_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "rl_checkpoint": str(rl_checkpoint),
        "base_checkpoint": str(base_checkpoint),
        "exported_params": exported,
        "exported_base_keys": exported_base_keys,
        "exported_param_policy": "drivepi0_action_io_only",
        "ignored_trainable_keys": ignored,
        "state_dict_metadata_preserved": metadata is not None,
        "state_dict_metadata_len": len(metadata) if metadata is not None else 0,
    }
    output_payload["meta"] = meta
    output_payload["model"] = merged_model
    torch.save(output_payload, output)
    return meta


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge DrivePi0 RL finetune action-expert weights into a base DrivePi0 checkpoint."
    )
    parser.add_argument("--rl-checkpoint", required=True, type=Path, help="Path to policy_latest.pt or rl_finetune_update_*.pt")
    parser.add_argument("--base-checkpoint", required=True, type=Path, help="Path to the base DrivePi0 checkpoint")
    parser.add_argument("--output", required=True, type=Path, help="Path to write the exported eval checkpoint")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    meta = export_drivepi0_checkpoint(
        rl_checkpoint=args.rl_checkpoint,
        base_checkpoint=args.base_checkpoint,
        output=args.output,
    )
    print(
        "Exported DrivePi0 eval checkpoint "
        f"to {args.output} with {len(meta['exported_params'])} parameter(s); "
        f"ignored {len(meta['ignored_trainable_keys'])} training-only key(s)."
    )


if __name__ == "__main__":
    main()
