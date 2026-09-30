"""Serialize DrivePi0 joint-model KV caches for rollout-buffer storage."""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch

KV_MIXTURES = ("vlm", "proprio")


def kv_obs_keys(prefix: str = "drivepi0_kv") -> List[str]:
    return [f"{prefix}_{name}_{slot}" for name in KV_MIXTURES for slot in ("k", "v")]


def pack_kv_caches(
    kv_caches: Dict[str, object],
    *,
    prefix: str = "drivepi0_kv",
    squeeze_batch: bool = True,
    storage_dtype: np.dtype = np.float16,
) -> Dict[str, np.ndarray]:
    """Pack mixture KV caches into numpy arrays keyed for observations."""
    packed: Dict[str, np.ndarray] = {}
    for name in KV_MIXTURES:
        cache = kv_caches[name]
        keys = torch.stack(cache.key_cache, dim=0).detach().cpu()
        values = torch.stack(cache.value_cache, dim=0).detach().cpu()
        if storage_dtype == np.float16:
            keys = keys.to(torch.float16)
            values = values.to(torch.float16)
        else:
            keys = keys.float()
            values = values.float()
        keys = keys.numpy()
        values = values.numpy()
        if squeeze_batch and keys.shape[1] == 1:
            keys = keys[:, 0]
            values = values[:, 0]
        packed[f"{prefix}_{name}_k"] = keys
        packed[f"{prefix}_{name}_v"] = values
    return packed


def unpack_kv_caches(
    packed: Dict[str, np.ndarray],
    *,
    device: torch.device,
    dtype: torch.dtype,
    prefix: str = "drivepi0_kv",
) -> Dict[str, object]:
    """Restore KVCache objects from packed observation arrays."""
    from src.model.kv_cache import KVCache

    kv_caches: Dict[str, object] = {}
    for name in KV_MIXTURES:
        keys = torch.as_tensor(packed[f"{prefix}_{name}_k"], device=device, dtype=dtype)
        values = torch.as_tensor(packed[f"{prefix}_{name}_v"], device=device, dtype=dtype)
        if keys.ndim == 4:
            # [layers, heads, seq, dim] for a single transition
            keys = keys.unsqueeze(1)
            values = values.unsqueeze(1)
        elif keys.ndim == 5:
            # [batch, layers, heads, seq, dim] from rollout-buffer batches
            keys = keys.permute(1, 0, 2, 3, 4).contiguous()
            values = values.permute(1, 0, 2, 3, 4).contiguous()
        else:
            raise ValueError(f"Unexpected KV tensor rank for {name}: {tuple(keys.shape)}")

        cache = KVCache()
        for layer_idx in range(keys.shape[0]):
            cache.key_cache.append(keys[layer_idx])
            cache.value_cache.append(values[layer_idx])
        kv_caches[name] = cache
    return kv_caches


def infer_kv_shapes(packed_example: Dict[str, np.ndarray], *, prefix: str = "drivepi0_kv") -> Dict[str, Tuple[int, ...]]:
    shapes: Dict[str, Tuple[int, ...]] = {}
    for key, value in packed_example.items():
        if not key.startswith(prefix):
            continue
        arr = np.asarray(value)
        if arr.ndim == 4:
            shapes[key] = tuple(arr.shape)
        elif arr.ndim == 5:
            shapes[key] = tuple(arr.shape[1:])
        else:
            raise ValueError(f"Unexpected packed KV shape for {key}: {arr.shape}")
    return shapes


def split_batched_storage_obs(
    batched: Dict[str, np.ndarray],
    batch_size: int,
) -> List[Dict[str, np.ndarray]]:
    items: List[Dict[str, np.ndarray]] = []
    for idx in range(batch_size):
        item = {}
        for key, value in batched.items():
            arr = np.asarray(value)
            if arr.ndim == len(item.get("_rank", (0,))) + 1:
                item[key] = arr[idx]
            elif arr.ndim >= 1 and arr.shape[0] == batch_size:
                item[key] = arr[idx]
            else:
                item[key] = arr
        items.append({k: v for k, v in item.items() if not k.startswith("_")})
    return items


def merge_raw_and_storage_obs(
    raw_obs: Dict[str, np.ndarray],
    kv_packed: Dict[str, np.ndarray],
    *,
    state_key: str,
    rgb_key: str,
) -> Dict[str, np.ndarray]:
    """Build buffer observation: proprio state + packed KV (drop raw RGB)."""
    out = {state_key: np.asarray(raw_obs[state_key])}
    out.update({k: np.asarray(v) for k, v in kv_packed.items()})
    return out
