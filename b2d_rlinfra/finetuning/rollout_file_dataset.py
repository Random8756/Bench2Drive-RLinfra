"""mmap-backed, sample-level dataset over per-episode RolloutPack files."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from b2d_rlinfra.finetuning.rollout_pack import RolloutPackReader, unflatten_sample_fields

FILTER_OUTCOMES = ("success", "failure", "truncated")


def _to_torch(value: Any, device: Optional[torch.device]) -> Any:
    if isinstance(value, Mapping):
        return {key: _to_torch(item, device) for key, item in value.items()}
    tensor = torch.as_tensor(value)
    if tensor.dtype == torch.float64:
        tensor = tensor.float()
    if device is not None:
        tensor = tensor.to(device)
    return tensor


class RolloutFileDataset:
    """Read only rank-local samples while sharing RolloutPack page cache."""

    def __init__(self, index_path: str, *, sample_filter: Optional[Mapping[str, Any]] = None):
        self.index_path = Path(index_path)
        if sample_filter:
            # Filtering is resolved once by rank 0 into the UpdateSamplePlan.
            self.sample_filter = dict(sample_filter)
        else:
            self.sample_filter = {}
        with open(self.index_path, "r", encoding="utf-8") as handle:
            self.index = json.load(handle)
        if self.index.get("format") != "b2d-update-sample-plan" or int(self.index.get("version", -1)) != 2:
            raise ValueError(f"invalid UpdateSamplePlan: {self.index_path}")
        self.entries = [dict(entry) for entry in self.index.get("entries", [])]
        if not self.entries:
            raise ValueError(f"update sample plan has no RolloutPack entries: {self.index_path}")
        self.samples_before_filter = int(self.index["samples_before_filter"])
        self.samples_after_filter = int(self.index["samples_after_filter"])
        self.filtered_sample_counts_by_outcome = {
            outcome: int((self.index.get("sample_counts_by_outcome") or {}).get(outcome, 0))
            for outcome in FILTER_OUTCOMES
        }
        self.filtered_advantage_mean_by_outcome = {outcome: 0.0 for outcome in FILTER_OUTCOMES}
        self.distributed_dropped_samples = 0
        self._readers = [RolloutPackReader(entry["source_path"]) for entry in self.entries]
        try:
            expected_schema = str(self.index["schema_fingerprint"])
            for entry, reader in zip(self.entries, self._readers):
                if reader.schema_fingerprint != expected_schema:
                    raise ValueError(
                        f"RolloutPack schema changed after update plan creation: {reader.path}"
                    )
                if reader.header_crc32 != int(entry["header_crc32"]):
                    raise ValueError(f"RolloutPack header changed after update plan creation: {reader.path}")
                if reader.path.stat().st_size != int(entry["file_size"]):
                    raise ValueError(f"RolloutPack size changed after update plan creation: {reader.path}")
            self._field_paths = [tuple(field["path"]) for field in self._readers[0].fields]
            self._global_stops = np.asarray(
                [int(entry["global_stop"]) for entry in self.entries], dtype=np.int64
            )
            if int(self._global_stops[-1]) != self.samples_after_filter:
                raise ValueError("UpdateSamplePlan global ranges do not match samples_after_filter")
        except Exception:
            self.close()
            raise

    def __len__(self) -> int:
        return self.samples_after_filter

    def __enter__(self) -> "RolloutFileDataset":
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()

    def close(self) -> None:
        readers = getattr(self, "_readers", [])
        self._readers = []
        for reader in readers:
            reader.close()

    def _locations(self, global_indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        indices = np.asarray(global_indices, dtype=np.int64)
        if np.any(indices < 0) or np.any(indices >= len(self)):
            raise IndexError("global rollout sample index is out of range")
        pack_ids = np.searchsorted(self._global_stops, indices, side="right")
        global_starts = np.asarray(
            [int(self.entries[int(pack_id)]["global_start"]) for pack_id in pack_ids],
            dtype=np.int64,
        )
        retained_starts = np.asarray(
            [int(self.entries[int(pack_id)]["retained_start"]) for pack_id in pack_ids],
            dtype=np.int64,
        )
        local_offsets = indices - global_starts + retained_starts
        return pack_ids, local_offsets

    def _gather_field(
        self,
        path: Sequence[str],
        *,
        sample_count: int,
        groups: Sequence[Tuple[int, np.ndarray, np.ndarray]],
    ) -> np.ndarray:
        if not groups:
            raise ValueError("cannot gather an empty rollout batch")
        first_pack_id = int(groups[0][0])
        first = self._readers[first_pack_id].array(path)
        result = np.empty((int(sample_count), *first.shape[1:]), dtype=first.dtype)
        for pack_id, sorted_positions, sorted_offsets in groups:
            result[sorted_positions] = self._readers[int(pack_id)].array(path)[sorted_offsets]
        return result

    def _batch_from_indices(
        self,
        batch_indices: np.ndarray,
        *,
        device: Optional[torch.device],
    ) -> Dict[str, Any]:
        indices = np.asarray(batch_indices, dtype=np.int64)
        pack_ids, local_offsets = self._locations(indices)
        groups = []
        for pack_id in np.unique(pack_ids):
            positions = np.flatnonzero(pack_ids == pack_id)
            offsets = local_offsets[positions]
            order = np.argsort(offsets, kind="stable")
            groups.append((int(pack_id), positions[order], offsets[order]))
        flattened = {
            path: self._gather_field(path, sample_count=len(indices), groups=groups)
            for path in self._field_paths
        }
        return _to_torch(unflatten_sample_fields(flattened), device)

    def iter_rank_batches(
        self,
        global_batch_size: int,
        *,
        rank: int,
        world_size: int,
        seed: int,
        epoch: int = 0,
        shuffle: bool = True,
        device: Optional[torch.device] = None,
    ) -> Iterator[Dict[str, Any]]:
        global_batch_size = int(global_batch_size)
        rank = int(rank)
        world_size = int(world_size)
        if global_batch_size <= 0:
            raise ValueError("global_batch_size must be positive")
        if world_size <= 0 or not 0 <= rank < world_size:
            raise ValueError(f"invalid distributed sampler rank={rank} world_size={world_size}")
        if global_batch_size % world_size != 0:
            raise ValueError("global_batch_size must be divisible by world_size")
        if len(self) < world_size:
            raise ValueError(
                f"dataset has {len(self)} samples, fewer than learner world_size={world_size}"
            )
        indices = np.arange(len(self), dtype=np.int64)
        if shuffle:
            np.random.default_rng(int(seed) + int(epoch)).shuffle(indices)
        dropped = len(indices) % world_size
        self.distributed_dropped_samples = int(dropped)
        if dropped:
            indices = indices[:-dropped]

        global_batches = [
            indices[start : start + global_batch_size]
            for start in range(0, len(indices), global_batch_size)
        ]
        # Drop a disproportionately small tail at global scope so every DDP
        # rank stays within the configured batch size and takes matching steps.
        local_batch_size = global_batch_size // world_size
        min_local_batch_size = max(2, (local_batch_size + 1) // 2)
        if (
            len(global_batches) > 1
            and len(global_batches[-1]) < min_local_batch_size * world_size
        ):
            self.distributed_dropped_samples += int(len(global_batches[-1]))
            global_batches.pop()

        for global_indices in global_batches:
            local_indices = global_indices[rank::world_size]
            if len(local_indices):
                yield self._batch_from_indices(local_indices, device=device)

    def advantage_statistics(self, learner_context: Any, *, chunk_size: int = 65536) -> Tuple[float, float]:
        rank = int(learner_context.rank)
        world_size = int(learner_context.world_size)
        values = np.zeros(2 + len(FILTER_OUTCOMES) * 2 + 1, dtype=np.float64)
        # [sum, sumsq, outcome_sum/count..., total_count]
        for entry, reader in zip(self.entries, self._readers):
            global_start = int(entry["global_start"])
            count = int(entry["retained_count"])
            retained_start = int(entry["retained_start"])
            delta = (rank - global_start) % world_size
            local_start = retained_start + delta
            local_stop = retained_start + count
            outcome_index = FILTER_OUTCOMES.index(str(entry["outcome"]))
            array = reader.array(("advantages",))
            step = max(1, int(chunk_size)) * world_size
            for chunk_start in range(local_start, local_stop, step):
                offsets = np.arange(
                    chunk_start,
                    min(local_stop, chunk_start + step),
                    world_size,
                    dtype=np.int64,
                )
                chunk = np.asarray(array[offsets], dtype=np.float64)
                chunk_sum = float(chunk.sum(dtype=np.float64))
                chunk_count = int(chunk.size)
                values[0] += chunk_sum
                values[1] += float(np.square(chunk).sum(dtype=np.float64))
                base = 2 + outcome_index * 2
                values[base] += chunk_sum
                values[base + 1] += chunk_count
                values[-1] += chunk_count
        tensor = torch.as_tensor(values, device=learner_context.device, dtype=torch.float64)
        tensor = learner_context.sum_tensor(tensor).cpu()
        total_sum = float(tensor[0].item())
        total_sumsq = float(tensor[1].item())
        total_count = int(round(float(tensor[-1].item())))
        if total_count != len(self):
            raise RuntimeError(
                f"distributed advantage scan counted {total_count} samples, expected {len(self)}"
            )
        mean = total_sum / total_count if total_count else 0.0
        if total_count > 1:
            variance = max(0.0, (total_sumsq - total_sum * total_sum / total_count) / (total_count - 1))
            std = math.sqrt(variance)
        else:
            std = 0.0
        for index, outcome in enumerate(FILTER_OUTCOMES):
            base = 2 + index * 2
            outcome_sum = float(tensor[base].item())
            outcome_count = int(round(float(tensor[base + 1].item())))
            self.filtered_advantage_mean_by_outcome[outcome] = (
                outcome_sum / outcome_count if outcome_count else 0.0
            )
        return float(mean), float(std)
