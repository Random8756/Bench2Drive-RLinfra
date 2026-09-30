"""Node-0-only learner process group and rank command coordination."""

from __future__ import annotations

import json
import socket
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch
import torch.distributed as dist

from b2d_rlinfra.finetuning.coordination import atomic_write_json
from b2d_rlinfra.finetuning.policy_adapter import LearnerSpec


@dataclass(frozen=True)
class LearnerContext:
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    device: torch.device = torch.device("cpu")

    @property
    def distributed(self) -> bool:
        return self.world_size > 1

    def mean_tensor(self, value: torch.Tensor) -> torch.Tensor:
        if not self.distributed:
            return value
        return self.sum_tensor(value) / float(self.world_size)

    def sum_tensor(self, value: torch.Tensor) -> torch.Tensor:
        if not self.distributed:
            return value
        result = value.detach().clone()
        dist.all_reduce(result, op=dist.ReduceOp.SUM)
        return result

    def all_true(self, condition: bool) -> bool:
        if not self.distributed:
            return bool(condition)
        value = torch.tensor(int(bool(condition)), device=self.device, dtype=torch.int32)
        dist.all_reduce(value, op=dist.ReduceOp.MIN)
        return bool(value.item())

    def barrier(self) -> None:
        if self.distributed:
            dist.barrier()


def wrap_training_module(spec: LearnerSpec, context: LearnerContext) -> torch.nn.Module:
    if not context.distributed:
        return spec.module
    options = spec.ddp_options
    device_index = context.device.index
    device_kwargs: Dict[str, Any] = {}
    if context.device.type == "cuda":
        if device_index is None:
            raise ValueError("CUDA DDP learner requires an explicit device index")
        device_kwargs = {"device_ids": [device_index], "output_device": device_index}
    return torch.nn.parallel.DistributedDataParallel(
        spec.module,
        find_unused_parameters=options.find_unused_parameters,
        broadcast_buffers=options.broadcast_buffers,
        gradient_as_bucket_view=options.gradient_as_bucket_view,
        static_graph=options.static_graph,
        **device_kwargs,
    )


class LearnerCommandBus:
    """Atomic shared-file commands; NCCL is used only while ranks update."""

    def __init__(self, run_dir: str | Path):
        self.path = Path(run_dir) / "control" / "learner_command.json"

    def publish_update(self, *, update_id: int, index_path: str, sampler_seed: int) -> None:
        atomic_write_json(
            self.path,
            {
                "seq": int(update_id),
                "command": "update",
                "update_id": int(update_id),
                "index_path": str(index_path),
                "sampler_seed": int(sampler_seed),
                "updated_at": time.time(),
            },
        )

    def publish_stop(self, *, seq: int, reason: str) -> None:
        atomic_write_json(
            self.path,
            {
                "seq": int(seq),
                "command": "stop",
                "reason": str(reason),
                "updated_at": time.time(),
            },
        )

    def read(self) -> Optional[Dict[str, Any]]:
        if not self.path.exists():
            return None
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                value = json.load(handle)
        except (FileNotFoundError, json.JSONDecodeError):
            return None
        return dict(value) if isinstance(value, Mapping) else None

    def wait_next(self, *, last_seq: int, poll_interval: float = 0.2) -> Dict[str, Any]:
        while True:
            command = self.read()
            if command is not None and int(command.get("seq", -1)) > int(last_seq):
                return command
            time.sleep(float(poll_interval))


def find_free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def init_local_process_group(
    *,
    rank: int,
    world_size: int,
    device: torch.device,
    backend: str,
    master_port: int,
    timeout_seconds: float,
) -> LearnerContext:
    if device.type == "cuda":
        torch.cuda.set_device(device)
    dist.init_process_group(
        backend=str(backend),
        init_method=f"tcp://127.0.0.1:{int(master_port)}",
        rank=int(rank),
        world_size=int(world_size),
        timeout=timedelta(seconds=float(timeout_seconds)),
    )
    return LearnerContext(
        rank=int(rank),
        local_rank=int(rank),
        world_size=int(world_size),
        device=torch.device(device),
    )


def _learner_process_entry(
    local_rank: int,
    config_path: str,
    run_dir: str,
    max_updates: Optional[int],
    init_from_checkpoint: Optional[str],
    master_port: int,
) -> None:
    from b2d_rlinfra.finetuning.config_schema import normalize_rl_finetune_config
    from b2d_rlinfra.finetuning.coordination import FileControlPlane, FileEventBus
    from b2d_rlinfra.finetuning.coordinator import Coordinator, run_learner_follower
    from b2d_rlinfra.finetuning.topology import topology_from_file
    from b2d_rlinfra.learning.utils.config import load_config

    config = load_config(config_path)
    raw_config = normalize_rl_finetune_config(config.to_dict())
    config.raw_config = raw_config
    config.env_config = raw_config.get("env")
    topology = topology_from_file(Path(run_dir) / "topology_resolved.json")
    learners = topology.node(0).learners
    if not learners or int(local_rank) >= len(learners):
        raise ValueError(f"missing node 0 learner plan for local_rank={local_rank}")
    learner_cfg = dict((raw_config.get("rl_finetune", {}) or {}).get("learner", {}) or {})
    device = torch.device(learners[local_rank].device)
    context = init_local_process_group(
        rank=local_rank,
        world_size=len(learners),
        device=device,
        backend=str(learner_cfg.get("backend", "nccl")),
        master_port=master_port,
        timeout_seconds=float(learner_cfg.get("timeout_seconds", 1800.0)),
    )
    command_bus = LearnerCommandBus(run_dir)
    try:
        if local_rank == 0:
            coordinator = Coordinator(
                config=config,
                config_path=config_path,
                run_dir=run_dir,
                device=device,
                topology=topology,
                event_bus=FileEventBus(Path(run_dir) / "events"),
                control_plane=FileControlPlane(Path(run_dir) / "control"),
                spawn_collectors=False,
                init_from_checkpoint=init_from_checkpoint,
                learner_context=context,
                learner_command_bus=command_bus,
            )
            coordinator.run(max_updates=max_updates)
        else:
            run_learner_follower(
                raw_config=raw_config,
                device=device,
                learner_context=context,
                command_bus=command_bus,
                init_from_checkpoint=init_from_checkpoint,
            )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def run_node0_learner_group(
    *,
    config_path: str,
    run_dir: str,
    world_size: int,
    max_updates: Optional[int],
    init_from_checkpoint: Optional[str],
) -> None:
    if int(world_size) < 2:
        raise ValueError("DDP learner group requires world_size >= 2")
    master_port = find_free_local_port()
    torch.multiprocessing.spawn(
        _learner_process_entry,
        args=(
            str(config_path),
            str(run_dir),
            max_updates,
            init_from_checkpoint,
            master_port,
        ),
        nprocs=int(world_size),
        join=True,
    )
