"""Collector topology resolution for single-node and distributed rl_finetune."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping

from b2d_rlinfra.simulation.runners.carla_env_pool_utils import parse_connection_config

from b2d_rlinfra.finetuning.config_schema import distributed_enabled


@dataclass(frozen=True)
class CollectorPlan:
    collector_id: int
    node_rank: int
    local_id: int
    device: str
    host: str
    port: int
    traffic_manager_port: int
    traffic_manager_seed: int
    carla_gpu_id: Any

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CollectorPlan":
        return cls(
            collector_id=int(data["collector_id"]),
            node_rank=int(data["node_rank"]),
            local_id=int(data["local_id"]),
            device=str(data["device"]),
            host=str(data["host"]),
            port=int(data["port"]),
            traffic_manager_port=int(data["traffic_manager_port"]),
            traffic_manager_seed=int(data.get("traffic_manager_seed", 0)),
            carla_gpu_id=data.get("carla_gpu_id", 0),
        )


@dataclass(frozen=True)
class LearnerPlan:
    """One learner process. Learners are deliberately restricted to node 0."""

    node_rank: int
    global_rank: int
    local_rank: int
    device: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LearnerPlan":
        node_rank = int(data.get("node_rank", 0))
        if node_rank != 0:
            raise ValueError("learner placement is restricted to node_rank=0")
        return cls(
            node_rank=node_rank,
            global_rank=int(data["global_rank"]),
            local_rank=int(data["local_rank"]),
            device=str(data["device"]),
        )


@dataclass(frozen=True)
class NodePlan:
    node_rank: int
    collectors: List[CollectorPlan] = field(default_factory=list)
    learners: List[LearnerPlan] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_rank": self.node_rank,
            "collectors": [plan.to_dict() for plan in self.collectors],
            "learners": [plan.to_dict() for plan in self.learners],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "NodePlan":
        return cls(
            node_rank=int(data["node_rank"]),
            collectors=[CollectorPlan.from_dict(item) for item in data.get("collectors", [])],
            learners=[LearnerPlan.from_dict(item) for item in data.get("learners", [])],
        )


@dataclass(frozen=True)
class TopologyPlan:
    distributed: bool
    num_nodes: int
    total_collectors: int
    min_active_collectors: int
    nodes: List[NodePlan] = field(default_factory=list)

    def collectors(self) -> List[CollectorPlan]:
        result: List[CollectorPlan] = []
        for node in self.nodes:
            result.extend(node.collectors)
        return result

    def learners(self) -> List[LearnerPlan]:
        result: List[LearnerPlan] = []
        for node in self.nodes:
            result.extend(node.learners)
        return result

    @property
    def learner_world_size(self) -> int:
        return len(self.learners())

    def node(self, node_rank: int) -> NodePlan:
        for node in self.nodes:
            if node.node_rank == int(node_rank):
                return node
        raise ValueError(f"node_rank={node_rank} is not part of the topology")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "distributed": self.distributed,
            "num_nodes": self.num_nodes,
            "total_collectors": self.total_collectors,
            "min_active_collectors": self.min_active_collectors,
            "nodes": [node.to_dict() for node in self.nodes],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TopologyPlan":
        return cls(
            distributed=bool(data.get("distributed", False)),
            num_nodes=int(data.get("num_nodes", 1)),
            total_collectors=int(data.get("total_collectors", 0)),
            min_active_collectors=int(data.get("min_active_collectors", 0)),
            nodes=[NodePlan.from_dict(item) for item in data.get("nodes", [])],
        )


def _list_value(values: Any, index: int, *, default: Any) -> Any:
    if values is None:
        return default
    if isinstance(values, (list, tuple)):
        if index >= len(values):
            raise ValueError(f"Expected at least {index + 1} entries, got {len(values)}")
        return values[index]
    return values


def _node_counts(dist_cfg: Mapping[str, Any]) -> List[int]:
    num_nodes = int(dist_cfg.get("num_nodes", 1))
    counts_cfg = dist_cfg.get("collectors_per_node", 1)
    if isinstance(counts_cfg, (list, tuple)):
        if len(counts_cfg) != num_nodes:
            raise ValueError("distributed.collectors_per_node list length must match num_nodes")
        counts = [int(value) for value in counts_cfg]
    else:
        counts = [int(counts_cfg)] * num_nodes
    if "learner_node_collectors" in dist_cfg:
        counts[0] = int(dist_cfg["learner_node_collectors"])
    elif counts:
        counts[0] = max(0, counts[0] - 1)
    if any(count < 0 for count in counts):
        raise ValueError("collector counts must be non-negative")
    return counts


def _default_device(*, distributed: bool, node_rank: int, local_id: int, learner_device: str) -> str:
    if not distributed:
        return learner_device
    if node_rank == 0:
        return f"cuda:{local_id + 1}"
    return f"cuda:{local_id}"


def _device_for(
    rl_cfg: Mapping[str, Any],
    *,
    distributed: bool,
    node_rank: int,
    local_id: int,
    collector_id: int,
) -> str:
    devices = rl_cfg.get("collector_devices")
    learner_device = str(rl_cfg.get("device", "cuda:0"))
    default = _default_device(
        distributed=distributed,
        node_rank=node_rank,
        local_id=local_id,
        learner_device=learner_device,
    )
    if devices is None:
        return default
    if distributed:
        return str(_list_value(devices, local_id, default=default))
    return str(_list_value(devices, collector_id, default=default))


def _connection_slots(env_cfg: Mapping[str, Any], count: int, *, distributed: bool) -> Dict[str, List[Any]]:
    local_env = dict(env_cfg or {})
    local_carla = dict(local_env.get("carla", {}) or {})
    if str(local_carla.get("gpu_id", "")).lower() == "auto":
        local_carla["gpu_id"] = 0
    local_env["carla"] = local_carla
    configured = int(local_carla.get("num_envs", count) or count)
    slots = count if distributed else max(configured, count)
    return parse_connection_config(
        local_env,
        num_envs=slots,
        carla_ports_override=None,
        tm_ports_override=None,
        gpu_ids_override=None,
        hosts_override=None,
    )


def build_topology(raw_config: Mapping[str, Any]) -> TopologyPlan:
    rl_cfg = dict((raw_config or {}).get("rl_finetune", {}) or {})
    env_cfg = dict((raw_config or {}).get("env", {}) or {})
    carla_cfg = dict(env_cfg.get("carla", {}) or {})
    distributed = distributed_enabled(raw_config)
    learner_cfg = dict(rl_cfg.get("learner", {}) or {})
    learner_devices = list(learner_cfg.get("devices", [rl_cfg.get("device", "cuda:0")]))
    learners = [
        LearnerPlan(node_rank=0, global_rank=rank, local_rank=rank, device=str(device))
        for rank, device in enumerate(learner_devices)
    ]

    if not distributed:
        num_collectors = int(rl_cfg.get("num_collectors", 1))
        conn = _connection_slots(env_cfg, num_collectors, distributed=False)
        gpu_cfg = carla_cfg.get("gpu_id", conn["gpu_ids"])
        if str(gpu_cfg).lower() == "auto":
            gpu_cfg = conn["gpu_ids"]
        collectors = []
        for collector_id in range(num_collectors):
            collectors.append(
                CollectorPlan(
                    collector_id=collector_id,
                    node_rank=0,
                    local_id=collector_id,
                    device=_device_for(
                        rl_cfg,
                        distributed=False,
                        node_rank=0,
                        local_id=collector_id,
                        collector_id=collector_id,
                    ),
                    host=str(conn["hosts"][collector_id]),
                    port=int(conn["ports"][collector_id]),
                    traffic_manager_port=int(conn["tm_ports"][collector_id]),
                    traffic_manager_seed=int(conn["tm_seeds"][collector_id]),
                    carla_gpu_id=_list_value(gpu_cfg, collector_id, default=conn["gpu_ids"][collector_id]),
                )
            )
        total = len(collectors)
        return TopologyPlan(
            distributed=False,
            num_nodes=1,
            total_collectors=total,
            min_active_collectors=total,
            nodes=[NodePlan(node_rank=0, collectors=collectors, learners=learners)],
        )

    dist_cfg = dict(rl_cfg.get("distributed", {}) or {})
    counts = _node_counts(dist_cfg)
    max_local_collectors = max(counts) if counts else 0
    carla_gpu_cfg = carla_cfg.get("gpu_id", "auto")
    conn = _connection_slots(env_cfg, max_local_collectors, distributed=True)

    nodes: List[NodePlan] = []
    gid = 0
    for node_rank, count in enumerate(counts):
        collectors = []
        for local_id in range(count):
            collectors.append(
                CollectorPlan(
                    collector_id=gid,
                    node_rank=node_rank,
                    local_id=local_id,
                    device=_device_for(
                        rl_cfg,
                        distributed=True,
                        node_rank=node_rank,
                        local_id=local_id,
                        collector_id=gid,
                    ),
                    host=str(conn["hosts"][local_id]),
                    port=int(conn["ports"][local_id]),
                    traffic_manager_port=int(conn["tm_ports"][local_id]),
                    traffic_manager_seed=int(conn["tm_seeds"][local_id]),
                    carla_gpu_id=(
                        "auto"
                        if str(carla_gpu_cfg).lower() == "auto"
                        else _list_value(carla_gpu_cfg, local_id, default=conn["gpu_ids"][local_id])
                    ),
                )
            )
            gid += 1
        nodes.append(
            NodePlan(
                node_rank=node_rank,
                collectors=collectors,
                learners=learners if node_rank == 0 else [],
            )
        )

    total = gid
    if total < 1:
        raise ValueError("distributed topology must contain at least one collector")
    learner_device_set = {plan.device for plan in learners}
    overlap = sorted(
        learner_device_set.intersection(plan.device for plan in nodes[0].collectors)
    )
    if overlap:
        raise ValueError(
            "node 0 learner and collector devices must not overlap; overlapping device(s): "
            + ", ".join(overlap)
        )
    min_active = dist_cfg.get("min_active_collectors")
    if min_active is None:
        min_active = int(math.ceil(total * 2.0 / 3.0)) if total else 0
    min_active = int(min_active)
    if min_active < 1 or min_active > total:
        raise ValueError(
            f"distributed.min_active_collectors must satisfy 1 <= value <= total_collectors "
            f"({total}), got {min_active}"
        )
    return TopologyPlan(
        distributed=True,
        num_nodes=len(counts),
        total_collectors=total,
        min_active_collectors=min_active,
        nodes=nodes,
    )


def apply_carla_gpu_mapping(plan: TopologyPlan, cuda_to_vulkan: Mapping[int, int]) -> TopologyPlan:
    nodes: List[NodePlan] = []
    for node in plan.nodes:
        collectors: List[CollectorPlan] = []
        for collector in node.collectors:
            gpu_id = collector.carla_gpu_id
            if str(gpu_id).lower() == "auto":
                cuda_index = _cuda_index_from_device(collector.device)
                if cuda_index not in cuda_to_vulkan:
                    raise ValueError(
                        f"No Vulkan adapter mapping found for collector {collector.collector_id} "
                        f"device {collector.device}"
                    )
                gpu_id = int(cuda_to_vulkan[cuda_index])
            collectors.append(
                CollectorPlan(
                    collector_id=collector.collector_id,
                    node_rank=collector.node_rank,
                    local_id=collector.local_id,
                    device=collector.device,
                    host=collector.host,
                    port=collector.port,
                    traffic_manager_port=collector.traffic_manager_port,
                    traffic_manager_seed=collector.traffic_manager_seed,
                    carla_gpu_id=gpu_id,
                )
            )
        nodes.append(
            NodePlan(
                node_rank=node.node_rank,
                collectors=collectors,
                learners=list(node.learners),
            )
        )
    return TopologyPlan(
        distributed=plan.distributed,
        num_nodes=plan.num_nodes,
        total_collectors=plan.total_collectors,
        min_active_collectors=plan.min_active_collectors,
        nodes=nodes,
    )


def _cuda_index_from_device(device: str) -> int:
    text = str(device).strip().lower()
    if text == "cuda":
        return 0
    if text.startswith("cuda:"):
        return int(text.split(":", 1)[1])
    raise ValueError(f"CARLA GPU auto mapping requires a cuda device, got {device!r}")


def topology_from_file(path: str | Path) -> TopologyPlan:
    import json

    with open(path, "r", encoding="utf-8") as f:
        return TopologyPlan.from_dict(json.load(f))
