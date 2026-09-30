"""Lightweight messages exchanged by rl finetune components."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


@dataclass
class RolloutMeta:
    collector_id: int
    episode_id: int
    file_path: str
    num_steps: int
    policy_version: int
    reward_sum: float
    terminated: bool
    truncated: bool
    route_completion_ratio: Optional[float] = None
    crashed: bool = False
    route_id: str = ""
    scenario_name: str = ""
    town: str = ""
    created_at: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        extra = data.pop("extra", {}) or {}
        data.update(extra)
        return data


@dataclass
class CrashMeta:
    collector_id: int
    episode_id: int
    policy_version: int
    reason: str
    crash_type: str = ""
    crash_detail: str = ""
    route_id: str = ""
    scenario_name: str = ""
    town: str = ""
    created_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["crashed"] = True
        return data


@dataclass
class CollectorEvent:
    type: str
    payload: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.type, "payload": dict(self.payload)}

    @classmethod
    def rollout(cls, meta: RolloutMeta) -> "CollectorEvent":
        return cls(type="rollout", payload=meta.to_dict())

    @classmethod
    def crash(cls, meta: CrashMeta) -> "CollectorEvent":
        return cls(type="crash", payload=meta.to_dict())
