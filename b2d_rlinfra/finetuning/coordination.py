"""File-backed coordination primitives for rl_finetune."""

from __future__ import annotations

import json
import logging
import os
import queue
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Set, Tuple

logger = logging.getLogger("RLFinetune.Coordination")

_EVENT_RE = re.compile(r"^(?P<seq>\d{6})_(?P<type>[A-Za-z0-9_]+)\.json$")


def utc_timestamp() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    final_path = Path(path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = final_path.with_name(f"{final_path.name}.tmp.{os.getpid()}.{threading.get_ident()}")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(dict(payload), f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, final_path)
    fsync_parent_best_effort(final_path.parent)
    return final_path


def fsync_parent_best_effort(path: Path) -> None:
    if os.name == "nt":
        return
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def atomic_torch_save(path: str | Path, payload: Mapping[str, Any]) -> Path:
    import torch

    final_path = Path(path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = final_path.with_name(f"{final_path.name}.tmp.{os.getpid()}")
    with open(tmp, "wb") as f:
        torch.save(dict(payload), f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, final_path)
    fsync_parent_best_effort(final_path.parent)
    return final_path


class QueueEventBus:
    """Process-local event bus backed by multiprocessing.Queue."""

    def __init__(self, event_queue: Any):
        self.event_queue = event_queue

    def emit(self, event: Mapping[str, Any]) -> None:
        self.event_queue.put(dict(event))

    def poll(self, timeout: float) -> List[Dict[str, Any]]:
        try:
            first = self.event_queue.get(timeout=timeout)
        except queue.Empty:
            return []
        events = [dict(first)]
        while True:
            try:
                events.append(dict(self.event_queue.get_nowait()))
            except queue.Empty:
                return events


class FileEventBus:
    """Shared-filesystem event bus.

    Writers publish one immutable JSON event per collector sequence number.
    Readers scan final JSON files, validate the envelope, and remember processed
    ``(collector_id, seq)`` pairs for the current process.
    """

    def __init__(self, events_dir: str | Path, *, collector_id: Optional[int] = None):
        self.events_dir = Path(events_dir)
        self.events_dir.mkdir(parents=True, exist_ok=True)
        self.collector_id = None if collector_id is None else int(collector_id)
        self._seq = 0
        self._seen: Set[Tuple[int, int]] = set()

    def emit(self, event: Mapping[str, Any]) -> None:
        payload = dict(event.get("payload") or {})
        collector_id = self.collector_id
        if collector_id is None:
            collector_id = int(payload.get("collector_id", -1))
        if collector_id < 0:
            raise ValueError("FileEventBus.emit requires collector_id in the payload or constructor")
        event_type = str(event.get("type") or "")
        if not event_type:
            raise ValueError("FileEventBus.emit requires an event type")
        self._seq += 1
        envelope = {
            "schema_version": 1,
            "type": event_type,
            "collector_id": int(collector_id),
            "seq": int(self._seq),
            "emitted_at": utc_timestamp(),
            "payload": payload,
        }
        collector_dir = self.events_dir / f"collector_{int(collector_id):03d}"
        final_path = collector_dir / f"{self._seq:06d}_{event_type}.json"
        atomic_write_json(final_path, envelope)

    def poll(self, timeout: float) -> List[Dict[str, Any]]:
        deadline = time.monotonic() + max(0.0, float(timeout))
        while True:
            events = self._scan_once()
            if events or timeout <= 0:
                return events
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return []
            time.sleep(min(0.1, remaining))

    def _scan_once(self) -> List[Dict[str, Any]]:
        files = [path for path in self.events_dir.glob("collector_*/*.json") if ".tmp." not in path.name]
        envelopes: Dict[Tuple[int, int], Path] = {}
        pending: List[Dict[str, Any]] = []
        for path in sorted(files):
            envelope = self._load_envelope(path)
            key = (int(envelope["collector_id"]), int(envelope["seq"]))
            previous = envelopes.get(key)
            if previous is not None and previous != path:
                raise RuntimeError(f"Duplicate FileEventBus event {key}: {previous} and {path}")
            envelopes[key] = path
            if key in self._seen:
                continue
            pending.append(envelope)
        pending.sort(key=lambda item: (int(item["collector_id"]), int(item["seq"])))
        for envelope in pending:
            self._seen.add((int(envelope["collector_id"]), int(envelope["seq"])))
        return [{"type": item["type"], "payload": dict(item["payload"])} for item in pending]

    @staticmethod
    def _load_envelope(path: Path) -> Dict[str, Any]:
        match = _EVENT_RE.match(path.name)
        if not match:
            raise RuntimeError(f"Invalid FileEventBus event filename: {path}")
        parent = path.parent.name
        if not parent.startswith("collector_"):
            raise RuntimeError(f"Invalid FileEventBus collector directory: {path.parent}")
        path_collector = int(parent.split("_", 1)[1])
        path_seq = int(match.group("seq"))
        path_type = match.group("type")
        try:
            with open(path, "r", encoding="utf-8") as f:
                envelope = json.load(f)
        except Exception as exc:
            raise RuntimeError(f"Failed to parse FileEventBus event {path}: {exc}") from exc
        if int(envelope.get("schema_version", 0)) != 1:
            raise RuntimeError(f"Unsupported FileEventBus schema in {path}")
        if int(envelope.get("collector_id", -1)) != path_collector:
            raise RuntimeError(f"FileEventBus collector_id mismatch in {path}")
        if int(envelope.get("seq", -1)) != path_seq:
            raise RuntimeError(f"FileEventBus seq mismatch in {path}")
        if str(envelope.get("type", "")) != path_type:
            raise RuntimeError(f"FileEventBus type mismatch in {path}")
        if not isinstance(envelope.get("payload"), dict):
            raise RuntimeError(f"FileEventBus payload must be an object in {path}")
        return envelope


class FileControlPlane:
    FINAL_STATES = {"finished", "failed"}
    STOP_STATES = {"stopping", "finished", "failed"}
    TERMINAL_STATES = FINAL_STATES

    def __init__(self, control_dir: str | Path):
        self.control_dir = Path(control_dir)
        self.control_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.control_dir / "run_state.json"
        self.heartbeat_path = self.control_dir / "learner_heartbeat.json"
        self._heartbeat_seq = 0

    def publish_state(self, state: str, *, reason: str = "", extra: Optional[Mapping[str, Any]] = None) -> None:
        next_state = str(state)
        payload = self.read_state(default={})
        current_state = str(payload.get("state", "")).lower()
        if not self._can_transition(current_state, next_state.lower()):
            return
        payload.update(dict(extra or {}))
        payload.update(
            {
                "state": next_state,
                "reason": str(reason or payload.get("reason", "")),
                "updated_at": utc_timestamp(),
            }
        )
        atomic_write_json(self.state_path, payload)

    def heartbeat(self, *, phase: str = "running", extra: Optional[Mapping[str, Any]] = None) -> None:
        payload = self.read_heartbeat(default={})
        self._heartbeat_seq = int(payload.get("learner_heartbeat_seq", self._heartbeat_seq)) + 1
        payload.update(dict(extra or {}))
        payload.update(
            {
                "learner_phase": str(phase),
                "learner_heartbeat_seq": self._heartbeat_seq,
                "learner_heartbeat_at": utc_timestamp(),
                "updated_at": utc_timestamp(),
            }
        )
        atomic_write_json(self.heartbeat_path, payload)

    def read_state(self, *, default: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        if not self.state_path.exists():
            return dict(default or {})
        try:
            with open(self.state_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            logger.warning("Failed to read control state %s: %s", self.state_path, exc)
            return dict(default or {})
        return dict(data or {})

    def read_heartbeat(self, *, default: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        if not self.heartbeat_path.exists():
            return dict(default or {})
        try:
            with open(self.heartbeat_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            logger.warning("Failed to read learner heartbeat %s: %s", self.heartbeat_path, exc)
            return dict(default or {})
        return dict(data or {})

    def should_stop(self) -> bool:
        state = str(self.read_state(default={}).get("state", "")).lower()
        return state in self.STOP_STATES

    def request_stop(self, *, reason: str = "stop_requested") -> None:
        self.publish_state("stopping", reason=reason)

    @classmethod
    def _can_transition(cls, current_state: str, next_state: str) -> bool:
        if not current_state:
            return True
        if current_state == "finished":
            return False
        if current_state == "failed":
            return False
        if current_state == "stopping":
            return next_state == "failed"
        return True


def read_control_state(control_plane: Any) -> Dict[str, Any]:
    if control_plane is None or not hasattr(control_plane, "read_state"):
        return {}
    return dict(control_plane.read_state(default={}) or {})


def publish_coordinator_exit_state(
    control_plane: Any,
    *,
    completed: bool,
    stop_requested: bool = False,
) -> None:
    if completed:
        control_plane.publish_state("finished", reason="coordinator_exit")
    elif stop_requested:
        control_plane.publish_state("stopping", reason="coordinator_exit")


class HeartbeatWriter:
    def __init__(
        self,
        path: str | Path,
        *,
        interval: float,
        payload_factory: Callable[[int], Mapping[str, Any]],
    ):
        self.path = Path(path)
        self.interval = max(0.1, float(interval))
        self.payload_factory = payload_factory
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._seq = 0

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name=f"heartbeat:{self.path.name}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=min(5.0, self.interval + 1.0))
            self._thread = None

    def write_once(self) -> None:
        self._seq += 1
        payload = dict(self.payload_factory(self._seq))
        payload.setdefault("seq", self._seq)
        payload.setdefault("updated_at", utc_timestamp())
        atomic_write_json(self.path, payload)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.write_once()
            except Exception as exc:
                logger.warning("Heartbeat write failed for %s: %s", self.path, exc)
            self._stop.wait(self.interval)


class HeartbeatMonitor:
    """Track heartbeat freshness using local observation time, not remote clocks."""

    def __init__(self, paths: Iterable[str | Path]):
        self.paths = [Path(path) for path in paths]
        self._last_seq: Dict[Path, int] = {}
        self._last_seen: Dict[Path, float] = {}

    def refresh(self) -> None:
        now = time.monotonic()
        for path in self.paths:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                seq = int(data.get("seq", -1))
            except Exception:
                continue
            if seq > self._last_seq.get(path, -1):
                self._last_seq[path] = seq
                self._last_seen[path] = now

    def alive_paths(self, *, timeout: float) -> Set[Path]:
        self.refresh()
        now = time.monotonic()
        return {
            path
            for path, last_seen in self._last_seen.items()
            if now - last_seen <= float(timeout)
        }
