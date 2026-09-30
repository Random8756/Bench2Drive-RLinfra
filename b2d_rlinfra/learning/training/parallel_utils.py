#!/usr/bin/env python3
"""
Utilities shared by the RL leaderboard parallel evaluator.
"""

from __future__ import annotations

import copy
import json
import logging
import re
import subprocess
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parents[2]
_PORT_AUTO_INCREMENT = 10
logger = logging.getLogger("Training Loop")


@dataclass(frozen=True)
class ShardSlot:
    index: int
    host: str
    port: int
    traffic_manager_port: int
    traffic_manager_seed: int
    gpu_id: Any


@dataclass(frozen=True)
class EpisodeRoute:
    episode_index: int
    route_id: str
    element: ET.Element


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def sanitize_tag(text: str) -> str:
    tag = re.sub(r"[^A-Za-z0-9._-]+", "_", text or "").strip("._-")
    return tag or "run"


def resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return path


def write_json_file(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_json_if_exists(path_like: Optional[str | Path]) -> Optional[Dict[str, Any]]:
    if not path_like:
        return None
    path = resolve_path(path_like)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# CARLA slots
def _is_sequence_value(value: Any) -> bool:
    return isinstance(value, (list, tuple))


def _expand_config_value(
    *,
    key: str,
    raw_value: Any,
    count: int,
    default_value: Any,
    auto_increment: Optional[int] = None,
    fallback_base: Optional[Sequence[Any]] = None,
    fallback_offset: Optional[int] = None,
) -> List[Any]:
    if _is_sequence_value(raw_value):
        values = list(raw_value[:count])
        if len(values) < count:
            raise ValueError(
                f"carla.{key} expects at least {count} values, "
                f"but only {len(values)} were configured."
            )
        return values

    if raw_value is not None:
        if auto_increment is not None:
            return [raw_value + i * auto_increment for i in range(count)]
        return [raw_value] * count

    if fallback_base is not None and fallback_offset is not None:
        return [base + fallback_offset for base in fallback_base]

    if auto_increment is not None:
        return [default_value + i * auto_increment for i in range(count)]
    return [default_value] * count


def normalize_carla_slots(env_config: Dict[str, Any], shard_count: int) -> List[ShardSlot]:
    if shard_count <= 0:
        raise ValueError(f"num_shards must be positive, got {shard_count}")

    carla_cfg = env_config.get("carla", {})

    hosts = _expand_config_value(
        key="host",
        raw_value=carla_cfg.get("host", "localhost"),
        count=shard_count,
        default_value="localhost",
    )
    ports = _expand_config_value(
        key="port",
        raw_value=carla_cfg.get("port"),
        count=shard_count,
        default_value=2000,
        auto_increment=_PORT_AUTO_INCREMENT,
    )
    tm_ports = _expand_config_value(
        key="traffic_manager_port",
        raw_value=carla_cfg.get("traffic_manager_port"),
        count=shard_count,
        default_value=None,
        fallback_base=ports,
        fallback_offset=6000,
    )
    tm_seeds = _expand_config_value(
        key="traffic_manager_seed",
        raw_value=carla_cfg.get("traffic_manager_seed", 0),
        count=shard_count,
        default_value=0,
    )
    gpu_ids = _expand_config_value(
        key="gpu_id",
        raw_value=carla_cfg.get("gpu_id", 0),
        count=shard_count,
        default_value=0,
    )

    return [
        ShardSlot(
            index=i,
            host=str(hosts[i]),
            port=int(ports[i]),
            traffic_manager_port=int(tm_ports[i]),
            traffic_manager_seed=int(tm_seeds[i]),
            gpu_id=gpu_ids[i],
        )
        for i in range(shard_count)
    ]


def validate_parallel_slots(slots: Sequence[ShardSlot]) -> None:
    if not slots:
        raise ValueError("No CARLA slots were prepared for parallel evaluation")

    hosts = [slot.host for slot in slots]
    duplicated_hosts = sorted({host for host in hosts if hosts.count(host) > 1})
    if duplicated_hosts:
        raise ValueError(
            "Parallel eval requires unique fake bind hosts because cleanup is done by host. "
            f"Duplicated hosts: {duplicated_hosts}"
        )

    seen_rpc: set[Tuple[str, int]] = set()
    seen_tm: set[Tuple[str, int]] = set()
    for slot in slots:
        rpc_key = (slot.host, slot.port)
        if rpc_key in seen_rpc:
            raise ValueError(f"Duplicate CARLA RPC slot: host={slot.host} port={slot.port}")
        seen_rpc.add(rpc_key)

        tm_key = (slot.host, slot.traffic_manager_port)
        if tm_key in seen_tm:
            raise ValueError(
                f"Duplicate Traffic Manager slot: host={slot.host} "
                f"tm_port={slot.traffic_manager_port}"
            )
        seen_tm.add(tm_key)


# Route files
def _resolve_subset_ids(route_ids: Sequence[str], routes_subset: Any) -> Optional[set[str]]:
    if routes_subset is None:
        return None
    subset_str = str(routes_subset).strip()
    if not subset_str:
        return None

    selected_ids: set[str] = set()
    tokens = [token.strip() for token in subset_str.split(",") if token.strip()]
    for token in tokens:
        if "-" in token:
            start_id, end_id = [part.strip() for part in token.split("-", 1)]
            if not start_id or not end_id:
                raise ValueError(f"Malformed routes_subset token: '{token}'")
            try:
                start_idx = route_ids.index(start_id)
                end_idx = route_ids.index(end_id)
            except ValueError as exc:
                raise ValueError(
                    f"routes_subset token '{token}' references an unknown route id"
                ) from exc
            if start_idx > end_idx:
                raise ValueError(
                    f"routes_subset token '{token}' is out of order"
                )
            selected_ids.update(route_ids[start_idx : end_idx + 1])
        else:
            if token not in route_ids:
                raise ValueError(f"routes_subset references unknown route id: '{token}'")
            selected_ids.add(token)
    return selected_ids


def load_selected_route_elements(
    route_file: Path,
    routes_subset: Any = None,
) -> Tuple[ET.Element, List[ET.Element]]:
    tree = ET.parse(route_file)
    root = tree.getroot()
    route_elements = [copy.deepcopy(elem) for elem in root.findall("route")]
    if not route_elements:
        raise ValueError(f"No <route> entries found in {route_file}")

    route_ids = [elem.attrib.get("id", "") for elem in route_elements]
    selected_ids = _resolve_subset_ids(route_ids, routes_subset)

    if selected_ids is None:
        selected_routes = route_elements
    else:
        selected_routes = [
            elem for elem in route_elements if elem.attrib.get("id", "") in selected_ids
        ]

    if not selected_routes:
        raise ValueError("Route selection is empty after applying routes_subset")

    template_root = ET.Element(root.tag, root.attrib)
    return template_root, selected_routes


def materialize_episode_routes(
    route_elements: Sequence[ET.Element],
    repetitions: int = 1,
    n_eval_episodes: int = 0,
) -> List[EpisodeRoute]:
    if repetitions <= 0:
        raise ValueError(f"env.routes.repetitions must be positive, got {repetitions}")
    if not route_elements:
        raise ValueError("Cannot materialize episodes from an empty route list")

    cycle_templates: List[Tuple[str, ET.Element]] = []
    for route in route_elements:
        route_id = route.attrib.get("id", "")
        for _ in range(repetitions):
            cycle_templates.append((route_id, route))

    total_episodes = (
        n_eval_episodes if n_eval_episodes and n_eval_episodes > 0 else len(cycle_templates)
    )
    materialized: List[EpisodeRoute] = []
    for episode_index in range(total_episodes):
        route_id, template = cycle_templates[episode_index % len(cycle_templates)]
        materialized.append(
            EpisodeRoute(
                episode_index=episode_index,
                route_id=route_id,
                element=copy.deepcopy(template),
            )
        )
    return materialized


def split_evenly(items: Sequence[Any], shard_count: int) -> List[List[Any]]:
    if shard_count <= 0:
        raise ValueError(f"shard_count must be positive, got {shard_count}")
    if not items:
        return []

    shard_count = min(shard_count, len(items))
    base_size, extra = divmod(len(items), shard_count)
    chunks: List[List[Any]] = []
    start = 0
    for idx in range(shard_count):
        size = base_size + (1 if idx < extra else 0)
        end = start + size
        chunk = list(items[start:end])
        if chunk:
            chunks.append(chunk)
        start = end
    return chunks


def _compute_target_chunk_sizes(total_items: int, shard_count: int) -> List[int]:
    if shard_count <= 0:
        raise ValueError(f"shard_count must be positive, got {shard_count}")
    if total_items <= 0:
        return []

    shard_count = min(shard_count, total_items)
    base_size, extra = divmod(total_items, shard_count)
    return [base_size + (1 if idx < extra else 0) for idx in range(shard_count)]


def split_episodes_balanced_by_town(
    episodes: Sequence[EpisodeRoute],
    shard_count: int,
    heavy_towns: Sequence[str],
) -> List[List[EpisodeRoute]]:
    """Split episodes while spreading heavy towns as evenly as possible.

    The algorithm keeps total route counts close to the same sizes as
    ``split_evenly`` while using a greedy placement policy to spread
    heavy towns (e.g. Town11/12/13) across shards.
    """
    if shard_count <= 0:
        raise ValueError(f"shard_count must be positive, got {shard_count}")
    if not episodes:
        return []

    target_sizes = _compute_target_chunk_sizes(len(episodes), shard_count)
    normalized_heavy_towns = {
        str(town).strip().lower() for town in heavy_towns if str(town).strip()
    }
    if not normalized_heavy_towns:
        return split_evenly(episodes, shard_count)

    buckets = []
    for _ in target_sizes:
        buckets.append({
            "episodes": [],
            "size": 0,
            "heavy_total": 0,
            "heavy_by_town": {town: 0 for town in normalized_heavy_towns},
        })

    heavy_episodes: List[Tuple[str, EpisodeRoute]] = []
    light_episodes: List[EpisodeRoute] = []
    for episode in episodes:
        town = str(episode.element.attrib.get("town", "")).strip().lower()
        if town in normalized_heavy_towns:
            heavy_episodes.append((town, episode))
        else:
            light_episodes.append(episode)

    if not heavy_episodes:
        return split_evenly(episodes, shard_count)

    def _available_bucket_indexes() -> List[int]:
        return [
            idx for idx, bucket in enumerate(buckets)
            if bucket["size"] < target_sizes[idx]
        ]

    def _assign_heavy(town: str, episode: EpisodeRoute) -> None:
        candidate_indexes = _available_bucket_indexes()
        chosen_idx = min(
            candidate_indexes,
            key=lambda idx: (
                buckets[idx]["heavy_by_town"].get(town, 0),
                buckets[idx]["heavy_total"],
                buckets[idx]["size"],
                idx,
            ),
        )
        bucket = buckets[chosen_idx]
        bucket["episodes"].append(episode)
        bucket["size"] += 1
        bucket["heavy_total"] += 1
        bucket["heavy_by_town"][town] = bucket["heavy_by_town"].get(town, 0) + 1

    def _assign_light(episode: EpisodeRoute) -> None:
        candidate_indexes = _available_bucket_indexes()
        chosen_idx = min(
            candidate_indexes,
            key=lambda idx: (
                buckets[idx]["size"],
                buckets[idx]["heavy_total"],
                idx,
            ),
        )
        bucket = buckets[chosen_idx]
        bucket["episodes"].append(episode)
        bucket["size"] += 1

    for town, episode in heavy_episodes:
        _assign_heavy(town, episode)

    for episode in light_episodes:
        _assign_light(episode)

    chunks: List[List[EpisodeRoute]] = []
    for bucket in buckets:
        chunk = sorted(bucket["episodes"], key=lambda episode: episode.episode_index)
        if chunk:
            chunks.append(chunk)
    return chunks


def write_route_shard(
    template_root: ET.Element,
    episodes: Sequence[EpisodeRoute],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    root = ET.Element(template_root.tag, template_root.attrib)
    for episode in episodes:
        root.append(copy.deepcopy(episode.element))
    tree = ET.ElementTree(root)
    if hasattr(ET, "indent"):
        ET.indent(tree, space="  ")
    tree.write(output_path, encoding="utf-8", xml_declaration=True)


# Process cleanup
def kill_hosts(hosts: Iterable[str], verbose: bool = True) -> int:
    unique_hosts = [host for host in dict.fromkeys(hosts) if host]
    if not unique_hosts:
        return 0

    script_path = _PROJECT_ROOT / "tools" / "runtime" / "kill_by_host.sh"
    if not script_path.exists():
        logger.warning("kill_by_host script not found: %s", script_path)
        return 0

    killed = 0
    if verbose:
        logger.info("Killing CARLA processes by host hosts=%d", len(unique_hosts))

    for host in unique_hosts:
        logger.debug("CARLA cleanup start host=%s", host)
        try:
            result = subprocess.run(
                ["bash", str(script_path), host],
                capture_output=True,
                text=True,
            )
        except Exception as exc:
            logger.warning("CARLA cleanup failed host=%s: %s", host, exc)
            continue

        output = (result.stdout or "") + (result.stderr or "")
        killed += sum(1 for line in output.splitlines() if line.strip().startswith("PID:"))
        if output.strip():
            logger.debug("CARLA cleanup output host=%s:\n%s", host, output.rstrip())
        if result.returncode != 0:
            logger.warning("CARLA cleanup failed host=%s rc=%s", host, result.returncode)
        logger.debug("CARLA cleanup done host=%s rc=%s", host, result.returncode)

    if killed > 0:
        time.sleep(2.0)

    if verbose:
        logger.info("CARLA host cleanup finished killed=%d", killed)
    return killed
