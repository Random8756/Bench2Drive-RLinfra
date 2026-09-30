#!/usr/bin/env python3
"""Rebuild Bench2Drive group statistics from a completed run.

The source of truth is the top-level ``statistics.json``.

Usage:
    python3 b2d_rlinfra/evaluation/leaderboard/summarize_b2d_groups.py \
        --run-dir <full_eval_run_dir> \
        --routes resources/routes/bench2drive_fix.xml

Dry run:
    python3 b2d_rlinfra/evaluation/leaderboard/summarize_b2d_groups.py \
        --run-dir <full_eval_run_dir> \
        --routes resources/routes/bench2drive_fix.xml \
        --dry-run

Default output:
    <run-dir>/group_results/b2d_abc/

The output directory contains per-group v2.1/v2.0 statistics, per-group summaries, one aggregate summary, and a group_manifest.json.
"""
from __future__ import annotations

import argparse
import copy
import importlib
import importlib.util
import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import xml.etree.ElementTree as ET


GROUP_PRESETS = {
    "b2d_abc": {
        "groups": {
            "a": [
                "ControlLoss",
                "DynamicObjectCrossing",
                "HardBreakRoute",
                "InvadingTurn",
                "MergerIntoSlowTraffic",
                "ParkingCrossingPedestrian",
                "ParkingCutIn",
                "T_Junction",
                "VanillaNonSignalizedTurn",
                "VanillaNonSignalizedTurnEncounterStopsign",
                "VanillaSignalizedTurnEncounterGreenLight",
                "VanillaSignalizedTurnEncounterRedLight",
            ],
            "b": [
                "Accident",
                "AccidentTwoWays",
                "BlockedIntersection",
                "ConstructionObstacle",
                "ConstructionObstacleTwoWays",
                "HazardAtSideLane",
                "HazardAtSideLaneTwoWays",
                "ParkedObstacle",
                "ParkedObstacleTwoWays",
                "VehicleOpensDoorTwoWays",
            ],
            "c": [
                "CrossingBicycleFlow",
                "EnterActorFlow",
                "HighwayCutIn",
                "HighwayExit",
                "InterurbanActorFlow",
                "InterurbanAdvancedActorFlow",
                "MergerIntoSlowTrafficV2",
                "NonSignalizedJunctionLeftTurn",
                "NonSignalizedJunctionLeftTurnEnterFlow",
                "NonSignalizedJunctionRightTurn",
                "OppositeVehicleRunningRedLight",
                "OppositeVehicleTakingPriority",
                "ParkingExit",
                "PedestrianCrossing",
                "SequentialLaneChange",
                "SignalizedJunctionLeftTurn",
                "SignalizedJunctionLeftTurnEnterFlow",
                "SignalizedJunctionRightTurn",
                "StaticCutIn",
                "VehicleTurningRoute",
                "VehicleTurningRoutePedestrian",
                "YieldToEmergencyVehicle",
            ],
        },
    },
}
DEFAULT_GROUP_PRESET = "b2d_abc"


_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parents[2]
_DEFAULT_ROUTES_FILE = _PROJECT_ROOT / "resources" / "routes" / "bench2drive_fix.xml"
_SCORE_VERSIONS_PATH = _THIS_DIR / "score_versions.py"
_ROUTE_RECORD_ID_RE = re.compile(r"^RouteScenario_(.+)_rep(\d+)$")
_B2D_SUCCESS_RULE = (
    "B2D: status in {Completed, Perfect} and no infractions except min_speed_infractions"
)
_MISSING_ROUTE_STATUSES = {
    "Failed - Missing route result after retries",
    "Failed - Missing shard result",
}


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _load_json(path: Path) -> Dict[str, Any]:
    with open(str(path), "r", encoding="utf-8") as fd:
        return json.load(fd)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path), "w", encoding="utf-8") as fd:
        json.dump(payload, fd, indent=2, sort_keys=True)
        fd.write("\n")


def _load_score_versions():
    module_name = "rl_leaderboard_score_versions"
    spec = importlib.util.spec_from_file_location(module_name, str(_SCORE_VERSIONS_PATH))
    if spec is None or spec.loader is None:
        raise ImportError("Unable to load score_versions.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_statistics_manager_symbols():
    runtime_root = _PROJECT_ROOT / "vendor" / "carla" / "evaluation-runtime"
    leaderboard_root = runtime_root / "leaderboard"
    scenario_runner_root = runtime_root / "scenario_runner"
    for path in (str(_PROJECT_ROOT), str(scenario_runner_root), str(leaderboard_root)):
        if path not in sys.path:
            sys.path.insert(0, path)

    module = importlib.import_module("leaderboard.utils.statistics_manager")
    return module.StatisticsManager, module.to_route_record


_SCORE_VERSIONS = _load_score_versions()
build_v20_statistics_from_payload = _SCORE_VERSIONS.build_v20_statistics_from_payload
extract_summary_metrics_from_statistics_payload = (
    _SCORE_VERSIONS.extract_summary_metrics_from_statistics_payload
)


def _extract_route_identity(route_record_id: Any) -> Tuple[str, Optional[int]]:
    raw = str(route_record_id or "").strip()
    match = _ROUTE_RECORD_ID_RE.match(raw)
    if match:
        return match.group(1), int(match.group(2))
    if raw.startswith("RouteScenario_"):
        return raw[len("RouteScenario_") :], None
    return raw, None


def _unique_preserve_order(values: Iterable[Any]) -> List[Any]:
    seen = set()
    result = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _resolve_existing_path(raw_path: Optional[Any], bases: Sequence[Path]) -> Optional[Path]:
    if raw_path is None:
        return None

    text = str(raw_path).strip()
    if not text:
        return None

    candidate = Path(text).expanduser()
    candidates = [candidate] if candidate.is_absolute() else [base / candidate for base in bases]
    for path in candidates:
        if path.exists():
            return path.resolve()
    return None


def _resolve_routes_file(
    *,
    args_routes: Optional[Path],
    manifest: Optional[Mapping[str, Any]],
    run_dir: Path,
) -> Path:
    bases = [run_dir, _PROJECT_ROOT]
    if args_routes is not None:
        path = _resolve_existing_path(args_routes, bases)
        if path is None:
            raise FileNotFoundError(f"Routes file not found: {args_routes}")
        return path

    manifest_path = (manifest or {}).get("routes_file")
    path = _resolve_existing_path(manifest_path, bases)
    if path is not None:
        return path

    if _DEFAULT_ROUTES_FILE.exists():
        return _DEFAULT_ROUTES_FILE.resolve()
    raise FileNotFoundError(f"Default routes file not found: {_DEFAULT_ROUTES_FILE}")


def _parse_route_xml(routes_file: Path) -> Dict[str, Dict[str, Any]]:
    tree = ET.parse(str(routes_file))
    root = tree.getroot()
    route_metadata: Dict[str, Dict[str, Any]] = {}

    for route in root.findall("route"):
        route_id = str(route.attrib.get("id", "")).strip()
        if not route_id:
            continue
        scenarios = route.findall(".//scenario")
        scenario_type = (
            str(scenarios[0].attrib.get("type", "Unknown") or "Unknown")
            if scenarios else "Unknown"
        )
        route_metadata[route_id] = {
            "route_id": route_id,
            "town": str(route.attrib.get("town", "Unknown") or "Unknown"),
            "scenario_type": scenario_type,
            "artifacts": {},
            "paths": {},
        }

    if not route_metadata:
        raise ValueError(f"No <route> entries found in {routes_file}")
    return route_metadata


def _record_id_from_job(job: Mapping[str, Any]) -> str:
    record_id = str(job.get("record_id") or "").strip()
    if record_id:
        return record_id

    route_id = str(job.get("route_id") or "").strip()
    repetition_index = int(job.get("repetition_index", 0) or 0)
    if not route_id:
        return ""
    return "RouteScenario_{}_rep{}".format(route_id, repetition_index)


def _metadata_from_manifest(
    manifest: Mapping[str, Any],
) -> Dict[str, Dict[str, Any]]:
    metadata: Dict[str, Dict[str, Any]] = {}
    for job in manifest.get("jobs", []) or []:
        if not isinstance(job, Mapping):
            continue

        record_id = _record_id_from_job(job)
        if not record_id:
            continue

        route_id = str(job.get("route_id") or "").strip()
        if not route_id:
            route_id, _ = _extract_route_identity(record_id)

        metadata[record_id] = {
            "record_id": record_id,
            "route_id": route_id,
            "repetition_index": job.get("repetition_index"),
            "town": str(job.get("town", "Unknown") or "Unknown"),
            "scenario_type": str(job.get("scenario_type", "Unknown") or "Unknown"),
            "artifacts": dict(job.get("final_artifacts", {}) or {}),
            "paths": dict(job.get("paths", {}) or {}),
            "result_source": job.get("result_source"),
            "final_record_found": job.get("final_record_found"),
            "status": job.get("status"),
        }

    if not metadata:
        raise ValueError("manifest.json does not contain any usable jobs")
    return metadata


def _metadata_from_routes_for_records(
    records: Sequence[Mapping[str, Any]],
    route_metadata: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    metadata: Dict[str, Dict[str, Any]] = {}
    for record in records:
        record_id = str(record.get("route_id") or "").strip()
        if not record_id:
            raise ValueError("Found a statistics record without route_id")

        route_id, repetition_index = _extract_route_identity(record_id)
        route_info = route_metadata.get(route_id)
        if route_info is None:
            raise ValueError(
                "Unable to resolve route_id {} from record {} in routes XML".format(
                    route_id, record_id
                )
            )

        metadata[record_id] = {
            "record_id": record_id,
            "route_id": route_id,
            "repetition_index": repetition_index,
            "town": route_info.get("town", "Unknown"),
            "scenario_type": route_info.get("scenario_type", "Unknown"),
            "artifacts": {},
            "paths": {},
            "result_source": None,
            "final_record_found": None,
            "status": None,
        }
    return metadata


def _invert_group_preset(preset_name: str) -> Dict[str, str]:
    if preset_name not in GROUP_PRESETS:
        raise ValueError(
            "Unknown preset '{}'. Available presets: {}".format(
                preset_name, ", ".join(sorted(GROUP_PRESETS))
            )
        )

    scenario_to_group: Dict[str, str] = {}
    duplicates: List[str] = []
    groups = GROUP_PRESETS[preset_name]["groups"]
    for group_name, scenario_types in groups.items():
        for scenario_type in scenario_types:
            previous = scenario_to_group.get(scenario_type)
            if previous is not None and previous != group_name:
                duplicates.append(scenario_type)
            scenario_to_group[scenario_type] = group_name

    if duplicates:
        raise ValueError(
            "Preset '{}' assigns scenario types to multiple groups: {}".format(
                preset_name, ", ".join(sorted(set(duplicates)))
            )
        )
    return scenario_to_group


def _validate_group_coverage(
    *,
    preset_name: str,
    scenario_to_group: Mapping[str, str],
    source_scenario_types: Iterable[str],
) -> None:
    missing = sorted(set(source_scenario_types) - set(scenario_to_group))
    if missing:
        raise ValueError(
            "Preset '{}' is missing scenario types: {}".format(
                preset_name, ", ".join(missing)
            )
        )


def _is_missing_record(record: Mapping[str, Any]) -> bool:
    return str(record.get("status", "")).strip() in _MISSING_ROUTE_STATUSES


def _route_record_is_b2d_success(record: Optional[Mapping[str, Any]]) -> bool:
    if not record:
        return False

    status = str(record.get("status", "")).strip()
    if status not in ("Completed", "Perfect"):
        return False

    infractions = record.get("infractions", {}) or {}
    for key, value in infractions.items():
        if key == "min_speed_infractions":
            continue
        if value:
            return False
    return True


def _build_success_statistics(
    records: Sequence[Mapping[str, Any]],
    metadata_by_record_id: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    planned_route_count = len(records)
    success_count = 0
    scenario_buckets: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"success_count": 0, "total_count": 0}
    )

    for record in records:
        record_id = str(record.get("route_id") or "").strip()
        scenario_type = str(
            (metadata_by_record_id.get(record_id) or {}).get("scenario_type") or "Unknown"
        )
        is_success = _route_record_is_b2d_success(record)
        scenario_bucket = scenario_buckets[scenario_type]
        scenario_bucket["total_count"] += 1
        if is_success:
            success_count += 1
            scenario_bucket["success_count"] += 1

    scenario_success_summary = []
    for scenario_type in sorted(scenario_buckets):
        bucket = scenario_buckets[scenario_type]
        total_count = int(bucket["total_count"])
        scenario_success_summary.append({
            "scenario_type": scenario_type,
            "success_count": int(bucket["success_count"]),
            "total_count": total_count,
            "success_rate": (
                float(bucket["success_count"]) / float(total_count)
                if total_count > 0 else 0.0
            ),
        })

    return {
        "success_rule": _B2D_SUCCESS_RULE,
        "planned_route_count": planned_route_count,
        "success_count": success_count,
        "non_success_count": planned_route_count - success_count,
        "success_rate": (
            float(success_count) / float(planned_route_count)
            if planned_route_count > 0 else 0.0
        ),
        "scenario_success_summary": scenario_success_summary,
    }


def _load_records(statistics_payload: Mapping[str, Any]) -> List[Dict[str, Any]]:
    checkpoint = statistics_payload.get("_checkpoint", {}) or {}
    records = checkpoint.get("records", []) or []
    if not isinstance(records, list) or not records:
        raise ValueError("statistics.json does not contain _checkpoint.records")

    result: List[Dict[str, Any]] = []
    seen = set()
    duplicates = []
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("statistics.json contains a non-object route record")
        record_id = str(record.get("route_id") or "").strip()
        if not record_id:
            raise ValueError("statistics.json contains a record without route_id")
        if record_id in seen:
            duplicates.append(record_id)
        seen.add(record_id)
        result.append(dict(record))

    if duplicates:
        raise ValueError(
            "statistics.json contains duplicate route records: {}".format(
                ", ".join(sorted(set(duplicates)))
            )
        )
    return result


def _build_group_records(
    *,
    records: Sequence[Mapping[str, Any]],
    metadata_by_record_id: Mapping[str, Mapping[str, Any]],
    scenario_to_group: Mapping[str, str],
    preset_name: str,
) -> Dict[str, List[Dict[str, Any]]]:
    groups = GROUP_PRESETS[preset_name]["groups"]
    group_records: Dict[str, List[Dict[str, Any]]] = {
        group_name: [] for group_name in groups
    }

    missing_metadata = []
    ungrouped_types = []
    for record in records:
        record_id = str(record.get("route_id") or "").strip()
        metadata = metadata_by_record_id.get(record_id)
        if metadata is None:
            missing_metadata.append(record_id)
            continue

        scenario_type = str(metadata.get("scenario_type") or "Unknown")
        group_name = scenario_to_group.get(scenario_type)
        if group_name is None:
            ungrouped_types.append(scenario_type)
            continue
        group_records[group_name].append(dict(record))

    if missing_metadata:
        raise ValueError(
            "No metadata found for statistics records: {}".format(
                ", ".join(sorted(missing_metadata))
            )
        )
    if ungrouped_types:
        raise ValueError(
            "Preset '{}' does not group scenario types present in records: {}".format(
                preset_name, ", ".join(sorted(set(ungrouped_types)))
            )
        )

    empty_groups = [name for name, group in group_records.items() if not group]
    if empty_groups:
        raise ValueError(
            "Preset '{}' produced empty groups: {}".format(
                preset_name, ", ".join(sorted(empty_groups))
            )
        )
    return group_records


def _record_metadata_list(
    record_ids: Sequence[str],
    metadata_by_record_id: Mapping[str, Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    items = []
    for record_id in record_ids:
        metadata = metadata_by_record_id.get(record_id) or {}
        items.append({
            "record_id": record_id,
            "route_id": metadata.get("route_id"),
            "repetition_index": metadata.get("repetition_index"),
            "town": metadata.get("town"),
            "scenario_type": metadata.get("scenario_type"),
            "artifacts": dict(metadata.get("artifacts", {}) or {}),
            "paths": dict(metadata.get("paths", {}) or {}),
            "result_source": metadata.get("result_source"),
        })
    return items


def _aggregate_group_statistics(
    *,
    group_records: Sequence[Mapping[str, Any]],
    sensors: Sequence[Any],
    statistics_path: Path,
) -> Dict[str, Any]:
    StatisticsManager, to_route_record = _load_statistics_manager_symbols()
    live_results_path = statistics_path.with_suffix(".live_results.txt")
    manager = StatisticsManager(str(statistics_path), str(live_results_path))
    manager._results.checkpoint.records = [
        to_route_record(copy.deepcopy(dict(record))) for record in group_records
    ]
    manager.save_progress(len(group_records), len(group_records))
    manager.save_sensors(list(sensors or []))
    manager.sort_records()
    manager.compute_global_statistics()
    if any(_is_missing_record(record) for record in group_records):
        manager.save_entry_status("Crashed")
    manager.write_statistics()
    return _load_json(statistics_path)


def _build_score_version_node(
    *,
    metrics: Mapping[str, Any],
    success_stats: Mapping[str, Any],
    statistics_path: Path,
) -> Dict[str, Any]:
    return {
        "status": "ok",
        "entry_status": metrics.get("entry_status"),
        "eligible": metrics.get("eligible"),
        "score_mean": metrics.get("score_mean"),
        "route_completion_mean": metrics.get("route_completion_mean"),
        "score_penalty_mean": metrics.get("score_penalty_mean"),
        "success_rate": success_stats.get("success_rate"),
        "success_count": success_stats.get("success_count"),
        "planned_route_count": success_stats.get("planned_route_count"),
        "collisions_pedestrian": metrics.get("collisions_pedestrian"),
        "collisions_vehicle": metrics.get("collisions_vehicle"),
        "min_speed_infractions": metrics.get("min_speed_infractions"),
        "statistics": str(statistics_path),
    }


def _build_group_summary(
    *,
    preset_name: str,
    group_name: str,
    group_records: Sequence[Mapping[str, Any]],
    metadata_by_record_id: Mapping[str, Mapping[str, Any]],
    statistics_path: Path,
    statistics_v20_path: Path,
    payload_v21: Mapping[str, Any],
    payload_v20: Mapping[str, Any],
) -> Dict[str, Any]:
    record_ids = [str(record.get("route_id") or "").strip() for record in group_records]
    route_ids = _unique_preserve_order(
        (metadata_by_record_id.get(record_id) or {}).get("route_id") for record_id in record_ids
    )
    missing_record_ids = [
        str(record.get("route_id") or "").strip()
        for record in group_records
        if _is_missing_record(record)
    ]
    success_stats = _build_success_statistics(group_records, metadata_by_record_id)
    metrics_v21 = extract_summary_metrics_from_statistics_payload(payload_v21)
    metrics_v20 = extract_summary_metrics_from_statistics_payload(payload_v20)

    return {
        "status": "failed" if missing_record_ids else "completed",
        "preset": preset_name,
        "group": group_name,
        "scenario_types": list(GROUP_PRESETS[preset_name]["groups"][group_name]),
        "route_ids": route_ids,
        "record_ids": record_ids,
        "record_count": len(record_ids),
        "planned_record_count": len(record_ids),
        "reported_record_count": len(record_ids) - len(missing_record_ids),
        "missing_record_count": len(missing_record_ids),
        "missing_record_ids": missing_record_ids,
        "entry_status": metrics_v21.get("entry_status"),
        "eligible": metrics_v21.get("eligible"),
        "success_rule": success_stats.get("success_rule"),
        "success_count": success_stats.get("success_count"),
        "non_success_count": success_stats.get("non_success_count"),
        "success_rate": success_stats.get("success_rate"),
        "scenario_success_summary": success_stats.get("scenario_success_summary"),
        "score_mean": metrics_v21.get("score_mean"),
        "route_completion_mean": metrics_v21.get("route_completion_mean"),
        "score_penalty_mean": metrics_v21.get("score_penalty_mean"),
        "collisions_pedestrian": metrics_v21.get("collisions_pedestrian"),
        "collisions_vehicle": metrics_v21.get("collisions_vehicle"),
        "min_speed_infractions": metrics_v21.get("min_speed_infractions"),
        "global_record": metrics_v21.get("global_record"),
        "score_version_default": "v2.1",
        "score_versions": {
            "v2.1": _build_score_version_node(
                metrics=metrics_v21,
                success_stats=success_stats,
                statistics_path=statistics_path,
            ),
            "v2.0": _build_score_version_node(
                metrics=metrics_v20,
                success_stats=success_stats,
                statistics_path=statistics_v20_path,
            ),
        },
        "statistics": str(statistics_path),
        "statistics_v20": str(statistics_v20_path),
    }


def _build_group_manifest(
    *,
    run_dir: Path,
    statistics_path: Path,
    manifest_path: Optional[Path],
    routes_file: Path,
    preset_name: str,
    group_records: Mapping[str, Sequence[Mapping[str, Any]]],
    metadata_by_record_id: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    groups: Dict[str, Any] = {}
    for group_name, records in group_records.items():
        record_ids = [str(record.get("route_id") or "").strip() for record in records]
        route_ids = _unique_preserve_order(
            (metadata_by_record_id.get(record_id) or {}).get("route_id")
            for record_id in record_ids
        )
        groups[group_name] = {
            "scenario_types": list(GROUP_PRESETS[preset_name]["groups"][group_name]),
            "route_ids": route_ids,
            "record_ids": record_ids,
            "records": _record_metadata_list(record_ids, metadata_by_record_id),
        }

    return {
        "created_at": _now_iso(),
        "source_run_dir": str(run_dir),
        "source_statistics": str(statistics_path),
        "source_manifest": str(manifest_path) if manifest_path else None,
        "routes_file": str(routes_file),
        "preset": preset_name,
        "groups": groups,
    }


def _print_dry_run(
    *,
    preset_name: str,
    group_records: Mapping[str, Sequence[Mapping[str, Any]]],
    metadata_by_record_id: Mapping[str, Mapping[str, Any]],
) -> None:
    print("Dry run for preset '{}'".format(preset_name))
    total_records = 0
    total_missing = 0
    for group_name in GROUP_PRESETS[preset_name]["groups"]:
        records = group_records[group_name]
        record_ids = [str(record.get("route_id") or "").strip() for record in records]
        route_ids = _unique_preserve_order(
            (metadata_by_record_id.get(record_id) or {}).get("route_id")
            for record_id in record_ids
        )
        missing_count = sum(1 for record in records if _is_missing_record(record))
        total_records += len(records)
        total_missing += missing_count
        print(
            "group {}: routes={} records={} missing={} scenario_types={}".format(
                group_name,
                len(route_ids),
                len(records),
                missing_count,
                len(GROUP_PRESETS[preset_name]["groups"][group_name]),
            )
        )
    print("total: records={} missing={}".format(total_records, total_missing))


def _build_overall_summary(
    *,
    run_dir: Path,
    output_dir: Path,
    preset_name: str,
    group_summaries: Mapping[str, Mapping[str, Any]],
    group_manifest_path: Path,
) -> Dict[str, Any]:
    groups = {}
    total_records = 0
    total_missing = 0
    total_success = 0

    for group_name, summary in group_summaries.items():
        record_count = int(summary.get("planned_record_count", 0) or 0)
        missing_count = int(summary.get("missing_record_count", 0) or 0)
        success_count = int(summary.get("success_count", 0) or 0)
        total_records += record_count
        total_missing += missing_count
        total_success += success_count
        groups[group_name] = {
            "status": summary.get("status"),
            "scenario_types": summary.get("scenario_types", []),
            "record_count": record_count,
            "missing_record_count": missing_count,
            "success_count": success_count,
            "success_rate": summary.get("success_rate"),
            "score_mean": summary.get("score_mean"),
            "route_completion_mean": summary.get("route_completion_mean"),
            "score_penalty_mean": summary.get("score_penalty_mean"),
            "entry_status": summary.get("entry_status"),
            "statistics": summary.get("statistics"),
            "statistics_v20": summary.get("statistics_v20"),
        }

    return {
        "created_at": _now_iso(),
        "source_run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "preset": preset_name,
        "status": "failed" if total_missing else "completed",
        "planned_record_count": total_records,
        "reported_record_count": total_records - total_missing,
        "missing_record_count": total_missing,
        "success_rule": _B2D_SUCCESS_RULE,
        "success_count": total_success,
        "non_success_count": total_records - total_success,
        "success_rate": (
            float(total_success) / float(total_records) if total_records > 0 else 0.0
        ),
        "group_manifest": str(group_manifest_path),
        "groups": groups,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline Bench2Drive group statistics summarizer."
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--routes", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--preset", default=DEFAULT_GROUP_PRESET)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    statistics_path = run_dir / "statistics.json"
    manifest_path = run_dir / "manifest.json"

    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    if not statistics_path.exists():
        raise FileNotFoundError(f"Top-level statistics.json not found: {statistics_path}")

    manifest = _load_json(manifest_path) if manifest_path.exists() else None
    routes_file = _resolve_routes_file(
        args_routes=args.routes,
        manifest=manifest,
        run_dir=run_dir,
    )

    source_statistics = _load_json(statistics_path)
    source_records = _load_records(source_statistics)
    route_metadata = _parse_route_xml(routes_file)
    if manifest is not None:
        metadata_by_record_id = _metadata_from_manifest(manifest)
        source_scenario_types = {
            str(metadata.get("scenario_type") or "Unknown")
            for metadata in metadata_by_record_id.values()
        }
    else:
        metadata_by_record_id = _metadata_from_routes_for_records(
            source_records,
            route_metadata,
        )
        source_scenario_types = {
            str(metadata.get("scenario_type") or "Unknown")
            for metadata in route_metadata.values()
        }

    scenario_to_group = _invert_group_preset(args.preset)
    _validate_group_coverage(
        preset_name=args.preset,
        scenario_to_group=scenario_to_group,
        source_scenario_types=source_scenario_types,
    )
    group_records = _build_group_records(
        records=source_records,
        metadata_by_record_id=metadata_by_record_id,
        scenario_to_group=scenario_to_group,
        preset_name=args.preset,
    )

    if args.dry_run:
        _print_dry_run(
            preset_name=args.preset,
            group_records=group_records,
            metadata_by_record_id=metadata_by_record_id,
        )
        return 0

    base_output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else run_dir / "group_results"
    )
    output_dir = base_output_dir / args.preset
    output_dir.mkdir(parents=True, exist_ok=True)

    sensors = source_statistics.get("sensors", []) or []
    group_summaries: Dict[str, Dict[str, Any]] = {}
    for group_name, records in group_records.items():
        statistics_group_path = output_dir / "statistics_{}_{}.json".format(
            args.preset, group_name
        )
        statistics_v20_path = output_dir / "statistics_{}_{}_v20.json".format(
            args.preset, group_name
        )
        summary_path = output_dir / "summary_{}_{}.json".format(args.preset, group_name)

        payload_v21 = _aggregate_group_statistics(
            group_records=records,
            sensors=sensors,
            statistics_path=statistics_group_path,
        )
        payload_v20 = build_v20_statistics_from_payload(payload_v21)
        _write_json(statistics_v20_path, payload_v20)

        summary = _build_group_summary(
            preset_name=args.preset,
            group_name=group_name,
            group_records=records,
            metadata_by_record_id=metadata_by_record_id,
            statistics_path=statistics_group_path,
            statistics_v20_path=statistics_v20_path,
            payload_v21=payload_v21,
            payload_v20=payload_v20,
        )
        _write_json(summary_path, summary)
        group_summaries[group_name] = summary

    group_manifest_path = output_dir / "group_manifest.json"
    group_manifest = _build_group_manifest(
        run_dir=run_dir,
        statistics_path=statistics_path,
        manifest_path=manifest_path if manifest_path.exists() else None,
        routes_file=routes_file,
        preset_name=args.preset,
        group_records=group_records,
        metadata_by_record_id=metadata_by_record_id,
    )
    _write_json(group_manifest_path, group_manifest)

    summary_all_path = output_dir / "summary_{}.json".format(args.preset)
    summary_all = _build_overall_summary(
        run_dir=run_dir,
        output_dir=output_dir,
        preset_name=args.preset,
        group_summaries=group_summaries,
        group_manifest_path=group_manifest_path,
    )
    _write_json(summary_all_path, summary_all)

    print("Wrote grouped results to {}".format(output_dir))
    for group_name in GROUP_PRESETS[args.preset]["groups"]:
        summary = group_summaries[group_name]
        print(
            "group {}: records={} missing={} success_rate={:.3f} score_v21={}".format(
                group_name,
                summary["planned_record_count"],
                summary["missing_record_count"],
                summary["success_rate"],
                summary["score_mean"],
            )
        )
    print("summary: {}".format(summary_all_path))
    print("manifest: {}".format(group_manifest_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
