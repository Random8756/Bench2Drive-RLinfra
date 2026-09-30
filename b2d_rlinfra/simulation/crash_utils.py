"""Crash-payload refinement helpers.

Classifies CARLA worker crashes into a small set of canonical reasons used
by the env pool to decide whether to restart, discard the episode, or
escalate.
"""
from typing import Tuple

__layer__ = (3, "Simulation")


def _normalize_text(value: str, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _contains_spawn_collision(detail: str) -> bool:
    detail_lower = detail.lower()
    return "spawn failed" in detail_lower or "collision at spawn" in detail_lower


def refine_crash_payload(
    crash_reason: str,
    crash_type: str,
    crash_detail: str,
) -> Tuple[str, str, str]:
    """
    Refine coarse crash typing into a more faithful root-cause label.

    `crash_reason` remains the high-level bucket used by summaries.
    `crash_type` is allowed to become more specific based on the actual detail text.
    """
    reason = _normalize_text(crash_reason, "unknown")
    detail = _normalize_text(crash_detail, reason)
    crash_type = _normalize_text(crash_type, "unknown")
    detail_lower = detail.lower()

    if reason == "skip_setup_error":
        if _contains_spawn_collision(detail):
            crash_type = "route_scenario_setup_spawn_collision"
        elif "no traffic lights" in detail_lower:
            crash_type = "route_scenario_setup_missing_traffic_lights"
        elif "intersection point" in detail_lower:
            crash_type = "route_scenario_setup_intersection_point_missing"
        elif "end position" in detail_lower:
            crash_type = "route_scenario_setup_end_position_missing"
        elif "proper plan" in detail_lower:
            crash_type = "route_scenario_setup_plan_failed"
        elif "list index out of range" in detail_lower:
            crash_type = "route_scenario_setup_internal_index_error"

    elif reason == "initial_reset_failed":
        if crash_type == "scenario_manager_reset_failed" and _contains_spawn_collision(detail):
            crash_type = "scenario_manager_reset_spawn_collision"
        elif crash_type == "scenario_creation_failed" and _contains_spawn_collision(detail):
            crash_type = "scenario_creation_spawn_collision"
        elif crash_type == "scenario_load_failed" and _contains_spawn_collision(detail):
            crash_type = "scenario_load_spawn_collision"

    elif reason == "simulation_crashed":
        if crash_type == "actor_destroyed":
            if detail.startswith("Scenario manager tick failed:") and _contains_spawn_collision(detail):
                crash_type = "scenario_tick_spawn_collision"
            elif "failed to check ego alive status" in detail_lower:
                crash_type = "ego_actor_check_failed"
            else:
                crash_type = "actor_handle_invalid"

    elif reason == "step_error":
        if "waiting for the simulator" in detail_lower and ("time-out" in detail_lower or "timeout" in detail_lower):
            crash_type = "simulator_step_timeout"
        elif "invalid world snapshot" in detail_lower or "server not ready" in detail_lower:
            crash_type = "server_not_ready"

    elif reason == "server_connection_error":
        if "timeout waiting for carla server" in detail_lower:
            crash_type = "server_start_timeout"
        elif "server not available" in detail_lower:
            crash_type = "server_not_available"

    return reason, crash_type, detail
