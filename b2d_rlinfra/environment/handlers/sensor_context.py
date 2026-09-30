"""Shared CARLA sensor observation context.

This module is intentionally dependency-light so model integrations can read
sensor packets without importing Gym observation handlers.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

SENSOR_CONTEXT_ATTR = "_carla_sensor_obs_context"


def get_sensor_context(provider: Any) -> Dict[str, Dict[str, Any]]:
    if provider is None:
        return {}
    context = getattr(provider, SENSOR_CONTEXT_ATTR, None)
    if not isinstance(context, dict):
        context = {}
        setattr(provider, SENSOR_CONTEXT_ATTR, context)
    return context


def publish_sensor_packet(
    provider: Any,
    sensor_id: str,
    sensor_type: str,
    frame: int,
    data: Any,
) -> Dict[str, Any]:
    packet = {
        "type": str(sensor_type),
        "frame": int(frame),
        "data": data,
    }
    if provider is not None:
        get_sensor_context(provider)[str(sensor_id)] = packet
    return packet


def get_sensor_packet(sensor_id: str, provider: Any) -> Optional[Dict[str, Any]]:
    if provider is None:
        return None
    context = getattr(provider, SENSOR_CONTEXT_ATTR, None)
    if not isinstance(context, dict):
        return None
    packet = context.get(str(sensor_id))
    return packet if isinstance(packet, dict) else None


def clear_sensor_packet(provider: Any, sensor_id: str) -> None:
    if provider is None:
        return
    context = getattr(provider, SENSOR_CONTEXT_ATTR, None)
    if isinstance(context, dict):
        context.pop(str(sensor_id), None)
