"""RL Leaderboard integration with runtime-dependent exports loaded lazily."""

from __future__ import annotations


def __getattr__(name: str):
    if name in {"RLLeaderboardAgent", "get_entry_point"}:
        from .agent import RLLeaderboardAgent, get_entry_point

        return {
            "RLLeaderboardAgent": RLLeaderboardAgent,
            "get_entry_point": get_entry_point,
        }[name]
    raise AttributeError(name)

__all__ = ["RLLeaderboardAgent", "get_entry_point"]
