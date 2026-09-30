"""Layer 5 — Evaluation.

Evaluation abstraction. Covers the official Leaderboard evaluation runtime
wrapper, route-level statistics aggregation, and optional training
visualisation.

Layer entries and implementation modules
----------------------------------------
    L5_LeaderboardEvalRuntime    ← b2d_rlinfra.evaluation.runtime
        Turns a trained model into a Leaderboard 2.0 / Bench2Drive
        autonomous-agent step function.

    L5_RLLeaderboardAgent        ← b2d_rlinfra.evaluation.leaderboard
        Concrete ``AutonomousAgent`` subclass loaded by the parallel
        evaluation launcher.

    L5_RLStatisticsManager       ← b2d_rlinfra.evaluation.rl_statistics_manager
        Per-route infraction and penalty bookkeeping with ``merge_results()``
        for multi-environment aggregation.

    L5_TrainingVisualizer        ← b2d_rlinfra.evaluation.visualization
        Async BEV video recorder with reward overlay for training.
"""

from __future__ import annotations

__layer__ = (5, "Evaluation")

from b2d_rlinfra.evaluation.runtime import LeaderboardEvalRuntime as L5_LeaderboardEvalRuntime

from b2d_rlinfra.evaluation.leaderboard import (
    RLLeaderboardAgent as L5_RLLeaderboardAgent,
    get_entry_point as L5_get_entry_point,
)

from b2d_rlinfra.evaluation.rl_statistics_manager import (
    RLStatisticsManager as L5_RLStatisticsManager,
    RouteRecord as L5_RouteRecord,
    EnvStatistics as L5_EnvStatistics,
    PENALTY_VALUE_DICT as L5_PENALTY_VALUE_DICT,
    PENALTY_PERC_DICT as L5_PENALTY_PERC_DICT,
    PENALTY_NAME_DICT as L5_PENALTY_NAME_DICT,
)

from b2d_rlinfra.evaluation.visualization import TrainingVisualizer as L5_TrainingVisualizer


__all__ = [
    "L5_LeaderboardEvalRuntime",
    "L5_RLLeaderboardAgent",
    "L5_get_entry_point",
    "L5_RLStatisticsManager",
    "L5_RouteRecord",
    "L5_EnvStatistics",
    "L5_PENALTY_VALUE_DICT",
    "L5_PENALTY_PERC_DICT",
    "L5_PENALTY_NAME_DICT",
    "L5_TrainingVisualizer",
]
