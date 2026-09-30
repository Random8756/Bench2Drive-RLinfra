"""Layer 1 — Scenario.

Scenario abstraction. Groups route indexing, scenario selection, and
``ScenarioManagerRL`` episode lifecycle management.

Layer entries and implementation modules
----------------------------------------
    L1_RouteIndexer                    ← leaderboard.utils.route_indexer
    L1_select_scenario_name            ← b2d_rlinfra.scenario.adaptive_route_sampler
    L1_ScenarioManagerRL               ← b2d_rlinfra.scenario.scenario_manager_rl
    L1_routes_dir                      ← <repo_root>/resources/routes
"""

from __future__ import annotations

import os as _os

from . import _PROJECT_ROOT

__layer__ = (1, "Scenario")

# Route XML loader from the Leaderboard 2.0 SDK.
from leaderboard.utils.route_indexer import RouteIndexer as L1_RouteIndexer

# Scenario sampler helpers used by training runners.
from b2d_rlinfra.scenario.adaptive_route_sampler import (
    DEFAULT_ADAPTIVE_CONFIG as L1_DEFAULT_ADAPTIVE_CONFIG,
    normalize_adaptive_config as L1_normalize_adaptive_config,
    compute_sampling_weight as L1_compute_sampling_weight,
    select_scenario_name as L1_select_scenario_name,
    select_scenario_name_for_sample as L1_select_scenario_name_for_sample,
    ensure_shared_scenarios as L1_ensure_shared_scenarios,
    update_shared_sampling_state as L1_update_shared_sampling_state,
    update_shared_sampling_state_with_periodic_snapshot as L1_update_shared_sampling_state_periodic,
)

from b2d_rlinfra.scenario.scenario_manager_rl import ScenarioManagerRL as L1_ScenarioManagerRL

# User-replaceable route files location.
L1_routes_dir: str = _os.path.join(_PROJECT_ROOT, "resources", "routes")

__all__ = [
    "L1_RouteIndexer",
    "L1_DEFAULT_ADAPTIVE_CONFIG",
    "L1_normalize_adaptive_config",
    "L1_compute_sampling_weight",
    "L1_select_scenario_name",
    "L1_select_scenario_name_for_sample",
    "L1_ensure_shared_scenarios",
    "L1_update_shared_sampling_state",
    "L1_update_shared_sampling_state_periodic",
    "L1_ScenarioManagerRL",
    "L1_routes_dir",
]
