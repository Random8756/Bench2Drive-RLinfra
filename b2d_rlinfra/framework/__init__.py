"""Framework abstraction-layer map for Bench2Drive-RLInfra.

The ``b2d_rlinfra/framework/`` folder follows the five-layer structure used to describe
responsibility boundaries in the codebase. Layer-prefixed names provide a
compact view of the corresponding implementation concepts.

Layer mapping
-------------
    L1  Scenario     → ``b2d_rlinfra.framework.scenario_layer``
    L2  Environment  → ``b2d_rlinfra.framework.environment_layer``
    L3  Simulation   → ``b2d_rlinfra.framework.simulation_layer``
    L4  Algorithm    → ``b2d_rlinfra.framework.algorithm_layer``
    L5  Evaluation   → ``b2d_rlinfra.framework.evaluation_layer``

See ``docs/source/core_architecture.md`` for the corresponding explanation.
"""

from __future__ import annotations

import os as _os
_FRAMEWORK_DIR = _os.path.dirname(_os.path.abspath(__file__))
_PACKAGE_DIR = _os.path.dirname(_FRAMEWORK_DIR)
_PROJECT_ROOT = _os.path.dirname(_PACKAGE_DIR)

__version__ = "1.0.0"
__layers__ = (
    (1, "Scenario"),
    (2, "Environment"),
    (3, "Simulation"),
    (4, "Algorithm"),
    (5, "Evaluation"),
)

__all__ = [
    "scenario_layer",
    "environment_layer",
    "simulation_layer",
    "algorithm_layer",
    "evaluation_layer",
]
