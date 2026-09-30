from __future__ import annotations

import importlib
import sys
from types import ModuleType

import pytest


REPRESENTATIVE_MODULES = (
    "b2d_rlinfra.scenario.adaptive_route_sampler",
    "b2d_rlinfra.environment.spaces",
    "b2d_rlinfra.simulation.crash_utils",
    "b2d_rlinfra.learning.utils.config",
    "b2d_rlinfra.finetuning.config_schema",
    "b2d_rlinfra.evaluation.leaderboard.score_versions",
    "b2d_rlinfra.framework.demo_utils",
)


@pytest.mark.parametrize("module_name", REPRESENTATIVE_MODULES)
def test_lightweight_domain_imports_do_not_require_carla_runtime(monkeypatch, module_name: str) -> None:
    # These stubs make an accidental eager import visible without loading CARLA or
    # either vendored Leaderboard runtime. The selected modules should remain light.
    monkeypatch.setitem(sys.modules, "carla", ModuleType("carla"))
    monkeypatch.setitem(sys.modules, "leaderboard", ModuleType("leaderboard"))
    monkeypatch.setitem(sys.modules, "srunner", ModuleType("srunner"))

    module = importlib.import_module(module_name)

    assert module.__name__ == module_name
