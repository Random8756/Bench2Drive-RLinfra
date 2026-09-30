from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]

pytestmark = [pytest.mark.carla_runtime, pytest.mark.timeout(300)]


_RUNTIME_PROBE = r"""
import importlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys

runtime_kind = sys.argv[1]
repo_root = Path(sys.argv[2]).resolve()
carla_root = Path(sys.argv[3]).resolve()
runtime_root = repo_root / "vendor" / "carla" / f"{runtime_kind}-runtime"
scenario_runner_root = runtime_root / "scenario_runner"
carla_python_api = carla_root / "PythonAPI" / "carla"

def module_origin(module):
    module_file = getattr(module, "__file__", None)
    if module_file:
        return Path(module_file).resolve()
    module_paths = list(getattr(module, "__path__", ()))
    if module_paths:
        return Path(module_paths[0]).resolve()
    raise AssertionError(f"module {module.__name__} has neither __file__ nor __path__")

def require_under(module, root, label):
    resolved = module_origin(module)
    try:
        resolved.relative_to(Path(root).resolve())
    except ValueError as exc:
        raise AssertionError(f"{label} came from {resolved}, expected it under {root}") from exc

if importlib.metadata.version("carla") != "0.9.15":
    raise AssertionError("CARLA binding must be version 0.9.15")

import carla
import agents
import leaderboard
import srunner

require_under(agents, carla_python_api / "agents", "official agents")
require_under(leaderboard, runtime_root / "leaderboard", "leaderboard")
require_under(srunner, scenario_runner_root / "srunner", "scenario runner")

client_calls = []

class ForbiddenClient:
    def __init__(self, *args, **kwargs):
        client_calls.append((args, kwargs))
        raise AssertionError("carla_runtime tests must not construct carla.Client")

carla.Client = ForbiddenClient

package_root = repo_root / "b2d_rlinfra"
if runtime_kind == "training":
    source_files = sorted(package_root.rglob("*.py"))
else:
    source_files = sorted((package_root / "evaluation").rglob("*.py"))

modules = []
for source in source_files:
    relative = source.relative_to(repo_root).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    if parts:
        modules.append(".".join(parts))

if runtime_kind == "evaluation":
    modules.append("b2d_rlinfra.framework.evaluation_layer")

for module_name in sorted(set(modules)):
    importlib.import_module(module_name)

from b2d_rlinfra.evaluation.runtime.action_adapter import ActionAdapter

adapter = ActionAdapter(
    {
        "type": "discrete",
        "discrete_actions_list": [[0.25, -0.5, 0.75]],
    }
)
control = adapter.to_control([0])
if not isinstance(control, carla.VehicleControl):
    raise AssertionError(f"expected carla.VehicleControl, got {type(control)!r}")
if (float(control.throttle), float(control.steer), float(control.brake)) != (0.25, -0.5, 0.75):
    raise AssertionError("action adapter did not preserve the configured control values")

entry_point = None
if runtime_kind == "evaluation":
    agent_path = package_root / "evaluation" / "leaderboard" / "agent.py"
    spec = importlib.util.spec_from_file_location("agent", agent_path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not create an import spec for {agent_path}")
    agent_module = importlib.util.module_from_spec(spec)
    sys.modules["agent"] = agent_module
    spec.loader.exec_module(agent_module)
    entry_point = agent_module.get_entry_point()
    if entry_point != "RLLeaderboardAgent":
        raise AssertionError(f"unexpected agent entry point: {entry_point!r}")

if client_calls:
    raise AssertionError(f"carla.Client was constructed: {client_calls!r}")

print(
    json.dumps(
        {
            "runtime": runtime_kind,
            "module_count": len(set(modules)),
            "carla_file": str(Path(carla.__file__).resolve()),
            "agents_file": str(module_origin(agents)),
            "leaderboard_file": str(module_origin(leaderboard)),
            "srunner_file": str(module_origin(srunner)),
            "agent_entry_point": entry_point,
            "client_calls": len(client_calls),
        },
        sort_keys=True,
    )
)
"""


def _carla_root() -> Path:
    raw_root = os.environ.get("CARLA_ROOT", "").strip()
    if not raw_root:
        pytest.fail("CARLA_ROOT is required for carla_runtime tests")
    root = Path(raw_root).expanduser().resolve()
    agents_dir = root / "PythonAPI" / "carla" / "agents" / "navigation"
    if not agents_dir.is_dir():
        pytest.fail(f"CARLA_ROOT does not provide official agents.navigation: {agents_dir}")
    return root


def _run_runtime_probe(runtime_kind: str) -> dict:
    carla_root = _carla_root()
    runtime_root = REPO_ROOT / "vendor" / "carla" / f"{runtime_kind}-runtime"
    scenario_runner_root = runtime_root / "scenario_runner"
    paths = [
        REPO_ROOT,
        runtime_root,
    ]
    if runtime_kind == "evaluation":
        paths.append(runtime_root / "leaderboard")
    paths.extend(
        [
            scenario_runner_root,
            carla_root / "PythonAPI" / "carla",
        ]
    )
    for path in paths:
        if not path.exists():
            pytest.fail(f"missing {runtime_kind} runtime path: {path}")

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(str(path) for path in paths)
    env["PYTHONNOUSERSITE"] = "1"
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(_RUNTIME_PROBE),
            runtime_kind,
            str(REPO_ROOT),
            str(carla_root),
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=240,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    assert lines, "runtime probe produced no result"
    return json.loads(lines[-1])


def test_training_runtime_composition_and_full_import() -> None:
    result = _run_runtime_probe("training")
    assert result["runtime"] == "training"
    assert result["module_count"] >= 150
    assert result["client_calls"] == 0


def test_evaluation_runtime_composition_and_agent_file_loading() -> None:
    result = _run_runtime_probe("evaluation")
    assert result["runtime"] == "evaluation"
    assert result["module_count"] >= 18
    assert result["agent_entry_point"] == "RLLeaderboardAgent"
    assert result["client_calls"] == 0
