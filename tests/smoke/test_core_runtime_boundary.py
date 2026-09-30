from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_core_contract_modules_do_not_import_carla_runtime() -> None:
    script = r'''
import importlib.abc
import sys

FORBIDDEN = ("carla", "agents", "leaderboard", "srunner")

class BlockRuntimeImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in FORBIDDEN or fullname.startswith(tuple(name + "." for name in FORBIDDEN)):
            raise AssertionError(f"core contract imported forbidden runtime module: {fullname}")
        return None

sys.meta_path.insert(0, BlockRuntimeImports())

modules = (
    "b2d_rlinfra.learning.algorithms.base_algorithm",
    "b2d_rlinfra.finetuning.config_schema",
    "b2d_rlinfra.finetuning.rollout_pack",
    "b2d_rlinfra.simulation.runners.carla_env_pool_utils",
    "b2d_rlinfra.evaluation.leaderboard.score_versions",
)
for module in modules:
    __import__(module)

unexpected = sorted(name for name in sys.modules if name.split(".", 1)[0] in FORBIDDEN)
if unexpected:
    raise AssertionError(f"forbidden runtime modules were imported: {unexpected}")
'''
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
