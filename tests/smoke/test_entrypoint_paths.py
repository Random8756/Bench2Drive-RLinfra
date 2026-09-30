from __future__ import annotations

from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]

DEFAULT_ENTRYPOINTS = {
    Path("tools/launch/baseline/train.sh"): "python -m b2d_rlinfra.learning.training.train",
    Path("tools/launch/baseline/resume_on_policy.sh"): (
        "python -m b2d_rlinfra.learning.training.resume_on_policy"
    ),
    Path("tools/launch/baseline/resume_off_policy.sh"): (
        "python -m b2d_rlinfra.learning.training.resume_off_policy"
    ),
    Path("tools/launch/finetune/train.sh"): "python -m b2d_rlinfra.finetuning.train",
    Path("tools/launch/evaluation/run_leaderboard_eval_parallel.sh"): (
        "python3 -m b2d_rlinfra.evaluation.leaderboard.run_leaderboard_eval_parallel"
    ),
}

STALE_ENTRYPOINT_TOKENS = (
    "agents/rl_agents",
    "$REPO_ROOT/agents",
    "$REPO_ROOT/leaderboard/",
    "leaderboard_ori",
    "/scripts/slurm/",
    "python -m rl_agents",
)


def _read(relative_path: Path | str) -> str:
    path = ROOT / relative_path
    assert path.is_file(), f"missing entrypoint: {path}"
    return path.read_text(encoding="utf-8")


@pytest.mark.parametrize(("relative_path", "module_call"), DEFAULT_ENTRYPOINTS.items())
def test_default_launchers_resolve_root_and_call_new_modules(
    relative_path: Path,
    module_call: str,
) -> None:
    text = _read(relative_path)

    assert 'REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"' in text
    assert 'cd "$REPO_ROOT"' in text
    assert module_call in text
    assert not any(token in text for token in STALE_ENTRYPOINT_TOKENS)


def test_runtime_selection_matches_launcher_role() -> None:
    training_launchers = (
        Path("tools/launch/baseline/train.sh"),
        Path("tools/launch/baseline/resume_on_policy.sh"),
        Path("tools/launch/baseline/resume_off_policy.sh"),
        Path("tools/launch/finetune/train.sh"),
        Path("b2d_rlinfra/framework/env_rollout_demo.sh"),
        Path("tools/slurm/node_entry.sh"),
    )
    for relative_path in training_launchers:
        text = _read(relative_path)
        assert "vendor/carla/training-runtime" in text
        assert "evaluation-runtime" not in text

    evaluation_text = _read("tools/launch/evaluation/run_leaderboard_eval_parallel.sh")
    assert "vendor/carla/evaluation-runtime" in evaluation_text
    assert "training-runtime" not in evaluation_text


def test_finetune_launcher_forwards_module_arguments() -> None:
    text = _read("tools/launch/finetune/train.sh")
    assert 'CMD+=("$@")' in text


def test_framework_demo_keeps_its_own_launcher_contract() -> None:
    text = _read("b2d_rlinfra/framework/env_rollout_demo.sh")

    assert 'REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"' in text
    assert "python -m b2d_rlinfra.framework.env_rollout_demo" in text


def test_all_slurm_helpers_use_moved_siblings_and_modules() -> None:
    slurm_dir = ROOT / "tools" / "slurm"
    shell_paths = sorted(slurm_dir.glob("*.sh"))
    python_paths = sorted(slurm_dir.glob("*.py"))

    assert {path.name for path in shell_paths} == {
        "cleanup_rl_finetune_job.sh",
        "node_entry.sh",
        "rl_finetune_resume_sbatch.sh",
        "rl_finetune_sbatch.sh",
    }
    assert {path.name for path in python_paths} == {"cleanup_rl_finetune_node.py"}
    assert 'NODE_CLEANUP="$SCRIPT_DIR/cleanup_rl_finetune_node.py"' in _read(
        "tools/slurm/cleanup_rl_finetune_job.sh"
    )
    assert "tools/slurm/node_entry.sh" in _read("tools/slurm/rl_finetune_sbatch.sh")
    assert "tools/slurm/node_entry.sh" in _read("tools/slurm/rl_finetune_resume_sbatch.sh")
    assert "python -m b2d_rlinfra.finetuning.node_agent" in _read("tools/slurm/node_entry.sh")

    for path in (*shell_paths, *python_paths):
        text = path.read_text(encoding="utf-8")
        assert not any(token in text for token in STALE_ENTRYPOINT_TOKENS)


def test_checkpoint_export_launchers_resolve_new_package_root() -> None:
    exporters = {
        "export_minddrive_checkpoint.sh": (
            "python -m b2d_rlinfra.finetuning.tools.export_minddrive_checkpoint"
        ),
        "export_drivepi0_checkpoint.sh": (
            "python -m b2d_rlinfra.finetuning.tools.export_drivepi0_checkpoint"
        ),
    }
    for name, module_call in exporters.items():
        text = _read(Path("b2d_rlinfra/finetuning/tools") / name)
        assert 'REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"' in text
        assert module_call in text
        assert "training-runtime" not in text
        assert "evaluation-runtime" not in text


def test_runtime_helper_sibling_reference_survived_move() -> None:
    runtime_dir = ROOT / "tools" / "runtime"
    assert {path.name for path in runtime_dir.glob("*.sh")} == {
        "clean_shm.sh",
        "kill_all.sh",
        "kill_by_host.sh",
    }

    kill_all = _read("tools/runtime/kill_all.sh")
    assert 'SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"' in kill_all
    assert '"$SCRIPT_DIR/kill_by_host.sh"' in kill_all
    assert (runtime_dir / "kill_by_host.sh").is_file()
