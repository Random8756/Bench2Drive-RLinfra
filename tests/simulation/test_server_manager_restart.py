from __future__ import annotations

from pathlib import Path
import shlex

import pytest
import yaml

from b2d_rlinfra.simulation.runners import carla_server_manager as manager_module
from b2d_rlinfra.simulation.runners.carla_server_manager import CARLAServerManager


REPO_ROOT = Path(__file__).resolve().parents[2]


class _FakeProcess:
    pid = 12345


def _make_manager(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> CARLAServerManager:
    carla_root = tmp_path / "carla"
    carla_root.mkdir()
    (carla_root / "CarlaUE4.sh").touch()

    monkeypatch.setattr(manager_module.subprocess, "Popen", lambda *args, **kwargs: _FakeProcess())
    monkeypatch.setattr(manager_module.os, "setsid", lambda: None, raising=False)
    manager = CARLAServerManager(str(carla_root))
    monkeypatch.setattr(manager, "_check_port_available", lambda port, host="localhost": True)
    monkeypatch.setattr(manager, "_wait_port_free", lambda host, port, timeout=30.0: True)

    def _stop_server(host: str, port: int, timeout: float = 10.0) -> bool:
        manager._servers.pop((host, port), None)
        return True

    monkeypatch.setattr(manager, "stop_server", _stop_server)
    return manager


@pytest.mark.parametrize(
    ("yaml_text", "expected"),
    [
        (
            """
env:
  carla:
    extra_args: '-RPCThreads=2 "-CustomFlag=two words" -nothreading'
""",
            ["-RPCThreads=2", "-CustomFlag=two words", "-nothreading"],
        ),
        (
            """
env:
  carla:
    extra_args:
      - -RPCThreads=4
      - -StreamingThreads=2
""",
            ["-RPCThreads=4", "-StreamingThreads=2"],
        ),
    ],
)
def test_restart_inherits_yaml_extra_args_when_not_overridden(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    yaml_text: str,
    expected: list[str],
) -> None:
    manager = _make_manager(tmp_path, monkeypatch)
    raw_config = yaml.safe_load(yaml_text)
    extra_args = raw_config["env"]["carla"]["extra_args"]

    assert manager.start_server(
        host="127.0.0.1",
        port=2200,
        extra_args=extra_args,
        wait=False,
    )
    assert manager.get_server_info("127.0.0.1", 2200).config.extra_args == expected

    assert manager.restart_server("127.0.0.1", 2200, gpu_id=3, wait=False)
    restarted = manager.get_server_info("127.0.0.1", 2200)
    assert restarted.config.extra_args == expected
    assert restarted.config.gpu_id == 3
    manager._servers.clear()


@pytest.mark.parametrize(
    ("override", "expected"),
    [
        ('-RPCThreads=8 "-CustomFlag=new value"', ["-RPCThreads=8", "-CustomFlag=new value"]),
        (["-RPCThreads=6", "-nothreading"], ["-RPCThreads=6", "-nothreading"]),
        ("", []),
        ([], []),
        (None, []),
    ],
)
def test_restart_explicit_extra_args_override_or_clear_saved_value(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    override: object,
    expected: list[str],
) -> None:
    manager = _make_manager(tmp_path, monkeypatch)
    assert manager.start_server(
        host="127.0.0.1",
        port=2200,
        extra_args=["-RPCThreads=2", "-OldFlag=1"],
        wait=False,
    )

    assert manager.restart_server(
        "127.0.0.1",
        2200,
        extra_args=override,
        wait=False,
    )
    restarted = manager.get_server_info("127.0.0.1", 2200)
    assert restarted.config.extra_args == expected
    manager._servers.clear()


def test_repository_baseline_yaml_extra_args_survive_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _make_manager(tmp_path, monkeypatch)
    checked_configs: list[str] = []

    for config_path in sorted((REPO_ROOT / "configs").glob("*.yaml")):
        raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        env_config = raw_config.get("env") or {}
        raw_extra_args = (env_config.get("carla") or {}).get("extra_args")
        if not raw_extra_args:
            continue

        expected = (
            shlex.split(raw_extra_args)
            if isinstance(raw_extra_args, str)
            else list(raw_extra_args)
        )
        checked_configs.append(config_path.name)

        assert manager.start_server(
            host="127.0.0.1",
            port=2200,
            extra_args=raw_extra_args,
            wait=False,
        )
        assert manager.restart_server("127.0.0.1", 2200, wait=False)
        restarted = manager.get_server_info("127.0.0.1", 2200)
        assert restarted.config.extra_args == expected, config_path.name
        restarted_command = manager._build_command(restarted.config, "127.0.0.1")
        assert all(argument in restarted_command for argument in expected), config_path.name
        manager._servers.clear()

    assert checked_configs, "expected at least one repository config with env.carla.extra_args"
