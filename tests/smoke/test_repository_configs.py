from __future__ import annotations

from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_FILES = sorted((REPO_ROOT / "configs").glob("*.yaml"))


@pytest.mark.parametrize("config_path", CONFIG_FILES, ids=lambda path: path.name)
def test_repository_config_references_existing_routes_and_maps(config_path: Path) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert isinstance(config, dict)

    env_config = config.get("env") or {}
    route_files = (env_config.get("routes") or {}).get("route_files") or []
    for relative_path in route_files:
        path = Path(relative_path)
        assert not path.is_absolute(), f"{config_path.name}: route path must stay repository-relative"
        assert (REPO_ROOT / path).is_file(), f"{config_path.name}: missing route file {relative_path}"

    vector_config = (env_config.get("observation_space") or {}).get("vector") or {}
    map_dir = vector_config.get("map_dir")
    if map_dir:
        path = Path(map_dir)
        assert not path.is_absolute(), f"{config_path.name}: map path must stay repository-relative"
        assert (REPO_ROOT / path).is_dir(), f"{config_path.name}: missing map directory {map_dir}"
