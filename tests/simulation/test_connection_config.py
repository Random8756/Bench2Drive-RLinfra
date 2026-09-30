from __future__ import annotations

import pytest

from b2d_rlinfra.simulation.runners.carla_env_pool_utils import parse_connection_config


def test_connection_config_keeps_worker_resources_aligned() -> None:
    parsed = parse_connection_config(
        config={
            "carla": {
                "host": ["127.0.0.11", "127.0.0.12"],
                "port": [2100, 2110],
                "traffic_manager_port": [8100, 8110],
                "traffic_manager_seed": [3, 5],
                "gpu_id": [2, 4],
            }
        },
        num_envs=2,
        carla_ports_override=None,
        tm_ports_override=None,
        gpu_ids_override=None,
    )

    assert parsed == {
        "hosts": ["127.0.0.11", "127.0.0.12"],
        "ports": [2100, 2110],
        "tm_ports": [8100, 8110],
        "tm_seeds": [3, 5],
        "gpu_ids": [2, 4],
    }


@pytest.mark.parametrize("key", ["host", "port", "traffic_manager_port", "gpu_id"])
def test_connection_config_rejects_short_explicit_worker_lists(key: str) -> None:
    config = {
        "carla": {
            "host": ["127.0.0.11", "127.0.0.12"],
            "port": [2100, 2110],
            "traffic_manager_port": [8100, 8110],
            "traffic_manager_seed": [0, 0],
            "gpu_id": [0, 1],
        }
    }
    config["carla"][key] = config["carla"][key][:1]

    with pytest.raises(ValueError, match=key):
        parse_connection_config(
            config=config,
            num_envs=2,
            carla_ports_override=None,
            tm_ports_override=None,
            gpu_ids_override=None,
        )
