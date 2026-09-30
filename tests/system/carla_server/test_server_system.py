from __future__ import annotations

import copy
import math
import os
from pathlib import Path
import queue
import random
import signal
import socket
import time
from typing import Iterable

import pytest
import yaml

from b2d_rlinfra.framework.env_rollout_demo import make_demo_env
from b2d_rlinfra.simulation.runners.carla_env_pool import CARLAEnvPool
from b2d_rlinfra.simulation.runners.carla_server_manager import CARLAServerManager


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_HOSTS = ("127.77.0.135", "127.77.0.136")

pytestmark = [
    pytest.mark.carla_server,
    pytest.mark.gpu,
    pytest.mark.slow,
    pytest.mark.timeout(1200),
]


def _carla_root() -> Path:
    raw_root = os.environ.get("CARLA_ROOT", "").strip()
    if not raw_root:
        pytest.fail("CARLA_ROOT is required for carla_server tests")
    root = Path(raw_root).expanduser().resolve()
    if not (root / "CarlaUE4.sh").is_file():
        pytest.fail(f"CARLA_ROOT does not contain CarlaUE4.sh: {root}")
    return root


def _graphics_adapters() -> list[int]:
    raw_ids = os.environ.get("B2D_TEST_GRAPHICS_ADAPTERS", "").strip()
    if not raw_ids:
        pytest.fail(
            "B2D_TEST_GRAPHICS_ADAPTERS must list two verified Vulkan graphicsadapter indices"
        )
    try:
        adapter_ids = [int(value.strip()) for value in raw_ids.split(",") if value.strip()]
    except ValueError as exc:
        pytest.fail(f"invalid B2D_TEST_GRAPHICS_ADAPTERS={raw_ids!r}: {exc}")
    if len(adapter_ids) != 2:
        pytest.fail("B2D_TEST_GRAPHICS_ADAPTERS must contain exactly two indices")
    return adapter_ids


def _hosts() -> list[str]:
    raw_hosts = os.environ.get("B2D_TEST_CARLA_HOSTS", "").strip()
    hosts = [value.strip() for value in raw_hosts.split(",") if value.strip()] if raw_hosts else list(DEFAULT_HOSTS)
    if len(hosts) != 2:
        pytest.fail("B2D_TEST_CARLA_HOSTS must contain exactly two loopback addresses")
    if any(not host.startswith("127.") for host in hosts):
        pytest.fail(f"server tests only accept loopback hosts, got: {hosts}")
    return hosts


def _fake_bind_library() -> Path:
    path = REPO_ROOT / "tools" / "fake_bind.so"
    if not path.is_file():
        pytest.fail(
            f"missing {path}; compile tools/fake_bind.c before running carla_server tests"
        )
    return path


def _scenario_runner_root() -> Path:
    raw_root = os.environ.get("SCENARIO_RUNNER_ROOT", "").strip()
    if not raw_root:
        pytest.fail("SCENARIO_RUNNER_ROOT is required for CARLAEnvPool system tests")
    root = Path(raw_root).expanduser().resolve()
    expected = REPO_ROOT / "vendor" / "carla" / "training-runtime" / "scenario_runner"
    if root != expected or not (root / "srunner" / "scenarios").is_dir():
        pytest.fail(
            "SCENARIO_RUNNER_ROOT must select the repository training runtime: "
            f"expected {expected}, got {root}"
        )
    return root


def _port_is_free(host: str, port: int) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.settimeout(0.25)
        return sock.connect_ex((host, port)) != 0
    finally:
        sock.close()


def _port_block_is_bindable(host: str, base_port: int) -> bool:
    sockets = []
    try:
        for offset in range(3):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((host, base_port + offset))
            sockets.append(sock)
        return True
    except OSError:
        return False
    finally:
        for sock in sockets:
            sock.close()


def _allocate_ports(hosts: Iterable[str]) -> tuple[list[int], list[int]]:
    rng = random.SystemRandom()
    host_list = list(hosts)
    for _ in range(200):
        first = rng.randrange(3000, 3900) * 10
        rpc_ports = [first, first + 10]
        tm_ports = [first + 6000, first + 6010]
        if all(
            _port_block_is_bindable(host, rpc_port)
            and _port_block_is_bindable(host, tm_port)
            for host, rpc_port, tm_port in zip(host_list, rpc_ports, tm_ports)
        ):
            return rpc_ports, tm_ports
    pytest.fail("could not allocate two test-owned CARLA/TM port blocks")


def _wait_ports_closed(hosts: list[str], ports: list[int], timeout: float = 45.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if all(_port_is_free(host, port) for host, port in zip(hosts, ports)):
            return True
        time.sleep(0.5)
    return False


def _load_rgb_pool_config(
    tmp_path: Path,
    hosts: list[str],
    rpc_ports: list[int],
    tm_ports: list[int],
    graphics_adapters: list[int],
) -> dict:
    document = yaml.safe_load((REPO_ROOT / "configs" / "env_rollout_demo.yaml").read_text(encoding="utf-8"))
    config = copy.deepcopy(document["env"])
    config["algorithm"] = {"name": "ppo"}
    config["ipc"] = {"use_rgb_shm": True}

    environment = config["environment"]
    environment["result_dir"] = str(tmp_path / "results")
    environment["max_episode_steps"] = 64
    environment["worker_recycle_interval_episodes"] = 0
    environment["worker_recycle_jitter_episodes"] = 0

    carla_config = config["carla"]
    carla_config.update(
        {
            "num_envs": 2,
            "host": hosts,
            "port": rpc_ports,
            "traffic_manager_port": tm_ports,
            "traffic_manager_seed": [101, 103],
            "gpu_id": graphics_adapters,
            "load_world_timeout": 180,
            "worker_ready_timeout": 300,
            "connection_timeout": 60,
            "server_wait_timeout": 180,
            "timeout": 60,
            "render_offscreen": True,
            "no_rendering_mode": False,
            "no_sound": True,
            "null_rhi": False,
            "opengl": False,
            "vulkan": True,
            "no_steam": True,
            "quality_level": "Low",
            "extra_args": "-RPCThreads=2 -StreamingThreads=2 -SecondaryThreads=2",
        }
    )

    routes = config["routes"]
    routes["sample_mode"] = "sequential"
    routes.pop("adaptive", None)
    routes["route_max_length_m"] = 80.0

    observation_space = config["observation_space"]
    observation_space["vector"]["enable"] = False
    observation_space["rgb"] = {
        "enable": True,
        "output_key": "rgb",
        "frame_timeout": 60.0,
        "frame_history_size": 2,
        "warmup_ticks": 2,
        "tick_timeout": 30.0,
        "camera_order": ["CAM_FRONT"],
        "sensors": [
            {
                "type": "sensor.camera.rgb",
                "x": 0.8,
                "y": 0.0,
                "z": 1.6,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "width": 96,
                "height": 64,
                "fov": 70,
                "id": "CAM_FRONT",
            }
        ],
    }
    return config


def test_parallel_servers_expose_matching_runtime_and_release_ports() -> None:
    import carla

    hosts = _hosts()
    rpc_ports, _ = _allocate_ports(hosts)
    adapters = _graphics_adapters()
    manager = CARLAServerManager(
        carla_root=str(_carla_root()),
        default_fps=10,
        default_quality="Low",
        fake_bind_lib=str(_fake_bind_library()),
    )

    try:
        results = manager.start_servers(
            ports=rpc_ports,
            gpu_ids=adapters,
            hosts=hosts,
            timeout=180,
            render_offscreen=True,
            no_sound=True,
            null_rhi=False,
            vulkan=True,
            no_steam=True,
            extra_args=["-RPCThreads=2", "-StreamingThreads=2"],
        )
        assert results and all(results.values()), results
        assert sorted(manager.get_running_servers()) == sorted(zip(hosts, rpc_ports))

        for server_index, (host, port) in enumerate(zip(hosts, rpc_ports)):
            client = carla.Client(host, port)
            client.set_timeout(60.0)
            assert client.get_server_version().startswith("0.9.15")
            assert client.get_client_version().startswith("0.9.15")
            world = client.get_world()
            assert world.get_map().name

            if server_index == 0:
                blueprint = world.get_blueprint_library().find("sensor.camera.rgb")
                blueprint.set_attribute("image_size_x", "64")
                blueprint.set_attribute("image_size_y", "48")
                blueprint.set_attribute("sensor_tick", "0.05")
                spawn_point = world.get_map().get_spawn_points()[0]
                camera_transform = carla.Transform(
                    carla.Location(
                        x=spawn_point.location.x,
                        y=spawn_point.location.y,
                        z=spawn_point.location.z + 2.0,
                    ),
                    spawn_point.rotation,
                )
                camera = world.spawn_actor(blueprint, camera_transform)
                frames: queue.Queue = queue.Queue()
                try:
                    camera.listen(frames.put)
                    frame = frames.get(timeout=30.0)
                    assert (frame.width, frame.height) == (64, 48)
                    assert len(frame.raw_data) == 64 * 48 * 4
                finally:
                    camera.stop()
                    camera.destroy()
    finally:
        manager.stop_all(timeout=30.0)

    assert _wait_ports_closed(hosts, rpc_ports), f"CARLA ports did not close: {list(zip(hosts, rpc_ports))}"


def test_two_worker_rgb_pool_recovers_one_server_and_cleans_resources(tmp_path: Path) -> None:
    _scenario_runner_root()
    hosts = _hosts()
    rpc_ports, tm_ports = _allocate_ports(hosts)
    adapters = _graphics_adapters()
    config = _load_rgb_pool_config(tmp_path, hosts, rpc_ports, tm_ports, adapters)

    pool = None
    server_manager = None
    shm_paths: list[Path] = []
    completed_results = 0
    saw_crash = False
    saw_recovered_reset = False
    target_worker = 0
    killed_server = False

    try:
        pool = CARLAEnvPool(
            env_fn=make_demo_env,
            config=config,
            num_envs=2,
            auto_reset=True,
            max_episode_steps=64,
            health_check_interval=1.0,
            worker_timeout=120.0,
            server_wait_timeout=180.0,
            start_method="spawn",
            manage_servers=True,
            carla_root=str(_carla_root()),
            gpu_ids=adapters,
            fake_bind_lib=str(_fake_bind_library()),
        )
        server_manager = pool._server_manager
        assert server_manager is not None
        shm_paths = [Path(buffer.path) for buffer in pool._rgb_shm_buffers.values()]
        assert len(shm_paths) == 2 and all(path.exists() for path in shm_paths)

        observations, reset_infos = pool.reset(min_ready=2, timeout=300.0)
        assert set(observations) == {0, 1}
        assert set(reset_infos) == {0, 1}
        observation_space = pool.single_observation_space
        action_space = pool.single_action_space
        assert observation_space is not None and action_space is not None
        for observation in observations.values():
            assert observation_space.contains(observation)
            assert observation["rgb"].shape[-3:] == (64, 96, 3)

        ready = set(observations)
        deadline = time.monotonic() + 720.0
        while time.monotonic() < deadline:
            actions = {worker_id: 8 for worker_id in ready}
            for action in actions.values():
                assert action_space.contains(action)
            ready.clear()

            next_obs, rewards, terms, truncs, infos = pool.step(
                actions,
                min_ready=1,
                timeout=90.0,
            )

            for worker_id, reward in rewards.items():
                assert math.isfinite(float(reward))
                assert isinstance(infos.get(worker_id), dict)
                completed_results += 1
                info = infos.get(worker_id, {})
                if info.get("crashed"):
                    saw_crash = True
                    continue
                if not terms.get(worker_id, False) and not truncs.get(worker_id, False):
                    observation = next_obs.get(worker_id)
                    assert observation is not None
                    assert observation_space.contains(observation)
                    ready.add(worker_id)

            for worker_id, observation in next_obs.items():
                info = infos.get(worker_id, {})
                if worker_id not in rewards or info.get("from_reset") or info.get("episode_start"):
                    assert observation_space.contains(observation)
                    ready.add(worker_id)
                    if killed_server and worker_id == target_worker:
                        saw_recovered_reset = True

            if completed_results >= 6 and not killed_server:
                server_info = server_manager.get_server_info(hosts[target_worker], rpc_ports[target_worker])
                assert server_info is not None and server_info.process is not None
                os.killpg(os.getpgid(server_info.process.pid), signal.SIGKILL)
                killed_server = True

            stats = pool.get_stats()
            target_stats = stats["worker_details"][target_worker]
            if killed_server and target_stats["restart_count"] >= 1 and target_worker in ready:
                saw_recovered_reset = True

            if completed_results >= 12 and saw_recovered_reset:
                break

        assert killed_server
        assert saw_crash or pool.get_stats()["worker_details"][target_worker]["restart_count"] >= 1
        assert saw_recovered_reset
        assert completed_results >= 12
    finally:
        if pool is not None:
            pool.close()

    assert _wait_ports_closed(hosts, rpc_ports), f"CARLA ports did not close: {list(zip(hosts, rpc_ports))}"
    assert all(not path.exists() for path in shm_paths), f"RGB shared memory leaked: {shm_paths}"
    if server_manager is not None:
        assert not server_manager.get_running_servers()
