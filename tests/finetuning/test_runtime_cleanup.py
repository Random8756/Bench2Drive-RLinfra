from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[2]

from b2d_rlinfra.finetuning.carla_slot import CarlaSlot, _new_rgb_shm_prefix
from b2d_rlinfra.finetuning.config_schema import bind_environment_result_dir
from b2d_rlinfra.finetuning.node_agent import (
    _cleanup_local_carla,
    _cleanup_local_rgb_shm,
    _join_processes,
    _run_shm_prefix,
    _stop_collectors,
)
from b2d_rlinfra.finetuning.sim_actor import SimActorProcess

_CLEANUP_NODE_PATH = ROOT / "tools" / "slurm" / "cleanup_rl_finetune_node.py"
_CLEANUP_NODE_SPEC = importlib.util.spec_from_file_location("cleanup_rl_finetune_node", _CLEANUP_NODE_PATH)
assert _CLEANUP_NODE_SPEC is not None and _CLEANUP_NODE_SPEC.loader is not None
_CLEANUP_NODE = importlib.util.module_from_spec(_CLEANUP_NODE_SPEC)
_CLEANUP_NODE_SPEC.loader.exec_module(_CLEANUP_NODE)


class _FakeProcess:
    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.alive = True
        self.join_timeouts = []
        self.terminated = False
        self.killed = False

    def join(self, timeout: float) -> None:
        self.join_timeouts.append(timeout)

    def is_alive(self) -> bool:
        return self.alive

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True
        self.alive = False


class RuntimeCleanupTest(unittest.TestCase):
    def test_inline_coordinator_passes_run_dir_to_collectors(self) -> None:
        from b2d_rlinfra.finetuning.coordinator import Coordinator

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp).resolve()
            plan = MagicMock()
            plan.collector_id = 0
            plan.device = "cpu"
            plan.to_dict.return_value = {"collector_id": 0}
            topology = MagicMock()
            topology.node.return_value.collectors = [plan]
            received = []

            class FakeCollector:
                def __init__(self, **kwargs):
                    received.append(kwargs)
                    self.slot = MagicMock()

            coordinator = object.__new__(Coordinator)
            coordinator.topology = topology
            coordinator.config = {}
            coordinator.rl_config = {}
            coordinator.rollout_dir = run_dir / "rollouts"
            coordinator.weight_dir = run_dir / "weights"
            coordinator.run_dir = run_dir
            coordinator.update_id = 0

            with patch("b2d_rlinfra.finetuning.coordinator.Collector", FakeCollector):
                coordinator._run_inline(max_updates=0)

            self.assertEqual(len(received), 1)
            self.assertEqual(Path(received[0]["run_dir"]), run_dir)

    def test_environment_result_binding_is_run_scoped_without_mutating_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp).resolve() / "run"
            source_config = {
                "environment": {"result_dir": "legacy-results", "max_episode_steps": 10},
                "carla": {"host": ["127.0.0.1"]},
            }

            rebound = bind_environment_result_dir(source_config, str(run_dir))

            self.assertEqual(Path(rebound["environment"]["result_dir"]), run_dir)
            self.assertEqual(rebound["environment"]["max_episode_steps"], 10)
            self.assertEqual(rebound["carla"]["host"], ["127.0.0.1"])
            self.assertEqual(source_config["environment"]["result_dir"], "legacy-results")

    def test_local_collector_uses_run_scoped_environment_results(self) -> None:
        from b2d_rlinfra.finetuning.collector import Collector

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp).resolve() / "run"
            source_config = {
                "algorithm": {
                    "learning_rate": 1.0e-4,
                    "gamma": 0.99,
                    "gae_lambda": 0.95,
                },
                "policy_adapter": {"type": "test", "config": {}},
                "training": {},
                "env": {"environment": {"result_dir": "legacy-results"}},
            }
            adapter = MagicMock()
            adapter.load_initial.return_value = MagicMock(policy_version=0)

            with (
                patch("b2d_rlinfra.finetuning.collector.resolve_policy_adapter", return_value=adapter),
                patch("b2d_rlinfra.finetuning.collector.make_slot", return_value=MagicMock()) as make_slot,
            ):
                Collector(
                    collector_id=0,
                    config=source_config,
                    rl_config={"distributed": {"enabled": False}},
                    rollout_dir=str(run_dir / "rollouts"),
                    weight_dir=str(run_dir / "weights"),
                    device="cpu",
                    run_dir=str(run_dir),
                )

            received_env_config = make_slot.call_args.kwargs["env_config"]
            self.assertEqual(Path(received_env_config["environment"]["result_dir"]), run_dir)
            self.assertEqual(source_config["env"]["environment"]["result_dir"], "legacy-results")

    def test_sim_actor_nests_env_results_without_mutating_source_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp).resolve()
            config = {
                "environment": {"result_dir": str(run_dir)},
                "carla": {},
            }
            received_configs = []

            fake_env = MagicMock()
            fake_env.action_space = "action-space"
            fake_env.observation_space = "observation-space"
            fake_env.reset.return_value = ({"obs": 1}, {})

            def make_env(worker_config, *_args):
                received_configs.append(worker_config)
                return fake_env

            actor = SimActorProcess(
                worker_id=3,
                worker_epoch=1,
                env_fn=make_env,
                config=config,
                host="127.0.0.1",
                carla_port=2000,
                traffic_manager_port=8000,
                traffic_manager_seed=3,
                max_episode_steps=10,
                auto_reset=True,
                result_dir=str(run_dir),
            )
            control_queue = MagicMock()
            control_queue.get_nowait.return_value = {"type": "stop"}

            with (
                patch("b2d_rlinfra.finetuning.sim_actor._server_available", return_value=True),
                patch("b2d_rlinfra.finetuning.sim_actor.time.sleep"),
            ):
                actor(MagicMock(), MagicMock(), control_queue)

            self.assertEqual(len(received_configs), 1)
            self.assertEqual(
                Path(received_configs[0]["environment"]["result_dir"]),
                run_dir / "env_results",
            )
            self.assertEqual(config["environment"]["result_dir"], str(run_dir))
            fake_env.close.assert_called_once_with()

    def test_sim_actor_crash_records_remain_at_run_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp).resolve()

            def fail_env(*_args):
                raise RuntimeError("environment construction failed")

            actor = SimActorProcess(
                worker_id=4,
                worker_epoch=2,
                env_fn=fail_env,
                config={"environment": {"result_dir": str(run_dir)}, "carla": {}},
                host="127.0.0.1",
                carla_port=2001,
                traffic_manager_port=8001,
                traffic_manager_seed=4,
                max_episode_steps=10,
                auto_reset=True,
                result_dir=str(run_dir),
            )

            with (
                patch("b2d_rlinfra.finetuning.sim_actor._server_available", return_value=True),
                patch("b2d_rlinfra.finetuning.sim_actor.time.sleep"),
                patch("b2d_rlinfra.finetuning.sim_actor.write_worker_crash_file") as write_crash,
            ):
                actor(MagicMock(), MagicMock(), MagicMock())

            write_crash.assert_called_once()
            self.assertEqual(Path(write_crash.call_args.args[0]), run_dir)

    def test_run_hostname_discovery_requires_complete_topology(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            nodes_dir = run_dir / "nodes"
            nodes_dir.mkdir()
            (run_dir / "topology_resolved.json").write_text(
                json.dumps({"num_nodes": 2}),
                encoding="utf-8",
            )
            (nodes_dir / "node_000.json").write_text(
                json.dumps({"node_rank": 0, "hostname": "node115"}),
                encoding="utf-8",
            )
            self.assertEqual(_CLEANUP_NODE.run_hostnames(run_dir), [])

            (nodes_dir / "node_001.json").write_text(
                json.dumps({"node_rank": 1, "hostname": "node116"}),
                encoding="utf-8",
            )
            self.assertEqual(_CLEANUP_NODE.run_hostnames(run_dir), ["node115", "node116"])

    def test_local_carla_cleanup_runs_hosts_in_parallel(self) -> None:
        active = 0
        max_active = 0
        lock = threading.Lock()

        def fake_run(*_args, **_kwargs):
            nonlocal active, max_active
            with lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.02)
            with lock:
                active -= 1

        node_plan = SimpleNamespace(
            collectors=[SimpleNamespace(host=f"127.0.0.{index}") for index in range(1, 5)]
        )
        raw_config = {"rl_finetune": {"sim_actor": {"env_type": "carla"}}}
        rl_config = {"sim_actor": {"manage_servers": True}, "carla_cleanup_timeout": 1.0}
        with patch("b2d_rlinfra.finetuning.node_agent.subprocess.run", side_effect=fake_run):
            _cleanup_local_carla(raw_config, rl_config, node_plan)

        self.assertGreater(max_active, 1)

    def test_run_scoped_shm_cleanup_does_not_cross_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_a = root / "run_a"
            run_b = root / "run_b"
            run_a.mkdir()
            run_b.mkdir()
            prefix_a = _run_shm_prefix(run_a)
            prefix_b = _run_shm_prefix(run_b)
            self.assertNotEqual(prefix_a, prefix_b)

            shm_a = root / f"{prefix_a}_abc_w0_rgb"
            shm_b = root / f"{prefix_b}_def_w0_rgb"
            legacy = root / "rl_finetune_rgb_legacy_w0_rgb"
            for path in (shm_a, shm_b, legacy):
                path.write_bytes(b"test")

            removed = _cleanup_local_rgb_shm(prefix_a, root)
            self.assertEqual(removed, [shm_a])
            self.assertFalse(shm_a.exists())
            self.assertTrue(shm_b.exists())
            self.assertTrue(legacy.exists())

    def test_carla_slot_uses_injected_run_prefix(self) -> None:
        with patch.dict(os.environ, {"B2D_RL_FINETUNE_SHM_PREFIX": "rl_finetune_rgb_run123"}):
            prefix = _new_rgb_shm_prefix()
        self.assertTrue(prefix.startswith("rl_finetune_rgb_run123_"))

    def test_join_processes_shares_one_timeout_budget(self) -> None:
        first = _FakeProcess(101)
        second = _FakeProcess(102)
        with patch(
            "b2d_rlinfra.finetuning.node_agent.time.monotonic",
            side_effect=[10.0, 10.25, 10.75],
        ):
            _join_processes([first, second], 1.0)
        self.assertAlmostEqual(first.join_timeouts[0], 0.75)
        self.assertAlmostEqual(second.join_timeouts[0], 0.25)

    def test_stop_collectors_terminates_then_kills_stragglers(self) -> None:
        stop_event = MagicMock()
        processes = [_FakeProcess(201), _FakeProcess(202)]
        with patch("b2d_rlinfra.finetuning.node_agent._join_processes"):
            remaining = _stop_collectors(processes, stop_event, 1.0)
        stop_event.set.assert_called_once_with()
        self.assertEqual(remaining, [])
        self.assertTrue(all(process.terminated for process in processes))
        self.assertTrue(all(process.killed for process in processes))

    def test_carla_slot_close_always_attempts_all_cleanup_steps(self) -> None:
        slot = CarlaSlot.__new__(CarlaSlot)
        slot.host = "127.0.0.1"
        slot.carla_port = 2000
        slot._stop_actor = MagicMock(side_effect=RuntimeError("actor stop failed"))
        slot._server_manager = MagicMock()
        slot._server_manager.stop_server.side_effect = RuntimeError("server stop failed")
        slot._cleanup_rgb_shm = MagicMock()

        with patch("b2d_rlinfra.finetuning.carla_slot.logger.exception"):
            slot.close()

        slot._server_manager.stop_server.assert_called_once_with(slot.host, slot.carla_port)
        slot._cleanup_rgb_shm.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
