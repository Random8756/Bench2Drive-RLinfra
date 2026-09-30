from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]

from b2d_rlinfra.finetuning.config_schema import normalize_rl_finetune_config
from b2d_rlinfra.finetuning.coordination import (
    FileControlPlane,
    FileEventBus,
    publish_coordinator_exit_state,
)
from b2d_rlinfra.finetuning.node_agent import (
    _requires_vulkan_loader,
    _resolve_fake_bind_path,
    _run_preflight,
    _validate_vulkan_loader,
    _validate_slurm_node_count,
    _wait_for_stop,
)
from b2d_rlinfra.finetuning.rollout_file_store import RolloutManifest, select_ready_with_policy_lag
from b2d_rlinfra.finetuning.topology import build_topology


def _base_config() -> dict:
    return {
        "algorithm": {"name": "ppo", "batch_size": 2, "n_epochs": 1},
        "rl_finetune": {
            "device": "cpu",
            "execution_mode": "process",
            "num_collectors": 2,
            "collector_devices": ["cpu", "cpu"],
            "rollouts_per_update": 2,
            "collect_timeout": 10,
            "control_poll_interval": 0.1,
            "shutdown_timeout": 5,
            "sim_actor": {"env_type": "fake"},
        },
        "policy_adapter": {
            "type": "tiny_rgb_discrete",
            "config": {
                "num_cameras": 1,
                "image_height": 4,
                "image_width": 4,
                "channels": 3,
                "pool_grid": [1, 1],
                "scalar_dim": 2,
                "action_dim": 2,
                "hidden_dim": 4,
                "rgb_key": "rgb",
                "scalar_key": "scalars",
            },
        },
        "env": {
            "environment": {"result_dir": "./results"},
            "carla": {
                "num_envs": 2,
                "host": ["127.0.0.1", "127.0.0.2"],
                "port": [2000, 2010],
                "traffic_manager_port": [8000, 8010],
                "traffic_manager_seed": [0, 0],
                "gpu_id": [0, 1],
            },
        },
    }


class ConfigTopologyCoordinationTest(unittest.TestCase):
    def _ddp_config(self) -> dict:
        cfg = _base_config()
        cfg["algorithm"]["batch_size"] = 6
        rl_cfg = cfg["rl_finetune"]
        rl_cfg.pop("num_collectors")
        rl_cfg["device"] = "cuda:0"
        rl_cfg["collector_devices"] = ["cuda:3", "cuda:4"]
        rl_cfg["learner"] = {
            "strategy": "ddp",
            "devices": ["cuda:0", "cuda:1", "cuda:2"],
            "backend": "nccl",
        }
        rl_cfg["distributed"] = {
            "enabled": True,
            "num_nodes": 2,
            "collectors_per_node": 2,
            "learner_node_collectors": 1,
        }
        cfg["env"]["carla"].pop("num_envs")
        return cfg

    def test_ddp_batch_size_is_global_and_divisible(self) -> None:
        cfg = self._ddp_config()
        normalized = normalize_rl_finetune_config(cfg)
        self.assertEqual(normalized["algorithm"]["batch_size"], 6)
        cfg["algorithm"]["batch_size"] = 5
        with self.assertRaisesRegex(ValueError, "global learner batch size"):
            normalize_rl_finetune_config(cfg)

    def test_ddp_learners_are_restricted_to_node_zero(self) -> None:
        topology = build_topology(normalize_rl_finetune_config(self._ddp_config()))
        self.assertEqual(topology.learner_world_size, 3)
        self.assertEqual([p.device for p in topology.node(0).learners], ["cuda:0", "cuda:1", "cuda:2"])
        self.assertEqual(topology.node(1).learners, [])

    def test_ddp_rejects_node_zero_collector_device_overlap(self) -> None:
        cfg = self._ddp_config()
        cfg["rl_finetune"]["collector_devices"][0] = "cuda:2"
        with self.assertRaisesRegex(ValueError, "must not overlap"):
            build_topology(normalize_rl_finetune_config(cfg))

    def test_top_level_rl_finetune_fields_remain_open(self) -> None:
        cfg = _base_config()
        cfg["rl_finetune"]["experiment_note"] = "future extension"
        normalized = normalize_rl_finetune_config(cfg)
        self.assertEqual(normalized["rl_finetune"]["experiment_note"], "future extension")

    def test_removed_rollout_compression_fails_fast(self) -> None:
        cfg = _base_config()
        cfg["rl_finetune"]["rollout_compression"] = "none"
        with self.assertRaisesRegex(ValueError, "rollout_compression was removed"):
            normalize_rl_finetune_config(cfg)

    def test_removed_rollout_staging_dir_fails_fast(self) -> None:
        cfg = _base_config()
        cfg["rl_finetune"]["learner"] = {"rollout_staging_dir": "scratch/rollouts"}
        with self.assertRaisesRegex(ValueError, "rollout_staging_dir"):
            normalize_rl_finetune_config(cfg)

    def test_unknown_distributed_fields_fail_fast(self) -> None:
        cfg = _base_config()
        cfg["rl_finetune"].pop("num_collectors")
        cfg["env"]["carla"].pop("num_envs")
        cfg["rl_finetune"]["distributed"] = {
            "enabled": True,
            "num_nodes": 1,
            "collectors_per_node": 1,
            "max_policy_lag": 2,
        }
        with self.assertRaisesRegex(ValueError, "max_policy_lag"):
            normalize_rl_finetune_config(cfg)

    def test_stale_rollout_defaults_apply_to_single_node(self) -> None:
        normalized = normalize_rl_finetune_config(_base_config())
        rl_cfg = normalized["rl_finetune"]
        self.assertEqual(rl_cfg["max_policy_lag"], 2)
        self.assertEqual(rl_cfg["stale_rollout_action"], "drop")
        self.assertTrue(rl_cfg["cleanup_stale_rollouts"])

    def test_stale_rollout_defaults_apply_to_distributed(self) -> None:
        cfg = _base_config()
        cfg["rl_finetune"].pop("num_collectors")
        cfg["env"]["carla"].pop("num_envs")
        cfg["rl_finetune"]["distributed"] = {
            "enabled": True,
            "num_nodes": 1,
            "collectors_per_node": 1,
        }
        normalized = normalize_rl_finetune_config(cfg)
        rl_cfg = normalized["rl_finetune"]
        self.assertEqual(rl_cfg["max_policy_lag"], 2)
        self.assertEqual(rl_cfg["stale_rollout_action"], "drop")
        self.assertTrue(rl_cfg["cleanup_stale_rollouts"])

    def test_distributed_forbids_single_node_topology_fields(self) -> None:
        cfg = _base_config()
        cfg["rl_finetune"]["distributed"] = {
            "enabled": True,
            "num_nodes": 2,
            "collectors_per_node": 2,
        }
        with self.assertRaisesRegex(ValueError, "num_collectors"):
            normalize_rl_finetune_config(cfg)
        cfg["rl_finetune"].pop("num_collectors")
        with self.assertRaisesRegex(ValueError, "num_envs"):
            normalize_rl_finetune_config(cfg)

    def test_distributed_forbids_custom_rollout_and_weight_dirs(self) -> None:
        cfg = _base_config()
        cfg["rl_finetune"].pop("num_collectors")
        cfg["env"]["carla"].pop("num_envs")
        cfg["rl_finetune"]["distributed"] = {
            "enabled": True,
            "num_nodes": 1,
            "collectors_per_node": 1,
        }
        cfg["env"]["carla"]["gpu_id"] = "auto"

        with self.assertRaisesRegex(ValueError, "rollout_dir"):
            candidate = dict(cfg)
            candidate["rl_finetune"] = dict(cfg["rl_finetune"])
            candidate["rl_finetune"]["rollout_dir"] = "./custom_rollouts"
            normalize_rl_finetune_config(candidate)

        with self.assertRaisesRegex(ValueError, "weight_dir"):
            candidate = dict(cfg)
            candidate["rl_finetune"] = dict(cfg["rl_finetune"])
            candidate["rl_finetune"]["weight_dir"] = "./custom_weights"
            normalize_rl_finetune_config(candidate)

    def test_drop_requires_max_policy_lag(self) -> None:
        cfg = _base_config()
        cfg["rl_finetune"]["max_policy_lag"] = None
        cfg["rl_finetune"]["stale_rollout_action"] = "drop"
        with self.assertRaisesRegex(ValueError, "max_policy_lag"):
            normalize_rl_finetune_config(cfg)

    def test_single_node_topology_preserves_global_slot_semantics(self) -> None:
        topology = build_topology(normalize_rl_finetune_config(_base_config()))
        self.assertFalse(topology.distributed)
        self.assertEqual([p.collector_id for p in topology.collectors()], [0, 1])
        self.assertEqual([p.local_id for p in topology.collectors()], [0, 1])
        self.assertEqual([p.port for p in topology.collectors()], [2000, 2010])
        self.assertEqual([p.host for p in topology.collectors()], ["127.0.0.1", "127.0.0.2"])

    def test_distributed_topology_uses_global_gid_and_local_slots(self) -> None:
        cfg = _base_config()
        rl_cfg = cfg["rl_finetune"]
        rl_cfg.pop("num_collectors")
        rl_cfg.pop("collector_devices")
        rl_cfg["device"] = "cuda:0"
        rl_cfg["distributed"] = {
            "enabled": True,
            "num_nodes": 2,
            "collectors_per_node": 2,
            "learner_node_collectors": 1,
        }
        cfg["env"]["carla"].pop("num_envs")
        cfg["env"]["carla"]["gpu_id"] = "auto"
        topology = build_topology(normalize_rl_finetune_config(cfg))
        self.assertTrue(topology.distributed)
        self.assertEqual(topology.total_collectors, 3)
        self.assertEqual([p.collector_id for p in topology.node(0).collectors], [0])
        self.assertEqual([p.collector_id for p in topology.node(1).collectors], [1, 2])
        self.assertEqual([p.local_id for p in topology.node(1).collectors], [0, 1])
        self.assertEqual([p.port for p in topology.node(1).collectors], [2000, 2010])
        self.assertEqual([p.host for p in topology.node(1).collectors], ["127.0.0.1", "127.0.0.2"])
        self.assertEqual(topology.node(0).collectors[0].device, "cuda:1")
        self.assertEqual(topology.node(1).collectors[0].device, "cuda:0")

    def test_distributed_topology_defaults_learner_node_to_one_fewer_collector(self) -> None:
        cfg = _base_config()
        rl_cfg = cfg["rl_finetune"]
        rl_cfg.pop("num_collectors")
        rl_cfg.pop("collector_devices")
        rl_cfg["device"] = "cuda:0"
        rl_cfg["distributed"] = {
            "enabled": True,
            "num_nodes": 2,
            "collectors_per_node": 2,
        }
        cfg["env"]["carla"].pop("num_envs")
        cfg["env"]["carla"]["gpu_id"] = "auto"
        topology = build_topology(normalize_rl_finetune_config(cfg))
        self.assertEqual([p.collector_id for p in topology.node(0).collectors], [0])
        self.assertEqual([p.collector_id for p in topology.node(1).collectors], [1, 2])
        self.assertEqual(topology.node(0).collectors[0].device, "cuda:1")

    def test_distributed_min_active_collectors_bounds(self) -> None:
        cfg = _base_config()
        rl_cfg = cfg["rl_finetune"]
        rl_cfg.pop("num_collectors")
        rl_cfg.pop("collector_devices")
        rl_cfg["distributed"] = {
            "enabled": True,
            "num_nodes": 2,
            "collectors_per_node": 2,
            "learner_node_collectors": 1,
        }
        cfg["env"]["carla"].pop("num_envs")
        cfg["env"]["carla"]["gpu_id"] = "auto"

        for invalid in (0, 4):
            with self.subTest(min_active_collectors=invalid):
                candidate = normalize_rl_finetune_config(cfg)
                candidate["rl_finetune"]["distributed"]["min_active_collectors"] = invalid
                with self.assertRaisesRegex(ValueError, "min_active_collectors"):
                    build_topology(candidate)

    def test_slurm_node_count_mismatch_fails_fast(self) -> None:
        with patch.dict(os.environ, {"SLURM_JOB_NUM_NODES": "1"}, clear=False):
            with self.assertRaisesRegex(ValueError, "distributed.num_nodes=2"):
                _validate_slurm_node_count({"num_nodes": 2})

    def test_fake_bind_default_path_matches_build_output(self) -> None:
        cfg = _base_config()
        self.assertEqual(_resolve_fake_bind_path(cfg), ROOT / "tools" / "fake_bind.so")

        cfg["rl_finetune"]["fake_bind_path"] = "/custom/fake_bind.so"
        self.assertEqual(_resolve_fake_bind_path(cfg), Path("/custom/fake_bind.so"))

    def test_vulkan_loader_requirement_tracks_runtime_mode(self) -> None:
        cfg = _base_config()
        self.assertFalse(_requires_vulkan_loader(cfg))

        cfg["rl_finetune"]["sim_actor"]["env_type"] = "carla"
        self.assertTrue(_requires_vulkan_loader(cfg))

        cfg["env"]["carla"]["opengl"] = True
        self.assertFalse(_requires_vulkan_loader(cfg))

        cfg["env"]["carla"]["opengl"] = False
        cfg["rl_finetune"]["sim_actor"]["manage_servers"] = False
        self.assertFalse(_requires_vulkan_loader(cfg))

    def test_missing_vulkan_loader_has_actionable_error(self) -> None:
        with patch("b2d_rlinfra.finetuning.node_agent.ctypes.CDLL", side_effect=OSError("missing")):
            with self.assertRaisesRegex(RuntimeError, "VULKAN_LIB_DIR"):
                _validate_vulkan_loader()

    def test_preflight_out_of_range_node_rank_writes_diagnostics(self) -> None:
        cfg = _base_config()
        rl_cfg = cfg["rl_finetune"]
        rl_cfg.pop("num_collectors")
        rl_cfg.pop("collector_devices")
        rl_cfg["distributed"] = {
            "enabled": True,
            "num_nodes": 1,
            "collectors_per_node": 1,
            "learner_node_collectors": 1,
        }
        cfg["env"]["carla"].pop("num_envs")
        cfg["env"]["carla"]["gpu_id"] = "auto"
        raw = normalize_rl_finetune_config(cfg)
        topology = build_topology(raw)
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            with self.assertRaisesRegex(RuntimeError, "node_rank=2"):
                _run_preflight(raw_config=raw, topology=topology, node_rank=2, run_dir=run_dir)
            node_json = run_dir / "nodes" / "node_002.json"
            self.assertTrue(node_json.exists())
            data = json.loads(node_json.read_text(encoding="utf-8"))
            self.assertEqual(data["preflight_status"], "preflight_failed")

    def test_file_event_bus_validates_and_orders_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            events_dir = Path(tmp) / "events"
            writer = FileEventBus(events_dir, collector_id=3)
            reader = FileEventBus(events_dir)
            writer.emit({"type": "rollout", "payload": {"collector_id": 3, "episode_id": 1}})
            writer.emit({"type": "crash", "payload": {"collector_id": 3, "episode_id": 2}})
            polled = reader.poll(0)
            self.assertEqual([event["type"] for event in polled], ["rollout", "crash"])
            self.assertEqual(reader.poll(0), [])
            bad = events_dir / "collector_003" / "000003_rollout.json"
            bad.write_text("{not json", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "Failed to parse"):
                reader.poll(0)

    def test_file_control_plane_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            control = FileControlPlane(Path(tmp) / "control")
            control.publish_state("starting", reason="test")
            self.assertFalse(control.should_stop())
            control.heartbeat(phase="running", extra={"update_id": 2})
            state = control.read_state()
            self.assertNotIn("learner_heartbeat_seq", state)
            self.assertNotIn("update_id", state)
            heartbeat = control.read_heartbeat()
            self.assertEqual(heartbeat["learner_heartbeat_seq"], 1)
            self.assertEqual(heartbeat["update_id"], 2)
            control.request_stop(reason="done")
            self.assertTrue(control.should_stop())

    def test_learner_heartbeat_does_not_overwrite_failed_control_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            control = FileControlPlane(Path(tmp) / "control")
            control.publish_state("failed", reason="node_preflight_failed")
            control.heartbeat(phase="running", extra={"update_id": 2})
            state = control.read_state()
            self.assertEqual(state["state"], "failed")
            self.assertEqual(state["reason"], "node_preflight_failed")
            self.assertEqual(control.read_heartbeat()["learner_heartbeat_seq"], 1)

    def test_final_control_states_are_sticky_and_stopping_can_fail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            for final_state in ("finished", "failed"):
                with self.subTest(final_state=final_state):
                    control = FileControlPlane(Path(tmp) / final_state / "control")
                    control.publish_state(final_state, reason="final")
                    control.publish_state("running", reason="late_running")
                    control.publish_state("failed", reason="late_failed")
                    state = control.read_state()
                    self.assertEqual(state["state"], final_state)
                    self.assertEqual(state["reason"], "final")

            control = FileControlPlane(Path(tmp) / "stopping" / "control")
            control.publish_state("stopping", reason="requested")
            control.publish_state("running", reason="late_running")
            state = control.read_state()
            self.assertEqual(state["state"], "stopping")
            self.assertEqual(state["reason"], "requested")
            control.publish_state("failed", reason="late_failed")
            state = control.read_state()
            self.assertEqual(state["state"], "failed")
            self.assertEqual(state["reason"], "late_failed")

    def test_wait_for_stop_reads_learner_heartbeat_file(self) -> None:
        class CountingControl(FileControlPlane):
            def __init__(self, control_dir: Path):
                super().__init__(control_dir)
                self.heartbeat_reads = 0

            def read_heartbeat(self, *, default=None):
                self.heartbeat_reads += 1
                return super().read_heartbeat(default=default)

        with tempfile.TemporaryDirectory() as tmp:
            control = CountingControl(Path(tmp) / "control")
            control.publish_state("running", reason="test")
            _wait_for_stop(control, timeout=0.0)
            self.assertGreater(control.heartbeat_reads, 0)
            state = control.read_state()
            self.assertEqual(state["state"], "failed")
            self.assertEqual(state["reason"], "learner_heartbeat_timeout")

    def test_coordinator_exit_does_not_overwrite_failed_control_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            control = FileControlPlane(Path(tmp) / "control")
            control.publish_state("failed", reason="node_preflight_failed")
            publish_coordinator_exit_state(control, completed=False)
            state = control.read_state()
            self.assertEqual(state["state"], "failed")
            self.assertEqual(state["reason"], "node_preflight_failed")

    def test_coordinator_exit_without_completion_does_not_mark_stopping(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            control = FileControlPlane(Path(tmp) / "control")
            control.publish_state("running", reason="test")
            publish_coordinator_exit_state(control, completed=False)
            state = control.read_state()
            self.assertEqual(state["state"], "running")
            self.assertEqual(state["reason"], "test")

    def test_coordinator_exit_stop_request_marks_stopping(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            control = FileControlPlane(Path(tmp) / "control")
            control.publish_state("running", reason="test")
            publish_coordinator_exit_state(control, completed=False, stop_requested=True)
            state = control.read_state()
            self.assertEqual(state["state"], "stopping")
            self.assertEqual(state["reason"], "coordinator_exit")

    def test_manifest_dropped_stale_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = RolloutManifest(str(Path(tmp) / "manifest.jsonl"))
            rollout = Path(tmp) / "episode.rollout"
            rollout.write_bytes(b"x")
            manifest.add_ready(
                {
                    "file_path": str(rollout),
                    "collector_id": 0,
                    "episode_id": 0,
                    "num_steps": 4,
                    "policy_version": 1,
                }
            )
            records = manifest.records()
            self.assertEqual(records[str(rollout)]["state"], "ready")
            manifest.mark_dropped_stale(
                str(rollout),
                update_id=2,
                current_policy_version=5,
                rollout_policy_version=1,
                lag=4,
            )
            records = manifest.records()
            self.assertEqual(records[str(rollout)]["state"], "dropped_stale")

    def test_single_node_ready_records_drop_stale_rollouts(self) -> None:
        raw_cfg = _base_config()
        raw_cfg["rl_finetune"]["rollouts_per_update"] = 1
        cfg = normalize_rl_finetune_config(raw_cfg)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            rollout = tmp_path / "old_episode.rollout"
            rollout.write_bytes(b"x")
            manifest = RolloutManifest(str(tmp_path / "manifest.jsonl"))
            manifest.add_ready(
                {
                    "file_path": str(rollout),
                    "collector_id": 0,
                    "episode_id": 0,
                    "num_steps": 4,
                    "policy_version": 1,
                }
            )
            selected, stale_count, fresh_count = select_ready_with_policy_lag(
                manifest,
                max_files=cfg["rl_finetune"]["rollouts_per_update"],
                policy_version=5,
                update_id=2,
                max_policy_lag=cfg["rl_finetune"]["max_policy_lag"],
                stale_action=cfg["rl_finetune"]["stale_rollout_action"],
                cleanup_stale=cfg["rl_finetune"]["cleanup_stale_rollouts"],
            )
            self.assertEqual(selected, [])
            self.assertEqual(stale_count, 1)
            self.assertEqual(fresh_count, 0)
            records = manifest.records()
            self.assertEqual(records[str(rollout)]["state"], "dropped_stale")
            self.assertFalse(rollout.exists())

    def test_single_node_ready_records_warn_keeps_stale_rollouts(self) -> None:
        cfg = _base_config()
        cfg["rl_finetune"]["rollouts_per_update"] = 1
        cfg["rl_finetune"]["stale_rollout_action"] = "warn"
        cfg = normalize_rl_finetune_config(cfg)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            rollout = tmp_path / "old_episode.rollout"
            rollout.write_bytes(b"x")
            manifest = RolloutManifest(str(tmp_path / "manifest.jsonl"))
            manifest.add_ready(
                {
                    "file_path": str(rollout),
                    "collector_id": 0,
                    "episode_id": 0,
                    "num_steps": 4,
                    "policy_version": 1,
                }
            )
            selected, stale_count, fresh_count = select_ready_with_policy_lag(
                manifest,
                max_files=cfg["rl_finetune"]["rollouts_per_update"],
                policy_version=5,
                update_id=2,
                max_policy_lag=cfg["rl_finetune"]["max_policy_lag"],
                stale_action=cfg["rl_finetune"]["stale_rollout_action"],
                cleanup_stale=cfg["rl_finetune"]["cleanup_stale_rollouts"],
            )
            self.assertEqual(len(selected), 1)
            self.assertEqual(stale_count, 1)
            self.assertEqual(fresh_count, 0)
            records = manifest.records()
            self.assertEqual(records[str(rollout)]["state"], "ready")
            self.assertTrue(rollout.exists())


if __name__ == "__main__":
    unittest.main()
