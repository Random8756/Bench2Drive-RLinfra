from __future__ import annotations

import json
import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace

import torch


class PolicyAdapterContractTest(unittest.TestCase):
    def test_policy_adapter_exposes_only_consolidated_training_contract(self) -> None:
        from b2d_rlinfra.finetuning.policy_adapter import PolicyAdapter

        self.assertIn("learner_spec", PolicyAdapter.__abstractmethods__)
        self.assertIn("trainable_components", PolicyAdapter.__abstractmethods__)
        self.assertFalse(hasattr(PolicyAdapter, "evaluate_actions"))
        self.assertFalse(hasattr(PolicyAdapter, "trainable_parameters"))
        self.assertFalse(hasattr(PolicyAdapter, "aux_loss_weights"))
        self.assertFalse(hasattr(PolicyAdapter, "optimizer"))

    def test_learner_output_accepts_only_the_mapping_protocol(self) -> None:
        from b2d_rlinfra.finetuning.coordinator import _validate_learner_output

        valid = _validate_learner_output(
            {
                "log_probs": torch.zeros(2),
                "values": torch.ones(2),
                "entropy": torch.full((2,), 0.5),
                "aux_losses": {"penalty": torch.tensor(1.0)},
                "aux_logs": {"full_log_probs": torch.zeros(2, 3)},
            }
        )
        self.assertEqual(set(valid), {"log_probs", "values", "entropy", "aux_losses", "aux_logs"})

        @dataclass
        class _LegacyOutput:
            log_probs: torch.Tensor
            values: torch.Tensor

        invalid_outputs = (
            (torch.zeros(2), torch.zeros(2), torch.zeros(2)),
            _LegacyOutput(torch.zeros(2), torch.zeros(2)),
            [torch.zeros(2), torch.zeros(2)],
        )
        for output in invalid_outputs:
            with self.subTest(output_type=type(output).__name__):
                with self.assertRaisesRegex(TypeError, "must return a mapping"):
                    _validate_learner_output(output)

        with self.assertRaisesRegex(ValueError, "missing required field.*values"):
            _validate_learner_output({"log_probs": torch.zeros(2)})
        with self.assertRaisesRegex(ValueError, "unknown field.*extra"):
            _validate_learner_output(
                {"log_probs": torch.zeros(2), "values": torch.zeros(2), "extra": 1}
            )

    def test_mock_tiny_and_minddrive_modules_return_mapping_outputs(self) -> None:
        from b2d_rlinfra.finetuning.coordinator import _validate_learner_output
        from b2d_rlinfra.finetuning.minddrive_policy_adapter import MindDrivePolicyAdapter
        from b2d_rlinfra.finetuning.policy_adapter import (
            MockPolicyAdapter,
            TinyRGBDiscretePolicyAdapter,
        )

        mock = MockPolicyAdapter(
            obs_dim=4,
            action_dim=2,
            hidden_dim=8,
            learning_rate=1e-3,
            device=torch.device("cpu"),
        )
        mock_output = mock.learner_spec().module(
            {"policy_input_state": torch.zeros(3, 4), "actions": torch.zeros(3, 2)}
        )
        _validate_learner_output(mock_output)

        tiny = TinyRGBDiscretePolicyAdapter(
            num_cameras=1,
            image_height=2,
            image_width=2,
            channels=3,
            pool_grid=(1, 1),
            scalar_dim=1,
            action_dim=3,
            hidden_dim=8,
            learning_rate=1e-3,
            device=torch.device("cpu"),
        )
        tiny_output = tiny.learner_spec().module(
            {"policy_input_state": {"latent": torch.zeros(3, 4)}, "actions": torch.zeros(3)}
        )
        _validate_learner_output(tiny_output)

        class _DecisionExpert(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lora_weight = torch.nn.Parameter(torch.zeros(3))

        class _MindDriveModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.decision_expert = _DecisionExpert()
                self.value_net_pro = torch.nn.Linear(1, 1)

            def forward(self, model_input, *, return_loss, is_rl_training):
                batch_size = model_input["inputs_embeds"].shape[0]
                logits = self.decision_expert.lora_weight.unsqueeze(0).expand(batch_size, -1)
                log_probs = torch.log_softmax(logits, dim=-1)
                values = self.value_net_pro(model_input["inputs_embeds"].float()).view(-1)
                return log_probs, log_probs, values

        minddrive_model = _MindDriveModel()
        minddrive = MindDrivePolicyAdapter(
            model=minddrive_model,
            obs_adapter=None,
            device=torch.device("cpu"),
            optimizer=torch.optim.Adam(minddrive_model.parameters(), lr=1e-3),
            trainable_names=[
                "decision_expert.lora_weight",
                "value_net_pro.weight",
                "value_net_pro.bias",
            ],
            config={"precision": "bf16", "ppo": {"use_kl": False}},
        )
        minddrive_output = minddrive.learner_spec().module(
            {
                "policy_input_state": {
                    "inputs_embeds": torch.zeros(3, 1),
                    "new_input_ids": torch.zeros(3, 1, dtype=torch.long),
                },
                "actions": torch.zeros(3, dtype=torch.long),
            }
        )
        _validate_learner_output(minddrive_output)
        self.assertEqual(
            minddrive.trainable_components(),
            ["decision_expert_lora", "value_net_pro"],
        )

    def test_learner_spec_aux_loss_weight_controls_the_update(self) -> None:
        from b2d_rlinfra.finetuning.coordinator import RLPPOUpdater
        from b2d_rlinfra.finetuning.policy_adapter import LearnerSpec

        class _AuxModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor(1.0))

            def forward(self, batch):
                zeros = self.scale * torch.zeros_like(batch["advantages"].float())
                return {
                    "log_probs": zeros,
                    "values": zeros,
                    "entropy": zeros,
                    "aux_losses": {"penalty": self.scale.square()},
                    "aux_logs": {},
                }

        class _Dataset:
            samples_before_filter = 2
            samples_after_filter = 2
            filtered_sample_counts_by_outcome = {"success": 0, "failure": 2, "truncated": 0}
            filtered_advantage_mean_by_outcome = {"success": 0.0, "failure": 0.0, "truncated": 0.0}
            distributed_dropped_samples = 0

            def __len__(self):
                return 2

            def advantage_statistics(self, _learner_context):
                return 0.0, 0.0

            def iter_rank_batches(self, *args, **kwargs):
                yield {
                    "policy_input_state": torch.zeros(2, 1),
                    "actions": torch.zeros(2),
                    "old_action_log_probs": torch.zeros(2),
                    "advantages": torch.zeros(2),
                    "returns": torch.zeros(2),
                }

        module = _AuxModule()
        spec = LearnerSpec(
            module=module,
            optimizer=torch.optim.SGD(module.parameters(), lr=0.1),
            aux_loss_weights={"penalty": 2.0},
        )
        policy = SimpleNamespace(learner_spec=lambda: spec, set_train=lambda mode: None)
        updater = RLPPOUpdater(
            policy=policy,
            algo_config=SimpleNamespace(
                batch_size=2,
                n_epochs=1,
                clip_range=0.2,
                vf_coef=0.5,
                ent_coef=0.0,
                max_grad_norm=100.0,
                normalize_advantage=False,
                target_kl=None,
            ),
            device=torch.device("cpu"),
            learner_spec=spec,
        )
        updater.update(_Dataset())
        torch.testing.assert_close(module.scale.detach(), torch.tensor(0.6))

    def test_learner_spec_rejects_optimizer_parameter_mismatch(self) -> None:
        from b2d_rlinfra.finetuning.policy_adapter import LearnerSpec

        module = torch.nn.Linear(2, 1)
        other_module = torch.nn.Linear(2, 1)
        spec = LearnerSpec(
            module=module,
            optimizer=torch.optim.Adam(other_module.parameters()),
        )
        with self.assertRaisesRegex(ValueError, "optimizer param groups"):
            spec.validate()

    def test_resolver_requires_one_explicit_concrete_adapter(self) -> None:
        from b2d_rlinfra.finetuning.policy_adapter import (
            MockPolicyAdapter,
            PolicyAdapter,
            resolve_policy_adapter,
        )

        self.assertIs(resolve_policy_adapter({"type": "mock"}), MockPolicyAdapter)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            resolve_policy_adapter({})
        with self.assertRaisesRegex(ValueError, "exactly one"):
            resolve_policy_adapter({"type": "mock", "class_path": "example.Adapter"})
        with self.assertRaisesRegex(ValueError, "Unknown policy_adapter.type"):
            resolve_policy_adapter({"type": "missing"})
        with self.assertRaisesRegex(ValueError, "Invalid policy adapter class_path"):
            resolve_policy_adapter({"class_path": "NotDotted"})

        class _CustomAdapter(MockPolicyAdapter):
            pass

        class _NotAdapter:
            pass

        class _AbstractAdapter(PolicyAdapter):
            pass

        module_name = "_rl_finetune_test_policy_adapters"
        module = ModuleType(module_name)
        module.CustomAdapter = _CustomAdapter
        module.NotAdapter = _NotAdapter
        module.AbstractAdapter = _AbstractAdapter
        module.not_a_class = 3
        sys.modules[module_name] = module
        try:
            self.assertIs(
                resolve_policy_adapter({"class_path": f"{module_name}.CustomAdapter"}),
                _CustomAdapter,
            )
            with self.assertRaisesRegex(TypeError, "must subclass PolicyAdapter"):
                resolve_policy_adapter({"class_path": f"{module_name}.NotAdapter"})
            with self.assertRaisesRegex(TypeError, "must resolve to a class"):
                resolve_policy_adapter({"class_path": f"{module_name}.not_a_class"})
            with self.assertRaisesRegex(TypeError, "must be a concrete"):
                resolve_policy_adapter({"class_path": f"{module_name}.AbstractAdapter"})
            with self.assertRaisesRegex(ImportError, "was not found"):
                resolve_policy_adapter({"class_path": f"{module_name}.MissingAdapter"})
        finally:
            sys.modules.pop(module_name, None)

    def test_new_weight_metadata_and_old_checkpoint_compatibility(self) -> None:
        from b2d_rlinfra.finetuning.coordinator import Coordinator
        from b2d_rlinfra.finetuning.policy_adapter import MockPolicyAdapter
        from b2d_rlinfra.finetuning.weight_store import WeightStore

        adapter = MockPolicyAdapter(
            obs_dim=4,
            action_dim=2,
            hidden_dim=8,
            learning_rate=1e-3,
            device=torch.device("cpu"),
        )
        with tempfile.TemporaryDirectory() as tmp:
            store = WeightStore(tmp)
            store.publish(
                policy_version=2,
                base_checkpoint=None,
                trainable_state_dict=adapter.trainable_state_dict(),
                trainable_components=adapter.trainable_components(),
                dtype="torch.float32",
            )
            payload = store.load_latest(map_location="cpu")
            self.assertEqual(payload["trainable_components"], adapter.trainable_components())
            self.assertNotIn("trainable_modules", payload)
            latest = json.loads(store.latest_json.read_text(encoding="utf-8"))
            self.assertEqual(latest["trainable_components"], adapter.trainable_components())
            self.assertNotIn("trainable_modules", latest)

            old_checkpoint = Path(tmp) / "old_checkpoint.pt"
            torch.save(
                {
                    "policy_version": 7,
                    "trainable_state_dict": adapter.trainable_state_dict(),
                    "trainable_modules": ["model", "actor", "critic", "log_std"],
                },
                old_checkpoint,
            )
            restored = MockPolicyAdapter.load_initial(
                {"obs_dim": 4, "action_dim": 2, "hidden_dim": 8},
                str(old_checkpoint),
                torch.device("cpu"),
            )
            self.assertEqual(restored.policy_version, 7)
            for expected, actual in zip(
                adapter.learner_spec().module.parameters(),
                restored.learner_spec().module.parameters(),
            ):
                torch.testing.assert_close(expected, actual)

        class _CaptureWeightStore:
            latest_path = Path("policy_latest.pt")

            def publish(self, **kwargs):
                self.kwargs = kwargs

        capture = _CaptureWeightStore()
        fake_coordinator = SimpleNamespace(
            policy=adapter,
            updater=SimpleNamespace(
                learner_spec=adapter.learner_spec(),
                state_dict=lambda: {},
            ),
            policy_version=3,
            raw_config={"policy_adapter": {"checkpoint": "base.pt"}},
            weight_store=capture,
            config_path="config.yaml",
        )
        Coordinator._publish_latest(fake_coordinator)
        self.assertEqual(capture.kwargs["dtype"], "torch.float32")
        self.assertEqual(
            capture.kwargs["trainable_components"],
            adapter.trainable_components(),
        )
        self.assertNotIn("trainable_modules", capture.kwargs)

        checkpoint = Coordinator._checkpoint_payload(
            fake_coordinator,
            checkpoint_phase="post_update",
            last_update_id=1,
        )
        self.assertEqual(checkpoint["trainable_components"], adapter.trainable_components())
        self.assertNotIn("trainable_modules", checkpoint)

    def test_drivepi0_export_accepts_v1_and_v2_standalone_states(self) -> None:
        from b2d_rlinfra.finetuning.tools.export_drivepi0_checkpoint import (
            _extract_trainable_state,
        )

        for version in ("drivepi0_trainable_v1", "drivepi0_trainable_v2"):
            with self.subTest(version=version):
                payload = {
                    "drivepi0_action_expert": {"action_encoder.weight": torch.ones(1)},
                    "metadata": {"format": version},
                }
                self.assertIs(_extract_trainable_state(payload), payload)


if __name__ == "__main__":
    unittest.main()
