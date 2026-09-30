from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]

if os.name == "nt":
    # The Windows CUDA wheel in the validation environment is built without
    # libuv. Gloo's TCP rendezvous supports the legacy backend explicitly.
    os.environ.setdefault("USE_LIBUV", "0")


def _tree_index(value, indices):
    if isinstance(value, dict):
        return {key: _tree_index(item, indices) for key, item in value.items()}
    return np.asarray(value)[indices]


def _tree_tensor(value, device):
    if isinstance(value, dict):
        return {key: _tree_tensor(item, device) for key, item in value.items()}
    tensor = torch.as_tensor(value)
    return tensor.float().to(device) if tensor.dtype == torch.float64 else tensor.to(device)


def _trainable_parameters(adapter):
    return [
        param
        for param in adapter.learner_spec().module.parameters()
        if param.requires_grad
    ]


class _EagerReferenceDataset:
    def __init__(self, data):
        self.data = data
        self.samples_before_filter = len(data["advantages"])
        self.samples_after_filter = self.samples_before_filter
        self.filtered_sample_counts_by_outcome = {
            "success": 0,
            "failure": self.samples_after_filter,
            "truncated": 0,
        }
        self.filtered_advantage_mean_by_outcome = {
            "success": 0.0,
            "failure": float(np.asarray(data["advantages"]).mean()),
            "truncated": 0.0,
        }
        self.distributed_dropped_samples = 0

    def __len__(self):
        return self.samples_after_filter

    def advantage_statistics(self, _learner_context):
        advantages = np.asarray(self.data["advantages"], dtype=np.float64)
        return float(advantages.mean()), float(advantages.std(ddof=1))

    def iter_rank_batches(
        self,
        global_batch_size,
        *,
        rank,
        world_size,
        seed,
        epoch=0,
        shuffle=True,
        device=None,
    ):
        if rank != 0 or world_size != 1:
            raise ValueError("eager reference only supports the single-rank test")
        indices = np.arange(len(self), dtype=np.int64)
        if shuffle:
            np.random.default_rng(int(seed) + int(epoch)).shuffle(indices)
        for start in range(0, len(indices), int(global_batch_size)):
            selected = indices[start : start + int(global_batch_size)]
            yield _tree_tensor(_tree_index(self.data, selected), device)


def _write_rollout_plan(root: Path, *, count: int, mock_policy: bool) -> Path:
    from b2d_rlinfra.finetuning.policy_adapter import MockPolicyAdapter
    from b2d_rlinfra.finetuning.rollout_file_store import RolloutFileStore, write_update_index

    rng = np.random.default_rng(5)
    states = rng.normal(size=(count, 4)).astype(np.float32)
    if mock_policy:
        actions = rng.normal(size=(count, 2)).astype(np.float32)
        adapter = MockPolicyAdapter(
            obs_dim=4,
            action_dim=2,
            hidden_dim=8,
            learning_rate=1e-3,
            device=torch.device("cpu"),
            seed=17,
        )
        with torch.no_grad():
            initial = adapter.learner_spec().module(
                {"policy_input_state": states, "actions": actions}
            )
        values = initial["values"].numpy()
        old_log_probs = initial["log_probs"].numpy()
        returns = (initial["values"] + 0.2).numpy()
    else:
        actions = np.arange(count, dtype=np.int64)
        values = np.zeros(count, dtype=np.float32)
        old_log_probs = np.zeros(count, dtype=np.float32)
        returns = np.ones(count, dtype=np.float32)
    store = RolloutFileStore(str(root / "rollouts"))
    meta = store.write_episode(
        collector_id=0,
        episode_id=0,
        policy_version=0,
        actions=actions,
        rewards=np.zeros(count, dtype=np.float32),
        episode_starts=np.zeros(count, dtype=np.bool_),
        values=values,
        old_action_log_probs=old_log_probs,
        advantages=np.ones(count, dtype=np.float32),
        returns=returns,
        policy_input_state=states,
        terminated=True,
        truncated=False,
    )
    return write_update_index(str(root / "update_plan.json"), 0, [meta.file_path])


def _gloo_update_worker(rank: int, port: int, output_dir: str, index_path: str) -> None:
    from b2d_rlinfra.finetuning.coordinator import RLPPOUpdater
    from b2d_rlinfra.finetuning.learner_distributed import (
        init_local_process_group,
        wrap_training_module,
    )
    from b2d_rlinfra.finetuning.policy_adapter import MockPolicyAdapter
    from b2d_rlinfra.finetuning.rollout_file_dataset import RolloutFileDataset

    device = torch.device("cpu")
    context = init_local_process_group(
        rank=rank,
        world_size=2,
        device=device,
        backend="gloo",
        master_port=port,
        timeout_seconds=30.0,
    )
    try:
        adapter = MockPolicyAdapter(
            obs_dim=4,
            action_dim=2,
            hidden_dim=8,
            learning_rate=1e-3,
            device=device,
            seed=17,
        )
        spec = adapter.learner_spec()
        updater = RLPPOUpdater(
            policy=adapter,
            algo_config=SimpleNamespace(
                batch_size=4,
                n_epochs=1,
                clip_range=0.2,
                vf_coef=0.5,
                ent_coef=0.0,
                max_grad_norm=0.5,
                normalize_advantage=False,
                target_kl=None,
            ),
            device=device,
            learner_spec=spec,
            training_module=wrap_training_module(spec, context),
            learner_context=context,
        )
        with RolloutFileDataset(index_path) as dataset:
            stats = updater.update(dataset, sampler_seed=23)
        torch.save(
            {
                "params": [param.detach().clone() for param in _trainable_parameters(adapter)],
                "stats": stats,
            },
            Path(output_dir) / f"rank_{rank}.pt",
        )
    finally:
        torch.distributed.destroy_process_group()


def _gloo_target_kl_early_stop_worker(rank: int, port: int, output_dir: str, index_path: str) -> None:
    """Force target-KL skip on update 1, then run a second update under DDP."""
    from b2d_rlinfra.finetuning.coordinator import RLPPOUpdater
    from b2d_rlinfra.finetuning.learner_distributed import (
        init_local_process_group,
        wrap_training_module,
    )
    from b2d_rlinfra.finetuning.policy_adapter import MockPolicyAdapter
    from b2d_rlinfra.finetuning.rollout_file_dataset import RolloutFileDataset

    device = torch.device("cpu")
    context = init_local_process_group(
        rank=rank,
        world_size=2,
        device=device,
        backend="gloo",
        master_port=port,
        timeout_seconds=30.0,
    )
    try:
        # Seed differs from the rollout writer (seed=17) so pre-step approx_kl is large.
        adapter = MockPolicyAdapter(
            obs_dim=4,
            action_dim=2,
            hidden_dim=8,
            learning_rate=1e-3,
            device=device,
            seed=99,
        )
        spec = adapter.learner_spec()
        training_module = wrap_training_module(spec, context)
        # Extremely tight KL threshold so the first pre-step check always early-stops.
        early_algo = SimpleNamespace(
            batch_size=4,
            n_epochs=1,
            clip_range=0.2,
            vf_coef=0.5,
            ent_coef=0.0,
            max_grad_norm=0.5,
            normalize_advantage=False,
            target_kl=1e-12,
        )
        continue_algo = SimpleNamespace(
            batch_size=4,
            n_epochs=1,
            clip_range=0.2,
            vf_coef=0.5,
            ent_coef=0.0,
            max_grad_norm=0.5,
            normalize_advantage=False,
            target_kl=None,
        )
        with RolloutFileDataset(index_path) as dataset:
            early_updater = RLPPOUpdater(
                policy=adapter,
                algo_config=early_algo,
                device=device,
                learner_spec=spec,
                training_module=training_module,
                learner_context=context,
            )
            stats_early = early_updater.update(dataset, sampler_seed=23)
            second_updater = RLPPOUpdater(
                policy=adapter,
                algo_config=continue_algo,
                device=device,
                learner_spec=spec,
                training_module=training_module,
                learner_context=context,
            )
            stats_second = second_updater.update(dataset, sampler_seed=29)
        torch.save(
            {
                "params": [param.detach().clone() for param in _trainable_parameters(adapter)],
                "stats_early": stats_early,
                "stats_second": stats_second,
            },
            Path(output_dir) / f"rank_{rank}.pt",
        )
    finally:
        torch.distributed.destroy_process_group()


def _gloo_global_variance_worker(rank: int, port: int, output_dir: str) -> None:
    from b2d_rlinfra.finetuning.coordinator import RLPPOUpdater
    from b2d_rlinfra.finetuning.learner_distributed import init_local_process_group
    from b2d_rlinfra.finetuning.policy_adapter import MockPolicyAdapter

    device = torch.device("cpu")
    context = init_local_process_group(
        rank=rank,
        world_size=2,
        device=device,
        backend="gloo",
        master_port=port,
        timeout_seconds=30.0,
    )
    try:
        adapter = MockPolicyAdapter(
            obs_dim=4,
            action_dim=2,
            hidden_dim=8,
            learning_rate=1e-3,
            device=device,
            seed=17,
        )
        updater = RLPPOUpdater(
            policy=adapter,
            algo_config=SimpleNamespace(),
            device=device,
            learner_context=context,
        )
        returns = torch.full((2,), 0.0 if rank == 0 else 10.0)
        explained_variance = updater._global_explained_variance(returns, returns)
        torch.save(
            {"explained_variance": explained_variance},
            Path(output_dir) / f"collective_{rank}.pt",
        )
    finally:
        torch.distributed.destroy_process_group()


class LearnerContractTest(unittest.TestCase):
    def test_nested_policy_states_stack_and_round_trip(self) -> None:
        from b2d_rlinfra.finetuning.policy_adapter import stack_policy_states
        from b2d_rlinfra.finetuning.rollout_pack import RolloutPackReader, RolloutPackWriter

        states = [
            {
                "camera": {"features": np.asarray([index, index + 1], dtype=np.float32)},
                "mask": np.asarray([True, False]),
            }
            for index in range(3)
        ]
        stacked = stack_policy_states(states)
        self.assertEqual(stacked["camera"]["features"].shape, (3, 2))
        self.assertEqual(stacked["camera"]["features"].dtype, np.float32)

        with tempfile.TemporaryDirectory() as tmp:
            path = RolloutPackWriter.write(
                Path(tmp) / "nested.rollout",
                metadata={},
                sample_fields={"policy_input_state": stacked},
                num_steps=3,
            )
            with RolloutPackReader(path) as reader:
                np.testing.assert_array_equal(
                    reader.array(("policy_input_state", "camera", "features")),
                    np.asarray([[0, 1], [1, 2], [2, 3]], dtype=np.float32),
                )

        with self.assertRaisesRegex(ValueError, "mapping keys are inconsistent"):
            stack_policy_states(
                [
                    {"nested": {"left": np.asarray([1])}},
                    {"nested": {"right": np.asarray([2])}},
                ]
            )

    def test_policy_adapter_requires_explicit_learner_spec(self) -> None:
        from b2d_rlinfra.finetuning.policy_adapter import PolicyAdapter

        self.assertIn("learner_spec", PolicyAdapter.__abstractmethods__)

    def test_mock_adapter_learner_spec_covers_optimizer_parameters(self) -> None:
        from b2d_rlinfra.finetuning.policy_adapter import MockPolicyAdapter

        adapter = MockPolicyAdapter(
            obs_dim=4,
            action_dim=2,
            hidden_dim=8,
            learning_rate=1e-3,
            device=torch.device("cpu"),
        )
        spec = adapter.learner_spec()
        self.assertIs(spec, adapter.learner_spec())
        spec.validate()
        output = spec.module(
            {
                "policy_input_state": torch.zeros(3, 4),
                "actions": torch.zeros(3, 2),
            }
        )
        self.assertEqual(tuple(output["log_probs"].shape), (3,))

    def test_rank_sampler_drops_world_size_remainder(self) -> None:
        from b2d_rlinfra.finetuning.rollout_file_dataset import RolloutFileDataset

        with tempfile.TemporaryDirectory() as tmp:
            index_path = _write_rollout_plan(Path(tmp), count=8, mock_policy=False)
            with RolloutFileDataset(str(index_path)) as dataset:
                rank_actions = []
                for rank in range(3):
                    batches = list(
                        dataset.iter_rank_batches(
                            6,
                            rank=rank,
                            world_size=3,
                            seed=7,
                            shuffle=False,
                            device=torch.device("cpu"),
                        )
                    )
                    rank_actions.append(
                        torch.cat([batch["actions"] for batch in batches]).tolist()
                    )
                self.assertEqual(dataset.distributed_dropped_samples, 2)
        self.assertEqual(rank_actions, [[0, 3], [1, 4], [2, 5]])

    def test_rank_sampler_drops_small_tail_consistently(self) -> None:
        from b2d_rlinfra.finetuning.rollout_file_dataset import RolloutFileDataset

        with tempfile.TemporaryDirectory() as tmp:
            index_path = _write_rollout_plan(Path(tmp), count=15, mock_policy=False)
            with RolloutFileDataset(str(index_path)) as dataset:
                rank_batch_sizes = []
                rank_actions = []
                for rank in range(3):
                    batches = list(
                        dataset.iter_rank_batches(
                            12,
                            rank=rank,
                            world_size=3,
                            seed=7,
                            shuffle=False,
                            device=torch.device("cpu"),
                        )
                    )
                    rank_batch_sizes.append([len(batch["actions"]) for batch in batches])
                    rank_actions.append(
                        torch.cat([batch["actions"] for batch in batches]).tolist()
                    )

                dropped_samples = dataset.distributed_dropped_samples

        self.assertEqual(rank_batch_sizes, [[4], [4], [4]])
        self.assertEqual(
            rank_actions,
            [
                [0, 3, 6, 9],
                [1, 4, 7, 10],
                [2, 5, 8, 11],
            ],
        )
        self.assertEqual(dropped_samples, 3)

    def test_rank_sampler_keeps_half_sized_tail(self) -> None:
        from b2d_rlinfra.finetuning.rollout_file_dataset import RolloutFileDataset

        with tempfile.TemporaryDirectory() as tmp:
            index_path = _write_rollout_plan(Path(tmp), count=18, mock_policy=False)
            with RolloutFileDataset(str(index_path)) as dataset:
                rank_batch_sizes = []
                for rank in range(3):
                    batches = list(
                        dataset.iter_rank_batches(
                            12,
                            rank=rank,
                            world_size=3,
                            seed=7,
                            shuffle=False,
                            device=torch.device("cpu"),
                        )
                    )
                    rank_batch_sizes.append([len(batch["actions"]) for batch in batches])

                dropped_samples = dataset.distributed_dropped_samples

        self.assertEqual(rank_batch_sizes, [[4, 2], [4, 2], [4, 2]])
        self.assertEqual(dropped_samples, 0)

    def test_rank_sampler_drops_tail_below_half_for_odd_batch_size(self) -> None:
        from b2d_rlinfra.finetuning.rollout_file_dataset import RolloutFileDataset

        with tempfile.TemporaryDirectory() as tmp:
            index_path = _write_rollout_plan(Path(tmp), count=7, mock_policy=False)
            with RolloutFileDataset(str(index_path)) as dataset:
                batches = list(
                    dataset.iter_rank_batches(
                        5,
                        rank=0,
                        world_size=1,
                        seed=7,
                        shuffle=False,
                        device=torch.device("cpu"),
                    )
                )
                batch_sizes = [len(batch["actions"]) for batch in batches]
                dropped_samples = dataset.distributed_dropped_samples

        self.assertEqual(batch_sizes, [5])
        self.assertEqual(dropped_samples, 2)

    def test_dataset_maps_filtered_samples_across_episode_packs(self) -> None:
        from b2d_rlinfra.finetuning.learner_distributed import LearnerContext
        from b2d_rlinfra.finetuning.rollout_file_dataset import RolloutFileDataset
        from b2d_rlinfra.finetuning.rollout_file_store import RolloutFileStore, write_update_index

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = RolloutFileStore(str(root / "rollouts"))
            files = []
            offset = 0
            for episode_id, count in enumerate((3, 5)):
                meta = store.write_episode(
                    collector_id=0,
                    episode_id=episode_id,
                    policy_version=0,
                    actions=np.arange(offset, offset + count, dtype=np.int64),
                    rewards=np.zeros(count, dtype=np.float32),
                    episode_starts=np.arange(count) == 0,
                    values=np.zeros(count, dtype=np.float32),
                    old_action_log_probs=np.zeros(count, dtype=np.float32),
                    advantages=np.arange(offset, offset + count, dtype=np.float32),
                    returns=np.ones(count, dtype=np.float32),
                    policy_input_state=np.zeros((count, 2), dtype=np.float32),
                    terminated=True,
                    truncated=False,
                )
                files.append(meta.file_path)
                offset += count
            plan = write_update_index(
                str(root / "plan.json"),
                0,
                files,
                sample_filter={"failure_terminal_window_steps": 2},
            )
            with RolloutFileDataset(str(plan)) as dataset:
                self.assertFalse(hasattr(dataset, "_data"))
                batches = list(
                    dataset.iter_rank_batches(
                        4,
                        rank=0,
                        world_size=1,
                        seed=0,
                        shuffle=False,
                        device=torch.device("cpu"),
                    )
                )
                actions = torch.cat([batch["actions"] for batch in batches]).tolist()
                mean, std = dataset.advantage_statistics(
                    LearnerContext(device=torch.device("cpu"))
                )
            self.assertEqual(actions, [1, 2, 6, 7])
            self.assertAlmostEqual(mean, 4.0)
            self.assertAlmostEqual(std, float(np.std([1, 2, 6, 7], ddof=1)), places=6)

    def test_single_rank_update_matches_eager_reference(self) -> None:
        from b2d_rlinfra.finetuning.coordinator import RLPPOUpdater
        from b2d_rlinfra.finetuning.learner_distributed import LearnerContext
        from b2d_rlinfra.finetuning.policy_adapter import MockPolicyAdapter
        from b2d_rlinfra.finetuning.rollout_file_dataset import RolloutFileDataset
        from b2d_rlinfra.finetuning.rollout_pack import RolloutPackReader

        algo = SimpleNamespace(
            batch_size=4,
            n_epochs=1,
            clip_range=0.2,
            vf_coef=0.5,
            ent_coef=0.0,
            max_grad_norm=0.5,
            normalize_advantage=True,
            target_kl=None,
        )
        device = torch.device("cpu")
        context = LearnerContext(device=device)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = _write_rollout_plan(root, count=8, mock_policy=True)
            source_path = json.loads(plan.read_text(encoding="utf-8"))["entries"][0]["source_path"]
            with RolloutPackReader(source_path) as reader:
                eager_payload = reader.materialize()
            sample_keys = (
                "actions",
                "rewards",
                "episode_starts",
                "values",
                "old_action_log_probs",
                "advantages",
                "returns",
                "policy_input_state",
            )
            eager = _EagerReferenceDataset(
                {key: eager_payload[key] for key in sample_keys}
            )
            mmap_adapter = MockPolicyAdapter(
                obs_dim=4,
                action_dim=2,
                hidden_dim=8,
                learning_rate=1e-3,
                device=device,
                seed=17,
            )
            eager_adapter = MockPolicyAdapter(
                obs_dim=4,
                action_dim=2,
                hidden_dim=8,
                learning_rate=1e-3,
                device=device,
                seed=17,
            )
            mmap_updater = RLPPOUpdater(
                policy=mmap_adapter,
                algo_config=algo,
                device=device,
                learner_context=context,
            )
            eager_updater = RLPPOUpdater(
                policy=eager_adapter,
                algo_config=algo,
                device=device,
                learner_context=context,
            )
            with RolloutFileDataset(str(plan)) as dataset:
                mmap_updater.update(dataset, sampler_seed=31)
            eager_updater.update(eager, sampler_seed=31)
        for mmap_param, eager_param in zip(
            _trainable_parameters(mmap_adapter), _trainable_parameters(eager_adapter)
        ):
            torch.testing.assert_close(mmap_param, eager_param, rtol=0.0, atol=0.0)

    def test_two_rank_gloo_update_stays_synchronized(self) -> None:
        from b2d_rlinfra.finetuning.learner_distributed import find_free_local_port

        with tempfile.TemporaryDirectory() as tmp:
            index_path = _write_rollout_plan(Path(tmp), count=7, mock_policy=True)
            torch.multiprocessing.spawn(
                _gloo_update_worker,
                args=(find_free_local_port(), tmp, str(index_path)),
                nprocs=2,
                join=True,
            )
            rank0 = torch.load(Path(tmp) / "rank_0.pt", weights_only=False)
            rank1 = torch.load(Path(tmp) / "rank_1.pt", weights_only=False)
        self.assertEqual(rank0["stats"]["batches"], 1.0)
        self.assertEqual(rank0["stats"]["samples_trained"], 4.0)
        self.assertEqual(rank0["stats"]["dropped_samples"], 3.0)
        for left, right in zip(rank0["params"], rank1["params"]):
            torch.testing.assert_close(left, right)

    def test_two_rank_explained_variance_uses_global_moments(self) -> None:
        from b2d_rlinfra.finetuning.learner_distributed import find_free_local_port

        with tempfile.TemporaryDirectory() as tmp:
            torch.multiprocessing.spawn(
                _gloo_global_variance_worker,
                args=(find_free_local_port(), tmp),
                nprocs=2,
                join=True,
            )
            results = [
                torch.load(Path(tmp) / f"collective_{rank}.pt", weights_only=False)
                for rank in range(2)
            ]
        for result in results:
            self.assertAlmostEqual(result["explained_variance"], 1.0, places=7)

    def test_two_rank_target_kl_early_stop_allows_second_update(self) -> None:
        """DDP target-KL skip must finish reduction so the next update can run."""
        from b2d_rlinfra.finetuning.learner_distributed import find_free_local_port

        with tempfile.TemporaryDirectory() as tmp:
            index_path = _write_rollout_plan(Path(tmp), count=7, mock_policy=True)
            torch.multiprocessing.spawn(
                _gloo_target_kl_early_stop_worker,
                args=(find_free_local_port(), tmp, str(index_path)),
                nprocs=2,
                join=True,
            )
            rank0 = torch.load(Path(tmp) / "rank_0.pt", weights_only=False)
            rank1 = torch.load(Path(tmp) / "rank_1.pt", weights_only=False)
        self.assertEqual(rank0["stats_early"]["batches"], 0.0)
        self.assertEqual(rank0["stats_early"]["skipped_by_target_kl"], 1.0)
        self.assertGreater(rank0["stats_second"]["batches"], 0.0)
        for left, right in zip(rank0["params"], rank1["params"]):
            torch.testing.assert_close(left, right)

    def test_drivepi0_learner_forward_clears_cached_action_distribution(self) -> None:
        """target-KL without backward must not leave Normal(mean) on the runtime."""
        from b2d_rlinfra.finetuning.drivepi0_policy_adapter import DrivePi0PolicyAdapter
        from b2d_rlinfra.learning.utils.distributions import DiagGaussianDistribution

        class _StubRuntime:
            def __init__(self):
                self.device = torch.device("cpu")
                self.rgb_key = "rgb"
                self.state_key = "state"
                self.action_dist = DiagGaussianDistribution(2)
                self.drivepi0_model = torch.nn.Linear(4, 2)
                self.value_net = torch.nn.Linear(4, 1)
                self.log_std = torch.nn.Parameter(torch.zeros(2))
                params = (
                    list(self.drivepi0_model.parameters())
                    + list(self.value_net.parameters())
                    + [self.log_std]
                )
                self.optimizer = torch.optim.Adam(params, lr=1e-3)

            def set_train(self, mode: bool = True) -> None:
                self.drivepi0_model.train(mode)
                self.value_net.train(mode)

            def distribution_from_batch(self, policy_input_state):
                state = torch.as_tensor(policy_input_state, dtype=torch.float32, device=self.device)
                if state.ndim == 1:
                    state = state.unsqueeze(0)
                mean = self.drivepi0_model(state)
                return self.action_dist.proba_distribution(mean, self.log_std)

            def value_from_state(self, policy_input_state):
                state = torch.as_tensor(policy_input_state, dtype=torch.float32, device=self.device)
                if state.ndim == 1:
                    state = state.unsqueeze(0)
                return self.value_net(state).squeeze(-1)

            def trainable_parameters(self):
                yield from self.drivepi0_model.parameters()
                yield from self.value_net.parameters()
                yield self.log_std

            def trainable_components(self):
                return ["drivepi0_model", "value_net", "log_std"]

            def trainable_state_dict(self):
                return {
                    "drivepi0_model": self.drivepi0_model.state_dict(),
                    "value_net": self.value_net.state_dict(),
                    "log_std": self.log_std.detach().clone(),
                }

            def load_trainable_state_dict(self, state_dict):
                self.drivepi0_model.load_state_dict(state_dict["drivepi0_model"])
                self.value_net.load_state_dict(state_dict["value_net"])
                self.log_std.data.copy_(state_dict["log_std"])

        runtime = _StubRuntime()
        adapter = DrivePi0PolicyAdapter(
            runtime=runtime,
            config={},
            trajectory_shape=(1, 2),
        )
        batch = {
            "policy_input_state": torch.randn(3, 4),
            "actions": torch.randn(3, 2),
        }
        eval_out = adapter.learner_spec().module(batch)
        self.assertIsNone(runtime.action_dist.distribution)
        self.assertTrue(eval_out["log_probs"].requires_grad)
        # Early-stop path: no backward; a second eval must not keep the prior Normal.
        eval_out2 = adapter.learner_spec().module(batch)
        self.assertIsNone(runtime.action_dist.distribution)
        self.assertEqual(tuple(eval_out2["log_probs"].shape), (3,))

    def test_cached_diag_gaussian_distribution_retains_graph_until_cleared(self) -> None:
        """Document the OOM holder: stateful Normal(mean) keeps the autograd graph."""
        from b2d_rlinfra.learning.utils.distributions import DiagGaussianDistribution

        holder = DiagGaussianDistribution(2)
        leaf = torch.randn(4, 8, requires_grad=True)
        weight = torch.randn(8, 2)
        mean = leaf @ weight
        holder.proba_distribution(mean, torch.zeros(2))
        del mean
        self.assertIsNotNone(holder.distribution)
        self.assertIsNotNone(holder.distribution.mean.grad_fn)
        holder.distribution = None
        self.assertIsNone(holder.distribution)


if __name__ == "__main__":
    unittest.main()
