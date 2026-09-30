from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from b2d_rlinfra.learning.algorithms.base_algorithm import BaseAlgorithm


class _TinyPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(3, 2)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)


class _TinyAlgorithm(BaseAlgorithm):
    def __init__(
        self,
        policy=_TinyPolicy,
        env=None,
        learning_rate: float = 1e-2,
        restored_counter: int = 0,
        **kwargs,
    ) -> None:
        self.restored_counter = int(restored_counter)
        self._n_updates = 0
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            **kwargs,
        )

    def _setup_model(self) -> None:
        self.policy = self.policy_class().to(self.device)

    def learn(self, total_timesteps: int, *args, **kwargs):
        self._total_timesteps = int(total_timesteps)
        return self

    def _get_save_data(self):
        return {"restored_counter": self.restored_counter}

    def _load_save_data(self, metadata) -> None:
        self.restored_counter = int(metadata.get("restored_counter", 0))


def _take_optimizer_step(model: _TinyAlgorithm) -> None:
    assert model.policy is not None
    inputs = torch.tensor([[1.0, -2.0, 0.5]], device=model.device)
    loss = model.policy.projection(inputs).square().mean()
    model.policy.optimizer.zero_grad(set_to_none=True)
    loss.backward()
    model.policy.optimizer.step()


def test_baseline_checkpoint_round_trips_state_optimizer_and_metadata(tmp_path: Path) -> None:
    model = _TinyAlgorithm(
        device="cpu",
        config={"algorithm": {"name": "contract"}},
        seed=17,
    )
    _take_optimizer_step(model)
    model.num_timesteps = 123
    model._total_timesteps = 456
    model._episode_num = 7
    model._n_updates = 9
    model._scenario_episode_counts = {"ScenarioA": 3}
    model.restored_counter = 11

    assert model.policy is not None
    expected_policy = {
        name: tensor.detach().clone()
        for name, tensor in model.policy.state_dict().items()
    }
    expected_optimizer = model.policy.optimizer.state_dict()

    checkpoint = tmp_path / "checkpoint"
    model.save(checkpoint)

    restored = _TinyAlgorithm.load(
        checkpoint,
        policy=_TinyPolicy,
        device="cpu",
        learning_rate=1e-2,
    )

    assert restored.policy is not None
    for name, expected in expected_policy.items():
        torch.testing.assert_close(restored.policy.state_dict()[name], expected)
    assert restored.policy.optimizer.state_dict()["param_groups"] == expected_optimizer["param_groups"]
    restored_states = list(restored.policy.optimizer.state_dict()["state"].values())
    expected_states = list(expected_optimizer["state"].values())
    assert len(restored_states) == len(expected_states) > 0
    for restored_state, expected_state in zip(restored_states, expected_states):
        assert restored_state.keys() == expected_state.keys()
        for key, expected_value in expected_state.items():
            actual_value = restored_state[key]
            if torch.is_tensor(expected_value):
                torch.testing.assert_close(actual_value, expected_value)
            else:
                assert actual_value == expected_value

    assert restored.num_timesteps == 123
    assert restored._total_timesteps == 456
    assert restored._episode_num == 7
    assert restored._n_updates == 9
    assert restored._scenario_episode_counts == {"ScenarioA": 3}
    assert restored.restored_counter == 11

    metadata = json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["policy_class"] == "_TinyPolicy"
    assert metadata["restored_counter"] == 11
