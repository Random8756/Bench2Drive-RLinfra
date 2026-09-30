"""Policy adapter interface for rl finetune collection and PPO update."""

from __future__ import annotations

import importlib
import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple, Type, TypedDict

import numpy as np
import torch


def to_numpy_tree(value: Any) -> Any:
    """Convert tensors/lists/scalars in a policy state to numpy-friendly values."""
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        try:
            return tensor.numpy()
        except TypeError:
            if tensor.is_floating_point():
                return tensor.float().numpy()
            raise
    if isinstance(value, np.ndarray):
        return np.array(value, copy=True)
    if isinstance(value, Mapping):
        return {k: to_numpy_tree(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return np.asarray(value)
    return np.asarray(value)


def stack_policy_states(states: List[Any]) -> Any:
    if not states:
        raise ValueError("cannot stack an empty policy state list")
    first = states[0]
    if isinstance(first, Mapping):
        keys = tuple(first.keys())
        expected_keys = set(keys)
        for state in states:
            if not isinstance(state, Mapping) or set(state.keys()) != expected_keys:
                raise ValueError("policy state mapping keys are inconsistent across the episode")
        return {
            key: stack_policy_states([state[key] for state in states])
            for key in keys
        }
    if any(isinstance(state, Mapping) for state in states[1:]):
        raise ValueError("policy state structure is inconsistent across the episode")
    try:
        return np.stack([to_numpy_tree(state) for state in states], axis=0)
    except ValueError as exc:
        raise ValueError("policy state values cannot be stacked across the episode") from exc


@dataclass
class PolicyStep:
    """One collector step with explicit train-action/env-action separation."""

    policy_input_state: Any
    train_action: Any
    env_action: Any
    value: float
    old_action_log_prob: float
    action_logprob_info: Dict[str, Any] = field(default_factory=dict)


class _RequiredLearnerOutput(TypedDict):
    log_probs: torch.Tensor
    values: torch.Tensor


class LearnerOutput(_RequiredLearnerOutput, total=False):
    """Mapping returned by ``LearnerSpec.module.forward``.

    In DDP, ``aux_losses`` keys and optional ``aux_logs`` presence are a
    static adapter contract: they must not vary by rank, batch, or sample
    content. An inactive loss must remain present as a zero tensor rather
    than disappearing or becoming ``None``.
    """

    entropy: Optional[torch.Tensor]
    aux_losses: Mapping[str, torch.Tensor]
    aux_logs: Mapping[str, Any]


@dataclass(frozen=True)
class DDPOptions:
    """Adapter-owned DDP behavior for its concrete training graph."""

    find_unused_parameters: bool = False
    broadcast_buffers: bool = True
    gradient_as_bucket_view: bool = True
    static_graph: bool = False


@dataclass(frozen=True)
class LearnerSpec:
    """Strict learner contract implemented by every policy adapter.

    ``module`` is the exact graph invoked by PPO and wrapped by DDP.  Keeping
    this adapter-owned avoids guessing which submodules a third-party model
    trains or relying on attribute-name conventions.
    """

    module: torch.nn.Module
    optimizer: torch.optim.Optimizer
    precision: Optional[str] = None
    ddp_options: DDPOptions = field(default_factory=DDPOptions)
    aux_loss_weights: Mapping[str, float] = field(default_factory=dict)

    def validate(self) -> None:
        module_params = {id(param) for param in self.module.parameters() if param.requires_grad}
        optimizer_params = {
            id(param)
            for group in self.optimizer.param_groups
            for param in group["params"]
            if param.requires_grad
        }
        if not module_params:
            raise ValueError("PolicyAdapter.learner_spec().module has no trainable parameters")
        if optimizer_params != module_params:
            raise ValueError(
                "PolicyAdapter learner contract mismatch: optimizer param groups and "
                "learner_spec().module.parameters() must describe the same parameters "
                f"(optimizer={len(optimizer_params)}, module={len(module_params)})"
            )


class PolicyAdapter(ABC):
    """Minimal contract between collectors, rollout files, and PPO update."""

    policy_version: int = 0

    @classmethod
    @abstractmethod
    def load_initial(
        cls,
        config: Mapping[str, Any],
        checkpoint: Optional[str],
        device: torch.device,
    ) -> "PolicyAdapter":
        """Load the base model and trainable modules."""

    @abstractmethod
    def collect_step(self, obs: Any) -> PolicyStep:
        """Map one observation to a :class:`PolicyStep` during collection."""

    def on_episode_start(self, info: Mapping[str, Any]) -> None:
        """Optional hook called once before the first action of an episode."""

    @abstractmethod
    def value_from_obs(self, obs: Any) -> float:
        """Return scalar bootstrap value for an observation."""

    @abstractmethod
    def learner_spec(self) -> LearnerSpec:
        """Return the module/optimizer contract used by the learner.

        There is intentionally no compatibility bridge: every adapter must
        explicitly define its trainable forward graph.
        """

    @abstractmethod
    def trainable_state_dict(self) -> Dict[str, Any]:
        """Return trainable-only weights/delta."""

    @abstractmethod
    def load_trainable_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load trainable-only weights/delta."""

    @abstractmethod
    def trainable_components(self) -> List[str]:
        """Logical trainable components described by the published payload."""

    def set_train(self, mode: bool = True) -> None:
        """Optional train/eval hook."""


class _MockLearnerModule(torch.nn.Module):
    def __init__(self, model, actor, critic, log_std, device: torch.device):
        super().__init__()
        self.model = model
        self.actor = actor
        self.critic = critic
        self.log_std = log_std
        self.device = torch.device(device)

    def forward(self, batch: Dict[str, Any]) -> LearnerOutput:
        state = batch["policy_input_state"]
        if isinstance(state, Mapping):
            state = state.get("latent", state.get("vector"))
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        if state.ndim == 1:
            state = state.unsqueeze(0)
        hidden = self.model(state)
        mean = self.actor(hidden)
        values = self.critic(hidden).squeeze(-1)
        dist = torch.distributions.Normal(mean, self.log_std.exp().expand_as(mean))
        actions = torch.as_tensor(batch["actions"], dtype=torch.float32, device=self.device)
        if actions.ndim == 1:
            actions = actions.unsqueeze(-1)
        return {
            "log_probs": dist.log_prob(actions).sum(dim=-1),
            "values": values,
            "entropy": dist.entropy().sum(dim=-1),
        }


class MockPolicyAdapter(PolicyAdapter):
    """Small Gaussian actor-critic used for local integration tests."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int,
        learning_rate: float,
        device: torch.device,
        seed: int = 0,
    ):
        torch.manual_seed(seed)
        self.device = torch.device(device)
        self.encoder_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.encoder_dim, int(hidden_dim)),
            torch.nn.Tanh(),
        ).to(self.device)
        self.actor = torch.nn.Linear(int(hidden_dim), self.action_dim).to(self.device)
        self.critic = torch.nn.Linear(int(hidden_dim), 1).to(self.device)
        self.log_std = torch.nn.Parameter(torch.full((self.action_dim,), -0.5, device=self.device))
        self._learner_module = _MockLearnerModule(
            self.model, self.actor, self.critic, self.log_std, self.device
        )
        optimizer = torch.optim.Adam(
            list(self.model.parameters())
            + list(self.actor.parameters())
            + list(self.critic.parameters())
            + [self.log_std],
            lr=float(learning_rate),
        )
        self._learner_spec = LearnerSpec(module=self._learner_module, optimizer=optimizer)
        self.policy_version = 0
        self._learner_spec.validate()

    @classmethod
    def load_initial(
        cls,
        config: Mapping[str, Any],
        checkpoint: Optional[str],
        device: torch.device,
    ) -> "MockPolicyAdapter":
        cfg = dict(config or {})
        adapter = cls(
            obs_dim=int(cfg.get("obs_dim", 4)),
            action_dim=int(cfg.get("action_dim", 2)),
            hidden_dim=int(cfg.get("hidden_dim", 32)),
            learning_rate=float(cfg.get("learning_rate", 3e-4)),
            device=device,
            seed=int(cfg.get("seed", 0)),
        )
        if checkpoint:
            payload = torch.load(checkpoint, map_location=device)
            state = payload.get("trainable_state_dict", payload)
            adapter.load_trainable_state_dict(state)
            adapter.policy_version = int(payload.get("policy_version", 0))
        return adapter

    def _obs_vector(self, obs: Any) -> np.ndarray:
        if isinstance(obs, Mapping):
            if "vector" in obs:
                obs = obs["vector"]
            elif "observation" in obs:
                obs = obs["observation"]
        vector = np.asarray(obs, dtype=np.float32).reshape(-1)
        if vector.shape[0] != self.encoder_dim:
            raise ValueError(f"expected obs_dim={self.encoder_dim}, got {vector.shape[0]}")
        return vector

    def _state_tensor(self, policy_input_state: Any) -> torch.Tensor:
        if isinstance(policy_input_state, Mapping):
            policy_input_state = policy_input_state.get("latent", policy_input_state.get("vector"))
        tensor = torch.as_tensor(policy_input_state, dtype=torch.float32, device=self.device)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        return tensor

    def _distribution_and_value(self, policy_input_state: Any):
        state = self._state_tensor(policy_input_state)
        hidden = self.model(state)
        mean = self.actor(hidden)
        value = self.critic(hidden).squeeze(-1)
        std = self.log_std.exp().expand_as(mean)
        return torch.distributions.Normal(mean, std), value

    @torch.no_grad()
    def collect_step(self, obs: Any) -> PolicyStep:
        self.model.eval()
        policy_input_state = self._obs_vector(obs)
        dist, value = self._distribution_and_value(policy_input_state)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        action_np = action.squeeze(0).detach().cpu().numpy().astype(np.float32)
        return PolicyStep(
            policy_input_state=policy_input_state,
            train_action=action_np,
            env_action=action_np,
            value=float(value.squeeze(0).detach().cpu().item()),
            old_action_log_prob=float(log_prob.squeeze(0).detach().cpu().item()),
            action_logprob_info={
                "mean": dist.mean.squeeze(0).detach().cpu().numpy(),
                "log_std": self.log_std.detach().cpu().numpy(),
            },
        )

    @torch.no_grad()
    def value_from_obs(self, obs: Any) -> float:
        _, value = self._distribution_and_value(self._obs_vector(obs))
        return float(value.squeeze(0).detach().cpu().item())

    def learner_spec(self) -> LearnerSpec:
        return self._learner_spec

    def trainable_state_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model.state_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "log_std": self.log_std.detach().cpu(),
        }

    def load_trainable_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.model.load_state_dict(state_dict["model"])
        self.actor.load_state_dict(state_dict["actor"])
        self.critic.load_state_dict(state_dict["critic"])
        with torch.no_grad():
            self.log_std.copy_(state_dict["log_std"].to(self.device))

    def trainable_components(self) -> List[str]:
        return ["model", "actor", "critic", "log_std"]

    def set_train(self, mode: bool = True) -> None:
        self.model.train(mode)
        self.actor.train(mode)
        self.critic.train(mode)


class TinyRGBDiscretePolicyAdapter(PolicyAdapter):
    """Small categorical actor-critic for RGB rl finetune smoke runs.

    The adapter consumes the real RGB observation layout but stores only a
    compact pooled latent in rollout files.
    """

    def __init__(
        self,
        *,
        num_cameras: int,
        image_height: int,
        image_width: int,
        channels: int,
        pool_grid: Tuple[int, int],
        scalar_dim: int,
        action_dim: int,
        hidden_dim: int,
        learning_rate: float,
        device: torch.device,
        rgb_key: str = "rgb",
        scalar_key: str = "scalars",
        seed: int = 0,
    ):
        torch.manual_seed(seed)
        self.device = torch.device(device)
        self.num_cameras = int(num_cameras)
        self.image_height = int(image_height)
        self.image_width = int(image_width)
        self.channels = int(channels)
        self.pool_grid = (int(pool_grid[0]), int(pool_grid[1]))
        self.scalar_dim = int(scalar_dim)
        self.action_dim = int(action_dim)
        self.rgb_key = str(rgb_key)
        self.scalar_key = str(scalar_key)
        self.encoder_dim = (
            self.num_cameras
            * self.pool_grid[0]
            * self.pool_grid[1]
            * self.channels
            + self.scalar_dim
        )
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.encoder_dim, int(hidden_dim)),
            torch.nn.Tanh(),
            torch.nn.Linear(int(hidden_dim), int(hidden_dim)),
            torch.nn.Tanh(),
        ).to(self.device)
        self.actor = torch.nn.Linear(int(hidden_dim), self.action_dim).to(self.device)
        self.critic = torch.nn.Linear(int(hidden_dim), 1).to(self.device)
        self._learner_module = _TinyRGBLearnerModule(
            self.model, self.actor, self.critic, self.device
        )
        optimizer = torch.optim.Adam(
            self._learner_module.parameters(),
            lr=float(learning_rate),
        )
        self._learner_spec = LearnerSpec(module=self._learner_module, optimizer=optimizer)
        self.policy_version = 0
        self._learner_spec.validate()

    @classmethod
    def load_initial(
        cls,
        config: Mapping[str, Any],
        checkpoint: Optional[str],
        device: torch.device,
    ) -> "TinyRGBDiscretePolicyAdapter":
        cfg = dict(config or {})
        pool_grid = tuple(cfg.get("pool_grid", (3, 5)))
        if len(pool_grid) != 2:
            raise ValueError("tiny_rgb_discrete pool_grid must contain [height_bins, width_bins]")
        adapter = cls(
            num_cameras=int(cfg.get("num_cameras", 6)),
            image_height=int(cfg.get("image_height", 900)),
            image_width=int(cfg.get("image_width", 1600)),
            channels=int(cfg.get("channels", 3)),
            pool_grid=(int(pool_grid[0]), int(pool_grid[1])),
            scalar_dim=int(cfg.get("scalar_dim", 12)),
            action_dim=int(cfg.get("action_dim", 16)),
            hidden_dim=int(cfg.get("hidden_dim", 128)),
            learning_rate=float(cfg.get("learning_rate", 3e-4)),
            device=device,
            rgb_key=str(cfg.get("rgb_key", "rgb")),
            scalar_key=str(cfg.get("scalar_key", "scalars")),
            seed=int(cfg.get("seed", 0)),
        )
        if checkpoint:
            payload = torch.load(checkpoint, map_location=device)
            state = payload.get("trainable_state_dict", payload)
            adapter.load_trainable_state_dict(state)
            adapter.policy_version = int(payload.get("policy_version", 0))
        return adapter

    def _obs_mapping(self, obs: Any) -> Mapping[str, Any]:
        if not isinstance(obs, Mapping):
            raise TypeError("tiny_rgb_discrete expects dict observations")
        if self.rgb_key in obs:
            return obs
        nested = obs.get("observation")
        if isinstance(nested, Mapping) and self.rgb_key in nested:
            return nested
        raise KeyError(f"observation has no RGB key {self.rgb_key!r}")

    def _rgb_array(self, obs: Any) -> np.ndarray:
        mapping = self._obs_mapping(obs)
        rgb = np.asarray(mapping[self.rgb_key])
        expected = (self.num_cameras, self.image_height, self.image_width, self.channels)
        if rgb.shape != expected:
            raise ValueError(f"expected RGB shape {expected}, got {rgb.shape}")
        return rgb

    def _scalar_vector(self, obs: Any) -> np.ndarray:
        mapping = self._obs_mapping(obs)
        if self.scalar_dim <= 0:
            return np.zeros((0,), dtype=np.float32)
        if self.scalar_key not in mapping:
            raise KeyError(f"observation has no scalar key {self.scalar_key!r}")
        scalars = np.asarray(mapping[self.scalar_key], dtype=np.float32).reshape(-1)
        if scalars.shape[0] != self.scalar_dim:
            raise ValueError(f"expected scalar_dim={self.scalar_dim}, got {scalars.shape[0]}")
        return scalars

    def _pool_rgb(self, rgb: np.ndarray) -> np.ndarray:
        grid_h, grid_w = self.pool_grid
        if self.image_height % grid_h == 0 and self.image_width % grid_w == 0:
            cell_h = self.image_height // grid_h
            cell_w = self.image_width // grid_w
            pooled = rgb.reshape(
                self.num_cameras,
                grid_h,
                cell_h,
                grid_w,
                cell_w,
                self.channels,
            ).mean(axis=(2, 4), dtype=np.float32)
        else:
            rows = np.linspace(0, self.image_height, grid_h + 1, dtype=np.int64)
            cols = np.linspace(0, self.image_width, grid_w + 1, dtype=np.int64)
            pooled = np.empty((self.num_cameras, grid_h, grid_w, self.channels), dtype=np.float32)
            for row in range(grid_h):
                for col in range(grid_w):
                    patch = rgb[:, rows[row] : rows[row + 1], cols[col] : cols[col + 1], :]
                    pooled[:, row, col, :] = patch.mean(axis=(1, 2), dtype=np.float32)
        return (pooled.reshape(-1) / 255.0).astype(np.float32, copy=False)

    def _state_tensor(self, policy_input_state: Any) -> torch.Tensor:
        if isinstance(policy_input_state, Mapping):
            policy_input_state = policy_input_state.get("latent")
        tensor = torch.as_tensor(policy_input_state, dtype=torch.float32, device=self.device)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.shape[-1] != self.encoder_dim:
            raise ValueError(f"expected latent dim={self.encoder_dim}, got {tensor.shape[-1]}")
        return tensor

    def _distribution_and_value(self, policy_input_state: Any):
        state = self._state_tensor(policy_input_state)
        hidden = self.model(state)
        logits = self.actor(hidden)
        value = self.critic(hidden).squeeze(-1)
        return torch.distributions.Categorical(logits=logits), value

    def _encode(self, obs: Any) -> Dict[str, np.ndarray]:
        rgb_latent = self._pool_rgb(self._rgb_array(obs))
        scalars = self._scalar_vector(obs)
        latent = np.concatenate([rgb_latent, scalars], axis=0).astype(np.float32, copy=False)
        return {"latent": latent}

    @torch.no_grad()
    def collect_step(self, obs: Any) -> PolicyStep:
        self.model.eval()
        self.actor.eval()
        self.critic.eval()
        policy_input_state = self._encode(obs)
        dist, value = self._distribution_and_value(policy_input_state)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        action_int = int(action.squeeze(0).detach().cpu().item())
        return PolicyStep(
            policy_input_state=policy_input_state,
            train_action=action_int,
            env_action=action_int,
            value=float(value.squeeze(0).detach().cpu().item()),
            old_action_log_prob=float(log_prob.squeeze(0).detach().cpu().item()),
        )

    @torch.no_grad()
    def value_from_obs(self, obs: Any) -> float:
        _, value = self._distribution_and_value(self._encode(obs))
        return float(value.squeeze(0).detach().cpu().item())

    def learner_spec(self) -> LearnerSpec:
        return self._learner_spec

    def trainable_state_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model.state_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
        }

    def load_trainable_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.model.load_state_dict(state_dict["model"])
        self.actor.load_state_dict(state_dict["actor"])
        self.critic.load_state_dict(state_dict["critic"])

    def trainable_components(self) -> List[str]:
        return ["model", "actor", "critic"]

    def set_train(self, mode: bool = True) -> None:
        self.model.train(mode)
        self.actor.train(mode)
        self.critic.train(mode)


class _TinyRGBLearnerModule(torch.nn.Module):
    def __init__(self, model, actor, critic, device: torch.device):
        super().__init__()
        self.model = model
        self.actor = actor
        self.critic = critic
        self.device = torch.device(device)

    def forward(self, batch: Dict[str, Any]) -> LearnerOutput:
        state = batch["policy_input_state"]
        if isinstance(state, Mapping):
            state = state.get("latent")
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        if state.ndim == 1:
            state = state.unsqueeze(0)
        hidden = self.model(state)
        dist = torch.distributions.Categorical(logits=self.actor(hidden))
        actions = torch.as_tensor(batch["actions"], dtype=torch.long, device=self.device).flatten()
        return {
            "log_probs": dist.log_prob(actions),
            "values": self.critic(hidden).squeeze(-1),
            "entropy": dist.entropy(),
        }


def _require_policy_adapter_class(candidate: Any, *, source: str) -> Type[PolicyAdapter]:
    if not inspect.isclass(candidate):
        raise TypeError(f"Policy adapter {source} must resolve to a class, got {type(candidate).__name__}")
    if not issubclass(candidate, PolicyAdapter):
        raise TypeError(f"Policy adapter {source} must subclass PolicyAdapter")
    if inspect.isabstract(candidate):
        raise TypeError(f"Policy adapter {source} must be a concrete PolicyAdapter subclass")
    return candidate


def resolve_policy_adapter(adapter_config: Mapping[str, Any]) -> Type[PolicyAdapter]:
    """Resolve an adapter class from config.

    Built-ins:
        mock
        tiny_rgb_discrete
        minddrive
        drivepi0

    Exactly one of built-in ``type`` or custom dotted ``class_path`` is
    required.
    """

    cfg = dict(adapter_config or {})
    raw_type = cfg.get("type")
    raw_class_path = cfg.get("class_path")
    adapter_type = str(raw_type).strip().lower() if raw_type is not None else ""
    class_path = str(raw_class_path).strip() if raw_class_path is not None else ""
    if bool(adapter_type) == bool(class_path):
        raise ValueError("policy_adapter must define exactly one of non-empty 'type' or 'class_path'")

    if adapter_type == "mock":
        return MockPolicyAdapter
    if adapter_type == "tiny_rgb_discrete":
        return TinyRGBDiscretePolicyAdapter
    if adapter_type == "minddrive":
        from b2d_rlinfra.finetuning.minddrive_policy_adapter import MindDrivePolicyAdapter

        return MindDrivePolicyAdapter
    if adapter_type == "drivepi0":
        from b2d_rlinfra.finetuning.drivepi0_policy_adapter import DrivePi0PolicyAdapter

        return DrivePi0PolicyAdapter

    if adapter_type:
        raise ValueError(f"Unknown policy_adapter.type={adapter_type!r}")

    module_name, _, class_name = class_path.rpartition(".")
    if not module_name or not class_name:
        raise ValueError(f"Invalid policy adapter class_path: {class_path!r}")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ImportError(f"Could not import policy adapter module {module_name!r}") from exc
    try:
        candidate = getattr(module, class_name)
    except AttributeError as exc:
        raise ImportError(
            f"Policy adapter class {class_name!r} was not found in module {module_name!r}"
        ) from exc
    return _require_policy_adapter_class(candidate, source=f"class_path={class_path!r}")
