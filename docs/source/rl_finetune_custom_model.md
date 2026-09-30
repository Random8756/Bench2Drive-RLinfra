# Custom Model Integration

This page describes how to connect your own model to the finetuning workflow in `b2d_rlinfra.finetuning`. The collector/learner loop, rollout files, and weight synchronization stay unchanged; a custom integration is a `PolicyAdapter` subclass that the workflow loads from your config. Read [Finetune Workflow](rl_finetune.md) first for the shared workflow, and treat the MindDrive and DrivePi0 adapters as reference implementations.

## Where the adapter sits

During collection, each collector calls the adapter to turn the current observation into an action for its CARLA worker and buffers what the adapter returns. Completed episodes are written to mmap-friendly `.rollout` RolloutPack files. Per-step data must stack into fixed-shape bool, integer, or floating-point arrays; represent variable-length data with padding and masks, or values and offsets. During updates, the learner reads rank-local batches from those files and evaluates the stored actions under the current weights. The adapter therefore owns both ends of the loop — model inference during collection, and action likelihood/value evaluation during PPO updates — while the collector, storage, and update machinery stay model-agnostic.

## The adapter contract

Subclass `PolicyAdapter` from `b2d_rlinfra/finetuning/policy_adapter.py` and implement:

| Method | Purpose |
| --- | --- |
| `load_initial(config, checkpoint, device)` | Build the model and optimizer from `policy_adapter.config` and the base checkpoint. |
| `collect_step(obs)` | One collection step: map the current observation to a `PolicyStep`. |
| `value_from_obs(obs)` | Scalar value estimate, used to bootstrap truncated episodes. |
| `learner_spec()` | Return the explicit training `nn.Module`, optimizer, precision, auxiliary-loss weights, and model-specific DDP options. |
| `trainable_state_dict()` / `load_trainable_state_dict(...)` | Save and restore the trainable-only weights that are checkpointed and synced to collectors. |
| `trainable_components()` | Return stable, logical names for the trainable components described by the published payload. |

The trainable parameters in `learner_spec().module` and `learner_spec().optimizer` must be exactly the same objects. The workflow validates this contract at startup and derives gradient clipping and dtype metadata from the module. Optional hooks such as `on_episode_start` and `set_train` follow the base-class docstrings.

### PolicyStep essentials

`collect_step` returns a `PolicyStep` that separates two actions:

- `env_action` is what the environment executes, for example low-level control produced by a downstream controller.
- `train_action` is what PPO optimizes the likelihood of, for example the model-level action that `old_action_log_prob` refers to.

They may be the same value for simple policies. `policy_input_state` is whatever the adapter needs to re-evaluate the step during updates; it may be an array or a nested mapping of arrays, but its structure, shapes, and dtypes must remain consistent across rollouts that may share an update. Additional per-step values can be attached through `action_logprob_info` with the same consistency requirement; they come back in the update batch.

### The update batch

The learner module returned by `learner_spec()` receives batches assembled from rollout files: the stored `policy_input_state`, `actions` (the train actions), `old_action_log_probs`, `advantages`, `returns`, and any per-step extras, as tensors on the learner device. Its `forward(batch)` must return the mapping-shaped `LearnerOutput`: required tensor fields `log_probs` and `values`, plus optional `entropy`, `aux_losses`, and `aux_logs`. `LearnerOutput` is a `TypedDict` that documents this mapping schema; at runtime it is an ordinary dictionary and does not wrap, copy, or transform tensors. The updater validates the mapping before computing PPO losses, rejecting tuple or dataclass outputs and unknown top-level fields. It calls the module directly for a single learner or through DDP for a multi-GPU learner. In DDP mode, auxiliary output keys must remain consistent across ranks and batches.

## Config

Choose exactly one adapter selector: use type for a built-in adapter, or omit type and set class_path to the dotted Python path of a custom PolicyAdapter subclass.

```yaml
policy_adapter:
  class_path: my_models.b2d_adapter.MyPolicyAdapter
  checkpoint: /abs/path/to/base_checkpoint.pt
  config:
    # Fields owned by your adapter; passed to load_initial().
```

Collectors run as spawned processes, so the module must be importable under the `PYTHONPATH` the launcher sets. For a local run, add the package path in `train.sh`. For a distributed run, make the code visible at the same path on every node and set `EXTRA_PYTHONPATH_VALUE` in both Slurm submission scripts. Everything else (`algorithm`, `rl_finetune`, `env`) works as in the built-in recipes.

A skeleton:

```python
import torch

from b2d_rlinfra.finetuning.policy_adapter import (
    DDPOptions,
    LearnerOutput,
    LearnerSpec,
    PolicyAdapter,
    PolicyStep,
)


class MyLearnerModule(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, batch) -> LearnerOutput:
        # Run the actual differentiable PPO graph here.
        return {
            "log_probs": ...,
            "values": ...,
            "entropy": ...,
            "aux_losses": {},
            "aux_logs": {},
        }


class MyPolicyAdapter(PolicyAdapter):
    @classmethod
    def load_initial(cls, config, checkpoint, device):
        adapter = cls(...)  # build/load the model and select trainable params
        adapter.learner_module = MyLearnerModule(adapter.model)
        optimizer = torch.optim.AdamW(adapter.learner_module.parameters(), lr=config["learning_rate"])
        adapter._learner_spec = LearnerSpec(
            module=adapter.learner_module,
            optimizer=optimizer,
            precision=config.get("precision"),
            ddp_options=DDPOptions(find_unused_parameters=False),
            aux_loss_weights={},
        )
        adapter._learner_spec.validate()
        return adapter

    @torch.no_grad()
    def collect_step(self, obs) -> PolicyStep:
        ...  # model inference -> PolicyStep(...)

    @torch.no_grad()
    def value_from_obs(self, obs) -> float:
        ...

    def learner_spec(self) -> LearnerSpec:
        return self._learner_spec

    def trainable_state_dict(self):
        return {"trainable_component": self.model.trainable_component.state_dict()}

    def load_trainable_state_dict(self, state_dict):
        self.model.trainable_component.load_state_dict(state_dict["trainable_component"])

    def trainable_components(self):
        return ["trainable_component"]
```

Create `self.learner_module`, the optimizer, and one cached `self._learner_spec` in `load_initial()` or the constructor; do not rebuild them on every `learner_spec()` call. Put static auxiliary-loss weights in `LearnerSpec.aux_loss_weights`. The adapter remains the checkpoint and weight-publication boundary, so `trainable_state_dict()` keys should not contain a DDP `module.` prefix.

## Environment side

The adapter consumes whatever the `env` section produces, so match the observation and action interfaces to your model:

- Observation branches (`rgb`, `scalars`, ego sensors, model-state branches) are configured under `env.observation_space`; see [Environment Layer](core_environment.md) and the recipe pages for concrete layouts.
- `env.action_space` supports discrete, continuous, and trajectory types; trajectory actions are converted to control by a trajectory controller (see the [DrivePi0 recipe](rl_finetune_drivepi0.md)).
- Reward and termination handlers are selected under `env.reward` and `env.termination`.
- Route-context wrappers under `env.model_integrations` accept a `class_path` for custom wrappers with the same `(env, config)` constructor as the built-in `minddrive_route` / `drivepi0_route` entries.

If your model needs a dedicated state observation that no existing branch provides, add an observation handler and space branch on the environment side first; the MindDrive and DrivePi0 state handlers show the pattern.

## Debugging a new adapter

Start small and iterate:

- Use one collector, one CARLA worker, a small `env.environment.max_episode_steps`, and `rollouts_per_update: 1` with `max_updates: 1`.
- `rl_finetune.execution_mode: inline` runs collection inside the coordinator process, which keeps stack traces in one place.
- The built-in `mock` and `tiny_rgb_discrete` adapter types are useful for verifying the infrastructure itself, independent of your model.
- Common early failures are schema-related: `policy_input_state` or `action_logprob_info` entries with inconsistent keys, shapes, or dtypes cannot be combined in one update.

Once one update completes end to end (a rollout `.rollout` file is written, the PPO update is logged, and `weights/policy_latest.pt` is republished), set `num_collectors`, `rollouts_per_update`, and `max_updates` to the values intended for full training.
