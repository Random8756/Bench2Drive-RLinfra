# Algorithm Layer

The algorithm layer connects asynchronous CARLA rollout to policy optimization. It converts worker-indexed outputs into learner-ready transitions, stores data in rollout or replay buffers, updates policy networks, and saves checkpoints.

## Components

| Component | Description |
|---|---|
| **On-policy** `L4_PPO`, `L4_A2C` | Usable reference implementations for on-policy training. |
| **Off-policy** `L4_SAC`, `L4_TD3` | Usable reference implementations for off-policy training. |
| `L4_EnvAdapter` | Preserves worker-indexed pool results, extracts the configured observation branch, and routes actions back to workers; batching/stacking happens inside the algorithm. |
| `L4_RolloutBuffer` | Per-worker on-policy rollout storage with GAE advantage computation. |
| `L4_ReplayBuffer` | Shared prioritized replay for off-policy algorithms, with an optional static scenario tier mixed into sampling. |
| `L4_ActorCriticPolicy` | MLP / CNN actor-critic with configurable feature extractors. |
| `L4_CNNFeatureExtractor` | Conv backbone for BEV or RGB camera observations. |
| `L4_load_config` | YAML → `Config` loader shared by the training and evaluation workflows. |

## Integrated baselines

The repository includes:

| Algorithm | Type | Main file |
| --- | --- | --- |
| PPO | On-policy actor-critic | `b2d_rlinfra/learning/algorithms/ppo.py` |
| A2C | On-policy actor-critic | `b2d_rlinfra/learning/algorithms/a2c.py` |
| SAC | Off-policy maximum-entropy actor-critic | `b2d_rlinfra/learning/algorithms/sac.py` |
| TD3 | Off-policy deterministic actor-critic | `b2d_rlinfra/learning/algorithms/td3.py` |

The same route suite, environment interface, observation builder, and evaluation runtime can be reused across algorithms.

## Standard adapter

`b2d_rlinfra/learning/adapters/standard_adapter.py` defines `StandardEnvAdapter`. It is a thin wrapper over `CARLAEnvPool` that passes `reset()`/`step()` through to the pool and extracts the configured observation branch (e.g. `vector`, `bev_mask`, or the full dict) from each worker's output. The worker-indexed dictionary protocol is preserved — the adapter does not batch or reshape transitions.

Dispatch, terminal observation handling, and crash-discard logic live in the algorithm code, not in the adapter. This matters because CARLA workers may be in different states:

- stepping,
- resetting,
- waiting for action,
- recovering after a crash,
- temporarily unavailable.

Learners should only store valid transitions in rollout or replay buffers. Crash markers carried through `info` let the algorithm discard corrupted episodes before they affect policy updates.

## Policies and feature extractors

Policy modules live under:

```text
b2d_rlinfra/learning/policies/
```

The feature extractor is selected by `policy.feature_extractor.type` and must match the observation modality.

**BEV feature extractor** (`combined_v2`) — used by the PPO/A2C BEV baseline configs. It uses configurable CNN channels for BEV masks and MLP branches for scalar state:

```yaml
policy:
  feature_extractor:
    type: combined_v2
    features_dim: 256
    use_layer_norm: true
  net_arch:
    pi: [256, 256]
    vf: [256, 256]
```

**RGB camera feature extractor** (`combined_v3_rgb`) — for RGB camera observation configs. Uses a shared CNN stem that processes each camera independently, pools the per-camera features, and fuses them with optional BEV, scalar, or other configured branches:

```yaml
policy:
  feature_extractor:
    type: combined_v3_rgb
    features_dim: 256
    use_layer_norm: true
    cnn_channels: [8, 16, 32, 64, 128, 256]
    rgb_camera_feature_dim: 128
    fusion_dims: [512]
```

SAC/TD3 BEV examples use the older `combined` extractor, which keeps a fixed CNN branch plus a smaller MLP branch and fusion layer for off-policy policies.

The code also includes distributional value utilities and optional mixture-of-experts modules for more advanced experiments.

## Buffers

On-policy algorithms use per-worker rollout buffers. Off-policy algorithms use a shared-memory prioritized dynamic replay buffer. When `algorithm.static_buffer.enabled` is true, the dynamic PER buffer is wrapped by `MixedReplayBuffer`, which mixes in per-scenario static buffers with an annealed static ratio; priority updates still apply to the dynamic PER samples.

Important files:

- `b2d_rlinfra/learning/buffers/per_worker_buffer.py`
- `b2d_rlinfra/learning/buffers/shared_buffer.py`
- `b2d_rlinfra/learning/buffers/static_buffer.py`
- `b2d_rlinfra/learning/buffers/mixed_buffer.py`
- `b2d_rlinfra/learning/algorithms/train_process.py`

Large BEV observations can be compressed in memory and decompressed at sampling time to reduce memory pressure and transfer overhead.

## Adding a new algorithm

A new algorithm should:

1. Consume the same adapter interface used by existing algorithms.
2. Build spaces through `b2d_rlinfra/environment/spaces.py`.
3. Respect worker IDs and discard crashed transitions.
4. Load its hyperparameters from `configs/*.yaml`.
5. Save checkpoints in the same layout expected by the leaderboard model loader if it needs evaluation.

Start by reading `b2d_rlinfra/learning/training/runner_utils.py`, because this file maps config choices to feature extractors, policies, and algorithm classes.

!!! note
    The end-to-end RL finetuning workflow (`b2d_rlinfra/finetuning/`) does not use the algorithm classes listed above. It brings its own learner loop, mmap-backed RolloutPack storage, and model-specific `PolicyAdapter` integrations. To add a new end-to-end model, implement a `PolicyAdapter` subclass and explicit learner contract; see [Custom Model Integration](rl_finetune_custom_model.md).
