# Configuration Guide

Experiments are driven by YAML files under `configs/`. The repository includes runnable example configs that can also serve as templates for new experiments. Use the example YAML files as references when editing algorithm settings, policy configuration, CARLA runtime options, route sampling, observations, actions, rewards, termination logic, and other experiment settings.

!!! note "Graphical config editor"
    [Open the editor](../../tools/config_editor.html) to create or edit a baseline training YAML visually: pick a preset or import an existing YAML, auto-fill CARLA workers, preview the BEV mask live, then export a runnable config.

## Example configs

| Config | Purpose |
| --- | --- |
| `ppo_bev_example.yaml`, `a2c_bev_example.yaml` | On-policy BEV baseline presets behind the `train.sh ppo` and `train.sh a2c` aliases |
| `sac_bev_example.yaml`, `td3_bev_example.yaml` | Off-policy BEV baseline presets behind the `train.sh sac` and `train.sh td3` aliases |
| `ppo_bev_simple_reward_example.yaml` | PPO BEV variant using the `simple` reward handler |
| `ppo_rgb_example.yaml` | RGB camera pipeline demonstration with `combined_v3_rgb` |
| `env_rollout_demo.yaml` | Async pool rollout demo, no training (see [Async Pool Tutorial](core_pool_tutorial.md)) |
| `rl_finetune_rgb_example.yaml` | Finetune workflow smoke test with a tiny RGB policy |
| `minddrive_rl_finetune.yaml`, `drivepi0_rl_finetune.yaml` | End-to-end VLA finetune recipes |
| `drivepi0_rl_finetune_distributed.yaml` | Multi-node Slurm example with a DDP learner |

## Top-level sections

| Section | Purpose |
| --- | --- |
| `algorithm` | RL algorithm name, learning rate, horizon, batch size, replay settings |
| `policy` | Feature extractor, policy/value heads, hidden dimensions |
| `training` | Seed, device, log directory, checkpoint frequency, adapter and visualization |
| `rl_finetune` | End-to-end learner strategy, collector topology, rollout, checkpoint, and distributed settings |
| `policy_adapter` | End-to-end model adapter, base checkpoint, and adapter-owned settings |
| `env.environment` | Episode limits, event termination, result directories |
| `env.carla` | CARLA workers, hosts, ports, GPU IDs, server launch options |
| `env.ipc` | Optional inter-process transport settings such as RGB shared-memory mode |
| `env.routes` | Route files, sampling mode, adaptive schedule |
| `env.action_space` | Discrete, continuous, or trajectory action definition |
| `env.observation_space` | BEV masks, RGB camera observations, scalar and vehicle-state branches, model-specific state, history indices |
| `env.reward` | Reward handler selection (`type`) |
| `env.termination` | Termination handler selection (`type`) |
| `env.model_integrations` | Optional model-specific route/context wrappers used by finetune recipes |

## Common edits

### Reduce workers

For local debugging, reduce the worker count and list lengths together:

```yaml
env:
  carla:
    num_envs: 2
    host: [127.0.0.1, 127.0.0.2]
    port: [2026, 2036]
    traffic_manager_port: [8202, 8302]
    gpu_id: [0, 0]
training:
  adapter:
    min_ready: 1
```

`training.adapter.min_ready` is how many workers must report a result before a pool step returns. See [Simulation Layer](core_simulation.md#readiness-based-stepping) for tuning guidance.

For multi-node finetuning, collector count comes from `rl_finetune.distributed` instead of `rl_finetune.num_collectors` or `env.carla.num_envs`. `rl_finetune.learner` selects a single learner or a DDP learner group. CARLA lists are node-local in that mode; see [Distributed RL Finetune](rl_finetune_distributed.md#collector-topology).

### Use a custom route file

```yaml
env:
  routes:
    route_files:
      - resources/routes/my_training_routes.xml
    sample_mode: random
```

`sample_mode` accepts `sequential` (default), `random`, or `adaptive`; see [Scenario Layer](core_scenarios.md#sampling-modes).

### Shorten a smoke test

```yaml
algorithm:
  total_timesteps: 10000
training:
  save_freq: 5000
env:
  environment:
    max_episode_steps: 256
```

### Change observation history

```yaml
env:
  observation_space:
    vector:
      history_index: [-16, -11, -6, -1]
    scalars:
      items:
        - type: speed
          use_history: true
```

### Enable RGB camera observations

Switch from BEV-only input to multi-camera input from CARLA RGB camera sensors. Scalar state, GNSS / IMU / speedometer signals, and model-specific branches can still be enabled when the policy expects them. RGB camera observations require compatible rendering settings (see below):

```yaml
env:
  observation_space:
    vector:
      enable: false
    rgb:
      enable: true
      output_key: rgb
      camera_order: [CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT, CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT]
      sensors:
        - { type: sensor.camera.rgb, x: 0.80, y: 0.0, z: 1.60, width: 1600, height: 900, fov: 70, id: CAM_FRONT }
        # ... additional cameras
  carla:
    no_rendering_mode: false
    null_rhi: false
```

When using RGB camera observations, also change `policy.feature_extractor.type` to `combined_v3_rgb`. See `configs/ppo_rgb_example.yaml` for a complete camera-pipeline demonstration example.

### Use trajectory actions

For end-to-end models that predict future waypoints instead of instantaneous controls:

```yaml
env:
  action_space:
    type: trajectory
    trajectory:
      num_points: 4
      state_dim: 3
```

### Select rewards and termination

Reward and termination handlers are selected in YAML by `type`:

```yaml
env:
  reward:
    type: ppo
  termination:
    type: ppo
```

Registered reward types are `simple`, `ppo`, `a2c`, `sac`, `td3`, and `minddrive_sparse` (a sparse variant used by the MindDrive finetune recipe). Registered termination types are `ppo`, `a2c`, `sac`, and `td3`.

The YAML reward block only selects a reward handler; the per-algorithm handlers are kept separate to make reward settings easier to edit and to make adding new handlers straightforward. Reward weights, event rewards, and shaping switches live in the selected handler file under `b2d_rlinfra/environment/handlers/reward_handler_*.py` — edit the `REWARD_CONFIG` dictionary at the top of the selected file for reward changes. New reward handlers can follow the existing handler structure and be added to the wrapper registry in `b2d_rlinfra/environment/wrappers.py`.

### Rendering settings

These `env.carla.*` entries are Bench2Drive-RLInfra YAML keys that map to CARLA launch arguments or runtime world settings.

| Setting | BEV-only training | RGB camera and E2E training |
| --- | --- | --- |
| `env.carla.render_offscreen` | `true` | `true` |
| `env.carla.null_rhi` | `true` | `false` |
| `env.carla.no_rendering_mode` | `true` | `false` |

`render_offscreen` keeps camera rendering available without a visible window. `null_rhi` removes render capability, and `no_rendering_mode` makes GPU / camera sensor data unavailable or empty, so both must be `false` for RGB camera training. The pool validates these settings at startup when RGB camera observations are enabled.

### Toggle visualization

```yaml
training:
  visualization:
    enabled: true
    save_interval: 10
    fps: 10
    lazy_capture: true
```

## Config loading

`b2d_rlinfra/learning/utils/config.py` parses YAML into typed dataclasses:

- `AlgorithmConfig`
- `PolicyConfig`
- `TrainingConfig`
- `AdapterConfig`
- `VisualizationConfig`
- `Config`

Environment-specific fields remain as a dictionary under `Config.env_config`, so the CARLA wrappers and runners can consume nested settings directly.

## End-to-End Finetune Configs

The finetuning workflow (`b2d_rlinfra/finetuning/`) uses the same `algorithm` and `env` sections as the baselines, and adds `rl_finetune` and `policy_adapter` sections for learner strategy, collector topology, weight synchronization, and model-specific adapter settings. See [Finetune Workflow](rl_finetune.md) for the shared config contract, [MindDrive Finetune Recipe](rl_finetune_minddrive.md) for MindDrive-specific YAML edits, and [DrivePi0 Finetune Recipe](rl_finetune_drivepi0.md) for DrivePi0-specific YAML edits.
