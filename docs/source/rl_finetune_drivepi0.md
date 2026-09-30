# DrivePi0 Finetune Recipe

This page walks through `configs/drivepi0_rl_finetune.yaml` — the recipe for finetuning a pretrained DrivePi0 trajectory policy with closed-loop PPO. It covers dependencies, configuration, launch, and checkpoint export. For the shared collector/learner workflow, rollout file layout, and adapter contract, read [Finetune Workflow](rl_finetune.md) first.

## Dependencies

Use the Bench2Drive-RLInfra Python 3.10 environment as the base. DrivePi0 is implemented in the external DriveMoE repository, whose main package pins `torch==2.5.0` and `torchvision==0.20.0`. Install the PyTorch wheel that matches your machine before installing DriveMoE in editable mode.

```bash
# Pick the CUDA wheel index for your machine.
python -m pip install torch==2.5.0 torchvision==0.20.0 \
  --index-url https://download.pytorch.org/whl/cu124

python -m pip install -r requirements.txt
python -m pip install -e /abs/path/to/DriveMoE
```

The DriveMoE repository and weights are expected to be available on disk before launch. The default recipe assumes this layout:

```text
workspace/
  Bench2Drive-RLinfra/
  DriveMoE/
    ckpts/
      DrivePi0_Base_bf16.pt
      paligemma-3b-pt-224/
```

## Configuration

Start from `configs/drivepi0_rl_finetune.yaml`. The checked-in file assumes an external `DriveMoE` repository next to this repository and a 16-collector / 8-GPU topology; both need editing before the first run.

### Model and adapter paths

Five paths must resolve to valid local files or directories:

```yaml
policy_adapter:
  type: drivepi0
  checkpoint: /abs/path/to/DriveMoE/ckpts/DrivePi0_Base_bf16.pt
  config:
    repo_dir: /abs/path/to/DriveMoE
    config_path: config/eval/DrivePi0/closed_loop.yaml
    pretrained_model_path: ckpts/paligemma-3b-pt-224
    statistics_path: config/statistics/b2d_statistics.json
```

`checkpoint` is the base DrivePi0 checkpoint that initializes the model before any RL updates. `repo_dir` is the root of the external DriveMoE repository — it gets added to `sys.path` (together with `repo_dir/src/agent`), and the runtime sets `DRIVEMOE_REPO_DIR` / `REPO_DIR` environment variables if they are not already defined. `config_path` is the DriveMoE YAML used to build the closed-loop DrivePi0 model. `pretrained_model_path` is the PaliGemma VLA backbone directory. `statistics_path` is the normalization statistics file used by DrivePi0 state preprocessing.

### Trainable scope and PPO settings

DrivePi0 uses **module-level finetuning with trajectory actions** (see [Finetune modes](rl_finetune.md#finetune-modes)): the adapter fully finetunes the action encoder/decoder (`action_encoder.*` and `action_decoder.*`) plus an adapter-initialized value head and `log_std`, while the PaliGemma VLA backbone remains frozen and is never exported as a trainable delta.

```yaml
policy_adapter:
  config:
    use_bf16: true
    num_inference_steps: 10
    text_prompt: predict trajectory
    image_preprocess: drivemoe
    image_jpeg_quality: 20
    image_augment: true
    action_init_mode: randn
    value_hidden_dim: 512
    log_std_init: -1.5
    ppo:
      mean_ref_l2_coef: 0.01
    state_shape: [5, 10]
```

`use_bf16` controls the model dtype after loading. `num_inference_steps` is the flow-matching action integration step count. `action_init_mode` controls the initial action state for the flow-matching integration process (`randn` or `zeros`).

`image_preprocess` selects the RGB preprocessing path — `drivemoe` (or its alias `official`) uses the DriveMoE pipeline, `rlinf` (or `default`) uses the RLInfra pipeline. `image_augment` enables random color/crop augmentation during training. `image_jpeg_quality` applies JPEG compression to camera frames before they enter the model, matching the quality level used during DrivePi0's original training.

`value_hidden_dim` and `log_std_init` configure the RL value head and action log-standard-deviation that the adapter creates on top of the frozen DrivePi0 backbone. `ppo.mean_ref_l2_coef` adds an L2 penalty between the current action mean and the reference action mean from rollout collection, discouraging the policy from drifting too far from its pre-finetune behavior.

### Single-node collector and CARLA topology

For the single-node recipe, these four counts must stay aligned: `rl_finetune.num_collectors`, the length of `rl_finetune.collector_devices`, `env.carla.num_envs`, and the lengths of the per-worker CARLA lists (`host`, `port`, `traffic_manager_port`, `traffic_manager_seed`, `gpu_id`). `collector_devices` controls where each collector runs model inference; `gpu_id` is passed to CARLA as the graphics adapter index.

The checked-in recipe runs 16 collectors across 8 GPUs. For a quick smoke test on a single GPU, shrink everything together:

```yaml
rl_finetune:
  device: cuda:0
  num_collectors: 2
  collector_devices: [cuda:0, cuda:0]
  max_updates: 2
  rollouts_per_update: 2
  collect_timeout: 3600
  control_poll_interval: 1.0
  shutdown_timeout: 120.0
  max_policy_lag: 2
  stale_rollout_action: drop
  cleanup_stale_rollouts: true

env:
  carla:
    num_envs: 2
    host: [127.0.0.135, 127.0.0.136]
    port: [2026, 2036]
    traffic_manager_port: [8202, 8302]
    traffic_manager_seed: [0, 0]
    gpu_id: [0, 0]
  environment:
    max_episode_steps: 256
```

Use host/port pairs that are free on your machine. If a previous launch left stale processes, clean them up before retrying — see the cleanup notes in [Launch Training](start_launch_training.md).

### Observation and rendering

DrivePi0 uses temporal front-camera RGB under `drivepi0_rgb` and normalized proprioceptive / route state under `drivepi0_state`. The raw `rgb` output is not emitted to the observation dict — only the temporal stack is:

```yaml
env:
  model_integrations:
    drivepi0_route:
      enable: true
      min_distance: 7.5
      max_distance: 25.0
      gps_sensor_id: GPS
  observation_space:
    reset_warmup_ticks: 11
    other_sensor_presets:
      - drivepi0
    rgb:
      enable: true
      output_key: rgb
      emit_output: false
      temporal_output_key: drivepi0_rgb
      temporal_camera_id: CAM_FRONT
      temporal_history_index: [-1, -3]
      frame_history_size: 8
      camera_order: [CAM_FRONT]
    drivepi0_state:
      enable: true
      output_key: drivepi0_state
      statistics_path: /abs/path/to/DriveMoE/config/statistics/b2d_statistics.json
      temporal_mode: drivemoe_20hz
      history_index: [-10, -8, -6, -4, -2]
      state_source: sensor
      route_source: drivepi0_route
      fallback_to_actor: false
      gps_sensor_id: GPS
      imu_sensor_id: IMU
      speed_sensor_id: SPEED
      route_lookahead_index: 25
```

`drivepi0_route` installs the route-context wrapper so the state handler can publish far-route target metadata. `reset_warmup_ticks: 11` gives the temporal history buffer enough frames to fill before the first real step.

If `temporal_output_key` or `drivepi0_state.output_key` is changed, also set matching keys in `policy_adapter.config.rgb_key` and `policy_adapter.config.state_key` — the adapter defaults are `drivepi0_rgb` and `drivepi0_state`.

Because DrivePi0 uses RGB cameras, CARLA rendering must stay available:

```yaml
env:
  carla:
    render_offscreen: true
    null_rhi: false
    no_rendering_mode: false
```

### Trajectory action

DrivePi0 predicts future trajectory waypoints, not instantaneous throttle/steer/brake. The `trajectory` shape in the adapter config and the environment action space must match:

```yaml
policy_adapter:
  config:
    trajectory:
      num_points: 10
      state_dim: 2

env:
  action_space:
    type: trajectory
    action_repeat: 2
    trajectory:
      num_points: 10
      state_dim: 2
      horizon: 1.0
      freq: 10.0
    controller:
      type: drivepi0
      interp_hz: 10.0
      points_per_meter: 10.0
      speed_lookahead_points: 3
      use_heading_interpolation: false
```

The trajectory controller converts the predicted waypoints into low-level vehicle controls at each tick. `action_repeat: 2` applies the same trajectory for two simulation ticks before requesting a new one.


## Launch

Edit `tools/launch/finetune/train.sh` for your machine, then run:

```bash
bash tools/launch/finetune/train.sh drivepi0
```

For multi-node finetuning, start from `configs/drivepi0_rl_finetune_distributed.yaml`, configure the Slurm submission script, and follow [Distributed RL Finetune](rl_finetune_distributed.md):

```bash
bash tools/slurm/rl_finetune_sbatch.sh \
  configs/drivepi0_rl_finetune_distributed.yaml
```

## Export and evaluation

DrivePi0 finetuning only saves trainable deltas. To produce a full checkpoint for DrivePi0 agent, merge the RL deltas back into the base checkpoint:

```bash
RL_CHECKPOINT=/path/to/rl_finetune_update_000100.pt \
BASE_CHECKPOINT=/path/to/DrivePi0_Base_bf16.pt \
OUTPUT_CHECKPOINT=/path/to/drivepi0_rl_export.pt \
bash b2d_rlinfra/finetuning/tools/export_drivepi0_checkpoint.sh
```

`RL_CHECKPOINT` can be either `weights/policy_latest.pt` or a periodic snapshot under `checkpoints/rl_finetune_update_*.pt`. `BASE_CHECKPOINT` should match the checkpoint used to initialize finetuning.

After export, evaluate through the official DriveMoE / Bench2Drive pipeline.
