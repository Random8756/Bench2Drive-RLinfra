# MindDrive Finetune Recipe

This page walks through `configs/minddrive_rl_finetune.yaml` — the recipe for finetuning a pretrained MindDrive policy with closed-loop PPO. It covers dependencies, configuration, launch, and checkpoint export. For the shared collector/learner workflow, rollout file layout, and adapter contract, read [Finetune Workflow](rl_finetune.md) first.

## Dependencies

Use the Bench2Drive-RLInfra Python 3.10 environment as the base. MindDrive needs a set of extra packages; do **not** install MindDrive's own `requirements.txt` directly, because it pins Python 3.8-era versions of `numba`, `numpy`, and other libraries that conflict with the 3.10 environment.

```bash
python -m pip install torch==2.4.0 torchvision==0.19.0
python -m pip install -r requirements.txt

python -m pip install \
  cython addict Pillow prettytable terminaltables \
  nuscenes-devkit scikit-image scikit-learn cityscapesscripts imagecorruptions \
  open3d ipython seaborn einops casadi torchmetrics \
  pytest pytest-cov pytest-runner yapf==0.40.1 flake8 \
  similaritymeasures loguru gym pyzmq sentencepiece \
  motmetrics==1.1.3 \
  "trimesh>=3.0" \
  "laspy>=2.5" "lazrs>=0.5" \
  "numba>=0.57" \
  "transformers>=4.45,<4.50" \
  "peft>=0.12,<0.14" \
  "diffusers>=0.32,<0.34" \
  stable-baselines3 \
  lyft_dataset_sdk

python -m pip install flash-attn==2.8.3
cd /abs/path/to/MindDrive
python -m pip install -e . --no-build-isolation --no-deps
```

Keep `numpy` on the version already installed in the base environment. `flash-attn` may need to build from source — match the command to your CUDA, PyTorch, and GPU architecture.

## Configuration

Start from `configs/minddrive_rl_finetune.yaml`. The checked-in file is a machine-local template with placeholder model paths and a 16-collector / 8-GPU topology; both need editing before the first run.

### Model and adapter paths

Three paths must resolve to valid local files or directories:

```yaml
policy_adapter:
  type: minddrive
  checkpoint: /abs/path/to/minddrive_rltrain.pth
  config:
    minddrive_root: /abs/path/to/MindDrive
    minddrive_config: /abs/path/to/MindDrive/adzoo/minddrive/configs/minddrive_rl_ppo_train.py
```

`checkpoint` is the base MindDrive checkpoint that initializes the model before any RL updates. `minddrive_root` is the root of the external MindDrive repository — it gets added to `sys.path` together with MindDrive's `rl_projects` directories, and the adapter temporarily `chdir`s into it while loading the config so that relative paths inside MindDrive (e.g. `ckpts/llava-qwen2-0.5b`) resolve correctly. `minddrive_config` is the MindDrive config file used to build the model and inference pipeline.

### Trainable scope and PPO settings

MindDrive uses **LoRA finetuning with meta actions** (see [Finetune modes](rl_finetune.md#finetune-modes)). PPO optimizes the discrete meta-action distribution. Only the decision-expert LoRA weights and the value head are updated; the rest of the model stays frozen.

The `trainable` list selects parameter-name substrings within that trainable scope. The default selects the decision-expert LoRA weights plus `value_net_pro`:

```yaml
policy_adapter:
  config:
    trainable:
      - decision_expert
      - value_net_pro
    precision: bf16
    reset_value_head: false
    collection_action_mode: sample
    pid_controller: eval_de
    ppo:
      kl_coef: 0.5
      use_kl: true
```

`precision: bf16` wraps inference in `torch.autocast(dtype=bfloat16)` without converting the model itself; `fp16` instead applies MindDrive/mmcv `wrap_fp16_model`.

`collection_action_mode` controls whether rollout actions are stochastic (`sample`) or greedy (`argmax`); PPO normally uses `sample`. `pid_controller` selects the MindDrive PID implementation — `rollout_decouple` imports from `rl_projects.utils`, `eval_de` imports from `team_code`. `ppo.kl_coef` and `ppo.use_kl` add an old-policy KL penalty to the PPO objective.

If `reset_value_head` is `true` (or unset), the adapter reinitializes `value_net_pro` before training so the value head starts fresh for the new reward function.

### Single-node collector and CARLA topology

For the single-node recipe, these four counts must stay aligned: `rl_finetune.num_collectors`, the length of `rl_finetune.collector_devices`, `env.carla.num_envs`, and the lengths of the per-worker CARLA lists (`host`, `port`, `traffic_manager_port`, `traffic_manager_seed`, `gpu_id`). `collector_devices` controls where each collector runs model inference; `gpu_id` is passed to CARLA as the graphics adapter index. On multi-GPU machines they are often related, but they are configured independently.

The checked-in recipe runs 16 collectors across 8 GPUs. For a quick smoke test on a single GPU, shrink everything together:

```yaml
rl_finetune:
  device: cuda:0
  num_collectors: 2
  collector_devices: [cuda:0, cuda:0]
  max_updates: 2
  rollouts_per_update: 2
  collect_timeout: 18000
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

MindDrive expects six RGB cameras under the `rgb` observation key and model-specific route/state metadata under `minddrive_state`. The recipe also enables a route-context integration so the state handler can publish near-target and route-command metadata:

```yaml
env:
  model_integrations:
    minddrive_route:
      enable: true
      downsample_factor: 50.0
  observation_space:
    other_sensor_presets:
      - minddrive
    rgb:
      enable: true
      output_key: rgb
    minddrive_state:
      enable: true
      output_key: minddrive_state
```

If you rename `rgb.output_key` or `minddrive_state.output_key`, also set matching keys in `policy_adapter.config.rgb_key` and `policy_adapter.config.state_key` — the adapter defaults are `rgb` and `minddrive_state`.

Because MindDrive uses RGB cameras, CARLA rendering must stay available:

```yaml
env:
  carla:
    render_offscreen: true
    null_rhi: false
    no_rendering_mode: false
```

`render_offscreen` tells CARLA to keep the GPU rendering pipeline active without a visible window. Setting `null_rhi: true` removes render capability entirely, and `no_rendering_mode: true` disables sensor data output — either one will make camera frames empty or time out.


## Launch

Edit `tools/launch/finetune/train.sh` for your machine, then run:

```bash
bash tools/launch/finetune/train.sh minddrive
```


## Export and evaluation

MindDrive finetuning only saves trainable deltas. To produce a full checkpoint for the official MindDrive closed-loop evaluation, merge the RL deltas back into the base checkpoint:

```bash
RL_CHECKPOINT=/path/to/rl_finetune_update_000054.pt \
BASE_CHECKPOINT=/path/to/minddrive_rltrain.pth \
OUTPUT_CHECKPOINT=/path/to/minddrive_rl_export.pth \
bash b2d_rlinfra/finetuning/tools/export_minddrive_checkpoint.sh
```

`RL_CHECKPOINT` can be either `weights/policy_latest.pt` or a periodic snapshot under `checkpoints/rl_finetune_update_*.pt`. `BASE_CHECKPOINT` should match the checkpoint used to initialize finetuning.

After export, evaluate through the official MindDrive / Bench2Drive pipeline.
