# Launch Training

Bench2Drive-RLInfra provides two training paths that share the same CARLA environments, wrapper stack, and YAML configuration format. Choose the one that matches your use case:

- **RL Baseline Training** starts from `tools/launch/baseline/train.sh`, uses BEV masks, RGB camera observations, scalar state, and other configured observation branches, and trains PPO / A2C / SAC / TD3 policies from scratch.
- **End-to-End RL Finetune** starts from `tools/launch/finetune/train.sh`, uses RGB camera observations plus model-specific and vehicle-state signals, and finetunes pretrained VLA models such as MindDrive or DrivePi0.

!!! note "Cleanup guidance"
    For local jobs, clear stale CARLA workers with `bash tools/runtime/kill_by_host.sh <host_ip>` or `bash tools/runtime/kill_all.sh`, and clean orphaned RGB shared memory with `bash tools/runtime/clean_shm.sh` (`-f` skips confirmation). For distributed jobs, use the [run-scoped cleanup helper](rl_finetune_distributed.md#stop-and-clean-up).


## RL Baseline Training

The baseline launcher resolves an algorithm alias or a YAML path, creates an output directory, and starts `b2d_rlinfra/learning/training/train.py`. The runner builds a `CARLAEnvPool`, wraps it with `StandardEnvAdapter`, constructs the policy and algorithm, and begins collecting transitions.

### BEV-observation baselines

BEV configs use rasterized bird's-eye-view masks plus scalar state and are the lightest way to get started. Four algorithm presets are included:

```bash
bash tools/launch/baseline/train.sh ppo    # configs/ppo_bev_example.yaml
bash tools/launch/baseline/train.sh a2c    # configs/a2c_bev_example.yaml
bash tools/launch/baseline/train.sh sac    # configs/sac_bev_example.yaml
bash tools/launch/baseline/train.sh td3    # configs/td3_bev_example.yaml
```

BEV masks are built from map data, route geometry, actor positions and other traffic states. They do not require CARLA camera rendering, so the example configs can use the repository YAML setting `null_rhi: true` for non-visual simulation.

### RGB camera observation baseline

An RGB camera PPO config attaches multi-camera sensors and uses the `combined_v3_rgb` feature extractor. This config is mainly a pipeline demonstration example for camera-based training, not the primary recommended baseline:

```bash
bash tools/launch/baseline/train.sh configs/ppo_rgb_example.yaml
```

RGB camera observations require CARLA rendering to remain available, so the config sets the repository YAML keys `no_rendering_mode: false` and `null_rhi: false`. Expect fewer workers per GPU compared to BEV-only training.

### Custom config

Pass any YAML path directly:

```bash
bash tools/launch/baseline/train.sh /path/to/custom_config.yaml
```

### Resume

Use the dedicated resume launchers:

```bash
bash tools/launch/baseline/resume_on_policy.sh ppo /path/to/checkpoint_dir
bash tools/launch/baseline/resume_off_policy.sh sac /path/to/checkpoint_dir
```

## End-to-End RL Finetune

The finetune launcher loads a recipe alias or YAML, starts parallel collectors that each own a CARLA worker and a local inference copy, and runs a learner that consumes RolloutPack files and publishes updated weights. The workflow is described in detail in [Finetune Workflow](rl_finetune.md).

### Available recipes

```bash
bash tools/launch/finetune/train.sh example     # configs/rl_finetune_rgb_example.yaml
bash tools/launch/finetune/train.sh minddrive    # configs/minddrive_rl_finetune.yaml
bash tools/launch/finetune/train.sh drivepi0     # configs/drivepi0_rl_finetune.yaml
bash tools/launch/finetune/train.sh /path/to/custom.yaml
```

`example` is a lightweight smoke-test config with a small RGB camera policy. `minddrive` and `drivepi0` are recipes for finetuning the corresponding pretrained VLA models. MindDrive requires the external [MindDrive](https://github.com/xiaomi-mlab/MindDrive) repository and the setup described in [MindDrive Finetune Recipe](rl_finetune_minddrive.md); DrivePi0 requires the external [DriveMoE](https://github.com/Thinklab-SJTU/DriveMoE) repository and the setup described in [DrivePi0 Finetune Recipe](rl_finetune_drivepi0.md).

### Distributed finetuning

To run finetuning across Slurm nodes, configure the `USER SETTINGS` in the submission script and use a distributed YAML:

```bash
bash tools/slurm/rl_finetune_sbatch.sh \
  configs/drivepi0_rl_finetune_distributed.yaml
```

Distributed runs require a shared POSIX-like filesystem and use node-local collector/CARLA slot lists. They do not use `rl_finetune.num_collectors` or `env.carla.num_envs`. See [Distributed RL Finetune](rl_finetune_distributed.md) before submitting the first job.

To continue from an RL checkpoint, submit a new run:

```bash
bash tools/slurm/rl_finetune_resume_sbatch.sh \
  <distributed_config.yaml> <rl_checkpoint.pt>
```

### Before launching

Edit the applicable launcher and YAML to match the machine or cluster. Key items to check:

- `CARLA_ROOT` points to CARLA 0.9.15.
- For a local finetune run, `env.carla.num_envs`, `rl_finetune.num_collectors`, `collector_devices`, and the CARLA slot lists match the intended topology.
- For a distributed run, the repository is on shared storage, `rl_finetune.distributed.num_nodes` matches the Slurm request, and node-local device and CARLA slot lists cover the largest per-node collector count.
- `policy_adapter.checkpoint` and model-specific paths are real paths on the machine.

## Practical tips

### Small-machine settings

The example configs use many CARLA workers. On a smaller workstation, reduce these together:

- `env.carla.num_envs` (baseline) or `rl_finetune.num_collectors` (finetune)
- `env.carla.host`, `port`, `traffic_manager_port`, `gpu_id`
- `training.adapter.min_ready` (baseline only)

A useful first smoke test is 2–4 workers, one short route file, and a reduced `algorithm.total_timesteps` or `rl_finetune.max_updates`.

### Pool rollout demo

The environment rollout demo drives the async pool with fixed actions — no policy or training involved:

```bash
bash b2d_rlinfra/framework/env_rollout_demo.sh
```

It can serve as a quick connectivity check before starting a real training job, but it is also a useful starting point for understanding how `CARLAEnvPool` works: the demo prints per-worker lifecycle events (reset, step, terminal, crash) and exposes the worker-indexed dictionary protocol, `min_ready` stepping, terminal observation ownership, and crash-discard logic that the algorithm code handles on top of the pool. If you plan to customize the pool interaction or build your own algorithm integration, the [Async Pool Tutorial](core_pool_tutorial.md) walks through each phase in detail.

### Route crashes

Some routes or scenarios may occasionally trigger CARLA crashes during long multi-worker jobs. This is expected because CARLA server itself is not perfectly stable. The training infrastructure restarts the affected worker/server, marks the crashed transition in `info["crashed"]`, and keeps the remaining workers running. If many workers crash at the same time, the machine usually cannot fit the current worker count; reduce `env.carla.num_envs` and the matching host/port/GPU lists.

### Output directories

Baseline and local finetune launchers write console logs under `output/<config_name>/`. Experiment logs and checkpoints go to the configured log directory:

```yaml
# baseline
training:
  log_dir: ./logs/ppo_bev_example
  save_freq: 100000

# finetune
rl_finetune:
  log_dir: ./logs/minddrive_rl_finetune
  checkpoint_interval: 1
  max_policy_lag: 2
  stale_rollout_action: drop
  cleanup_stale_rollouts: true
```

The distributed launcher writes per-node output, checkpoints, control state, and rollout artifacts under `logs/<config_stem>/rl_finetune_<timestamp>_<job_id>/` on shared storage.
