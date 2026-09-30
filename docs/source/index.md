# Bench2Drive-RLInfra

Bench2Drive-RLInfra is a CARLA-based reinforcement learning infrastructure for training, comparing, and diagnosing autonomous-driving planners in closed-loop scenarios. It wraps CARLA Leaderboard and Bench2Drive-style route execution into a reproducible RL workflow with parallel rollout, configurable observations and rewards, integrated RL baselines, and benchmark-compatible evaluation.

## Getting started

New users should begin with:

- [Introduction](start_introduction.md), for the motivation and project scope.
- [Setup Guide](start_setup.md), for CARLA, Python, and repository setup.
- [Launch Training](start_launch_training.md), for running RL baselines or end-to-end finetune jobs.

## System components

The codebase is organized around five abstraction layers. The detailed layer-oriented view is described in [Architecture Overview](core_architecture.md).

| Layer | Responsibility | Architecture map | Key implementation paths |
| --- | --- | --- | --- |
| Scenario | Route files, task sampling, curriculum, scenario grouping | `b2d_rlinfra.framework.scenario_layer` | `b2d_rlinfra/scenario/adaptive_route_sampler.py`, `resources/routes/` |
| Environment | Gym-style CARLA interface, wrappers, observations, actions, rewards, termination | `b2d_rlinfra.framework.environment_layer` | `b2d_rlinfra/environment/carla_env.py`, `b2d_rlinfra/environment/wrappers.py`, `b2d_rlinfra/environment/handlers/`, `b2d_rlinfra/environment/spaces.py` |
| Simulation | Asynchronous CARLA worker pool, server management, crash recovery | `b2d_rlinfra.framework.simulation_layer` | `b2d_rlinfra/simulation/runners/carla_env_pool.py`, `b2d_rlinfra/simulation/runners/carla_server_manager.py` |
| Algorithm | PPO, A2C, SAC, TD3 training, buffers, policies, runners | `b2d_rlinfra.framework.algorithm_layer` | `b2d_rlinfra/learning/algorithms/`, `b2d_rlinfra/learning/training/`, `b2d_rlinfra/learning/policies/` |
| Evaluation | Leaderboard-compatible runtime, scoring, route diagnostics, videos | `b2d_rlinfra.framework.evaluation_layer` | `b2d_rlinfra/evaluation/leaderboard/`, `b2d_rlinfra/evaluation/runtime/` |

Built on the shared scenario, environment, and simulation infrastructure, the [Finetune Workflow](rl_finetune.md) (`b2d_rlinfra/finetuning`) post-trains pretrained driving models (MindDrive, DrivePi0, custom VLAs) with closed-loop PPO, combining any trainable scope (full / module-level / LoRA) with any action interface (meta action / trajectory / direct control); see [Finetune modes](rl_finetune.md#finetune-modes).

## Common workflows

| Goal | Entry point | Guide |
| --- | --- | --- |
| Train a BEV baseline | `tools/launch/baseline/train.sh` | [Launch Training](start_launch_training.md#rl-baseline-training) |
| Use RGB observations | `configs/ppo_rgb_example.yaml` | [Launch Training](start_launch_training.md#rgb-camera-observation-baseline) |
| Resume training | `tools/launch/baseline/resume_on_policy.sh`, `tools/launch/baseline/resume_off_policy.sh` | [Launch Training](start_launch_training.md#resume) |
| Finetune an E2E VLA model<br>(full-model / module-level / LoRA) | `tools/launch/finetune/train.sh` | [Finetune Workflow](rl_finetune.md) |
| Run E2E finetuning across Slurm nodes | `tools/slurm/rl_finetune_sbatch.sh` | [Distributed RL Finetune](rl_finetune_distributed.md) |
| Evaluate a checkpoint | `tools/launch/evaluation/run_leaderboard_eval_parallel.sh` | [Evaluation and Diagnostics](eval_diagnostics.md) |
| Customize an experiment | `configs/*.yaml`, or the [graphical config editor](../../tools/config_editor.html) | [Configuration Guide](configuration.md) |

## Repository layout

- `b2d_rlinfra/` - scenario, environment, simulation, learning, finetuning, evaluation, and framework modules.
- `configs/` - YAML recipes for baseline training, evaluation, and finetune.
- `resources/routes/`, `resources/maps/` - route definitions and BEV map assets.
- `tests/` - finetuning tests and lightweight layout/import contracts.
- `tools/` - launchers, Slurm helpers, runtime tools, and development configuration.
- `vendor/carla/` - independent training and evaluation Leaderboard / ScenarioRunner runtimes.
- `docs/` - MkDocs configuration and documentation sources.

For a deeper code map, see [Architecture Overview](core_architecture.md) and [API Map](api_map.md).

## Documentation status

The documentation covers setup, baseline training, evaluation, configuration, the five-layer architecture, and local or distributed end-to-end finetuning. API-level details will continue to expand as the implementation stabilizes.
