<h1 align="center">Bench2Drive-RLInfra: Scalable Infrastructure for Reinforcement Learning Planners in Autonomous Driving</h1>

<p align="center">
<img src="https://img.shields.io/badge/arXiv-coming-red?style=for-the-badge&logo=arxiv" alt="Paper coming">
<a href="https://huggingface.co/rethinklab/Bench2Drive-RLinfra-V0.0.1"><img src="https://img.shields.io/badge/HuggingFace-Model-orange?style=for-the-badge&logo=huggingface" alt="Model on Hugging Face"></a>
</p>

Bench2Drive-RLInfra is a CARLA-based infrastructure for closed-loop reinforcement learning in autonomous driving. It wraps CARLA Leaderboard and Bench2Drive-style route execution into a reproducible and diagnosable workflow for training RL planners, comparing baseline algorithms, and evaluating checkpoints. The same scenario, environment, and simulation infrastructure also supports end-to-end VLA finetuning through a parallel collector–learner workflow.

## Highlights and Overview

- Five abstraction layers: scenario, environment, simulation, algorithm, and evaluation.
- Gym-style CARLA interfaces with configurable observations (BEV, RGB, scalar), actions, rewards, and termination.
- Asynchronous multi-worker rollout pool with `min_ready` stepping for higher training throughput and automatic crash recovery.
- Built-in PPO, A2C, SAC, and TD3 training recipes.
- End-to-end RL finetuning for VLA models (MindDrive, DrivePi0) with parallel collectors, weight synchronization, and flexible full-model / module-level / LoRA training scopes paired with meta-action, trajectory, or direct-control interfaces.

### Modular RL training pipeline

![Bench2Drive-RLInfra training pipeline](docs/source/assets/training_pipeline.png)

Bench2Drive-RLInfra organizes baseline training into five connected layers. Route and scenario context flows through configurable Gym-style environments into an asynchronous CARLA simulation pool, which produces transitions for PPO, A2C, SAC, or TD3. Rollout events and learner metrics feed the diagnostics layer for BEV videos, episode statistics, reward overlays, and TensorBoard records. See the [Architecture Overview](docs/source/core_architecture.md) and [Launch Training](docs/source/start_launch_training.md) guides for details.

### End-to-end RL finetuning workflow

![Closed-loop end-to-end RL finetuning workflow](docs/source/assets/rl_finetune_loop.svg)

Built on the same scenario, environment, and simulation infrastructure as the RL baselines, the [RL finetuning workflow](docs/source/rl_finetune.md) turns the shared CARLA stack into a scalable post-training engine for pretrained VLA models and other end-to-end driving policies. `PolicyAdapter` isolates model-specific logic behind one unified interface, so external models can plug in with full-model, module-level, or LoRA optimization and expose meta-action, trajectory, or direct-control outputs while reusing the same collector, learner, and distributed runtime. Parallel collectors continuously produce file-backed RolloutPack episodes, single-GPU or multi-GPU DDP learners update the policy, and WeightStore publishes fresh trainable state back to collectors in either local or Slurm-managed deployments. See [Finetune Workflow](docs/source/rl_finetune.md) for the complete training lifecycle and [Custom Model Integration](docs/source/rl_finetune_custom_model.md) for the adapter contract.

## 🚀 Quick Start

1. **Set up the environment** — Install CARLA 0.9.15, create the Python environment, and prepare the required assets. See the [Setup Guide](docs/source/start_setup.md).

2. **Choose a recipe** — Start from an example YAML in `configs/`, then adjust CARLA paths, ports, GPUs, and training settings for your machine. See the [Configuration Guide](docs/source/configuration.md).

3. **Launch training** — Run a baseline: `bash tools/launch/baseline/train.sh <YAML>`

## Common Workflows

| Goal | Entry point | Guide |
| --- | --- | --- |
| Train a BEV baseline | `tools/launch/baseline/train.sh` | [Launch Training](docs/source/start_launch_training.md#rl-baseline-training) |
| Use RGB observations | `configs/ppo_rgb_example.yaml` | [Launch Training](docs/source/start_launch_training.md#rgb-camera-observation-baseline) |
| Resume training | `tools/launch/baseline/resume_on_policy.sh`, `tools/launch/baseline/resume_off_policy.sh` | [Launch Training](docs/source/start_launch_training.md#resume) |
| Finetune an E2E VLA model | `tools/launch/finetune/train.sh` | [Finetune Workflow](docs/source/rl_finetune.md) |
| Run distributed E2E finetuning | `tools/slurm/rl_finetune_sbatch.sh` | [Distributed RL Finetune](docs/source/rl_finetune_distributed.md) |
| Evaluate a checkpoint | `tools/launch/evaluation/run_leaderboard_eval_parallel.sh` | [Evaluation](docs/source/eval_diagnostics.md) |
| Customize an experiment | `configs/*.yaml`, or the [graphical config editor](tools/config_editor.html) | [Configuration Guide](docs/source/configuration.md) |

## 📖 Documentation

[**Browse the documentation**](docs/source/index.md) — covering setup, architecture, configuration, training, finetuning, evaluation, and troubleshooting.

To preview locally, run these commands from the repository root:

```bash
pip install -r tools/requirements/docs.txt
mkdocs serve -f docs/mkdocs.yml
```

## Acknowledgements

Our code builds on the following projects:

- [Leaderboard](https://github.com/carla-simulator/leaderboard)
- [ScenarioRunner](https://github.com/carla-simulator/scenario_runner)
- [Bench2Drive](https://github.com/Thinklab-SJTU/Bench2Drive)
- [CaRL](https://github.com/autonomousvision/CaRL)
- [Roach](https://github.com/zhejz/carla-roach)
- [CARLA Garage](https://github.com/autonomousvision/carla_garage)
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)
- [MindDrive](https://github.com/xiaomi-mlab/MindDrive)
- [DriveMoE](https://github.com/Thinklab-SJTU/DriveMoE)

Many thanks to their authors and contributors for their valuable contributions!

## License

All assets and code are under the [CC BY-NC-ND 4.0](LICENSE) license unless specified otherwise.

Third-party code and assets remain subject to their respective licenses and copyright notices, including those provided under `vendor/`.
