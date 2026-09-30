# Introduction

Bench2Drive-RLInfra provides infrastructure for reinforcement learning planners in autonomous driving. The project targets the engineering gap between CARLA's realistic closed-loop simulation and the standard workflow expected by RL algorithms.

CARLA Leaderboard and Bench2Drive are strong evaluation substrates, but they primarily expose route execution and scoring interfaces. RL training adds another layer of difficulty: simulator orchestration, asynchronous rollout, expensive resets, crash recovery, reward shaping, replay or rollout buffering, policy optimization, and benchmark-compatible evaluation must work together over long jobs.

Bench2Drive-RLInfra turns this into a modular workflow rather than an ad-hoc simulator integration.

## What this project provides

- A view of five abstraction layers for scenario specification, environment interaction, simulator execution, policy optimization, and evaluation.
- Gym-style CARLA environments with configurable observations (BEV masks, RGB camera observations, scalar state, GNSS / IMU / speedometer signals, model-specific state), action (discrete, continuous, trajectory), reward, and termination logic.
- An asynchronous multi-worker CARLA rollout pool with configurable `min_ready` stepping and shared-memory transport for RGB camera pipelines.
- Integrated PPO, A2C, SAC, and TD3 baselines under shared route and environment protocols.
- An end-to-end RL finetuning workflow with local or multi-node execution, pluggable policy adapters (MindDrive, DrivePi0), mmap-backed RolloutPack storage, optional DDP learning, and latest-weight synchronization.
- Leaderboard-style parallel evaluation with route-level, scenario-level, and infraction-level diagnostics.

## When to use it

Use Bench2Drive-RLInfra when you want to:

- Train closed-loop RL planners in CARLA with BEV, RGB camera, scalar, vehicle-state, or model-specific observations.
- Compare on-policy and off-policy RL algorithms under a shared driving protocol.
- Finetune end-to-end driving models (e.g. MindDrive, DrivePi0) through closed-loop RL collection and online updates.
- Study scenario sampling, curriculum learning, reward design, and replay design.
- Evaluate trained policies using CARLA Leaderboard or Bench2Drive-compatible outputs.
- Convert route-based driving tasks into reusable RL experiments.

## Design principles

The project follows three practical principles:

1. Configuration should drive experiments. Routes, workers, ports, observations, actions, rewards, and algorithm settings live in YAML files.
2. Simulator failures should be recoverable. A single crashed CARLA worker should not invalidate a long training run.
3. Evaluation should remain benchmark-compatible. RL policies should be evaluated through the same route execution and scoring concepts used by CARLA-style benchmarks.

## Two training paths

Bench2Drive-RLInfra supports two complementary training paths that share the same CARLA environments, wrapper stack, and configuration format:

1. **RL Baseline Training** — PPO, A2C, SAC, and TD3 algorithms train lightweight policies from BEV masks, RGB camera observations, scalar state, and other configured observation branches through the asynchronous `CARLAEnvPool` and `StandardEnvAdapter`. Launch with `bash tools/launch/baseline/train.sh <algorithm|config>`.
2. **End-to-End RL Finetune** — large pretrained models (MindDrive, DrivePi0, or custom VLA policies) are finetuned through parallel collectors that each own a CARLA worker and a local inference copy. Collectors write RolloutPack files; one learner or a DDP learner group consumes them for PPO updates and publishes fresh weights. Use `tools/launch/finetune/train.sh` locally or the [distributed Slurm launcher](rl_finetune_distributed.md) for multi-node finetuning.

Both paths reuse the same observation handlers, action wrappers, CARLA server management, and route/scenario infrastructure. The main difference is how rollout data flows to the learner: the baseline path keeps online worker-indexed pool outputs in process and lets the algorithm stack ready workers for updates, while the finetune path decouples collection and learning through mmap-backed files and weight synchronization.
