# API Map

This page maps high-level concepts to source files. It is not a full generated API reference yet, but it should help new contributors find the right implementation quickly.

## Framework orientation files

`b2d_rlinfra/framework/` provides code-oriented orientation files and small examples for the main abstraction boundaries.

| File | Purpose |
| --- | --- |
| `b2d_rlinfra/framework/layer_walkthrough.py` | Code-oriented walkthrough for the five abstraction layers |
| `b2d_rlinfra/framework/env_rollout_demo.py` | Runnable async CARLA env pool rollout example |
| `b2d_rlinfra/framework/scenario_layer.py` | Scenario-layer imports and route-indexing helpers |
| `b2d_rlinfra/framework/environment_layer.py` | Environment factory and space builders |
| `b2d_rlinfra/framework/simulation_layer.py` | CARLA pool exports |
| `b2d_rlinfra/framework/algorithm_layer.py` | Algorithm, adapter, config, and callback exports |
| `b2d_rlinfra/framework/evaluation_layer.py` | Evaluation runtime and statistics exports |

## Environment

| Symbol | File |
| --- | --- |
| `CARLAEnv` | `b2d_rlinfra/environment/carla_env.py` |
| `CARLAEnvPool` | `b2d_rlinfra/simulation/runners/carla_env_pool.py` |
| `CARLAServerManager` | `b2d_rlinfra/simulation/runners/carla_server_manager.py` |
| `ShmRgbBuffer` | `b2d_rlinfra/simulation/runners/shm_rgb_buffer.py` |
| `build_observation_space` | `b2d_rlinfra/environment/spaces.py` |
| `build_action_space` | `b2d_rlinfra/environment/spaces.py` |
| `BirdViewObsManager` | `b2d_rlinfra/environment/handlers/birdview_obs_handler.py` |
| `RGBSensorObsHandler` | `b2d_rlinfra/environment/handlers/rgb_sensor_obs_handler.py` |
| `ScalarObsHandler` | `b2d_rlinfra/environment/handlers/scalars_obs_handler.py` |
| `GnssSensorObsHandler`, `ImuSensorObsHandler`, `SpeedometerSensorObsHandler` | `b2d_rlinfra/environment/handlers/ego_sensor_obs_handler.py` |
| `MindDriveStateObsHandler` | `b2d_rlinfra/environment/handlers/minddrive_state_obs_handler.py` |
| `DrivePi0StateObsHandler` | `b2d_rlinfra/environment/handlers/drivepi0_state_obs_handler.py` |
| `TrajectoryController` | `b2d_rlinfra/environment/controllers/trajectory_controller.py` |
| `RewardHandler` | `b2d_rlinfra/environment/handlers/reward_handler_ppo.py`, `b2d_rlinfra/environment/handlers/reward_handler_a2c.py`, `b2d_rlinfra/environment/handlers/reward_handler_sac.py`, `b2d_rlinfra/environment/handlers/reward_handler_td3.py`, `b2d_rlinfra/environment/handlers/reward_handler_minddrive.py` |
| `TerminationHandler` | `b2d_rlinfra/environment/handlers/termination_handler_ppo.py`, `b2d_rlinfra/environment/handlers/termination_handler_a2c.py`, `b2d_rlinfra/environment/handlers/termination_handler_sac.py`, `b2d_rlinfra/environment/handlers/termination_handler_td3.py` |

## Model integrations

| Symbol | File |
| --- | --- |
| MindDrive route/geometry helpers | `b2d_rlinfra/environment/model_integrations/minddrive_route.py` |
| DrivePi0 route/geometry helpers | `b2d_rlinfra/environment/model_integrations/drivepi0_route.py` |
| Route geometry utilities | `b2d_rlinfra/environment/model_integrations/route_geo.py` |
| Sensor context (publish/get) | `b2d_rlinfra/environment/handlers/sensor_context.py` |

## Algorithms

| Symbol | File |
| --- | --- |
| `BaseAlgorithm` | `b2d_rlinfra/learning/algorithms/base_algorithm.py` |
| `OnPolicyAlgorithm` | `b2d_rlinfra/learning/algorithms/on_policy_algorithm.py` |
| `OffPolicyAlgorithm` | `b2d_rlinfra/learning/algorithms/off_policy_algorithm.py` |
| `PPO` | `b2d_rlinfra/learning/algorithms/ppo.py` |
| `A2C` | `b2d_rlinfra/learning/algorithms/a2c.py` |
| `SAC` | `b2d_rlinfra/learning/algorithms/sac.py` |
| `TD3` | `b2d_rlinfra/learning/algorithms/td3.py` |
| `StandardEnvAdapter` | `b2d_rlinfra/learning/adapters/standard_adapter.py` |

## Policies and buffers

| Component | File |
| --- | --- |
| Actor-critic policy | `b2d_rlinfra/learning/policies/actor_critic_policy_v2.py` |
| SAC policy | `b2d_rlinfra/learning/policies/sac_policy.py` |
| TD3 policy | `b2d_rlinfra/learning/policies/td3_policy.py` |
| BEV feature extractors | `b2d_rlinfra/learning/policies/feature_extractor.py`, `b2d_rlinfra/learning/policies/feature_extractor_v2.py` |
| RGB camera feature extractor | `b2d_rlinfra/learning/policies/feature_extractor_v3_rgb.py` |
| Rollout buffer | `b2d_rlinfra/learning/buffers/per_worker_buffer.py` |
| Shared prioritized replay buffer | `b2d_rlinfra/learning/buffers/shared_buffer.py` |
| Static scenario replay buffer | `b2d_rlinfra/learning/buffers/static_buffer.py` |
| Mixed dynamic/static replay buffer | `b2d_rlinfra/learning/buffers/mixed_buffer.py` |

## End-to-End RL Finetune

`b2d_rlinfra/finetuning/` contains the VLA and end-to-end policy finetuning workflow, including parallel collectors, single- or multi-GPU learners, mmap-backed rollout storage, policy adapters, and checkpoint/export utilities.

| Component | File |
| --- | --- |
| Finetune entry point | `b2d_rlinfra/finetuning/train.py` |
| Shell launcher | `tools/launch/finetune/train.sh` |
| Distributed node entry point | `b2d_rlinfra/finetuning/node_agent.py` |
| Distributed topology and config normalization | `b2d_rlinfra/finetuning/topology.py`, `b2d_rlinfra/finetuning/config_schema.py` |
| File-backed coordination and GPU mapping | `b2d_rlinfra/finetuning/coordination.py`, `b2d_rlinfra/finetuning/gpu_mapping.py` |
| DDP learner runtime | `b2d_rlinfra/finetuning/learner_distributed.py` |
| Slurm launch and run-scoped cleanup | `tools/slurm/` |
| Finetune coordinator | `b2d_rlinfra/finetuning/coordinator.py` |
| Parallel collector process | `b2d_rlinfra/finetuning/collector.py` |
| CARLA actor process | `b2d_rlinfra/finetuning/sim_actor.py` |
| Policy adapter contract | `b2d_rlinfra/finetuning/policy_adapter.py` |
| MindDrive adapter | `b2d_rlinfra/finetuning/minddrive_policy_adapter.py` |
| DrivePi0 adapter | `b2d_rlinfra/finetuning/drivepi0_policy_adapter.py` |
| RolloutPack format | `b2d_rlinfra/finetuning/rollout_pack.py` |
| Rollout file storage and sample plans | `b2d_rlinfra/finetuning/rollout_file_store.py` |
| Rollout dataset reader | `b2d_rlinfra/finetuning/rollout_file_dataset.py` |
| Latest-weight store | `b2d_rlinfra/finetuning/weight_store.py` |
| MindDrive checkpoint exporter | `b2d_rlinfra/finetuning/tools/export_minddrive_checkpoint.py` |
| DrivePi0 checkpoint exporter | `b2d_rlinfra/finetuning/tools/export_drivepi0_checkpoint.py` |
| CARLA slot manager | `b2d_rlinfra/finetuning/carla_slot.py` |
| MindDrive observation adapter | `b2d_rlinfra/finetuning/minddrive_obs_adapter.py` |
| DrivePi0 observation adapter | `b2d_rlinfra/finetuning/drivepi0_obs_adapter.py` |
| DrivePi0 core inference | `b2d_rlinfra/finetuning/drivepi0_core.py` |
| Inter-process messages | `b2d_rlinfra/finetuning/messages.py` |
| TensorBoard logger | `b2d_rlinfra/finetuning/tensorboard_logger.py` |

## Evaluation

| Component | File |
| --- | --- |
| Leaderboard agent | `b2d_rlinfra/evaluation/leaderboard/agent.py` |
| Parallel evaluator | `b2d_rlinfra/evaluation/leaderboard/run_leaderboard_eval_parallel.py` |
| Model loader | `b2d_rlinfra/evaluation/leaderboard/model_loader.py` |
| Runtime config | `b2d_rlinfra/evaluation/leaderboard/runtime_config.py` |
| Score conversion | `b2d_rlinfra/evaluation/leaderboard/score_versions.py` |
| Bench2Drive group summary | `b2d_rlinfra/evaluation/leaderboard/summarize_b2d_groups.py` |
| Space builder | `b2d_rlinfra/evaluation/leaderboard/space_builder.py` |
| Evaluation runtime | `b2d_rlinfra/evaluation/runtime/runtime.py` |
| Route tracker | `b2d_rlinfra/evaluation/runtime/route_tracker.py` |
