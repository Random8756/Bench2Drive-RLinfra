"""Code-oriented walkthrough for the five-layer RL pipeline.

The executable launchers live under ``tools/launch/baseline`` and
``tools/launch/evaluation``. This file keeps the same ideas in one place,
arranged as a readable training and evaluation pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable


# Layer 1 - Scenario
from b2d_rlinfra.framework.scenario_layer import (
    L1_RouteIndexer,
    L1_normalize_adaptive_config,
)

# Layer 2 - Environment
from b2d_rlinfra.framework.environment_layer import (
    L2_build_action_space,
    L2_build_observation_space,
    L2_make_env,
)

# Layer 3 - Simulation
from b2d_rlinfra.framework.simulation_layer import L3_CARLAEnvPool

# Layer 4 - Algorithm
from b2d_rlinfra.framework.algorithm_layer import (
    L4_CallbackList,
    L4_CheckpointCallback,
    L4_EnvAdapter,
    L4_load_config,
)
from b2d_rlinfra.learning.training.runner_utils import (
    get_algorithm_components,
    get_feature_extractor,
)

# Layer 5 - Evaluation
from b2d_rlinfra.framework.evaluation_layer import (
    L5_LeaderboardEvalRuntime,
    L5_RLLeaderboardAgent,
    L5_RLStatisticsManager,
    L5_TrainingVisualizer,
)
from b2d_rlinfra.evaluation.leaderboard.model_loader import load_model_bundle
from b2d_rlinfra.evaluation.leaderboard.runtime_config import load_agent_config


CONFIG_PATH = "configs/ppo_bev_example.yaml"
FINAL_MODEL_NAME = "final_model"
EVAL_RESULT_DIR = "outputs/eval"


def training_pipeline(config_path: str = CONFIG_PATH) -> None:
    # L4 - Load YAML once; the same config drives every layer below.
    config = L4_load_config(config_path)
    algo = config.algorithm
    training = config.training
    env_config = config.env_config

    log_dir = Path(training.log_dir) / "walkthrough_run"
    checkpoint_dir = log_dir / "checkpoints"

    # L1 - Route indexing and curriculum settings sit below the environment.
    route_config = env_config.get("routes") or {}
    route_file = route_config.get("route_files") or "resources/routes/train_routes_demo.xml"
    if isinstance(route_file, (list, tuple)):
        route_file = route_file[0] if route_file else "resources/routes/train_routes_demo.xml"
    l1_route_indexer = L1_RouteIndexer(
        routes_file=str(route_file),
        repetitions=int(route_config.get("repetitions", 1)),
        routes_subset=route_config.get("routes_subset"),
    )
    l1_adaptive_config = L1_normalize_adaptive_config(route_config.get("adaptive"))
    print(
        f"[L1] routes={l1_route_indexer.get_length()} "
        f"adaptive={l1_adaptive_config.get('enabled', False)}"
    )

    # L2 - The policy-facing spaces are derived from the same YAML as CARLAEnv.
    l2_observation_space = L2_build_observation_space(env_config)
    l2_action_space = L2_build_action_space(env_config)

    # L4 - Feature extractor and algorithm kwargs are selected by config.
    _feature_extractor_cls, _feature_extractor_kwargs, obs_key, policy_kwargs = (
        get_feature_extractor(config)
    )
    algorithm_config = config.to_dict().get("algorithm", {})

    # L3 - CARLAEnvPool owns worker processes, ports, server lifecycle, and
    # crash recovery. Each worker uses L2_make_env to build CARLAEnv plus the
    # route, observation, termination, reward, and action wrappers.
    carla_config = env_config.get("carla") or {}
    environment_config = env_config.get("environment") or {}
    worker_config = dict(env_config)
    worker_config["algorithm"] = dict(algorithm_config)

    l3_pool = L3_CARLAEnvPool(
        env_fn=L2_make_env,
        config=worker_config,
        num_envs=int(carla_config.get("num_envs", 1)),
        auto_reset=True,
        manage_servers=True,
        max_episode_steps=int(environment_config.get("max_episode_steps", 10000)),
    )

    # L4 - StandardEnvAdapter is the learner's view of the async worker pool.
    l4_adapter = L4_EnvAdapter(
        pool=l3_pool,
        observation_space=l2_observation_space,
        action_space=l2_action_space,
        default_timeout=config.adapter.timeout,
        obs_key=obs_key,
    )

    # L5 - Training video is attached as an observer of collected transitions.
    l5_visualizer = None
    if config.visualization.enabled:
        l5_visualizer = L5_TrainingVisualizer(
            output_dir=str(log_dir / "videos"),
            fps=config.visualization.fps,
            save_interval_episodes=config.visualization.save_interval,
            max_episodes_to_keep=config.visualization.max_videos,
            lazy_capture=config.visualization.lazy_capture,
            overlay_info=config.visualization.overlay_info,
            enabled=True,
        )

    # L4 - The production runner dispatches PPO, A2C, SAC, or TD3 through this
    # factory instead of hard-coding a learner class here.
    algo_class, algo_kwargs = get_algorithm_components(
        config,
        l4_adapter,
        obs_key,
        policy_kwargs,
        visualizer=l5_visualizer,
        log_dir=log_dir,
        config_path=config_path,
        checkpoint_path=None,
    )
    l4_learner = algo_class(
        device=training.device,
        seed=training.seed,
        **algo_kwargs,
    )

    callbacks = L4_CallbackList(
        [
            L4_CheckpointCallback(
                save_freq=training.save_freq,
                save_path=str(checkpoint_dir),
                name_prefix=f"{algo.name}_model",
                verbose=1,
                save_on="update_end" if algo.name.lower() in ("ppo", "a2c") else "step",
            ),
        ]
    )

    try:
        l4_learner.learn(
            total_timesteps=algo.total_timesteps,
            callback=callbacks,
            log_interval=training.log_interval,
            reset_num_timesteps=True,
        )
        l4_learner.save(str(checkpoint_dir / FINAL_MODEL_NAME))
    finally:
        l3_pool.close()
        if l5_visualizer is not None:
            l5_visualizer.close()


def leaderboard_evaluation_pipeline(
    agent_config_path: str,
    carla_host: str,
    carla_port: int,
    route_context: Dict[str, Any],
    global_plan_gps: Iterable[Any],
    global_plan_world_coord: Iterable[Any],
    num_sim_ticks: int = 1000,
) -> None:
    # L5 - The parallel launcher starts the official evaluator. The evaluator
    # loads RLLeaderboardAgent, injects route context / global plan, and then
    # drives the agent through setup -> run_step -> destroy.
    l5_agent = L5_RLLeaderboardAgent(carla_host, carla_port, debug=False)

    try:
        l5_agent.set_route_context(route_context)
        l5_agent.set_global_plan(global_plan_gps, global_plan_world_coord)
        l5_agent.setup(agent_config_path)

        for _sim_tick in range(num_sim_ticks):
            _vehicle_control = l5_agent.run_step(input_data={}, timestamp=_sim_tick)
    finally:
        l5_agent.destroy()


def build_leaderboard_runtime_from_agent_config(
    agent_config_path: str,
    dense_route_plan: Iterable[Any],
    route_name: str,
) -> L5_LeaderboardEvalRuntime:
    # This is the core of RLLeaderboardAgent.setup(): restore the L4 learner
    # from checkpoint, then wrap it in the L5 closed-loop runtime.
    l5_agent_config = load_agent_config(agent_config_path)
    l4_bundle = load_model_bundle(
        l5_agent_config.rl_config_path,
        l5_agent_config.checkpoint_path,
    )

    l5_runtime = L5_LeaderboardEvalRuntime(
        model=l4_bundle.model,
        env_config=l4_bundle.env_config,
        route=dense_route_plan,
        route_name=route_name,
        output_root=l5_agent_config.output_root,
        stochastic=l5_agent_config.stochastic,
    )
    l5_runtime.reset()
    return l5_runtime


def merge_evaluation_results(num_envs: int) -> Dict[str, Any]:
    return L5_RLStatisticsManager.merge_results(
        result_dir=EVAL_RESULT_DIR,
        num_envs=num_envs,
    )


if __name__ == "__main__":
    raise SystemExit(
        "This file is a pipeline walkthrough. Use "
        "tools/launch/baseline/train.sh or "
        "tools/launch/evaluation/run_leaderboard_eval_parallel.sh "
        "for actual jobs."
    )
