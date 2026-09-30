"""CARLAEnvPool async rollout demo.

A runnable tutorial showing how to interact with CARLAEnvPool directly.
No training, no policy network -- just fixed actions driving the pool to
illustrate the worker-indexed async lifecycle:

    pool.reset()  → {worker_id: obs}         # not a fixed-length array
    pool.step()   → obs, rewards, terms, truncs, infos   # dicts

Key differences from Gym/VecEnv that this demo highlights:

  - step() input/output are {worker_id: value} dicts with varying key sets.
  - Terminal observation belongs to the OLD episode; reset obs arrives later.
  - Crash produces no valid observation; the whole episode should be discarded.
  - pool.step({}) is a valid poll that drains queued results without dispatch.

Usage:
    bash b2d_rlinfra/framework/env_rollout_demo.sh
"""

from __future__ import annotations
import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Set

from b2d_rlinfra.simulation.runners.carla_env_pool import CARLAEnvPool
from b2d_rlinfra.framework.demo_utils import (
    DemoRolloutBuffer,
    load_demo_runtime_config,
    print_crash_discard,
    print_reset_ready,
    print_terminal_observation,
    print_worker_step,
)

# =============================================================================
# Env factory -- called once per worker process inside CARLAEnvPool
# =============================================================================
# Wrapper stacking order (bottom to top):
#   CARLAEnv → RoutePlan → Observation → EventTermination → Reward → Action

def make_demo_env(
    config: Dict[str, Any],
    worker_id: int,
    carla_port: int,
    traffic_manager_port: int,
    adaptive_shared_stats: Optional[Any] = None,
    adaptive_shared_lock: Optional[Any] = None,
) -> Any:
    """Build the wrapped env for a single pool worker."""
    from b2d_rlinfra.environment.carla_env import CARLAEnv
    from b2d_rlinfra.environment.wrappers import (
        ActionWrapper,
        EventTerminationWrapper,
        ObservationWrapper,
        RewardWrapper,
        RoutePlanWrapper,
    )

    env = CARLAEnv(
        config,
        env_index=worker_id,
        port=carla_port,
        traffic_manager_port=traffic_manager_port,
        adaptive_shared_stats=adaptive_shared_stats,
        adaptive_shared_lock=adaptive_shared_lock,
    )
    env = RoutePlanWrapper(env)
    env = ObservationWrapper(env)
    env = EventTerminationWrapper(env)
    env = RewardWrapper(env)
    env = ActionWrapper(env)
    return env

# =============================================================================
# Main demo
# =============================================================================

def run_demo(config_path: Path) -> None:
    runtime = load_demo_runtime_config(config_path)
    demo_buffer = DemoRolloutBuffer()

    pool: Optional[CARLAEnvPool] = None
    ready_obs: Dict[int, Any] = {}          # next obs that can be passed to the agent
    pending_action: Dict[int, Any] = {}     # actions waiting for step results
    pending_obs: Dict[int, Any] = {}        # obs paired with pending actions
    completed_step_results = 0
    pool_step_index = 0

    print(
        f"[demo] config={runtime.config_path} "
        f"workers={runtime.num_envs} steps={runtime.steps} "
        f"agent_mode={runtime.agent.mode}",
        flush=True,
    )

    try:
        # ---------------------------------------------------------------------
        # 1. Start pool and collect initial reset observations.
        # ---------------------------------------------------------------------
        pool = CARLAEnvPool(
            env_fn=make_demo_env,
            config=runtime.env_config,
            num_envs=runtime.num_envs,
            auto_reset=True,
            manage_servers=runtime.manage_servers,
            max_episode_steps=runtime.max_episode_steps,
        )

        reset_obs, reset_infos = pool.reset(
            min_ready=runtime.min_ready,
            timeout=runtime.reset_timeout,
        )

        print(f"[demo] initial reset ready workers={list(reset_obs)}", flush=True)
        if not reset_obs:
            raise RuntimeError(
                "pool.reset() returned no observations -- "
                "check CARLA startup and port config."
            )

        for worker_id, observation in reset_obs.items():
            reset_info = reset_infos.get(worker_id, {})
            demo_buffer.start_episode(worker_id, reset_info)
            ready_obs[worker_id] = observation
            print_reset_ready(worker_id, observation, reset_info)

        # ---------------------------------------------------------------------
        # 2. Async rollout loop.
        # ---------------------------------------------------------------------
        # Each iteration: dispatch actions → collect async pool results → handle returns.
        # The returned worker set may differ from the dispatched set because
        # the pool is asynchronous.
        while completed_step_results < runtime.steps:

            # -----------------------------------------------------------------
            # 2a. Dispatch actions for workers that currently have ready obs.
            # -----------------------------------------------------------------
            # Dispatch: send actions only to workers with a usable observation.
            # An empty dispatch ({}) is a poll that drains queued results.
            if ready_obs:
                dispatch_workers = list(ready_obs)
                actions = {}
                for wid in dispatch_workers:
                    observation = ready_obs.pop(wid)
                    action = runtime.agent.act(wid, observation)
                    actions[wid] = action
                    pending_obs[wid] = observation
                    pending_action[wid] = action
            else:
                dispatch_workers = []
                actions = {}

            print(
                f"\n[pool step {pool_step_index}] "
                f"dispatch workers={dispatch_workers} actions={actions}",
                flush=True,
            )

            # -----------------------------------------------------------------
            # 2b. Step the pool and collect whichever workers are ready.
            # -----------------------------------------------------------------
            # Step: returns 5 worker-indexed dicts whose key sets can differ.
            # A reset observation can arrive without a reward.
            obs_dict, rewards, terms, truncs, infos = pool.step(
                actions,
                min_ready=runtime.min_ready,
                timeout=runtime.step_timeout,
            )

            print(
                f"[pool step {pool_step_index}] "
                f"return   obs={list(obs_dict)} rewards={list(rewards)}",
                flush=True,
            )
            pool_step_index += 1

            # -----------------------------------------------------------------
            # 2c. Process step results from workers that have a reward.
            # -----------------------------------------------------------------
            handled: Set[int] = set()

            for worker_id in rewards:
                handled.add(worker_id)
                info = infos.get(worker_id, {})
                reward = float(rewards[worker_id])
                terminated = bool(terms.get(worker_id, False))
                truncated = bool(truncs.get(worker_id, False))

                # CRASH: no valid obs; discard entire episode and wait for
                # the worker to restart and emit a new reset obs.
                if info.get("crashed"):
                    pending_obs.pop(worker_id, None)
                    pending_action.pop(worker_id, None)
                    print_crash_discard(
                        worker_id, info, demo_buffer.buffered_steps(worker_id)
                    )
                    demo_buffer.discard_episode(worker_id)
                    ready_obs.pop(worker_id, None)
                    completed_step_results += 1
                    continue

                transition_obs = pending_obs.pop(worker_id)
                action = pending_action.pop(worker_id)
                observation = obs_dict.get(worker_id)

                print_worker_step(
                    worker_id, observation, reward,
                    terminated, truncated, info,
                )

                demo_buffer.append_step(
                    worker_id,
                    transition_obs,
                    action,
                    reward,
                    terminated,
                    truncated,
                    info,
                )
                completed_step_results += 1

                if terminated or truncated:
                    # TERMINAL: this obs is the LAST frame of the old episode.
                    # Do NOT reuse it as the start of the next episode.
                    # Remove from ready table; the reset obs arrives later.
                    print_terminal_observation(
                        worker_id, observation, terminated, truncated
                    )
                    demo_buffer.finish_episode(worker_id)
                    ready_obs.pop(worker_id, None)
                elif observation is not None:
                    # NORMAL: observation is usable for the next dispatch.
                    ready_obs[worker_id] = observation

            # -----------------------------------------------------------------
            # 2d. Process obs-only entries, typically reset observations.
            # -----------------------------------------------------------------
            # These appear when a worker finishes auto-reset after a terminal.
            # Identified by info["from_reset"] or info["episode_start"].
            for worker_id in obs_dict:
                if worker_id in handled:
                    continue
                info = infos.get(worker_id, {})

                if info.get("from_reset") or info.get("episode_start"):
                    observation = obs_dict[worker_id]
                    demo_buffer.start_episode(worker_id, info)
                    ready_obs[worker_id] = observation
                    print_reset_ready(worker_id, observation, info)
                else:
                    ready_obs[worker_id] = obs_dict[worker_id]

        # ---------------------------------------------------------------------
        # 3. Summary.
        # ---------------------------------------------------------------------
        print(
            f"\n[demo] done. step_results={completed_step_results} "
            f"committed_episodes={demo_buffer.committed_episodes} "
            f"discarded_episodes={demo_buffer.discarded_episodes}",
            flush=True,
        )

    finally:
        demo_buffer.clear()
        if pool is not None:
            pool.close()

# =============================================================================
# CLI entry point
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CARLAEnvPool async rollout demo."
    )
    parser.add_argument(
        "--config",
        default="configs/env_rollout_demo.yaml",
        help="YAML config path. All demo behavior is configured there.",
    )
    return parser.parse_args()

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()
    run_demo(Path(args.config).expanduser().resolve())

if __name__ == "__main__":
    main()
