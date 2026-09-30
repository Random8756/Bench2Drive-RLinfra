"""Leaderboard agent entry point for RL checkpoint evaluation."""
from __future__ import annotations

from typing import Dict, Optional

from leaderboard.autoagents.autonomous_agent import AutonomousAgent

from b2d_rlinfra.evaluation.leaderboard.runtime_config import load_agent_config
from b2d_rlinfra.evaluation.leaderboard.model_loader import load_model_bundle
from b2d_rlinfra.evaluation.runtime.runtime import LeaderboardEvalRuntime

__layer__ = (5, "Evaluation")


def get_entry_point():
    return "RLLeaderboardAgent"


class RLLeaderboardAgent(AutonomousAgent):
    def __init__(self, carla_host, carla_port, debug=False):
        super().__init__(carla_host, carla_port, debug)
        self._route_context: Dict = {}
        self._dense_global_plan = None
        self._agent_config = None
        self._l5_runtime: Optional[LeaderboardEvalRuntime] = None

    def set_route_context(self, context: Dict) -> None:
        self._route_context = dict(context or {})

    def set_global_plan(self, global_plan_gps, global_plan_world_coord):
        self._dense_global_plan = list(global_plan_world_coord)
        super().set_global_plan(global_plan_gps, global_plan_world_coord)

    def setup(self, path_to_conf_file):
        self._agent_config = load_agent_config(path_to_conf_file)
        l4_bundle = load_model_bundle(
            self._agent_config.rl_config_path,
            self._agent_config.checkpoint_path,
        )
        if not self._dense_global_plan:
            raise ValueError("Dense global plan was not set before agent setup")

        route_name = self._route_context.get("route_name", "route")
        self._l5_runtime = LeaderboardEvalRuntime(
            model=l4_bundle.model,
            env_config=l4_bundle.env_config,
            route=self._dense_global_plan,
            route_name=route_name,
            output_root=self._agent_config.output_root,
            stochastic=self._agent_config.stochastic,
        )
        self._l5_runtime.reset()

    def sensors(self):
        return []

    def run_step(self, input_data, timestamp):
        if self._l5_runtime is None:
            raise RuntimeError("Runtime is not initialized")
        return self._l5_runtime.run_step()

    def destroy(self):
        if self._l5_runtime is not None:
            self._l5_runtime.close()
            self._l5_runtime = None
