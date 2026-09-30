"""ScenarioManagerRL: RL fork of Leaderboard's ScenarioManager.

* Owns the lifecycle of one scenario route (load -> tick -> cleanup).
* Does not use a Leaderboard Agent; the RL env consumes actions via
  ``step()`` instead.
* Replaces the original ``StatisticsManager`` with the RL-specific
  ``RLStatisticsManager`` (multi-env safe, crash-recoverable).
"""

import time
import logging
from typing import TYPE_CHECKING

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.timer import GameTime

if TYPE_CHECKING:
    from b2d_rlinfra.evaluation.rl_statistics_manager import RLStatisticsManager

logger = logging.getLogger("Carla Env")

__layer__ = (1, "Scenario")


class ScenarioManagerRL:
    """Scenario manager adapted for external RL actions and RL statistics."""

    def __init__(self, timeout: float, statistics_manager: 'RLStatisticsManager'):
        """Initialize with a timeout in seconds and an RL statistics manager."""
        self.config = None
        self.scenario = None
        self.scenario_tree = None
        self.ego_vehicles = None

        self._timeout = float(timeout)
        self._running = False
        self._timestamp_last_run = 0.0

        self.start_system_time = 0.0
        self.start_game_time = 0.0
        self.end_system_time = 0.0
        self.end_game_time = 0.0
        self.scenario_duration_system = 0.0
        self.scenario_duration_game = 0.0

        self._watchdog = None
        self._agent_watchdog = None

        self._statistics_manager = statistics_manager

    def load_scenario(self, scenario, config):
        """Load a scenario and notify the statistics manager."""
        GameTime.restart()
        self.config = config
        self.scenario = scenario
        self.scenario_tree = scenario.scenario_tree
        self.ego_vehicles = scenario.ego_vehicles

        if self._statistics_manager:
            route_id = f"{config.name}_rep{config.repetition_index}" if hasattr(config, 'name') else str(config)
            route_length = self._resolve_route_length(config, scenario)
            self._statistics_manager.start_route(route_id, route_length)
            self._statistics_manager.set_scenario(scenario, route_length)
    
    def _resolve_route_length(self, config, scenario) -> float:
        """
        Resolve the runtime route used for leaderboard statistics.

        In this RL pipeline the reset caller passes the route config object, which
        may only contain keypoints while the fully interpolated route lives on the
        constructed RouteScenario. Prefer scenario.route so the stored route length
        matches the actual executed route.
        """
        route = getattr(scenario, 'route', None)
        if not route:
            route = getattr(config, 'route', None)
        return self._compute_route_length(route)

    def _compute_route_length(self, route) -> float:
        """Compute the total route length in metres."""
        if not route:
            return 0.0
        
        route_length = 0.0
        previous_location = None
        
        for transform, _ in route:
            location = transform.location
            if previous_location:
                dist_vec = location - previous_location
                route_length += dist_vec.length()
            previous_location = location
        
        return route_length
    
    def stop_scenario(self):
        """Stop the scenario, finalise statistics, and stop watchdogs."""
        self._running = False

        # Stop the watchdog so timeout exceptions cannot fire afterwards.
        if self._watchdog:
            try:
                self._watchdog.stop()
            except Exception:
                pass
        
        if self._agent_watchdog:
            try:
                self._agent_watchdog.stop()
            except Exception:
                pass
        
        self.end_system_time = time.time()
        self.end_game_time = GameTime.get_time()
        
        self.scenario_duration_system = self.end_system_time - self.start_system_time
        self.scenario_duration_game = self.end_game_time - self.start_game_time

        # Mirror the official leaderboard stop order: terminate the scenario
        # before reading criteria so terminate()-only events (for example
        # timeout / yield / min-speed finalization) are visible to statistics.
        if self.scenario is not None:
            try:
                self.scenario.terminate()
            except Exception as e:
                logger.warning(
                    "ScenarioManagerRL.stop_scenario: scenario.terminate() failed; "
                    "falling back to partial statistics: %s",
                    e,
                )

        if self._statistics_manager:
            return self._statistics_manager.end_route(
                duration_game=self.scenario_duration_game,
                duration_system=self.scenario_duration_system
            )
        return None
    
    def cleanup(self):
        """Release scenario / actor / data-provider resources."""
        self._timestamp_last_run = 0.0
        self.scenario_duration_system = 0.0
        self.scenario_duration_game = 0.0
        
        if self._watchdog:
            self._watchdog.stop()
            self._watchdog = None
        
        if self._agent_watchdog:
            self._agent_watchdog.stop()
            self._agent_watchdog = None
        
        if self.scenario:
            try:
                self.scenario.terminate()
            except Exception as e:
                logger.warning(f"ScenarioManagerRL.cleanup: failed to terminate scenario: {e}")
            self.scenario = None
        
        self.scenario_tree = None
        self.ego_vehicles = None
        self.config = None
        
        try:
            CarlaDataProvider.cleanup()
        except Exception as e:
            logger.warning(f"ScenarioManagerRL.cleanup: failed to cleanup CarlaDataProvider: {e}")
