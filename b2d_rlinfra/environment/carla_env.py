"""CARLAEnv: Gym-style environment wrapping CARLA + Leaderboard 2.0.

Implements ``step() / reset() / render()`` with configurable observation,
action and reward spaces driven by YAML.
"""
from collections import defaultdict
from typing import Optional, Dict, Any, List

__layer__ = (2, "Environment")

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.timer import GameTime
from srunner.scenariomanager.watchdog import Watchdog

from leaderboard.utils.route_indexer import RouteIndexer

from leaderboard.scenarios.route_scenario import RouteScenario, RouteScenarioSetupError
from leaderboard.utils.route_manipulation import downsample_route

from b2d_rlinfra.evaluation.rl_statistics_manager import RLStatisticsManager
from b2d_rlinfra.scenario.scenario_manager_rl import ScenarioManagerRL

from b2d_rlinfra.simulation.carla_connection import CARLAConnection, CARLAConnectionError
from b2d_rlinfra.simulation.crash_utils import refine_crash_payload
from b2d_rlinfra.scenario.adaptive_route_sampler import (
    ensure_shared_scenarios,
    normalize_adaptive_config,
    select_scenario_name_for_sample,
    update_shared_sampling_state_with_periodic_snapshot,
)

import py_trees
import random

import signal
import logging
import os

import carla
import cv2 as cv
import gymnasium as gym
from gymnasium import spaces
import time

logger = logging.getLogger("Carla Env")

SIMULATOR_TICK_TIMEOUT_LIMIT = 60.0  # seconds
SIMULATOR_RESET_TIMEOUT_LIMIT = 30   # s
TILE_STREAM_DISTANCE = 100
ACTOR_ACTIVE_DISTANCE = 100
# Default to conservative behavior: always reload the world on reset.
# Set to False to allow same-town resets to skip client.load_world().
FORCE_FULL_LOAD_EVERY_RESET = True


class EpisodeCrashError(RuntimeError):
    """Structured crash propagated from env reset/step to the worker."""

    def __init__(self, crash_reason: str, crash_type: str, crash_detail: str):
        super().__init__(crash_detail or crash_reason)
        self.crash_reason = crash_reason
        self.crash_type = crash_type
        self.crash_detail = crash_detail or crash_reason


class CARLATimingStats:
    """Pure CARLA-API timing statistics.

    Only direct CARLA client calls are tracked, no wrapper or higher-level
    logic, so it can be used to diagnose FPS regressions.
    """

    def __init__(self):
        self.reset_stats()

    def reset_stats(self):
        """Reset all timing counters."""
        # Cumulative durations in seconds.
        self.world_tick_total = 0.0
        self.apply_control_total = 0.0
        self.get_velocity_total = 0.0
        self.get_control_total = 0.0
        self.get_location_total = 0.0
        self.scenario_tree_tick_total = 0.0
        self.on_carla_tick_total = 0.0      # GameTime + CarlaDataProvider tick
        self.get_snapshot_total = 0.0
        self.load_world_total = 0.0
        self.reset_total = 0.0              # full reset() duration

        self.world_tick_count = 0
        self.apply_control_count = 0
        self.get_velocity_count = 0
        self.get_control_count = 0
        self.get_location_count = 0
        self.scenario_tree_tick_count = 0
        self.on_carla_tick_count = 0
        self.get_snapshot_count = 0
        self.load_world_count = 0
        self.reset_count = 0

        # Sliding windows over the most recent N samples.
        self._window_size = 100
        self._recent_world_tick = []
        self._recent_step_total = []

    def record(self, name: str, elapsed: float):
        """Record a single API-call duration."""
        total_attr = f"{name}_total"
        count_attr = f"{name}_count"
        if hasattr(self, total_attr):
            setattr(self, total_attr, getattr(self, total_attr) + elapsed)
            setattr(self, count_attr, getattr(self, count_attr) + 1)

        if name == 'world_tick':
            self._recent_world_tick.append(elapsed)
            if len(self._recent_world_tick) > self._window_size:
                self._recent_world_tick.pop(0)

    def record_step_total(self, elapsed: float):
        """Record one full ``step()`` duration."""
        self._recent_step_total.append(elapsed)
        if len(self._recent_step_total) > self._window_size:
            self._recent_step_total.pop(0)

    def get_summary(self) -> dict:
        """Return a dict of averaged timings (in milliseconds)."""
        def avg(total, count):
            return (total / count * 1000) if count > 0 else 0.0

        def recent_avg(lst):
            return (sum(lst) / len(lst) * 1000) if lst else 0.0
        
        return {
            'world_tick_avg_ms': avg(self.world_tick_total, self.world_tick_count),
            'world_tick_recent_avg_ms': recent_avg(self._recent_world_tick),
            'apply_control_avg_ms': avg(self.apply_control_total, self.apply_control_count),
            'get_velocity_avg_ms': avg(self.get_velocity_total, self.get_velocity_count),
            'get_control_avg_ms': avg(self.get_control_total, self.get_control_count),
            'get_location_avg_ms': avg(self.get_location_total, self.get_location_count),
            'scenario_tree_tick_avg_ms': avg(self.scenario_tree_tick_total, self.scenario_tree_tick_count),
            'on_carla_tick_avg_ms': avg(self.on_carla_tick_total, self.on_carla_tick_count),
            'get_snapshot_avg_ms': avg(self.get_snapshot_total, self.get_snapshot_count),
            'load_world_avg_ms': avg(self.load_world_total, self.load_world_count),
            'reset_avg_ms': avg(self.reset_total, self.reset_count),
            'step_recent_avg_ms': recent_avg(self._recent_step_total),
            'world_tick_count': self.world_tick_count,
            'load_world_count': self.load_world_count,
            'reset_count': self.reset_count,
        }
    
    def log_summary(self, env_index: int, step_count: int):
        """Log a formatted timing summary."""
        s = self.get_summary()
        logger.debug(
            f"\n{'='*70}\n"
            f"  [CARLA Timing] Env[{env_index}] after {step_count} steps\n"
            f"  world.tick()        : avg={s['world_tick_avg_ms']:.2f}ms  recent={s['world_tick_recent_avg_ms']:.2f}ms  (n={s['world_tick_count']})\n"
            f"  apply_control()     : avg={s['apply_control_avg_ms']:.2f}ms\n"
            f"  get_velocity()      : avg={s['get_velocity_avg_ms']:.2f}ms\n"
            f"  get_control()       : avg={s['get_control_avg_ms']:.2f}ms\n"
            f"  get_location()      : avg={s['get_location_avg_ms']:.2f}ms\n"
            f"  get_snapshot()      : avg={s['get_snapshot_avg_ms']:.2f}ms\n"
            f"  on_carla_tick()     : avg={s['on_carla_tick_avg_ms']:.2f}ms\n"
            f"  scenario_tree.tick(): avg={s['scenario_tree_tick_avg_ms']:.2f}ms\n"
            f"  load_world()        : avg={s['load_world_avg_ms']:.2f}ms  (n={s['load_world_count']})\n"
            f"  reset() total       : avg={s['reset_avg_ms']:.2f}ms  (n={s['reset_count']})\n"
            f"  step total (recent) : avg={s['step_recent_avg_ms']:.2f}ms\n"
            f"{'='*70}"
        )


class CARLAEnv(gym.Env):
    """Gymnasium environment backed by a configured CARLA server."""

    def __init__(
        self,
        config: Dict[str, Any],
        env_index: int = 0,
        # Optional direct port override (takes precedence over config).
        port: Optional[int] = None,
        traffic_manager_port: Optional[int] = None,
        adaptive_shared_stats: Optional[Any] = None,
        adaptive_shared_lock: Optional[Any] = None,
        # Whether to wait for the CARLA server before initialising.
        wait_for_server: bool = True
    ):
        """Initialize one worker, optionally overriding its configured ports."""
        self.config = config
        self.env_index = env_index
        self._wait_for_server = wait_for_server

        self._port, self._tm_port, self._tm_seed = self._parse_port_config(
            config, env_index, port, traffic_manager_port
        )

        logger.debug(f"CARLAEnv[{env_index}] initializing with port={self._port}, "
                   f"traffic_manager_port={self._tm_port}")

        # Connection manager (connects lazily).
        self._connection: Optional[CARLAConnection] = None
        self._connected = False

        # CARLA-related state.
        self.client = None
        self.world = None
        self.traffic_manager = None
        self.scenario = None
        self.count = 0
        self.crash_message = ""
        self.crash_reason = ""
        self.crash_type = ""
        self.crash_detail = ""

        # CARLA-API-level timing (excludes wrapper logic).
        self.timing_stats = CARLATimingStats()

        # Adaptive route-sampling state.
        self._adaptive_config: Dict[str, Any] = {}
        self._adaptive_route_buckets: Dict[str, List[int]] = {}
        self._adaptive_shared_stats = adaptive_shared_stats
        self._adaptive_shared_lock = adaptive_shared_lock
        self._adaptive_retry_route_index: Optional[int] = None
        self._adaptive_sampled_count = 0
        self._last_sampled_route_filtered_index: Optional[int] = None
        
        # Initialise Leaderboard components (no CARLA connection required).
        self._init_leaderboard_components()

        # action_space / observation_space are set by the wrappers; CARLAEnv
        # leaves them as None to avoid conflicts.
        self.action_space = None
        self.observation_space = None
        self._current_route_id = "unknown"
        self.scenario_name = "unknown"
        self.scenario_instance_name = ""
        self._current_town = "unknown"
        self._init_info()

        self._connect()

    def _parse_port_config(
        self,
        config: Dict,
        env_index: int,
        port_override: Optional[int],
        tm_port_override: Optional[int]
    ) -> tuple:
        """Resolve ports from config + overrides.

        Returns:
            ``(port, traffic_manager_port, traffic_manager_seed)``.
        """
        carla_config = config.get('carla', {})

        # CARLA RPC port.
        if port_override is not None:
            port = port_override
        elif 'port' in carla_config:
            carla_port = carla_config['port']
            if isinstance(carla_port, list):
                if env_index >= len(carla_port):
                    raise ValueError(f"env_index {env_index} out of range, only {len(carla_port)} ports configured")
                port = carla_port[env_index]
            else:
                port = carla_port
        else:
            port = 2000

        # Traffic Manager port.
        if tm_port_override is not None:
            tm_port = tm_port_override
        elif 'traffic_manager_port' in carla_config:
            carla_tm_port = carla_config['traffic_manager_port']
            if isinstance(carla_tm_port, list):
                tm_port = carla_tm_port[env_index] if env_index < len(carla_tm_port) else port + 6000
            else:
                tm_port = carla_tm_port
        else:
            tm_port = port + 6000
        
        # Traffic Manager seed.
        if 'traffic_manager_seed' in carla_config:
            carla_tm_seed = carla_config['traffic_manager_seed']
            if isinstance(carla_tm_seed, list):
                tm_seed = carla_tm_seed[env_index] if env_index < len(carla_tm_seed) else env_index
            else:
                tm_seed = carla_tm_seed
        else:
            tm_seed = 0
        
        return port, tm_port, tm_seed
    
    def _init_leaderboard_components(self):
        """Initialise Leaderboard components that do not require a CARLA connection.

        Sets up:
            - ``RLStatisticsManager``: RL-specific statistics manager (multi-env
              safe, crash-recoverable, resume-friendly).
            - ``ScenarioManagerRL``: simplified scenario manager.
            - ``RouteIndexer``: route sampler.

        ``observation_space`` itself is set later by ``ObservationWrapper`` and
        synchronised to ``CarlaDataProvider``.
        """
        env_config = self.config.get('environment', {})
        result_dir_base = env_config.get('result_dir', './results')

        # Resolve to an absolute path so subprocesses see the same dir.
        result_dir_base = os.path.abspath(result_dir_base)

        # Compose the result-dir suffix from optional experiment name +
        # timestamp. The timestamp may be injected by ``CARLAEnvPool`` to
        # keep all workers consistent.
        experiment_name = env_config.get('experiment_name', None)
        use_timestamp = env_config.get('use_timestamp', True)
        experiment_timestamp = env_config.get('_experiment_timestamp', None)

        if experiment_timestamp is None and use_timestamp:
            from datetime import datetime
            experiment_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if experiment_name:
            if use_timestamp and experiment_timestamp:
                result_dir = os.path.join(result_dir_base, f"{experiment_name}_{experiment_timestamp}")
            else:
                result_dir = os.path.join(result_dir_base, experiment_name)
        elif use_timestamp and experiment_timestamp:
            result_dir = os.path.join(result_dir_base, f"exp_{experiment_timestamp}")
        else:
            result_dir = result_dir_base

        os.makedirs(result_dir, exist_ok=True)
        logger.debug(f"Result directory: {result_dir}")

        self.result_dir = result_dir

        # Persist a snapshot after every completed route.
        self.statistics_manager = RLStatisticsManager(
            result_dir=result_dir,
            env_id=self.env_index,
            auto_save=True,
            save_interval=1,
        )

        carla_config = self.config.get('carla', {})
        timeout = carla_config.get('timeout', 60)
        self.scenario_manager = ScenarioManagerRL(
            timeout=timeout,
            statistics_manager=self.statistics_manager
        )

        routes_config = self.config.get('routes', {})
        routes_file = routes_config.get('route_files', ['resources/routes/train_routes_demo.xml'])
        if isinstance(routes_file, list):
            routes_file = routes_file[0] if routes_file else 'resources/routes/train_routes_demo.xml'
        repetitions = routes_config.get('repetitions', 1)
        routes_subset = routes_config.get('routes_subset', None)

        # Sampling mode: sequential / random / adaptive.
        self._sample_mode = str(routes_config.get('sample_mode', 'sequential')).strip().lower()
        if self._sample_mode not in {'sequential', 'random', 'adaptive'}:
            raise ValueError(f"Unsupported route sample_mode: {self._sample_mode}")

        self.route_sampler = RouteIndexer(
            routes_file=routes_file,
            repetitions=repetitions,
            routes_subset=routes_subset,
            warmup=routes_config.get('warmup', False),
            training=routes_config.get('training', True)
        )
        raw_total = self.route_sampler.get_length()

        # Town partition: enabled in random/adaptive sampling modes only -
        # routes from heavy towns are pinned to dedicated workers to reduce
        # town-switching memory leaks. Sequential mode must keep the full
        # route order intact.
        town_partition = routes_config.get('town_partition', {})
        heavy_towns = set(town_partition.get('heavy_towns', []))
        heavy_ratio = town_partition.get('heavy_ratio', 0.0)

        self._route_index_map = None  # filtered index -> actual route index
        if self._sample_mode in {'random', 'adaptive'} and heavy_towns and heavy_ratio > 0:
            num_envs = self.config.get('carla', {}).get('num_envs', 1)
            n_heavy = max(1, round(num_envs * heavy_ratio))

            is_heavy_worker = (self.env_index < n_heavy)

            valid_indices = []
            for i in range(raw_total):
                rc = self.route_sampler.get_index_config(i)
                town = rc.town if hasattr(rc, 'town') else ''
                if is_heavy_worker and town in heavy_towns:
                    valid_indices.append(i)
                elif not is_heavy_worker and town not in heavy_towns:
                    valid_indices.append(i)

            self._route_index_map = valid_indices
            self._total_routes = len(valid_indices)
            worker_tag = "heavy" if is_heavy_worker else "normal"
            logger.info(
                f"CARLAEnv[{self.env_index}] town_partition: {worker_tag} worker, "
                f"{self._total_routes}/{raw_total} routes "
                f"(heavy_towns={heavy_towns}, n_heavy={n_heavy})"
            )
        else:
            self._total_routes = raw_total
            if self._sample_mode != 'random' and heavy_towns and heavy_ratio > 0:
                logger.debug(
                    f"CARLAEnv[{self.env_index}] sample_mode={self._sample_mode}, "
                    "ignoring town_partition"
                )
        
        # Random mode uses shuffle-then-repeat: shuffle once and walk the
        # list, reshuffling at the start of each new cycle so every route is
        # visited evenly.
        self._shuffle_indices = []
        self._shuffle_pos = 0
        self._shuffle_cycle = 0
        self._random_sampled_count = 0
        if self._sample_mode == 'random' and self._total_routes > 0:
            self._shuffle_indices = list(range(self._total_routes))
            random.shuffle(self._shuffle_indices)
        elif self._sample_mode == 'adaptive':
            self._init_adaptive_sampler(routes_config)

        # Last sampled route, used to retry on crash recovery.
        self._last_sampled_route = None

        # Resume: skip past routes already completed in the statistics file.
        self._resume_route_index_from_statistics()

        logger.debug(f"Route sampler initialized: {self._total_routes} routes, mode={self._sample_mode}")

    def _get_filtered_route_id(self, filtered_idx: int) -> str:
        """Map a filtered index (after ``town_partition``) back to a route id."""
        actual = (
            self._route_index_map[filtered_idx]
            if self._route_index_map is not None
            else filtered_idx
        )
        cfg = self.route_sampler._configs_list[actual]
        return cfg.name + "_rep" + str(cfg.repetition_index)

    def _resume_route_index_from_statistics(self):
        """Advance ``RouteIndexer`` past routes already completed.

        Used after a worker crash: when the worker restarts the
        ``RouteIndexer`` starts at 0 again, but the statistics file may
        already contain completed entries. Skipping them here ensures the
        sampler resumes at the first unfinished route.
        """
        if not self.statistics_manager:
            return
        completed_ids = self.statistics_manager.get_completed_route_ids()
        if not completed_ids:
            return
        
        if self._sample_mode == 'sequential':
            original_index = self.route_sampler.index
            while self.route_sampler.index < self._total_routes:
                config = self.route_sampler._configs_list[self.route_sampler.index]
                route_id = config.name + "_rep" + str(config.repetition_index)
                if route_id in completed_ids:
                    self.route_sampler.index += 1
                else:
                    break
            if self.route_sampler.index > original_index:
                logger.debug(
                    f"Route sampler resumed: index {original_index} -> {self.route_sampler.index} "
                    f"(skipped {self.route_sampler.index - original_index} completed routes)"
                )
        elif self._sample_mode == 'random':
            def _get_route_id(filtered_idx):
                actual = self._route_index_map[filtered_idx] if self._route_index_map else filtered_idx
                cfg = self.route_sampler._configs_list[actual]
                return cfg.name + "_rep" + str(cfg.repetition_index)
            self._shuffle_indices = [
                i for i in self._shuffle_indices
                if _get_route_id(i) not in completed_ids
            ]
            self._shuffle_pos = 0
            if len(self._shuffle_indices) < self._total_routes:
                skipped = self._total_routes - len(self._shuffle_indices)
                logger.debug(
                    f"Route sampler resumed (random): removed {skipped} completed routes "
                    f"from shuffle list, {len(self._shuffle_indices)} remaining"
                )

    def _get_actual_route_index(self, filtered_index: int) -> int:
        return self._route_index_map[filtered_index] if self._route_index_map else filtered_index

    def _get_route_config_by_filtered_index(self, filtered_index: int):
        actual_index = self._get_actual_route_index(filtered_index)
        return self.route_sampler.get_index_config(actual_index)

    def _init_adaptive_sampler(self, routes_config: Dict[str, Any]) -> None:
        self._adaptive_config = normalize_adaptive_config(routes_config.get('adaptive', {}))
        self._adaptive_shared_stats = routes_config.get('_adaptive_shared_stats', self._adaptive_shared_stats)
        self._adaptive_shared_lock = routes_config.get('_adaptive_shared_lock', self._adaptive_shared_lock)
        self._adaptive_route_buckets = self._build_adaptive_route_buckets()

        try:
            ensure_shared_scenarios(
                self._adaptive_shared_stats,
                self._adaptive_shared_lock,
                self._adaptive_route_buckets.keys(),
            )
        except Exception as e:
            logger.warning(
                "CARLAEnv[%s] failed to initialize adaptive shared stats: %s",
                self.env_index,
                e,
            )

        logger.info(
            "CARLAEnv[%s] adaptive sampler ready: %s scenario buckets, %s routes",
            self.env_index,
            len(self._adaptive_route_buckets),
            self._total_routes,
        )

    def _build_adaptive_route_buckets(self) -> Dict[str, List[int]]:
        buckets: Dict[str, List[int]] = defaultdict(list)
        for filtered_index in range(self._total_routes):
            route = self._get_route_config_by_filtered_index(filtered_index)
            context = self._resolve_episode_context(route=route)
            buckets[context['scenario_name']].append(filtered_index)
        return dict(buckets)

    def _update_adaptive_sampling_state(self, record: Optional[Any]) -> None:
        if self._sample_mode != 'adaptive' or record is None:
            return
        if self._adaptive_shared_stats is None or self._adaptive_shared_lock is None:
            return

        scenario_name = self._normalize_context_value(self.scenario_name, default="unknown")
        try:
            update_result = update_shared_sampling_state_with_periodic_snapshot(
                self._adaptive_shared_stats,
                self._adaptive_shared_lock,
                scenario_name,
                record,
                self._adaptive_config,
            )
            snapshot = (update_result or {}).get("snapshot")
            update_count = (update_result or {}).get("update_count")
            if snapshot:
                lines = [
                    f"CARLAEnv[{self.env_index}] adaptive sampler snapshot "
                    f"(global_updates={update_count})"
                ]
                for name, state in sorted(
                    snapshot.items(),
                    key=lambda item: (-float(item[1]["sampling_weight"]), item[0]),
                ):
                    recent = "".join(str(int(v)) for v in state["recent_results"]) or "-"
                    success_rate = state.get("success_rate")
                    success_rate_text = "-" if success_rate is None else f"{success_rate:.2f}"
                    lines.append(
                        f"  {name}: weight={float(state['sampling_weight']):.2f}, "
                        f"success_rate={success_rate_text}, recent={recent}"
                    )
                logger.info("\n".join(lines))
        except Exception as e:
            logger.warning(
                "CARLAEnv[%s] failed to update adaptive stats for scenario=%s: %s",
                self.env_index,
                scenario_name,
                e,
            )

    def _sample_adaptive_route(self):
        if self._total_routes <= 0:
            return None

        if self._adaptive_retry_route_index is not None:
            retry_route_index = self._adaptive_retry_route_index
            self._adaptive_retry_route_index = None
            route = self._get_route_config_by_filtered_index(retry_route_index)
            self._last_sampled_route = route
            self._last_sampled_route_filtered_index = retry_route_index
            self._adaptive_sampled_count += 1
            return route

        scenario_names = list(self._adaptive_route_buckets.keys())
        if not scenario_names:
            return None

        try:
            selected_scenario = select_scenario_name_for_sample(
                scenario_names,
                self._adaptive_shared_stats,
                self._adaptive_shared_lock,
                self._adaptive_config,
            )
        except Exception as e:
            logger.warning(
                "CARLAEnv[%s] failed to read adaptive shared stats, falling back to uniform sampling: %s",
                self.env_index,
                e,
            )
            selected_scenario = random.choice(scenario_names)
        route_index = random.choice(self._adaptive_route_buckets[selected_scenario])
        route = self._get_route_config_by_filtered_index(route_index)
        self._last_sampled_route = route
        self._last_sampled_route_filtered_index = route_index
        self._adaptive_sampled_count += 1

        logger.debug(
            "CARLAEnv[%s] adaptive sampled scenario=%s route_index=%s",
            self.env_index,
            selected_scenario,
            route_index,
        )
        return route
    
    def _mark_route_as_crashed(
        self,
        route_id: str,
        *,
        crash_reason: str = "",
        crash_type: str = "",
        crash_detail: str = "",
    ):
        """Record a route as crashed before start_route was called.

        This handles "poison routes" that fail during map loading or scenario
        creation (before ``scenario_manager.load_scenario`` → ``start_route``).
        Without this, the route would never appear in the completed-routes set
        and ``RouteIndexer`` would keep re-attempting it on every restart.
        """
        if not route_id or not hasattr(self, 'statistics_manager') or not self.statistics_manager:
            return
        try:
            sm = self.statistics_manager
            record = sm.record_route_crash(
                route_id=route_id,
                failure_message=crash_detail,
                crash_reason=crash_reason,
                crash_type=crash_type,
                crash_detail=crash_detail,
            )
            self._update_adaptive_sampling_state(record)
            logger.warning(
                f"CARLAEnv[{self.env_index}] marked poison route as crashed "
                f"(pre-scenario failure) | {self._episode_context_for_log()}"
            )
        except Exception as e:
            logger.debug(f"Error marking route as crashed: {e}")

    def _record_route_crash(
        self,
        *,
        crash_reason: str,
        crash_type: str,
        crash_detail: str,
    ) -> None:
        route_id = str(self._current_route_id or "").strip()
        if not route_id or route_id == "unknown":
            return
        self._mark_route_as_crashed(
            route_id,
            crash_reason=crash_reason,
            crash_type=crash_type,
            crash_detail=crash_detail,
        )

    def _raise_episode_crash(
        self,
        *,
        crash_reason: str,
        crash_type: str,
        crash_detail: str,
    ) -> None:
        crash_reason, crash_type, crash_detail = refine_crash_payload(
            crash_reason,
            crash_type,
            crash_detail,
        )
        self._set_crash_state(
            crash_reason=crash_reason,
            crash_type=crash_type,
            crash_detail=crash_detail,
        )
        self._record_route_crash(
            crash_reason=crash_reason,
            crash_type=crash_type,
            crash_detail=crash_detail,
        )
        raise EpisodeCrashError(crash_reason, crash_type, crash_detail)

    @staticmethod
    def _normalize_context_value(value: Any, default: str = "unknown") -> str:
        if value is None:
            return default
        text = str(value).strip()
        return text if text else default

    def _resolve_episode_context(
        self,
        route: Optional[Any] = None,
        scenario: Optional[Any] = None,
    ) -> Dict[str, str]:
        if route is None:
            route_id = self._current_route_id
            town = self._current_town
            scenario_name = self.scenario_name
            scenario_instance_name = self.scenario_instance_name
        else:
            route_id = "unknown"
            town = getattr(route, "town", "unknown")
            scenario_name = "unknown"
            scenario_instance_name = ""
            route_id = (
                f"{route.name}_rep{route.repetition_index}"
                if hasattr(route, "name") and hasattr(route, "repetition_index")
                else route_id
            )

            scenario_configs = list(getattr(route, "scenario_configs", []) or [])
            if len(scenario_configs) == 1:
                scenario_cfg = scenario_configs[0]
                scenario_name = getattr(scenario_cfg, "type", scenario_name)
                scenario_instance_name = getattr(scenario_cfg, "name", scenario_instance_name)
            elif len(scenario_configs) > 1:
                scenario_name = "MultiScenario"
                scenario_instance_name = ""
            elif len(scenario_configs) == 0:
                scenario_name = "NoScenario"
                scenario_instance_name = ""

        if scenario is not None:
            runtime_scenario_name = getattr(scenario, "scenario_name", None)
            if runtime_scenario_name and runtime_scenario_name != "NoScenario":
                scenario_name = runtime_scenario_name

        scenario_name = self._normalize_context_value(scenario_name, default="unknown")
        scenario_instance_name = self._normalize_context_value(scenario_instance_name, default="")
        town = self._normalize_context_value(town, default="unknown")
        route_id = self._normalize_context_value(route_id, default="unknown")

        return {
            "route_id": route_id,
            "scenario_name": scenario_name,
            "scenario_instance_name": scenario_instance_name,
            "town": town,
        }

    def _set_episode_context(
        self,
        route: Optional[Any] = None,
        scenario: Optional[Any] = None,
    ) -> Dict[str, str]:
        context = self._resolve_episode_context(route=route, scenario=scenario)
        self._current_route_id = context["route_id"]
        self.scenario_name = context["scenario_name"]
        self.scenario_instance_name = context["scenario_instance_name"]
        self._current_town = context["town"]
        return context

    def get_episode_context(self) -> Dict[str, str]:
        return {
            "route_id": self._current_route_id,
            "scenario_name": self.scenario_name,
            "scenario_instance_name": self.scenario_instance_name,
            "town": self._current_town,
        }

    def _episode_context_for_log(self) -> str:
        context = self.get_episode_context()
        scenario_display = context["scenario_name"] or context["scenario_instance_name"]
        if (
            context["scenario_name"]
            and context["scenario_instance_name"]
            and context["scenario_instance_name"] != context["scenario_name"]
        ):
            scenario_display = f"{scenario_display} ({context['scenario_instance_name']})"
        return (
            f"route={context['route_id']} | "
            f"scenario={scenario_display} | "
            f"town={context['town']}"
        )

    def _with_episode_context(self, info: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        info_dict = dict(info or {})
        info_dict.update(self.get_episode_context())
        info_dict.setdefault("crash_message", self.crash_message)
        info_dict.update(kwargs)
        return info_dict

    def _set_crash_state(
        self,
        *,
        crash_reason: str,
        crash_type: str,
        crash_detail: str,
    ) -> None:
        self.crash_reason = str(crash_reason or "").strip()
        self.crash_type = str(crash_type or "").strip()
        self.crash_detail = str(crash_detail or crash_reason or "").strip()
        self.crash_message = self.crash_detail or self.crash_reason

    def _runtime_crash_result(
        self,
        *,
        crash_type: str,
        crash_detail: str,
        crash_reason: str = "simulation_crashed",
    ):
        crash_reason, crash_type, crash_detail = refine_crash_payload(
            crash_reason,
            crash_type,
            crash_detail,
        )
        self._set_crash_state(
            crash_reason=crash_reason,
            crash_type=crash_type,
            crash_detail=crash_detail,
        )
        return {}, 0.0, True, False, self._with_episode_context(
            error=self.crash_detail,
            crashed=True,
            crash_reason=self.crash_reason,
            crash_type=self.crash_type,
            crash_detail=self.crash_detail,
        )

    def _sample_route(self):
        """Sample the next route according to ``self._sample_mode``.

        Supported modes are ``sequential``, ``random`` (shuffle-then-repeat)
        and ``adaptive`` (scenario-weighted sampling).

        Returns:
            ``RouteConfig`` for the next route, or ``None`` if the sampler
            is empty.
        """
        if self._sample_mode == 'adaptive':
            return self._sample_adaptive_route()

        if self._sample_mode == 'random':
            if self._total_routes <= 0:
                return None

            # Reshuffle at the start of every new cycle.
            if self._shuffle_pos >= len(self._shuffle_indices):
                self._shuffle_cycle += 1
                self._shuffle_indices = list(range(self._total_routes))
                random.shuffle(self._shuffle_indices)
                self._shuffle_pos = 0
                logger.debug(f"Shuffle cycle {self._shuffle_cycle}: reshuffled {self._total_routes} routes")

            route_index = self._shuffle_indices[self._shuffle_pos]
            self._shuffle_pos += 1
            self._random_sampled_count += 1

            route = self._get_route_config_by_filtered_index(route_index)
            self._last_sampled_route = route
            self._last_sampled_route_filtered_index = route_index

            logger.debug(f"Shuffle sampled route index: {route_index} "
                        f"(pos {self._shuffle_pos}/{self._total_routes}, "
                        f"cycle {self._shuffle_cycle})")
            return route

        # Sequential mode.
        if not self.route_sampler.peek():
            logger.debug("All routes completed, resetting route sampler")
            self.route_sampler.reset()

        route = self.route_sampler.get_next_config()
        self._last_sampled_route = route
        self._last_sampled_route_filtered_index = None
        return route

    def get_route_progress(self) -> dict:
        """Return progress information about the route sampler.

        The dict always contains ``total_routes``, ``sample_mode`` and
        ``completed_routes``. ``current_index`` is mode-dependent:

        - ``sequential``: index of the next route to sample.
        - ``random``: position within the current shuffle cycle; extra
          fields ``shuffle_cycle`` and ``random_sampled_count`` are also
          provided.
        - ``adaptive``: cumulative number of adaptive samples drawn.
        """
        progress = {
            'total_routes': self._total_routes,
            'sample_mode': self._sample_mode,
            'completed_routes': len(self.statistics_manager.get_completed_route_ids()) if self.statistics_manager else 0,
        }
        
        if self._sample_mode == 'adaptive':
            progress['current_index'] = self._adaptive_sampled_count
            progress['adaptive_bucket_count'] = len(self._adaptive_route_buckets)
        elif self._sample_mode == 'random':
            progress['current_index'] = self._shuffle_pos
            progress['shuffle_cycle'] = self._shuffle_cycle
            progress['random_sampled_count'] = self._random_sampled_count
        else:
            progress['current_index'] = getattr(self.route_sampler, 'index', 0)
        
        return progress
    
    def _connect(
        self,
        connect_retries: int = 1,
        connect_retry_interval: float = 2.0
    ) -> bool:
        """Connect to the CARLA server.

        Returns:
            True on success.
        """
        if self._connected and self._connection is not None:
            if self._connection.is_connected():
                return True

        carla_config = self.config.get('carla', {})

        # Pick host from the ``carla`` section (supports a list keyed by env_index).
        host = 'localhost'
        if 'host' in carla_config:
            carla_host = carla_config['host']
            if isinstance(carla_host, list):
                host = carla_host[self.env_index] if self.env_index < len(carla_host) else carla_host[0]
            else:
                host = carla_host

        self._host = host

        connection_timeout = carla_config.get('connection_timeout', 10.0)
        server_wait_timeout = carla_config.get('server_wait_timeout', 120.0)
        max_retries = carla_config.get('max_retries', 60)

        self._connection = CARLAConnection(
            host=host,
            port=self._port,
            traffic_manager_port=self._tm_port,
            traffic_manager_seed=self._tm_seed,
            connection_timeout=connection_timeout,
            server_wait_timeout=server_wait_timeout,
            # retry_interval throttles wait-for-server polling.
            retry_interval=2.0,
            max_retries=max_retries
        )
        
        try:
            self._connection.connect(
                wait_for_server=self._wait_for_server,
                connect_retries=connect_retries,
                connect_retry_interval=connect_retry_interval
            )
            self.client = self._connection.client
            self._connected = True
            logger.debug(f"CARLAEnv[{self.env_index}] connected to server")
            return True
        except CARLAConnectionError as e:
            logger.error(f"CARLAEnv[{self.env_index}] connection failed: {e}")
            self._connected = False
            raise
    
    def reconnect(
        self,
        wait_for_server: bool = True,
        cleanup_first: bool = False,
        connect_retries: Optional[int] = None,
        connect_retry_interval: Optional[float] = None
    ) -> bool:
        """Reconnect to the CARLA server after a crash.

        Args:
            wait_for_server: Wait for the server to be ready (recommended).
            cleanup_first: Run crash cleanup before reconnecting.
            connect_retries: Number of connect attempts (default from config).
            connect_retry_interval: Seconds between attempts (default from config).

        Returns:
            True on success.

        Example::

            env.cleanup_on_crash(retry_current_route=True)
            env.reconnect(wait_for_server=True)
            obs, info = env.reset()
        """
        logger.debug(f"CARLAEnv[{self.env_index}] reconnecting...")

        if cleanup_first:
            self.cleanup_on_crash(retry_current_route=False)
        else:
            # Drop only the connection-related resources.
            if self._connection is not None:
                try:
                    self._connection.close()
                except Exception:
                    pass

            self._connected = False
            self.client = None
            self.world = None
            self.traffic_manager = None

        self.crash_message = ""

        try:
            carla_config = self.config.get('carla', {})
            if connect_retries is None:
                connect_retries = carla_config.get('connect_retries', 3)
            if connect_retry_interval is None:
                connect_retry_interval = carla_config.get('connect_retry_interval', 2.0)

            self._wait_for_server = wait_for_server
            self._connect(
                connect_retries=connect_retries,
                connect_retry_interval=connect_retry_interval
            )
            logger.debug(f"CARLAEnv[{self.env_index}] reconnected successfully")
            return True
        except CARLAConnectionError as e:
            logger.error(f"CARLAEnv[{self.env_index}] reconnect failed: {e}")
            return False
        except Exception as e:
            logger.error(f"CARLAEnv[{self.env_index}] reconnect failed (unexpected): {e}")
            return False

    def reward_space(self):
        return self.config.get('reward_space', None)

    def load_world(self, town: str, retry_times: Optional[int] = None) -> bool:
        """Load a CARLA map.

        By default a full load is forced on every reset. When
        ``FORCE_FULL_LOAD_EVERY_RESET`` is false and the town is unchanged,
        ``client.load_world`` is skipped in favour of a lighter state reset.

        Args:
            town: Map name to load.
            retry_times: Override for retry count (defaults to config).

        Returns:
            True on success.
        """
        env_config = self.config.get('environment', {})
        carla_config = self.config.get('carla', {})
        
        if retry_times is None:
            retry_times = env_config.get('retry_times', 1)
        
        load_world_timeout = carla_config.get('load_world_timeout', 60.0)
        load_world_server_check_interval = float(
            carla_config.get('load_world_server_check_interval', 20.0) or 0.0
        )
        load_world_server_check_jitter = float(
            carla_config.get('load_world_server_check_jitter', 5.0) or 0.0
        )
        
        if not self._connected:
            try:
                self._connect()
            except CARLAConnectionError as e:
                logger.error(
                    f"Cannot load world - connection failed: {e} | "
                    f"{self._episode_context_for_log()}"
                )
                self.crash_message = "Simulation crashed - connection failed"
                return False
        
        # Decide whether a town switch is needed.
        current_map = None
        if self.world is not None:
            try:
                current_map = self.world.get_map().name.split('/')[-1]
            except Exception:
                current_map = None

        need_full_load = (
            FORCE_FULL_LOAD_EVERY_RESET
            or current_map is None
            or current_map != town
        )

        logger.debug(f"CARLAEnv[{self.env_index}] load_world: cleaning up CarlaDataProvider...")
        try:
            CarlaDataProvider.cleanup()
            logger.debug(f"CARLAEnv[{self.env_index}] load_world: CarlaDataProvider cleanup done")
        except Exception as e:
            logger.debug(f"CARLAEnv[{self.env_index}] load_world: CarlaDataProvider cleanup failed: {e}")
        
        if need_full_load:
            last_error = None
            if (
                FORCE_FULL_LOAD_EVERY_RESET
                and current_map is not None
                and current_map == town
            ):
                logger.debug(
                    f"CARLAEnv[{self.env_index}] load_world: force_full_load enabled, "
                    f"reloading same town {town}"
                )
            for attempt in range(retry_times):
                try:
                    logger.debug(f"CARLAEnv[{self.env_index}] load_world: FULL load {town}, attempt {attempt+1}/{retry_times}")
                    self.world = self._connection.load_world(
                        town,
                        timeout=load_world_timeout,
                        reset_settings=False,
                        server_check_interval=load_world_server_check_interval,
                        server_check_jitter=load_world_server_check_jitter,
                    )
                    break
                except Exception as e:
                    last_error = e
                    logger.debug(f"CARLAEnv[{self.env_index}] load_world: attempt {attempt + 1} FAILED: {e}")
                    if attempt < retry_times - 1:
                        time.sleep(2)
            else:
                logger.error(
                    f"CARLAEnv[{self.env_index}] load_world: FAILED after {retry_times} attempts: "
                    f"{last_error} | {self._episode_context_for_log()}"
                )
                self.crash_message = "Simulation crashed"
                return False
            logger.info(f"CARLAEnv[{self.env_index}] load_world: loaded new town {town}")
        else:
            logger.debug(
                f"CARLAEnv[{self.env_index}] load_world: same town {town}, "
                "skipping client.load_world (FORCE_FULL_LOAD_EVERY_RESET=False)"
            )
        
        # Apply world settings after loading or reusing the town.
        logger.debug(f"CARLAEnv[{self.env_index}] load_world: configuring world settings...")
        frequency_hz = carla_config.get('frequency_hz', 10)
        
        settings = self.world.get_settings()
        settings.fixed_delta_seconds = 1.0 / frequency_hz
        settings.no_rendering_mode = bool(carla_config.get('no_rendering_mode', True))
        settings.synchronous_mode = True
        settings.tile_stream_distance = TILE_STREAM_DISTANCE
        settings.actor_active_distance = ACTOR_ACTIVE_DISTANCE
        self.world.apply_settings(settings)
        logger.debug(
            f"CARLAEnv[{self.env_index}] load_world: world settings applied "
            f"(sync mode, {frequency_hz}Hz, no_rendering_mode={settings.no_rendering_mode})"
        )
        
        logger.debug(f"CARLAEnv[{self.env_index}] load_world: resetting traffic lights...")
        self.world.reset_all_traffic_lights()
        logger.debug(f"CARLAEnv[{self.env_index}] load_world: traffic lights reset")
        
        logger.debug(f"CARLAEnv[{self.env_index}] load_world: getting Traffic Manager on port {self._tm_port}...")
        if self.traffic_manager is None:
            self.traffic_manager = self._connection.get_traffic_manager()
        logger.debug(f"CARLAEnv[{self.env_index}] load_world: Traffic Manager obtained")

        logger.debug(f"CARLAEnv[{self.env_index}] load_world: setting up CarlaDataProvider...")
        CarlaDataProvider.set_client(self.client)
        CarlaDataProvider.set_world(self.world)
        CarlaDataProvider.set_traffic_manager_port(self._tm_port)
        logger.debug(f"CARLAEnv[{self.env_index}] load_world: CarlaDataProvider configured")

        logger.debug(f"CARLAEnv[{self.env_index}] load_world: configuring Traffic Manager...")
        self.traffic_manager.set_synchronous_mode(True)
        self.traffic_manager.set_hybrid_physics_mode(True)
        self.traffic_manager.set_random_device_seed(self._tm_seed)
        logger.debug(f"CARLAEnv[{self.env_index}] load_world: Traffic Manager configured")

        logger.debug(f"CARLAEnv[{self.env_index}] load_world: initial world tick...")
        self.world.tick(SIMULATOR_TICK_TIMEOUT_LIMIT)
        logger.debug(
            f"CARLAEnv[{self.env_index}] load_world: COMPLETED "
            f"(full_load={need_full_load}, force_full_load={FORCE_FULL_LOAD_EVERY_RESET})"
        )
        return True

    def _cleanup_after_route(self) -> None:
        """End-of-route cleanup.

        Stops the scenario, removes all actors and resets the
        ``ScenarioManager`` state without touching the sync/async mode or
        the underlying connection.
        """
        logger.debug(f"CARLAEnv[{self.env_index}] _cleanup_after_route: starting...")
        
        if self.scenario_manager:
            try:
                record = self.scenario_manager.stop_scenario()
                self._update_adaptive_sampling_state(record)
                logger.debug(f"CARLAEnv[{self.env_index}] _cleanup_after_route: scenario stopped")
            except Exception as e:
                logger.debug(f"CARLAEnv[{self.env_index}] stop_scenario failed: {e}")

        if self.scenario:
            try:
                self.scenario.remove_all_actors()
                logger.debug(f"CARLAEnv[{self.env_index}] _cleanup_after_route: actors removed")
            except Exception as e:
                logger.debug(f"CARLAEnv[{self.env_index}] remove_all_actors failed: {e}")
            self.scenario = None

        # Use cleanup_route() rather than cleanup() to keep cross-route state.
        try:
            CarlaDataProvider.cleanup_route()
            logger.debug(f"CARLAEnv[{self.env_index}] _cleanup_after_route: CarlaDataProvider.cleanup_route done")
        except Exception as e:
            logger.debug(f"CARLAEnv[{self.env_index}] cleanup_route failed: {e}")

        logger.debug(f"CARLAEnv[{self.env_index}] _cleanup_after_route: done")

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """
        Start a new episode.

        Args:
            seed: Random seed for reproducible episodes
            options: Additional configuration 

        Returns:
            tuple: (observation, info) for the initial state
        """
        _reset_t0 = time.perf_counter()
        
        # IMPORTANT: must be called first so the RNG is seeded properly.
        super().reset(seed=seed)

        self.count = 0
        self.crash_message = ""
        self.crash_reason = ""
        self.crash_type = ""
        self.crash_detail = ""

        # End-of-route cleanup (drops actors from the previous route).
        if self.scenario is not None:
            logger.debug(f"CARLAEnv[{self.env_index}] reset: cleaning up previous route...")
            self._cleanup_after_route()

        # Make sure the connection is alive (do not force a reconnect).
        connection_ok = False
        if self._connection is not None:
            try:
                connection_ok = self._connection.is_connected()
            except Exception:
                connection_ok = False
        if not self._connected or not connection_ok:
            logger.debug(f"CARLAEnv[{self.env_index}] reset: reconnecting to server...")
            if not self.reconnect(wait_for_server=True, cleanup_first=False):
                raise RuntimeError(
                    f"Failed to establish connection to CARLA server at {self._host}:{self._port}. "
                    "Please ensure the server is running."
                )
        else:
            logger.debug(f"CARLAEnv[{self.env_index}] reset: connection already established")

        # Sample the next route according to the sampling mode.
        logger.debug(f"CARLAEnv[{self.env_index}] reset: [1/8] sampling route...")
        route = self._sample_route()
        if not route:
            raise RuntimeError("No route available")
        self._set_episode_context(route=route)
        logger.debug(
            f"CARLAEnv[{self.env_index}] reset: [1/8] route sampled | "
            f"{self._episode_context_for_log()}"
        )
        
        # Load the map (load_world records its own timing).
        logger.debug(f"CARLAEnv[{self.env_index}] reset: [2/8] loading world {route.town}...")
        _t = time.perf_counter()
        try:
            if not self.load_world(route.town):
                raise EpisodeCrashError(
                    "initial_reset_failed",
                    "load_world_failed",
                    f"Failed to load world: {route.town}",
                )
        except Exception as exc:
            if isinstance(exc, EpisodeCrashError):
                self._raise_episode_crash(
                    crash_reason=exc.crash_reason,
                    crash_type=exc.crash_type,
                    crash_detail=exc.crash_detail,
                )
            self._raise_episode_crash(
                crash_reason="initial_reset_failed",
                crash_type="load_world_failed",
                crash_detail=str(exc),
            )
        self.timing_stats.record('load_world', time.perf_counter() - _t)
        logger.debug(f"CARLAEnv[{self.env_index}] reset: [2/8] world loaded successfully")
        
        # CarlaDataProvider was already set up inside load_world().
        logger.debug(f"CARLAEnv[{self.env_index}] reset: [3/8] CarlaDataProvider ready (set in load_world)")
        
        # create a scenario
        logger.debug(f"CARLAEnv[{self.env_index}] reset: [5/8] creating scenario...")
        try:
            scenario = RouteScenario(route, config=self.config)
        except RouteScenarioSetupError as exc:
            self._raise_episode_crash(
                crash_reason="skip_setup_error",
                crash_type="route_scenario_setup",
                crash_detail=str(exc),
            )
        except Exception as exc:
            self._raise_episode_crash(
                crash_reason="initial_reset_failed",
                crash_type="scenario_creation_failed",
                crash_detail=str(exc),
            )
        if scenario:
            self._set_episode_context(route=route, scenario=scenario)
            logger.debug(
                f"CARLAEnv[{self.env_index}] reset: [5/8] Scenario {scenario.name} created | "
                f"{self._episode_context_for_log()}"
            )
        self.scenario = scenario
        
        # Tick the world once so the ego is properly synced to the server.
        logger.debug(f"CARLAEnv[{self.env_index}] reset: [6/8] first world tick...")
        _t = time.perf_counter()
        try:
            CarlaDataProvider.get_world().tick(SIMULATOR_TICK_TIMEOUT_LIMIT)
        except Exception as exc:
            self._raise_episode_crash(
                crash_reason="initial_reset_failed",
                crash_type="reset_world_tick_failed",
                crash_detail=str(exc),
            )
        self.timing_stats.record('world_tick', time.perf_counter() - _t)
        logger.debug(f"CARLAEnv[{self.env_index}] reset: [6/8] first world tick completed")
        
        # Hand the scenario to the manager (also notifies the statistics manager).
        logger.debug(f"CARLAEnv[{self.env_index}] reset: [7/8] loading scenario to manager...")
        try:
            self.scenario_manager.load_scenario(scenario, route)
        except Exception as exc:
            self._raise_episode_crash(
                crash_reason="initial_reset_failed",
                crash_type="scenario_load_failed",
                crash_detail=str(exc),
            )
        logger.debug(f"CARLAEnv[{self.env_index}] reset: [7/8] scenario loaded to manager")

        self.scenario_manager_init_status()

        logger.debug(f"CARLAEnv[{self.env_index}] reset: [8/8] second world tick and scenario reset...")
        _t = time.perf_counter()
        try:
            CarlaDataProvider.get_world().tick(SIMULATOR_TICK_TIMEOUT_LIMIT)
        except Exception as exc:
            self._raise_episode_crash(
                crash_reason="initial_reset_failed",
                crash_type="reset_world_tick_failed",
                crash_detail=str(exc),
            )
        self.timing_stats.record('world_tick', time.perf_counter() - _t)
        
        # initial tick
        try:
            self.scenario_manager_reset()
        except Exception as exc:
            if isinstance(exc, RouteScenarioSetupError):
                self._raise_episode_crash(
                    crash_reason="skip_setup_error",
                    crash_type="route_scenario_setup",
                    crash_detail=str(exc),
                )
            self._raise_episode_crash(
                crash_reason="initial_reset_failed",
                crash_type="scenario_manager_reset_failed",
                crash_detail=str(exc),
            )
        logger.debug(f"CARLAEnv[{self.env_index}] reset: [8/8] scenario manager reset completed")
        
        observation = {}  # process in the ObservationWrapper
        info = self.update_info()
        
        # Pause the watchdog while we wait for an external action; it is
        # resumed at the start of step() via unlock_scenario_manager().
        self.lock_scenario_manager()

        self.timing_stats.record('reset', time.perf_counter() - _reset_t0)
        
        logger.debug(f"CARLAEnv[{self.env_index}] reset: COMPLETED "
                    f"(total={time.perf_counter() - _reset_t0:.2f}s)")
        return observation, info
        

    def step(self, action):
        """Run a single simulation step.

        Args:
            action: Vehicle control to apply this tick.

        Returns:
            ``(observation, reward, terminated, truncated, info)``.

        Notes:
            - ``terminated`` is true when the task finishes or a custom
              termination event fires (collision, off-road, ...).
            - ``truncated`` is true when the max-step limit is reached.
            - ``info['crashed']`` flags simulator crashes so the buffer can
              discard the episode.

        Crash types (``info['crash_type']``):
            - ``watchdog_timeout``: simulator unresponsive.
            - ``world_tick_failed``: ``world.tick()`` raised.
            - ``ego_not_alive``: ego vehicle was destroyed.
            - ``apply_control_failed``: ``apply_control`` raised.
            - ``scenario_tick_spawn_collision``: dynamic spawn collided
              during a scenario tick.
            - ``actor_handle_invalid``: actor handle no longer valid.
            - ``ego_actor_check_failed``: ``is_alive`` check itself raised.
        """
        # 1. Watchdog status check.
        watchdog_ok = self.scenario_manager._watchdog is None or self.scenario_manager._watchdog.get_status()
        if not watchdog_ok or not self.scenario_manager._running:
            error_msg = "Watchdog timeout - simulation not responding"
            logger.error(
                f"CARLAEnv[{self.env_index}] {error_msg} | "
                f"{self._episode_context_for_log()}"
            )
            return self._runtime_crash_result(
                crash_type='watchdog_timeout',
                crash_detail=error_msg,
            )
        
        # 2. Ego-vehicle liveness check.
        ego_actor = CarlaDataProvider._ego_actor
        if ego_actor is None:
            error_msg = "Ego actor is None"
            logger.error(
                f"CARLAEnv[{self.env_index}] {error_msg} | "
                f"{self._episode_context_for_log()}"
            )
            return self._runtime_crash_result(
                crash_type='ego_not_alive',
                crash_detail=error_msg,
            )
        
        try:
            _is_alive = ego_actor.is_alive
            if not _is_alive:
                error_msg = "Ego actor is not alive (destroyed externally)"
                logger.error(
                    f"CARLAEnv[{self.env_index}] {error_msg} | "
                    f"{self._episode_context_for_log()}"
                )
                return self._runtime_crash_result(
                    crash_type='ego_not_alive',
                    crash_detail=error_msg,
                )
        except Exception as e:
            # is_alive check itself raised (likely connection dropped).
            error_msg = f"Failed to check ego alive status: {e}"
            logger.error(
                f"CARLAEnv[{self.env_index}] {error_msg} | "
                f"{self._episode_context_for_log()}"
            )
            return self._runtime_crash_result(
                crash_type='actor_destroyed',
                crash_detail=error_msg,
            )
        
        _step_t0 = time.perf_counter()
        
        # 3. Apply control command.
        try:
            self.unlock_scenario_manager(action)
            _t = time.perf_counter()
            ego_actor.apply_control(action)
            self.timing_stats.record('apply_control', time.perf_counter() - _t)
            py_trees.blackboard.Blackboard().set("AV_control", action, overwrite=True)
        except Exception as e:
            error_msg = f"apply_control failed: {e}"
            logger.error(
                f"CARLAEnv[{self.env_index}] {error_msg} | "
                f"{self._episode_context_for_log()}"
            )
            return self._runtime_crash_result(
                crash_type='apply_control_failed',
                crash_detail=error_msg,
            )
        
        # 4. World tick.
        try:
            _t = time.perf_counter()
            CarlaDataProvider.get_world().tick(SIMULATOR_TICK_TIMEOUT_LIMIT)
            self.timing_stats.record('world_tick', time.perf_counter() - _t)
        except Exception as e:
            error_msg = f"World tick failed: {e}"
            logger.error(
                f"CARLAEnv[{self.env_index}] {error_msg} | "
                f"{self._episode_context_for_log()}"
            )
            return self._runtime_crash_result(
                crash_type='world_tick_failed',
                crash_detail=error_msg,
            )
        
        # 5. Scenario-manager tick (records its own sub-timings).
        try:
            self.scenario_manager_step_tick()
        except Exception as e:
            error_msg = f"Scenario manager tick failed: {e}"
            _e_str = str(e)
            if "Spawn failed" in _e_str or "collision at spawn" in _e_str:
                logger.warning(
                    f"CARLAEnv[{self.env_index}] {error_msg} | "
                    f"{self._episode_context_for_log()}"
                )
            else:
                logger.error(
                    f"CARLAEnv[{self.env_index}] {error_msg} | "
                    f"{self._episode_context_for_log()}"
                )
            return self._runtime_crash_result(
                crash_type='actor_destroyed',
                crash_detail=error_msg,
            )
        
        # 6. Compute termination flags.
        time_limit = self.config.get('environment', {}).get('time_limit', 10000)

        # ``terminated`` and ``truncated`` are mutually exclusive:
        #   - terminated: task finished or a termination event fired.
        #   - truncated:  step/time limit reached (safety fallback; normally
        #                 the EnvPool's max_episode_steps fires first).
        # ``terminated`` takes precedence to keep RL bootstrap unambiguous.
        task_finished = not self.scenario_manager._running
        time_limit_reached = self.count >= (time_limit - 1)
        if task_finished:
            terminated = True
            truncated = False
        elif time_limit_reached:
            terminated = False
            truncated = True
        else:
            terminated = False
            truncated = False
        
        # 7. Read vehicle state for debug logging (timed CARLA API calls).
        speed, steer, throttle, brake = 0.0, 0.0, 0.0, 0.0
        try:
            if ego_actor is not None and ego_actor.is_alive:
                _t = time.perf_counter()
                velocity = ego_actor.get_velocity()
                self.timing_stats.record('get_velocity', time.perf_counter() - _t)
                speed = (velocity.x**2 + velocity.y**2 + velocity.z**2)**0.5  # m/s
                
                _t = time.perf_counter()
                control = ego_actor.get_control()
                self.timing_stats.record('get_control', time.perf_counter() - _t)
                steer = control.steer
                throttle = control.throttle
                brake = control.brake
        except Exception:
            # Failure to read vehicle state must not break the main loop.
            pass

        # Log every 100 steps and on the last step.
        if self.count % 100 == 0 or terminated or truncated:
            logger.debug(f"Env[{self.env_index}] step={self.count} | "
                        f"spd={speed:.1f}m/s str={steer:+.2f} thr={throttle:.2f} brk={brake:.2f}")
        
        observation = {}  # process in the ObservationWrapper
        reward = 0.0  # process in the RewardWrapper
        info = self.update_info()
        if self.statistics_manager:
            try:
                self.statistics_manager.update_step()
            except Exception as e:
                logger.debug(f"Failed to update route step statistics: {e}")
        
        self.lock_scenario_manager()

        self.timing_stats.record_step_total(time.perf_counter() - _step_t0)

        # Log a CARLA timing summary every 200 steps.
        if self.count > 0 and self.count % 200 == 0:
            self.timing_stats.log_summary(self.env_index, self.count)
        
        return observation, reward, terminated, truncated, info
    
    def scenario_manager_init_status(self):
        """Initialise scenario-manager state for a new episode."""
        self.scenario_manager.start_system_time = time.time()
        self.scenario_manager.start_game_time = GameTime.get_time()

        carla_config = self.config.get('carla', {})
        timeout = carla_config.get('timeout', 60)

        # Optionally disable the watchdog (recommended in debug runs).
        disable_watchdog = carla_config.get('disable_watchdog', False)

        # Stop any pre-existing watchdog.
        if hasattr(self.scenario_manager, '_watchdog') and self.scenario_manager._watchdog:
            try:
                self.scenario_manager._watchdog.stop()
            except Exception:
                pass
        
        if hasattr(self.scenario_manager, '_agent_watchdog') and self.scenario_manager._agent_watchdog:
            try:
                self.scenario_manager._agent_watchdog.stop()
            except Exception:
                pass
        
        if disable_watchdog:
            logger.debug(f"CARLAEnv[{self.env_index}] Watchdog DISABLED (debug mode)")
            self.scenario_manager._watchdog = None
            self.scenario_manager._agent_watchdog = None
        else:
            self.scenario_manager._watchdog = Watchdog(timeout)
            self.scenario_manager._watchdog.start()

            self.scenario_manager._agent_watchdog = Watchdog(timeout)
            self.scenario_manager._agent_watchdog.start()

            # Pause immediately so the reset path (e.g. observation handler
            # breakpoints) does not trip the watchdog. step() resumes it
            # through unlock_scenario_manager().
            self.scenario_manager._watchdog.pause()
            self.scenario_manager._agent_watchdog.pause()
            logger.debug(f"CARLAEnv[{self.env_index}] Watchdog paused after init")

        self.scenario_manager._running = True
    
    # include a initial 'tick' of scenario manager
    def scenario_manager_reset(self):
        timestamp = CarlaDataProvider.get_world().get_snapshot().timestamp

        self.scenario_manager._timestamp_last_run = timestamp.elapsed_seconds
        if self.scenario_manager._watchdog:
            self.scenario_manager._watchdog.update()
        # Update game time and actor information
        GameTime.on_carla_tick(timestamp)
        CarlaDataProvider.on_carla_tick()
        
        # tick scenario_tree
        self.scenario_manager.scenario_tree.tick_once()


    def unlock_scenario_manager(self, action):
        if self.scenario_manager._agent_watchdog:
            self.scenario_manager._agent_watchdog.update()
            self.scenario_manager._agent_watchdog.pause()
        if self.scenario_manager._watchdog:
            self.scenario_manager._watchdog.resume()

    def scenario_manager_step_tick(self):
        _t = time.perf_counter()
        timestamp = CarlaDataProvider.get_world().get_snapshot().timestamp
        self.timing_stats.record('get_snapshot', time.perf_counter() - _t)
        
        # Guard against invalid snapshot during server warm-up/reconnect
        if getattr(timestamp, "delta_seconds", None) is None:
            raise RuntimeError("Invalid world snapshot: delta_seconds is None (server not ready)")
        if self.scenario_manager._timestamp_last_run < timestamp.elapsed_seconds and self.scenario_manager._running:
            self.scenario_manager._timestamp_last_run = timestamp.elapsed_seconds
            if self.scenario_manager._watchdog:
                self.scenario_manager._watchdog.update()
            # Update game time and actor information
            _t = time.perf_counter()
            GameTime.on_carla_tick(timestamp)   
            CarlaDataProvider.on_carla_tick()
            self.timing_stats.record('on_carla_tick', time.perf_counter() - _t)

        # tick scenario_tree
        _t = time.perf_counter()
        self.scenario_manager.scenario_tree.tick_once()
        self.timing_stats.record('scenario_tree_tick', time.perf_counter() - _t)
        
        if self.scenario_manager.scenario_tree.status != py_trees.common.Status.RUNNING:
            self.scenario_manager._running = False

    def lock_scenario_manager(self):
        if self.scenario_manager._watchdog:
            self.scenario_manager._watchdog.pause()
        if self.scenario_manager._agent_watchdog:
            self.scenario_manager._agent_watchdog.resume()

    def _init_info(self):
        if self.config.get("info", None):
            for k,v in self.config["info"].items():
                setattr(self, k, v)
        else:
            self.running_info = {}
        return
    
    def cleanup_on_crash(self, retry_current_route: bool = True):
        """Release CARLA-connection resources after a server crash.

        ``route_sampler`` and ``statistics_manager`` state are preserved so
        the schedule and aggregate metrics survive the crash.

        After this call the typical recovery flow is:
            1. Restart the CARLA server.
            2. Call ``reconnect()``.
            3. Call ``reset()`` to continue the schedule (or retry the
               current route).

        Args:
            retry_current_route: If true, the next ``reset()`` retries the
                failed route; if false, it jumps to the next route.
        """
        logger.debug(f"CARLAEnv[{self.env_index}] cleaning up after crash...")

        if not self.crash_reason:
            self.crash_reason = "simulation_crashed"
        if not self.crash_type:
            self.crash_type = "unknown"
        if not self.crash_detail:
            self.crash_detail = self.crash_message or "Simulation crashed"
        self.crash_message = self.crash_detail or "Simulation crashed"
        
        if self.scenario_manager:
            self.scenario_manager._running = False
            if hasattr(self.scenario_manager, '_watchdog') and self.scenario_manager._watchdog:
                try:
                    self.scenario_manager._watchdog.stop()
                except Exception:
                    pass
            if hasattr(self.scenario_manager, '_agent_watchdog') and self.scenario_manager._agent_watchdog:
                try:
                    self.scenario_manager._agent_watchdog.stop()
                except Exception:
                    pass
        
        # Drop the scenario without calling scenario_manager.cleanup() /
        # scenario.terminate() - after a crash the CARLA connection, actor
        # and sensor handles may already be invalid, and forcing terminate
        # could re-raise or hang the cleanup chain.
        self.scenario = None

        # Record the crash through the RLStatisticsManager. We deliberately
        # skip terminate-only finalisation here, so events that are only
        # emitted by terminate() (min-speed, scenario-timeout, yield, ...)
        # may be missing from crash-route statistics. Normal endings always
        # run terminate() first.
        if hasattr(self, 'statistics_manager') and self.statistics_manager:
            try:
                record = self.statistics_manager.end_route(
                    crashed=True,
                    failure_message=self.crash_message,
                    crash_reason=self.crash_reason,
                    crash_type=self.crash_type,
                    crash_detail=self.crash_detail,
                )
                self._update_adaptive_sampling_state(record)
            except Exception as e:
                logger.debug(f"Error recording crash in statistics: {e}")
        
        try:
            CarlaDataProvider.cleanup()
        except Exception as e:
            logger.debug(f"Error cleaning up CarlaDataProvider: {e}")

        if self._connection is not None:
            try:
                self._connection.close()
            except Exception:
                pass
            self._connection = None
        
        self._connected = False
        self.client = None
        self.world = None
        self.traffic_manager = None
        
        # Decide how route_sampler progress should be rewound.
        if not retry_current_route:
            # Skip to the next route; route_sampler already advanced.
            self._last_sampled_route = None
            self._last_sampled_route_filtered_index = None
            self._adaptive_retry_route_index = None
            logger.debug(f"CARLAEnv[{self.env_index}] will skip to next route")
        else:
            if self._sample_mode == 'adaptive':
                if self._last_sampled_route_filtered_index is not None:
                    self._adaptive_retry_route_index = self._last_sampled_route_filtered_index
                    self._adaptive_sampled_count = max(0, self._adaptive_sampled_count - 1)
                    logger.debug(
                        f"CARLAEnv[{self.env_index}] will retry current route "
                        f"(adaptive_route_index={self._adaptive_retry_route_index})"
                    )
            elif self._sample_mode == 'random':
                # Rewind the shuffle position so the same index is drawn
                # again on the next _sample_route() call.
                if self._shuffle_pos > 0:
                    self._shuffle_pos -= 1
                    self._random_sampled_count = max(0, self._random_sampled_count - 1)
                    logger.debug(f"CARLAEnv[{self.env_index}] will retry current route "
                               f"(shuffle_pos={self._shuffle_pos}, "
                               f"route_index={self._shuffle_indices[self._shuffle_pos]})")
                else:
                    # shuffle_pos == 0 means we just started a new cycle and
                    # cannot rewind further; keep _last_sampled_route only.
                    logger.debug(f"CARLAEnv[{self.env_index}] cannot retry: "
                                  f"shuffle_pos is 0, will proceed with next route")
            else:
                # Sequential mode: rewind the RouteIndexer index.
                if hasattr(self, 'route_sampler') and self.route_sampler:
                    if hasattr(self.route_sampler, 'index') and self.route_sampler.index > 0:
                        self.route_sampler.index -= 1
                        logger.debug(f"CARLAEnv[{self.env_index}] will retry current route "
                                   f"(index={self.route_sampler.index})")

        self.count = 0
        
        logger.debug(f"CARLAEnv[{self.env_index}] crash cleanup complete")
    
    def destroy_env(self):
        """Tear down the env after a normal episode end.

        Stops the scenario, persists statistics, switches the world back to
        async mode, and clears ``CarlaDataProvider``.
        """
        if self.scenario_manager:
            self.scenario_manager._running = False

        logger.debug("Stopping the route")
        if self.crash_message:
            logger.debug(self.crash_message)

        env_config = self.config.get('environment', {})

        # Stop the scenario and persist statistics (skip on crash).
        if self.crash_message != "Simulation crashed" and self.scenario_manager:
            try:
                record = self.scenario_manager.stop_scenario()
                self._update_adaptive_sampling_state(record)
            except Exception as e:
                logger.debug(f"Error stopping scenario: {e}")

        if env_config.get('record', False) and self.client:
            try:
                self.client.stop_recorder()
            except Exception:
                pass

        # RLStatisticsManager auto-saves; force a save here just in case.
        if hasattr(self, 'statistics_manager') and self.statistics_manager:
            try:
                self.statistics_manager.save()
            except Exception as e:
                logger.debug(f"Error saving statistics: {e}")

        if self.config.get("info", None):
            for k, v in self.config["info"].items():
                setattr(self, k, v)
        else:
            self.running_info = {}

        if self.scenario_manager:
            try:
                self.scenario_manager.cleanup()
            except Exception as e:
                logger.debug(f"Error cleaning up scenario manager: {e}")

        # Switch back to async mode (skip on crash).
        if self.crash_message != "Simulation crashed":
            if hasattr(self, 'world') and self.world:
                try:
                    self.world.tick(10)
                    settings = self.world.get_settings()
                    settings.synchronous_mode = False
                    settings.fixed_delta_seconds = None
                    self.world.apply_settings(settings)
                except Exception:
                    pass
            
            if self.traffic_manager is not None:
                try:
                    self.traffic_manager.set_synchronous_mode(False)
                    self.traffic_manager.set_hybrid_physics_mode(False)
                except Exception:
                    pass

        CarlaDataProvider.cleanup()
    
    def update_info(self) -> Dict:
        """Increment step counter and return the env info dict."""
        self.count += 1
        return {
            'env_index': self.env_index,
            'step': self.count,
            'port': self._port,
            'crash_message': self.crash_message,
            'crash_reason': self.crash_reason,
            'crash_type': self.crash_type,
            'crash_detail': self.crash_detail,
            **self.get_episode_context(),
        }
    
    def close(self):
        """Close the environment and release CARLA resources."""
        logger.debug(f"CARLAEnv[{self.env_index}] closing...")

        # Crash path uses the safe cleanup to avoid touching stale handles.
        if self.crash_message:
            try:
                self.cleanup_on_crash(retry_current_route=True)
            except Exception as e:
                logger.debug(f"CARLAEnv[{self.env_index}] crash cleanup failed: {e}")
        else:
            self.destroy_env()

        if self._connection is not None:
            try:
                self._connection.close()
            except Exception:
                pass
            self._connection = None
        
        self._connected = False
        self.client = None
        self.world = None
        self.traffic_manager = None
        
        logger.debug(f"CARLAEnv[{self.env_index}] closed")
    
    def _get_running_status(self) -> bool:
        """Return whether the scenario manager is currently running."""
        if hasattr(self, 'scenario_manager') and self.scenario_manager:
            return self.scenario_manager._running
        return False
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
    
    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
                
    
