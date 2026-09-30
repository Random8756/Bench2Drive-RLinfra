#!/usr/bin/env python
"""Crash-safe per-environment statistics with resume and result merging."""

import os
import json
import time
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime

from srunner.scenariomanager.traffic_events import TrafficEventType

logger = logging.getLogger("Interface Wrapper")

__layer__ = (5, "Evaluation")


PENALTY_VALUE_DICT = {
    TrafficEventType.COLLISION_PEDESTRIAN: 0.5,
    TrafficEventType.COLLISION_VEHICLE: 0.6,
    TrafficEventType.COLLISION_STATIC: 0.65,
    TrafficEventType.TRAFFIC_LIGHT_INFRACTION: 0.7,
    TrafficEventType.STOP_INFRACTION: 0.8,
    TrafficEventType.SCENARIO_TIMEOUT: 0.7,
    TrafficEventType.YIELD_TO_EMERGENCY_VEHICLE: 0.7
}

PENALTY_PERC_DICT = {
    TrafficEventType.OUTSIDE_ROUTE_LANES_INFRACTION: [0, 'increases'],
    TrafficEventType.MIN_SPEED_INFRACTION: [0.7, 'decreases'],
}

PENALTY_NAME_DICT = {
    TrafficEventType.COLLISION_STATIC: 'collisions_layout',
    TrafficEventType.COLLISION_PEDESTRIAN: 'collisions_pedestrian',
    TrafficEventType.COLLISION_VEHICLE: 'collisions_vehicle',
    TrafficEventType.TRAFFIC_LIGHT_INFRACTION: 'red_light',
    TrafficEventType.STOP_INFRACTION: 'stop_infraction',
    TrafficEventType.OUTSIDE_ROUTE_LANES_INFRACTION: 'outside_route_lanes',
    TrafficEventType.MIN_SPEED_INFRACTION: 'min_speed_infractions',
    TrafficEventType.YIELD_TO_EMERGENCY_VEHICLE: 'yield_emergency_vehicle_infractions',
    TrafficEventType.SCENARIO_TIMEOUT: 'scenario_timeouts',
    TrafficEventType.ROUTE_DEVIATION: 'route_dev',
    TrafficEventType.VEHICLE_BLOCKED: 'vehicle_blocked',
}


@dataclass
class RouteRecord:
    route_id: str = ""
    env_id: int = 0
    status: str = "Started"  # Started, Running, Completed, Failed, Crashed
    
    score_route: float = 0.0
    score_penalty: float = 1.0
    score_composed: float = 0.0
    
    num_infractions: int = 0
    infractions: Dict[str, List[str]] = field(default_factory=dict)
    
    route_length: float = 0.0
    duration_game: float = 0.0
    duration_system: float = 0.0
    total_steps: int = 0
    total_reward: float = 0.0
    
    start_time: str = ""
    end_time: str = ""
    failure_message: str = ""
    crash_reason: str = ""
    crash_type: str = ""
    crash_detail: str = ""
    
    def __post_init__(self):
        if not self.infractions:
            self.infractions = {name: [] for name in PENALTY_NAME_DICT.values()}
            self.infractions['route_timeout'] = []
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'RouteRecord':
        return cls(**data)


@dataclass  
class EnvStatistics:
    env_id: int = 0
    total_routes: int = 0
    completed_routes: int = 0
    failed_routes: int = 0
    crashed_routes: int = 0
    
    total_score: float = 0.0
    total_reward: float = 0.0
    total_steps: int = 0
    total_infractions: int = 0
    
    completed_route_ids: List[str] = field(default_factory=list)
    
    records: List[Dict] = field(default_factory=list)
    
    last_update: str = ""
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'EnvStatistics':
        return cls(**data)


class RLStatisticsManager:
    """Persist route statistics independently for each environment worker."""
    
    def __init__(
        self,
        result_dir: str = "./results",
        env_id: int = 0,
        auto_save: bool = True,
        save_interval: int = 1,
    ):
        self.result_dir = Path(result_dir)
        self.env_id = env_id
        self.auto_save = auto_save
        self.save_interval = save_interval
        
        self.result_dir.mkdir(parents=True, exist_ok=True)
        
        self._result_file = self.result_dir / f"results_env_{env_id}.json"
        self._lock_file = self.result_dir / f".lock_env_{env_id}"
        
        self._stats = EnvStatistics(env_id=env_id)
        self._current_record: Optional[RouteRecord] = None
        self._completed_route_ids: Set[str] = set()
        self._unsaved_count = 0
        
        self._scenario = None
        self._route_length = 0.0
        
        self._load_existing_data()
        
        logger.info(f"RLStatisticsManager[{env_id}] initialized, "
                   f"completed routes: {len(self._completed_route_ids)}")
    
    def _load_existing_data(self):
        if self._result_file.exists():
            try:
                with open(self._result_file, 'r') as f:
                    data = json.load(f)
                self._stats = EnvStatistics.from_dict(data)
                self._completed_route_ids = set(self._stats.completed_route_ids)
                logger.info(f"Loaded existing data: {len(self._completed_route_ids)} completed routes")
            except Exception as e:
                logger.warning(f"Failed to load existing data: {e}")
                self._stats = EnvStatistics(env_id=self.env_id)
    
    def is_route_completed(self, route_id: str) -> bool:
        return route_id in self._completed_route_ids
    
    def get_completed_route_ids(self) -> Set[str]:
        return self._completed_route_ids.copy()
    
    def start_route(self, route_id: str, route_length: float = 0.0):
        """Begin collecting statistics for one route."""
        self._current_record = RouteRecord(
            route_id=route_id,
            env_id=self.env_id,
            status="Running",
            route_length=route_length,
            start_time=datetime.now().isoformat()
        )
        self._route_length = route_length
        logger.debug(f"Started route: {route_id}")
    
    def set_scenario(self, scenario, route_length: float = 0.0):
        self._scenario = scenario
        self._route_length = route_length
    
    def update_step(self, reward: float = 0.0):
        if self._current_record:
            self._current_record.total_steps += 1
            self._current_record.total_reward += reward
    
    def end_route(
        self,
        duration_game: float = 0.0,
        duration_system: float = 0.0,
        failure_message: str = "",
        crashed: bool = False,
        crash_reason: str = "",
        crash_type: str = "",
        crash_detail: str = "",
    ):
        """Finalize and persist the current route record."""
        if not self._current_record:
            logger.warning("No current route to end")
            return None
        
        record = self._current_record
        record.end_time = datetime.now().isoformat()
        record.duration_game = duration_game
        record.duration_system = duration_system
        record.failure_message = failure_message
        record.crash_reason = crash_reason
        record.crash_type = crash_type
        record.crash_detail = crash_detail or failure_message
        
        self._compute_route_statistics(record, failure_message, crashed)
        
        if crashed:
            record.status = "Crashed"
            self._stats.crashed_routes += 1
        elif record.score_route >= 100:
            record.status = "Completed" if record.num_infractions > 0 else "Perfect"
            self._stats.completed_routes += 1
        else:
            record.status = "Failed"
            if failure_message:
                record.status += f" - {failure_message}"
            self._stats.failed_routes += 1
        
        self._stats.records.append(record.to_dict())
        self._stats.completed_route_ids.append(record.route_id)
        self._completed_route_ids.add(record.route_id)
        self._stats.total_routes += 1
        self._stats.total_score += record.score_composed
        self._stats.total_reward += record.total_reward
        self._stats.total_steps += record.total_steps
        self._stats.total_infractions += record.num_infractions
        
        self._current_record = None
        self._scenario = None
        self._unsaved_count += 1
        
        if self.auto_save and self._unsaved_count >= self.save_interval:
            self.save()
        
        logger.debug(f"Route {record.route_id} ended: {record.status}, "
                     f"score={record.score_composed:.2f}")
        return RouteRecord.from_dict(record.to_dict())

    def record_route_crash(
        self,
        route_id: str = "",
        *,
        failure_message: str = "",
        crash_reason: str = "",
        crash_type: str = "",
        crash_detail: str = "",
    ) -> Optional[RouteRecord]:
        target_route_id = str(route_id or "").strip()
        if not target_route_id and self._current_record is not None:
            target_route_id = self._current_record.route_id
        if not target_route_id:
            logger.warning("No route id available to record crash")
            return None
        if self._current_record is None:
            if self.is_route_completed(target_route_id):
                return None
            self.start_route(target_route_id)
        record = self.end_route(
            failure_message=failure_message,
            crashed=True,
            crash_reason=crash_reason,
            crash_type=crash_type,
            crash_detail=crash_detail,
        )
        self.save()
        return record
    
    def _compute_route_statistics(
        self,
        record: RouteRecord,
        failure_message: str = "",
        crashed: bool = False
    ):
        score_penalty = 1.0
        score_route = 0.0
        
        record.infractions = {name: [] for name in PENALTY_NAME_DICT.values()}
        record.infractions['route_timeout'] = []
        
        if self._scenario:
            if hasattr(self._scenario, 'timeout_node') and self._scenario.timeout_node:
                if self._scenario.timeout_node.timeout:
                    record.infractions['route_timeout'].append('Route timeout.')
            
            for node in self._scenario.get_criteria():
                for event in node.events:
                    event_type = event.get_type()
                    
                    if event_type in PENALTY_VALUE_DICT:
                        score_penalty *= PENALTY_VALUE_DICT[event_type]
                        infraction_name = PENALTY_NAME_DICT[event_type]
                        record.infractions[infraction_name].append(event.get_message())
                    
                    elif event_type in PENALTY_PERC_DICT:
                        event_value = event.get_dict().get('percentage', 0)
                        penalty_value, penalty_type = PENALTY_PERC_DICT[event_type]
                        if penalty_type == "decreases":
                            score_penalty *= (1 - (1 - penalty_value) * (1 - event_value / 100))
                        elif penalty_type == "increases":
                            score_penalty *= (1 - (1 - penalty_value) * event_value / 100)
                        infraction_name = PENALTY_NAME_DICT[event_type]
                        record.infractions[infraction_name].append(event.get_message())
                    
                    elif event_type == TrafficEventType.ROUTE_DEVIATION:
                        record.infractions['route_dev'].append(event.get_message())
                    
                    elif event_type == TrafficEventType.VEHICLE_BLOCKED:
                        record.infractions['vehicle_blocked'].append(event.get_message())
                    
                    elif event_type == TrafficEventType.ROUTE_COMPLETION:
                        score_route = event.get_dict().get('route_completed', 0)
        
        record.score_route = round(score_route, 6)
        record.score_penalty = round(score_penalty, 6)
        record.score_composed = round(max(score_route * score_penalty, 0.0), 6)
        record.num_infractions = sum(len(v) for v in record.infractions.values())
    
    def save(self):
        self._stats.last_update = datetime.now().isoformat()
        
        self.result_dir.mkdir(parents=True, exist_ok=True)
        
        temp_file = self._result_file.with_suffix('.tmp')
        try:
            with open(temp_file, 'w') as f:
                json.dump(self._stats.to_dict(), f, indent=2, ensure_ascii=False)
            
            shutil.move(str(temp_file), str(self._result_file))
            self._unsaved_count = 0
            logger.debug(f"Saved statistics to {self._result_file}")
            
        except Exception as e:
            logger.error(f"Failed to save statistics: {e}")
            if temp_file.exists():
                temp_file.unlink()
    
    def get_summary(self) -> Dict[str, Any]:
        total = self._stats.total_routes
        return {
            'env_id': self.env_id,
            'total_routes': total,
            'completed_routes': self._stats.completed_routes,
            'failed_routes': self._stats.failed_routes,
            'crashed_routes': self._stats.crashed_routes,
            'avg_score': self._stats.total_score / max(total, 1),
            'avg_reward': self._stats.total_reward / max(total, 1),
            'total_steps': self._stats.total_steps,
            'total_infractions': self._stats.total_infractions,
        }
    
    @staticmethod
    def convert_to_leaderboard_format(rl_results: Dict) -> Dict:
        """Convert stored RL results to the Leaderboard result schema."""
        import math
        
        records = []
        for idx, r in enumerate(rl_results.get('records', [])):
            status = r.get('status', 'Started')
            if 'Crashed' in status:
                status = 'Failed - Simulation crashed'
            elif 'Failed' in status and ' - ' not in status:
                status = 'Failed'
            
            record = {
                'index': idx,
                'route_id': r.get('route_id', f'Route_{idx}'),
                'status': status,
                'num_infractions': r.get('num_infractions', 0),
                'infractions': r.get('infractions', {}),
                'scores': {
                    'score_route': r.get('score_route', 0.0),
                    'score_penalty': r.get('score_penalty', 1.0),
                    'score_composed': r.get('score_composed', 0.0)
                },
                'meta': {
                    'route_length': r.get('route_length', 0.0),
                    'duration_game': r.get('duration_game', 0.0),
                    'duration_system': r.get('duration_system', 0.0)
                }
            }
            records.append(record)
        
        total_routes = len(records)
        if total_routes == 0:
            global_record = {
                'index': -1,
                'route_id': -1,
                'status': 'Perfect',
                'infractions': {name: 0 for name in PENALTY_NAME_DICT.values()},
                'scores_mean': {'score_composed': 0, 'score_route': 0, 'score_penalty': 0},
                'scores_std_dev': {'score_composed': 'NaN', 'score_route': 'NaN', 'score_penalty': 'NaN'},
                'meta': {'total_length': 0, 'duration_game': 0, 'duration_system': 0, 'exceptions': []}
            }
        else:
            scores_mean = {'score_composed': 0.0, 'score_route': 0.0, 'score_penalty': 0.0}
            total_length = 0.0
            total_duration_game = 0.0
            total_duration_system = 0.0
            exceptions = []
            global_status = 'Perfect'
            
            infractions_count = {name: 0 for name in PENALTY_NAME_DICT.values()}
            infractions_count['route_timeout'] = 0
            
            for record in records:
                scores_mean['score_composed'] += record['scores']['score_composed'] / total_routes
                scores_mean['score_route'] += record['scores']['score_route'] / total_routes
                scores_mean['score_penalty'] += record['scores']['score_penalty'] / total_routes
                
                total_length += record['meta']['route_length']
                total_duration_game += record['meta']['duration_game']
                total_duration_system += record['meta']['duration_system']
                
                for key, infractions in record['infractions'].items():
                    if key in infractions_count:
                        infractions_count[key] += len(infractions) if isinstance(infractions, list) else 0
                
                route_status = record['status']
                if 'Failed' in route_status:
                    exceptions.append((record['route_id'], record['index'], route_status))
                    global_status = 'Failed'
                elif global_status == 'Perfect' and route_status not in ('Perfect', 'Completed'):
                    global_status = route_status
            
            for key in scores_mean:
                scores_mean[key] = round(scores_mean[key], 6)
            
            if total_routes == 1:
                scores_std_dev = {'score_composed': 'NaN', 'score_route': 'NaN', 'score_penalty': 'NaN'}
            else:
                scores_std_dev = {'score_composed': 0.0, 'score_route': 0.0, 'score_penalty': 0.0}
                for record in records:
                    for key in scores_std_dev:
                        score_key = key
                        diff = record['scores'][score_key] - scores_mean[key]
                        scores_std_dev[key] += diff ** 2
                
                for key in scores_std_dev:
                    scores_std_dev[key] = round(math.sqrt(scores_std_dev[key] / (total_routes - 1)), 3)
            
            km_driven = max(0.001, total_length / 1000 * scores_mean['score_route'] / 100)
            for key in infractions_count:
                infractions_count[key] = round(infractions_count[key] / km_driven, 3)
            
            global_record = {
                'index': -1,
                'route_id': -1,
                'status': global_status,
                'infractions': infractions_count,
                'scores_mean': scores_mean,
                'scores_std_dev': scores_std_dev,
                'meta': {
                    'total_length': round(total_length, 3),
                    'duration_game': round(total_duration_game, 3),
                    'duration_system': round(total_duration_system, 3),
                    'exceptions': exceptions
                }
            }
        
        entry_status = 'Finished'
        for record in records:
            if 'Simulation crashed' in record['status']:
                entry_status = 'Crashed'
                break
            elif "Agent's sensors were invalid" in record['status']:
                entry_status = 'Rejected'
                break
        
        eligible_values = {
            'Started': False, 'Finished': True, 'Rejected': False, 
            'Crashed': False, 'Invalid': False
        }
        
        values = [
            str(total_routes),
            str(global_record['scores_mean']['score_composed']),
            str(global_record['scores_mean']['score_route']),
            str(global_record['scores_mean']['score_penalty']),
            str(global_record['infractions'].get('collisions_pedestrian', 0)),
            str(global_record['infractions'].get('collisions_vehicle', 0)),
            str(global_record['infractions'].get('collisions_layout', 0)),
            str(global_record['infractions'].get('red_light', 0)),
            str(global_record['infractions'].get('stop_infraction', 0)),
            str(global_record['infractions'].get('outside_route_lanes', 0)),
            str(global_record['infractions'].get('route_dev', 0)),
            str(global_record['infractions'].get('route_timeout', 0)),
            str(global_record['infractions'].get('vehicle_blocked', 0)),
            str(global_record['infractions'].get('yield_emergency_vehicle_infractions', 0)),
            str(global_record['infractions'].get('scenario_timeouts', 0)),
            str(global_record['infractions'].get('min_speed_infractions', 0)),
        ]
        
        labels = [
            "route number",
            "Avg. driving score",
            "Avg. route completion",
            "Avg. infraction penalty",
            "Collisions with pedestrians",
            "Collisions with vehicles",
            "Collisions with layout",
            "Red lights infractions",
            "Stop sign infractions",
            "Off-road infractions",
            "Route deviations",
            "Route timeouts",
            "Agent blocked",
            "Yield emergency vehicles infractions",
            "Scenario timeouts",
            "Min speed infractions"
        ]
        
        return {
            '_checkpoint': {
                'global_record': global_record,
                'progress': [total_routes, total_routes],
                'records': records
            },
            'entry_status': entry_status,
            'eligible': eligible_values.get(entry_status, False),
            'sensors': [],
            'values': values,
            'labels': labels
        }
    
    @staticmethod
    def export_leaderboard_results(result_dir: str, output_file: str = None, num_envs: int = None) -> str:
        """Merge worker results and export them in Leaderboard format."""
        merged_data = RLStatisticsManager.merge_results(result_dir, num_envs)
        
        leaderboard_data = RLStatisticsManager.convert_to_leaderboard_format(merged_data)
        
        if output_file is None:
            output_file = str(Path(result_dir) / "leaderboard_results.json")
        
        with open(output_file, 'w') as f:
            json.dump(leaderboard_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Exported Leaderboard format results to {output_file}")
        return output_file

    @staticmethod
    def merge_results(result_dir: str, num_envs: int = None) -> Dict:
        """Merge all available per-environment result files."""
        result_path = Path(result_dir)
        
        if num_envs is None:
            result_files = list(result_path.glob("results_env_*.json"))
        else:
            result_files = [result_path / f"results_env_{i}.json" for i in range(num_envs)]
            result_files = [f for f in result_files if f.exists()]
        
        if not result_files:
            logger.warning(f"No result files found in {result_dir}")
            return {}
        
        all_records = []
        total_stats = {
            'total_routes': 0,
            'completed_routes': 0,
            'failed_routes': 0,
            'crashed_routes': 0,
            'total_score': 0.0,
            'total_reward': 0.0,
            'total_steps': 0,
            'total_infractions': 0,
        }
        
        for result_file in result_files:
            try:
                with open(result_file, 'r') as f:
                    data = json.load(f)
                
                all_records.extend(data.get('records', []))
                total_stats['total_routes'] += data.get('total_routes', 0)
                total_stats['completed_routes'] += data.get('completed_routes', 0)
                total_stats['failed_routes'] += data.get('failed_routes', 0)
                total_stats['crashed_routes'] += data.get('crashed_routes', 0)
                total_stats['total_score'] += data.get('total_score', 0)
                total_stats['total_reward'] += data.get('total_reward', 0)
                total_stats['total_steps'] += data.get('total_steps', 0)
                total_stats['total_infractions'] += data.get('total_infractions', 0)
                
            except Exception as e:
                logger.warning(f"Failed to load {result_file}: {e}")
        
        total = max(total_stats['total_routes'], 1)
        merged_data = {
            **total_stats,
            'avg_score': total_stats['total_score'] / total,
            'avg_reward': total_stats['total_reward'] / total,
            'records': all_records,
            'merge_time': datetime.now().isoformat(),
        }
        
        merged_file = result_path / "results_merged.json"
        with open(merged_file, 'w') as f:
            json.dump(merged_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Merged {len(result_files)} result files, "
                   f"total {len(all_records)} records -> {merged_file}")
        
        return merged_data
    
    def __del__(self):
        if self._unsaved_count > 0:
            try:
                self.save()
            except Exception:
                pass
