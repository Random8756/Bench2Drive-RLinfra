"""Training visualisation module.

Monitors RL algorithm behaviour during training in CARLA:
1. Real-time BEV-view recording.
2. Periodic video export.
3. Key-metric overlay (reward, speed, steering, ...).
4. Async writer thread that does not block training.

Usage - register the callback on ``CARLAEnvPool``:

    ```python
    from b2d_rlinfra.evaluation.visualization import TrainingVisualizer
    
    visualizer = TrainingVisualizer(
        output_dir="./vis_logs",
        save_interval_episodes=10,  # save a video every 10 episodes
    )
    
    pool = CARLAEnvPool(
        env_fn=make_env,
        config=config,
        step_callback=visualizer.on_step,
        episode_callback=visualizer.on_episode_end,
    )
    ```
"""

import os
import time
import threading
import queue
import numpy as np
import cv2
from datetime import datetime
from typing import Dict, Any, Optional, Callable, List, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import logging
from b2d_rlinfra.evaluation.sensor_visualization import extract_sensor_visual_frame

logger = logging.getLogger("Visualization")

__layer__ = (5, "Evaluation")


VIS_EXPORT_CONFIG = {
    "output_width": 1280,
    "output_height": 720,
    "panel_width": 420,
    "panel_padding_x": 8,
    "panel_padding_y": 14,
    "font_scale": 0.45,
    "min_font_scale": 0.32,
    "line_height": 15,
    "min_line_height": 11,
    "max_text_chars": 34,
    "background_color": (0, 0, 0),
    "text_color": (255, 255, 255),
    "dim_text_color": (180, 180, 180),
}


@dataclass
class EpisodeBuffer:
    """Frame buffer for a single episode."""
    worker_id: int
    episode_id: int
    frames: List[np.ndarray] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    actions: List[Any] = field(default_factory=list)
    infos: List[Dict] = field(default_factory=list)
    reward_components: List[Dict] = field(default_factory=list)  # per-frame reward breakdown
    start_time: float = field(default_factory=time.time)
    route_id: str = "unknown"
    
    @property
    def total_reward(self) -> float:
        return sum(self.rewards)
    
    @property
    def num_steps(self) -> int:
        return len(self.frames)


class TrainingVisualizer:
    """Buffer and asynchronously export annotated training videos."""
    
    def __init__(
        self,
        output_dir: str = "./vis_logs",
        fps: int = 10,
        save_interval_episodes: int = 10,
        save_interval_steps: int = 0,
        max_episodes_to_keep: int = 50,
        overlay_info: bool = True,
        enabled: bool = True,
        lazy_capture: bool = True,
        initial_step: int = 0,
        initial_episode: int = 0,
    ):
        self.output_dir = output_dir
        self.fps = fps
        self.save_interval_episodes = save_interval_episodes
        self.save_interval_steps = save_interval_steps
        self.max_episodes_to_keep = max_episodes_to_keep
        self.overlay_info = overlay_info
        self.enabled = enabled
        self.lazy_capture = lazy_capture
        
        self._episode_buffers: Dict[int, EpisodeBuffer] = {}
        self._episode_counts: Dict[int, int] = defaultdict(int)
        self._global_step = initial_step
        self._global_episode = initial_episode
        self._next_global_episode_id = initial_episode + 1
        self._worker_active_episode_ids: Dict[int, int] = {}
        
        self._should_capture: Dict[int, bool] = defaultdict(bool)
        
        self._worker_episode_ids: Dict[int, int] = {}
        
        self._write_queue: queue.Queue = queue.Queue()
        self._writer_thread: Optional[threading.Thread] = None
        self._shutdown = threading.Event()
        
        self._saved_videos: List[str] = []
        
        if enabled:
            os.makedirs(output_dir, exist_ok=True)
            self._start_writer_thread()
            logger.info(f"[TrainingVisualizer] Initialized, output_dir: {output_dir}")
    
    def _start_writer_thread(self):
        self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._writer_thread.start()
    
    def _writer_loop(self):
        while True:
            try:
                task = self._write_queue.get(timeout=1.0)
                if task is None:
                    break
                self._process_write_task(task)
            except queue.Empty:
                if self._shutdown.is_set():
                    break
                continue
            except Exception as e:
                logger.error(f"[TrainingVisualizer] Write error: {e}")
    
    def _process_write_task(self, task: Dict):
        task_type = task.get('type')
        
        if task_type == 'save_video':
            self._save_video(
                frames=task['frames'],
                filepath=task['filepath'],
                overlay_data=task.get('overlay_data'),
                reward_components=task.get('reward_components'),
            )

    @staticmethod
    def _ensure_rgb_frame(frame: np.ndarray) -> np.ndarray:
        if frame is None:
            return frame
        if len(frame.shape) == 2:
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        if frame.shape[2] == 4:
            return cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
        return frame

    def _prepare_frames_for_export(
        self,
        frames: List[np.ndarray],
        overlay_data: Optional[List[Dict]] = None,
        reward_components: Optional[List[Dict]] = None,
    ) -> List[np.ndarray]:
        """Normalize frames and compose optional overlays for export."""
        if not frames:
            return []

        rgb_frames = [self._ensure_rgb_frame(frame) for frame in frames]
        rgb_frames = [frame for frame in rgb_frames if frame is not None]
        if not rgb_frames:
            return []

        use_overlay = bool(self.overlay_info and overlay_data)
        output_w = int(VIS_EXPORT_CONFIG["output_width"])
        output_h = int(VIS_EXPORT_CONFIG["output_height"])
        if use_overlay:
            panel_width = int(VIS_EXPORT_CONFIG["panel_width"])
            panel_height = output_h
            target_bev_w = max(1, output_w - panel_width)
        else:
            panel_width = 0
            panel_height = 0
            target_bev_w = output_w

        target_h = output_h
        target_w = output_w

        processed_frames: List[np.ndarray] = []
        for i, frame in enumerate(rgb_frames):
            bev_frame = self._resize_to_canvas(frame, target_bev_w, target_h)

            if use_overlay:
                if i < len(overlay_data):
                    overlay_info = overlay_data[i].copy()
                    if reward_components and i < len(reward_components):
                        overlay_info['reward_components'] = reward_components[i]
                    info_panel = self._create_info_panel(
                        overlay_info,
                        panel_width=panel_width,
                        target_height=panel_height,
                    )
                else:
                    info_panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
                combined = np.hstack([bev_frame, info_panel])
            else:
                combined = bev_frame

            if combined.shape[0] != target_h or combined.shape[1] != target_w:
                combined = self._resize_to_canvas(combined, target_w, target_h)
            processed_frames.append(combined)

        return processed_frames

    @staticmethod
    def _resize_to_canvas(frame: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
        """Resize without changing aspect ratio and center on a fixed canvas."""
        if frame is None:
            return np.zeros((target_h, target_w, 3), dtype=np.uint8)

        frame = TrainingVisualizer._ensure_rgb_frame(frame)
        h, w = frame.shape[:2]
        if h <= 0 or w <= 0:
            return np.zeros((target_h, target_w, 3), dtype=np.uint8)

        scale = min(target_w / float(w), target_h / float(h))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        x0 = (target_w - new_w) // 2
        y0 = (target_h - new_h) // 2
        canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
        return canvas
    
    def _save_video(
        self,
        frames: List[np.ndarray],
        filepath: str,
        overlay_data: Optional[List[Dict]] = None,
        reward_components: Optional[List[Dict]] = None,
    ):
        if not frames:
            return
        
        
        try:
            processed_frames_rgb = self._prepare_frames_for_export(
                frames=frames,
                overlay_data=overlay_data,
                reward_components=reward_components,
            )
            if not processed_frames_rgb:
                return
            processed_frames = [
                cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in processed_frames_rgb
            ]
            
            height, width = processed_frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(filepath, fourcc, self.fps, (width, height))
            
            for frame in processed_frames:
                writer.write(frame)
            
            writer.release()
            route_id, scenario_name = self._extract_video_context(overlay_data)
            logger.info(
                "[artifact] type=video path=%s frames=%d route=%s scenario=%s",
                filepath,
                len(frames),
                route_id,
                scenario_name,
            )
            
            self._saved_videos.append(filepath)
            self._cleanup_old_videos()
            
            
        except Exception as e:
            logger.error(f"[TrainingVisualizer] Failed to save video: {e}")

    @staticmethod
    def _extract_video_context(overlay_data: Optional[List[Dict]]) -> Tuple[str, str]:
        route_id = "unknown"
        scenario_name = "unknown"
        for info in overlay_data or []:
            if not isinstance(info, dict):
                continue
            route_value = str(info.get("route_id") or info.get("route") or "").strip()
            scenario_value = str(
                info.get("scenario_name")
                or info.get("scenario_instance_name")
                or ""
            ).strip()
            if route_value:
                route_id = route_value
            if scenario_value:
                scenario_name = scenario_value
            if route_value or scenario_value:
                break
        return route_id, scenario_name

    @staticmethod
    def _flatten_action(action: Any) -> Optional[np.ndarray]:
        """Convert action-like inputs to a flat numpy vector when possible."""
        if action is None:
            return None
        if isinstance(action, list):
            action = np.asarray(action, dtype=np.float32)
        if isinstance(action, np.ndarray):
            return action.astype(np.float32, copy=False).reshape(-1)
        return None

    def _format_action_lines(self, data: Dict, compact: bool = False) -> List[str]:
        """Format planner-action / executed-control info for overlays."""
        action = self._flatten_action(data.get('action'))
        executed = self._flatten_action(data.get('executed_control'))
        action_type = data.get('action_type')

        if action_type == 'trajectory' and action is not None and action.size >= 2:
            state_dim = int(data.get('trajectory_state_dim', 2) or 2)
            if state_dim <= 0:
                state_dim = 2
            num_points = int(data.get('trajectory_num_points', max(action.size // state_dim, 1)) or 1)
            if compact:
                lines = [f"Tr:{num_points}x{state_dim}"]
            else:
                lines = [f"Traj: {num_points}x{state_dim}"]
                lines.append(f"P1: x={action[0]:.2f} y={action[1]:.2f}")
                last_base = min(max((num_points - 1) * state_dim, 0), max(action.size - 2, 0))
                if num_points > 1 and action.size >= last_base + 2:
                    lines.append(f"Pend: x={action[last_base]:.2f} y={action[last_base + 1]:.2f}")
                if state_dim >= 3 and action.size >= 3:
                    lines.append(f"H1: {action[2]:.2f} rad")

            if executed is not None and executed.size >= 3:
                if compact:
                    lines.append(
                        f"C:{executed[0]:.1f},{executed[1]:.1f},{executed[2]:.1f}"
                    )
                else:
                    lines.append(
                        f"Ctrl: T={executed[0]:.2f} S={executed[1]:.2f} B={executed[2]:.2f}"
                    )
            return lines

        if executed is not None and executed.size >= 3:
            if compact:
                return [f"A:{executed[0]:.1f},{executed[1]:.1f},{executed[2]:.1f}"]
            return [f"Act: T={executed[0]:.2f} S={executed[1]:.2f} B={executed[2]:.2f}"]

        if action is None:
            return []
        if action.size >= 3:
            if compact:
                return [f"A:{action[0]:.1f},{action[1]:.1f},{action[2]:.1f}"]
            return [f"Act: T={action[0]:.2f} S={action[1]:.2f} B={action[2]:.2f}"]
        if action.size >= 2:
            if compact:
                return [f"A:{action[0]:.1f},{action[1]:.1f}"]
            return [f"Act: {action[0]:.2f},{action[1]:.2f}"]
        return []

    @staticmethod
    def _short_text(value: Any, max_len: int = None) -> str:
        if max_len is None:
            max_len = int(VIS_EXPORT_CONFIG["max_text_chars"])
        text = str(value or "").strip()
        if not text:
            return "unknown"
        if len(text) <= max_len:
            return text
        return text[:max_len - 3] + "..."

    @staticmethod
    def _normalize_route_completion(value: Any, ratio_hint: bool = False) -> Optional[float]:
        if not isinstance(value, (int, float, np.number)):
            return None
        percent = float(value)
        if ratio_hint:
            percent *= 100.0
        return max(0.0, min(percent, 100.0))

    @staticmethod
    def _to_finite_float(value: Any, default: Optional[float] = None) -> Optional[float]:
        if isinstance(value, (bool, np.bool_)):
            return default
        if not isinstance(value, (int, float, np.number)):
            return default
        value = float(value)
        if not np.isfinite(value):
            return default
        return value

    @classmethod
    def _format_float(
        cls,
        value: Any,
        precision: int = 4,
        default: str = "N/A",
        suffix: str = "",
    ) -> str:
        value = cls._to_finite_float(value)
        if value is None:
            return default
        return f"{value:.{precision}f}{suffix}"

    @classmethod
    def _extract_route_completion(cls, info: Optional[Dict]) -> Optional[float]:
        if not isinstance(info, dict):
            return None

        for key in ("route_completed_ratio", "truncation_route_completed_ratio"):
            value = cls._normalize_route_completion(info.get(key), ratio_hint=True)
            if value is not None:
                return value

        value = cls._normalize_route_completion(info.get("simple_reward_RC"), ratio_hint=False)
        if value is not None:
            return value

        best_value = None
        for event in info.get("all_events", []) or []:
            if not isinstance(event, dict):
                continue
            event_type = event.get("type") or event.get("event_type")
            if event_type != "ROUTE_COMPLETION":
                continue
            details = event.get("details") or {}
            if not isinstance(details, dict):
                continue
            value = cls._normalize_route_completion(details.get("route_completed"), ratio_hint=False)
            if value is not None:
                best_value = value if best_value is None else max(best_value, value)
        return best_value

    @staticmethod
    def _fit_text_to_width(
        text: str,
        max_width: int,
        font: int,
        font_scale: float,
        thickness: int,
    ) -> str:
        if cv2.getTextSize(text, font, font_scale, thickness)[0][0] <= max_width:
            return text
        if max_width <= 0:
            return ""

        suffix = "..."
        candidate = text
        while len(candidate) > len(suffix):
            candidate = candidate[:-1]
            fitted = candidate.rstrip() + suffix
            if cv2.getTextSize(fitted, font, font_scale, thickness)[0][0] <= max_width:
                return fitted
        return suffix
        
    
    def _create_info_panel(self, data: Dict, panel_width: int = None, target_height: int = None) -> np.ndarray:
        """Render an information panel at the requested dimensions."""
        cfg = VIS_EXPORT_CONFIG
        font_scale = float(cfg["font_scale"])
        line_height = int(cfg["line_height"])
        x_offset = int(cfg["panel_padding_x"])
        
        y_offset = int(cfg["panel_padding_y"])
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = tuple(cfg["text_color"])
        color_dim = tuple(cfg["dim_text_color"])
        thickness = 1
        line_type = cv2.LINE_AA
        
        overlay_items = []
        item_colors = []
        
        overlay_items.append(f"Step: {data.get('step', 'N/A')}")
        item_colors.append(color)
        if 'global_step' in data:
            overlay_items.append(f"GStep: {data.get('global_step', 'N/A')}")
            item_colors.append(color_dim)
        overlay_items.append(f"Route: {self._short_text(data.get('route_id'))}")
        item_colors.append(color)
        overlay_items.append(f"Scenario: {self._short_text(data.get('scenario_name'))}")
        item_colors.append(color)
        route_completion = data.get('route_completion')
        route_completion_value = self._to_finite_float(route_completion)
        if route_completion_value is not None:
            overlay_items.append(f"RC: {route_completion_value:.1f}%")
            item_colors.append(color)
        else:
            overlay_items.append("RC: N/A")
            item_colors.append(color_dim)
        reward_value = self._to_finite_float(data.get('reward'), 0.0)
        total_reward_value = self._to_finite_float(data.get('total_reward'), 0.0)
        overlay_items.append(f"Reward: {reward_value:.3f}")
        item_colors.append(color)
        overlay_items.append(f"Total: {total_reward_value:.3f}")
        item_colors.append(color)
        
        if 'speed' in data:
            overlay_items.append(f"Speed: {self._format_float(data['speed'], 2)} m/s")
            item_colors.append(color)
        if 'desired_speed' in data:
            overlay_items.append(f"Desired: {self._format_float(data['desired_speed'], 2)} m/s")
            item_colors.append(color)
        for line in self._format_action_lines(data, compact=False):
            overlay_items.append(line)
            item_colors.append(color)
        
        reward_comp = data.get('reward_components', {})
        overlay_items.append("")
        item_colors.append(color)
        overlay_items.append("--- Reward Components ---")
        item_colors.append(color)
        
        val = self._to_finite_float(reward_comp.get('micro_deviation_penalty'), 0.0)
        overlay_items.append(f"Deviation: {val:.4f}")
        item_colors.append(color if val != 0 else color_dim)
        
        val = self._to_finite_float(reward_comp.get('reward_per_meter'), 0.0)
        overlay_items.append(f"Per Meter: {val:.4f}")
        item_colors.append(color if val != 0 else color_dim)
        
        val = self._to_finite_float(reward_comp.get('reward_speed'), 0.0)
        overlay_items.append(f"Speed Rwd: {val:.4f}")
        item_colors.append(color if val != 0 else color_dim)
        
        val = self._to_finite_float(reward_comp.get('steer_consistency_penalty'), 0.0)
        overlay_items.append(f"Steer Pen: {val:.4f}")
        item_colors.append(color if val != 0 else color_dim)
        
        val = self._to_finite_float(reward_comp.get('terminate_reward'), 0.0)
        overlay_items.append(f"Terminate: {val:.4f}")
        item_colors.append(color if val != 0 else color_dim)
        
        val = self._to_finite_float(reward_comp.get('reverse_penalty'), 0.0)
        overlay_items.append(f"Reverse: {val:.4f}")
        item_colors.append(color if val != 0 else color_dim)
        
        val = self._to_finite_float(reward_comp.get('turn'), 0.0)
        overlay_items.append(f"Turn: {val:.4f}")
        item_colors.append(color if val != 0 else color_dim)
        
        val = self._to_finite_float(reward_comp.get('pass_green'), 0.0)
        overlay_items.append(f"Pass Green: {val:.4f}")
        item_colors.append(color if val != 0 else color_dim)

        val = self._to_finite_float(reward_comp.get('stop_wait_penalty'), 0.0)
        overlay_items.append(f"Stop Wait: {val:.4f}")
        item_colors.append(color if val != 0 else color_dim)
        
        case_val = reward_comp.get('case', '-')
        overlay_items.append(f"Case: {case_val}")
        item_colors.append(color if case_val and case_val != '-' else color_dim)
        
        merged = reward_comp.get('merged', False)
        overlay_items.append(f"Merged: {merged}")
        item_colors.append(color if merged else color_dim)
        
        red = reward_comp.get('red', False)
        overlay_items.append(f"Red Light: {red}")
        item_colors.append(color if red else color_dim)
        
        emergency = reward_comp.get('emergency', False)
        overlay_items.append(f"Emergency: {emergency}")
        item_colors.append(color if emergency else color_dim)

        debug_red_latched_id = reward_comp.get('debug_red_latched_id', None)
        debug_second_stage = reward_comp.get('debug_second_red_stage', None)
        if debug_red_latched_id is not None or debug_second_stage is not None:
            overlay_items.append("")
            item_colors.append(color)
            overlay_items.append("--- TL Debug ---")
            item_colors.append(color)

            debug_red_locked = reward_comp.get('debug_red_is_locked', False)
            debug_red_dist = reward_comp.get('debug_red_recent_distance', -1.0)
            debug_red_min_dist = reward_comp.get('debug_red_min_distance', -1.0)
            red_dist_text = self._format_float(debug_red_dist, 1)
            red_min_dist_text = self._format_float(debug_red_min_dist, 1)
            overlay_items.append(
                f"TL Lock: {debug_red_latched_id} | Locked: {debug_red_locked} | Dist: {red_dist_text}/{red_min_dist_text}"
            )
            item_colors.append(color if debug_red_locked else color_dim)

            lock_source = reward_comp.get('debug_red_lock_source', '-')
            release_reason = reward_comp.get('debug_red_release_reason', '-')
            overlay_items.append(f"TL Src/Rel: {lock_source} / {release_reason}")
            item_colors.append(color if (lock_source not in (None, 'none') or release_reason not in (None, '-', 'none')) else color_dim)

            second_tracked = reward_comp.get('debug_second_red_tracked_id', -1)
            second_current = reward_comp.get('debug_second_red_current_id', -1)
            second_state = reward_comp.get('debug_second_red_current_state', 'None')
            overlay_items.append(f"2ndRed: {debug_second_stage} | T:{second_tracked} C:{second_current} {second_state}")
            item_colors.append(color if debug_second_stage not in (None, 'init', 'no_current_light') else color_dim)

            second_stop = reward_comp.get('debug_second_red_stopped', False)
            second_green = reward_comp.get('debug_second_red_seen_green', False)
            overlay_items.append(f"2ndFlags: stopped={second_stop} seen_green={second_green}")
            item_colors.append(color if (second_stop or second_green) else color_dim)

        debug_stop_latched_id = reward_comp.get('debug_stop_latched_id', None)
        if debug_stop_latched_id is not None:
            overlay_items.append("")
            item_colors.append(color)
            overlay_items.append("--- Stop Debug ---")
            item_colors.append(color)

            debug_stop_locked = reward_comp.get('debug_stop_is_locked', False)
            debug_stop_completed = reward_comp.get('debug_stop_completed', False)
            debug_stop_dist = reward_comp.get('debug_stop_recent_distance', -1.0)
            debug_stop_min_dist = reward_comp.get('debug_stop_min_distance', -1.0)
            overlay_items.append(f"Stop Lock: {debug_stop_latched_id}")
            item_colors.append(color if (debug_stop_locked or debug_stop_completed) else color_dim)

            overlay_items.append(
                f"Stop Flags: locked={debug_stop_locked} done={debug_stop_completed}"
            )
            item_colors.append(color if (debug_stop_locked or debug_stop_completed) else color_dim)

            stop_dist_text = self._format_float(debug_stop_dist, 1)
            stop_min_dist_text = self._format_float(debug_stop_min_dist, 1)
            overlay_items.append(f"Stop Dist: {stop_dist_text}/{stop_min_dist_text}")
            item_colors.append(color if debug_stop_locked else color_dim)

            stop_lock_source = reward_comp.get('debug_stop_lock_source', '-')
            stop_release_reason = reward_comp.get('debug_stop_release_reason', '-')
            overlay_items.append(f"Stop Src/Rel: {stop_lock_source} / {stop_release_reason}")
            item_colors.append(
                color if (stop_lock_source not in (None, 'none') or stop_release_reason not in (None, '-', 'none')) else color_dim
            )
        
        if reward_comp.get('simple_reward_RC') is not None:
            val = self._to_finite_float(reward_comp.get('simple_reward_RC'))
            if val is None:
                overlay_items.append("Simple RC: N/A")
                item_colors.append(color_dim)
            else:
                overlay_items.append(f"Simple RC: {val:.4f}")
                item_colors.append(color if val != 0 else color_dim)
        if reward_comp.get('simple_reward_penal') is not None:
            val = reward_comp.get('simple_reward_penal')
            if isinstance(val, dict):
                active = any(
                    bool(item)
                    for key, item in val.items()
                    if not str(key).endswith('_factor')
                )
                overlay_items.append(f"Simple Pen: {'active' if active else '-'}")
                item_colors.append(color if active else color_dim)
            else:
                numeric_val = self._to_finite_float(val)
                if numeric_val is None:
                    overlay_items.append("Simple Pen: N/A")
                    item_colors.append(color_dim)
                else:
                    overlay_items.append(f"Simple Pen: {numeric_val:.4f}")
                    item_colors.append(color if numeric_val != 0 else color_dim)
        
        if panel_width is None:
            panel_width = int(cfg["panel_width"])
        panel_height = int(target_height or cfg["output_height"])

        available_h = max(1, panel_height - y_offset * 2)
        if overlay_items:
            fit_scale = min(1.0, available_h / float(len(overlay_items) * line_height))
            font_scale = max(float(cfg["min_font_scale"]), font_scale * fit_scale)
            line_height = max(int(cfg["min_line_height"]), int(round(line_height * fit_scale)))

        max_lines = max(1, available_h // max(1, line_height))
        if len(overlay_items) > max_lines:
            hidden = len(overlay_items) - max_lines + 1
            overlay_items = overlay_items[:max_lines - 1] + [f"... {hidden} more"]
            item_colors = item_colors[:max_lines - 1] + [color_dim]

        max_text_width = panel_width - x_offset * 2
        overlay_items = [
            self._fit_text_to_width(item, max_text_width, font, font_scale, thickness)
            for item in overlay_items
        ]

        panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
        panel[:, :] = tuple(cfg["background_color"])
        
        for i, text in enumerate(overlay_items):
            y = y_offset + (i + 1) * line_height
            if y >= panel_height:
                break
            cv2.putText(panel, text, (x_offset, y), font, font_scale, item_colors[i], thickness, line_type)
        
        return panel
    
    def _add_overlay(self, frame: np.ndarray, data: Dict) -> np.ndarray:
        h, w = frame.shape[:2]
        
        if w <= 64:
            font_scale = 0.25
            line_height = 8
            y_offset = 7
            bg_width = 55
            x_offset = 2
        elif w <= 128:
            font_scale = 0.3
            line_height = 10
            y_offset = 9
            bg_width = 70
            x_offset = 3
        else:
            font_scale = 0.4
            line_height = 14
            y_offset = 12
            bg_width = 120
            x_offset = 4
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (255, 255, 255)
        thickness = 1
        line_type = cv2.LINE_AA
        
        if w <= 64:
            overlay_items = [
                f"S:{data.get('step', '?')}",
                f"R:{data.get('reward', 0):.1f}",
            ]
            overlay_items.extend(self._format_action_lines(data, compact=True))
        else:
            overlay_items = [
                f"Step:{data.get('step', 'N/A')}",
                f"R:{data.get('reward', 0):.2f}",
                f"Tot:{data.get('total_reward', 0):.2f}",
            ]
            if 'speed' in data:
                overlay_items.append(f"Spd:{data['speed']:.1f}")
            overlay_items.extend(self._format_action_lines(data, compact=False))
            
            reward_comp = data.get('reward_components', {})
            if reward_comp:
                comp_items = []
                if reward_comp.get('micro_deviation_penalty') is not None and reward_comp.get('micro_deviation_penalty') != 0:
                    comp_items.append(f"Dev:{reward_comp['micro_deviation_penalty']:.3f}")
                if reward_comp.get('reward_per_meter') is not None and reward_comp.get('reward_per_meter') != 0:
                    comp_items.append(f"PerM:{reward_comp['reward_per_meter']:.3f}")
                if reward_comp.get('reward_speed') is not None and reward_comp.get('reward_speed') != 0:
                    comp_items.append(f"SpdR:{reward_comp['reward_speed']:.3f}")
                if reward_comp.get('steer_consistency_penalty') is not None and reward_comp.get('steer_consistency_penalty') != 0:
                    comp_items.append(f"Steer:{reward_comp['steer_consistency_penalty']:.3f}")
                if reward_comp.get('turn') is not None and reward_comp.get('turn') != 0:
                    comp_items.append(f"Turn:{reward_comp['turn']:.3f}")
                if reward_comp.get('pass_green') is not None and reward_comp.get('pass_green') != 0:
                    comp_items.append(f"Green:{reward_comp['pass_green']:.3f}")
                if reward_comp.get('case') is not None:
                    comp_items.append(f"Case:{reward_comp['case']}")
                if reward_comp.get('merged'):
                    comp_items.append("Merged")
                if reward_comp.get('red'):
                    comp_items.append("Red")
                if reward_comp.get('emergency'):
                    comp_items.append("Emergency")
                
                if comp_items:
                    overlay_items.append("Reward Components:")
                    if len(comp_items) <= 3:
                        overlay_items.append(" | ".join(comp_items))
                    else:
                        for i in range(0, len(comp_items), 3):
                            overlay_items.append(" | ".join(comp_items[i:i+3]))
        
        overlay_height = len(overlay_items) * line_height + 4
        max_text_width = max([len(item) for item in overlay_items] + [bg_width // 2])
        actual_bg_width = min(max_text_width * int(font_scale * 8), w - 2)
        overlay = frame.copy()
        cv2.rectangle(overlay, (1, 1), (actual_bg_width, overlay_height), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        for i, text in enumerate(overlay_items):
            y = y_offset + i * line_height
            cv2.putText(frame, text, (x_offset, y), font, font_scale, color, thickness, line_type)
        
        return frame

    @staticmethod
    def _extract_visual_image(observation: Any) -> Optional[np.ndarray]:
        if observation is None:
            return None
        sensor_frame = extract_sensor_visual_frame(observation)
        if sensor_frame is not None:
            return sensor_frame
        if isinstance(observation, dict):
            for key in ['bev_image', 'rgb_front', 'rgb_third_person', 'semantic_bev']:
                if key in observation and observation[key] is not None:
                    return observation[key]
        elif isinstance(observation, np.ndarray):
            return observation
        return None
    
    def _cleanup_old_videos(self):
        while len(self._saved_videos) > self.max_episodes_to_keep:
            old_file = self._saved_videos.pop(0)
            try:
                if os.path.exists(old_file):
                    os.remove(old_file)
                    logger.debug(f"[TrainingVisualizer] Removed old video: {old_file}")
            except Exception as e:
                logger.warning(f"[TrainingVisualizer] Cleanup failed: {e}")
    
    def on_worker_crash(self, worker_id: int, crash_info: Optional[Dict] = None):
        """Finalize any buffered video after a worker crash."""
        crash_type = "unknown"
        crash_error = ""
        if isinstance(crash_info, dict):
            crash_type = str(crash_info.get('crash_type', 'unknown') or 'unknown')
            crash_error = str(crash_info.get('error', '') or '')

        global_episode_id = self._worker_active_episode_ids.pop(worker_id, None)
        if global_episode_id is None:
            self._global_episode += 1
            global_episode_id = self._global_episode
        else:
            self._global_episode = max(self._global_episode, global_episode_id)
        self._episode_counts[worker_id] += 1

        if worker_id in self._episode_buffers:
            old_buffer = self._episode_buffers[worker_id]
            if old_buffer.num_steps > 0:
                safe_crash_type = ''.join(
                    ch if (ch.isalnum() or ch in ('_', '-')) else '_'
                    for ch in crash_type
                )[:48] or "unknown"
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = (
                    f"ep{global_episode_id}_step{self._global_step}_"
                    f"w{worker_id}_crash_{safe_crash_type}_{timestamp}.mp4"
                )
                route_dir = os.path.join(self.output_dir, old_buffer.route_id)
                os.makedirs(route_dir, exist_ok=True)
                filepath = os.path.join(route_dir, filename)

                if old_buffer.infos:
                    old_buffer.infos[-1]['crashed'] = True
                    old_buffer.infos[-1]['crash_type'] = crash_type

                self._write_queue.put({
                    'type': 'save_video',
                    'frames': old_buffer.frames.copy(),
                    'filepath': filepath,
                    'overlay_data': old_buffer.infos.copy(),
                    'reward_components': old_buffer.reward_components.copy(),
                })

                logger.debug(
                    f"[TrainingVisualizer] Worker {worker_id} crashed; "
                    f"queued crash video: {filepath} ({old_buffer.num_steps} frames), "
                    f"type={crash_type}, error={crash_error}"
                )
            else:
                logger.debug(
                    f"[TrainingVisualizer] Worker {worker_id} crashed; "
                    f"no frames available to save (type={crash_type}, error={crash_error})"
                )
            del self._episode_buffers[worker_id]
        else:
            logger.debug(
                f"[TrainingVisualizer] Worker {worker_id} crashed; "
                "current episode is not in the recording cache, so no crash video can be exported. "
                "Set visualization.save_interval=1 or disable lazy_capture to guarantee saving."
            )
        
        self._should_capture[worker_id] = False
        
        if worker_id in self._worker_episode_ids:
            del self._worker_episode_ids[worker_id]
        self._worker_active_episode_ids.pop(worker_id, None)
    
    def on_episode_start(self, worker_id: int, episode_id: int = None):
        """Initialize capture state for a worker episode."""
        if not self.enabled:
            return
        
        if episode_id is not None and worker_id in self._worker_episode_ids:
            expected_id = self._worker_episode_ids[worker_id] + 1
            if episode_id != expected_id and episode_id != 0:
                logger.warning(
                    f"[TrainingVisualizer] Worker {worker_id} episode id is not continuous "
                    f"(expected {expected_id}, got {episode_id}); possible crash restart"
                )
                self.on_worker_crash(worker_id)
        
        if episode_id is not None:
            self._worker_episode_ids[worker_id] = episode_id
        
        global_episode_id = self._next_global_episode_id
        self._next_global_episode_id += 1
        self._worker_active_episode_ids[worker_id] = global_episode_id
        
        if self.lazy_capture:
            should_save = (
                self.save_interval_episodes > 0 and 
                global_episode_id % self.save_interval_episodes == 0
            )
            self._should_capture[worker_id] = should_save
            
            if should_save:
                logger.debug(
                    f"[TrainingVisualizer] Worker {worker_id} started recording "
                    f"(global_episode={global_episode_id})"
                )
        else:
            self._should_capture[worker_id] = True
        
        if worker_id in self._episode_buffers:
            del self._episode_buffers[worker_id]
    
    def on_step(
        self,
        worker_id: int,
        observation: Dict,
        action: Any,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: Dict,
    ):
        """Record one worker transition when capture is enabled."""
        if not self.enabled:
            return
        
        self._global_step += 1
        
        
        if info:
            if info.get('crashed', False):
                self.on_worker_crash(worker_id, crash_info=info)
                return
        
        if self.lazy_capture and not self._should_capture.get(worker_id, False):
            if terminated or truncated:
                self._episode_counts[worker_id] += 1
                active_id = self._worker_active_episode_ids.pop(worker_id, None)
                if active_id is not None:
                    self._global_episode = max(self._global_episode, active_id)
                else:
                    self._global_episode += 1
                self._should_capture[worker_id] = False
            return
        
        if worker_id not in self._episode_buffers:
            self._episode_buffers[worker_id] = EpisodeBuffer(
                worker_id=worker_id,
                episode_id=self._episode_counts[worker_id],
                route_id=info.get('route_id', 'unknown') if info else 'unknown',
            )
        
        buffer = self._episode_buffers[worker_id]
        
        vis_image = self._extract_visual_image(observation)
        
        if vis_image is not None:
            buffer.frames.append(vis_image.copy())
            buffer.rewards.append(reward)
            buffer.actions.append(action)
            buffer.infos.append({
                'step': buffer.num_steps,
                'global_step': self._global_step,
                'reward': reward,
                'total_reward': buffer.total_reward,
                'route_id': info.get('route_id', 'unknown') if info else 'unknown',
                'scenario_name': (
                    info.get('scenario_name') or info.get('scenario_instance_name') or 'unknown'
                ) if info else 'unknown',
                'route_completion': self._extract_route_completion(info),
                'speed': info.get('speed', 0) if info else 0,
                'desired_speed': info.get('desired_speed', 0) if info else 0,
                'action': action,
                'action_type': info.get('action_type') if info else None,
                'executed_control': info.get('executed_control') if info else None,
                'trajectory_num_points': info.get('trajectory_num_points') if info else None,
                'trajectory_state_dim': info.get('trajectory_state_dim') if info else None,
            })
            
            reward_comp = {}
            if info:
                reward_comp = {
                    'micro_deviation_penalty': info.get('micro_deviation_penalty', None),
                    'reward_per_meter': info.get('reward_per_meter', None),
                    'reward_speed': info.get('reward_speed', None),
                    'reward_speed_raw': info.get('reward_speed_raw', None),
                    'steer_consistency_penalty': info.get('steer_consistency_penaly', None),
                    'terminate_reward': info.get('terminate_reward', None),
                    'reverse_penalty': info.get('reverse_penalty', None),
                    'turn': info.get('turn', None),
                    'pass_green': info.get('pass_green', None),
                    'stop_wait_penalty': info.get('stop_wait_penalty', None),
                    'cumulative_length': info.get('cumulative_length', None),
                    'desired_speed': info.get('desired_speed', None),
                    'speed': info.get('speed', None),
                    'case': info.get('case', None),
                    'merged': info.get('merged', False),
                    'red': info.get('red', False),
                    'has_overlap': info.get('has_overlap', None),
                    'emergency': info.get('emergency', False),
                    'debug_red_latched_id': info.get('debug_red_latched_id', None),
                    'debug_red_recent_id': info.get('debug_red_recent_id', None),
                    'debug_red_lock_source': info.get('debug_red_lock_source', None),
                    'debug_red_release_reason': info.get('debug_red_release_reason', None),
                    'debug_red_recent_distance': info.get('debug_red_recent_distance', None),
                    'debug_red_min_distance': info.get('debug_red_min_distance', None),
                    'debug_red_is_locked': info.get('debug_red_is_locked', None),
                    'debug_stop_latched_id': info.get('debug_stop_latched_id', None),
                    'debug_stop_lock_source': info.get('debug_stop_lock_source', None),
                    'debug_stop_release_reason': info.get('debug_stop_release_reason', None),
                    'debug_stop_recent_distance': info.get('debug_stop_recent_distance', None),
                    'debug_stop_min_distance': info.get('debug_stop_min_distance', None),
                    'debug_stop_is_locked': info.get('debug_stop_is_locked', None),
                    'debug_stop_completed': info.get('debug_stop_completed', None),
                    'debug_second_red_stage': info.get('debug_second_red_stage', None),
                    'debug_second_red_tracked_id': info.get('debug_second_red_tracked_id', None),
                    'debug_second_red_current_id': info.get('debug_second_red_current_id', None),
                    'debug_second_red_current_state': info.get('debug_second_red_current_state', None),
                    'debug_second_red_seen_green': info.get('debug_second_red_seen_green', None),
                    'debug_second_red_stopped': info.get('debug_second_red_stopped', None),
                    'simple_reward_RC': info.get('simple_reward_RC', None),
                    'simple_reward_penal': info.get('simple_reward_penal', None),
                }
                
            buffer.reward_components.append(reward_comp)
            
        
        if terminated or truncated:
            self._on_episode_complete(worker_id, info)
    
    def _on_episode_complete(
        self,
        worker_id: int,
        info: Dict,
    ):
        self._episode_counts[worker_id] += 1
        
        global_episode_id = self._worker_active_episode_ids.pop(worker_id, None)
        if global_episode_id is None:
            self._global_episode += 1
            global_episode_id = self._global_episode
        else:
            self._global_episode = max(self._global_episode, global_episode_id)
        
        if worker_id not in self._episode_buffers:
            if self.lazy_capture:
                self._should_capture[worker_id] = False
            return
        
        buffer = self._episode_buffers[worker_id]
        
        if self.lazy_capture:
            should_save = self._should_capture.get(worker_id, False)
        else:
            should_save = (
                self.save_interval_episodes > 0 and 
                global_episode_id % self.save_interval_episodes == 0
            )
        
        if should_save and buffer.frames:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ep{global_episode_id}_step{self._global_step}_w{worker_id}_done_{timestamp}.mp4"
            route_dir = os.path.join(self.output_dir, buffer.route_id)
            os.makedirs(route_dir, exist_ok=True)
            filepath = os.path.join(route_dir, filename)
            
            self._write_queue.put({
                'type': 'save_video',
                'frames': buffer.frames.copy(),
                'filepath': filepath,
                'overlay_data': buffer.infos.copy(),
                'reward_components': buffer.reward_components.copy(),
            })
            
            logger.debug(
                f"[TrainingVisualizer] Recorded episode {global_episode_id} "
                f"(worker {worker_id}): {buffer.num_steps} steps, "
                f"total_reward: {buffer.total_reward:.2f}"
            )
        
        if worker_id in self._episode_buffers:
            del self._episode_buffers[worker_id]
        
        if self.lazy_capture:
            self._should_capture[worker_id] = False
        else:
            self._episode_buffers[worker_id] = EpisodeBuffer(
                worker_id=worker_id,
                episode_id=self._episode_counts[worker_id],
            )
    
    def on_episode_end(
        self,
        worker_id: int,
        episode_stats: Dict,
    ):
        """Hook for subclasses that aggregate episode statistics."""
        pass
    
    def process_step_result(
        self,
        actions: Dict[int, Any],
        obs: Dict[int, Any],
        rewards: Dict[int, float],
        terminateds: Dict[int, bool],
        truncateds: Dict[int, bool],
        infos: Dict[int, Dict],
    ):
        """Consume one dictionary-style environment-pool step result."""
        if not self.enabled:
            return
        
        for wid in rewards:
            info = infos.get(wid, {})
            
            if info.get('crashed', False):
                self.on_worker_crash(wid, crash_info=info)
                continue
            
            obs_item = obs.get(wid)
            
            self.on_step(
                worker_id=wid,
                observation=obs_item,
                action=actions.get(wid),
                reward=rewards[wid],
                terminated=terminateds.get(wid, False),
                truncated=truncateds.get(wid, False),
                info=info,
            )
    
    def process_step_sync_result(
        self,
        actions: List[Any],
        obs_list: List[Any],
        rewards: np.ndarray,
        terminateds: np.ndarray,
        truncateds: np.ndarray,
        info_list: List[Dict],
    ):
        """Consume one list-style synchronous environment-pool step result."""
        if not self.enabled:
            return
        
        num_envs = len(rewards)
        for i in range(num_envs):
            if actions[i] is None:
                continue
            
            info = info_list[i] if info_list else {}
            
            if info.get('crashed', False):
                self.on_worker_crash(i, crash_info=info)
                continue
            
            self.on_step(
                worker_id=i,
                observation=obs_list[i] if obs_list else None,
                action=actions[i],
                reward=float(rewards[i]),
                terminated=bool(terminateds[i]),
                truncated=bool(truncateds[i]),
                info=info,
            )
    
    def save_checkpoint_video(self, tag: str = "checkpoint"):
        """Queue videos for all currently buffered episodes."""
        if not self.enabled:
            return
        
        for worker_id, buffer in self._episode_buffers.items():
            if buffer.frames:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{tag}_step{self._global_step}_w{worker_id}_{timestamp}.mp4"
                route_dir = os.path.join(self.output_dir, buffer.route_id)
                os.makedirs(route_dir, exist_ok=True)
                filepath = os.path.join(route_dir, filename)
                
                self._write_queue.put({
                    'type': 'save_video',
                    'frames': buffer.frames.copy(),
                    'filepath': filepath,
                    'overlay_data': buffer.infos.copy(),
                })
    
    def get_stats(self) -> Dict:
        return {
            'global_step': self._global_step,
            'global_episode': self._global_episode,
            'episodes_per_worker': dict(self._episode_counts),
            'pending_writes': self._write_queue.qsize(),
            'saved_videos': len(self._saved_videos),
        }
    
    def close(self):
        if not self.enabled:
            return
        
        self.save_checkpoint_video(tag="final")
        
        self._shutdown.set()
        self._write_queue.put(None)
        
        if self._writer_thread and self._writer_thread.is_alive():
            self._writer_thread.join(timeout=120)
            if self._writer_thread.is_alive():
                logger.warning(
                    "[TrainingVisualizer] Writer thread did not stop within 120s; "
                    f"{self._write_queue.qsize()} tasks remain in the queue"
                )
        
        logger.info("[TrainingVisualizer] Closed")
    
    def __del__(self):
        self.close()


def create_visualized_env_pool(
    env_fn: Callable,
    config: Dict,
    vis_output_dir: str = "./vis_logs",
    vis_interval_episodes: int = 10,
    **pool_kwargs,
):
    """Create a CARLA environment pool and its training visualizer."""
    from b2d_rlinfra.simulation.runners.carla_env_pool import CARLAEnvPool
    
    visualizer = TrainingVisualizer(
        output_dir=vis_output_dir,
        save_interval_episodes=vis_interval_episodes,
    )
    
    pool = CARLAEnvPool(
        env_fn=env_fn,
        config=config,
        **pool_kwargs,
    )
    
    return pool, visualizer
