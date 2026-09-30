"""Trajectory-to-control converter.

Converts a sparse ego-centric trajectory into ``(throttle, steer, brake)``
using a lateral PID controller and a longitudinal linear-regression
controller ported from carla_garage/team_code. Used by ``ActionWrapper``
when the action space is of type ``trajectory``.

The controller is coordinate-system agnostic at the PID level: since the
planner outputs ego-centric waypoints every tick, the vehicle is always at
the origin with heading 0. The heading-error scalar that feeds into the
PID history is identical to what the world-frame formulation would produce.

When waypoint headings are available, they are used during temporal
interpolation to build a heading-aware dense path before feeding the
ported low-level controller.
"""

from typing import Dict, Any, Tuple, Optional

import numpy as np
from scipy.interpolate import CubicHermiteSpline, CubicSpline, interp1d

__layer__ = (2, "Environment")


class DrivePi0PID:
    """Small PID helper matching DrivePi0's official closed-loop agent."""

    def __init__(self, k_p: float = 1.0, k_i: float = 0.0, k_d: float = 0.0, n: int = 20):
        from collections import deque

        self.k_p = float(k_p)
        self.k_i = float(k_i)
        self.k_d = float(k_d)
        self.window = deque([0.0 for _ in range(int(n))], maxlen=int(n))
        self.error = 0.0
        self.integral = 0.0
        self.derivative = 0.0

    def reset(self) -> None:
        maxlen = self.window.maxlen or 20
        self.window.clear()
        self.window.extend([0.0 for _ in range(maxlen)])
        self.error = 0.0
        self.integral = 0.0
        self.derivative = 0.0

    def step(self, error: float) -> float:
        self.error = float(error)
        self.window.append(self.error)
        if len(self.window) >= 2:
            self.integral = float(np.mean(self.window))
            self.derivative = float(self.window[-1] - self.window[-2])
        else:
            self.integral = 0.0
            self.derivative = 0.0
        return self.k_p * self.error + self.k_i * self.integral + self.k_d * self.derivative


class DrivePi0PIDController:
    """DrivePi0 official trajectory-to-control PID port."""

    def __init__(self, cfg: Dict[str, Any]):
        self.brake_speed = float(cfg.get("brake_speed", 0.1))
        self.brake_ratio = float(cfg.get("brake_ratio", 1.1))
        self.clip_delta = float(cfg.get("clip_delta", 0.25))
        self.aim_distance_threshold = float(cfg.get("aim_distance_threshold", 5.5))
        self.aim_distance_slow = float(cfg.get("aim_distance_slow", 5.5))
        self.aim_distance_fast = float(cfg.get("aim_distance_fast", 8.5))
        self.clip_throttle = float(cfg.get("clip_throttle", 0.75))
        self.use_official_throttle_cap = bool(cfg.get("use_official_throttle_cap", True))
        self.turn_speed_threshold = float(cfg.get("turn_speed_threshold", 4.5))
        self.straight_speed_threshold = float(cfg.get("straight_speed_threshold", 7.0))
        self.turn_steer_threshold = float(cfg.get("turn_steer_threshold", 0.15))
        self.turn_controller = DrivePi0PID(
            k_p=cfg.get("turn_KP", 1.25),
            k_i=cfg.get("turn_KI", 0.75),
            k_d=cfg.get("turn_KD", 0.3),
            n=cfg.get("turn_n", 20),
        )
        self.speed_controller = DrivePi0PID(
            k_p=cfg.get("speed_KP", 5.0),
            k_i=cfg.get("speed_KI", 0.5),
            k_d=cfg.get("speed_KD", 1.0),
            n=cfg.get("speed_n", 20),
        )

    def reset(self) -> None:
        self.turn_controller.reset()
        self.speed_controller.reset()

    def control_pid(self, waypoints: np.ndarray, speed: float) -> Tuple[float, float, float]:
        waypoints = np.asarray(waypoints, dtype=np.float64)
        if waypoints.ndim != 2 or waypoints.shape[1] < 2:
            raise ValueError(f"DrivePi0 PID expects waypoints shaped (N, >=2), got {waypoints.shape}")
        if waypoints.shape[0] < 8:
            raise ValueError(f"DrivePi0 PID expects at least 8 waypoints, got {waypoints.shape[0]}")

        desired_speed = float(np.linalg.norm(waypoints[7, :2] - waypoints[0, :2]) * 2.0)
        brake = bool((desired_speed < self.brake_speed) or ((float(speed) / max(desired_speed, 1e-6)) > self.brake_ratio))
        delta = float(np.clip(desired_speed - float(speed), 0.0, self.clip_delta))
        throttle = self.speed_controller.step(delta)
        throttle = 0.0 if brake else throttle
        throttle = float(np.clip(throttle, 0.0, self.clip_throttle))

        aim_distance = self.aim_distance_slow if desired_speed < self.aim_distance_threshold else self.aim_distance_fast
        aim_index = waypoints.shape[0] - 1
        for index, predicted_waypoint in enumerate(waypoints[:, :2]):
            if np.linalg.norm(predicted_waypoint) >= aim_distance:
                aim_index = index
                break
        aim = waypoints[aim_index, :2]
        angle = float(np.degrees(np.arctan2(aim[1], aim[0])) / 90.0)
        if float(speed) < 0.01:
            angle = 0.0
        steer = float(np.clip(self.turn_controller.step(angle), -1.0, 1.0))

        if self.use_official_throttle_cap:
            speed_threshold = self.turn_speed_threshold if abs(steer) > self.turn_steer_threshold else self.straight_speed_threshold
            max_throttle = 0.05 if float(speed) > speed_threshold else self.clip_throttle
            throttle = float(np.clip(throttle, 0.0, max_throttle))
        if float(brake) > 0.5:
            throttle = 0.0
        return steer, throttle, float(brake)


class LateralPIDController:
    """Speed-dependent lookahead PID over dense route points.

    Ported from carla_garage/team_code/lateral_controller.py with
    behaviour-preserving parameter semantics.
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.kp: float = cfg.get('kp', 3.118357247806046)
        self.kd: float = cfg.get('kd', 1.3782508892109167)
        self.ki: float = cfg.get('ki', 0.6406067986034124)
        self.speed_scale: float = cfg.get('speed_scale', 0.9755321901954155)
        self.speed_offset: float = cfg.get('speed_offset', 1.9152884533402488)
        self.window_size: int = int(cfg.get('window_size', 6))
        self.min_lookahead: float = cfg.get('min_lookahead', 24.0)
        self.max_lookahead: float = cfg.get('max_lookahead', 105.0)

        self.error_history: list = []
        self._saved_error_history: list = []

    def reset(self):
        self.error_history.clear()

    def step(
        self,
        route_points: np.ndarray,
        current_speed_mps: float,
        vehicle_position: np.ndarray,
        vehicle_heading: float,
    ) -> float:
        """Return steering in [-1, 1]."""
        speed_kph = current_speed_mps * 3.6

        lookahead = self.speed_scale * speed_kph + self.speed_offset
        lookahead = np.clip(lookahead, self.min_lookahead, self.max_lookahead)
        idx = int(min(lookahead, route_points.shape[0] - 1))

        desired_vec = route_points[idx, :2] - vehicle_position[:2]
        desired_angle = np.arctan2(desired_vec[1], desired_vec[0])

        heading_error = (desired_angle - vehicle_heading) % (2 * np.pi)
        if heading_error >= np.pi:
            heading_error -= 2 * np.pi
        heading_error = heading_error * 180.0 / np.pi / 90.0

        self.error_history.append(heading_error)
        self.error_history = self.error_history[-self.window_size:]

        derivative = (
            0.0
            if len(self.error_history) == 1
            else self.error_history[-1] - self.error_history[-2]
        )
        integral = float(np.mean(self.error_history))

        steer = float(
            np.clip(
                self.kp * heading_error + self.kd * derivative + self.ki * integral,
                -1.0,
                1.0,
            )
        )
        return steer

    def save_state(self):
        self._saved_error_history = self.error_history.copy()

    def load_state(self):
        self.error_history = self._saved_error_history.copy()


class LongitudinalLinearRegressionController:
    """Stateless longitudinal controller.

    Ported from carla_garage/team_code/longitudinal_controller.py.
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.minimum_target_speed: float = cfg.get('minimum_target_speed', 0.278)
        self.params: np.ndarray = np.array(
            cfg.get(
                'params',
                [
                    1.1990342347353184,
                    -0.8057602384167799,
                    1.710818710950062,
                    0.921890257450335,
                    1.556497522998393,
                    -0.7013479734904027,
                    1.031266635497984,
                ],
            )
        )
        self.maximum_acceleration: float = cfg.get('maximum_acceleration', 1.89)
        self.maximum_deceleration: float = cfg.get('maximum_deceleration', -4.82)

    def get_throttle_and_brake(
        self,
        hazard_brake: bool,
        target_speed_mps: float,
        current_speed_mps: float,
    ) -> Tuple[float, bool]:
        if target_speed_mps < 1e-5 or hazard_brake:
            return 0.0, True

        target_speed_mps = max(target_speed_mps, self.minimum_target_speed)
        current_kph = current_speed_mps * 3.6
        target_kph = target_speed_mps * 3.6
        speed_error = target_kph - current_kph

        if speed_error > self.maximum_acceleration:
            return 1.0, False

        if current_kph / target_kph > self.params[-1] or hazard_brake:
            return 0.0, True

        e = max(speed_error, 0.0) / 100.0
        v = current_kph / 100.0
        features = np.array(
            [v, v * v, 100 * e, e * e, v * e, v * v * e]
        )
        throttle = float(np.clip(features @ self.params[:-1], 0.0, 1.0))
        return throttle, False


class TrajectoryController:
    """Converts an ego-centric trajectory to (throttle, steer, brake).

    Parameters are read from the ``controller`` sub-dict of the YAML
    ``action_space`` section so that the same controller can be reused by
    any planner (IL or RL) simply by switching ``action_space.type``.
    """

    def __init__(self, action_cfg: Dict[str, Any]):
        traj_cfg = action_cfg.get('trajectory', {})
        ctrl_cfg = action_cfg.get('controller', {})

        self.num_points: int = int(traj_cfg.get('num_points', 4))
        self.controller_type: str = str(ctrl_cfg.get('type', 'garage')).lower()
        self.state_dim: int = int(traj_cfg.get('state_dim', 3))
        self.horizon: float = float(traj_cfg.get('horizon', 2.0))
        self.freq: float = float(traj_cfg.get('freq', 2.0))

        self.interp_hz: float = float(ctrl_cfg.get('interp_hz', 10.0))
        self.points_per_meter: float = float(ctrl_cfg.get('points_per_meter', 10.0))
        self.speed_lookahead_points: int = int(ctrl_cfg.get('speed_lookahead_points', 3))
        self.steer_noise: float = float(ctrl_cfg.get('steer_noise', 0.0))
        self.use_heading_interpolation: bool = bool(
            ctrl_cfg.get('use_heading_interpolation', True)
        )

        lat_cfg = ctrl_cfg.get('lateral', {})
        lon_cfg = ctrl_cfg.get('longitudinal', {})

        self._drivepi0_pid = (
            DrivePi0PIDController(ctrl_cfg.get('drivepi0', {}) or {})
            if self.controller_type == 'drivepi0'
            else None
        )
        self._lateral = LateralPIDController(lat_cfg)
        self._longitudinal = LongitudinalLinearRegressionController(lon_cfg)

    def reset(self):
        """Call at the start of every episode."""
        if self._drivepi0_pid is not None:
            self._drivepi0_pid.reset()
        self._lateral.reset()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(
        self,
        trajectory: np.ndarray,
        ego_speed_mps: float,
    ) -> Tuple[float, float, float]:
        """Convert *trajectory* to ``(throttle, steer, brake)``.

        Args:
            trajectory: ``(N, 2|3)`` ego-centric waypoints (x-forward,
                y-left, [heading]).  ``N`` must equal ``self.num_points``.
            ego_speed_mps: current ego speed in m/s.

        Returns:
            ``(throttle, steer, brake)`` tuple ready for
            ``carla.VehicleControl``.
        """
        trajectory = np.asarray(trajectory, dtype=np.float64)
        if trajectory.ndim == 1:
            trajectory = trajectory.reshape(self.num_points, self.state_dim)

        if self._drivepi0_pid is not None:
            steer, throttle, brake = self._drivepi0_pid.control_pid(trajectory[:, :2], ego_speed_mps)
            return throttle, steer, brake

        interp_pts = self._temporal_interp(trajectory)
        dense_pts = self._spatial_densify(interp_pts)

        steer = self._lateral.step(
            route_points=dense_pts,
            current_speed_mps=ego_speed_mps,
            vehicle_position=np.zeros(2),
            vehicle_heading=0.0,
        )
        if self.steer_noise > 0:
            steer += self.steer_noise * np.random.randn()
            steer = float(np.clip(steer, -1.0, 1.0))

        target_speed = self._derive_target_speed(interp_pts)

        throttle, brake_flag = self._longitudinal.get_throttle_and_brake(
            hazard_brake=False,
            target_speed_mps=target_speed,
            current_speed_mps=ego_speed_mps,
        )

        return throttle, steer, float(brake_flag)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_tangent_speeds(pts: np.ndarray, t_src: np.ndarray) -> np.ndarray:
        """Estimate per-knot speed magnitudes for Hermite interpolation."""
        if len(pts) <= 1:
            return np.zeros(len(pts), dtype=np.float64)

        seg_dt = np.diff(t_src)
        seg_dist = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        seg_speed = np.divide(
            seg_dist,
            seg_dt,
            out=np.zeros_like(seg_dist),
            where=seg_dt > 1e-9,
        )

        tangent_speed = np.zeros(len(pts), dtype=np.float64)
        tangent_speed[0] = seg_speed[0]
        tangent_speed[-1] = seg_speed[-1]
        if len(pts) > 2:
            tangent_speed[1:-1] = 0.5 * (seg_speed[:-1] + seg_speed[1:])
        return tangent_speed

    def _temporal_interp(self, trajectory: np.ndarray) -> np.ndarray:
        """Interpolate sparse trajectory from ``freq`` Hz to ``interp_hz``."""
        xy = trajectory[:, :2]
        origin = np.zeros((1, 2))
        pts = np.vstack([origin, xy])

        dt_src = 1.0 / self.freq
        t_src = np.arange(len(pts)) * dt_src

        dt_dst = 1.0 / self.interp_hz
        t_dst = np.arange(0, t_src[-1] + 1e-9, dt_dst)

        if self.use_heading_interpolation and trajectory.shape[1] >= 3 and len(pts) >= 2:
            headings = np.concatenate([[0.0], trajectory[:, 2].astype(np.float64)])
            tangent_speed = self._estimate_tangent_speeds(pts, t_src)
            dx_dt = tangent_speed * np.cos(headings)
            dy_dt = tangent_speed * np.sin(headings)

            spline_x = CubicHermiteSpline(t_src, pts[:, 0], dx_dt)
            spline_y = CubicHermiteSpline(t_src, pts[:, 1], dy_dt)
            return np.stack([spline_x(t_dst), spline_y(t_dst)], axis=1)

        if len(pts) < 3:
            kind = min(len(pts) - 1, 1)
            interp_fn = interp1d(t_src, pts, axis=0, kind=kind, fill_value='extrapolate')
            return interp_fn(t_dst)

        cs = CubicSpline(t_src, pts, axis=0, bc_type='clamped')
        return cs(t_dst)

    def _spatial_densify(self, interp_pts: np.ndarray) -> np.ndarray:
        """Resample *interp_pts* at ``points_per_meter`` along arc-length."""
        diffs = np.diff(interp_pts, axis=0)
        seg_lengths = np.linalg.norm(diffs, axis=1)
        cum_length = np.concatenate([[0.0], np.cumsum(seg_lengths)])
        total_length = cum_length[-1]

        if total_length < 1e-6:
            return interp_pts

        n_dense = max(int(total_length * self.points_per_meter), 2)
        target_dists = np.linspace(0, total_length, n_dense)
        dense_x = np.interp(target_dists, cum_length, interp_pts[:, 0])
        dense_y = np.interp(target_dists, cum_length, interp_pts[:, 1])
        return np.stack([dense_x, dense_y], axis=1)

    def _derive_target_speed(self, interp_pts: np.ndarray) -> float:
        """Estimate target speed from the first segments of the 10 Hz path."""
        dt = 1.0 / self.interp_hz

        n = min(self.speed_lookahead_points, len(interp_pts) - 1)
        if n <= 0:
            return 0.0

        speeds = []
        for i in range(n):
            d = float(np.linalg.norm(interp_pts[i + 1] - interp_pts[i]))
            speeds.append(d / dt)
        return float(np.mean(speeds))
