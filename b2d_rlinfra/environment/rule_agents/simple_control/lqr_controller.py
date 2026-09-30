"""
LQR (Linear Quadratic Regulator) Controller for CARLA Autonomous Driving.

Uses a kinematic bicycle model linearized at a reference speed and solves the
discrete-time Algebraic Riccati Equation (DARE) for lateral and longitudinal
feedback gains.

The cost penalizes lateral, heading, speed, and lateral-rate errors together
with steering and acceleration effort.

Interface:
    controller = LQRController(action_space=env.action_space)
    obs, info = env.reset()
    controller.reset()
    action = controller(obs)
"""

import math
import logging
from collections import deque
from typing import Optional, Tuple

import numpy as np
from scipy import linalg

import carla
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

logger = logging.getLogger("LQRController")


DEFAULT_LQR_WEIGHTS = {
    # State cost weights Q = diag(q_lateral, q_heading, q_speed, q_lateral_rate)
    'q_lateral': 8.0,       # lateral deviation
    'q_heading': 5.0,       # heading error
    'q_speed': 3.0,         # speed tracking error
    'q_lateral_rate': 1.0,  # lateral-rate damping
    # Control cost weights R = diag(r_steer, r_accel)
    'r_steer': 2.0,         # steering effort
    'r_accel': 0.5,         # acceleration effort
}


def solve_dare(A, B, Q, R):
    """
    Solve the Discrete-time Algebraic Riccati Equation:
        P = A^T P A - A^T P B (R + B^T P B)^{-1} B^T P A + Q

    Returns:
        P: Solution matrix
        K: Optimal feedback gain K = (R + B^T P B)^{-1} B^T P A
    """
    try:
        P = linalg.solve_discrete_are(A, B, Q, R)
    except (linalg.LinAlgError, ValueError) as e:
        logger.warning(f"DARE solve failed ({e}), using LQR fallback with identity P")
        P = np.eye(Q.shape[0]) * 10.0

    BtP = B.T @ P
    K = np.linalg.solve(R + BtP @ B, BtP @ A)
    return P, K


class KinematicBicycleModelLQR:
    """
    Simplified kinematic bicycle model for LQR linearisation.

    State vector x = [e_lat, e_heading, e_speed, de_lat/dt]
        e_lat      : lateral error (m) from route centerline
        e_heading  : heading error (rad) from route tangent
        e_speed    : speed error (m/s) = ego_speed - desired_speed
        de_lat/dt  : lateral error rate (m/s)

    Control vector u = [steer_cmd, accel_cmd]
        steer_cmd : steering command
        accel_cmd : longitudinal acceleration command
    """

    def __init__(self, wheelbase: float = 2.9, dt: float = 0.1):
        """
        Args:
            wheelbase: Vehicle wheelbase in metres (L_f + L_r).
            dt: Control timestep in seconds.
        """
        self.L = wheelbase
        self.dt = dt

    def linearize(self, v_ref: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Linearize the bicycle model about a reference speed v_ref.

        Returns:
            A: (4, 4) state transition matrix
            B: (4, 2) control input matrix
        """
        v = max(v_ref, 0.5)  # avoid singularity at zero speed
        dt = self.dt
        L = self.L

        # State: [e_lat, e_heading, e_speed, de_lat_dt]
        # Control: [steer, accel]

        # Continuous-time A, B from bicycle kinematics:
        # d(e_lat)/dt     = v * sin(e_heading) ≈ v * e_heading
        # d(e_heading)/dt = v / L * tan(delta) - kappa * v ≈ v/L * delta  (kappa≈0)
        # d(e_speed)/dt   = a_accel
        # d(de_lat)/dt    = v * e_heading_dot ≈ v * (v/L) * delta

        # Discretize with Euler:
        A = np.array([
            [1.0,  v * dt,  0.0,  0.0],
            [0.0,  1.0,     0.0,  0.0],
            [0.0,  0.0,     1.0,  0.0],
            [0.0,  v * dt,  0.0,  1.0],
        ])

        B = np.array([
            [0.0,            0.0],
            [v / L * dt,     0.0],
            [0.0,            dt],
            [v**2 / L * dt,  0.0],
        ])

        return A, B


class LQRController:
    """
    LQR-based autonomous driving controller for CARLA.

    Reads route and ego state directly from CarlaDataProvider (privileged access,
    same pattern as PDMLiteExpert) and outputs [throttle, steer, brake] actions.

    Q penalizes state-tracking errors; R penalizes steering and acceleration
    effort.
    """

    # Speed control constants
    MAX_THROTTLE = 0.8
    MAX_BRAKE = 1.0
    MAX_STEER = 1.0
    SAFETY_DISTANCE = 10.0        # m, target standstill gap
    FRONT_ACTOR_THRESHOLD = 15.0  # m, detection range for front actors
    JUNCTION_SPEED_FACTOR = 0.6   # slow down ratio in junctions

    def __init__(self, action_space=None, config: Optional[dict] = None,
                 visualize: bool = False):
        """
        Args:
            action_space: Gym action space (compatibility, not strictly needed).
            config: Optional dict with LQR weight overrides and vehicle params.
            visualize: Draw debug info in CARLA world.
        """
        self.action_space = action_space
        self.visualize = visualize
        self.config = config or {}

        # Vehicle parameters
        self.wheelbase = float(self.config.get('wheelbase', 2.9))
        self.dt = float(self.config.get('dt', 0.1))
        self.max_speed = float(self.config.get('max_speed', 20.0))  # m/s ≈ 72 km/h

        # LQR weights
        weights = self.config.get('lqr_weights', DEFAULT_LQR_WEIGHTS)
        self.Q = np.diag([
            float(weights.get('q_lateral', DEFAULT_LQR_WEIGHTS['q_lateral'])),
            float(weights.get('q_heading', DEFAULT_LQR_WEIGHTS['q_heading'])),
            float(weights.get('q_speed', DEFAULT_LQR_WEIGHTS['q_speed'])),
            float(weights.get('q_lateral_rate', DEFAULT_LQR_WEIGHTS['q_lateral_rate'])),
        ])
        self.R = np.diag([
            float(weights.get('r_steer', DEFAULT_LQR_WEIGHTS['r_steer'])),
            float(weights.get('r_accel', DEFAULT_LQR_WEIGHTS['r_accel'])),
        ])

        # Models
        self.bicycle_model = KinematicBicycleModelLQR(
            wheelbase=self.wheelbase, dt=self.dt
        )

        # State
        self.initialized = False
        self.step_count = -1
        self._vehicle = None
        self._world = None
        self._map = None

        # History for smoothing
        self._prev_steer = 0.0
        self._prev_lateral_error = 0.0
        self._steer_history = deque(maxlen=5)
        self._K_cache = None
        self._K_cache_speed = -1.0

        # Speed planning
        self._list_traffic_lights = []
        self._list_stop_signs = []

    def reset(self):
        """Reset the controller for a new episode."""
        self.initialized = False
        self.step_count = -1
        self._vehicle = None
        self._world = None
        self._map = None
        self._prev_steer = 0.0
        self._prev_lateral_error = 0.0
        self._steer_history.clear()
        self._K_cache = None
        self._K_cache_speed = -1.0
        self._list_traffic_lights = []
        self._list_stop_signs = []

    def __call__(self, obs, **kwargs) -> np.ndarray:
        """
        Compute LQR control action from current world state.

        Args:
            obs: Observation from environment (not used directly; reads
                 privileged state from CarlaDataProvider).

        Returns:
            np.ndarray: [throttle, steer, brake] of shape (3,).
        """
        self.step_count += 1

        if not self.initialized:
            self._init()

        control = self._compute_control()
        return np.array([control.throttle, control.steer, control.brake],
                        dtype=np.float32)

    def _init(self):
        """Lazy initialisation once the CARLA world is ready."""
        self._vehicle = CarlaDataProvider.get_hero_actor()
        if self._vehicle is None:
            self._vehicle = CarlaDataProvider._ego_actor
        if self._vehicle is None:
            raise RuntimeError("LQRController._init: ego vehicle not found")

        self._world = self._vehicle.get_world()
        self._map = CarlaDataProvider.get_map()

        # Cache traffic lights and stop signs
        for actor in self._world.get_actors():
            if 'traffic_light' in actor.type_id:
                self._list_traffic_lights.append(actor)
            elif 'traffic.stop' in actor.type_id:
                self._list_stop_signs.append(actor)

        # Estimate wheelbase from the bounding box.
        extent = self._vehicle.bounding_box.extent
        self.wheelbase = max(2.0 * extent.x * 0.8, 2.0)  # approximate
        self.bicycle_model.L = self.wheelbase

        self.initialized = True
        logger.info(f"LQRController initialised: wheelbase={self.wheelbase:.2f}m")

    def _compute_control(self) -> 'carla.VehicleControl':
        """Compute the vehicle control command using LQR."""
        # Get ego state
        ego_transform = self._vehicle.get_transform()
        ego_location = ego_transform.location
        ego_rotation = ego_transform.rotation
        ego_velocity = self._vehicle.get_velocity()
        ego_speed = math.sqrt(ego_velocity.x**2 + ego_velocity.y**2)
        ego_heading = math.radians(ego_rotation.yaw)

        # Get route info
        route = CarlaDataProvider._ego_vehicle_route
        if not route or len(route) < 2:
            # No route, just brake
            return self._make_control(0.0, 0.0, 1.0)

        # Find reference point on route
        ref_transform, ref_heading, curvature = self._get_reference_state(
            ego_location, route
        )

        # Compute error state
        e_lat = self._compute_lateral_error(
            ego_location, ref_transform, ref_heading
        )
        e_heading = self._normalize_angle(ego_heading - ref_heading)

        # Compute desired speed
        desired_speed = self._compute_desired_speed(
            ego_location, ego_speed, route, curvature
        )
        e_speed = ego_speed - desired_speed

        # Lateral error rate
        de_lat = (e_lat - self._prev_lateral_error) / max(self.dt, 1e-3)
        self._prev_lateral_error = e_lat

        # State vector
        x = np.array([e_lat, e_heading, e_speed, de_lat])

        # Get LQR gain
        K = self._get_lqr_gain(max(ego_speed, desired_speed))

        # Optimal control: u* = -K x
        u = -K @ x
        steer_cmd = u[0]
        accel_cmd = u[1]

        # Convert to throttle/brake
        steer = self._smooth_steer(float(np.clip(steer_cmd, -self.MAX_STEER, self.MAX_STEER)))

        if accel_cmd >= 0:
            throttle = float(np.clip(accel_cmd / 3.0, 0.0, self.MAX_THROTTLE))
            brake = 0.0
        else:
            throttle = 0.0
            brake = float(np.clip(-accel_cmd / 8.0, 0.0, self.MAX_BRAKE))

        # Safety: if too close to an obstacle or at red light, force brake
        if desired_speed < 0.1 and ego_speed > 0.5:
            throttle = 0.0
            brake = min(1.0, ego_speed / 5.0)

        # Hold the brake near standstill.
        if throttle < 0.01 and ego_speed < 0.3:
            brake = 1.0

        if self.visualize and self._world is not None:
            self._draw_debug(ego_location, ref_transform, route)

        return self._make_control(throttle, steer, brake)

    def _get_lqr_gain(self, v_ref: float) -> np.ndarray:
        """
        Get or recompute LQR gain K for the given reference speed.
        Uses caching to avoid solving DARE every tick.
        """
        # Recompute if speed changed significantly (> 1 m/s)
        if self._K_cache is not None and abs(v_ref - self._K_cache_speed) < 1.0:
            return self._K_cache

        A, B = self.bicycle_model.linearize(v_ref)
        _, K = solve_dare(A, B, self.Q, self.R)

        self._K_cache = K
        self._K_cache_speed = v_ref
        return K

    def _get_reference_state(self, ego_location, route):
        """
        Find the closest route point and compute the reference heading
        and local curvature.

        Returns:
            ref_transform: Transform of the closest route point
            ref_heading: Reference heading angle (rad)
            curvature: Estimated local curvature (1/m)
        """
        # Find closest point
        min_dist = float('inf')
        closest_idx = 0
        route_len = len(route)

        search_range = min(route_len, 50)
        for i in range(search_range):
            loc = route[i][0].location
            d = math.sqrt(
                (ego_location.x - loc.x)**2 + (ego_location.y - loc.y)**2
            )
            if d < min_dist:
                min_dist = d
                closest_idx = i

        ref_transform = route[closest_idx][0]

        # Compute heading from current to next route point
        if closest_idx + 1 < route_len:
            loc0 = route[closest_idx][0].location
            loc1 = route[closest_idx + 1][0].location
            dx = loc1.x - loc0.x
            dy = loc1.y - loc0.y
            if dx**2 + dy**2 > 0.01:
                ref_heading = math.atan2(dy, dx)
            else:
                ref_heading = math.radians(ref_transform.rotation.yaw)
        else:
            ref_heading = math.radians(ref_transform.rotation.yaw)

        # Estimate curvature using 3 points
        curvature = 0.0
        if closest_idx + 2 < route_len and closest_idx > 0:
            p0 = route[max(0, closest_idx - 1)][0].location
            p1 = route[closest_idx][0].location
            p2 = route[min(closest_idx + 2, route_len - 1)][0].location
            curvature = self._compute_curvature(
                (p0.x, p0.y), (p1.x, p1.y), (p2.x, p2.y)
            )

        return ref_transform, ref_heading, curvature

    def _compute_lateral_error(self, ego_location, ref_transform, ref_heading):
        """
        Compute signed lateral error from the ego to the route centerline.
        Positive = ego is to the right of the route direction in CARLA coordinates.
        """
        dx = ego_location.x - ref_transform.location.x
        dy = ego_location.y - ref_transform.location.y
        # Cross product with route forward direction gives signed lateral offset
        e_lat = -math.sin(ref_heading) * dx + math.cos(ref_heading) * dy
        return e_lat

    @staticmethod
    def _compute_curvature(p0, p1, p2):
        """Compute curvature through three 2D points using Menger curvature."""
        ax, ay = p0
        bx, by = p1
        cx, cy = p2
        # Twice the signed area of the triangle
        cross = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
        d01 = math.sqrt((bx - ax)**2 + (by - ay)**2)
        d12 = math.sqrt((cx - bx)**2 + (cy - by)**2)
        d02 = math.sqrt((cx - ax)**2 + (cy - ay)**2)
        denom = d01 * d12 * d02
        if denom < 1e-6:
            return 0.0
        return 2.0 * cross / denom

    @staticmethod
    def _normalize_angle(angle):
        """Normalize angle to [-pi, pi]."""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def _compute_desired_speed(self, ego_location, ego_speed, route, curvature):
        """
        Compute desired speed considering:
          - Speed limit
          - Curvature (slow down in curves)
          - Traffic lights (stop at red)
          - Front actors (follow at safe distance)
          - Junction (reduce speed)

        Inspired by RewardHandler.get_desired_speed()
        """
        if len(route) < 2:
            return 0.0

        # Base desired speed from speed limit
        speed_limit = self._vehicle.get_speed_limit() / 3.6  # convert km/h → m/s
        desired_speed = min(speed_limit, self.max_speed)

        # Curvature-based speed reduction
        if abs(curvature) > 0.01:
            # v_max = sqrt(a_lat_max / |kappa|), lateral comfort limit ≈ 3 m/s²
            curvature_speed = math.sqrt(3.0 / max(abs(curvature), 0.01))
            desired_speed = min(desired_speed, curvature_speed)

        # Junction speed reduction
        ego_wp = self._map.get_waypoint(ego_location)
        route_wp = self._map.get_waypoint(route[0][0].location)
        if ego_wp.is_junction or route_wp.is_junction:
            desired_speed = min(desired_speed, speed_limit * self.JUNCTION_SPEED_FACTOR)

        # Red traffic light detection
        red_light_speed = self._check_traffic_lights(ego_location, route)
        if red_light_speed is not None:
            desired_speed = min(desired_speed, red_light_speed)

        # Front actor detection (IDM-like simple version)
        front_speed = self._check_front_actors(ego_location, ego_speed, route)
        if front_speed is not None:
            desired_speed = min(desired_speed, front_speed)

        return max(desired_speed, 0.0)

    def _check_traffic_lights(self, ego_location, route):
        """Check for a nearby red light along the sampled route points."""
        for tl in self._list_traffic_lights:
            if not tl.is_alive:
                continue
            if tl.state != carla.TrafficLightState.Red:
                continue

            tl_location = tl.get_location()
            dist = math.sqrt(
                (ego_location.x - tl_location.x)**2 +
                (ego_location.y - tl_location.y)**2
            )

            if dist > 50.0:
                continue

            # Check if the light is on our route (within ~15m of a route point)
            for i in range(min(len(route), 30)):
                r_loc = route[i][0].location
                d = math.sqrt(
                    (tl_location.x - r_loc.x)**2 +
                    (tl_location.y - r_loc.y)**2
                )
                if d < 15.0:
                    # Decelerate linearly as we approach
                    stop_distance = max(dist - 5.0, 0.0)
                    target = (stop_distance / 30.0) * self.max_speed
                    return max(target, 0.0)

        return None

    def _check_front_actors(self, ego_location, ego_speed, route):
        """Simple front-actor distance-based speed planning."""
        if self._world is None:
            return None

        ego_wp = self._map.get_waypoint(ego_location)
        ego_forward = self._vehicle.get_transform().get_forward_vector()

        for actor in self._world.get_actors():
            if not actor.is_alive:
                continue
            if actor.id == self._vehicle.id:
                continue
            if not (actor.type_id.startswith('vehicle') or actor.type_id.startswith('walker')):
                continue

            a_loc = actor.get_location()
            dx = a_loc.x - ego_location.x
            dy = a_loc.y - ego_location.y
            dist = math.sqrt(dx**2 + dy**2)

            if dist > self.FRONT_ACTOR_THRESHOLD:
                continue

            # Check if actor is in front (dot product with ego forward)
            dot = ego_forward.x * dx + ego_forward.y * dy
            if dot <= 0:
                continue

            # Check same road/lane
            a_wp = self._map.get_waypoint(a_loc)
            if a_wp.road_id != ego_wp.road_id or a_wp.lane_id != ego_wp.lane_id:
                continue

            # Scale target speed linearly above the standstill gap.
            distance_span = max(self.FRONT_ACTOR_THRESHOLD - self.SAFETY_DISTANCE, 1.0)
            scale = np.clip(dist - self.SAFETY_DISTANCE, 0.0, distance_span)
            target_speed = (scale / distance_span) * self.max_speed
            return target_speed

        return None

    def _smooth_steer(self, raw_steer: float) -> float:
        """Apply exponential smoothing to reduce jitter."""
        alpha = 0.3  # smoothing factor (0=max smooth, 1=no smooth)
        smoothed = alpha * raw_steer + (1.0 - alpha) * self._prev_steer
        self._prev_steer = smoothed
        self._steer_history.append(smoothed)
        return float(np.clip(smoothed, -self.MAX_STEER, self.MAX_STEER))

    @staticmethod
    def _make_control(throttle, steer, brake):
        """Create a carla.VehicleControl."""
        control = carla.VehicleControl()
        control.throttle = float(np.clip(throttle, 0.0, 1.0))
        control.steer = float(np.clip(steer, -1.0, 1.0))
        control.brake = float(np.clip(brake, 0.0, 1.0))
        control.hand_brake = False
        control.manual_gear_shift = False
        return control

    def _draw_debug(self, ego_location, ref_transform, route):
        """Draw debug visualisation in the CARLA world."""
        life_time = 0.1

        # Draw reference point
        ref_loc = carla.Location(
            ref_transform.location.x,
            ref_transform.location.y,
            ref_transform.location.z + 0.5
        )
        self._world.debug.draw_point(
            location=ref_loc, size=0.15,
            color=carla.Color(255, 0, 0),
            life_time=life_time
        )

        # Draw future route
        for i in range(min(len(route), 30)):
            loc = route[i][0].location
            draw_loc = carla.Location(loc.x, loc.y, loc.z + 0.3)
            self._world.debug.draw_point(
                location=draw_loc, size=0.08,
                color=carla.Color(0, 255, 0),
                life_time=life_time
            )

        # Draw lateral error line
        self._world.debug.draw_line(
            ego_location, ref_loc,
            thickness=0.05,
            color=carla.Color(255, 255, 0),
            life_time=life_time
        )
