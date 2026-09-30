# Environment Layer

The environment layer exposes CARLA route execution through a Gym-style interface:

```python
obs, info = env.reset()
next_obs, reward, terminated, truncated, info = env.step(action)
```

Internally, it composes CARLA runtime state, route tracking, observations, termination rules, rewards, and action decoding.

## Components

| Component | Description |
|---|---|
| `L2_CARLAEnv` | Core Gym environment that connects to a CARLA server, manages sensors, and implements `step()` / `reset()` / `render()`. |
| `L2_make_env` | Training env factory that composes the wrapper stack around `L2_CARLAEnv` for `CARLAEnvPool` workers. |
| `L2_build_observation_space` | Derives a `gym.spaces.Dict` from YAML so the wrapper view and the policy network always agree. |
| `L2_build_action_space` | Builds a discrete, continuous, or trajectory action space from the same YAML. |
| `L2_*Wrapper` | Modular wrappers — `ActionWrapper`, `ObservationWrapper`, `RewardWrapper`, `EventTerminationWrapper`, `RoutePlanWrapper`, `LQRExpertWrapper`. |
| `L2_BirdViewObsManager` | Rasterised BEV observation producer. |
| `L2_RGBSensorObsHandler` | Multi-camera CARLA RGB camera observation producer. |
| `L2_ScalarObsHandler` | Scalar feature vector (speed, steer, etc.) producer. |
| `L2_RewardHandler` | Multi-signal reward shaping. |

## Core environment

`b2d_rlinfra/environment/carla_env.py` defines `CARLAEnv`. It owns the CARLA connection, scenario manager, route state, watchdog, and low-level simulator calls.

The wrapper stack is defined in `b2d_rlinfra/environment/wrappers.py`:

| Wrapper | Role |
| --- | --- |
| `RoutePlanWrapper` | Maintains route-progress signals and route context |
| `ObservationWrapper` | Builds configured BEV, RGB camera, scalar, vehicle-state, and model-specific observations |
| `EventTerminationWrapper` | Converts events and rules into termination signals |
| `RewardWrapper` | Computes scalar reward through reward handlers |
| `ActionWrapper` | Decodes policy outputs into `carla.VehicleControl` |

## Observation space

Observation spaces are created from YAML by `b2d_rlinfra/environment/spaces.py`. The wrapper stack (`ObservationWrapper`) instantiates the matching handlers at runtime. Several observation branches can be enabled simultaneously; the resulting `gym.spaces.Dict` contains one key per active branch.

### BEV and scalar observations

The BEV baselines expose a `Dict` observation with two complementary branches:

| Branch | Contents |
| --- | --- |
| `vector` | Rasterized map, route, ego vehicle, surrounding actors, traffic lights, obstacles, lane markings, and stop signs |
| `scalars` | Ego states and previously executed low-level controls |

BEV masks are built directly from map data and CARLA actor state. The current presets use a 64 m square at 2 pixels per metre, producing a `128 x 128` mask. Dynamic actors use `[-16, -11, -6, -1]` history indices; at the configured 10 Hz policy frequency these correspond to approximately 1.5 s, 1.0 s, 0.5 s, and the current frame.

The following excerpt shows an example PPO layout:

```yaml
env:
  observation_space:
    vector:
      enable: true
      type: bev_mask
      map_dir: resources/maps
      mask_width: 64
      pixels_per_meter: 2
      history_index: [-16, -11, -6, -1]
    scalars:
      items:
        - type: speed
          use_history: true
        - type: throttle
          use_history: true
        - type: brake
          use_history: true
      history_index: [-16, -11, -6, -1]
```

`BirdViewObsManager` produces the `vector` branch, while `ScalarObsHandler` reads states and the actual control reported by `ego_actor.get_control()` for the `scalars` branch. 

### RGB camera observations

RGB camera observation configs attach multi-camera sensors to the ego vehicle. Each camera is specified with position, orientation, resolution, and field of view. The resulting camera branch is a stacked array of shape `(num_cameras, height, width, 3)`.

```yaml
env:
  observation_space:
    vector:
      enable: false
    rgb:
      enable: true
      output_key: rgb
      camera_order: [CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT, CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT]
      sensors:
        - type: sensor.camera.rgb
          x: 0.80
          y: 0.0
          z: 1.60
          width: 1600
          height: 900
          fov: 70
          id: CAM_FRONT
        # ... additional cameras
```

RGB camera observations are produced by `b2d_rlinfra/environment/handlers/rgb_sensor_obs_handler.py`. The branch is named `rgb` because it uses CARLA `sensor.camera.rgb` sensors; the runtime array is stored in CARLA / OpenCV-style BGR channel order unless an adapter converts it. Enabling RGB camera observations requires CARLA rendering to remain available, so the repository YAML settings `env.carla.no_rendering_mode` and `env.carla.null_rhi` must both be `false`. In the multi-worker pool, camera frames are transported between the worker process and the main process through per-worker shared-memory buffers (`b2d_rlinfra/simulation/runners/shm_rgb_buffer.py`).

### Model-specific observations

End-to-end models often require proprioceptive state beyond raw images, such as GPS coordinates, IMU readings, route context, or model-specific navigation tensors. The wrapper stack supports these through dedicated handlers:

| Handler | Output key | Purpose |
| --- | --- | --- |
| `GnssSensorObsHandler` / `ImuSensorObsHandler` / `SpeedometerSensorObsHandler` | configurable (e.g. `gnss`, `imu`) | GNSS / IMU / speedometer observation branches |
| `MindDriveStateObsHandler` | `minddrive_state` | Navigation geometry and camera projection matrices for MindDrive |
| `DrivePi0StateObsHandler` | `drivepi0_state` | Normalized proprioceptive state with temporal history for DrivePi0 |

These handlers are enabled in the YAML `observation_space` block with `enable: true` under their respective keys. They read from the shared sensor context published through `b2d_rlinfra/environment/handlers/sensor_context.py`.

## Action space

`b2d_rlinfra/environment/spaces.py` builds the Gym space, and `ActionWrapper` converts policy outputs to `carla.VehicleControl`:

| `type` | Policy output | Control semantics |
| --- | --- | --- |
| `discrete` | `Discrete(N)` | Index into `(throttle, steer, brake)` actions |
| `continuous_signed_pedal_steer` | `Box(2)` | `[signed_pedal, steer]` |
| `continuous_throttle_steer_brake` | `Box(3)` | Direct `[throttle, steer, brake]` |
| `trajectory` | `Box(num_points * state_dim)` | Ego-centric waypoints converted by a controller |

### Discrete

PPO/A2C examples select from `discrete_actions_list`; its length determines `N`. Each row uses `(throttle, steer, brake)` order, and `discrete_action_num` should match the list length.

### Continuous

A continuous example uses two bounded dimensions:

```yaml
type: continuous_signed_pedal_steer
action_dim: 2
continuous_actions_list:
  - [-1.0, 1.0]  # signed_pedal
  - [-1.0, 1.0]  # steer
```

```text
signed_pedal >= 0  -> throttle = signed_pedal, brake = 0
signed_pedal < 0   -> throttle = 0, brake = -signed_pedal
```

This keeps throttle and brake mutually exclusive. `continuous_throttle_steer_brake` instead exposes all three controls directly.

### Trajectory

Trajectory actions are flattened ego-centric waypoints. `state_dim=2` represents `(x forward, y left)`; `state_dim=3` also includes heading. The `garage` or `drivepi0` controller converts them to low-level control.

`action_repeat` repeats one decoded control, sums rewards, and returns the last observation. The executed `(throttle, steer, brake)` command is available in `info["executed_control"]`. Leaderboard evaluation shares the discrete and continuous decoders, but controls repetition through the simulator-to-policy frequency ratio.

## Reward and termination

Reward computation is intentionally isolated because reward shaping changes frequently across experiments.

Common customization points:

- `b2d_rlinfra/environment/handlers/reward_handler_ppo.py`
- `b2d_rlinfra/environment/handlers/reward_handler_a2c.py`
- `b2d_rlinfra/environment/handlers/reward_handler_sac.py`
- `b2d_rlinfra/environment/handlers/reward_handler_td3.py`
- `b2d_rlinfra/environment/handlers/simple_reward_handler.py`
- `b2d_rlinfra/environment/handlers/reward_handler_minddrive.py` — sparse reward variant for MindDrive finetune

Termination checks combine route completion, traffic infractions, blocked vehicle detection, route deviation, off-road behavior, and other CARLA/Leaderboard events.

Relevant modules:

- `b2d_rlinfra/environment/handlers/termination_handler_ppo.py`
- `b2d_rlinfra/environment/handlers/termination_handler_a2c.py`
- `b2d_rlinfra/environment/handlers/termination_handler_sac.py`
- `b2d_rlinfra/environment/handlers/termination_handler_td3.py`
- `b2d_rlinfra/environment/handlers/criteria/`

## Design note

The YAML file is the single source of truth for observation and action spaces. The same space construction is reused by wrappers, policies, training scripts, and evaluation runtime so the learner and evaluator stay aligned.
