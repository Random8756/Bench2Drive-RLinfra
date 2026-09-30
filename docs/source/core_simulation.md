# Simulation Layer

The simulation layer provides high-throughput CARLA rollout through `CARLAEnvPool`.

CARLA is expensive compared with lightweight RL environments: resets can take many seconds, route durations are heterogeneous, and long-running jobs can encounter simulator or traffic-manager failures. The simulation layer keeps these realities out of the algorithm code.

## Components

| Component | Description |
|---|---|
| `L3_CARLAEnvPool` | Async env worker pool. Each worker runs in its own process; the pool exposes a unified `step()` / `reset()` returning worker-indexed dictionaries. |
| `L3_CARLAServerManager` | Manages CARLA server processes: port allocation, health checks, and automatic restart on crash. |
| `L3_CARLAConnection` | Single-server connection helper with retry and timeout logic. |
| `L3_HealthWorker` | Background health-check worker that periodically checks environment worker state and triggers recovery for abnormal workers. |

For a hands-on demonstration of pool lifecycle without training code, see the [Async Pool Tutorial](core_pool_tutorial.md).

## `CARLAEnvPool`

`b2d_rlinfra/simulation/runners/carla_env_pool.py` defines `CARLAEnvPool`, an asynchronous worker pool with:

- One process per CARLA worker.
- Worker-indexed `reset()` and `step()` outputs.
- Configurable `min_ready` stepping.
- CARLA server launch and health checks.
- Fault-tolerant restart and structured crash information.
- Auto-reset support for terminated episodes.

## Readiness-based stepping

CARLA workers do not progress at identical speeds. Routes have different lengths, reset cost can be high, and a worker may need to restart after a simulator failure. A synchronous vectorized environment waits for the slowest worker at every step.

The adapter calls the pool with a minimum number of ready workers:

```python
next_obs, rewards, terminated, truncated, infos = env.step(actions, min_ready=4)
```

The pool returns once at least `min_ready` workers have produced transitions. Resetting or recovering workers temporarily leave the ready set while the rest of the rollout continues. This avoids waiting for every worker on each step, so a slow reset, map load, or recovering worker does not block all other workers.

`step()` also takes a `timeout`. If it expires before `min_ready` workers are ready, the call returns the results collected so far — possibly fewer than `min_ready`, or none. Consumers should treat the returned key set as authoritative rather than assuming a fixed batch size.

## CARLA worker configuration

Worker slots are configured in YAML:

```yaml
env:
  carla:
    num_envs: 24
    host:
      - 127.0.3.135
    port:
      - 2026
    traffic_manager_port:
      - 8202
    gpu_id:
      - 0
    frequency_hz: 10
    render_offscreen: true
    null_rhi: true
```

For `num_envs > 1`, the host, port, traffic-manager port, and GPU lists should have enough entries for every worker.

## RGB camera shared-memory transport

When RGB camera observations are enabled, each CARLA worker produces multi-camera images that need to reach the main process for policy inference. Copying large image arrays through multiprocessing pipes or queues is expensive, so the pool uses per-worker shared-memory buffers backed by `/dev/shm` (`b2d_rlinfra/simulation/runners/shm_rgb_buffer.py`).

Each `ShmRgbBuffer` is a fixed-slot ring buffer: the worker writes the latest RGB camera observation into the next slot, and the main process reads from the most recent slot. This avoids serialization overhead and keeps memory usage bounded. The pool automatically creates and cleans up these buffers based on the observation space.

The IPC mode can be controlled with `env.ipc.use_rgb_shm` (default `auto`): when RGB camera observations are detected in the config, shared-memory transport is activated automatically.

## Rendering modes

The `env.carla.*` entries below are Bench2Drive-RLInfra YAML keys, not CARLA official config names. They map to CARLA launch arguments or runtime world settings:

| Setting | Repository YAML key | Effect |
| --- | --- | --- |
| `render_offscreen` | `env.carla.render_offscreen` | Runs CARLA without a visible window while keeping rendering available for camera / GPU sensors. |
| `null_rhi` | `env.carla.null_rhi` | Starts CARLA without render capability. Good for non-visual BEV training, not for RGB camera training. |
| `no_rendering_mode` | `env.carla.no_rendering_mode` | Runtime CARLA world setting that makes GPU / camera sensor data unavailable or empty. Do not use it for RGB camera training. |

For BEV-only training, `null_rhi: true` and `no_rendering_mode: true` give the best performance. For RGB camera or end-to-end training, both must be `false`. The pool validates this at startup when RGB camera observations are enabled.

## Server management

`CARLAServerManager` handles CARLA server processes and launch options such as offscreen rendering, sound, RHI mode, and extra Unreal arguments.

Relevant files:

- `b2d_rlinfra/simulation/runners/carla_server_manager.py`
- `b2d_rlinfra/simulation/carla_connection.py`
- `b2d_rlinfra/simulation/runners/shm_rgb_buffer.py`
- `tools/fake_bind.c`
- `tools/runtime/kill_by_host.sh`
- `tools/runtime/clean_shm.sh`

## Auto-reset and terminal observation ownership

When an episode terminates, the pool automatically resets the worker for a new episode. The terminal observation (the last `obs` returned alongside `terminated=True`) belongs to the **old** episode. The first observation of the new episode arrives as a separate reset observation in a subsequent step. Algorithm code should not treat the terminal observation as the start of a new episode.

## Crash handling

Worker crashes are treated as recoverable events. The pool records route and scenario context, marks the affected transition or episode, restarts the worker or server, and reinserts it into the rollout pool.

Algorithm code should discard data marked as crashed (`info["crashed"]`). This prevents corrupted simulator episodes from entering rollout or replay buffers.

## Tuning `min_ready`

`min_ready` trades off policy inference efficiency and waiting time. Very small values reduce waiting but may create inefficient tiny inference batches. Very large values approach synchronous vectorized stepping and can wait for the slowest worker.

The paper reports that, in the PPO BEV pipeline with 24 workers, an intermediate `min_ready` value was the fastest setting.
