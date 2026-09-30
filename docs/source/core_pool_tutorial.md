# Async Pool Tutorial

This page walks through `b2d_rlinfra/framework/env_rollout_demo.py` — a runnable tutorial that drives the asynchronous CARLA environment pool with fixed actions. No policy network or training is involved; the demo exists to illustrate the `CARLAEnvPool` lifecycle and the semantics that distinguish it from standard Gym/VecEnv.

## Run the demo

```bash
bash b2d_rlinfra/framework/env_rollout_demo.sh
```

Before running, edit `b2d_rlinfra/framework/env_rollout_demo.sh` to set `CARLA_ROOT` to your CARLA installation path, and verify that the port/host entries in `configs/env_rollout_demo.yaml` match your CARLA server setup.

The demo config (`configs/env_rollout_demo.yaml`) contains two top-level blocks:

- `env` — standard environment configuration (workers, ports, observations, actions, routes).
- `demo` — demo-specific settings: `min_ready`, total `steps`, timeouts, and the fixed-action agent mode.

## What the demo shows

The demo runs three phases:

### Phase 1 — Pool startup and initial reset

```python
pool = CARLAEnvPool(env_fn=make_demo_env, config=..., num_envs=4, auto_reset=True)
reset_obs, reset_infos = pool.reset(min_ready=1, timeout=120)
```

`pool.reset()` returns `{worker_id: observation}` — a dictionary keyed by integer worker IDs. Unlike VecEnv, this is not a fixed-length array. Workers that are still loading their CARLA world may not appear yet.

### Phase 2 — Async rollout loop

Each iteration:

1. **Dispatch** — send actions to workers that have a ready observation.
2. **Collect** — call `pool.step(actions, min_ready=k)` to drain results from whichever workers are ready.
3. **Process** — handle normal transitions, terminal observations, crash events, and reset observations.

```python
obs_dict, rewards, terms, truncs, infos = pool.step(actions, min_ready=1)
```

The returned dictionaries have **varying key sets** — not every worker appears in every call. A worker that is resetting or recovering will not have a reward entry.

### Phase 3 — Summary

The demo reports committed episodes (cleanly terminated) and discarded episodes (crashed mid-episode).

## Key lifecycle semantics

| Concept | Behavior |
|---|---|
| Worker-indexed dicts | All pool I/O uses `{worker_id: value}` dicts with varying key sets — not fixed-length arrays. |
| Terminal observation ownership | The observation returned with `terminated=True` belongs to the **old** episode. Do not reuse it as the start of a new episode. |
| Auto-reset observation | After a terminal, the worker automatically resets. The first observation of the new episode arrives as a separate entry in a later `pool.step()` call, identified by `info["from_reset"]` or `info["episode_start"]`. |
| Crash → discard | If `info["crashed"]` is set, no valid observation exists. The entire in-progress episode must be discarded. The worker restarts and eventually emits a new reset observation. |
| Empty dispatch (poll) | `pool.step({})` is valid — it dispatches no new actions but drains any queued results from workers that finished earlier. |
| `min_ready` | The pool blocks until at least `min_ready` workers have produced results, then returns immediately. Workers that are resetting, loading, or recovering do not count toward readiness. |
| Step timeout | `pool.step()` also takes a `timeout` (seconds). If it expires before `min_ready` workers are ready — for example when all workers happen to be resetting — the call returns whatever is available, possibly fewer entries than `min_ready` or an empty batch. Always iterate over the returned keys instead of assuming the batch size. |

## How the demo agent works

The demo uses a trivial agent that selects either a fixed discrete action index or a random index:

```yaml
demo:
  action:
    mode: fixed   # or "random"
    index: 8      # discrete action index (throttle=0.3, steer=0.0)
```

This lets you observe pool behavior without any policy interference.

## Connection to training code

In real training, the algorithm code (e.g. `b2d_rlinfra/learning/algorithms/ppo.py`) handles dispatch, terminal observation ownership, and crash-discard on top of the pool. `StandardEnvAdapter` (in `b2d_rlinfra/learning/adapters/standard_adapter.py`) is a thin wrapper that passes `reset()`/`step()` through to the pool and extracts the configured observation branch — it does not add lifecycle logic itself. The demo exposes the same pool-level protocol so you can see exactly what the algorithm consumes.

## Relevant files

- `b2d_rlinfra/framework/env_rollout_demo.py` — the demo script
- `b2d_rlinfra/framework/env_rollout_demo.sh` — launch wrapper
- `b2d_rlinfra/framework/demo_utils.py` — config parsing, toy buffer, console output
- `configs/env_rollout_demo.yaml` — demo configuration
