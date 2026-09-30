# Troubleshooting

This page lists common setup and runtime issues.

## ModuleNotFoundError: carla

Check `CARLA_ROOT` and `PYTHONPATH`.

```bash
export CARLA_ROOT=/path/to/carla
export PYTHONPATH="$CARLA_ROOT/PythonAPI/carla:${PYTHONPATH:-}"
```

The exact CARLA PythonAPI layout depends on how CARLA was installed.

## CARLA servers do not become ready

Check:

- `env.carla.port` and `env.carla.traffic_manager_port` are free.
- `CARLA_ROOT` points to a valid CARLA 0.9.15 installation.
- The selected GPU IDs exist.
- The host list length matches `env.carla.num_envs`.
- `server_wait_timeout` is long enough for the machine and map.

## Port or process conflicts

Clean stale CARLA processes before relaunching:

```bash
bash tools/runtime/kill_by_host.sh 127.0.1.135
```

Replace the host with the value from `env.carla.host` for the worker you want to clean.

If shared-memory resources are left behind after a forced stop:

```bash
bash tools/runtime/clean_shm.sh
```

## Low throughput

Try:

- Lowering `num_envs` if the machine is overloaded.
- Increasing `num_envs` if the GPU is idle and CPU memory is sufficient.
- Tuning `training.adapter.min_ready`.
- Increasing episode length to amortize reset cost.
- Disabling visualization for throughput-only runs.
- Using offscreen or null-RHI CARLA settings when rendering is unnecessary.

## Training crashes after a simulator failure

The pool is designed to recover from worker-level failures, but repeated crashes usually indicate a setup issue:

- invalid route XML,
- missing map,
- bad CARLA binary path,
- port collision,
- insufficient timeout,
- incompatible Leaderboard/ScenarioRunner root.

Check the structured crash records under the configured `env.environment.result_dir`. For finetuning runs, check `<run_dir>/crash_events/`; finetune collectors bind environment artifacts to the current run directory.

## CARLA worker crashes during training

Some routes or scenarios may occasionally trigger CARLA crashes during long multi-worker jobs. This is expected — CARLA server itself is not perfectly stable. The training infrastructure automatically restarts the affected worker and server, marks the crashed transition in `info["crashed"]`, and keeps the remaining workers running.

Occasional isolated crashes are normal and do not require intervention. If many workers crash at the same time, the machine usually cannot support the current worker count; reduce `env.carla.num_envs` and the matching host/port/GPU lists.

## Initial CARLA server failures during RL finetuning

During initial startup, it is normal for multiple CARLA servers to fail on their first attempt because concurrent launches can create a brief peak in resource demand. The collector retries the failed servers one by one, so a correctly configured run may take several retry cycles before all workers are ready. Investigate only if a server keeps failing and never becomes ready; possible causes include an invalid CARLA path, a port conflict, a Vulkan/graphics-adapter problem, or an insufficient startup timeout.

For a distributed run, inspect `nodes/node_<node_rank>.json` and `logs/nodes/node_<node_rank>.out` first. Preflight records the resolved CUDA-to-Vulkan mapping and fails the run before collection if it cannot prove a safe mapping. See [Distributed RL Finetune](rl_finetune_distributed.md#troubleshooting).

## Stale shared memory after RGB training

RGB camera training uses `/dev/shm` for inter-process image transport. If a job is killed without cleanup, shared-memory files may be left behind. Clear them before relaunching:

```bash
bash tools/runtime/clean_shm.sh
```

For a distributed run, use the run-scoped cleanup helper:

```bash
bash tools/slurm/cleanup_rl_finetune_job.sh --run-dir <run_dir>
bash tools/slurm/cleanup_rl_finetune_job.sh --run-dir <run_dir> --force
```

## Evaluation config not found

`tools/launch/evaluation/run_leaderboard_eval_parallel.sh` contains placeholder paths by default:

```bash
CONFIG_PATH="/path/to/eval_config.yaml"
CHECKPOINT_PATH="/path/to/checkpoint_dir"
ROUTES_FILE="/path/to/routes/eval_routes.xml"
```

Edit these before running evaluation.
