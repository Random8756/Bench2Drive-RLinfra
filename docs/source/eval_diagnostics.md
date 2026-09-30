# Evaluation and Diagnostics

Bench2Drive-RLInfra supports policy assessment at two levels. During training, BEV video diagnostics help inspect closed-loop behavior, reward timing, and failure modes without interrupting rollout. For benchmark-style evaluation, trained RL baseline checkpoints are loaded through a CARLA Leaderboard-compatible autonomous agent, executed on route files, and scored with CARLA Leaderboard 2.0 / 2.1 route-completion and infraction criteria.

## Components

| Component | Description |
|---|---|
| `L5_LeaderboardEvalRuntime` | Turns a trained model into a Leaderboard 2.0 `AutonomousAgent` step function for closed-loop evaluation. |
| `L5_RLLeaderboardAgent` | Concrete agent class loaded by the parallel evaluation launcher. |
| `L5_RLStatisticsManager` | Per-route infraction / penalty bookkeeping with crash-safe atomic writes and `merge_results()` for multi-environment aggregation. |
| `L5_TrainingVisualizer` | Async BEV video recorder with reward overlay — observe training progress without stopping the run. |

## Launch script

Use:

```bash
bash tools/launch/evaluation/run_leaderboard_eval_parallel.sh
```

Before launching, edit the top of the script:

```bash
export CARLA_ROOT="/path/to/carla"

CONFIG_PATH="/path/to/eval_config.yaml"
CHECKPOINT_PATH="/path/to/checkpoint_dir"
ROUTES_FILE="/path/to/routes/eval_routes.xml"

NUM_WORKERS=16
STOCHASTIC=false
```

`NUM_WORKERS` is a launcher setting for this evaluation script. It is passed to `run_leaderboard_eval_parallel.py --num-workers` and controls how many parallel route jobs are launched, capped by the number of route episodes. During evaluation, the YAML still supplies CARLA slot settings such as `env.carla.host`, `port`, `traffic_manager_port`, and `gpu_id`; these lists need at least `NUM_WORKERS` entries.

## Runtime layout

The evaluation launcher uses:

```text
vendor/carla/evaluation-runtime/
```

as the Leaderboard root, and uses:

```text
b2d_rlinfra/evaluation/leaderboard/agent.py
```

as the policy entry point.

## Supported checkpoints

This launcher is intended for RL baseline checkpoints trained in this repository. The current runtime supports BEV-style vector observations (`bev_mask` or `bev_image`) with optional scalar observations, and control-style actions such as discrete controls or continuous throttle / steering / brake variants.

End-to-end model evaluation should use the model-specific external evaluation flow instead of this launcher.

## Outputs

Evaluation results are written under:

```text
leaderboard_eval_logs/
```

The shell script creates a timestamped directory named from the config stem. Important top-level artifacts include:

- `run.log` - top-level launcher log.
- `manifest.json` - workers, route jobs, attempts, paths, and artifact references.
- `statistics.json` - merged CARLA Leaderboard statistics using the default v2.1-style score view.
- `statistics_v20.json` - converted v2.0-style statistics when conversion succeeds.
- `summary.json` - compact run summary with status, worker counts, record counts, success rate, scores, infractions, and artifact paths.
- `summary_v20.json` - compact v2.0 summary when conversion succeeds.
- `live_results.txt` - concatenated live result output from route attempts.
- `combined_eval.log` - concatenated per-attempt evaluation logs.
- `videos/` - consolidated BEV videos exported by the evaluation runtime.
- `workers/` - per-worker job and attempt directories, including route shards, logs, and attempt statistics.
- `records/` - recorder output root used by the default shell script.

## Route crashes and retries

The parallel evaluator treats each route as an independent job. When a child evaluator process exits, the launcher checks the attempt statistics for the expected route record. If the route crashed before producing a route record, the attempt is marked as `missing_record` and the job is retried once by default.

If the retry also finishes without a route record, the job is marked as `synthetic_missing`. During statistics merge, the launcher inserts a failed route record with status `Failed - Missing route result after retries`, so the final `statistics.json` still covers every planned route. The top-level `summary.json` reports this through `missing_record_count`, `routes_failed_without_record`, job status fields, and the overall run status.

This lets the run keep completed route results instead of leaving the whole evaluation incomplete because one route crashed. The missing route is still included in the final CARLA Leaderboard 2.0 / 2.1 statistics as a failed zero-score record.

## Metrics

The evaluation layer reports both leaderboard-style and diagnostic views.

| Metric | Meaning |
| --- | --- |
| SR | Success rate over evaluated routes |
| DS | Driving score, combining route completion and infraction penalties |
| RC | Route completion |
| Infractions | Collision, traffic-light, stop-sign, lane, blocked, and speed-related events |

The code includes helpers for CARLA Leaderboard 2.0 and 2.1-style statistics:

```text
b2d_rlinfra/evaluation/leaderboard/score_versions.py
```

## Diagnostics

The evaluation layer provides diagnostic tools that cover both training-time observation and post-evaluation route analysis.

### Training-time video diagnostics

`TrainingVisualizer` records asynchronous BEV videos during training with reward overlays. Use these videos to inspect policy behavior, reward timing, and episode crash or failure modes without stopping the training run. Videos are written per-worker at the interval configured in `training.visualization.save_interval`.

### Post-evaluation route analysis

After leaderboard evaluation, `b2d_rlinfra/evaluation/leaderboard/summarize_b2d_groups.py` can regroup the top-level `statistics.json` by scenario type.

```bash
python3 b2d_rlinfra/evaluation/leaderboard/summarize_b2d_groups.py \
    --run-dir leaderboard_eval_logs/<run_dir> \
    --routes resources/routes/bench2drive_fix.xml
```

Grouped summaries are written under: `<run-dir>/group_results/`.


The default script preset groups routes by scenario groups, but the grouping logic is not fixed. You can define your own route groups or scenario taxonomy in the summarizer when you need project-specific diagnostics.

## End-to-end model evaluation

The leaderboard evaluator described above is designed for RL baseline checkpoints trained within this repository. For end-to-end models adapted through the finetuning workflow (MindDrive, DrivePi0, etc.), evaluation typically goes through the official Bench2Drive evaluation pipeline (following their own evaluation instructions) rather than this leaderboard launcher.

The general workflow is:

1. Export finetuned weights using the model-specific export script (for MindDrive, see [MindDrive Finetune Recipe](rl_finetune_minddrive.md#export-and-evaluation); for DrivePi0, see [DrivePi0 Finetune Recipe](rl_finetune_drivepi0.md#export-and-evaluation)).
2. Run the [Bench2Drive](https://github.com/Thinklab-SJTU/Bench2Drive/) closed-loop evaluation with the exported checkpoint.

See the [MindDrive](https://github.com/xiaomi-mlab/MindDrive) and [DriveMoE](https://github.com/Thinklab-SJTU/DriveMoE) repositories for their evaluation instructions.
