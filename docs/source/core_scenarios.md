# Scenario Layer

The scenario layer defines RL training tasks through CARLA Leaderboard-style route files. A route determines the town, start state, destination, weather, and scenario parameters used to initialize an episode.

## Components

| Component | Description |
|---|---|
| `L1_RouteIndexer` | Parses Leaderboard 2.0 route XMLs into sampleable route entries organized around town, route, and scenario context. |
| `L1_select_scenario_name` | Picks the next scenario according to the current sampling strategy. |
| `L1_ScenarioManagerRL` | Lifecycle manager for a single scenario episode: spawn, tick, and cleanup. |

## Route files

Route files live under:

```text
resources/routes/
```

The example training config uses:

```yaml
env:
  routes:
    route_files:
      - resources/routes/train_routes_demo.xml
```

You can replace this with your own route XML files as long as they follow the expected Leaderboard route format.

## Sampling modes

`env.carla.num_envs` controls the number of CARLA workers, while `env.routes.sample_mode` controls how route tasks are sampled. Three modes are supported:

| Mode | Behavior |
| --- | --- |
| `sequential` | Walk through the route list in order (the default). |
| `random` | Shuffle the available routes, walk through the shuffled list once, then reshuffle for the next cycle. |
| `adaptive` | Weight scenarios by recent success statistics shared across workers. |

Custom sampling rules are also easy to add because route selection is centralized
in `b2d_rlinfra/environment/carla_env.py`. Add a new `env.routes.sample_mode` name to the allowed
mode set, initialize any small sampler state next to the existing random and
adaptive setup, and add one branch in `_sample_route()` that returns a
`RouteConfig`. If the rule needs progress reporting, extend `get_route_progress()`
with a few diagnostic fields. Most custom policies can reuse the already parsed
route list, `_get_route_config_by_filtered_index()`, and the per-route context
helpers instead of touching the CARLA episode runtime.

The PPO example uses adaptive sampling:

```yaml
env:
  routes:
    sample_mode: adaptive
    adaptive:
      window_size: 10
      success_threshold: 100.0
      low_success_rate: 0.5
      high_success_rate: 0.8
      low_success_weight: 2.0
      high_success_weight: 0.75
      warmup_min_samples: 8
```

Adaptive sampling tracks recent route outcomes and increases exposure to scenarios with low success. The config also supports a `hard_scenarios` list and `hard_scenario_max_gap_samples` so safety-critical scenarios are revisited regularly.

## Short scenario-centered routes

The paper uses short routes centered on single interactive scenarios. This design reduces credit-assignment ambiguity: if a route focuses on one event, a failure can usually be tied to a specific driving skill such as yielding, obstacle bypassing, merging, or pedestrian interaction.

## Practical route workflow

1. Place route XML files in `resources/routes/`.
2. Reference them in `env.routes.route_files`.
3. Keep `route_max_length_m` aligned with the intended episode horizon.
4. Start with a small subset for smoke tests.
5. Use adaptive sampling for mixed-difficulty scenario sets, or stage grouped route files for curriculum-style training.

## Related modules

- `b2d_rlinfra/framework/scenario_layer.py`
- `b2d_rlinfra/scenario/adaptive_route_sampler.py`
- `b2d_rlinfra/learning/training/parallel_utils.py`
- `resources/routes/train_routes_demo.xml`
