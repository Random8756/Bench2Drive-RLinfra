"""Private ``info`` keys used for wrapper/handler coordination."""

# Produced by RoutePlanWrapper; consumed by termination/reward handlers.
WRAPPER_ROUTE_DISTANCE_ON_TICK = "wrapper/route/distance_on_tick"
WRAPPER_ROUTE_DISTANCE_ON_TICK_REWARD = "wrapper/route/distance_on_tick_reward"

# Produced by ObservationWrapper; consumed by reward handlers.
WRAPPER_OBS_EMERGENCY_IN_VISION = "wrapper/obs/emergency_vehicles_in_vision"

# Shared by termination and reward wrappers/handlers.
WRAPPER_TERM_TRIGGERED = "wrapper/term/triggered"

# Reward-handler terminal-state fallback consumed by RewardWrapper.
WRAPPER_REWARD_SIMPLE_TERMINATED = "wrapper/reward/simple_terminated"
WRAPPER_REWARD_SIMPLE_TRUNCATED = "wrapper/reward/simple_truncated"
WRAPPER_REWARD_PARKING_EXIT_DEVIATION_EXEMPT = (
    "wrapper/reward/parking_exit_deviation_exempt"
)
WRAPPER_REWARD_BLOCKED_BY_OBSTACLES = "wrapper/reward/blocked_by_obstacles"


__all__ = [
    "WRAPPER_ROUTE_DISTANCE_ON_TICK",
    "WRAPPER_ROUTE_DISTANCE_ON_TICK_REWARD",
    "WRAPPER_OBS_EMERGENCY_IN_VISION",
    "WRAPPER_TERM_TRIGGERED",
    "WRAPPER_REWARD_SIMPLE_TERMINATED",
    "WRAPPER_REWARD_SIMPLE_TRUNCATED",
    "WRAPPER_REWARD_PARKING_EXIT_DEVIATION_EXEMPT",
    "WRAPPER_REWARD_BLOCKED_BY_OBSTACLES",
]
