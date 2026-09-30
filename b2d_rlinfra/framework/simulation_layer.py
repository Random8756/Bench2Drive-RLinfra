"""Layer 3 — Simulation.

Simulation abstraction. Covers the parallel CARLA environment pool, CARLA
server process management, connection handling, and worker health checks.

Layer entries and implementation modules
----------------------------------------
    L3_CARLAEnvPool            ← b2d_rlinfra.simulation.runners.carla_env_pool
    L3_HealthWorker            ← b2d_rlinfra.simulation.runners.carla_env_pool
    L3_CARLAServerManager      ← b2d_rlinfra.simulation.runners.carla_server_manager
    L3_CARLAConnection         ← b2d_rlinfra.simulation.carla_connection
"""

from __future__ import annotations

__layer__ = (3, "Simulation")

from b2d_rlinfra.simulation.runners.carla_env_pool import (
    CARLAEnvPool as L3_CARLAEnvPool,
    EnvPoolState as L3_EnvPoolState,
    HealthWorker as L3_HealthWorker,
    WorkerInfo as L3_WorkerInfo,
    WorkerState as L3_WorkerState,
)

from b2d_rlinfra.simulation.runners.carla_server_manager import (
    CARLAServerManager as L3_CARLAServerManager,
    ServerConfig as L3_ServerConfig,
    ServerInfo as L3_ServerInfo,
    ServerState as L3_ServerState,
)

from b2d_rlinfra.simulation.carla_connection import (
    CARLAConnection as L3_CARLAConnection,
    CARLAConnectionError as L3_CARLAConnectionError,
)


__all__ = [
    "L3_CARLAEnvPool",
    "L3_EnvPoolState",
    "L3_HealthWorker",
    "L3_WorkerInfo",
    "L3_WorkerState",
    "L3_CARLAServerManager",
    "L3_ServerConfig",
    "L3_ServerInfo",
    "L3_ServerState",
    "L3_CARLAConnection",
    "L3_CARLAConnectionError",
]
