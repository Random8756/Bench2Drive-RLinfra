"""CARLA client connection manager.

1. Wait-for-server-ready handshake on startup.
2. Periodic connection health checks.
3. Automatic reconnect on transient failures.
4. Traffic Manager connection management.
"""

import time
import socket
import logging
import random
import threading
from typing import Optional, Tuple
from dataclasses import dataclass

import carla

logger = logging.getLogger("Carla")

__layer__ = (3, "Simulation")


@dataclass
class CARLAConnectionConfig:
    host: str = "localhost"
    port: int = 2000
    traffic_manager_port: int = 8000
    traffic_manager_seed: int = 0
    connection_timeout: float = 10.0
    server_wait_timeout: float = 120.0
    retry_interval: float = 2.0
    max_retries: int = 60


class CARLAConnectionError(Exception):
    pass


class CARLAConnection:
    """Manage a CARLA client connection and its cached world resources."""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 2000,
        traffic_manager_port: int = 8000,
        traffic_manager_seed: int = 0,
        connection_timeout: float = 10.0,
        server_wait_timeout: float = 120.0,
        retry_interval: float = 2.0,
        max_retries: int = 60
    ):
        self.host = host
        self.port = port
        self.traffic_manager_port = traffic_manager_port
        self.traffic_manager_seed = traffic_manager_seed
        self.connection_timeout = connection_timeout
        self.server_wait_timeout = server_wait_timeout
        self.retry_interval = retry_interval
        self.max_retries = max_retries
        
        self._client: Optional[carla.Client] = None
        self._world: Optional[carla.World] = None
        self._traffic_manager: Optional[carla.TrafficManager] = None
        self._connected = False
    
    @classmethod
    def from_config(cls, config: dict, env_index: int = 0) -> 'CARLAConnection':
        """Build a connection for ``env_index`` from scalar or per-env values."""
        carla_config = config.get('carla', {})

        host_cfg = carla_config.get('host', 'localhost')
        if isinstance(host_cfg, list):
            if not host_cfg:
                host = 'localhost'
            else:
                host = host_cfg[env_index] if env_index < len(host_cfg) else host_cfg[-1]
        else:
            host = host_cfg

        port_cfg = carla_config.get('port', 2000)
        if isinstance(port_cfg, list):
            if not port_cfg:
                port = 2000
            elif env_index < len(port_cfg):
                port = port_cfg[env_index]
            else:
                raise ValueError(f"env_index {env_index} out of range, only {len(port_cfg)} ports configured")
        else:
            port = port_cfg

        tm_port_cfg = carla_config.get('traffic_manager_port', port + 6000)
        if isinstance(tm_port_cfg, list):
            traffic_manager_port = tm_port_cfg[env_index] if env_index < len(tm_port_cfg) else port + 6000
        else:
            traffic_manager_port = tm_port_cfg

        tm_seed_cfg = carla_config.get('traffic_manager_seed', 0)
        if isinstance(tm_seed_cfg, list):
            traffic_manager_seed = tm_seed_cfg[env_index] if env_index < len(tm_seed_cfg) else env_index
        else:
            traffic_manager_seed = tm_seed_cfg
        
        return cls(
            host=host,
            port=port,
            traffic_manager_port=traffic_manager_port,
            traffic_manager_seed=traffic_manager_seed,
            connection_timeout=carla_config.get('connection_timeout', 10.0),
            server_wait_timeout=carla_config.get('server_wait_timeout', 120.0),
            retry_interval=2.0,
            max_retries=carla_config.get('max_retries', 60)
        )
    
    def is_server_available(self) -> bool:
        """Return whether the configured TCP port accepts connections."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2.0)
            result = sock.connect_ex((self.host, self.port))
            sock.close()
            return result == 0
        except Exception:
            return False
    
    def wait_for_server(self, timeout: Optional[float] = None) -> bool:
        """Wait until CARLA responds or the timeout expires."""
        timeout = timeout or self.server_wait_timeout
        start_time = time.time()
        retry_count = 0
        
        # Ensure the retry budget spans the requested timeout.
        max_retries = self.max_retries
        if max_retries is None or max_retries <= 0:
            max_retries = float('inf')
        min_retries = int(timeout / self.retry_interval) + 1
        max_retries = max(max_retries, min_retries)
        
        logger.info(f"Waiting for CARLA server at {self.host}:{self.port}...")
        
        while time.time() - start_time < timeout and retry_count < max_retries:
            if self.is_server_available():
                try:
                    client = carla.Client(self.host, self.port)
                    client.set_timeout(self.connection_timeout)
                    version = client.get_server_version()
                    logger.debug(f"CARLA server ready! Version: {version}")
                    return True
                except Exception as e:
                    logger.debug(f"Server port open but not ready yet: {e}")
            
            retry_count += 1
            if retry_count % 10 == 0:
                logger.info(f"Still waiting for CARLA server... ({retry_count} retries)")
            
            time.sleep(self.retry_interval)
        
        logger.error(f"Timeout waiting for CARLA server at {self.host}:{self.port}")
        return False
    
    def connect(
        self,
        wait_for_server: bool = True,
        connect_retries: int = 1,
        connect_retry_interval: float = 2.0
    ) -> bool:
        """Connect to CARLA, optionally waiting and retrying first."""
        if self._connected:
            return True
        
        if wait_for_server:
            if not self.wait_for_server():
                raise CARLAConnectionError(
                    f"Failed to connect to CARLA server at {self.host}:{self.port} - server not ready"
                )
        
        last_error = None
        attempts = max(1, int(connect_retries))
        for attempt in range(1, attempts + 1):
            try:
                self._client = carla.Client(self.host, self.port)
                self._client.set_timeout(self.connection_timeout)
                
                version = self._client.get_server_version()
                logger.debug(f"Connected to CARLA server (version: {version}) at {self.host}:{self.port}")
                
                self._connected = True
                return True
                
            except Exception as e:
                last_error = e
                self._client = None
                self._connected = False
                if attempt < attempts:
                    logger.warning(
                        f"Connect attempt {attempt}/{attempts} failed for {self.host}:{self.port}: {e}. "
                        f"Retrying in {connect_retry_interval}s..."
                    )
                    time.sleep(connect_retry_interval)
        
        raise CARLAConnectionError(f"Failed to connect to CARLA server after {attempts} attempts: {last_error}")
    
    def reconnect(
        self,
        wait_for_server: bool = True,
        connect_retries: int = 1,
        connect_retry_interval: float = 2.0
    ) -> bool:
        """Close the current connection and connect again."""
        self.close()
        return self.connect(
            wait_for_server=wait_for_server,
            connect_retries=connect_retries,
            connect_retry_interval=connect_retry_interval
        )
    
    def is_connected(self) -> bool:
        if not self._connected or self._client is None:
            return False
        
        try:
            self._client.get_server_version()
            return True
        except Exception:
            self._connected = False
            return False
    
    @property
    def client(self) -> carla.Client:
        if not self._connected or self._client is None:
            raise CARLAConnectionError("Not connected to CARLA server")
        return self._client
    
    def get_world(self) -> carla.World:
        if self._world is None:
            self._world = self.client.get_world()
        return self._world
    
    def load_world(
        self,
        town: str,
        timeout: float = 60.0,
        reset_settings: bool = False,
        server_check_interval: float = 20.0,
        server_check_jitter: float = 5.0,
    ) -> carla.World:
        """Load ``town`` while monitoring timeout and server liveness."""
        current_timeout = None
        client = self._client
        if client is None:
            raise CARLAConnectionError("Not connected to CARLA server")

        if hasattr(client, "get_timeout"):
            try:
                current_timeout = client.get_timeout()
            except Exception:
                current_timeout = None
        
        logger.debug(
            f"CARLAConnection[{self.host}:{self.port}] load_world: set timeout={timeout}s "
            f"(current={current_timeout if current_timeout is not None else 'unknown'})"
        )
        client.set_timeout(timeout)

        server_check_interval = max(0.0, float(server_check_interval or 0.0))
        server_check_jitter = max(0.0, float(server_check_jitter or 0.0))
        timeout = max(0.0, float(timeout or 0.0))

        start_ts = time.monotonic()
        if server_check_interval > 0.0 and not self.is_server_available():
            self._connected = False
            raise CARLAConnectionError(
                f"CARLA server unavailable before load_world({town})"
            )

        load_result = {}
        load_done = threading.Event()
        load_thread = None

        def next_check_delay() -> float:
            if server_check_interval <= 0.0:
                return float("inf")
            return server_check_interval + random.uniform(0.0, server_check_jitter)

        def load_world_rpc() -> None:
            try:
                load_result["world"] = client.load_world(town, reset_settings=reset_settings)
            except BaseException as exc:
                load_result["error"] = exc
            finally:
                load_done.set()

        try:
            logger.debug(f"CARLAConnection[{self.host}:{self.port}] load_world: calling client.load_world({town}, reset_settings={reset_settings})...")
            load_thread = threading.Thread(
                target=load_world_rpc,
                name=f"carla_load_world_{self.host}_{self.port}",
                daemon=True,
            )
            load_thread.start()

            next_check_ts = start_ts + next_check_delay()
            deadline = start_ts + timeout if timeout > 0.0 else None
            while not load_done.wait(timeout=0.1):
                now = time.monotonic()
                if deadline is not None and now >= deadline:
                    raise CARLAConnectionError(
                        f"load_world({town}) timed out after {timeout:.2f}s"
                    )
                if now >= next_check_ts:
                    if not self.is_server_available():
                        duration = now - start_ts
                        self._connected = False
                        raise CARLAConnectionError(
                            f"CARLA server became unavailable during load_world({town}) "
                            f"after {duration:.2f}s"
                        )
                    next_check_ts = now + next_check_delay()

            if "error" in load_result:
                raise load_result["error"]
            self._world = load_result["world"]
            duration = time.monotonic() - start_ts
            logger.debug(f"CARLAConnection[{self.host}:{self.port}] load_world: loaded {town} in {duration:.2f}s")
            return self._world
        except Exception as e:
            duration = time.monotonic() - start_ts
            logger.error(
                f"CARLAConnection[{self.host}:{self.port}] load_world: FAILED after {duration:.2f}s: {e}"
            )
            raise
        finally:
            if load_thread is None or not load_thread.is_alive():
                client.set_timeout(self.connection_timeout)
                logger.debug(
                    f"CARLAConnection[{self.host}:{self.port}] load_world: timeout restored to {self.connection_timeout}s"
                )
            else:
                logger.debug(
                    f"CARLAConnection[{self.host}:{self.port}] load_world: timeout restore skipped "
                    "because RPC thread is still running"
                )
    
    def get_traffic_manager(self) -> carla.TrafficManager:
        if self._traffic_manager is None:
            self._traffic_manager = self.client.get_trafficmanager(self.traffic_manager_port)
            self._traffic_manager.set_random_device_seed(self.traffic_manager_seed)
            logger.debug(f"Traffic Manager connected on port {self.traffic_manager_port}")
        return self._traffic_manager
    
    def setup_synchronous_mode(
        self,
        frequency_hz: float = 10.0,
        no_rendering: bool = True,
        tile_stream_distance: int = 650,
        actor_active_distance: int = 650,
        hybrid_physics: bool = True
    ) -> None:
        """Configure the world and Traffic Manager for synchronous stepping."""
        world = self.get_world()
        settings = world.get_settings()
        
        settings.fixed_delta_seconds = 1.0 / frequency_hz
        settings.no_rendering_mode = no_rendering
        settings.synchronous_mode = True
        settings.tile_stream_distance = tile_stream_distance
        settings.actor_active_distance = actor_active_distance
        
        world.apply_settings(settings)
        
        tm = self.get_traffic_manager()
        tm.set_synchronous_mode(True)
        tm.set_hybrid_physics_mode(hybrid_physics)
        
        logger.info(f"Synchronous mode enabled at {frequency_hz} Hz")
    
    def disable_synchronous_mode(self) -> None:
        if self._world is not None:
            settings = self._world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            self._world.apply_settings(settings)
        
        if self._traffic_manager is not None:
            self._traffic_manager.set_synchronous_mode(False)
            self._traffic_manager.set_hybrid_physics_mode(False)
    
    def close(self) -> None:
        if self._traffic_manager is not None:
            try:
                self._traffic_manager.set_synchronous_mode(False)
            except Exception:
                pass
            self._traffic_manager = None
        
        if self._world is not None:
            try:
                settings = self._world.get_settings()
                settings.synchronous_mode = False
                settings.fixed_delta_seconds = None
                self._world.apply_settings(settings)
            except Exception:
                pass
            self._world = None
        
        self._client = None
        self._connected = False
    
    def __enter__(self) -> 'CARLAConnection':
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
    
    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def wait_for_carla_server(
    host: str = "localhost",
    port: int = 2000,
    timeout: float = 120.0,
    retry_interval: float = 2.0
) -> bool:
    """Wait for a CARLA server without retaining a connection object."""
    conn = CARLAConnection(
        host=host,
        port=port,
        server_wait_timeout=timeout,
        retry_interval=retry_interval
    )
    return conn.wait_for_server()


def create_carla_client(
    host: str = "localhost",
    port: int = 2000,
    wait_for_server: bool = True,
    timeout: float = 120.0
) -> Tuple[carla.Client, carla.TrafficManager]:
    """Create a connected CARLA client and Traffic Manager pair."""
    conn = CARLAConnection(
        host=host,
        port=port,
        server_wait_timeout=timeout
    )
    conn.connect(wait_for_server=wait_for_server)
    return conn.client, conn.get_traffic_manager()
