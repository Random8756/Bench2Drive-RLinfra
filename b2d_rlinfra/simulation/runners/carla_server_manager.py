"""Manage CARLA server processes, ports, health checks, and restarts."""

import subprocess
import os
import time
import signal
import socket
import logging
import re
import shlex
from typing import List, Optional, Dict, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum, auto
import threading
from pathlib import Path

logger = logging.getLogger("Carla Manager")

__layer__ = (3, "Simulation")

_CARLA_PROCESS_PATTERN = "CarlaUE4|carla-rpc-port"


class ServerState(Enum):
    STOPPED = auto()
    STARTING = auto()
    RUNNING = auto()
    ERROR = auto()
    STOPPING = auto()


@dataclass
class ServerConfig:
    carla_root: str
    port: int = 2000
    # UE4 graphics adapter index, passed as -graphicsadapter=N to CarlaUE4.
    # This follows UE4 RHI enumeration, which does NOT match CUDA device order.
    gpu_id: int = 0
    fps: Optional[int] = None
    render_offscreen: bool = True
    no_sound: bool = True
    null_rhi: bool = True
    opengl: bool = False
    vulkan: bool = False
    no_steam: bool = False
    quality_level: str = "Low"
    timeout: float = 60.0
    host: str = "localhost"
    multihome: Optional[str] = None
    extra_args: List[str] = field(default_factory=list)
    fake_bind_lib: Optional[str] = None


@dataclass
class ServerInfo:
    config: ServerConfig
    state: ServerState = ServerState.STOPPED
    process: Optional[subprocess.Popen] = None
    pid: Optional[int] = None
    start_time: Optional[float] = None
    error_message: Optional[str] = None


class CARLAServerManager:
    """Start, monitor, and stop CARLA server processes."""
    
    def __init__(
        self,
        carla_root: str,
        default_fps: Optional[int] = 10,
        default_quality: str = "Default",  # "Default" = don't add quality-level arg
        auto_restart: bool = True,
        fake_bind_lib: Optional[str] = None,
    ):
        """Initialize a manager for the CARLA installation at ``carla_root``."""
        self.carla_root = Path(carla_root)
        self.default_fps = default_fps
        self.default_quality = default_quality
        self.auto_restart = auto_restart
        self.fake_bind_lib = fake_bind_lib
        
        if self.fake_bind_lib is None:
            _project_root = Path(__file__).resolve().parents[3]
            default_fake_bind = _project_root / "tools" / "fake_bind.so"
            if default_fake_bind.exists():
                self.fake_bind_lib = str(default_fake_bind)
        
        self._validate_carla_path()
        
        self._servers: Dict[Tuple[str, int], ServerInfo] = {}  # (host, port) -> ServerInfo
        self._lock = threading.Lock()
        
    
    def _validate_carla_path(self):
        carla_sh = self.carla_root / "CarlaUE4.sh"
        if not carla_sh.exists():
            for subdir in ["", "Dist", "Build"]:
                path = self.carla_root / subdir / "CarlaUE4.sh"
                if path.exists():
                    self.carla_root = self.carla_root / subdir
                    return
            raise FileNotFoundError(
                f"CarlaUE4.sh not found in {self.carla_root}. "
                "Please provide the correct CARLA installation path."
            )
    
    def _get_carla_executable(self) -> str:
        return str(self.carla_root / "CarlaUE4.sh")
    
    def _build_command(self, config: ServerConfig, host: str = "localhost") -> str:
        """Build the shell command for one server process."""
        cmd_prefix = []
        
        fake_bind_lib = config.fake_bind_lib or self.fake_bind_lib
        if fake_bind_lib and Path(fake_bind_lib).exists():
            cmd_prefix.append(f"export FAKE_BIND_IP={host} &&")
            cmd_prefix.append(f"LD_PRELOAD={fake_bind_lib}")
        
        cmd_parts = [self._get_carla_executable()]
        
        if config.render_offscreen:
            cmd_parts.append("-RenderOffScreen")
        
        if config.no_sound:
            cmd_parts.append("-nosound")
        
        if config.null_rhi:
            cmd_parts.append("-nullrhi")
        
        if config.fps is not None:
            cmd_parts.append(f"-fps={config.fps}")
        
        cmd_parts.append(f"-carla-rpc-port={config.port}")

        cmd_parts.append(f"-graphicsadapter={int(config.gpu_id)}")
        
        if config.quality_level and config.quality_level != "Default":
            cmd_parts.append(f"-quality-level={config.quality_level}")
        
        if config.multihome:
            cmd_parts.append(f"-multihome={config.multihome}")

        if config.opengl:
            cmd_parts.append("-opengl")

        if config.vulkan:
            cmd_parts.append("-vulkan")

        if config.no_steam:
            cmd_parts.append("-nosteam")
        
        cmd_parts.extend(config.extra_args)
        
        if cmd_prefix:
            return " ".join(cmd_prefix) + " " + " ".join(cmd_parts)
        return " ".join(cmd_parts)
    
    def _make_env(self, host: str = "localhost") -> Dict[str, str]:
        env = os.environ.copy()
        env["FAKE_BIND_IP"] = host
        return env
    
    def _check_port_available(self, port: int, host: str = "localhost") -> bool:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex((host, port))
            sock.close()
            return result != 0
        except:
            return True
    
    def _wait_for_server(self, port: int, host: str = "localhost", 
                         timeout: float = 60.0) -> bool:
        start_time = time.time()
        attempt = 0
        
        while time.time() - start_time < timeout:
            attempt += 1
            try:
                import carla
                client = carla.Client(host, port)
                client.set_timeout(2.0)
                version = client.get_server_version()
                logger.info(f"CARLA server on {host}:{port} is ready! Version: {version}")
                return True
            except Exception as e:
                if attempt % 10 == 0:
                    logger.debug(f"Waiting for {host}:{port}... attempt {attempt}, error: {e}")
                    with self._lock:
                        server_key = (host, port)
                        if server_key in self._servers:
                            proc = self._servers[server_key].process
                            if proc and proc.poll() is not None:
                                logger.error(f"CARLA process for {host}:{port} exited with code {proc.returncode}")
                                return False
                time.sleep(1)
        
        return False
    
    def _wait_for_server_simple(self, port: int, host: str = "localhost",
                                 timeout: float = 60.0) -> bool:
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(2)
                result = sock.connect_ex((host, port))
                sock.close()
                if result == 0:
                    time.sleep(3)
                    logger.info(f"CARLA server on port {port} is ready!")
                    return True
            except:
                pass
            time.sleep(1)
        
        return False
    
    def start_server(
        self,
        port: int = 2000,
        gpu_id: int = 0,
        fps: Optional[int] = None,
        quality_level: Optional[str] = None,
        timeout: float = 60.0,
        host: str = "localhost",
        multihome: Optional[str] = None,
        render_offscreen: Optional[bool] = None,
        no_sound: Optional[bool] = None,
        null_rhi: Optional[bool] = None,
        opengl: Optional[bool] = None,
        vulkan: Optional[bool] = None,
        no_steam: Optional[bool] = None,
        extra_args: Optional[List[str]] = None,
        wait: bool = True,
        **_ignored,
    ) -> bool:
        """Start one server and optionally wait until it is ready."""
        server_key = (host, port)
        
        with self._lock:
            if server_key in self._servers and self._servers[server_key].state == ServerState.RUNNING:
                logger.warning(f"Server on {host}:{port} is already running")
                return True
            
            if not self._check_port_available(port, host):
                logger.error(f"Port {host}:{port} is already in use")
                return False
            
            resolved_fps = self.default_fps if fps is None else fps
            if extra_args is None:
                normalized_extra_args = []
            elif isinstance(extra_args, str):
                normalized_extra_args = shlex.split(extra_args)
            else:
                normalized_extra_args = list(extra_args)

            config = ServerConfig(
                carla_root=str(self.carla_root),
                port=port,
                gpu_id=gpu_id,
                fps=resolved_fps,
                render_offscreen=True if render_offscreen is None else bool(render_offscreen),
                no_sound=True if no_sound is None else bool(no_sound),
                null_rhi=True if null_rhi is None else bool(null_rhi),
                opengl=False if opengl is None else bool(opengl),
                vulkan=False if vulkan is None else bool(vulkan),
                no_steam=False if no_steam is None else bool(no_steam),
                quality_level=quality_level or self.default_quality,
                timeout=timeout,
                host=host,
                multihome=multihome,
                extra_args=normalized_extra_args
            )
            
            cmd = self._build_command(config, host)
            # cmd = f"su -s /bin/bash carla -c {shlex.quote(cmd)}"
            logger.info(f"Starting CARLA server: {cmd}")
            
            env = self._make_env(host)
            
            try:
                stdout_target = subprocess.DEVNULL
                stderr_target = subprocess.DEVNULL
                log_dir = os.environ.get("B2D_CARLA_LOG_DIR")
                if log_dir:
                    Path(log_dir).mkdir(parents=True, exist_ok=True)
                    log_path = Path(log_dir) / f"carla_{host.replace('.', '_')}_{port}.log"
                    log_handle = open(log_path, "wb")
                    stdout_target = log_handle
                    stderr_target = subprocess.STDOUT
                    logger.info("CARLA server log: %s", log_path)
                process = subprocess.Popen(
                    cmd,
                    shell=True,
                    env=env,
                    preexec_fn=os.setsid,
                    stdout=stdout_target,
                    stderr=stderr_target,
                )
                
                server_info = ServerInfo(
                    config=config,
                    state=ServerState.STARTING,
                    process=process,
                    pid=process.pid,
                    start_time=time.time()
                )
                self._servers[server_key] = server_info
                
            except Exception as e:
                logger.error(f"Failed to start CARLA server: {e}")
                return False
        
        if wait:
            try:
                success = self._wait_for_server(port, host, timeout)
            except ImportError:
                success = self._wait_for_server_simple(port, host, timeout)
            
            with self._lock:
                if success:
                    self._servers[server_key].state = ServerState.RUNNING
                else:
                    self._servers[server_key].state = ServerState.ERROR
                    self._servers[server_key].error_message = "Startup timeout"
                    logger.error(f"CARLA server on {host}:{port} failed to start")
            
            return success
        
        return True
    
    def start_servers(
        self,
        ports: List[int],
        gpu_ids: Optional[List[int]] = None,
        hosts: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[Tuple[str, int], bool]:
        """Start servers for aligned port, GPU, and host lists."""
        num_servers = len(ports)
        
        if gpu_ids is None:
            gpu_ids = [0] * num_servers
        if hosts is None:
            hosts = ['localhost'] * num_servers
        
        assert len(gpu_ids) >= num_servers, "gpu_ids length must >= ports length"
        assert len(hosts) >= num_servers, "hosts length must >= ports length"
        
        results = {}
        timeout = kwargs.get('timeout', 60.0)
        
        for i, (port, gpu_id, host) in enumerate(zip(ports, gpu_ids, hosts)):
            if not self._check_port_available(port, host):
                logger.info(f"Server on {host}:{port} is already running, skipping...")
                results[(host, port)] = True
                continue
            
            logger.info(
                f"Starting CARLA server {i}: {host}:{port} "
                f"(graphicsadapter={gpu_id})"
            )
            self.start_server(
                port=port,
                gpu_id=gpu_id,
                host=host,
                wait=False,
                **kwargs,
            )
        
        for port, host in zip(ports, hosts):
            if (host, port) in results:
                continue
            
            try:
                success = self._wait_for_server(port, host, timeout)
            except ImportError:
                success = self._wait_for_server_simple(port, host, timeout)
            
            results[(host, port)] = success
            
            with self._lock:
                server_key = (host, port)
                if server_key in self._servers:
                    if success:
                        self._servers[server_key].state = ServerState.RUNNING
                    else:
                        self._servers[server_key].state = ServerState.ERROR
        
        return results
    
    def stop_server(self, host: str, port: int, timeout: float = 60.0) -> bool:
        """Stop the server identified by host and RPC port."""
        server_key = (host, port)
        
        with self._lock:
            if server_key not in self._servers:
                logger.warning(f"No server found on {host}:{port}")
                return True
            
            server_info = self._servers[server_key]
            
            if server_info.state == ServerState.STOPPED:
                return True
            
            server_info.state = ServerState.STOPPING

            try:
                carla_pids = set(self._find_carla_pids_by_host(host, port))
                if server_info.process is not None:
                    carla_pids.add(server_info.process.pid)
                    carla_pids.update(self._collect_descendant_pids(server_info.process.pid))
                carla_pids = sorted(carla_pids)

                if not carla_pids:
                    logger.warning(f"No CARLA processes found for host {host}")
                    if server_info.process is not None:
                        try:
                            os.killpg(os.getpgid(server_info.process.pid), signal.SIGKILL)
                            server_info.process.wait(timeout=5)
                        except:
                            pass
                else:
                    logger.debug(f"Found CARLA processes for {host}: {carla_pids}")

                    for pid in carla_pids:
                        try:
                            os.kill(pid, signal.SIGKILL)
                            logger.debug(f"Killed CARLA process {pid}")
                        except (ProcessLookupError, PermissionError) as e:
                            logger.info(f"Failed to kill process {pid}: {e}")

                    for pid in carla_pids:
                        try:
                            subprocess.run(['pkill', '-9', '-P', str(pid)],
                                         capture_output=True, timeout=2)
                        except (subprocess.TimeoutExpired, FileNotFoundError):
                            pass

                    time.sleep(1.0)

                    still_running = []
                    for pid in carla_pids:
                        if self._is_process_running(pid):
                            still_running.append(pid)

                    if still_running:
                        logger.warning(f"Some processes still running after first kill, doing second kill: {still_running}")
                        for pid in still_running:
                            try:
                                os.kill(pid, signal.SIGKILL)
                                logger.info(f"Second kill: CARLA process {pid}")
                            except (ProcessLookupError, PermissionError) as e:
                                logger.info(f"Failed to second kill process {pid}: {e}")

                        for pid in still_running:
                            try:
                                subprocess.run(['pkill', '-9', '-P', str(pid)],
                                             capture_output=True, timeout=2)
                            except (subprocess.TimeoutExpired, FileNotFoundError):
                                pass

                        time.sleep(0.5)

                        final_still_running = []
                        for pid in carla_pids:
                            if self._is_process_running(pid):
                                final_still_running.append(pid)

                        if final_still_running:
                            poll_timeout = max(0.0, timeout - 1.5)
                            logger.info(
                                "Waiting up to %.1fs for CARLA processes to exit on %s:%s: %s",
                                poll_timeout,
                                host,
                                port,
                                final_still_running,
                            )
                            final_still_running, waited = self._wait_for_processes_exit(
                                final_still_running,
                                timeout=poll_timeout,
                            )
                            if final_still_running:
                                logger.warning(
                                    "Processes still running after second kill and %.1fs poll on %s:%s: %s",
                                    waited,
                                    host,
                                    port,
                                    final_still_running,
                                )
                            elif waited >= 0.2:
                                logger.info(
                                    "Lingering CARLA processes exited after %.1fs on %s:%s",
                                    waited,
                                    host,
                                    port,
                                )
                        else:
                            logger.info("All processes killed after second attempt")
                    else:
                        logger.debug("All CARLA processes killed successfully on first attempt")

                server_info.state = ServerState.STOPPED
                server_info.process = None
                logger.info(f"CARLA server on {host}:{port} stopped")
                return True
                
            except Exception as e:
                logger.error(f"Error stopping server on {host}:{port}: {e}")
                server_info.state = ServerState.ERROR
                server_info.error_message = str(e)
                return False

    def _find_carla_pids_by_host(self, host_ip: str, port: Optional[int] = None) -> list:
        """Find CARLA-related PIDs for a fake-bind host.

        Processes started through wrappers such as ``su - carla -c`` may hide
        ``/proc/<pid>/environ`` from the caller.  In that case the wrapper
        command line still carries ``FAKE_BIND_IP=...`` and the CARLA child can
        be connected through its ``-carla-rpc-port=...`` argument.
        """
        all_pids = set()
        matched_ports = set()

        try:
            all_pids.update(self._find_listening_pids(host_ip, port))
            candidate_pids = self._candidate_carla_pids()

            for pid in candidate_pids:
                environ = self._read_proc_environ(pid)
                cmdline = self._read_proc_cmdline(pid)
                proc_text = f"{environ}\n{cmdline}"
                if not self._has_fake_bind_ip(proc_text, host_ip):
                    continue

                pid_ports = self._extract_carla_rpc_ports(proc_text)
                if port is not None and pid_ports and int(port) not in pid_ports:
                    continue

                all_pids.add(pid)
                if port is not None:
                    matched_ports.add(int(port))
                else:
                    matched_ports.update(pid_ports)

            if matched_ports:
                for pid in candidate_pids:
                    cmdline = self._read_proc_cmdline(pid)
                    if self._extract_carla_rpc_ports(cmdline) & matched_ports:
                        all_pids.add(pid)

            for pid in list(all_pids):
                all_pids.update(self._collect_descendant_pids(pid))

        except Exception as e:
            logger.info(f"Error finding CARLA PIDs for {host_ip}: {e}")

        return [pid for pid in sorted(all_pids) if self._is_process_running(pid)]

    def _candidate_carla_pids(self) -> List[int]:
        try:
            result = subprocess.run(
                ['pgrep', '-f', _CARLA_PROCESS_PATTERN],
                capture_output=True,
                text=True,
                timeout=3
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return []

        if result.returncode != 0:
            return []

        pids = []
        for value in result.stdout.strip().splitlines():
            if not value:
                continue
            try:
                pids.append(int(value))
            except ValueError:
                continue
        return pids

    def _find_listening_pids(self, host_ip: str, port: Optional[int] = None) -> Set[int]:
        pids = set()
        try:
            result = subprocess.run(
                ['ss', '-tlnp'],
                capture_output=True,
                text=True,
                timeout=5
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return pids

        if result.returncode != 0:
            return pids

        port_pattern = None
        if port is not None:
            port_pattern = re.compile(r':{}(?:\s|$)'.format(int(port)))

        for line in result.stdout.splitlines():
            if 'pid=' not in line:
                continue
            if not self._ss_line_matches_host_port(line, host_ip, port):
                continue
            if port_pattern is not None and not port_pattern.search(line):
                continue
            for pid_text in re.findall(r'pid=(\d+)', line):
                try:
                    pids.add(int(pid_text))
                except ValueError:
                    continue
        return pids

    def _has_fake_bind_ip(self, text: str, host_ip: str) -> bool:
        return re.search(
            r'(^|[\s\0])FAKE_BIND_IP={}($|[\s\0])'.format(re.escape(host_ip)),
            text,
        ) is not None

    def _ss_line_matches_host_port(
        self,
        line: str,
        host_ip: str,
        port: Optional[int] = None,
    ) -> bool:
        parts = line.split()
        if len(parts) < 4:
            return False

        local_addr = parts[3]
        local_host, local_port = self._split_ss_local_address(local_addr)
        if local_host != host_ip:
            return False
        if port is not None and local_port != int(port):
            return False
        return True

    def _split_ss_local_address(self, local_addr: str) -> Tuple[str, Optional[int]]:
        addr = local_addr.strip()

        if addr.startswith('['):
            end = addr.find(']')
            if end != -1:
                host = addr[1:end]
                rest = addr[end + 1:]
                if rest.startswith(':') and rest[1:].isdigit():
                    return host, int(rest[1:])
                return host, None

        if ':' not in addr:
            return addr, None

        host, port_text = addr.rsplit(':', 1)
        if port_text.isdigit():
            return host, int(port_text)
        return host, None

    def _read_proc_cmdline(self, pid: int) -> str:
        try:
            with open(f'/proc/{pid}/cmdline', 'rb') as f:
                return f.read().replace(b'\0', b' ').decode('utf-8', errors='ignore')
        except (FileNotFoundError, PermissionError, OSError):
            return ""

    def _read_proc_environ(self, pid: int) -> str:
        try:
            with open(f'/proc/{pid}/environ', 'rb') as f:
                return f.read().replace(b'\0', b'\n').decode('utf-8', errors='ignore')
        except (FileNotFoundError, PermissionError, OSError):
            return ""

    def _extract_carla_rpc_ports(self, text: str) -> Set[int]:
        ports = set()
        for match in re.finditer(r'-carla-rpc-port(?:=|\s+)(\d+)', text):
            try:
                ports.add(int(match.group(1)))
            except ValueError:
                continue
        return ports

    def _collect_descendant_pids(self, pid: int) -> Set[int]:
        descendants = set()
        try:
            result = subprocess.run(
                ['pgrep', '-P', str(pid)],
                capture_output=True,
                text=True,
                timeout=2
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return descendants

        if result.returncode != 0:
            return descendants

        for value in result.stdout.strip().splitlines():
            if not value:
                continue
            try:
                child_pid = int(value)
            except ValueError:
                continue
            if child_pid in descendants:
                continue
            descendants.add(child_pid)
            descendants.update(self._collect_descendant_pids(child_pid))
        return descendants

    def _wait_for_processes_exit(
        self,
        pids: List[int],
        timeout: float,
        poll_interval: float = 0.2,
    ) -> Tuple[List[int], float]:
        start = time.monotonic()
        remaining = [pid for pid in pids if self._is_process_running(pid)]

        if not remaining or timeout <= 0:
            return remaining, time.monotonic() - start

        deadline = start + timeout
        while remaining:
            now = time.monotonic()
            if now >= deadline:
                break
            time.sleep(min(poll_interval, deadline - now))
            remaining = [pid for pid in remaining if self._is_process_running(pid)]

        return remaining, time.monotonic() - start

    def _is_process_running(self, pid: int) -> bool:
        stat_path = f'/proc/{pid}/stat'
        try:
            with open(stat_path, 'r') as f:
                stat_fields = f.read().split()
                if len(stat_fields) >= 3 and stat_fields[2] == 'Z':
                    return False
        except FileNotFoundError:
            return False
        except (PermissionError, OSError):
            pass

        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
    
    def stop_all(self, timeout: float = 10.0):
        server_keys = list(self._servers.keys())
        for host, port in server_keys:
            self.stop_server(host, port, timeout)
        
    def restart_server(self, host: str, port: int, **kwargs) -> bool:
        server_key = (host, port)
        if server_key in self._servers:
            config = self._servers[server_key].config
            self.stop_server(host, port)
            self._wait_port_free(host, port)
            _config_keys = {
                'gpu_id', 'fps', 'quality_level', 'multihome',
                'render_offscreen', 'no_sound', 'null_rhi',
                'opengl', 'vulkan', 'no_steam', 'extra_args',
                'host', 'port',
            }
            extra = {k: v for k, v in kwargs.items() if k not in _config_keys}
            return self.start_server(
                port=port,
                gpu_id=kwargs.get('gpu_id', config.gpu_id),
                fps=config.fps,
                quality_level=config.quality_level,
                host=host,
                multihome=config.multihome,
                render_offscreen=config.render_offscreen,
                no_sound=config.no_sound,
                null_rhi=config.null_rhi,
                opengl=config.opengl,
                vulkan=config.vulkan,
                no_steam=config.no_steam,
                extra_args=kwargs.get('extra_args', config.extra_args),
                **extra,
            )
        self._wait_port_free(host, port)
        return self.start_server(port=port, host=host, **kwargs)

    def _wait_port_free(self, host: str, port: int, timeout: float = 30.0) -> bool:
        start = time.time()
        while time.time() - start < timeout:
            if self._check_port_available(port, host):
                return True
            time.sleep(1.0)
        logger.warning("Port %s:%s still in use after %.0fs", host, port, timeout)
        return False
    
    def is_server_running(self, host: str, port: int) -> bool:
        server_key = (host, port)
        
        with self._lock:
            if server_key not in self._servers:
                return False
            
            server_info = self._servers[server_key]
            
            if server_info.state != ServerState.RUNNING:
                return False
            
            if server_info.process is not None:
                if server_info.process.poll() is not None:
                    server_info.state = ServerState.STOPPED
                    return False
            
            return True
    
    def get_server_info(self, host: str, port: int) -> Optional[ServerInfo]:
        return self._servers.get((host, port))
    
    def get_all_servers(self) -> Dict[Tuple[str, int], ServerInfo]:
        return dict(self._servers)
    
    def get_running_servers(self) -> List[Tuple[str, int]]:
        running = []
        for host, port in self._servers:
            if self.is_server_running(host, port):
                running.append((host, port))
        return running
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_all()
        return False
    
    def __del__(self):
        try:
            self.stop_all()
        except:
            pass
