import asyncio
import json
from typing import Callable, Dict, Optional, Set

import tenacity
from aiohttp import ClientSession, ClientTimeout
from daytona_sdk import (
    CreateWorkspaceParams,
    Daytona,
    DaytonaConfig,
    SessionExecuteRequest,
    Workspace,
)

from openhands.core.config.app_config import AppConfig
from openhands.events.stream import EventStream
from openhands.runtime.impl.action_execution.action_execution_client import (
    ActionExecutionClient,
)
from openhands.runtime.plugins.requirement import PluginRequirement
from openhands.runtime.utils.command import get_action_execution_server_startup_command
from openhands.utils.tenacity_stop import stop_if_should_exit

WORKSPACE_PREFIX = 'openhands-sandbox-'


class DaytonaRuntime(ActionExecutionClient):
    """The DaytonaRuntime class is an DockerRuntime that utilizes Daytona workspace as a runtime environment."""

    _sandbox_port: int = 4444
    _vscode_port: int = 4445

    def __init__(
        self,
        config: AppConfig,
        event_stream: EventStream,
        sid: str = 'default',
        plugins: list[PluginRequirement] | None = None,
        env_vars: dict[str, str] | None = None,
        status_callback: Callable | None = None,
        attach_to_existing: bool = False,
        headless_mode: bool = True,
    ):
        assert config.daytona_api_key, 'Daytona API key is required'

        self.config = config
        self.sid = sid
        self.workspace_id = WORKSPACE_PREFIX + sid
        self.workspace: Optional[Workspace] = None
        self._vscode_url: str | None = None

        daytona_config = DaytonaConfig(
            api_key=config.daytona_api_key.get_secret_value(),
            server_url=config.daytona_api_url,
            target=config.daytona_target,
        )
        self.daytona = Daytona(daytona_config)

        # workspace_base cannot be used because we can't bind mount into a workspace.
        if self.config.workspace_base is not None:
            self.log(
                'warning',
                'Workspace mounting is not supported in the Daytona runtime.',
            )

        # TODO: Add automatic image registration here with Daytona SDK

        super().__init__(
            config,
            event_stream,
            sid,
            plugins,
            env_vars,
            status_callback,
            attach_to_existing,
            headless_mode,
        )
        self._closed = False
        self._port_proxies: Dict[int, str] = {}  # Track active port proxies
        self._proxy_servers: Dict[int, asyncio.Server] = {}
        self._proxy_tasks: Set[asyncio.Task] = set()
        # Add lock for port polling
        self._port_poll_lock = asyncio.Lock()

    async def _proxy_handler(
        self,
        local_reader: asyncio.StreamReader,
        local_writer: asyncio.StreamWriter,
        target_url: str,
    ):
        """Handle a single proxy connection."""
        try:
            # Read the incoming HTTP request
            request_data = await local_reader.read(8192)
            if not request_data:
                return

            # Parse the HTTP request to get the method
            request_lines = request_data.decode('utf-8', 'ignore').split('\n')
            if not request_lines:
                return

            # Get HTTP method from first line (e.g., "GET /path HTTP/1.1")
            method = request_lines[0].split(' ')[0]

            # Forward the request to the target URL
            timeout = ClientTimeout(total=30)
            async with ClientSession(timeout=timeout) as session:
                async with session.request(
                    method=method,
                    url=target_url,
                    data=request_data,
                    headers={'Connection': 'close', 'Host': target_url.split('://')[1]},
                ) as response:
                    response_data = await response.read()
                    # Write response status line
                    status_line = f'HTTP/1.1 {response.status} {response.reason}\r\n'
                    local_writer.write(status_line.encode())
                    # Write response headers
                    for header, value in response.headers.items():
                        local_writer.write(f'{header}: {value}\r\n'.encode())
                    local_writer.write(b'\r\n')
                    # Write response body
                    local_writer.write(response_data)
                    await local_writer.drain()
        except Exception as e:
            self.log('error', f'Proxy error: {str(e)}')
        finally:
            local_writer.close()
            await local_writer.wait_closed()

    async def _start_proxy(self, remote_port: int, proxy_url: str) -> int:
        """Start a proxy server for a given port."""
        # Try to use the same port number as the remote port
        try:

            async def handle_connection(reader, writer):
                task = asyncio.create_task(
                    self._proxy_handler(reader, writer, proxy_url)
                )
                self._proxy_tasks.add(task)
                task.add_done_callback(self._proxy_tasks.discard)

            server = await asyncio.start_server(
                handle_connection, 'localhost', remote_port
            )
            self._proxy_servers[remote_port] = server
            return remote_port
        except OSError as e:
            # Log the specific error details
            error_code = getattr(e, 'errno', None)
            error_msg = str(e)
            if error_code == 98:  # Address already in use
                self.log(
                    'warning',
                    f'Port {remote_port} is already in use on localhost (errno={error_code}): {error_msg}',
                )
            elif error_code == 13:  # Permission denied
                self.log(
                    'error',
                    f'Permission denied when trying to bind to port {remote_port} (errno={error_code}): {error_msg}',
                )
            else:
                self.log(
                    'error',
                    f'Failed to start proxy on port {remote_port} (errno={error_code}): {error_msg}',
                )
            return -1

    async def _stop_proxy(self, port: int):
        """Stop a proxy server for a given port."""
        if server := self._proxy_servers.pop(port, None):
            server.close()
            await server.wait_closed()

    def _get_workspace(self) -> Optional[Workspace]:
        try:
            workspace = self.daytona.get_current_workspace(self.workspace_id)
            self.log(
                'info', f'Attached to existing workspace with id: {self.workspace_id}'
            )
        except Exception:
            self.log(
                'warning',
                f'Failed to attach to existing workspace with id: {self.workspace_id}',
            )
            workspace = None

        return workspace

    def _get_creation_env_vars(self) -> dict[str, str]:
        env_vars: dict[str, str] = {
            'port': str(self._sandbox_port),
            'PYTHONUNBUFFERED': '1',
            'VSCODE_PORT': str(self._vscode_port),
        }

        if self.config.debug:
            env_vars['DEBUG'] = 'true'

        return env_vars

    def _create_workspace(self) -> Workspace:
        workspace_params = CreateWorkspaceParams(
            id=self.workspace_id,
            language='python',
            image=self.config.sandbox.runtime_container_image,
            public=True,
            env_vars=self._get_creation_env_vars(),
        )
        workspace = self.daytona.create(workspace_params)
        return workspace

    def _get_workspace_status(self) -> str:
        assert self.workspace is not None, 'Workspace is not initialized'
        provider_metadata = json.loads(self.workspace.instance.info.provider_metadata)
        return provider_metadata.get('status', 'unknown')

    def _construct_api_url(self, port: int) -> str:
        assert self.workspace is not None, 'Workspace is not initialized'
        node_domain = json.loads(self.workspace.instance.info.provider_metadata)[
            'nodeDomain'
        ]
        return f'https://{port}-{self.workspace.id}.{node_domain}'

    def _get_action_execution_server_host(self) -> str:
        return self.api_url

    def _start_action_execution_server(self) -> None:
        assert self.workspace is not None, 'Workspace is not initialized'
        self.workspace.process.exec(
            f'mkdir -p {self.config.workspace_mount_path_in_sandbox}'
        )

        command_args = get_action_execution_server_startup_command(
            server_port=self._sandbox_port,
            plugins=self.plugins,
            app_config=self.config,
        )
        start_command: str = ' '.join(command_args)

        exec_session_id = 'action-execution-server'
        self.workspace.process.create_session(exec_session_id)
        self.workspace.process.execute_session_command(
            exec_session_id,
            SessionExecuteRequest(command='cd /openhands/code', var_async=True),
        )

        exec_command = self.workspace.process.execute_session_command(
            exec_session_id,
            SessionExecuteRequest(command=start_command, var_async=True),
        )

        self.log('debug', f'exec_command_id: {exec_command.cmd_id}')

    @tenacity.retry(
        stop=tenacity.stop_after_delay(120) | stop_if_should_exit(),
        wait=tenacity.wait_fixed(1),
        reraise=(ConnectionRefusedError,),
    )
    def _wait_until_alive(self):
        super().check_if_alive()

    async def _get_active_ports(self) -> Set[int]:
        """Get currently active ports in the workspace."""
        async with self._port_poll_lock:
            assert self.workspace is not None, 'Workspace is not initialized'
            # Using netstat to check for listening TCP ports
            output = self.workspace.process.execute_session_command(
                'port-poller-session',
                SessionExecuteRequest(
                    command="netstat -tln | grep ':[0-9].*LISTEN' | awk '{print $4}' | sed 's/.*://'",
                    var_async=False,
                ),
            )

            ports = set()
            # Assuming the output is in the result directly
            for line in output.output.splitlines():
                try:
                    port = int(line.strip())
                    # Between 3000 and 10000 and not the sandbox or vscode port
                    if 3000 <= port <= 10000 and port not in [
                        self._sandbox_port,
                        self._vscode_port,
                    ]:  # Filter ports in our range of interest
                        ports.add(port)
                except ValueError:
                    continue
            return ports

    def _create_proxy_url(self, port: int) -> str:
        """Create a proxy URL for the given port."""
        return self._construct_api_url(port)

    async def _monitor_ports(self):
        """Continuously monitor ports in the workspace and create proxies for new ports."""
        previous_ports: set[int] = set()
        self.log('info', 'Starting port monitoring...')

        while not self._closed:
            try:
                current_ports = await self._get_active_ports()

                # Check for newly opened ports
                new_ports = current_ports - previous_ports
                for port in new_ports:
                    proxy_url = self._create_proxy_url(port)
                    local_port = await self._start_proxy(port, proxy_url)
                    if (
                        local_port != -1
                    ):  # Only add proxy if port was successfully bound
                        self._port_proxies[port] = f'http://localhost:{local_port}'
                        self.log(
                            'info',
                            f'Opening proxy {port} -> localhost:{local_port} -> {proxy_url}',
                        )

                # Check for closed ports
                closed_ports = previous_ports - current_ports
                for port in closed_ports:
                    await self._stop_proxy(port)
                    proxy_url = self._port_proxies.pop(port, '')
                    if proxy_url:
                        self.log(
                            'info', f'Closing proxy for port {port} (was {proxy_url})'
                        )

                previous_ports = current_ports - {
                    p for p in new_ports if p not in self._port_proxies
                }  # Remove ports that weren't successfully proxied
                await asyncio.sleep(2)  # Sleep for 2 seconds before checking again

            except Exception as e:
                self.log('error', f'Error monitoring ports: {str(e)}')
                await asyncio.sleep(5)

    @property
    def web_hosts(self) -> dict[str, int]:
        """Return a dictionary of active web hosts and their ports."""
        return {url: port for port, url in self._port_proxies.items()}

    async def connect(self):
        self.send_status_message('STATUS$STARTING_RUNTIME')

        if self.attach_to_existing:
            self.workspace = self._get_workspace()

        if self.workspace is None:
            self.send_status_message('STATUS$PREPARING_CONTAINER')
            self.workspace = self._create_workspace()
            self.log('info', f'Created new workspace with id: {self.workspace_id}')

        if self._get_workspace_status() == 'stopped':
            self.log('info', 'Starting Daytona workspace...')
            self.workspace.start()

        self.api_url = self._construct_api_url(self._sandbox_port)

        if not self.attach_to_existing:
            self._start_action_execution_server()
            self.log(
                'info',
                f'Container started. Action execution server url: {self.api_url}',
            )
            # Install netstat in the workspace
            exec_session_id = 'port-poller-session'
            self.workspace.process.create_session(exec_session_id)
            output = self.workspace.process.execute_session_command(
                exec_session_id,
                SessionExecuteRequest(
                    command='sudo apt update && sudo apt install net-tools -y',
                    var_async=False,
                ),
            )

            self.log('info', f'Netstat installation output: {output.output}')

        self.log('info', 'Waiting for client to become ready...')
        self.send_status_message('STATUS$WAITING_FOR_CLIENT')
        self._wait_until_alive()

        if not self.attach_to_existing:
            self.setup_initial_env()

        self.log(
            'info',
            f'Container initialized with plugins: {[plugin.name for plugin in self.plugins]}',
        )

        # Start port monitoring after workspace is ready
        asyncio.create_task(self._monitor_ports())

        if not self.attach_to_existing:
            self.send_status_message(' ')
        self._runtime_initialized = True

    async def close(self):
        """Override close to cleanup proxy servers."""
        self._closed = True

        # Stop all proxy servers and wait for them to complete
        stop_tasks = [
            self._stop_proxy(port) for port in list(self._proxy_servers.keys())
        ]
        if stop_tasks:
            await asyncio.gather(*stop_tasks)

        # Clear all proxy records
        self._proxy_servers.clear()
        self._port_proxies.clear()

        super().close()

        if self.attach_to_existing:
            return

        if self.workspace:
            self.daytona.remove(self.workspace)

    @property
    def vscode_url(self) -> str | None:
        if self._vscode_url is not None:  # cached value
            return self._vscode_url
        token = super().get_vscode_token()
        if not token:
            self.log(
                'warning', 'Failed to get VSCode token while trying to get VSCode URL'
            )
            return None
        if not self.workspace:
            self.log(
                'warning', 'Workspace is not initialized while trying to get VSCode URL'
            )
            return None
        self._vscode_url = (
            self._construct_api_url(self._vscode_port)
            + f'/?tkn={token}&folder={self.config.workspace_mount_path_in_sandbox}'
        )

        self.log(
            'debug',
            f'VSCode URL: {self._vscode_url}',
        )

        return self._vscode_url
