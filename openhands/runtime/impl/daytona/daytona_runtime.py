import json
from typing import Callable, Optional

import tenacity
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
DAYTONA_API_URL = (
    'https://stage.daytona.work/api'  # TODO: Parametrize and change to production
)


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
            server_url=DAYTONA_API_URL,
            target='eu',  # TODO: Parametrize
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

    # def _set_env_vars_in_workspace(self, exec_session_id: str) -> None:
    #     # Quickfix for env vars not being set in the workspace, TODO remove this
    #     for key, value in self._get_creation_env_vars().items():
    #         self.workspace.process.execute_session_command(
    #             exec_session_id,
    #             SessionExecuteRequest(
    #                 command=f'export {key}="{value}"', var_async=True
    #             ),
    #         )

    def _start_action_execution_server(self) -> None:
        self.workspace.process.exec(
            f'mkdir -p {self.config.workspace_mount_path_in_sandbox}'
        )

        start_command = get_action_execution_server_startup_command(
            server_port=self._sandbox_port,
            plugins=self.plugins,
            app_config=self.config,
        )
        start_command: str = ' '.join(start_command)

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

    async def connect(self):
        self.send_status_message('STATUS$STARTING_RUNTIME')

        if self.attach_to_existing:
            self.workspace: Optional[Workspace] = self._get_workspace()

        if self.workspace is None:
            self.send_status_message('STATUS$PREPARING_CONTAINER')
            self.workspace: Workspace = self._create_workspace()
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

        self.log('info', 'Waiting for client to become ready...')
        self.send_status_message('STATUS$WAITING_FOR_CLIENT')
        self._wait_until_alive()

        if not self.attach_to_existing:
            self.setup_initial_env()

        self.log(
            'info',
            f'Container initialized with plugins: {[plugin.name for plugin in self.plugins]}',
        )

        if not self.attach_to_existing:
            self.send_status_message(' ')
        self._runtime_initialized = True

    def close(self):
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
