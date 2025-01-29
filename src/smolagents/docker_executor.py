import base64
import pickle
import re
import tarfile
import textwrap
from io import BytesIO
from typing import Any, List, Tuple

from PIL import Image

from .tool_validation import validate_tool_attributes
from .tools import Tool
from .utils import BASE_BUILTIN_MODULES, instance_to_source


class DockerExecutor:
    """
    Executes Python code within a Docker container.
    """

    def __init__(self, additional_imports: List[str], tools: List[Tool], logger, host="127.0.0.1", port=65432):
        """
        Initializes the Docker executor.

        Args:
            additional_imports (List[str]): List of additional Python packages to install.
            tools (List[Tool]): List of tools to make available in the execution environment.
            logger: Logger for logging messages.
            host (str): Host IP to bind the container to.
            port (int): Port to bind the container to.
        """
        try:
            import docker
            from docker.models.containers import Container
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                """Please install 'docker' extra to use DockerExecutor: pip install `pip install "smolagents[docker]"`"""
            )
        try:
            self.client = docker.from_env()
            self.client.ping()
        except docker.errors.DockerException:
            raise RuntimeError("Could not connect to Docker daemon. Please ensure Docker is installed and running.")
        try:
            self.container: Container = self.client.containers.run(
                "python:3.12-slim",
                command=["sleep", "infinity"],
                detach=True,
                ports={f"{port}/tcp": (host, port)},
                volumes={"/tmp/smolagents": {"bind": "/app", "mode": "rw"}},
            )
        except docker.errors.DockerException as e:
            raise RuntimeError(f"Failed to create Docker container: {e}")

        self.logger = logger
        self.final_answer_pattern = re.compile(r"^final_answer\((.*)\)$")
        self.final_answer = False

        # Install additional imports
        for imp in additional_imports:
            exit_code, output = self.container.exec_run(f"pip install {imp}")
            if exit_code != 0:
                raise Exception(f"Error installing {imp}: {output.decode()}")

        # Generate and inject tool definitions
        tool_definition_code = self._generate_tool_code(tools)
        self._inject_code_into_container("/app/tools.py", tool_definition_code)
        exit_code, output = self.container.exec_run("python /app/tools.py")
        if exit_code != 0:
            raise Exception(f"Tools setup failed: {output.decode()}")

    def _generate_tool_code(self, tools: List[Tool]) -> str:
        """
        Generates Python code to define and instantiate tools.

        Args:
            tools (List[Tool]): List of tools to generate code for.

        Returns:
            str: Generated Python code.
        """
        tool_codes = []
        for tool in tools:
            validate_tool_attributes(tool.__class__, check_imports=False)
            tool_code = instance_to_source(tool, base_cls=Tool)
            tool_code = tool_code.replace("from smolagents.tools import Tool", "")
            tool_code += f"\n{tool.name} = {tool.__class__.__name__}()\n"
            tool_codes.append(tool_code)

        tool_definition_code = "\n".join([f"import {module}" for module in BASE_BUILTIN_MODULES])
        tool_definition_code += "\nfrom typing import Any"
        tool_definition_code += textwrap.dedent("""
        class Tool:
            def __call__(self, *args, **kwargs):
                return self.forward(*args, **kwargs)

            def forward(self, *args, **kwargs):
                pass  # To be implemented in child class
        """)
        tool_definition_code += "\n\n".join(tool_codes)
        return tool_definition_code

    def _inject_code_into_container(self, path: str, code: str):
        """
        Injects Python code into the Docker container at the specified path.

        Args:
            path (str): Path inside the container where the code will be written.
            code (str): Python code to write.
        """
        stream = BytesIO()
        with tarfile.open(fileobj=stream, mode="w") as tar:
            code_data = code.encode("utf-8")
            info = tarfile.TarInfo(name=path.lstrip("/"))
            info.size = len(code_data)
            tar.addfile(info, BytesIO(code_data))
        stream.seek(0)
        self.container.put_archive("/", stream)

    def __call__(self, code_action: str, additional_args: dict) -> Tuple[Any, str, bool]:
        """
        Executes the provided Python code in the Docker container.

        Args:
            code_action (str): Python code to execute.
            additional_args (dict): Additional arguments to pass to the code.

        Returns:
            Tuple[Any, str, bool]: A tuple containing the result, execution logs, and a flag indicating if this is a final answer.
        """
        # Handle additional_args by pickling into the container
        if additional_args:
            state_path = "/app/state.pkl"
            with BytesIO() as bio:
                pickle.dump(additional_args, bio)
                bio.seek(0)
                self._inject_code_into_container(state_path, bio.getvalue().decode("latin1"))
                setup_code = textwrap.dedent(f"""
                    import pickle
                    with open('{state_path}', 'rb') as f:
                        state = pickle.load(f)
                    locals().update(state)
                """)
                code_action = setup_code + code_action

        # Check if this is a final answer
        if self.final_answer_pattern.match(code_action):
            self.final_answer = True

        # Inject and execute the code
        code_action = "from tools import *\n" + code_action
        self._inject_code_into_container("/app/code.py", code_action)
        exit_code, output = self.container.exec_run(
            "python /app/code.py",
            demux=True,  # Separate stdout and stderr
        )
        stdout, stderr = output
        stdout = stdout.decode() if stdout else ""
        stderr = stderr.decode() if stderr else ""
        execution_logs = stdout + "\n" + stderr

        if exit_code != 0:
            raise ValueError(f"Code execution failed:\n{execution_logs}")

        # Parse output for final answer or specific results
        result = self._parse_output(stdout)
        return result, execution_logs, self.final_answer

    def _parse_output(self, stdout: str) -> Any:
        """
        Parses the output from the executed code.

        Args:
            stdout (str): Standard output from the executed code.

        Returns:
            Any: Parsed result (e.g., image, text, etc.).
        """
        if "IMAGE_BASE64:" in stdout:
            img_data = stdout.split("IMAGE_BASE64:")[1].split("\n")[0]
            return Image.open(BytesIO(base64.b64decode(img_data)))
        return stdout

    def __del__(self):
        """
        Cleans up the Docker container when the executor is no longer needed.
        """
        if hasattr(self, "container"):
            self.container.stop()
            self.container.remove()


__all__ = ["DockerExecutor"]
