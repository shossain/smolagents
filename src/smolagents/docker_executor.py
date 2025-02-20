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
import socket

class DockerExecutor:
    """
    Executes Python code within a Docker container.
    """

    def __init__(self, additional_imports: List[str], tools: List[Tool], logger, initial_state, host="127.0.0.1", port=65432):
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

        # # Generate and inject tool definitions
        # tool_definition_code = self._generate_tool_code(tools)
        # self._inject_code_into_container("/app/tools.py", tool_definition_code)
        # exit_code, output = self.container.exec_run("python /app/tools.py")
        # if exit_code != 0:
        #     raise Exception(f"Tools setup failed: {output.decode()}")
        # Start a single persistent Python interactive session
        self.python_session = self.container.exec_run(
            "python -i",
            stdin=True,
            stream=True,  # To handle continuous I/O
            demux=True  # Get separate stdout/stderr streams
        )
        # Connect with socket
        self.sock = socket.create_connection(("localhost", 65432))



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

    def send_variables_to_server(self, state):
        """Pickle state to server"""
        state_path = "/app/state.pkl"

        pickle.dump(state, state_path)
        remote_unloading_code = """import pickle
import os
print("File path", os.path.getsize('/home/state.pkl'))
with open('/home/state.pkl', 'rb') as f:
pickle_dict = pickle.load(f)
locals().update({key: value for key, value in pickle_dict.items()})
"""
        execution = self.run_code_raise_errors(remote_unloading_code)
        execution_logs = "\n".join([str(log) for log in execution.logs.stdout])
        self.logger.log(execution_logs, 1)

    def __call__(self, code_action: str) -> Tuple[Any, str, bool]:
        """Check if code is a final answer and run it accordingly"""
        return self.run_code(code_action, return_final_answer=self.final_answer_pattern.match(code_action))

    def run_code(self, code_action: str, return_final_answer=False) -> Tuple[Any, str, bool]:
        """
        Executes the provided Python code in the Docker container.

        Args:
            code_action (str): Python code to execute.

        Returns:
            Tuple[Any, str, bool]: A tuple containing the result, execution logs, and a flag indicating if this is a final answer.
        """

        # Inject and execute the code
        marked_code = f"""
{code_action}
print('---END---')
"""
        if return_final_answer:
            marked_code += """with open('/app/result.pkl', 'wb') as f:
    pickle.dump(_result, f)
print('---OUTPUT_END---')
"""

        print("FLAGG", dir(self.python_session.output))

        self.python_session.write(marked_code.encode('utf-8'))
        self.python_session.flush()

        # Read output until we see our marker
        output = ""
        for line in self.python_session.output:
            if line:
                output += line.decode()
            if "---END---" in output:
                break
        with open('/tmp/smolagents/result.pkl', 'rb') as f:
            result = pickle.load(f)
        print("OK reached eth end")

        # Return both the object and the printed output
        output = output.replace("---END---", "").strip()
        return result, output

        stderr=False

        if stderr:
            raise ValueError(f"Code execution failed:\n{output}")
        else:
            if return_final_answer:
                while "---OUTPUT_END---" not in output:
                    out, err = self.python_session.output  # Get both streams
                    if out:
                        output += out.decode()
                    if err:
                        stderr += err.decode()

                # Load output for final answer or specific results
                with open('/tmp/smolagents/result.pkl', 'rb') as f:  # Note path on host side
                    result = pickle.load(f)
            else:
                result = None
            return result, execution_logs, return_final_answer

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
