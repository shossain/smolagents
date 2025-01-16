import docker
from typing import List, Optional
import warnings
import socket
from smolagents.tools import Tool

class DockerPythonInterpreter:
    def __init__(self):
        self.container = None
        try:
            self.client = docker.from_env()
            self.client.ping()
        except docker.errors.DockerException:
            raise RuntimeError(
                "Could not connect to Docker daemon. Please ensure Docker is installed and running."
            )
        try:
            self.container = self.client.containers.run(
                "pyrunner:latest",
                ports={"65432/tcp": 65432},
                detach=True,
                remove=True,
            )
        except docker.errors.DockerException as e:
            raise RuntimeError(f"Failed to create Docker container: {e}")
    def stop(self):
        """Cleanup: Stop and remove container when object is destroyed"""
        if self.container:
            try:
                self.container.kill()  # can consider .stop(), but this is faster
            except Exception as e:
                warnings.warn(f"Failed to stop Docker container: {e}")
    def execute(self, code: str, tools: Optional[List[Tool]] = None) -> str:
        """
        Execute Python code in the container and return stdout and stderr
        """
        if tools is not None:
            tool_instance = tools[0]()
            import_code = f"""
module_path = '{tool_instance.__class__.__module__}'
class_name = '{tool_instance.__class__.__name__}'
import importlib
module = importlib.import_module(module_path)
web_search = getattr(module, class_name)()
"""
            code = import_code + "\n" + code
        try:
            # Connect to the server running inside the container
            with socket.create_connection(("localhost", 65432)) as sock:
                sock.sendall(code.encode("utf-8"))
                output = sock.recv(4096)
                return output.decode("utf-8")
        except Exception as e:
            return f"Error executing code: {str(e)}"

__all__ = ["DockerPythonInterpreter"]