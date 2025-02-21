import base64
import json
import pickle
import re
import time
from pathlib import Path
from typing import Any, List, Tuple

import docker
import requests


class DockerExecutor:
    """
    Executes Python code using Jupyter Kernel Gateway in a Docker container.
    """

    def __init__(
        self,
        additional_imports: List[str],
        tools,
        logger,
        initial_state: dict = None,
        host: str = "127.0.0.1",
        port: int = 8888,
    ):
        """
        Initialize the Docker-based Jupyter Kernel Gateway executor.
        """
        self.logger = logger
        self.host = host
        self.port = port
        self.final_answer_pattern = re.compile(r"^final_answer\((.*)\)$")

        # Initialize Docker
        try:
            self.client = docker.from_env()
        except docker.errors.DockerException as e:
            raise RuntimeError("Could not connect to Docker daemon") from e

        # Build and start container
        try:
            # Build the Docker image
            self.logger.info("Building Docker image...")
            dockerfile_path = Path(__file__).parent / "Dockerfile"
            if not dockerfile_path.exists():
                with open(dockerfile_path, "w") as f:
                    f.write("""FROM python:3.12-slim

RUN pip install jupyter_kernel_gateway requests numpy pandas
RUN pip install jupyter_client notebook

EXPOSE 8888
CMD ["jupyter", "kernelgateway", "--KernelGatewayApp.ip='0.0.0.0'", "--KernelGatewayApp.port=8888", "--KernelGatewayApp.allow_origin='*'"]
""")
            image, build_logs = self.client.images.build(
                path=str(dockerfile_path.parent), dockerfile=str(dockerfile_path), tag="jupyter-kernel"
            )
            # Run the container

            self.logger.info(f"Starting container on {host}:{port}...")
            self.container = self.client.containers.run(
                "jupyter-kernel", ports={"8888/tcp": (host, port)}, detach=True
            )
            # Wait for kernel gateway to start

            self.logger.info("Waiting for kernel gateway to start...")
            time.sleep(2)
            # Initialize kernel session

            self.base_url = f"http://{host}:{port}"
            # Create new kernel via HTTP

            r = requests.post(f"{self.base_url}/api/kernels")
            if r.status_code != 201:
                error_details = {
                    "status_code": r.status_code,
                    "headers": dict(r.headers),
                    "url": r.url,
                    "body": r.text,
                    "request_method": r.request.method,
                    "request_headers": dict(r.request.headers),
                    "request_body": r.request.body,
                }
                self.logger.error(f"Failed to create kernel. Details: {json.dumps(error_details, indent=2)}")
                raise RuntimeError(f"Failed to create kernel: Status {r.status_code}\nResponse: {r.text}") from None

            self.kernel_id = r.json()["id"]
            # Initialize WebSocket connection

            from websocket import create_connection

            ws_url = f"ws://{host}:{port}/api/kernels/{self.kernel_id}/channels"
            self.ws = create_connection(ws_url)
            # Install additional packages

            for package in additional_imports:
                self.execute_code(f"!pip install {package}")

            # Initialize state if provided
            if initial_state:
                self.send_variables_to_kernel(initial_state)

            self.logger.info(f"Container {self.container.short_id} is running with kernel {self.kernel_id}")

        except Exception as e:
            self.cleanup()
            # Re-raise with the original traceback preserved
            raise RuntimeError(f"Failed to initialize Jupyter kernel: {e}") from e

    def execute_code(self, code: str) -> str:
        """Execute code and return output"""
        result, output, _ = self.run_code(code)
        return output

    def __call__(self, code_action: str) -> Tuple[Any, str, bool]:
        """Check if code is a final answer and run it accordingly"""
        return self.run_code(code_action, return_final_answer=bool(self.final_answer_pattern.match(code_action)))

    def run_code(self, code_action: str, return_final_answer: bool = False) -> Tuple[Any, str, bool]:
        """
        Execute code and return result based on whether it's a final answer.
        """
        try:
            if return_final_answer:
                match = self.final_answer_pattern.match(code_action)
                if match:
                    result_expr = match.group(1)
                    wrapped_code = f"""
    import pickle, base64
    _result = {result_expr}
    print("RESULT_PICKLE:" + base64.b64encode(pickle.dumps(_result)).decode())
    """
            else:
                wrapped_code = code_action

            # Send execute request
            msg_id = self._send_execute_request(wrapped_code)

            # Collect output and results
            outputs = []
            result = None
            waiting_for_idle = False

            while True:
                msg = json.loads(self.ws.recv())
                msg_type = msg.get("msg_type", "")
                parent_msg_id = msg.get("parent_header", {}).get("msg_id")

                # Only process messages related to our execute request
                if parent_msg_id != msg_id:
                    continue

                if msg_type == "stream":
                    text = msg["content"]["text"]
                    if return_final_answer and text.startswith("RESULT_PICKLE:"):
                        pickle_data = text[len("RESULT_PICKLE:") :].strip()
                        result = pickle.loads(base64.b64decode(pickle_data))
                        waiting_for_idle = True
                    else:
                        outputs.append(text)
                elif msg_type == "error":
                    traceback = msg["content"].get("traceback", [])
                    raise RuntimeError("\n".join(traceback)) from None
                elif msg_type == "status" and msg["content"]["execution_state"] == "idle":
                    if not return_final_answer or waiting_for_idle:
                        break

            return result, "".join(outputs), return_final_answer

        except Exception as e:
            self.logger.error(f"Code execution failed: {e}")
            raise

    def send_variables_to_kernel(self, variables: dict):
        """
        Send variables to the kernel namespace using pickle.
        """
        pickled_vars = base64.b64encode(pickle.dumps(variables)).decode()
        code = f"""
import pickle, base64
vars_dict = pickle.loads(base64.b64decode('{pickled_vars}'))
globals().update(vars_dict)
"""
        self.run_code(code)

    def get_variable_from_kernel(self, var_name: str) -> Any:
        """
        Retrieve a variable from the kernel namespace.
        """
        code = f"""
import pickle, base64
print("RESULT_PICKLE:" + base64.b64encode(pickle.dumps({var_name})).decode())
"""
        result, _, _ = self.run_code(code, return_final_answer=True)
        return result

    def _send_execute_request(self, code: str) -> str:
        """Send code execution request to kernel."""
        import uuid

        # Generate a unique message ID
        msg_id = str(uuid.uuid4())

        # Create execute request
        execute_request = {
            "header": {
                "msg_id": msg_id,
                "username": "anonymous",
                "session": str(uuid.uuid4()),
                "msg_type": "execute_request",
                "version": "5.0",
            },
            "parent_header": {},
            "metadata": {},
            "content": {
                "code": code,
                "silent": False,
                "store_history": True,
                "user_expressions": {},
                "allow_stdin": False,
            },
        }

        self.ws.send(json.dumps(execute_request))
        return msg_id

    def cleanup(self):
        """Clean up resources."""
        try:
            if hasattr(self, "kernel_id"):
                self.session.delete(f"{self.base_url}/api/kernels/{self.kernel_id}")
            if hasattr(self, "container"):
                self.logger.info(f"Stopping and removing container {self.container.short_id}...")
                self.container.stop()
                self.container.remove()
                self.logger.info("Container cleanup completed")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

    def __del__(self):
        """Ensure cleanup on deletion."""
        self.cleanup()
