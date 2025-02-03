import logging
from unittest import TestCase

import docker
from PIL import Image

from smolagents.docker_executor import DockerExecutor


class TestDockerExecutor(TestCase):
    def setUp(self):
        self.logger = logging.getLogger("DockerExecutorTest")
        self.executor = DockerExecutor(
            additional_imports=["pillow"],  # Ensure PIL is installed
            tools=[],
            logger=self.logger,
        )

    def test_initialization(self):
        """Check if DockerExecutor initializes without errors"""
        self.assertIsNotNone(self.executor.container, "Container should be initialized")

    def test_execute_basic_code(self):
        """Test executing a simple print statement"""
        code_action = 'print("Hello, Docker!")'
        result, logs, final_answer = self.executor(code_action, {})

        self.assertIn("Hello, Docker!", logs, "Output should contain 'Hello, Docker!'")
        self.assertFalse(final_answer, "Final answer flag should be False")

    def test_execute_image_output(self):
        """Test execution that returns a base64 image"""
        code_action = """
import base64
from PIL import Image
from io import BytesIO

img = Image.new("RGB", (10, 10), (255, 0, 0))
buffer = BytesIO()
img.save(buffer, format="PNG")
encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
print("IMAGE_BASE64:" + encoded)
"""
        result, logs, final_answer = self.executor(code_action, {})

        self.assertIsInstance(result, Image.Image, "Result should be a PIL Image")

    def test_syntax_error_handling(self):
        """Test handling of syntax errors"""
        code_action = 'print("Missing Parenthesis'  # Syntax error
        with self.assertRaises(ValueError) as context:
            self.executor(code_action, {})

        self.assertIn("SyntaxError", str(context.exception), "Should raise a syntax error")

    def test_cleanup_on_deletion(self):
        """Test if Docker container stops and removes on deletion"""
        container_id = self.executor.container.id
        del self.executor  # Trigger cleanup

        client = docker.from_env()
        containers = [c.id for c in client.containers.list(all=True)]
        self.assertNotIn(container_id, containers, "Container should be removed")
