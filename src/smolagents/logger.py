import importlib
import json
from typing import Dict, List, Optional

from rich.console import Console
from rich.rule import Rule
from rich.syntax import Syntax


if importlib.util.find_spec("opentelemetry") is not None:
    from openinference.instrumentation.smolagents import SmolagentsInstrumentor
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    endpoint = "http://0.0.0.0:6006/v1/traces"
    trace_provider = TracerProvider()
    trace_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))

    SmolagentsInstrumentor().instrument(tracer_provider=trace_provider)

from smolagents.models import MessageRole
from smolagents.utils import ActionStep, LogLevel, PlanningStep, SystemPromptStep, TaskStep


console = Console()


class AgentLogger:
    def __init__(self, level: LogLevel = LogLevel.INFO):
        self.level = level
        self.steps = []
        self.console = Console()
        # todos:
        # - add a way to save/load logs to/from a file
        # - group agent logs and user logs
        # - add log configurations details

    def log(self, *args, level: LogLevel = LogLevel.INFO, **kwargs):
        """Logs a message to the console.

        Args:
            level (LogLevel, optional): Defaults to LogLevel.INFO.
        """
        if level <= self.level:
            self.console.print(*args, **kwargs)

    def log_step(self, step, position: int = None):
        """Logs an agent execution step for ulterior processing.

        Args:
            step (_type_): _description_
            position (int, optional): Position at which to insert the item.
                Defaults to None, in which case item is added to the end of the list.
                Should only be used to insert the system prompt at the start of the list.
        """
        if position is None:
            self.steps.append(step)
        else:
            assert position == 0, "Position should only be 0 for system prompt."
            self.steps = [step] + self.steps[1:]  # we replace the system prompt

    def reset(self):
        self.steps = []

    def get_succinct_logs(self):
        return [{key: value for key, value in log.items() if key != "agent_memory"} for log in self.steps]

    def write_inner_memory_from_logs(
        self,
        summary_mode: Optional[bool] = False,
        return_memory: Optional[bool] = False,
    ) -> List[Dict[str, str]]:
        """
        Reads past llm_outputs, actions, and observations or errors from the logs into a series of messages
        that can be used as input to the LLM. Adds a number of keywords (such as PLAN, error, etc) to help
        the LLM.
        """
        # todo: we might want to homogeneize all keywords?
        memory = []
        for i, step_log in enumerate(self.steps):
            if isinstance(step_log, SystemPromptStep):
                if not summary_mode:
                    thought_message = {
                        "role": MessageRole.SYSTEM,
                        "content": step_log.system_prompt.strip(),
                    }
                    memory.append(thought_message)

            elif isinstance(step_log, PlanningStep):
                thought_message = {
                    "role": MessageRole.ASSISTANT,
                    "content": "[FACTS LIST]:\n" + step_log.facts.strip(),
                }
                memory.append(thought_message)

                if not summary_mode:
                    thought_message = {
                        "role": MessageRole.ASSISTANT,
                        "content": "[PLAN]:\n" + step_log.plan.strip(),
                    }
                    memory.append(thought_message)

            elif isinstance(step_log, TaskStep):
                task_message = {
                    "role": MessageRole.USER,
                    "content": "New task:\n" + step_log.task,
                }
                memory.append(task_message)

            elif isinstance(step_log, ActionStep):
                if step_log.agent_memory is not None and return_memory:
                    thought_message = {
                        "role": MessageRole.SYSTEM,
                        "content": step_log.agent_memory,
                    }
                    memory.append(thought_message)
                if step_log.llm_output is not None and not summary_mode:
                    thought_message = {
                        "role": MessageRole.ASSISTANT,
                        "content": step_log.llm_output.strip(),
                    }
                    memory.append(thought_message)

                if step_log.tool_calls is not None:
                    tool_call_message = {
                        "role": MessageRole.ASSISTANT,
                        "content": str(
                            [
                                {
                                    "id": tool_call.id,
                                    "type": "function",
                                    "function": {
                                        "name": tool_call.name,
                                        "arguments": tool_call.arguments,
                                    },
                                }
                                for tool_call in step_log.tool_calls
                            ]
                        ),
                    }
                    memory.append(tool_call_message)

                if step_log.tool_calls is None and step_log.error is not None:
                    message_content = (
                        "Error:\n"
                        + str(step_log.error)
                        + "\nNow let's retry: take care not to repeat previous errors! If you have retried several times, try a completely different approach.\n"
                    )
                    tool_response_message = {
                        "role": MessageRole.ASSISTANT,
                        "content": message_content,
                    }
                if step_log.tool_calls is not None and (
                    step_log.error is not None or step_log.observations is not None
                ):
                    if step_log.error is not None:
                        message_content = (
                            "Error:\n"
                            + str(step_log.error)
                            + "\nNow let's retry: take care not to repeat previous errors! If you have retried several times, try a completely different approach.\n"
                        )
                    elif step_log.observations is not None:
                        message_content = f"Observation:\n{step_log.observations}"
                    tool_response_message = {
                        "role": MessageRole.TOOL_RESPONSE,
                        "content": f"Call id: {(step_log.tool_calls[0].id if getattr(step_log.tool_calls[0], 'id') else 'call_0')}\n"
                        + message_content,
                    }
                    memory.append(tool_response_message)

        return memory

    def replay(self, with_memory: bool = False):
        """Prints a replay of the agent's steps.

        Args:
            with_memory (bool, optional): If True, also displays the memory at each step. Defaults to False.
                Careful: will increase log length exponentially. Use only for debugging.
        """
        memory = self.write_inner_memory_from_logs(return_memory=with_memory)
        self.console.log("Replaying the agent's steps:")
        ix = 0
        for step in memory:
            role = step["role"].strip()
            if ix > 0 and role == "system":
                role == "memory"
            theme = "default"
            match role:
                case "assistant":
                    theme = "monokai"
                    ix += 1
                case "system":
                    theme = "monokai"
                case "tool-response":
                    theme = "github_dark"

            content = step["content"]
            try:
                content = eval(content)
            except Exception:
                content = [step["content"]]

            for substep_ix, item in enumerate(content):
                self.console.log(
                    Rule(
                        f"{role.upper()}, STEP {ix}, SUBSTEP {substep_ix + 1}/{len(content)}",
                        align="center",
                        style="orange",
                    ),
                    Syntax(
                        json.dumps(item, indent=4) if isinstance(item, dict) else str(item),
                        lexer="json",
                        theme=theme,
                        word_wrap=True,
                    ),
                )


__all__ = ["AgentLogger"]
