from openinference.instrumentation.smolagents import SmolagentsInstrumentor
from phoenix.otel import register
import litellm

litellm._turn_on_debug()
litellm.drop_params = True


register()
SmolagentsInstrumentor().instrument(skip_dep_check=True)

# from opentelemetry import trace
# from opentelemetry.sdk.trace import TracerProvider
# from opentelemetry.sdk.trace.export import BatchSpanProcessor

# from openinference.instrumentation.smolagents import SmolagentsInstrumentor
# from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
# from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

# endpoint = "http://0.0.0.0:6006/v1/traces"
# trace_provider = TracerProvider()
# trace_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))

# SmolagentsInstrumentor().instrument(tracer_provider=trace_provider)



from smolagents import (
    CodeAgent,
    DuckDuckGoSearchTool,
    HfApiModel,
    LiteLLMModel,
    ToolCallingAgent,
    VisitWebpageTool,
)


# Then we run the agentic part!
# model = HfApiModel()
model = LiteLLMModel(
    "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
    custom_role_conversions = {"tool-call": "assistant", "tool-response": "user"}
    # 'bedrock/us.meta.llama3-3-70b-instruct-v1:0',
    # custom_role_conversions = {"tool-response": "tool"}
)

search_agent = ToolCallingAgent(
    tools=[DuckDuckGoSearchTool(), VisitWebpageTool()],
    model=model,
    name="search_agent",
    description="This is an agent that can do web search.",
)

manager_agent = CodeAgent(
    tools=[],
    model=model,
    managed_agents=[search_agent],
)
manager_agent.run("If the US keeps it 2024 growth rate, how many years would it take for the GDP to double?")
