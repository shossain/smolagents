from typing import Optional

from smolagents import HfApiModel, LiteLLMModel, TransformersModel, tool
from smolagents.agents import CodeAgent, ToolCallingAgent


# Choose which inference type to use!

available_inferences = ["hf_api", "transformers", "ollama", "litellm"]
chosen_inference = "transformers"

print(f"Chose model: '{chosen_inference}'")

model = LiteLLMModel(
    model_id="openai/deepseek-ai/DeepSeek-R1",
    api_base="https://huggingface.co/api/inference-proxy/together",
    api_key=""
)

@tool
def get_weather(location: str, celsius: Optional[bool] = False) -> str:
    """
    Get weather in the next days at given location.
    Secretly this tool does not care about the location, it hates the weather everywhere.

    Args:
        location: the location
        celsius: the temperature
    """
    return "The weather is UNGODLY with torrential rains and temperatures below -10°C"


agent = ToolCallingAgent(tools=[get_weather], model=model)

print("ToolCallingAgent:", agent.run("What's the weather like in Paris?"))

agent = CodeAgent(tools=[get_weather], model=model)

print("CodeAgent:", agent.run("What's the weather like in Paris?"))
