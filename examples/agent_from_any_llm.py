from typing import Optional

from smolagents import HfApiModel, LiteLLMModel, TransformersModel, tool
from smolagents.agents import CodeAgent, ToolCallingAgent


# Choose which inference type to use!

available_inferences = ["hf_api", "transformers", "ollama", "litellm"]
chosen_inference = "transformers"

print(f"Chose model: '{chosen_inference}'")

model = LiteLLMModel(
    model_id="hosted_vllm//fsx/anton/deepseek-r1-checkpoint",
    api_base="http://ip-26-0-165-202:8000/v1/",
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



agent = CodeAgent(tools=[get_weather], model=model, verbosity_level=2)
print("CodeAgent:", agent.run("What's the weather like in Paris?"))
