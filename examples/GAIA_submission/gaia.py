import os

import datasets
import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import login
from scripts.run_agents import answer_questions
from scripts.text_inspector_tool import TextInspectorTool
from scripts.text_web_browser import (
    ArchiveSearchTool,
    FinderTool,
    FindNextTool,
    NavigationalSearchTool,
    PageDownTool,
    PageUpTool,
    SearchInformationTool,
    VisitTool,
)
from scripts.visual_qa import VisualQAGPT4Tool, visualizer

from smolagents import CodeAgent, HfApiModel, LiteLLMModel, ManagedAgent, ToolCallingAgent, OpenAIServerModel


load_dotenv(override=True)
login(os.getenv("HF_TOKEN"))

### IMPORTANT: EVALUATION SWITCHES

print("Make sure you deactivated Tailscale VPN, else some URLs will be blocked!")

OUTPUT_DIR = "output"
USE_OPEN_MODELS = False

SET = "validation"

custom_role_conversions = {"tool-response": "user"}
proprietary_model = OpenAIServerModel(
    "o3-mini",
    custom_role_conversions=custom_role_conversions,
    max_completion_tokens=8192
)

websurfer_model = proprietary_model

repo_id_llama3 = "meta-llama/Meta-Llama-3-70B-Instruct"
repo_id_command_r = "CohereForAI/c4ai-command-r-plus"
repo_id_gemma2 = "google/gemma-2-27b-it"
repo_id_llama = "meta-llama/Meta-Llama-3.1-70B-Instruct"

hf_model = HfApiModel(model=repo_id_llama)

model = hf_model if USE_OPEN_MODELS else proprietary_model

### LOAD EVALUATION DATASET

eval_ds = datasets.load_dataset("gaia-benchmark/GAIA", "2023_all")[SET]
eval_ds = eval_ds.rename_columns(
    {"Question": "question", "Final answer": "true_answer", "Level": "task"}
)


def preprocess_file_paths(row):
    if len(row["file_name"]) > 0:
        row["file_name"] = f"data/gaia/{SET}/" + row["file_name"]
    return row


eval_ds = eval_ds.map(preprocess_file_paths)

eval_df = pd.DataFrame(eval_ds)
print("Loaded evaluation dataset:")
print(pd.Series(eval_ds["task"]).value_counts())

### BUILD AGENTS & TOOLS


text_limit = 100000
ti_tool = TextInspectorTool(websurfer_model, text_limit)

WEB_TOOLS = [
    SearchInformationTool(),
    NavigationalSearchTool(),
    VisitTool(),
    PageUpTool(),
    PageDownTool(),
    FinderTool(),
    FindNextTool(),
    ArchiveSearchTool(),
    TextInspectorTool(websurfer_model, text_limit),
]

surfer_agent = ToolCallingAgent(
    model=websurfer_model,
    tools=WEB_TOOLS,
    max_steps=20,
    verbosity_level=2,
    # grammar = DEFAULT_JSONAGENT_REGEX_GRAMMAR,
    planning_interval=4,
)

search_agent = ManagedAgent(
    surfer_agent,
    "web_search",
    description="""A team member that will browse the internet to answer your question.
Ask him for all your web-search related questions, but he's unable to do problem-solving.
Provide him as much context as possible, in particular if you need to search on a specific timeframe!
And don't hesitate to provide him with a complex search task, like finding a difference between two webpages.
Your request must be a real sentence, not a google search! Like "Find me this information (...)" rather than a few keywords.
""",
    additional_prompting= """You can navigate to .txt online files.
If a non-html page is in another format, especially .pdf, use tool 'inspect_file_as_text' to download and inspect it.
Additionally, if after some searching you find out that you need more information to answer the question, you can use `final_answer` with your request for clarification as argument to request for more information.""",
    provide_run_summary=True
)

TASK_SOLVING_TOOLBOX = [
    visualizer,  # VisualQATool(),
    ti_tool,
]


manager_agent = CodeAgent(
    model=model,
    tools=TASK_SOLVING_TOOLBOX,
    max_steps=12,
    verbosity_level=1,
    # grammar=DEFAULT_CODEAGENT_REGEX_GRAMMAR,
    additional_authorized_imports=[
        "requests",
        "zipfile",
        "os",
        "pandas",
        "numpy",
        "sympy",
        "json",
        "bs4",
        "pubchempy",
        "xml",
        "yahoo_finance",
        "Bio",
        "sklearn",
        "scipy",
        "pydub",
        "io",
        "PIL",
        "chess",
        "PyPDF2",
        "pptx",
        "torch",
        "datetime",
        "fractions",
        "csv"
    ],
    planning_interval=4,
    managed_agents=[search_agent]
)

### EVALUATE

results = answer_questions(
    eval_ds,
    manager_agent,
    "code_o1_preview_01-02_text",
    output_folder=f"{OUTPUT_DIR}/{SET}",
    visual_inspection_tool = VisualQAGPT4Tool(),
    text_inspector_tool = ti_tool,
    reformulation_model=model,
)
