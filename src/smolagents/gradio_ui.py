#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import mimetypes
import os
import re
import shutil
from typing import Optional

from .agents import ActionStep, AgentStepLog, MultiStepAgent
from .types import AgentAudio, AgentImage, AgentText, handle_agent_output_types
from .utils import _is_package_available


def pull_messages_from_step(step_log: AgentStepLog, test_mode: bool = True):
    """Extract ChatMessage objects from agent steps with proper nesting"""
    if isinstance(step_log, ActionStep):
        # First yield the thought/reasoning from the LLM
        if step_log.llm_output is not None and not test_mode:
            yield gr.ChatMessage(role="assistant", content=step_log.llm_output.strip())

        # For tool calls, create a parent message 
        if step_log.tool_calls is not None:
            first_tool_call = step_log.tool_calls[0]
            used_code = first_tool_call.name == "python_interpreter"
            parent_id = f"call_{len(step_log.tool_calls)}"
            
            # Calculate token information
            token_str = ""
            if hasattr(step_log, "input_token_count") and hasattr(step_log, "output_token_count"):
                token_str = f"Input-tokens:{step_log.input_token_count:,} Output-tokens:{step_log.output_token_count:,}"
            
            step_str = f"Step {step_log.step}" if step_log.step is not None else ""
            
            timing_str = " | ".join(filter(None, [step_str, token_str]))
            
            # Tool call becomes the parent message with timing info
            content = first_tool_call.arguments
            if used_code:
                content = f"```python\n{content}\n```"
            
            parent_message = gr.ChatMessage(
                role="assistant",
                content=content,
                metadata={
                    "title": f"🛠️ Used tool {first_tool_call.name}" + (f" ({timing_str})" if timing_str else ""),
                    "id": parent_id,
                    "status": "pending",
                    "duration": step_log.duration if step_log.duration else None
                }
            )
            yield parent_message

            # Nest execution logs under the tool call if they exist
            if step_log.observations is not None:
                log_content = step_log.observations
                yield gr.ChatMessage(
                    role="assistant",
                    content=f"Execution logs:\n{log_content}",
                    metadata={
                        "title": "📝 Execution Logs",
                        "parent_id": parent_id,
                        "status": "done"
                    }
                )
                
            # Nest any errors under the tool call
            if step_log.error is not None:
                yield gr.ChatMessage(
                    role="assistant", 
                    content=str(step_log.error),
                    metadata={
                        "title": "💥 Error",
                        "parent_id": parent_id,
                        "status": "done"
                    }
                )

            # Update parent message metadata to done status (without yielding a new message)
            parent_message.metadata["status"] = "done"

        # Handle standalone errors (not from tool calls)
        elif step_log.error is not None:
            yield gr.ChatMessage(
                role="assistant",
                content=str(step_log.error),
                metadata={"title": "💥 Error"}
            )


def stream_to_gradio(
    agent,
    task: str,
    reset_agent_memory: bool = False,
    additional_args: Optional[dict] = None,
):
    """Runs an agent with the given task and streams the messages from the agent as gradio ChatMessages."""
    total_input_tokens = 0
    total_output_tokens = 0

    for step_log in agent.run(
        task, stream=True, reset=reset_agent_memory, additional_args=additional_args
    ):
        # Track tokens if model provides them
        if hasattr(agent.model, "last_input_token_count"):
            total_input_tokens += agent.model.last_input_token_count
            total_output_tokens += agent.model.last_output_token_count
            if isinstance(step_log, ActionStep):
                step_log.input_token_count = agent.model.last_input_token_count
                step_log.output_token_count = agent.model.last_output_token_count
        
        for message in pull_messages_from_step(step_log, test_mode=test_mode):
            yield message

    final_answer = step_log  # Last log is the run's final_answer
    final_answer = handle_agent_output_types(final_answer)

    if isinstance(final_answer, AgentText):
        yield gr.ChatMessage(
            role="assistant",
            content=f"**Final answer:**\n{final_answer.to_string()}\n",
        )
    elif isinstance(final_answer, AgentImage):
        yield gr.ChatMessage(
            role="assistant",
            content={"path": final_answer.to_string(), "mime_type": "image/png"},
        )
    elif isinstance(final_answer, AgentAudio):
        yield gr.ChatMessage(
            role="assistant",
            content={"path": final_answer.to_string(), "mime_type": "audio/wav"},
        )
    else:
        yield gr.ChatMessage(role="assistant", content=str(final_answer))


class GradioUI:
    """A one-line interface to launch your agent in Gradio"""

    def __init__(self, agent: MultiStepAgent, file_upload_folder: str | None = None):
        if not _is_package_available("gradio"):
            raise ModuleNotFoundError(
                "Please install 'gradio' extra to use the GradioUI: `pip install 'smolagents[audio]'`"
            )
        self.agent = agent
        self.file_upload_folder = file_upload_folder
        if self.file_upload_folder is not None:
            if not os.path.exists(file_upload_folder):
                os.mkdir(file_upload_folder)

    def interact_with_agent(self, prompt, messages):
        import gradio as gr

        messages.append(gr.ChatMessage(role="user", content=prompt))
        yield messages
        for msg in stream_to_gradio(self.agent, task=prompt, reset_agent_memory=False):
            messages.append(msg)
            yield messages
        yield messages

    def upload_file(
        self,
        file,
        file_uploads_log,
        allowed_file_types=[
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "text/plain",
        ],
    ):
        """
        Handle file uploads, default allowed types are .pdf, .docx, and .txt
        """
        import gradio as gr

        if file is None:
            return gr.Textbox("No file uploaded", visible=True), file_uploads_log

        try:
            mime_type, _ = mimetypes.guess_type(file.name)
        except Exception as e:
            return gr.Textbox(f"Error: {e}", visible=True), file_uploads_log

        if mime_type not in allowed_file_types:
            return gr.Textbox("File type disallowed", visible=True), file_uploads_log

        # Sanitize file name
        original_name = os.path.basename(file.name)
        sanitized_name = re.sub(
            r"[^\w\-.]", "_", original_name
        )  # Replace any non-alphanumeric, non-dash, or non-dot characters with underscores

        type_to_ext = {}
        for ext, t in mimetypes.types_map.items():
            if t not in type_to_ext:
                type_to_ext[t] = ext

        # Ensure the extension correlates to the mime type
        sanitized_name = sanitized_name.split(".")[:-1]
        sanitized_name.append("" + type_to_ext[mime_type])
        sanitized_name = "".join(sanitized_name)

        # Save the uploaded file to the specified folder
        file_path = os.path.join(self.file_upload_folder, os.path.basename(sanitized_name))
        shutil.copy(file.name, file_path)

        return gr.Textbox(f"File uploaded: {file_path}", visible=True), file_uploads_log + [file_path]

    def log_user_message(self, text_input, file_uploads_log):
        return (
            text_input
            + (
                f"\nYou have been provided with these files, which might be helpful or not: {file_uploads_log}"
                if len(file_uploads_log) > 0
                else ""
            ),
            "",
        )

    def launch(self):
        with gr.Blocks(fill_height=True) as demo:
            stored_messages = gr.State([])
            file_uploads_log = gr.State([])
            chatbot = gr.Chatbot(
                label="Agent",
                type="messages",
                avatar_images=(
                    None,
                    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/smolagents/mascot_smol.png",
                ),
                resizeable=True,
                scale=1
            )
            # If an upload folder is provided, enable the upload feature
            if self.file_upload_folder is not None:
                upload_file = gr.File(label="Upload a file")
                upload_status = gr.Textbox(label="Upload Status", interactive=False, visible=False)
                upload_file.change(
                    self.upload_file,
                    [upload_file, file_uploads_log],
                    [upload_status, file_uploads_log],
                )
            text_input = gr.Textbox(lines=1, label="Chat Message")
            text_input.submit(
                self.log_user_message,
                [text_input, file_uploads_log],
                [stored_messages, text_input],
            ).then(self.interact_with_agent, [stored_messages, chatbot], [chatbot])

        demo.launch(debug=True, share=True)

__all__ = ["stream_to_gradio", "GradioUI"]
