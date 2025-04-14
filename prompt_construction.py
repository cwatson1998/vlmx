from PIL import Image
from pathlib import Path
from vlmx.agent import Agent, AgentConfig
from dotenv import load_dotenv
from vlmx.context_agent import ContextAwareAgent
import os
import tempfile
from vlmx.tool_use_agent import ToolUseAgent, TOOL_INSTRUCTION
from vlmx.artifact import ArtifactDisplayHandler, artifacts_to_prompt_parts, ArtifactCollector
from typing import Dict, Any, Optional, List, Union
import sys
from io import StringIO
from vlmx.utils import string_to_file, join_path, extract_code_from_string
from contextlib import contextmanager
from io import StringIO
from PIL import Image
from pathlib import Path


SYSTEM_INSTRUCTION = "Please follow the instructions in the prompt."


def construct_prompt(prompt_txt_path, content_dict, special_separator="<|>"):
    with open(prompt_txt_path, 'r') as f:
        prompt = f.read()
    for key, value in content_dict.items():
        if isinstance(value, str):
            # assert not os.path.exists(value), f"Expected a string value for {key}, but got a file path: {value}"
            prompt = prompt.replace(
                f"{special_separator}{key}{special_separator}", value)
    prompt_parts = []
    current_pos = 0
    while True:
        # Find next separator
        next_sep = prompt.find(special_separator, current_pos)
        if next_sep == -1:
            # Add remaining text
            if current_pos < len(prompt):
                prompt_parts.append(prompt[current_pos:])
            break

        # Add text before separator
        if current_pos < next_sep:
            prompt_parts.append(prompt[current_pos:next_sep])

        # Find end of key
        end_sep = prompt.find(
            special_separator, next_sep + len(special_separator))
        if end_sep == -1:
            raise ValueError(f"Unmatched separator at position {next_sep}")

        # Extract key and get corresponding image
        key = prompt[next_sep + len(special_separator):end_sep]
        if key in content_dict:
            prompt_parts.append(content_dict[key])
        else:
            raise ValueError(f"Key {key} not found in content_dict")

        current_pos = end_sep + len(special_separator)
    return prompt_parts


class HelperAgent(Agent):
    OUT_RESULT_PATH = "test.txt"

    def _make_system_instruction(self):
        return SYSTEM_INSTRUCTION

    def _make_prompt_parts(self, question: str):
        return question

    def parse_response(self, response):
        print("response:", response.text)
        # string_to_file(response.txt, "path.txt")


def as_bool(my_str):
    """Convert various string representations of boolean values to actual booleans.

    Args:
        my_str: String to convert to boolean

    Returns:
        bool: The boolean value

    Raises:
        ValueError: If the string cannot be converted to a boolean
    """
    # Handle string type
    if not isinstance(my_str, str):
        raise ValueError(f"Expected string, got {type(my_str)}")
    # Convert to lowercase and strip whitespace
    my_str = my_str.lower().strip()
    truthy_values = ["true", "t", "yes", "y"]
    falsey_values = ["false", "f", "no", "n"]
    if my_str in truthy_values:
        return True
    if my_str in falsey_values:
        return False
    maybe_truthy = False
    maybe_falsey = False
    for truthy in truthy_values:
        if truthy in my_str:
            maybe_truthy = True
    for falsey in falsey_values:
        if falsey in my_str:
            maybe_falsey = True
    if maybe_truthy and not maybe_falsey:
        return True
    if maybe_falsey and not maybe_truthy:
        return False
    raise ValueError(f"Could not convert '{my_str}' to boolean. Expected one of: "
                     "true, false, yes, no, t, f, y, n (case insensitive)")


def get_agent(model_name, api_key, out_dir="chris_temp_deletable"):
    # It is important to provide the api key
    # I will save my api keys as CHRIS_OPENAI_API_KEY in my .env file.
    cfg = AgentConfig(model_name=model_name, out_dir=out_dir, api_key=api_key)
    return HelperAgent(cfg)


def check_skill_completion(agent, skill, current_image, return_bool=True):
    prompt_txt_file = os.path.join(os.path.dirname(
        __file__), "prompts", "skill_completion.txt")
    prompt_parts = construct_prompt(
        prompt_txt_file, {"SKILL": skill, "CURRENT_IMAGE": current_image})
    response = agent.generate_prediction(prompt_parts)
    if return_bool:
        return as_bool(response.text)
    else:
        return response.text


def check_skill_sequencing_two_choices(agent, skill, current_image, next_skill, return_bool=True):
    prompt_txt_file = os.path.join(os.path.dirname(
        __file__), "prompts", "skill_sequencing_two_choices.txt")
    prompt_parts = construct_prompt(
        prompt_txt_file, {"SKILL": skill, "CURRENT_IMAGE": current_image, "NEXT_SKILL": next_skill})
    response = agent.generate_prediction(prompt_parts)
    if return_bool:
        return as_bool(response.text)
    else:
        return response.text
