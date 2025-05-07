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
import cv2
import numpy as np
from google.generativeai import types
from time import sleep
from google import genai


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
            assert content_dict[key] is not None, f"Key {key} was None in the content_dict"
            prompt_parts.append(content_dict[key])
        else:
            raise ValueError(f"Key {key} not found in content_dict")

        current_pos = end_sep + len(special_separator)
    return prompt_parts


class HelperAgent(Agent):
    # This is what gets created.
    # Notably, in Long's VLMX there is no Gemini wrapper.
    OUT_RESULT_PATH = "test.txt"

    def _make_system_instruction(self):
        return SYSTEM_INSTRUCTION

    def _make_prompt_parts(self, question):
        # Incredibly cursed, question is already concatenated.

        # The commented out code here does not above.
        # # Here I am going to add in some hackiness.
        # # This only works for Gemini.
        # # It lets you pass in an mp4 file up to 20MB.
        # includes_mp4 = False
        # for p in question:
        #     print(f"we are checking: {p}")

        #     if str(p).endswith('.mp4'):
        #         print("we think it does in in mp4")
        #         includes_mp4 = True

        # if includes_mp4:
        #     print("Includes mp4 is true")
        #     question = hacky_prompt_processing(question)
        # print("_make_prompt_parts is returning")
        # print(question)
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
    # This is the function that main_chris uses to make the agent.
    # It is important to provide the api key
    # I will save my api keys as CHRIS_OPENAI_API_KEY in my .env file.
    cfg = AgentConfig(model_name=model_name, out_dir=out_dir, api_key=api_key)
    return HelperAgent(cfg)


def check_skill_completion(agent, skill, current_image, return_bool=True, scene_description=None, prompt="skill_completion.txt"):
    prompt_txt_file = os.path.join(os.path.dirname(
        __file__), "prompts", prompt)
    prompt_dict = {"SKILL": skill, "CURRENT_IMAGE": current_image}
    if scene_description is not None:
        prompt_dict["SCENE_DESCRIPTION"] = scene_description
    prompt_parts = construct_prompt(
        prompt_txt_file, prompt_dict)
    response = agent.generate_prediction(prompt_parts)
    if return_bool:
        return as_bool(response.text)
    else:
        return response.text


def check_skill_completion_general(
        agent,
        skill,
        current_image,
        current_wrist_image=None,
        previous_image=None,
        overall_task=None,
        entire_plan=None,
        prompt="skill_completion_contrastive_v2.txt",
        return_bool=True
):
    prompt_txt_file = os.path.join(os.path.dirname(
        __file__), "prompts", prompt)
    prompt_dict = {"SKILL": skill, "CURRENT_IMAGE": current_image}
    # This not putting things in the dict is mostly unnecessary, since None counts as an error in construct_prompt
    if current_wrist_image is not None:
        prompt_dict["CURRENT_WRIST_IMAGE"] = current_wrist_image
    if previous_image is not None:
        prompt_dict["PRE_IMAGE"] = previous_image
    if overall_task is not None:
        prompt_dict["OVERALL_TASK"] = overall_task
    if entire_plan is not None:
        entire_plan = str(entire_plan)
        prompt_dict["ENTIRE_PLAN"] = entire_plan
    prompt_parts = construct_prompt(prompt_txt_file, prompt_dict)
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


def hacky_prompt_part_processing(prompt_part):
    assert isinstance(
        prompt_part, str), "Hacky video processing only works when everything is a string. (use a path to your mp4)"
    if prompt_part.endswith(".mp4"):
        video_bytes = open(prompt_part, 'rb').read()
        return types.Part(inline_data=types.Blob(data=video_bytes, mime_type='video/mp4'))
    else:
        return types.Part(text=prompt_part)


def hacky_prompt_processing(prompt_parts):
    ''' This gives you the contents field you can put into a generate_content gemini request.'''
    parts = [hacky_prompt_part_processing(p) for p in prompt_parts]
    return types.Content(parts)


def video_feedback_gemini(gemini_client, gemini_model, video, overall_task, path, prompt="video_feedback_v2.txt", max_retries=10, retry_sleep=0.5):
    ''' This works if video is a str path to the video. 
        Will only work with gemini models. 
        # VLMX is broken for videos, so this uses a client and model name.
        '''
    # Check if video is a path (either as a Path object or a string)
    if not isinstance(video, (str, Path)):
        raise ValueError(
            "Video must be a string path or Path object pointing to an MP4 file")

    # Convert to string if it's a Path object
    video_path = str(video)

    # Check if it's an absolute path
    if not os.path.isabs(video_path):
        raise ValueError(
            f"Video path must be an absolute path, got: {video_path}")

    # Check if it has .mp4 extension
    if not video_path.lower().endswith('.mp4'):
        raise ValueError(f"Video file must be an MP4 file, got: {video_path}")

    # Check if the file exists
    if not os.path.exists(video_path):
        raise ValueError(f"Video file does not exist: {video_path}")

    # Now it is time to upload the file:
    video = gemini_client.files.upload(file=str(video))
    # I wish this would block, but maybe I need to sleep.

    prompt_txt_file = os.path.join(os.path.dirname(
        __file__), "prompts", "video_feedback", prompt)
    prompt_parts = construct_prompt(
        prompt_txt_file, {"VIDEO": video, "OVERALL_TASK": overall_task, "PATH": path})

    for i in range(max_retries):
        sleep((2**i) * retry_sleep)
        try:
            response = gemini_client.models.generate_content(
                model=gemini_model, contents=prompt_parts
            )
            break
        except genai.errors.ClientError:
            continue

    return response.text

    # I modified HelperAgent to be able to cope with this when everything is a String.
    response = agent.generate_prediction(prompt_parts)
    return response.text


def scene_description(agent, overall_task, image):
    prompt_txt_file = os.path.join(os.path.dirname(
        __file__), "prompts", "scene_description", "scene_description_v1.txt")
    prompt_parts = construct_prompt(
        prompt_txt_file, {"OVERALL_TASK": overall_task, "IMAGE": image})
    response = agent.generate_prediction(prompt_parts)
    return response.text
