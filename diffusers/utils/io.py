import json
import os
from typing import Dict, Optional, Tuple, Union

import requests
from PIL import Image

import torch
from diffusers import (ControlNetModel, DiffusionPipeline,
                       DPMSolverMultistepScheduler,
                       StableDiffusionControlNetPipeline,
                       StableDiffusionPipeline)


def load_diffusers_pipeline(
    model_base: str,
    lora_path: Optional[str],
    controlnet_path: Optional[str],
    device: str,
    mode: str = 'base',
    close_safety: bool = False,
    custom_pipeline: str = '/home/pai/lpw_stable_diffusion.py'
) -> Union[DiffusionPipeline, StableDiffusionControlNetPipeline]:
    """
    Loads a DiffusionPipeline or StableDiffusionControlNetPipeline with a LoRA checkpoint,
    based on the specified mode of operation.

    Args:
        model_base (str): The path to the base model checkpoint
        lora_path (str, optional): The path to the LoRA checkpoint
        controlnet_path (str, optional): The path to the controlnet checkpoint (if mode='controlnet')
        device (str): The device where the pipeline will run (e.g. 'cpu' or 'cuda')
        mode (str): The mode of operation ('base', or 'controlnet')
        close_safety (bool): Whether to disable safety checks in the pipeline
        custom_pipeline (str): The path to a custom pipeline script (if any)

    Returns:
        Union[DiffusionPipeline, StableDiffusionControlNetPipeline]:
            A DiffusionPipeline (LPW) or StableDiffusionControlNetPipeline object with a LoRA checkpoint loaded.
    """
    if mode == 'base':
        if close_safety:
            pipe = DiffusionPipeline.from_pretrained(
                model_base,
                custom_pipeline=custom_pipeline,
                torch_dtype=torch.float16,
                safety_checker=None)
        else:
            pipe = DiffusionPipeline.from_pretrained(
                model_base,
                custom_pipeline=custom_pipeline,
                torch_dtype=torch.float16,
            )

    elif mode == 'controlnet':
        controlnet = ControlNetModel.from_pretrained(controlnet_path,
                                                     torch_dtype=torch.float16)

        if close_safety:
            pipe = StableDiffusionControlNetPipeline.from_pretrained(
                model_base,
                controlnet=controlnet,
                revision='fp16',
                torch_dtype=torch.float16,
                safety_checker=None)
        else:
            pipe = StableDiffusionControlNetPipeline.from_pretrained(
                model_base,
                controlnet=controlnet,
                revision='fp16',
                torch_dtype=torch.float16)
    else:
        raise ValueError(
            'Unrecognized function name: {}. We support base(t2i)/controlnet'.
            format(mode))

    pipe.to(device)

    if lora_path is not None:
        pipe.unet.load_attn_procs(lora_path, use_safetensors=False)

    return pipe


def download_image(image_link: str) -> Image.Image:
    """
    Download an image from the given image_link and return it as a PIL Image object.

    Args:
        image_link (str): The URL of the image to download.

    Returns:
        Image.Image: The downloaded image as a PIL Image object.
    """
    response = requests.get(image_link)
    image_name = image_link.split('/')[-1]
    with open(image_name, 'ab') as f:
        f.write(response.content)
        f.flush()
    img = Image.open(image_name).convert('RGB')

    return img


def get_result_str(result_dict: Optional[Dict[str, Union[int, str]]] = None,
                   error: Optional[Exception] = None) -> Tuple[bytes, int]:
    """
    Generates a result string in JSON format based on the provided result dictionary and error.

    Args:
        result_dict (Optional[Dict[str, Union[int, str]]]): A dictionary containing the result information.
        error (Optional[Exception]): An error object representing any occurred error.

    Returns:
        Tuple[bytes, int]: A tuple containing the result string encoded in UTF-8 and the HTTP status code.

    """
    result = {}

    if error is not None:
        result['success'] = 0
        result['error_code'] = error.code
        result['error_msg'] = error.msg[:200]
        stat = error.code

        if result_dict is not None and 'task_id' in result_dict.keys():
            result['task_id'] = result_dict['task_id']

    elif result_dict is not None:
        result['success'] = 1
        result.update(result_dict)
        stat = 200

    result_str = json.dumps(result).encode('utf-8')

    return result_str, stat
