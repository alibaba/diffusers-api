"""
    convert differnt model type to the standard diffuser type
"""

import torch
from diffusers.pipelines.stable_diffusion.convert_from_ckpt import \
    download_from_original_stable_diffusion_ckpt
from safetensors.torch import load_file

LORA_PREFIX_UNET = 'lora_unet'


def convert_name_to_bin(name: str) -> str:
    """
    Convert a name to binary format.

    Args:
        name (str): Name to be converted.

    Returns:
        str: Converted name in binary format.
    """

    # down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q.lora_up
    new_name = name.replace(LORA_PREFIX_UNET + '_', '')
    new_name = new_name.replace('.weight', '')

    # ['down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q', 'lora.up']
    parts = new_name.split('.')

    #parts[0] = parts[0].replace('_0', '')
    if 'out' in parts[0]:
        parts[0] = '_'.join(parts[0].split('_')[:-1])
    parts[1] = parts[1].replace('_', '.')

    # ['down', 'blocks', '0', 'attentions', '0', 'transformer', 'blocks', '0', 'attn1', 'to', 'q']
    # ['mid', 'block', 'attentions', '0', 'transformer', 'blocks', '0', 'attn2', 'to', 'out']
    sub_parts = parts[0].split('_')

    # down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q_
    new_sub_parts = ''
    for i in range(len(sub_parts)):
        if sub_parts[i] in [
                'block', 'blocks', 'attentions'
        ] or sub_parts[i].isnumeric() or 'attn' in sub_parts[i]:
            if 'attn' in sub_parts[i]:
                new_sub_parts += sub_parts[i] + '.processor.'
            else:
                new_sub_parts += sub_parts[i] + '.'
        else:
            new_sub_parts += sub_parts[i] + '_'

    # down_blocks.0.attentions.0.transformer_blocks.0.attn1.processor.to_q_lora.up
    new_sub_parts += parts[1]

    new_name = new_sub_parts + '.weight'

    return new_name


def convert_lora_safetensor_to_bin(safetensor_path: str,
                                   bin_path: str) -> None:
    """
    Convert LoRA safetensor file to binary format and save it. (only the attn parameters will be saved)

    Args:
        safetensor_path (str): Path to the safetensor file.
        bin_path (str): Path to save the binary file.
    """

    bin_state_dict = {}
    safetensors_state_dict = load_file(safetensor_path)

    for key_safetensors in safetensors_state_dict:
        # these if are required  by current diffusers' API
        # remove these may have negative effect as not all LoRAs are used
        if 'text' in key_safetensors:
            continue
        if 'unet' not in key_safetensors:
            continue
        if 'transformer_blocks' not in key_safetensors:
            continue
        if 'ff_net' in key_safetensors or 'alpha' in key_safetensors:
            continue
        key_bin = convert_name_to_bin(key_safetensors)
        bin_state_dict[key_bin] = safetensors_state_dict[key_safetensors]

    torch.save(bin_state_dict, bin_path)


def convert_base_model_to_diffuser(checkpoint_path: str,
                                   target_path: str,
                                   from_safetensors: bool = False,
                                   save_half: bool = False,
                                   controlnet: str = None,
                                   to_safetensors: bool = False) -> None:
    """
    Convert base model to diffuser format and save it.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
        target_path (str): Path to save the diffuser model.
        from_safetensors (bool, optional): Flag indicating whether to load from safetensors.
        save_half (bool, optional): Flag indicating whether to save the model in half precision.
        controlnet (str, optional): Controlnet model path.
        to_safetensors (bool, optional): Flag indicating whether to serialize in safetensors format.
    """

    pipe = download_from_original_stable_diffusion_ckpt(
        checkpoint_path=checkpoint_path,
        from_safetensors=from_safetensors,
        controlnet=controlnet)

    if save_half:
        pipe.to(torch_dtype=torch.float16)

    if controlnet:
        # only save the controlnet model
        pipe.controlnet.save_pretrained(target_path,
                                        safe_serialization=to_safetensors)
    else:
        pipe.save_pretrained(target_path, safe_serialization=to_safetensors)
