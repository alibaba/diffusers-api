"""
    The file is used for blade optimization.
    The main function are three fold:
    1. optimize_and_save_blade_model: do blade optimization and save optimized models (it takes about 30min)
    2. load_blade_model: use to load the optimized blade model
    3. load_attn_procs / unload_lora: online change and merge multiple lora weights
"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch_blade
from safetensors.torch import load_file
from torch import Tensor, nn
from torch_blade import optimize as blade_optimize

# ------------------------ 1. optimize_and_save_blade_model ------------------------


def gen_inputs(
    pipe,
    use_controlnet: bool = False
) -> Tuple[Tensor, Tuple[Tensor, ...], Tuple[Tensor, ...], Tensor]:
    """
    Generate inputs for the specified pipe to forward pipeline.

    Args:
        pipe: The diffusion pipeline.
        use_controlnet (bool, optional): Flag indicating whether to use controlnet inputs. Default is False.

    Returns:
        Tuple[Tensor, Tuple[Tensor, ...], Tuple[Tensor, ...], Tensor]: The generated inputs consisting of:
            - encoder_inputs: Tensor of shape (1, text_max_length) containing integer values.
            - controlnet_inputs: Tuple of tensors containing inputs for controlnet.
            - unet_inputs: Tuple of tensors containing inputs for unet.
            - decoder_inputs: Tensor of shape (1, unet_out_channels, 128, 128) containing float values.
    """
    device = torch.device('cuda:0')
    # use bs=1 to trace and optimize
    text_max_length = pipe.tokenizer.model_max_length
    text_embedding_size = pipe.text_encoder.config.hidden_size
    sample_size = pipe.unet.config.sample_size
    unet_in_channels = pipe.unet.config.in_channels
    unet_out_channels = pipe.unet.config.out_channels

    encoder_inputs = torch.randint(1,
                                   999, (1, text_max_length),
                                   device=device,
                                   dtype=torch.int64)

    unet_inputs = [
        torch.randn(
            (2, unet_in_channels, sample_size, sample_size),
            dtype=torch.half,
            device=device,
        ),
        torch.tensor(999, device=device, dtype=torch.half),
        torch.randn((2, text_max_length, text_embedding_size),
                    dtype=torch.half,
                    device=device),
    ]

    # controlnet has same inputs as unet, with additional condition
    controlnet_inputs = unet_inputs + [
        torch.randn((2, 3, 512, 512), dtype=torch.half, device=device),
    ]

    decoder_inputs = torch.randn(1,
                                 unet_out_channels,
                                 128,
                                 128,
                                 device=device,
                                 dtype=torch.half)

    return encoder_inputs, controlnet_inputs, unet_inputs, decoder_inputs


def optimize_and_save_blade_model(
        pipe: nn.Module,
        encoder_path: str,
        unet_path: str,
        decoder_path: str,
        controlnet_path: Optional[str] = None) -> None:
    """
    Optimize and save the Blade model.

    Args:
        pipe (nn.Module): The pipeline module.
        encoder_path (str): The path to save the optimized encoder model.
        unet_path (str): The path to save the optimized UNet model.
        decoder_path (str): The path to save the optimized decoder model.
        controlnet_path (str, optional): The path to save the optimized controlnet model. Default is None.

    Returns:
        None
    """

    if controlnet_path is not None:
        use_controlnet = True
    else:
        use_controlnet = False

    encoder_inputs, controlnet_inputs, unet_inputs, decoder_inputs = gen_inputs(
        pipe, use_controlnet=use_controlnet)

    if not use_controlnet:
        # base
        class UnetWrapper(torch.nn.Module):
            def __init__(self, unet):
                super().__init__()
                self.unet = unet

            def forward(self, sample, timestep, encoder_hidden_states):
                return self.unet(
                    sample,
                    timestep,
                    encoder_hidden_states=encoder_hidden_states,
                )

        opt_cfg = torch_blade.Config()
        opt_cfg.enable_fp16 = True
        opt_cfg.freeze_module = False  # allow to change the lora weight when inferring

        from torch_blade.monkey_patch import patch_utils
        # change layout for conv layer [NCHW]->[NHWC] for a better inference time
        patch_utils.patch_conv2d(pipe.unet)
        patch_utils.patch_conv2d(pipe.vae.decoder)

        with opt_cfg, torch.no_grad():
            unet = torch.jit.trace(
                UnetWrapper(pipe.unet).eval(),
                tuple(unet_inputs),
                strict=False,
                check_trace=False,
            )
            # unet = torch.jit.trace(pipe.unet, unet_inputs, strict=False, check_trace=False)

            unet = torch_blade.optimize(unet,
                                        model_inputs=tuple(unet_inputs),
                                        allow_tracing=True)

            encoder = torch_blade.optimize(pipe.text_encoder,
                                           model_inputs=encoder_inputs,
                                           allow_tracing=True)

            decoder = torch.jit.trace(pipe.vae.decoder,
                                      decoder_inputs,
                                      strict=False,
                                      check_trace=False)
            decoder = torch_blade.optimize(decoder,
                                           model_inputs=decoder_inputs,
                                           allow_tracing=True)

        torch.jit.save(encoder, encoder_path)
        torch.jit.save(unet, unet_path)
        torch.jit.save(decoder, decoder_path)

    else:
        # controlnet
        opt_cfg = torch_blade.Config()
        opt_cfg.enable_fp16 = True

        class UnetWrapper(torch.nn.Module):
            def __init__(self, unet):
                super().__init__()
                self.unet = unet

            def forward(
                self,
                sample,
                timestep,
                encoder_hidden_states,
                down_block_additional_residuals,
                mid_block_additional_residual,
            ):
                return self.unet(
                    sample,
                    timestep,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=
                    down_block_additional_residuals,
                    mid_block_additional_residual=mid_block_additional_residual,
                )

        import functools

        from torch_blade.monkey_patch import patch_utils

        patch_utils.patch_conv2d(pipe.unet)
        patch_utils.patch_conv2d(pipe.controlnet)

        pipe.controlnet.forward = functools.partial(pipe.controlnet.forward,
                                                    return_dict=False)

        with opt_cfg, torch.no_grad():
            encoder = torch_blade.optimize(pipe.text_encoder,
                                           model_inputs=encoder_inputs,
                                           allow_tracing=True)
            # decoder = torch.jit.trace(pipe.vae.decoder, decoder_inputs, strict=False, check_trace=False)
            decoder = torch_blade.optimize(pipe.vae.decoder,
                                           model_inputs=decoder_inputs,
                                           allow_tracing=True)

            # not freeze to load other weights
            opt_cfg.freeze_module = False

            controlnet = torch.jit.trace(pipe.controlnet,
                                         tuple(controlnet_inputs),
                                         strict=False,
                                         check_trace=False)
            controlnet = torch_blade.optimize(
                controlnet,
                model_inputs=tuple(controlnet_inputs),
                allow_tracing=True)
            # add controlnet outputs to unet inputs
            down_block_res_samples, mid_block_res_sample = controlnet(
                *controlnet_inputs)

            device = torch.device('cuda:0')

            unet_inputs += [
                tuple(down_block_res_samples),
                mid_block_res_sample,
            ]

            unet = torch.jit.trace(
                UnetWrapper(pipe.unet).eval(),
                tuple(unet_inputs),
                strict=False,
                check_trace=False,
            )

            unet = torch_blade.optimize(unet,
                                        model_inputs=tuple(unet_inputs),
                                        allow_tracing=True)

        torch.jit.save(encoder, encoder_path)
        torch.jit.save(controlnet, controlnet_path)
        torch.jit.save(unet, unet_path)
        torch.jit.save(decoder, decoder_path)


# ------------------------ 2. load_blade_model ------------------------


def load_blade_model(pipe: nn.Module,
                     encoder_path: str,
                     unet_path: str,
                     decoder_path: str,
                     controlnet_path: Optional[str] = None) -> nn.Module:
    """
    Load the Blade model.

    Args:
        pipe (nn.Module): The pipeline module.
        encoder_path (str): The path to the optimized encoder model.
        unet_path (str): The path to the optimized UNet model.
        decoder_path (str): The path to the optimized decoder model.
        controlnet_path (str, optional): The path to the optimized controlnet model. Default is None.

    Returns:
        nn.Module: The loaded Blade model.
    """

    if controlnet_path is not None:
        use_controlnet = True
    else:
        use_controlnet = False

    encoder_inputs, controlnet_inputs, unet_inputs, decoder_inputs = gen_inputs(
        pipe, use_controlnet=use_controlnet)

    # encoder = torch.jit.load(encoder_path).eval().cuda()
    unet = torch.jit.load(unet_path).eval().cuda()
    decoder = torch.jit.load(decoder_path).eval().cuda()

    # load weights from current model
    if not use_controlnet:
        unet_state_dict = {
            'unet.' + k: v
            for k, v in pipe.unet.state_dict().items()
        }
        _, unexpected = unet.load_state_dict(unet_state_dict, strict=False)
        print(unexpected)

        _, unexpected = decoder.load_state_dict(pipe.vae.decoder.state_dict(),
                                                strict=False)
        print(unexpected)

        # warmup
        # encoder(encoder_inputs)
        unet(*unet_inputs)
        decoder(*decoder_inputs)

    patch_conv_weights(unet)
    patch_conv_weights(decoder)

    if use_controlnet:
        controlnet = torch.jit.load(controlnet_path).eval().cuda()

    @dataclass
    class UNet2DConditionOutput:
        sample: torch.FloatTensor

    class TracedEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = pipe.text_encoder.config
            self.device = pipe.text_encoder.device
            self.dtype = torch.half

        def forward(self, input_ids, **kwargs):
            embeddings = encoder(input_ids.long())
            return [embeddings['last_hidden_state']]

    if use_controlnet:
        # controlnet
        class TracedControlNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.controlnet_conditioning_channel_order = 'rgb'

            def forward(self, sample, timestep, encoder_hidden_states,
                        **kwargs):
                if self.controlnet_conditioning_channel_order == 'rgb':
                    return controlnet(sample.half(), timestep.half(),
                                      encoder_hidden_states.half(),
                                      kwargs['controlnet_cond'])
                else:
                    return controlnet(
                        sample.half(), timestep.half(),
                        encoder_hidden_states.half(),
                        torch.flip(kwargs['controlnet_cond'], dims=[1]))

            def load_state_dict(self, state_dict, strict=False):
                _, unexpected = controlnet.load_state_dict(state_dict,
                                                           strict=strict)
                if unexpected:
                    print(
                        f'load controlNet with unexpected keys: {unexpected}')
                return

            def state_dict(self):
                return controlnet.state_dict()

            def set_channel_order(self, channel_order):
                self.controlnet_conditioning_channel_order = channel_order

        class TracedUNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = pipe.unet.config
                self.in_channels = pipe.unet.in_channels
                self.device = pipe.unet.device
                self.device = pipe.unet.device
                self.lora_weights = {}
                self.cur_lora = {}

            def state_dict(self):
                return unet.state_dict()

            def forward(self, latent_model_input, t, encoder_hidden_states,
                        **kwargs):
                if kwargs.get('down_block_additional_residuals', None) is None:
                    kwargs['down_block_additional_residuals'] = tuple([
                        torch.tensor(
                            [[[[0.0]]]], device=self.device, dtype=torch.half)
                    ] * 13)
                if kwargs.get('mid_block_additional_residual', None) is None:
                    kwargs['mid_block_additional_residual'] = torch.tensor(
                        [[[[0.0]]]], device=self.device, dtype=torch.half)

                sample = unet(
                    latent_model_input.half(),
                    t.half(),
                    encoder_hidden_states.half(),
                    kwargs['down_block_additional_residuals'],
                    kwargs['mid_block_additional_residual'],
                )['sample']

                return UNet2DConditionOutput(sample=sample)

    else:
        # base model
        class TracedUNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = pipe.unet.config
                self.in_channels = pipe.unet.in_channels
                self.device = pipe.unet.device
                self.lora_weights = {}
                self.cur_lora = {}

            def state_dict(self):
                return unet.state_dict()

            def forward(self, latent_model_input, t, encoder_hidden_states,
                        **kwargs):
                sample = unet(latent_model_input.half(), t.half(),
                              encoder_hidden_states.half())['sample']
                return UNet2DConditionOutput(sample=sample)

    class TracedDecoder(torch.nn.Module):
        def forward(self, input):
            return decoder(input.half())

    # pipe.text_encoder = TracedEncoder() # lead to incorrect output

    if use_controlnet:
        controlnet_wrapper = TracedControlNet()
        pipe.controlnet.forward = controlnet_wrapper.forward
        pipe.controlnet.load_state_dict = controlnet_wrapper.load_state_dict
        pipe.controlnet.state_dict = controlnet_wrapper.state_dict
        pipe.controlnet.set_channel_order = controlnet_wrapper.set_channel_order

    pipe.unet = TracedUNet()
    pipe.vae.decoder = TracedDecoder()

    return pipe


# ------------------------ 3. load_attn_procs / unload_lora: online change and merge multiple lora weights ------------------

LORA_PREFIX_TEXT_ENCODER, LORA_PREFIX_UNET = 'lora_te', 'lora_unet'


@lru_cache(maxsize=32)
def load_lora_and_mul(
        lora_path: str,
        dtype: torch.dtype) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
    """
    Load and process LoRA weights from the specified path.

    Args:
        lora_path (str): Path to the LoRA weights file.
        dtype (torch.dtype): Desired data type for the processed weights.

    Returns:
        Tuple[Dict[str, Tensor], Dict[str, Tensor]]: A tuple containing two dictionaries:
            - text_encoder_state_dict: Dictionary containing the state dictionary for the text encoder.
            - unet_state_dict: Dictionary containing the state dictionary for the UNet.
    """
    if lora_path.endswith('.safetensors'):
        # lora model trained by webui script (e.g., model from civitai)

        state_dict = load_file(lora_path)

        visited, unet_state_dict, text_encoder_state_dict = [], {}, {}

        # directly update weight in diffusers model
        for key in state_dict:
            # it is suggested to print out the key, it usually will be something like below
            # "lora_te_text_model_encoder_layers_0_self_attn_k_proj.lora_down.weight"

            # as we have set the alpha beforehand, so just skip
            if '.alpha' in key or key in visited:
                continue

            if 'text' in key:
                diffusers_key = key.split(
                    '.')[0].split(LORA_PREFIX_TEXT_ENCODER + '_')[-1].replace(
                        '_', '.').replace('text.model', 'text_model').replace(
                            '.proj', '_proj').replace('self.attn',
                                                      'self_attn') + '.weight'
                curr_state_dict = text_encoder_state_dict
            else:
                diffusers_key = 'unet.' + key.split('.')[0].split(
                    LORA_PREFIX_UNET + '_')[-1].replace('_', '.').replace(
                        '.block', '_block').replace('to.', 'to_').replace(
                            'proj.', 'proj_') + '.weight'
                curr_state_dict = unet_state_dict

            pair_keys = []
            if 'lora_down' in key:
                alpha = state_dict.get(
                    key.replace('lora_down.weight', 'alpha'), None)
                pair_keys.append(key.replace('lora_down', 'lora_up'))
                pair_keys.append(key)
            else:
                alpha = state_dict.get(key.replace('lora_up.weight', 'alpha'),
                                       None)
                pair_keys.append(key)
                pair_keys.append(key.replace('lora_up', 'lora_down'))

            # update weight
            if alpha:
                alpha = alpha.item() / state_dict[pair_keys[0]].shape[1]
            else:
                alpha = 0.75

            if len(state_dict[pair_keys[0]].shape) == 4:
                weight_up = state_dict[pair_keys[0]].squeeze(3).squeeze(2).to(
                    torch.float32)
                weight_down = state_dict[pair_keys[1]].squeeze(3).squeeze(
                    2).to(torch.float32)
                if len(weight_up.shape) == len(weight_down.shape):
                    curr_state_dict[diffusers_key] = alpha * torch.mm(
                        weight_up.cuda(),
                        weight_down.cuda()).unsqueeze(2).unsqueeze(3).to(dtype)
                else:
                    curr_state_dict[diffusers_key] = alpha * torch.einsum(
                        'a b, b c h w -> a c h w', weight_up.cuda(),
                        weight_down.cuda()).to(dtype)
            else:
                weight_up = state_dict[pair_keys[0]].to(torch.float32)
                weight_down = state_dict[pair_keys[1]].to(torch.float32)
                curr_state_dict[diffusers_key] = alpha * torch.mm(
                    weight_up.cuda(), weight_down.cuda()).to(dtype)

            # update visited list
            for item in pair_keys:
                visited.append(item)
        return text_encoder_state_dict, unet_state_dict
    else:
        # model trained by diffusers api (lora attn only in unet)
        state_dict = torch.load(lora_path)
        multied_state_dict = {}
        for k, v in state_dict.items():
            if '_lora.up.weight' in k:
                up_weight = v
                new_k = 'unet.' + k.replace(
                    '_lora.up.weight', '.weight').replace(
                        'processor.', '').replace('to_out', 'to_out.0')
                down_weight = state_dict[k.replace('up.weight', 'down.weight')]
                # xxxx_lora.up.weight
                multied_state_dict[new_k] = torch.matmul(
                    up_weight.cuda(), down_weight.cuda())
        return {}, multied_state_dict


def patch_conv_weights(model: nn.Module) -> nn.Module:
    """
    Patch the convolutional weights in the model to be compatible with NHWC format.
    For model acceleration in blade optimization

    Args:
        model (nn.Module): The model to be patched.

    Returns:
        nn.Module: The patched model.
    """
    origin_state_dict = model.state_dict()
    state_dict = {}
    for k, v in origin_state_dict.items():
        if k.endswith('_nhwc'):
            state_dict[k] = origin_state_dict[k[:-5]].permute([0, 2, 3, 1])
    model.load_state_dict(state_dict, strict=False)
    return model


def merge_lora_weights(origin: Dict[str, Tensor], to_merge: Dict[str, Tensor],
                       scale: float) -> Dict[str, Tensor]:
    """
    Merge LoRA weights into the origin dictionary with the specified scale.

    Args:
        origin (Dict[str, Tensor]): The original weights dictionary.
        to_merge (Dict[str, Tensor]): The weights dictionary to be merged.
        scale (float): The scaling factor for the merged weights.

    Returns:
        Dict[str, Tensor]: The merged weights dictionary.
    """
    for k, v in to_merge.items():
        v = v.to('cuda')
        weight = v * scale
        if origin.get(k, None) is None:
            origin[k] = weight
        else:
            origin[k] += weight


def apply_lora_weights(model_state_dict: Dict[str, Tensor],
                       weights: Dict[str, Tensor]) -> None:
    """
    Apply LoRA weights to the model state dictionary.

    Args:
        model_state_dict (Dict[str, Tensor]): The model's state dictionary.
        weights (Dict[str, Tensor]): The LoRA weights to be applied.

    Returns:
        None
    """

    with torch.no_grad():
        for k, v in weights.items():
            v = v.to('cuda')
            model_state_dict[k].add_(v)


def unload_lora_weights(model_state_dict: Dict[str, Tensor],
                        weights: Dict[str, Tensor]) -> None:
    """
    Unload LoRA weights from the model state dictionary.

    Args:
        model_state_dict (Dict[str, Tensor]): The model's state dictionary.
        weights (Dict[str, Tensor]): The LoRA weights to be unloaded.

    Returns:
        None
    """
    with torch.no_grad():
        for k, v in weights.items():
            v = v.to('cuda')
            model_state_dict[k].sub_(v)


def load_attn_procs(pipe,
                    attn_procs_paths: Union[str, List[str]],
                    scales: Union[float, List[float]] = 0.75) -> None:
    """
    Load and merge multiple lora model weights into the pipeline.

    Args:
        pipe: Stable diffusion pipeline
        attn_procs_paths (Union[str, List[str]]): The paths to the attention processor weights.
        scales (Union[float, List[float]], optional): The scaling factor(s) for the merged weights. Default is 0.75.

    Returns:
        None
    """

    if isinstance(scales, str):
        attn_procs_paths = [attn_procs_paths]
    if isinstance(scales, float):
        scales = [scales] * len(attn_procs_paths)

    pipe.text_encoder_merged_weights, pipe.unet_merged_weights = {}, {}

    for attn_procs_path, scale in zip(attn_procs_paths, scales):
        text_encoder_state_dict, unet_state_dict = load_lora_and_mul(
            attn_procs_path, dtype=torch.half)
        # merge weights from multiple lora models with scale
        merge_lora_weights(pipe.text_encoder_merged_weights,
                           text_encoder_state_dict, scale)
        merge_lora_weights(pipe.unet_merged_weights, unet_state_dict, scale)

    # apply the final lora weights to text_encoder and unet
    apply_lora_weights(pipe.text_encoder.state_dict(),
                       pipe.text_encoder_merged_weights)
    apply_lora_weights(pipe.unet.state_dict(), pipe.unet_merged_weights)

    patch_conv_weights(pipe.unet)


def unload_lora(pipe):
    # unload the lora weight after each infer
    unload_lora_weights(pipe.text_encoder.state_dict(),
                        pipe.text_encoder_merged_weights)
    unload_lora_weights(pipe.unet.state_dict(), pipe.unet_merged_weights)
    pipe.text_encoder_merged_weights, pipe.unet_merged_weights = {}, {}
    patch_conv_weights(pipe.unet)
