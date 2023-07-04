import argparse
import base64
import datetime
import io
import json
import logging
import multiprocessing
# -*- coding: utf-8 -*-
import os
import random
import shutil
import traceback
from glob import glob
from typing import Any, Dict, List, Optional, Union

from PIL import Image

import allspark
import torch
from diffusers import (ControlNetModel, DiffusionPipeline,
                       DPMSolverMultistepScheduler,
                       StableDiffusionControlNetPipeline,
                       StableDiffusionPipeline)
from ev_error import InputFormatError, JsonParseError, UnExpectedServerError
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from safetensors.torch import load_file
from utils.blade import (load_attn_procs, load_blade_model,
                         optimize_and_save_blade_model, patch_conv_weights,
                         unload_lora)
from utils.convert import (convert_base_model_to_diffuser,
                           convert_lora_safetensor_to_bin)
from utils.image_process import (generate_mask_and_img_expand,
                                 preprocess_control, transform_image)
from utils.io import download_image, get_result_str, load_diffusers_pipeline

logging.basicConfig(
    format='[%(levelname)s] %(asctime)s %(filename)s:%(lineno)d : %(message)s',
    level=logging.INFO)

logging.getLogger().setLevel(logging.INFO)

os.environ['DISC_ENABLE_DOT_MERGE'] = '0'


class MyProcessor(allspark.BaseProcessor):
    def initialize(self):
        """
            Initialize function for the diffuser class.
            This function loads the model and
            sets up required attributes for diffuser.
        """

        # ----------- 1. Set default model path/ file path for eas config -----------
        # default for mount path (can not be changed or should be the same with eas config)
        model_dir = '/oss'
        save_dir = '/result'

        # default for file/model path (can not be changed or should be the same with your image)
        custom_pipeline = '/home/pai/lpw_stable_diffusion.py'
        default_blade_dir = '/home/pai/optimized_model'
        self.pretrain_dir = '/home/pai/pretrained_models'

        defaults = allspark.default_properties()
        logging.info(defaults)

        if FLAGS.local_debug:
            # mount path when local debug
            model_dir = FLAGS.model_dir
            save_dir = FLAGS.save_dir

            # only need change if you want to change the default model or pipeline
            custom_pipeline = '/mnt/xinyi.zxy/dl_eas_processor/diffusers/lpw_stable_diffusion.py'
            default_blade_dir = '/mnt/xinyi.zxy/diffuser/min_dependency/v211/optimized_model'
            self.pretrain_dir = '/mnt/xinyi.zxy/diffuser/min_dependency/v211/pretrained_models'

            self.local_debug = True

            defaults.put(
                b'model.model_config',
                ("{\"type\":\"test\", \"predictor_cls_name\":\"test\" }"
                 ).encode('utf8'))

            defaults.put(b'rpc.worker_threads', b'2')
            defaults.put(b'rpc.keepalive', b'500000')
        else:
            self.local_debug = False

        # set the default save root / model dir
        self.oss_save_dir = FLAGS.oss_save_dir
        self.oss_region = FLAGS.region
        self.save_dir = save_dir
        self.model_dir = model_dir

        os.makedirs(self.save_dir, exist_ok=True)

        # ----------- 2. Set service name and model type -----------
        # TODO: merge the use of base/controlnet
        # service_name: ['base','controlnet'] to use different func
        # you can use func_name: t2i/i2i/inpaint/outpaint when deploying a service_name as base
        self.service_name = FLAGS.func_name

        self.close_safety = FLAGS.close_safety

        if self.service_name == 'controlnet':
            self.use_controlnet = True
        else:
            self.use_controlnet = False

        # whether to build with blade optimization
        self.use_blade = FLAGS.use_blade
        # if the default blade model can not be load successfully, please try to use --not_use_default_blade to optimize your own blade model
        # the optimization process may be taken 30 min
        self.not_use_default_blade = FLAGS.not_use_default_blade

        # whether to use the translate model by modelscope
        self.use_translate = FLAGS.use_translate

        # to mark whether the lora model is added and need to be subed after each post request (when use blade)
        self.mount_lora = False

        # remove optimized blade model only when blade model needs to be re-optimized
        need_remove = FLAGS.remove_opt
        if need_remove:
            blade_dir = os.path.join(model_dir, 'optimized_model')
            if os.path.exists(blade_dir):
                logging.info(
                    'Removing optimized blade model when first use the new image!'
                )
                shutil.rmtree(blade_dir)

        # ----------- 3. Check/Convert models -----------
        # 3.1 Base model
        self.ckpt = glob('%s/base_model' % model_dir, recursive=True)

        if len(self.ckpt) == 0:
            raise ValueError(
                'base_model dir must be provided to load the base model for diffusers!'
            )
        else:
            self.ckpt = self.ckpt[0]

        files = os.listdir(self.ckpt)
        need_file_list = [
            'feature_extractor', 'model_index.json', 'safety_checker',
            'scheduler', 'text_encoder', 'tokenizer', 'unet', 'vae'
        ]

        if set(need_file_list).issubset(set(files)):
            logging.info('The provided base model is ready to load!')
        else:
            # need convert
            ckpt_files = [f for f in files if f.endswith('.ckpt')]
            safetensor_files = [f for f in files if f.endswith('.safetensors')]

            # raise ValueError(
            #     'Please refer to https://github.com/huggingface/diffusers/blob/main/scripts/convert_original_stable_diffusion_to_diffusers.py to convert your model in multi dir'
            # )

            # automatic convert
            if 1:
                # only one .ckpt/.safetensors should be given
                if (len(ckpt_files) + len(safetensor_files)) == 1:
                    if len(ckpt_files) == 1:
                        try:
                            logging.info('Start converting {} to {}'.format(
                                os.path.join(self.ckpt, ckpt_files[0]),
                                self.ckpt))
                            convert_base_model_to_diffuser(
                                os.path.join(self.ckpt, ckpt_files[0]),
                                self.ckpt)
                        except Exception as e:
                            raise e
                    else:
                        # convert safetensor
                        try:
                            logging.info('Start converting {} to {}'.format(
                                os.path.join(self.ckpt, safetensor_files[0]),
                                self.ckpt))
                            convert_base_model_to_diffuser(
                                os.path.join(self.ckpt, safetensor_files[0]),
                                self.ckpt,
                                from_safetensors=True)
                        except Exception as e:
                            raise e

                    logging.info('Convert base model successfully!')

                else:
                    raise ValueError(
                        'You need to provide only one .ckpt model or one .safetensors model to convert. But got .ckpt: {} and .safetensors: {}'
                        .format(len(ckpt_files), len(safetensor_files)))

        # 3.2 Lora model (lora model will be loaded only when not use blade or it will be added/subbed during post request)
        lora_model_dir = os.path.join(model_dir, 'lora_model')

        if not self.use_blade:
            # We only support use one lora model when not use blade.
            if os.path.exists(lora_model_dir):
                lora_files = os.listdir(lora_model_dir)
                lora_bin_files = [f for f in lora_files if f.endswith('.bin')]
                lora_safetensor_files = [
                    f for f in lora_files if f.endswith('.safetensors')
                ]

                if len(lora_bin_files) == 1:
                    logging.info('The provided lora model is ready to load!')
                    self.lora = os.path.join(lora_model_dir, lora_bin_files[0])

                elif len(lora_bin_files) > 1:
                    raise ValueError(
                        'only one lora model is allowed when init!')
                else:
                    if len(lora_safetensor_files) == 1:
                        # convert
                        try:
                            # if not use blade only the attn in unet will be load to adapt diffusers api
                            lora_safetensor_file = lora_safetensor_files[0]
                            bin_file = lora_safetensor_file.replace(
                                '.safetensors', '.bin')
                            self.lora = os.path.join(lora_model_dir, bin_file)

                            logging.info(
                                'Start converting lora model {} to {}'.format(
                                    os.path.join(lora_model_dir,
                                                 lora_safetensor_file),
                                    self.lora))
                            convert_lora_safetensor_to_bin(
                                os.path.join(lora_model_dir,
                                             lora_safetensor_file), self.lora)

                        except Exception as e:
                            traceback.print_exc()
                            raise e

                    elif len(lora_bin_files) > 1:
                        raise ValueError(
                            'only one lora model is allowed when init!')
                    else:
                        self.lora = None
            else:
                self.lora = None
        else:
            # blade model donot use the lora model when init
            self.lora = None

        # 3.3 controlnet The model should be put in controlnet dir
        if self.service_name == 'controlnet':
            self.controlnet = glob(
                '%s/controlnet' % model_dir, recursive=True) + glob(
                    '%s/**/controlnet' % model_dir, recursive=True)

            if len(self.controlnet) == 0:
                raise ValueError(
                    'The controlnet dir must be provided when using the function controlnet'
                )
            else:
                self.controlnet = self.controlnet[0]
        else:
            self.controlnet = None

        # ----------- 4. Load models -----------
        # 4.1 Diffuser pipeline
        self.device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')

        # load diffusers model
        print(self.controlnet)

        self.pipe = load_diffusers_pipeline(self.ckpt,
                                            self.lora,
                                            self.controlnet,
                                            self.device,
                                            mode=self.service_name,
                                            close_safety=self.close_safety,
                                            custom_pipeline=custom_pipeline)

        # default hyperparameters
        self.prompt = 'i am a prompt'
        self.inference_steps = 50  # number of sampling steps
        self.dpm_solver = True  # use DPM-Solver sampling
        self.n_samples = 1  # how many samples to produce for each given prompt. A.k.a. batch size
        self.guidance_scale = 7  # unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))
        self.lora_attn = 0.75
        self.negative_prompt = ''
        self.width = 512
        self.height = 512
        self.seed = 0

        if self.dpm_solver:
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipe.scheduler.config)

        # 4.2 Tanslate pipeline from modelscope
        # TODO: put translate in a sub service
        if self.use_translate:
            self.model_trans = glob(
                '%s/translate' % model_dir, recursive=True) + glob(
                    '%s/**/translate' % model_dir, recursive=True)

            if len(self.model_trans) == 0:
                raise ValueError(
                    'Please put the translation model in the translate dir.')

            self.model_trans = self.model_trans[0]

            self.pipe_trans = pipeline(task=Tasks.translation,
                                       model=self.model_trans)

        # ----------- 5. Blade optimization -----------
        # If the optimized_model dir exists, the blade optimized model can be used directly with --use_blade
        # or we will copy the default optimized model (in image) and only change weight for service_name base
        # TODO: the default model for service_name controlnet should also be provided (key error by blade now)
        # If the default model can not be load successfully, you can set --not_use_default_blade --remove_opt to optimize your own blade model,
        # the optimization process will take 30 min in subprocess, and the model will automaticaly loaded util the blade model exists during the infer time

        # mark whether the blade model is successully load
        self.blade_load = False

        if self.use_blade:
            if os.path.exists(lora_model_dir):
                # warning for not load lora model when service initialization
                lora_files = os.listdir(lora_model_dir)
                if len(lora_files) > 0:
                    logging.warning(
                        'We do not use the lora model during service intialization. You are allowed to add the lora model by post with key \'lora_path\' after optimization.'
                    )

            # save optimized model
            blade_dir = os.path.join(model_dir, 'optimized_model')
            os.makedirs(blade_dir, exist_ok=True)

            self.encoder_path = os.path.join(blade_dir, 'encoder.pt')
            self.unet_path = os.path.join(blade_dir, 'unet.pt')
            self.decoder_path = os.path.join(blade_dir, 'decoder.pt')

            if self.use_controlnet:
                # blade with controlnet still can not use the default optimized model for different model keys [fixing by yh]
                self.controlnet_path = os.path.join(blade_dir, 'controlnet.pt')

                if all(
                        os.path.exists(p) for p in [
                            self.encoder_path, self.unet_path,
                            self.decoder_path, self.controlnet_path
                        ]):
                    # load optimized model
                    try:
                        self.pipe = load_blade_model(self.pipe,
                                                     self.encoder_path,
                                                     self.unet_path,
                                                     self.decoder_path,
                                                     self.controlnet_path)
                        self.blade_load = True
                    except Exception as e:
                        logging.error(str(e))
                        traceback.print_exc()
                        logging.warning(
                            'Load blade model with controlnet failed! Use the original diffuser model!'
                        )
                else:
                    # begin to optimize
                    blade_optimize_process = multiprocessing.Process(
                        target=optimize_and_save_blade_model,
                        kwargs={
                            'pipe': self.pipe,
                            'encoder_path': self.encoder_path,
                            'unet_path': self.unet_path,
                            'decoder_path': self.decoder_path,
                            'controlnet_path': self.controlnet_path
                        })
                    blade_optimize_process.start()

                    logging.warning(
                        'The blade model is optimizing now! We will use eager diffusers API and change the mode once blade optimization is done.\n'
                        'See use_blade in res to mark whether the model is in blade optimization.\n'
                        'Note that during the blade optimization, it is likely to cause the CUDA OOM problem for a large image size!'
                    )

            else:
                # sevice_name base
                if all(
                        os.path.exists(p) for p in
                    [self.encoder_path, self.unet_path, self.decoder_path]):
                    try:
                        self.pipe = load_blade_model(self.pipe,
                                                     self.encoder_path,
                                                     self.unet_path,
                                                     self.decoder_path)
                        self.blade_load = True
                    except:
                        traceback.print_exc()
                        logging.warning(
                            'Load blade model failed! Use the original diffuser model!'
                        )

                else:
                    # copy default optimized model and reload model weight
                    if os.path.exists(default_blade_dir
                                      ) and not self.not_use_default_blade:
                        logging.info('Copying default blade model!')
                        if os.path.exists(blade_dir):
                            shutil.rmtree(blade_dir)
                        shutil.copytree(default_blade_dir, blade_dir)

                        try:
                            self.pipe = load_blade_model(
                                self.pipe, self.encoder_path, self.unet_path,
                                self.decoder_path)
                            self.blade_load = True
                        except:
                            traceback.print_exc()
                            logging.warning(
                                'Load pre-optimized blade model failed! Use the original diffuser model! Or you can try to optimize blade model with --not_use_default_blade --remove_opt when service initialization'
                            )

                    else:
                        # begin to optimize
                        # optimize_and_save_blade_model(self.pipe,self.encoder_path,self.unet_path,self.decoder_path)
                        blade_optimize_process = multiprocessing.Process(
                            target=optimize_and_save_blade_model,
                            kwargs={
                                'pipe': self.pipe,
                                'encoder_path': self.encoder_path,
                                'unet_path': self.unet_path,
                                'decoder_path': self.decoder_path
                            })
                        blade_optimize_process.start()

                        logging.warning(
                            'The blade model is optimizing now! We will use eager diffusers API and change the mode once blade optimization is done.\n'
                            'See use_blade in res to mark whether the model is in blade optimization.\n'
                            'Note that during the blade optimization, it is likely to cause the CUDA OOM problem for a large image size!'
                        )

        return

    def change_lora(self, lora_paths: List[str],
                    lora_attns: List[float]) -> Dict[str, Any]:
        """
        Change the Lora model used by the lora_paths and lora_attns.

        Args:
            lora_paths (List[str]): List of Lora paths.
            lora_attns (List[float]): List of Lora attentions.

        Returns:
            Dict[str, Any]: Return data containing the success flag and error message (if any).
        """

        ret_data = {}

        if len(lora_paths) != len(lora_attns):
            error = InputFormatError(
                'The input length of lora_paths and lora_attns must be the same'
            )
            ret_data['error'] = error
            ret_data['success'] = 0
        else:
            # check and convert each lora path
            for i, sub_lora_path in enumerate(lora_paths):
                lora_path = os.path.join(self.model_dir, sub_lora_path)

                if not os.path.exists(lora_path):
                    error = InputFormatError(
                        'lora path: {} not exists! Make sure you have mount the correct oss path'
                        .format(sub_lora_path))
                    ret_data['error'] = error
                    ret_data['success'] = 0
                    return ret_data
                # not convert for blade since it can process safetensors directly
                elif not self.use_blade:
                    if lora_path.endswith('.safetensors'):
                        try:
                            bin_path = lora_path.replace(
                                '.safetensors', '.bin')
                            logging.info(
                                'Converting lora model from {} to {}'.format(
                                    lora_path, bin_path))
                            convert_lora_safetensor_to_bin(lora_path, bin_path)
                            lora_path = bin_path
                        except Exception as e:
                            traceback.print_exc()
                            error = UnExpectedServerError(
                                'Convert lora model error: ' + str(e))
                            ret_data['error'] = error
                            ret_data['success'] = 0
                            return ret_data

                lora_paths[i] = lora_path

            logging.info('lora_paths: {}'.format(lora_paths))
            logging.info('lora_attns: {}'.format(lora_attns))

            # load lora model
            try:
                if self.use_blade:
                    if self.blade_load:
                        # load multi lora in blade
                        load_attn_procs(self.pipe, lora_paths, lora_attns)
                        self.mount_lora = True
                    else:
                        error = InputFormatError(
                            'Please wait util the blade optimization done to add lora model'
                        )
                        ret_data['error'] = error
                        ret_data['success'] = 0
                else:
                    # only one lora model is allowed for ori diffuser model
                    if len(lora_paths) != 1:
                        error = InputFormatError(
                            'Ori diffusers api only allow one lora model. You can use blade optimization for multiple lora models.'
                        )
                        ret_data['error'] = error
                        ret_data['success'] = 0
                    else:
                        self.pipe.unet.load_attn_procs(lora_paths[0],
                                                       use_safetensors=False)
                        ret_data['success'] = 1

            except Exception as e:
                traceback.print_exc()
                torch.cuda.empty_cache()
                error = UnExpectedServerError(str(e))
                ret_data['error'] = error
                ret_data['success'] = 0

        return ret_data

    def load_controlnet(self, pipe, path: str, process_func: str):
        """
            Load the controlnet from the given path and apply processing based on the process_func.

            Args:
                pipe: The pipe object.
                path (str): Path to the controlnet file.
                process_func (str): Function to process the controlnet.

            Returns:
                None
        """
        if (path.endswith('.safetensors')):
            state_dict = load_file(path)
        else:
            state_dict = torch.load(path, map_location='cpu')

        pipe.controlnet.load_state_dict(state_dict)

        if self.use_blade and self.blade_load:
            patch_conv_weights(pipe.controlnet)
            if process_func == 'normal':
                pipe.controlnet.set_channel_order('bgr')
            else:
                pipe.controlnet.set_channel_order('rgb')
        else:
            if process_func == 'normal':
                pipe.controlnet.config.controlnet_conditioning_channel_order = 'bgr'
            else:
                pipe.controlnet.config.controlnet_conditioning_channel_order = 'rgb'

    def change_controlnet(self, controlnet_path: str,
                          process_func: Union[str, None]) -> Dict[str, Any]:
        """
            Change the controlnet used by the given controlnet_path.

            Args:
                controlnet_path (str): Path to the controlnet file.
                process_func (str): Function name to mark slight different by different controlnet model.

            Returns:
                dict: Return data containing the success flag and error message (if any).
        """
        ret_data = {}

        controlnet_path = os.path.join(self.model_dir, controlnet_path)

        if not os.path.exists(controlnet_path):
            error = InputFormatError(
                'controlnet_path path: {} not exists! Make sure you have mount the correct oss path'
                .format(controlnet_path))
            ret_data['error'] = error
            ret_data['success'] = 0
            return ret_data
        else:
            try:
                # change controlnet weight and some attrs
                self.load_controlnet(self.pipe, controlnet_path, process_func)

            except Exception as e:
                traceback.print_exc()
                error = UnExpectedServerError('Load controlnet error: ' +
                                              str(e))
                ret_data['error'] = error
                ret_data['success'] = 0
                return ret_data

        return ret_data

    def text_to_image(self, input_datas: Dict[str, Any]) -> Any:
        """
            Generate images based on the input text.

            :param input_datas: dict, input data containing various parameters.
            :return: generated images.
        """

        self.seed = input_datas.get('seed', random.getrandbits(64))
        self.generator = torch.Generator(device=self.device).manual_seed(
            (int)(self.seed))

        # diffusers only support the input width height that can be devided by 8
        new_width = int(
            (self.width // 8) * 8) if (self.width % 8 == 0) else int(
                (self.width // 8 + 1) * 8)
        new_height = int(
            (self.height // 8) * 8) if (self.height % 8 == 0) else int(
                (self.height // 8 + 1) * 8)

        # generate
        with torch.no_grad():
            images = self.pipe.text2img(
                self.prompt,
                height=new_height,
                width=new_width,
                generator=self.generator,
                num_inference_steps=self.inference_steps,
                guidance_scale=self.guidance_scale,
                negative_prompt=self.negative_prompt,
                num_images_per_prompt=self.n_samples,
            )

        return images

    def image_to_image(self, input_datas: Dict[str, Any]) -> Any:
        """
            Perform image-to-image transformation based on the input data.

            :param input_datas: dict, input data containing various parameters.
            :return: transformed images or error msg.
        """
        self.seed = input_datas.get('seed', random.getrandbits(64))
        self.generator = torch.Generator(device=self.device).manual_seed(
            (int)(self.seed))

        init_image = input_datas['image_pil']

        # transform image
        # mode: Specifies the mode of image transformation. 0 - Stretch, 1 - Crop, 2 - Padding. Defaults to 0.
        resize_mode = int(input_datas.get('resize_mode', 0))
        try:
            init_image = transform_image(init_image, self.width, self.height,
                                         resize_mode)
        except Exception as e:
            traceback.print_exc()
            error = UnExpectedServerError(str(e))

            return error

        strength = input_datas.get('denoising_strength', 0.55)
        # The too small value will cause unexpected error in blade
        if strength < 0.1:
            logging.warning(
                'You should put denoising_strength larger than 0.1. Change to 0.1 by default.'
            )
            strength = 0.1

        # generate
        with torch.no_grad():
            images = self.pipe.img2img(
                prompt=self.prompt,
                image=init_image,
                strength=strength,
                generator=self.generator,
                num_inference_steps=self.inference_steps,
                guidance_scale=self.guidance_scale,
                negative_prompt=self.negative_prompt,
                num_images_per_prompt=self.n_samples,
            )

        return images

    def inpaint_image(self, input_datas: Dict[str, Any]) -> Any:
        """
            Inpaint an image based on the input data.

            :param input_datas: dict, input data containing various parameters.
            :return: inpainted images or error msg.
        """
        strength = input_datas.get('denoising_strength', 0.55)
        # The too small value will cause unexpected error in blade
        if strength < 0.1:
            logging.warning(
                'You should put denoising_strength larger than 0.1. Change to 0.1 by default.'
            )
            strength = 0.1

        self.seed = input_datas.get('seed', random.getrandbits(64))
        self.generator = torch.Generator(device=self.device).manual_seed(
            (int)(self.seed))

        init_image = input_datas['image_pil']
        mask_image = input_datas['mask_pil']

        # transform image
        # mode: Specifies the mode of image transformation. 0 - Stretch, 1 - Crop, 2 - Padding. Defaults to 0.
        resize_mode = int(input_datas.get('resize_mode', 0))

        try:
            init_image = transform_image(init_image, self.width, self.height,
                                         resize_mode)
            mask_image = transform_image(mask_image, self.width, self.height,
                                         resize_mode)
        except Exception as e:
            traceback.print_exc()
            error = UnExpectedServerError(str(e))

            return error

        # generate
        with torch.no_grad():
            images = self.pipe.inpaint(
                prompt=self.prompt,
                image=init_image,
                mask_image=mask_image,
                generator=self.generator,
                num_inference_steps=self.inference_steps,
                guidance_scale=self.guidance_scale,
                negative_prompt=self.negative_prompt,
                num_images_per_prompt=self.n_samples,
                strength=strength,
            )

        return images

    def outpaint_image(self, input_datas: Dict[str, Any]) -> Any:
        """
            Outpaint an image based on the input data.

            :param input_datas: dict, input data containing various parameters.
            :return: outpainted images or error msg.
        """
        strength = input_datas.get('denoising_strength', 0.55)
        # The too small value will cause unexpected error in blade
        if strength < 0.1:
            logging.warning(
                'You should put denoising_strength larger than 0.1. Change to 0.1 by default.'
            )
            strength = 0.1

        self.seed = input_datas.get('seed', random.getrandbits(64))
        self.generator = torch.Generator(device=self.device).manual_seed(
            (int)(self.seed))

        init_image = input_datas['image_pil']

        # generate mask
        # expand pixels [left,right,up,down]
        expand = input_datas.get('expand', [128, 128, 128, 128])
        expand_type = input_datas.get('expand_type', 'copy')
        init_image, mask_image = generate_mask_and_img_expand(
            init_image, expand, expand_type)

        with torch.no_grad():
            images = self.pipe.inpaint(
                prompt=self.prompt,
                image=init_image,
                mask_image=mask_image,
                generator=self.generator,
                num_inference_steps=self.inference_steps,
                guidance_scale=self.guidance_scale,
                negative_prompt=self.negative_prompt,
                num_images_per_prompt=self.n_samples,
                strength=strength,
            )

        return images

    def controlnet_image(self, input_datas: Dict[str, Any]) -> Any:
        """
            Process the input image using ControlNet with control image generated by process_func.

            :param input_datas: Dict[str, Any], a dictionary containing input data.
            :return: Any, the processed images or error msg.
        """
        self.seed = input_datas.get('seed', random.getrandbits(64))
        self.generator = torch.Generator(device=self.device).manual_seed(
            (int)(self.seed))

        init_image = input_datas['image_pil']

        # transform
        resize_mode = int(input_datas.get('resize_mode', 0))

        try:
            init_image = transform_image(init_image, self.width, self.height,
                                         resize_mode)
        except Exception as e:
            traceback.print_exc()
            error = UnExpectedServerError(
                'Transform img failed in controlnet:' + str(e))
            return error

        # process the img to generate control image
        process_func = input_datas.get('process_func', None)

        if process_func is not None:
            process_func_list = [
                'canny', 'depth', 'hed', 'mlsd', 'normal', 'openpose',
                'scribble', 'seg'
            ]
            if process_func not in process_func_list:
                error = InputFormatError(
                    'We only support process function list of {}. But got {}.'.
                    format(process_func_list, process_func))

                return error

            try:
                init_image = preprocess_control(init_image, process_func,
                                                self.pretrain_dir)
            except Exception as e:
                traceback.print_exc()
                error = UnExpectedServerError(
                    'Preprocess image for controlnet error:' + str(e))

                return error

        controlnet_conditioning_scale = input_datas.get('control_scale', 1.0)

        with torch.no_grad():
            images = self.pipe(
                prompt=self.prompt,
                image=init_image,
                height=self.height,
                width=self.width,
                generator=self.generator,
                num_inference_steps=self.inference_steps,
                guidance_scale=self.guidance_scale,
                negative_prompt=self.negative_prompt,
                num_images_per_prompt=self.n_samples,
                controlnet_conditioning_scale=controlnet_conditioning_scale)

        return images

    def generation_process(self, input_datas):
        """
            Process the data received from the client for image generation.
            inputs format:
                input_datas (dict): A dictionary with the following fields:
                    'text'(str): the prompt guidance.
                    'skip_translation'(bool): Whether skipping the translation.
                    'num_inference_steps'(int): how many steps to inference in ldm.
                    'num_images'(int): how many images to generate
                    'lora_path'(str): the path of the lora model.
                    'controlnet_path'(str): the path of the controlnet model.
                    'process_func'(str): the preporcess type for controlnet
                    'prompt'(str): the prompt for generation.
                    'steps'(int): the number of inference steps.
                    'image_num'(int): the number of images to generate.
                    'cfg_scale'(float): the scale of the guidance.
                    'negative_prompt'(str): the negative prompt for generating diverse samples.
                    'width'(int): the width of the image to generate.
                    'height'(int): the height of the image to generate.
                    'lora_attn'(float): the attention scale for lora.
                    'task_id'(str): the id of the task for generation.
                    'save_dir'(str): the directory to save the generated images.
                    'use_base64'(bool): whether to encode the images in base64.
                    'seed'(int): the random seed for generation.
                    'image_link'(str): the link of the initial image.
                    'mask_link'(str): the link of the mask image for inpainting.
                    'resize_mode'(int): the resize mode for the initial image.
                    'denoising_strength'(float): the strength of the image denoising for i2i and inpainting.

            Returns:
                dict: A dictionary with the following fields:
                    'text'(srt): the prompt guidence.
                    'images_base64'(list[array]): array contains the generated images which were encoded from base64.
                    'success'(bool): Whether the process was successful
                    'error': error info if existed
                    'image_link'(str): http link in oss
                    'oss_link'(list[str]): oss link
                    'is_nsfw'(list[bool]): whether the generate image is black
        """

        ret_data = {}

        # ----------- 1. Change lora model -----------
        if 'lora_path' in input_datas.keys():
            need_change_lora = True
        else:
            need_change_lora = False

        # change lora
        if need_change_lora:
            lora_paths = input_datas['lora_path']
            if isinstance(lora_paths, str):
                lora_paths = [lora_paths]

            lora_attns = input_datas.get('lora_attn', [0.75] * len(lora_paths))

            if isinstance(lora_attns, float):
                lora_attns = [lora_attns] * len(lora_paths)

            result_change_lora = self.change_lora(lora_paths, lora_attns)

            if 'error' in result_change_lora.keys():
                # change lora failed
                return result_change_lora
            else:
                logging.info('successully change lora model!')

        # ----------- 2. Change controlnet model -----------
        if 'controlnet_path' in input_datas.keys(
        ) and self.service_name != 'controlnet':
            error = InputFormatError(
                'You should set --func_name controlnet to use controlnet.')
            ret_data['error'] = error
            ret_data['success'] = 0
            return ret_data

        if 'controlnet_path' in input_datas.keys():
            need_change_controlnet = True
        else:
            need_change_controlnet = False

        if need_change_controlnet:
            controlnet_path = input_datas['controlnet_path']
            process_func = input_datas.get('process_func', None)

            result_change_controlnet = self.change_controlnet(
                controlnet_path, process_func)

            if 'error' in result_change_controlnet.keys():
                # change controlnet failed
                return result_change_controlnet
            else:
                logging.info('successully change controlnet model!')

        # ----------- 3. Load post prameters -----------
        self.prompt = input_datas.get('prompt', self.prompt)
        self.inference_steps = input_datas.get('steps', self.inference_steps)
        self.n_samples = input_datas.get('image_num', self.n_samples)
        self.guidance_scale = input_datas.get('cfg_scale', self.guidance_scale)
        self.negative_prompt = input_datas.get('negative_prompt',
                                               self.negative_prompt)
        self.width = input_datas.get('width', self.width)
        self.height = input_datas.get('height', self.height)
        self.task_id = input_datas.get('task_id', 'default')

        # ----------- 4. Set save root -----------
        self.save_sub_dir = input_datas.get('save_dir', 'result')

        # save path
        sample_path = self.save_dir
        sample_path = os.path.join(sample_path, self.save_sub_dir)
        os.makedirs(sample_path, exist_ok=True)

        # return image path
        oss_path = self.oss_save_dir
        oss_path = os.path.join(oss_path, self.save_sub_dir)

        bucket_name = self.oss_save_dir.split('/')[2]
        http_path = oss_path.replace(
            'oss://{}'.format(bucket_name),
            'https://{}.oss-cn-{}.aliyuncs.com'.format(bucket_name,
                                                       self.oss_region))

        # ----------- 5. Image generation -----------
        action_map = {
            't2i': self.text_to_image,
            'i2i': self.image_to_image,
            'inpaint': self.inpaint_image,
            'outpaint': self.outpaint_image,
            'controlnet': self.controlnet_image
        }

        if self.func_name not in action_map:
            error = InputFormatError('Invalid func_name: {}'.format(
                self.func_name))
            ret_data['error'] = error
            ret_data['success'] = 0
            ret_data['task_id'] = self.task_id
            return ret_data

        try:
            res = action_map[self.func_name](input_datas)
            if isinstance(res, dict):
                # success
                images = res
                ret_data['success'] = 1
                ret_data['task_id'] = self.task_id
            else:
                # error
                ret_data['error'] = res
                ret_data['success'] = 0
                ret_data['task_id'] = self.task_id

        except Exception as e:
            traceback.print_exc()
            torch.cuda.empty_cache()
            error = UnExpectedServerError(str(e))
            ret_data['error'] = error
            ret_data['success'] = 0
            ret_data['task_id'] = self.task_id

        if 'error' in ret_data.keys():
            return ret_data

        # ----------- 6. Save result -----------
        try:
            use_base64 = input_datas.get('use_base64', False)

            images_base64 = []
            image_url = []
            oss_url = []

            for i in range(0, self.n_samples):
                if self.func_name == 't2i':
                    # transform image (resize to the defined size)
                    resize_mode = int(input_datas.get('resize_mode', 0))
                    image = transform_image(images.images[i], self.width,
                                            self.height, resize_mode)
                    image = images.images[i].resize((self.width, self.height))

                else:
                    image = images.images[i]

                imgtime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                rtg = random.randint(0, 100)
                tmp_path = '%s_%s_%d.png' % (self.task_id, imgtime, rtg)

                save_path = os.path.join(sample_path, tmp_path)

                save_oss_path = os.path.join(oss_path, tmp_path)
                save_http_path = os.path.join(http_path, tmp_path)

                image.save(save_path)
                oss_url.append(save_oss_path)
                image_url.append(save_http_path)

                # save base64
                if use_base64:
                    with open(save_path, 'rb') as f:
                        img_data = f.read()
                        base64_data = base64.urlsafe_b64encode(
                            img_data)  # base64
                        images_base64.append(str(base64_data, 'utf-8'))

            torch.cuda.empty_cache()

            logging.info(
                f'Your samples are ready and waiting for you here: \n{sample_path} \n'
                f' \nEnjoy.')

            ret_data['prompt'] = self.prompt
            ret_data['seed'] = str(self.seed)
            ret_data['image_url'] = image_url
            ret_data['oss_url'] = oss_url
            ret_data['success'] = 1
            ret_data['task_id'] = self.task_id
            ret_data['is_nsfw'] = images.nsfw_content_detected

            # save base64
            if use_base64:
                ret_data['images_base64'] = images_base64

        except Exception as e:
            traceback.print_exc()
            torch.cuda.empty_cache()
            error = UnExpectedServerError('Save result error: ' + str(e))
            ret_data['error'] = error
            ret_data['success'] = 0
            ret_data['task_id'] = self.task_id

        return ret_data

    def post_process(self, data):
        """
        encode the data after process
        """
        result_str = json.dumps(data).encode('utf-8')

        return result_str

    def data_process(self, data):
        """
            Processes the input data and returns a dictionary containing the processed information.

            Args:
            data: A byte string containing the input data.

            Returns:
            A dictionary containing the processed information.

            Raises:
            JsonParseError: If there is an error in parsing the JSON from the input data.
            InputFormatError: If any of the required keys are missing from the input data or if the input data is not in the valid format.
            UnExpectedServerError: If there is an unexpected error during translation or if the use_translate flag is not set properly.
        """

        ret_data = {}

        try:
            ret_data = data.decode('utf-8')
            ret_data = json.loads(ret_data)
        except Exception as e:
            traceback.print_exc()
            error = JsonParseError(str(e))
            ret_data['error'] = error
            ret_data['success'] = 0
            return ret_data

        # ----------- 1. Check the primary key -----------
        need_keys = ['task_id', 'prompt']

        for need_key in need_keys:
            if need_key not in ret_data.keys():
                error = InputFormatError(
                    'key: {} must be provided!'.format(need_key))
                ret_data['error'] = error
                ret_data['success'] = 0
                return ret_data

        # get task_id
        if 'task_id' in ret_data.keys():
            self.task_id = ret_data.get('task_id', 'no_task_id')

        # ----------- 2. Choose a function name -----------
        if self.service_name == 'base':
            self.func_name = ret_data.get('func_name', 't2i')
            base_func_list = ['t2i', 'i2i', 'inpaint', 'outpaint']
            if self.func_name not in base_func_list:
                logging.warning(
                    'We only allow function list {}, but got {}. We use t2i by default'
                    .format(base_func_list, self.func_name))
                self.func_name = 't2i'

        elif self.service_name == 'controlnet':
            self.func_name = 'controlnet'

        logging.info('using function: {}'.format(self.func_name))

        # ----------- 3. Download/process image link -----------
        # check and read image
        if self.func_name == 'i2i' or self.func_name == 'controlnet' or self.func_name == 'inpaint' or self.func_name == 'outpaint':
            if 'image_link' not in ret_data.keys(
            ) and 'image_base64' not in ret_data.keys():
                error = InputFormatError(
                    'key: image_link or image_base64 must be provided when using function i2i/controlnet/inpaint/outpaint'
                )
                ret_data['error'] = error
                ret_data['success'] = 0
            elif 'image_link' in ret_data.keys(
            ) and 'image_base64' in ret_data.keys():
                error = InputFormatError(
                    'key: image_link or image_base64 must be provided when using function  i2i/controlnet/inpaint/outpaint, but got both!'
                )
                ret_data['error'] = error
                ret_data['success'] = 0
            else:
                # read image
                if 'image_link' in ret_data.keys():
                    # check the link must with internal when using oss link in eas
                    if not self.local_debug and 'oss-cn-{}.aliyuncs.com'.format(
                            self.oss_region
                    ) in ret_data['image_link'] and 'internal' not in ret_data[
                            'image_link']:
                        logging.warning(
                            'oss link must be provided with internal in eas service! auto converting now'
                        )
                        ret_data['image_link'] = ret_data[
                            'image_link'].replace('.aliyuncs.com/',
                                                  '-internal.aliyuncs.com/')

                    try:
                        ret_data['image_pil'] = download_image(
                            ret_data['image_link'])
                    except:
                        traceback.print_exc()
                        error = InputFormatError(
                            'image url %s response faild!' %
                            (ret_data['image_link']))
                        ret_data['error'] = error
                        ret_data['success'] = 0
                        ret_data['task_id'] = self.task_id
                else:
                    try:
                        image_b64 = base64.b64decode(ret_data['image_base64'])
                        # ret_data['image_pil'] = Image.fromarray(
                        #     np.array(imdecode(image_b64)))
                        ret_data['image_pil'] = Image.open(
                            io.BytesIO(image_b64))
                    except:
                        traceback.print_exc()
                        error = InputFormatError(
                            'Image data not valid, please check your image data (image_base64)'
                        )
                        ret_data['error'] = error
                        ret_data['success'] = 0
                        ret_data['task_id'] = self.task_id

        # check and read image for mask
        if self.func_name == 'inpaint':
            if 'mask_link' not in ret_data.keys(
            ) and 'mask_base64' not in ret_data.keys():
                error = InputFormatError(
                    'key: mask_link or mask_base64 must be provided when using function inpaint'
                )
                ret_data['error'] = error
                ret_data['success'] = 0
            elif 'mask_link' in ret_data.keys(
            ) and 'mask_base64' in ret_data.keys():
                error = InputFormatError(
                    'key: mask_link or mask_base64 must be provided when using function i2i or controlnet, but got both!'
                )
                ret_data['error'] = error
                ret_data['success'] = 0
            else:
                # read image
                if 'mask_link' in ret_data.keys():
                    # check the link must with internal when using oss link in eas
                    if not self.local_debug and 'oss-cn-{}.aliyuncs.com'.format(
                            self.oss_region
                    ) in ret_data['mask_link'] and 'internal' not in ret_data[
                            'mask_link']:
                        logging.warning(
                            'oss mask link must be provided with internal! Auto convert now.'
                        )
                        ret_data['mask_link'] = ret_data['mask_link'].replace(
                            '.aliyuncs.com/', '-internal.aliyuncs.com/')

                    try:
                        ret_data['mask_pil'] = download_image(
                            ret_data['mask_link'])
                    except:
                        traceback.print_exc()
                        error = InputFormatError(
                            'mask url %s response faild!' %
                            (ret_data['mask_link']))
                        ret_data['error'] = error
                        ret_data['success'] = 0
                        ret_data['task_id'] = self.task_id
                else:
                    try:
                        image_b64 = base64.b64decode(ret_data['mask_base64'])
                        # ret_data['mask_pil'] = Image.fromarray(
                        #     np.array(imdecode(image_b64)))
                        ret_data['mask_pil'] = Image.open(
                            io.BytesIO(image_b64))

                    except:
                        traceback.print_exc()
                        error = InputFormatError(
                            'Mask image data not valid, please check your image data (mask_base64)'
                        )
                        ret_data['error'] = error
                        ret_data['success'] = 0
                        ret_data['task_id'] = self.task_id

        if 'error' in ret_data.keys():
            return ret_data

        # ----------- 4. Traslate prompt (chinese to english) -----------
        need_translate = ret_data.get('use_translate', self.use_translate)

        if need_translate:
            try:
                prompt = ret_data.get('prompt', 'fake result')
                outputs = self.pipe_trans(input=prompt)

                translate_prompt = outputs['translation']
                logging.info('Translate Result: ' + translate_prompt)
                ret_data['prompt'] = translate_prompt

            except Exception as e:
                traceback.print_exc()
                error = UnExpectedServerError(
                    'You need to set --use_translate to use the translation or post use_translate to False'
                )
                ret_data['error'] = error
                ret_data['success'] = 0
                ret_data['task_id'] = self.task_id

        return ret_data

    def process(self, data):
        """
        Process the given data.

        Parameters:
        data -- The data to be processed

        Returns:
        The processed data

        """

        # ----------- 1. Process data -----------
        data = self.data_process(data)

        # ----------- 2. Use blade optimzation model (once blade optimization done in subprocess) -----------
        if self.use_blade and not self.blade_load:
            if self.use_controlnet:
                if all(
                        os.path.exists(p) for p in [
                            self.encoder_path, self.unet_path,
                            self.decoder_path, self.controlnet_path
                        ]):
                    try:
                        logging.info(
                            'Load blade model with controlnet to init')
                        self.pipe = load_blade_model(self.pipe,
                                                     self.encoder_path,
                                                     self.unet_path,
                                                     self.decoder_path,
                                                     self.controlnet_path)
                        self.blade_load = True
                    except:
                        traceback.print_exc()
                        logging.warning(
                            'Load blade model with controlnet failed! Use the original diffuser model!'
                        )
            else:
                if all(
                        os.path.exists(p) for p in
                    [self.encoder_path, self.unet_path, self.decoder_path]):
                    try:
                        logging.info('load blade model to init')
                        self.pipe = load_blade_model(self.pipe,
                                                     self.encoder_path,
                                                     self.unet_path,
                                                     self.decoder_path)

                        self.blade_load = True
                    except:
                        traceback.print_exc()
                        logging.warning(
                            'Load blade model failed! Use the original diffuser model!'
                        )

        # ----------- 3. Image Generation -----------
        if 'error' in data.keys():
            # data process failed
            return get_result_str(error=data['error'])
        else:
            ret_data = self.generation_process(data)

            # mark whether blade optimation is done
            ret_data['use_blade'] = self.blade_load

            # sub lora to change lora next time
            if self.mount_lora:
                unload_lora(self.pipe)
                self.mount_lora = False

            # return
            if ret_data['success']:
                return self.post_process(ret_data), 200
            else:
                torch.cuda.empty_cache()
                return get_result_str(result_dict=ret_data,
                                      error=ret_data['error'])


def run(args_unused):
    torch.multiprocessing.set_start_method('spawn', force=True)

    # paramter worker_threads indicates concurrency of processing
    allspark.default_properties().put('rpc.keepalive', 500000)

    runner = MyProcessor(worker_threads=1, endpoint='0.0.0.0:8000/')
    runner.run()


if __name__ == '__main__':
    # paramter worker_threads indicates concurrency of processing
    parser = argparse.ArgumentParser('ev eas processor local runner')
    parser.add_argument('--func_name',
                        type=str,
                        required=True,
                        help='use func name base or controlnet')
    parser.add_argument('--oss_save_dir',
                        type=str,
                        required=True,
                        help='oss save dir use to generate to a http link')
    parser.add_argument('--region',
                        type=str,
                        required=True,
                        help='oss region use to generate to a http link')
    parser.add_argument('--save_dir',
                        type=str,
                        required=False,
                        default='result',
                        help='save dir only work when local debug')
    parser.add_argument('--model_dir',
                        type=str,
                        default='models',
                        help='local model dir only work when local debug')
    parser.add_argument('--local_debug',
                        action='store_true',
                        help='in local debug mode')
    parser.add_argument('--use_blade',
                        action='store_true',
                        help='whether to use blade')
    parser.add_argument('--use_translate',
                        action='store_true',
                        help='whether to use translation')
    parser.add_argument('--close_safety',
                        action='store_true',
                        help='whether to close_safety')
    parser.add_argument('--not_use_default_blade',
                        action='store_true',
                        help='whether to use the pre optimized blade model')
    parser.add_argument(
        '--remove_opt',
        action='store_true',
        help='remove optimized model when first init by new image.')

    FLAGS = parser.parse_args()
    run(FLAGS)
