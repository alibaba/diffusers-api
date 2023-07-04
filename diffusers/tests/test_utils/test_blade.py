import os
import unittest

from PIL import Image

import torch
import torch_blade
from tests.ut_config import (BASE_MODEL_PATH, CONTROLNET_MODEL_PATH,
                             CUSTOM_PIPELINE, IMAGE_DIR, MODEL_DIR,
                             PRETRAIN_DIR, SAVE_DIR)
from utils.blade import load_blade_model, optimize_and_save_blade_model
from utils.image_process import preprocess_control
from utils.io import load_diffusers_pipeline

# important！or the blade result will be incorrect
os.environ['DISC_ENABLE_DOT_MERGE'] = '0'


class TestBladeOptimization(unittest.TestCase):
    def setUp(self):
        # hyper parameters
        self.prompt = 'a dog'
        img_path = os.path.join(IMAGE_DIR, 'image.png')
        mask_path = os.path.join(IMAGE_DIR, 'mask.png')
        self.image = Image.open(img_path).convert('RGB')
        self.mask = Image.open(mask_path).convert('RGB')
        self.num_inference_steps = 20
        self.num_images_per_prompt = 1
        self.device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')

    def test_optimize_and_save_blade_model_base(self):
        # save and optimize base model
        blade_dir = os.path.join(MODEL_DIR, 'optimized_model')
        os.makedirs(blade_dir, exist_ok=True)
        encoder_path = os.path.join(blade_dir, 'encoder.pt')
        unet_path = os.path.join(blade_dir, 'unet.pt')
        decoder_path = os.path.join(blade_dir, 'decoder.pt')
        controlnet_path = None
        mode = 'base'
        close_safety = False
        pipe = load_diffusers_pipeline(BASE_MODEL_PATH, None, None,
                                       self.device, mode, close_safety,
                                       CUSTOM_PIPELINE)
        optimize_and_save_blade_model(pipe, encoder_path, unet_path,
                                      decoder_path, controlnet_path)

        # save and optimize base model
        assert os.path.exists(
            encoder_path), f"Encoder path '{encoder_path}' does not exist!"
        assert os.path.exists(
            unet_path), f"UNet path '{unet_path}' does not exist!"
        assert os.path.exists(
            decoder_path), f"Decoder path '{decoder_path}' does not exist!"
        # load
        pipe = load_blade_model(pipe, encoder_path, unet_path, decoder_path,
                                controlnet_path)
        with torch.no_grad():
            res = pipe.text2img(
                prompt=self.prompt,
                num_inference_steps=self.num_inference_steps,
                num_images_per_prompt=self.num_images_per_prompt)
            image = res.images[0]
            self.assertIsInstance(image, Image.Image)
            image.save(os.path.join(SAVE_DIR, 't2i_blade.jpg'))

    def test_optimize_and_save_blade_model_controlnet(self):
        # save and optimize base model
        blade_dir = os.path.join(MODEL_DIR, 'optimized_control_model')
        os.makedirs(blade_dir, exist_ok=True)
        encoder_path = os.path.join(blade_dir, 'encoder.pt')
        unet_path = os.path.join(blade_dir, 'unet.pt')
        decoder_path = os.path.join(blade_dir, 'decoder.pt')
        controlnet_path = os.path.join(blade_dir, 'controlnet.pt')
        mode = 'controlnet'
        close_safety = False
        pipe = load_diffusers_pipeline(BASE_MODEL_PATH, None,
                                       CONTROLNET_MODEL_PATH, self.device,
                                       mode, close_safety, CUSTOM_PIPELINE)

        optimize_and_save_blade_model(pipe, encoder_path, unet_path,
                                      decoder_path, controlnet_path)

        # save and optimize base model
        assert os.path.exists(
            encoder_path), f"Encoder path '{encoder_path}' does not exist!"
        assert os.path.exists(
            unet_path), f"UNet path '{unet_path}' does not exist!"
        assert os.path.exists(
            decoder_path), f"Decoder path '{decoder_path}' does not exist!"
        assert os.path.exists(
            controlnet_path
        ), f"ControlNet path '{controlnet_path}' does not exist!"
        # load
        pipe = load_blade_model(pipe, encoder_path, unet_path, decoder_path,
                                controlnet_path)
        with torch.no_grad():
            process_image = preprocess_control(self.image, 'canny',
                                               PRETRAIN_DIR)
            res = pipe(prompt=self.prompt,
                       image=process_image,
                       num_inference_steps=self.num_inference_steps,
                       num_images_per_prompt=self.num_images_per_prompt)
            image = res.images[0]
            self.assertIsInstance(image, Image.Image)
            image.save(os.path.join(SAVE_DIR, 'control_blade.jpg'))


if __name__ == '__main__':
    unittest.main()
