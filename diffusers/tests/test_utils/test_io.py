import os
import unittest

from PIL import Image

import torch
from tests.ut_config import (BASE_MODEL_PATH, CONTROLNET_MODEL_PATH,
                             CUSTOM_PIPELINE, IMAGE_DIR, PRETRAIN_DIR,
                             SAVE_DIR)
from utils.image_process import preprocess_control
from utils.io import load_diffusers_pipeline


class TestLoadDiffusersPipeline(unittest.TestCase):
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

    def test_base_mode(self):
        mode = 'base'
        close_safety = False

        pipe = load_diffusers_pipeline(BASE_MODEL_PATH, None, None,
                                       self.device, mode, close_safety,
                                       CUSTOM_PIPELINE)

        self.assertIsNotNone(pipe)

        # t2i
        with torch.no_grad():
            res = pipe.text2img(
                prompt=self.prompt,
                num_inference_steps=self.num_inference_steps,
                num_images_per_prompt=self.num_images_per_prompt)
            image = res.images[0]
            self.assertIsInstance(image, Image.Image)
            image.save(os.path.join(SAVE_DIR, 't2i.jpg'))

        # i2i
        with torch.no_grad():
            res = pipe.img2img(
                prompt=self.prompt,
                image=self.image,
                num_inference_steps=self.num_inference_steps,
                num_images_per_prompt=self.num_images_per_prompt)
            image = res.images[0]
            self.assertIsInstance(image, Image.Image)
            image.save(os.path.join(SAVE_DIR, 'i2i.jpg'))

        # inpaint
        with torch.no_grad():
            res = pipe.inpaint(
                prompt=self.prompt,
                image=self.image,
                mask_image=self.mask,
                num_inference_steps=self.num_inference_steps,
                num_images_per_prompt=self.num_images_per_prompt)
            image = res.images[0]
            self.assertIsInstance(image, Image.Image)
            image.save(os.path.join(SAVE_DIR, 'inpaint.jpg'))

    def test_controlnet_mode(self):
        mode = 'controlnet'
        close_safety = False

        pipe = load_diffusers_pipeline(BASE_MODEL_PATH, None,
                                       CONTROLNET_MODEL_PATH, self.device,
                                       mode, close_safety, CUSTOM_PIPELINE)

        self.assertIsNotNone(pipe)

        with torch.no_grad():
            process_image = preprocess_control(self.image, 'canny',
                                               PRETRAIN_DIR)
            res = pipe(prompt=self.prompt,
                       image=process_image,
                       num_inference_steps=self.num_inference_steps,
                       num_images_per_prompt=self.num_images_per_prompt)
            image = res.images[0]
            self.assertIsInstance(image, Image.Image)
            image.save(os.path.join(SAVE_DIR, 'control.jpg'))

    def test_invalid_mode(self):
        with self.assertRaises(ValueError):
            mode = 'invalid'
            close_safety = False
            load_diffusers_pipeline(BASE_MODEL_PATH, None, None, self.device,
                                    mode, close_safety, CUSTOM_PIPELINE)


if __name__ == '__main__':
    unittest.main()
