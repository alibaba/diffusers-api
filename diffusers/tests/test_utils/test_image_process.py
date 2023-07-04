import os
import unittest

import numpy as np
from PIL import Image

import torch
# need to be import or an malloc error will be occured by controlnet_aux
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
from tests.ut_config import IMAGE_DIR, PRETRAIN_DIR, SAVE_DIR
from utils.image_process import (generate_mask_and_img_expand,
                                 preprocess_control, transform_image)


class TestImageProcessing(unittest.TestCase):
    def setUp(self):
        # image for testing
        img_path = os.path.join(IMAGE_DIR, 'room.png')
        self.image = Image.open(img_path).convert('RGB')

    def test_preprocess_control(self):
        # Test with valid process_func
        process_func_list = [
            'canny', 'depth', 'hed', 'mlsd', 'normal', 'openpose', 'scribble',
            'seg'
        ]
        for process_func in process_func_list:
            print('Process: {}'.format(process_func))
            processed_image = preprocess_control(self.image, process_func,
                                                 PRETRAIN_DIR)
            self.assertIsInstance(processed_image, Image.Image)
            processed_image.save(
                os.path.join(SAVE_DIR, 'pre_{}.jpg'.format(process_func)))

        # Test with an invalid process_func
        error_message = preprocess_control(self.image, 'invalid_func',
                                           PRETRAIN_DIR)
        self.assertIsInstance(error_message, str)

    def test_transform_image(self):
        test_params = {0: 'Stretch', 1: 'Crop', 2: 'Pad'}

        expected_sizes = [(1024, 1024), (768, 1024), (1024, 768)]

        for expected_size in expected_sizes:
            width, height = expected_size
            for mode, mode_type in test_params.items():
                print('Process: {}, width: {}, height: {}'.format(
                    mode, width, height))
                transformed_image = transform_image(self.image, width, height,
                                                    mode)
                self.assertEqual(transformed_image.size, (width, height))
                transformed_image.save(
                    os.path.join(SAVE_DIR, '{}.jpg'.format(mode_type)))

    def test_generate_mask_and_img_expand(self):
        expand = (10, 20, 30, 40)

        left, right, up, down = expand
        width, height = self.image.size
        new_width, new_height = width + left + right, height + up + down

        expand_list = ['copy', 'reflect']
        for expand_type in expand_list:
            expanded_image, mask = generate_mask_and_img_expand(
                self.image, expand, expand_type)
            self.assertEqual(expanded_image.size, (new_width, new_height))
            self.assertEqual(mask.size, (new_width, new_height))

            expanded_image.save(
                os.path.join(SAVE_DIR,
                             'expanded_image_{}.jpg'.format(expand_type)))
            mask.save(os.path.join(SAVE_DIR,
                                   'mask_{}.jpg'.format(expand_type)))


if __name__ == '__main__':
    unittest.main()
