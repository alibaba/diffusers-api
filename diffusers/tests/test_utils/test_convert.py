import os
import tempfile
import unittest

import torch
from diffusers import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion.convert_from_ckpt import \
    download_from_original_stable_diffusion_ckpt
from tests.ut_config import BASE_MODEL_PATH, CKPT_PATH, LORA_PATH
from utils.convert import (convert_base_model_to_diffuser,
                           convert_lora_safetensor_to_bin, convert_name_to_bin)


class TestModelConvert(unittest.TestCase):
    def test_convert_lora_safetensor_to_bin(self):
        # for .safetensors to load by ori diffuser api (only attn in unet will be loaded)
        with tempfile.TemporaryDirectory() as temp_dir:
            bin_path = LORA_PATH.replace('.safetensors', '.bin')
            bin_path = os.path.join(temp_dir, 'bin_model.pth')
            convert_lora_safetensor_to_bin(LORA_PATH, bin_path)
            self.assertTrue(os.path.exists(bin_path))

            # Load the converted lora model
            pipe = StableDiffusionPipeline.from_pretrained(
                BASE_MODEL_PATH,
                revision='fp16',
                torch_dtype=torch.float16,
                safety_checker=None)
            pipe.unet.load_attn_procs(bin_path, use_safetensors=False)

    def test_convert_base_model_to_diffuser(self):
        # convert .safetensors to multiple dirs
        from_safetensors = True

        with tempfile.TemporaryDirectory() as temp_dir:
            convert_base_model_to_diffuser(CKPT_PATH, temp_dir,
                                           from_safetensors)
            files = os.listdir(temp_dir)
            print(files)
            need_file_list = [
                'feature_extractor', 'model_index.json', 'safety_checker',
                'scheduler', 'text_encoder', 'tokenizer', 'unet', 'vae'
            ]
            self.assertTrue(set(need_file_list).issubset(set(files)))

            # Load the converted base model
            pipe = StableDiffusionPipeline.from_pretrained(
                temp_dir,
                revision='fp16',
                torch_dtype=torch.float16,
                safety_checker=None)


if __name__ == '__main__':
    unittest.main()
