import os

UT_ROOT = '/mnt/xinyi.zxy/diffuser/ut_test/'
MODEL_DIR = os.path.join(UT_ROOT, 'models/test_model')
BASE_MODEL_PATH = os.path.join(MODEL_DIR, 'base_model')
CONTROLNET_MODEL_PATH = os.path.join(MODEL_DIR, 'controlnet')
LORA_PATH = os.path.join(MODEL_DIR, 'lora_model/animeoutlineV4_16.safetensors')
LORA_PATH_BIN = os.path.join(MODEL_DIR, 'lora_model/pytorch_lora_weights.bin')

MODEL_DIR_NEW = os.path.join(UT_ROOT, 'models/new_model')
CKPT_PATH = os.path.join(MODEL_DIR_NEW, 'colorful_v26.safetensors')

PRETRAIN_DIR = os.path.join(UT_ROOT, 'models/pretrained_models')
SAVE_DIR = os.path.join(UT_ROOT, 'results')
IMAGE_DIR = os.path.join(UT_ROOT, 'images')

CUSTOM_PIPELINE = './lpw_stable_diffusion.py'
