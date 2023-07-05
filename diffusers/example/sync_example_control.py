"""
    post example when deploy the service name as controlnet
"""

import base64
import json
import os
import sys
from io import BytesIO

import requests
from PIL import Image, PngImagePlugin

ENCODING = 'utf-8'

hosts = 'http://xxx.cn-hangzhou.pai-eas.aliyuncs.com/api/predict/service_name'
head = {
    'Authorization': 'xxx'
}


def decode_base64(image_base64, save_file):
    img = Image.open(BytesIO(base64.urlsafe_b64decode(image_base64)))
    img.save(save_file)


def select_data(process_func):
    if process_func == 'canny':
        datas = json.dumps({
            'task_id': 'canny',
            'steps': 50,
            'image_num': 1,
            'width': 512,
            'height': 512,
            'image_link':
            'https://huggingface.co/lllyasviel/sd-controlnet-hed/resolve/main/images/man.png',
            'prompt': 'man',
            'process_func': 'canny',
        })
    elif process_func == 'depth':
        datas = json.dumps({
            'task_id': 'depth',
            'steps': 50,
            'image_num': 1,
            'width': 512,
            'height': 512,
            'image_link':
            'https://huggingface.co/lllyasviel/sd-controlnet-depth/resolve/main/images/stormtrooper.png',
            'prompt': "Stormtrooper's lecture",
            'controlnet_path':
            'new_controlnet/models--lllyasviel--sd-controlnet-depth/diffusion_pytorch_model.safetensors',  # use to change the controlnet path
            'process_func': 'depth',
        })
    elif process_func == 'hed':
        datas = json.dumps({
            'task_id': 'hed',
            'steps': 50,
            'image_num': 1,
            'width': 512,
            'height': 512,
            'image_link':
            'https://huggingface.co/lllyasviel/sd-controlnet-hed/resolve/main/images/man.png',
            'prompt': 'oil painting of handsome old man, masterpiece',
            'controlnet_path':
            'new_controlnet/models--lllyasviel--sd-controlnet-hed/diffusion_pytorch_model.safetensors',
            'process_func': 'hed',
        })
    elif process_func == 'mlsd':
        datas = json.dumps({
            'task_id': 'mlsd',
            'steps': 50,
            'image_num': 1,
            'width': 512,
            'height': 512,
            'image_link':
            'https://huggingface.co/lllyasviel/sd-controlnet-mlsd/resolve/main/images/room.png',
            'prompt': 'room',
            'controlnet_path':
            'new_controlnet/models--lllyasviel--sd-controlnet-mlsd/diffusion_pytorch_model.safetensors',
            'process_func': 'mlsd',
        })
    elif process_func == 'normal':
        datas = json.dumps({
            'task_id': 'normal',
            'steps': 50,
            'image_num': 1,
            'width': 512,
            'height': 512,
            'image_link':
            'https://huggingface.co/lllyasviel/sd-controlnet-normal/resolve/main/images/toy.png',
            'prompt': 'cute toy',
            'controlnet_path':
            'new_controlnet/models--fusing--stable-diffusion-v1-5-controlnet-normal/diffusion_pytorch_model.safetensors',
            'process_func': 'normal',
        })
    elif process_func == 'openpose':
        datas = json.dumps({
            'task_id': 'openpose',
            'steps': 50,
            'image_num': 1,
            'width': 512,
            'height': 512,
            'image_link':
            'https://huggingface.co/lllyasviel/sd-controlnet-openpose/resolve/main/images/pose.png',
            'prompt': 'chef in the kitchen',
            'controlnet_path':
            'new_controlnet/models--lllyasviel--sd-controlnet-openpose/diffusion_pytorch_model.safetensors',
            'process_func': 'openpose',
        })
    elif process_func == 'scribble':

        datas = json.dumps({
            'task_id': 'scribble',
            'steps': 50,
            'image_num': 1,
            'width': 512,
            'height': 512,
            'image_link':
            'https://huggingface.co/lllyasviel/sd-controlnet-scribble/resolve/main/images/bag.png',
            'prompt': 'bag',
            'controlnet_path':
            'new_controlnet/models--lllyasviel--sd-controlnet-scribble/diffusion_pytorch_model.safetensors',
            'process_func': 'scribble',
        })

    elif process_func == 'seg':
        datas = json.dumps({
            'task_id': 'seg',
            'steps': 50,
            'image_num': 1,
            'width': 512,
            'height': 512,
            'image_link':
            'https://huggingface.co/lllyasviel/sd-controlnet-seg/resolve/main/images/house.png',
            'prompt': 'house',
            'controlnet_path':
            'new_controlnet/models--lllyasviel--sd-controlnet-seg/diffusion_pytorch_model.safetensors',
            'process_func': 'seg',
        })
    else:
        raise ValueError('Invalid process_func value')

    return datas


process_func_list = [
    'canny', 'depth', 'hed', 'mlsd', 'normal', 'openpose', 'scribble', 'seg'
]

for process_func in process_func_list:
    datas = select_data(process_func)

    r = requests.post(hosts, data=datas, headers=head)
    # r = requests.post("http://0.0.0.0:8000/test", data=datas, timeout=1500)

    data = json.loads(r.content.decode('utf-8'))
    print(data.keys())

    if data['success']:
        print(data['image_url'])
        print(data['oss_url'])
        print(data['task_id'])
        print(data['use_blade'])
        print(data['seed'])
        print(data['is_nsfw'])
        if 'images_base64' in data.keys():
            for i, image_base64 in enumerate(data['images_base64']):
                decode_base64(image_base64,
                              './decode_ldm_base64_{}.png'.format(str(i)))

    else:
        print(data['error_msg'])
