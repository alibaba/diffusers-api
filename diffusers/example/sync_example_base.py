"""
    post example when deploy the service name as base
    set --use_transalte and upload the translate model to post chinese prompt
    translate model: https://www.modelscope.cn/models/damo/nlp_csanmt_translation_zh2en/summary
"""

import base64
import json
import os
import sys
from io import BytesIO

import requests
from PIL import Image, PngImagePlugin

ENCODING = 'utf-8'

hosts = 'http://1502318844610933.cn-hangzhou.pai-eas.aliyuncs.com/api/predict/diffuser_base_ch'
head = {
    'Authorization': 'ZmViY2IyNTY2ZDA3ZWUyMjcxZWQyZDgzNjA1NzhjODE3YmE5MDUyNw=='
}

# func_list = ['t2i','i2i','inpaint','outpaint']
func_list = ['outpaint']


def decode_base64(image_base64, save_file):
    img = Image.open(BytesIO(base64.urlsafe_b64decode(image_base64)))
    img.save(save_file)


def select_data(func_name):
    if func_name == 't2i':
        datas = json.dumps({
            'task_id':
            func_name,
            'prompt':
            '一只可爱的小猫',
            'func_name':
            func_name,  # or default is t2i
            'negative_prompt':
            'NSFW',
            'steps':
            50,
            'image_num':
            1,
            'width':
            512,
            'height':
            512,
            'lora_path':
            ['lora/animeLineartMangaLike_v30MangaLike.safetensors'],
            'lora_attn':
            1.0
        })
    elif func_name == 'i2i':
        datas = json.dumps({
            'image_link':
            'https://converter-offline-installer.oss-cn-hangzhou.aliyuncs.com/zxy/image.png',
            # 'image_base64': base64.b64encode(open('/mnt/xinyi.zxy/diffuser/models/bosi2/result/a001_20230602_075912_12.png', 'rb').read()).decode(ENCODING),
            'task_id':
            func_name,
            'prompt':
            'a cat',
            'func_name':
            func_name,
            'negative_prompt':
            'NSFW',
            'steps':
            50,
            'image_num':
            1,
            'width':
            512,
            'height':
            512,
            'lora_path':
            ['lora/animeLineartMangaLike_v30MangaLike.safetensors']
        })
    elif func_name == 'inpaint':
        datas = json.dumps({
            'image_link':
            'https://converter-offline-installer.oss-cn-hangzhou.aliyuncs.com/zxy/image.png',
            'mask_link':
            'https://converter-offline-installer.oss-cn-hangzhou.aliyuncs.com/zxy/mask.png',
            'task_id':
            func_name,
            'prompt':
            'a cat',
            'func_name':
            func_name,
            'negative_prompt':
            'NSFW',
            'steps':
            50,
            'image_num':
            1,
            'width':
            512,
            'height':
            512,
            'lora_path':
            ['lora/animeLineartMangaLike_v30MangaLike.safetensors']
        })
    elif func_name == 'outpaint':
        datas = json.dumps({
            'image_link':
            'https://converter-offline-installer.oss-cn-hangzhou.aliyuncs.com/zxy/image.png',
            # 'image_base64': base64.b64encode(open('/mnt/xinyi.zxy/diffuser/models/bosi2/result/a001_20230602_075912_12.png', 'rb').read()).decode(ENCODING),
            'task_id': func_name,
            'prompt': 'a cat',
            'func_name': func_name,
            'negative_prompt': 'NSFW',
            'steps': 50,
            'image_num': 1,
            'width': 512,
            'height': 512,
            'expand': [256, 256, 0, 0],  # [left,right,up,down]
            'expand_type': 'copy',  # or 'reflect',
            'denoising_strength': 0.6
            # 'lora_path': ['path_to_your_lora_model']
        })
    else:
        raise ValueError('Invalid process_func value')

    return datas


for func_name in func_list:
    datas = select_data(func_name)

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
                decode_base64(image_base64, './result_{}.png'.format(str(i)))

    else:
        print(data['error_msg'])
