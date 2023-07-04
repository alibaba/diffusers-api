#!/usr/bin/env python

import json

from eas_prediction import PredictClient, StringRequest

if __name__ == '__main__':
    client = PredictClient('http://1502318844610933.cn-hangzhou.pai-eas.aliyuncs.com/',
                           'diffuser_base_ch_async')
    client.set_token(
        'NDJkZjAwMzdlNDMzMjFmYWI4ODZmYmJkMzgwZDEzNmFlNTkyNDBiOQ==')

    client.init()

    datas = json.dumps({
        'task_id': 'async',
        'prompt': '一个可爱的女孩',
        'steps': 100,
        'image_num': 3,
        'width': 512,
        'height': 512,
        'seed': '123',
    })

    request = StringRequest(datas)

    for x in range(0, 1):
        resp = client.predict(request)
        print(resp)
