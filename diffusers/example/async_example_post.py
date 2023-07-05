#!/usr/bin/env python

import json

from eas_prediction import PredictClient, StringRequest

if __name__ == '__main__':
    client = PredictClient('http://xxx.cn-hangzhou.pai-eas.aliyuncs.com/',
                           'service_name')
    client.set_token(
        'xxx')

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
