import json

from eas_prediction import QueueClient

if __name__ == '__main__':
    # 创建输出队列对象，⽤于订阅读取输出结果数据。

    sink_queue = QueueClient(
        'http://1502318844610933.cn-hangzhou.pai-eas.aliyuncs.com',
        'diffuser_base_ch_async/sink')
    sink_queue.set_token(
        'NDJkZjAwMzdlNDMzMjFmYWI4ODZmYmJkMzgwZDEzNmFlNTkyNDBiOQ==')

    sink_queue.init()

    # 从输出队列中watch数据，窗⼝为1。
    i = 0
    watcher = sink_queue.watch(0, 1, auto_commit=False)
    for x in watcher.run():
        data = x.data.decode('utf-8')
        data = json.loads(data)
        print(data.keys())
        if data['success']:
            print(data['image_url'])
            print(data['oss_url'])
            print(data['task_id'])
            print(data['use_blade'])
            print(data['seed'])
            print(data['is_nsfw'])
        else:
            print(data['error_msg'])
        # 每次收到⼀个请求数据后处理完成后⼿动commit。
        sink_queue.commit(x.index)
        i += 1
        if i == 10:
            break

    # 关闭已经打开的watcher对象，每个客户端实例只允许存在⼀个watcher对象，若watcher对象不关闭，再运⾏时会报错。
    watcher.close()
