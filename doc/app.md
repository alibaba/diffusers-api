## EAS 自定义processor开发

以Python SDK为例，本文档介绍 基于PAI-EAS的自定义processor开发，其核心在于维护主文件[app.py](../diffusers/app.py)。

### 模版介绍

您需要继承PAI-EAS提供的基类BaseProcessor，实现**initialize()**和**process()**函数。其**process()**函数的输入输出均为BYTES类型，输出参数分别为**response_data**和**status_code**，正常请求**status_code**可以返回**0**或**200**。

| 函数                                                     | 功能描述                                                     | 参数描述                                                     |
| -------------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| init(worker_threads=5, worker_processes=1,endpoint=None) | Processor构建函数。                                          | **worker_threads**：Worker线程数，默认值为5。**worker_processes**：进程数，默认值为1。如果**worker_processes**为1，则表示单进程多线程模式。如果**worker_processes**大于1，则**worker_threads**只负责读取数据，请求由多进程并发处理，每个进程均会执行**initialize()**函数。**endpoint**：服务监听的Endpoint，通过该参数可以指定服务监听的地址和端口，例如**endpoint=’0.0.0.0:8079’**。 |
| initialize()                                             | Processor初始化函数。服务启动时，进行模型加载等初始化工作。  | 无参数。                                                     |
| process(data)                                            | 请求处理函数。每个请求会将Request Body作为参数传递给**process()**进行处理，并将函数返回值返回至客户端。 | **data**为Request Body，类型为BYTES。返回值也为BYTES类型。   |
| run()                                                    | 启动服务。                                                   | 无参数。                                                     |



### 二次开发

我们在[app.py](../diffusers/app.py)已经实现了部分基于diffusers api的功能实现，本节通过流程图的显示，对核心代码流程进行展示，方便您进行二次开发。

#### initialize()
<img src="https://pai-vision-exp.oss-cn-zhangjiakou.aliyuncs.com/zxy/diffusers/assets/init.jpg" width="80%">

#### process(data)

![img](https://pai-vision-exp.oss-cn-zhangjiakou.aliyuncs.com/zxy/diffusers/assets/app.jpg)
