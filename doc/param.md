## 服务输入输出参数说明

- post 输入参数

| **参数名**         | **说明**                                                     | **类型**           | **默认值**                         |
| ------------------ | ------------------------------------------------------------ | ------------------ | ---------------------------------- |
| task_id            | 任务ID                                                       | string             | 必须                               |
| prompt             | 用户输入的正向提示词                                         | string             | 必须                               |
| func_name          | post的功能部署服务为base时支持传入t2i/i2i/inpaint 进行功能转换 | string             | t2i                                |
| steps              | 用户输入的步数                                               | int                | 50                                 |
| cfg_scale          | guidance_scale                                               | int                | 7                                  |
| denoising_strength | 与原图的合并比例【只在图生图中有效】                         | float              | 0.55                               |
| width              | 生成图片宽度                                                 | int                | 512                                |
| height             | 生成图片高度                                                 | int                | 512                                |
| negative_prompt    | 用户输入的负向提示词                                         | string             | “”                                 |
| image_num          | 用户输入的图片数量                                           | int                | 1                                  |
| resize_mode        | 调整生成图片缩放方式 0 拉伸 1 裁剪 2 填充                    | int                | 0                                  |
| image_link         | 用户输入的图片url地址                                        | string             | 图生图，inpaint controlnet必须提供 |
| mask_link          | 用户输入的mask url 地址                                      | string             | inpaint 必须提供                   |
| image_base64       | 用户输入的图片 base64格式                                    | base64             | 与image_link二选一                 |
| mask_base64        | 用户输入的mask base64格式                                    | base64             | 与mask_link二选一                  |
| use_base64         | 是否返回imagebase64的图像结果                                | bool               | False                              |
| lora_attn          | lora使用的比例当使用多LoRA融合时支持列表的输入               | floatList[float]   | 0.75                               |
| lora_path          | 需要更新的lora模型在oss挂载路径的相对位置使用多LoRA融合时支持列表的输入 | stringList[string] | 无                                 |
| controlnet_path    | 需要更新的controlnet模型在oss挂载路径的相对位置（huggingface 上可下载的safetensors/bin 文件） | string             | 无                                 |
| process_func       | 图像预处理方式，用于生成controlnet的控制图像                 | string             | 具体支持的列表见下表               |
| expand      | outpaint时 各个方向需要填充的像素数[left,right,up,down]      | list   | 无   |
| expand_type | 原始图像的扩充方式（影响outpaint的出图效果）copy（复制边缘）reflect（镜像翻转边缘） | string | copy |
| save_dir           | 传入文件夹的名字文件将存放在部署挂载的result路径中的save_dir文件夹中 | string             | result                             |

 - controlnet支持列表（仅支持下表中的8种格式的端到端处理，对于其他controlnet 您可自行处理后用于控制生成的图像）

| process_func | 实现功能 | controlnet参考下载地址                                       |
| ------------ | -------- | ------------------------------------------------------------ |
| canny        | 边缘检测 | [边缘检测](https://converter-offline-installer.oss-cn-hangzhou.aliyuncs.com/zxy/diffusers/dbt/demo_controlnet/new_controlnet/models--lllyasviel--sd-controlnet-canny/diffusion_pytorch_model.safetensors) |
| depth        | 深度检测 | [深度检测](https://converter-offline-installer.oss-cn-hangzhou.aliyuncs.com/zxy/diffusers/dbt/demo_controlnet/new_controlnet/models--lllyasviel--sd-controlnet-depth/diffusion_pytorch_model.safetensors) |
| hed          | 线稿上色 | [线稿上色](https://converter-offline-installer.oss-cn-hangzhou.aliyuncs.com/zxy/diffusers/dbt/demo_controlnet/new_controlnet/models--lllyasviel--sd-controlnet-hed/diffusion_pytorch_model.safetensors) |
| mlsd         | 线段识别 | [线段识别](https://converter-offline-installer.oss-cn-hangzhou.aliyuncs.com/zxy/diffusers/dbt/demo_controlnet/new_controlnet/models--lllyasviel--sd-controlnet-mlsd/diffusion_pytorch_model.safetensors) |
| normal       | 物体识别 | [物体识别](https://converter-offline-installer.oss-cn-hangzhou.aliyuncs.com/zxy/diffusers/dbt/demo_controlnet/new_controlnet/models--fusing--stable-diffusion-v1-5-controlnet-normal/diffusion_pytorch_model.safetensors) |
| openpose     | 姿态识别 | [姿态识别](https://converter-offline-installer.oss-cn-hangzhou.aliyuncs.com/zxy/diffusers/dbt/demo_controlnet/new_controlnet/models--lllyasviel--sd-controlnet-openpose/diffusion_pytorch_model.safetensors) |
| scribble     | 线稿上色 | [线稿上色](https://converter-offline-installer.oss-cn-hangzhou.aliyuncs.com/zxy/diffusers/dbt/demo_controlnet/new_controlnet/models--lllyasviel--sd-controlnet-scribble/diffusion_pytorch_model.safetensors) |
| seg          | 语义分割 | [语义分割](https://converter-offline-installer.oss-cn-hangzhou.aliyuncs.com/zxy/diffusers/dbt/demo_controlnet/new_controlnet/models--lllyasviel--sd-controlnet-seg/diffusion_pytorch_model.safetensors) |

- post 输出参数

| **参数名**    | **说明**                                                     | **类型**   |
| ------------- | ------------------------------------------------------------ | ---------- |
| image_url     | 生成图像的公网可访问链接 【在开放acl权限后有效】             | list       |
| images_base64 | 生成的图像列表 base64格式（use_base64开启时会返回）          | list       |
| oss_url       | 生成图像的oss地址                                            | list       |
| success       | 是否成功 0-失败 1-成功                                       | int        |
| seed          | 生成图像的种子                                               | string     |
| task_id       | 任务ID                                                       | string     |
| error_msg     | 错误的原因【只在success=0时返回错误】                        | string     |
| use_blade     | 是否使用了blade 进行推理优化【blade模型成功优化后，会在第一次推理时默认使用】 | bool       |
| is_nsfw       | 用于表示生成图片是否不合法【True为黑图】                     | list[bool] |
