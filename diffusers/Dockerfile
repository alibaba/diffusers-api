FROM bladedisc/bladedisc:latest-devel-cu113
ENV BLADE_GEMM_TUNE_JIT=1 DISC_ENABLE_PREDEFINED_PDL=true DISC_ENABLE_PACK_QKV=true
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple

RUN pip install https://pai-blade.oss-cn-zhangjiakou.aliyuncs.com/pytorch/wheels/torch-1.12.0%2Bcu113-cp38-cp38-linux_x86_64.whl

RUN pip install https://pai-blade.oss-cn-zhangjiakou.aliyuncs.com/temp/xformers-0.0.17%2B658ebab.d20230327-cp38-cp38-linux_x86_64.whl &&\
    pip install transformers &&\
    pip install opencv-python-headless  &&\
    pip install diffusers==0.15.0 &&\
    pip install -U http://eas-data.oss-cn-shanghai.aliyuncs.com/sdk/allspark-0.15-py2.py3-none-any.whl &&\
    pip install https://pai-vision-exp.oss-cn-zhangjiakou.aliyuncs.com/zxy/diffusers/torch_blade-0.0.1%2B1.12.0.cu113-cp38-cp38-linux_x86_64.whl &&\
    pip install safetensors &&\
    pip install modelscope &&\
    pip install subword_nmt &&\
    pip install jieba &&\
    pip install sacremoses &&\
    pip install tensorflow &&\
    pip install omegaconf
RUN pip install scikit-image
RUN pip install https://pai-vision-exp.oss-cn-zhangjiakou.aliyuncs.com/zxy/diffusers/controlnet_aux-0.0.3-py3-none-any.whl --no-deps
RUN pip install torchvision==0.13.0
RUN pip install timm
RUN pip install mediapipe
RUN pip cache purge

RUN apt-get install wget
RUN mkdir /home/pai/

ADD ./app.py /home/pai/app.py
ADD ./utils /home/pai/utils
ADD ./ev_error.py /home/pai/ev_error.py
ADD ./lpw_stable_diffusion.py /home/pai/lpw_stable_diffusion.py

RUN wget https://converter-offline-installer.oss-cn-hangzhou.aliyuncs.com/zxy/diffusers/image/optimized_model.tar.gz \
    && tar -xvf optimized_model.tar.gz \
    && mv optimized_model /home/pai/optimized_model \
    && rm optimized_model.tar.gz

RUN wget https://converter-offline-installer.oss-cn-hangzhou.aliyuncs.com/zxy/diffusers/image/pretrained_models.tar.gz \
    && tar -xvf pretrained_models.tar.gz \
    && mv pretrained_models /home/pai/pretrained_models \
    && rm pretrained_models.tar.gz
