import os
from typing import Tuple, Union

import cv2
import numpy as np
from PIL import Image, ImageDraw

import torch
from controlnet_aux import HEDdetector, MLSDdetector, OpenposeDetector
# need to be import or an malloc error will be occured by controlnet_aux
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
from transformers import (AutoImageProcessor, UperNetForSemanticSegmentation,
                          pipeline)


def canny(image: Image.Image,
          pretrain_dir: str,
          low_threshold: int = 100,
          high_threshold: int = 200) -> Image.Image:
    """
    Apply the Canny edge detection algorithm to the image.

    Args:
        image (Image.Image): The input image.
        low_threshold (int): The lower threshold for edge detection (default: 100).
        high_threshold (int): The higher threshold for edge detection (default: 200).

    Returns:
        Image.Image: The processed image with detected edges.
    """
    image = np.array(image)
    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    image = Image.fromarray(image)
    return image


def depth(image: Image.Image, pretrain_dir: str) -> Image.Image:
    """
    Estimate the depth map of the image using a pre-trained depth estimation model.

    Args:
        image (Image.Image): The input image.
        pretrain_dir (str): The directory containing the pre-trained models.

    Returns:
        Image.Image: The estimated depth map of the image.
    """
    depth_estimator = pipeline('depth-estimation',
                               model=os.path.join(pretrain_dir,
                                                  'models--Intel--dpt-large'))
    image = depth_estimator(image)['depth']
    image = np.array(image)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    image = Image.fromarray(image)
    return image


def hed(image: Image.Image, pretrain_dir: str) -> Image.Image:
    """
    Apply the Holistically-Nested Edge Detection (HED) algorithm to the image.

    Args:
        image (Image.Image): The input image.
        pretrain_dir (str): The directory containing the pre-trained models.

    Returns:
        Image.Image: The processed image with detected edges.
    """
    hed = HEDdetector.from_pretrained(
        os.path.join(pretrain_dir, 'models--lllyasviel--ControlNet'))
    image = hed(image)
    return image


def mlsd(image: Image.Image, pretrain_dir: str) -> Image.Image:
    """
    Apply MLSD (Multi-Line Segment Detection) model to the input image.

    Args:
        image (Image.Image): The input image.
        pretrain_dir (str): The directory path where the pre-trained model is located.

    Returns:
        Image.Image: The processed image.

    """
    mlsd = MLSDdetector.from_pretrained(
        os.path.join(pretrain_dir, 'models--lllyasviel--ControlNet'))
    image = mlsd(image)
    return image


def normal(image: Image.Image,
           pretrain_dir: str,
           bg_threshold: float = 0.4) -> Image.Image:
    """
    Perform normal estimation on the input image.

    Args:
        image (Image.Image): The input image.
        pretrain_dir (str): The directory path where the pre-trained model is located.
        bg_threshold (float, optional): Background depth threshold. Default is 0.4.

    Returns:
        Image.Image: The image with normal estimation.

    """
    depth_estimator = pipeline('depth-estimation',
                               model=os.path.join(
                                   pretrain_dir,
                                   'models--Intel--dpt-hybrid-midas'))
    image = depth_estimator(image)['predicted_depth'][0]
    image = image.numpy()

    image_depth = image.copy()
    image_depth -= np.min(image_depth)
    image_depth /= np.max(image_depth)

    x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    x[image_depth < bg_threshold] = 0

    y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    y[image_depth < bg_threshold] = 0

    z = np.ones_like(x) * np.pi * 2.0

    image = np.stack([x, y, z], axis=2)
    image /= np.sum(image**2.0, axis=2, keepdims=True)**0.5
    image = (image * 127.5 + 127.5).clip(0, 255).astype(np.uint8)
    image = Image.fromarray(image)

    return image


def openpose(image: Image.Image, pretrain_dir: str) -> Image.Image:
    """
    Apply OpenPose model to the input image.

    Args:
        image (Image.Image): The input image.
        pretrain_dir (str): The directory path where the pre-trained model is located.

    Returns:
        Image.Image: The processed image.

    """
    openpose = OpenposeDetector.from_pretrained(
        os.path.join(pretrain_dir, 'models--lllyasviel--ControlNet'))
    image = openpose(image)
    return image


def scribble(image: Image.Image, pretrain_dir: str) -> Image.Image:
    """
    Apply scribble-based HED (Holistically-Nested Edge Detection) model to the input image.

    Args:
        image (Image.Image): The input image.
        pretrain_dir (str): The directory path where the pre-trained model is located.

    Returns:
        Image.Image: The processed image.

    """
    hed = HEDdetector.from_pretrained(pretrained_model_or_path=os.path.join(
        pretrain_dir, 'models--lllyasviel--ControlNet'))
    image = hed(image, scribble=True)
    return image


def seg(image: Image.Image, pretrain_dir: str) -> Image.Image:
    """
    Apply semantic segmentation to the input image.

    Args:
        image (Image.Image): The input image.
        pretrain_dir (str): The directory path where the pre-trained models are located.

    Returns:
        Image.Image: The processed image.

    """

    palette = np.asarray([
        [0, 0, 0],
        [120, 120, 120],
        [180, 120, 120],
        [6, 230, 230],
        [80, 50, 50],
        [4, 200, 3],
        [120, 120, 80],
        [140, 140, 140],
        [204, 5, 255],
        [230, 230, 230],
        [4, 250, 7],
        [224, 5, 255],
        [235, 255, 7],
        [150, 5, 61],
        [120, 120, 70],
        [8, 255, 51],
        [255, 6, 82],
        [143, 255, 140],
        [204, 255, 4],
        [255, 51, 7],
        [204, 70, 3],
        [0, 102, 200],
        [61, 230, 250],
        [255, 6, 51],
        [11, 102, 255],
        [255, 7, 71],
        [255, 9, 224],
        [9, 7, 230],
        [220, 220, 220],
        [255, 9, 92],
        [112, 9, 255],
        [8, 255, 214],
        [7, 255, 224],
        [255, 184, 6],
        [10, 255, 71],
        [255, 41, 10],
        [7, 255, 255],
        [224, 255, 8],
        [102, 8, 255],
        [255, 61, 6],
        [255, 194, 7],
        [255, 122, 8],
        [0, 255, 20],
        [255, 8, 41],
        [255, 5, 153],
        [6, 51, 255],
        [235, 12, 255],
        [160, 150, 20],
        [0, 163, 255],
        [140, 140, 140],
        [250, 10, 15],
        [20, 255, 0],
        [31, 255, 0],
        [255, 31, 0],
        [255, 224, 0],
        [153, 255, 0],
        [0, 0, 255],
        [255, 71, 0],
        [0, 235, 255],
        [0, 173, 255],
        [31, 0, 255],
        [11, 200, 200],
        [255, 82, 0],
        [0, 255, 245],
        [0, 61, 255],
        [0, 255, 112],
        [0, 255, 133],
        [255, 0, 0],
        [255, 163, 0],
        [255, 102, 0],
        [194, 255, 0],
        [0, 143, 255],
        [51, 255, 0],
        [0, 82, 255],
        [0, 255, 41],
        [0, 255, 173],
        [10, 0, 255],
        [173, 255, 0],
        [0, 255, 153],
        [255, 92, 0],
        [255, 0, 255],
        [255, 0, 245],
        [255, 0, 102],
        [255, 173, 0],
        [255, 0, 20],
        [255, 184, 184],
        [0, 31, 255],
        [0, 255, 61],
        [0, 71, 255],
        [255, 0, 204],
        [0, 255, 194],
        [0, 255, 82],
        [0, 10, 255],
        [0, 112, 255],
        [51, 0, 255],
        [0, 194, 255],
        [0, 122, 255],
        [0, 255, 163],
        [255, 153, 0],
        [0, 255, 10],
        [255, 112, 0],
        [143, 255, 0],
        [82, 0, 255],
        [163, 255, 0],
        [255, 235, 0],
        [8, 184, 170],
        [133, 0, 255],
        [0, 255, 92],
        [184, 0, 255],
        [255, 0, 31],
        [0, 184, 255],
        [0, 214, 255],
        [255, 0, 112],
        [92, 255, 0],
        [0, 224, 255],
        [112, 224, 255],
        [70, 184, 160],
        [163, 0, 255],
        [153, 0, 255],
        [71, 255, 0],
        [255, 0, 163],
        [255, 204, 0],
        [255, 0, 143],
        [0, 255, 235],
        [133, 255, 0],
        [255, 0, 235],
        [245, 0, 255],
        [255, 0, 122],
        [255, 245, 0],
        [10, 190, 212],
        [214, 255, 0],
        [0, 204, 255],
        [20, 0, 255],
        [255, 255, 0],
        [0, 153, 255],
        [0, 41, 255],
        [0, 255, 204],
        [41, 0, 255],
        [41, 255, 0],
        [173, 0, 255],
        [0, 245, 255],
        [71, 0, 255],
        [122, 0, 255],
        [0, 255, 184],
        [0, 92, 255],
        [184, 255, 0],
        [0, 133, 255],
        [255, 214, 0],
        [25, 194, 194],
        [102, 255, 0],
        [92, 0, 255],
    ])

    image_processor = AutoImageProcessor.from_pretrained(
        os.path.join(pretrain_dir,
                     'models--openmmlab--upernet-convnext-small'))
    image_segmentor = UperNetForSemanticSegmentation.from_pretrained(
        os.path.join(pretrain_dir,
                     'models--openmmlab--upernet-convnext-small'))

    pixel_values = image_processor(image, return_tensors='pt').pixel_values

    with torch.no_grad():
        outputs = image_segmentor(pixel_values)

    seg = image_processor.post_process_semantic_segmentation(
        outputs, target_sizes=[image.size[::-1]])[0]

    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)

    for label, color in enumerate(palette):
        color_seg[seg == label, :] = color

    color_seg = color_seg.astype(np.uint8)

    image = Image.fromarray(color_seg)

    return image


def preprocess_control(image: Image.Image, process_func: str,
                       pretrain_dir: str) -> Union[str, Image.Image]:
    """
    Apply the specified image processing function to the input image for controlnet.

    Args:
        image (Image.Image): The input image to be processed.
        process_func (str): The name of the processing function to be applied.
        pretrain_dir (str): The directory containing the pre-trained models.

    Returns:
        Union[str, Image.Image]: The processed image if successful, or an error message as a string if the specified process_func is not supported.
    """
    process_func_dict = {
        'canny': canny,
        'depth': depth,
        'hed': hed,
        'mlsd': mlsd,
        'normal': normal,
        'openpose': openpose,
        'scribble': scribble,
        'seg': seg
    }

    if process_func not in process_func_dict:
        return 'We only support process functions: {}. But got {}.'.format(
            list(process_func_dict.keys()), process_func)

    process_func = process_func_dict[process_func]
    processed_image = process_func(image, pretrain_dir)
    return processed_image


def transform_image(image: Image.Image,
                    width: int,
                    height: int,
                    mode: int = 0) -> Image.Image:
    """
    Transform the input image to the specified width and height using the specified mode.

    Args:
        image (PIL Image object): The image that needs to be transformed.
        width (int): The width of the output image.
        height (int): The height of the output image.
        mode (int, optional): Specifies the mode of image transformation.
            0 - Stretch 拉伸, 1 - Crop 裁剪, 2 - Padding 填充. Defaults to 0. align with webui

    Returns:
        PIL Image object: The transformed image.
    """

    if mode == 0:  # Stretch
        image = image.resize((width, height))
    elif mode == 1:  # Crop
        aspect_ratio = float(image.size[0]) / float(image.size[1])
        new_aspect_ratio = float(width) / float(height)

        if aspect_ratio > new_aspect_ratio:
            # Crop the width
            new_width = int(float(height) * aspect_ratio)
            left = int((new_width - width) / 2)
            right = new_width - left
            image = image.resize((new_width, height))
            image = image.crop((left, 0, left + width, height))
        else:
            # Crop the height
            new_height = int(float(width) / aspect_ratio)
            up = int((new_height - height) / 2)
            down = new_height - up
            image = image.resize((width, new_height))
            image = image.crop((0, up, width, up + height))

    elif mode == 2:  # Padding
        new_image = Image.new('RGB', (width, height), (255, 255, 255))
        new_image.paste(image, ((width - image.size[0]) // 2,
                                (height - image.size[1]) // 2))
        image = new_image

    return image


def generate_mask_and_img_expand(
        img: Image.Image,
        expand: Tuple[int, int, int, int],
        expand_type: str = 'copy') -> Tuple[Image.Image, Image.Image]:
    """
    Generate a mask and an expanded image based on the given image and expand parameters.

    Args:
        img (Image.Image): The original image.
        expand (Tuple[int, int, int, int]): The expansion values for left, right, up, and down directions.
        expand_type (str, optional): The type of expansion ('copy' or 'reflect'). Defaults to 'copy'.

    Returns:
        Tuple[Image.Image, Image.Image]: The expanded image and the corresponding mask.
    """

    left, right, up, down = expand

    width, height = img.size
    new_width, new_height = width + left + right, height + up + down

    # ----------- 1. Create mask where the image is black and the expanded region is white -----------
    mask = Image.new('L', (new_width, new_height), 0)
    draw = ImageDraw.Draw(mask)
    # Add white edge
    color = 255
    draw.rectangle((0, 0, new_width, up), fill=color)  # up
    draw.rectangle((0, new_height - down, new_width, new_height),
                   fill=color)  # down
    draw.rectangle((0, 0, left, new_height), fill=color)  # left
    draw.rectangle((new_width - right, 0, new_width, new_height),
                   fill=color)  # right

    # ----------- 2. Expand the image by a copy or reflection operation -----------
    # simply use the filled pixel can not generate meaningful image in unified pipeline
    # img_expand = Image.new('RGB', (new_width, new_height), (255, 255, 255))
    # img_expand.paste(img, (left, up))

    # Convert the image to a NumPy array
    image_array = np.array(img)

    # new img
    expanded_image_array = np.zeros((new_height, new_width, 3), dtype=np.uint8)

    # copy ori img
    expanded_image_array[up:up + height, left:left + width, :] = image_array

    if expand_type == 'reflect':
        # Reflect the boundary pixels to the new boundaries
        expanded_image_array[:up, left:left + width, :] = np.flipud(
            expanded_image_array[up:2 * up, left:left + width, :])  # up
        expanded_image_array[up + height:, left:left + width, :] = np.flipud(
            expanded_image_array[up + height - 2:up + height - 2 - down:-1,
                                 left:left + width, :])  # down
        expanded_image_array[:, :left, :] = np.fliplr(
            expanded_image_array[:, left:2 * left, :])  # left
        expanded_image_array[:, left + width:, :] = np.fliplr(
            expanded_image_array[:, left + width - 2:left + width - 2 -
                                 right:-1, :])  # right

    else:
        # Copy the boundary pixels to the new boundaries
        expanded_image_array[:up, left:left +
                             width, :] = image_array[0:1, :, :]  # up
        expanded_image_array[up + height:, left:left +
                             width, :] = image_array[height -
                                                     1:height, :, :]  # down
        expanded_image_array[:, :left, :] = expanded_image_array[:, left:left +
                                                                 1, :]  # left
        expanded_image_array[:, left +
                             width:, :] = expanded_image_array[:, left +
                                                               width - 1:left +
                                                               width, :]  # right

    # Create a new image from the expanded image array
    img_expand = Image.fromarray(expanded_image_array)

    return img_expand, mask
