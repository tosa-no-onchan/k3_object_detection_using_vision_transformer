# -*- coding: utf-8 -*-
"""
Title: Object detection with Vision Transformers
Author: [Karan V. Dave](https://www.linkedin.com/in/karan-dave-811413164/)
Date created: 2022/03/27
Last modified: 2023/11/20
Description: A simple Keras implementation of object detection using Vision Transformers.
Accelerator: GPU
"""

"""
## Introduction

The article
[Vision Transformer (ViT)](https://arxiv.org/abs/2010.11929)
architecture by Alexey Dosovitskiy et al.
demonstrates that a pure transformer applied directly to sequences of image
patches can perform well on object detection tasks.

In this Keras example, we implement an object detection ViT
and we train it on the
[Caltech 101 dataset](http://www.vision.caltech.edu/datasets/)
to detect an airplane in the given image.
"""

"""
## Imports and setup
"""

import tensorflow as tf
try: [tf.config.experimental.set_memory_growth(gpu, True) for gpu in tf.config.experimental.list_physical_devices("GPU")]
except: pass

import os
import glob

os.environ["KERAS_BACKEND"] = "tensorflow"  # @param ["tensorflow", "jax", "torch"]

# add by nishi 2024.6.6
os.environ["MLIR_CRASH_REPRODUCER_DIRECTORY"] = "enable"

import numpy as np
import keras
from keras import layers
from keras import ops
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import scipy.io
import shutil

import sys

from object_detection_using_vision_transformer import *

import tools as tol



#-----------------
# main start
#-----------------
if __name__ == "__main__":

    resize_true=True

    use_load_weights=False
    #use_load_weights=True

    image_size = 224  # resize input images to this size
    patch_size = 32  # Size of the patches to be extracted from the input images

    # 4
    input_shape = (image_size, image_size, 3)  # input image shape
    learning_rate = 0.001
    weight_decay = 0.0001
    batch_size = 32
    num_epochs = 100
    num_patches = (image_size // patch_size) ** 2
    projection_dim = 64
    num_heads = 4
    # Size of the transformer layers
    transformer_units = [
        projection_dim * 2,
        projection_dim,
    ]
    transformer_layers = 4
    mlp_head_units = [2048, 1024, 512, 64, 32]  # Size of the dense layers


    history = []
    num_patches = (image_size // patch_size) ** 2

    if use_load_weights==True:
        print('load weights')
        vit_object_detector = create_vit_object_detector(
            input_shape,
            patch_size,
            num_patches,
            projection_dim,
            num_heads,
            transformer_units,
            transformer_layers,
            mlp_head_units,
        )

        vit_object_detector.load_weights('vit_object_detector.weights.h5')

    else:
        print('load model')
        # https://keras.io/api/models/model_saving_apis/model_saving_and_loading/
        # https://qiita.com/taiga518/items/b2154b661e7baf56031e
        vit_object_detector = keras.saving.load_model("vit_object_detector_best.keras",
                    custom_objects={'Patches':Patches,'PatchEncoder':PatchEncoder},
                    compile=True)
        
    #vit_object_detector.summary(expand_nested=True)      
    #sys.exit(0)

    path_image ='101_ObjectCategories/airplanes'

    list_of_files = glob.glob(path_image+'/*') # * means all if need specific format then *.csv

    from PIL import Image, ImageDraw

    for i in range(8):

        print('list_of_files[i]:',list_of_files[i])

        image = keras.utils.load_img(
            list_of_files[i],
        )
        (w, h) = image.size[:2]

        if resize_true==True:
            (fit_ratio,pad_top,pad_left,img_resized) = tol.letter_box_image(image, image_size, image_size, 255)
            #print('>>> ',fit_ratio,pad_top,pad_left)
            image_np=img_resized.astype(np.float32)
        else:
            # resize images
            # 単純に、リサイズしているので、よろしくない。
            # aspect を維持してリサイズして余白を入れる。
            # https://stackoverflow.com/questions/2232742/does-python-pil-resize-maintain-the-aspect-ratio
            image_in = image.resize((image_size, image_size))
            image_np = keras.utils.img_to_array(image_in)

        input_image = np.expand_dims(image_np, axis=0)
        preds = vit_object_detector.predict(input_image,verbose=0)[0]

        print('preds:',preds)
        # use_load_weights=True
        # preds: [ 30.240198  87.390045 191.9165   138.66414 ]
        # preds: [ 30.172216  87.19467  191.48807  138.35385 ]
        # preds: [ 30.183987  87.22705  191.55856  138.4041  ]
        #
        # use_load_weights=False
        # preds: [ 28.180092  80.771866 180.28598  129.76776 ]
        # preds: [ 28.0671   80.44585 179.55853 129.24286]
        # preds: [ 28.09199  80.5184  179.72025 129.35988]

        if resize_true==True:
            top_left_x = int((preds[0] - pad_left)/fit_ratio)
            top_left_y = int((preds[1] - pad_top)/fit_ratio)
            bottom_right_x = int((preds[2] - pad_left)/fit_ratio)
            bottom_right_y = int((preds[3] - pad_top)/fit_ratio)

        else:
            top_left_x, top_left_y = int(preds[0] * w), int(preds[1] * h)
            bottom_right_x, bottom_right_y = int(preds[2] * w), int(preds[3] * h)

        box_predicted = [top_left_x, top_left_y, bottom_right_x, bottom_right_y]

        rect_d = ImageDraw.Draw(image)
        rect_d.rectangle(box_predicted, outline=(0, 255, 0), width=2)

        #image.show()

        image_np = keras.utils.img_to_array(image)
        plt.imshow(image_np.astype("uint8"))
        plt.show()
