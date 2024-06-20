# -*- coding: utf-8 -*-
"""
雑草の Object Detection
k3_object_detection_using_vision_transformer/tarin_zasou.py

"""

"""
## Imports and setup
"""

import tensorflow as tf
try: [tf.config.experimental.set_memory_growth(gpu, True) for gpu in tf.config.experimental.list_physical_devices("GPU")]
except: pass

import os

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

import tools as tol

from PIL import Image, ImageDraw

from object_detection_using_vision_transformer import *

import tools as tol



#-----------------
# main start
#-----------------
if __name__ == "__main__":

    resize_true=True

    # Path to images and annotations
    path_images = "../annotation/data-backup/雑草/"
    path_annot = "../annotation/annotation/雑草/"

    # list of paths to images and annotations
    image_paths = [
        f for f in os.listdir(path_images) if os.path.isfile(os.path.join(path_images, f))
    ]
    annot_paths = [
        f for f in os.listdir(path_annot) if os.path.isfile(os.path.join(path_annot, f))
    ]

    print('len(image_paths):',len(image_paths))
    print('len(annot_paths):',len(annot_paths))

    image_paths.sort()
    annot_paths.sort()

    image_size = 224  # resize input images to this size

    images, targets = [], []

    # loop over the annotations and images, preprocess them and store in lists
    for i in range(0, len(annot_paths)):
        # Access bounding box coordinates
        a_path=path_annot + annot_paths[i]
        #print('a_path:',a_path)
        f=open(a_path)
        a_text=f.read()
        f.close()
        # YOLO annotaion
        # https://qiita.com/yarakigit/items/4d4044bc2740cecba92a
        (class_id,yolo_x, yolo_y, yolo_w, yolo_h)=a_text.split(' ')
        yolo_x=float(yolo_x)
        yolo_y=float(yolo_y)
        yolo_w=float(yolo_w)
        yolo_h=float(yolo_h)

        print('yolo:',yolo_x,',',yolo_y,',',yolo_w,',',yolo_h)

        g_path=path_images + image_paths[i]
        #print('g_path:',g_path)

        image = keras.utils.load_img(
            path_images + image_paths[i],
        )
        (w, h) = image.size[:2]

        print('w:',w,' h:',h)

        top_left_x = int((yolo_x - yolo_w*0.5) * float(w))
        top_left_y = int((yolo_y - yolo_h*0.5) * float(h))
        bottom_right_x = int((yolo_x + yolo_w*0.5) * float(w))
        bottom_right_y = int((yolo_y + yolo_h*0.5) * float(h))

        print('top_left_x:',top_left_x, ' top_left_y:',top_left_y,' bottom_right_x:', bottom_right_x,' bottom_right_y:', bottom_right_y)

        if False:
            pil_image=image
            box_predicted = [top_left_x, top_left_y, bottom_right_x, bottom_right_y]

            rect_d = ImageDraw.Draw(pil_image)
            rect_d.rectangle(box_predicted, outline=(0, 255, 0), width=2)
            pil_image.show()


        if resize_true==True:
            (fit_ratio,pad_top,pad_left,img_resized) = tol.letter_box_image(image, image_size, image_size, 255)
            #print('>>> ',fit_ratio,pad_top,pad_left)
            image=img_resized
            images.append(image.astype(np.float32))
        else:
            # resize images
            # 単純に、リサイズしているので、よろしくない。
            # aspect を維持してリサイズして余白を入れる。
            # https://stackoverflow.com/questions/2232742/does-python-pil-resize-maintain-the-aspect-ratio
            image = image.resize((image_size, image_size))
            # convert image to array and append to list
            images.append(keras.utils.img_to_array(image))
            #x=images[0]
            #print('x.dtype:',x.dtype)
            #sys.exit(0)


        if False:
            image.show()

            plt.imshow(img_resized.astype("uint8"))
            plt.show()
            sys.exit(0)

        if resize_true==True:
            targets.append(
                (
                    float(top_left_x)*fit_ratio + pad_left,
                    float(top_left_y)*fit_ratio + pad_top,
                    float(bottom_right_x)*fit_ratio + pad_left,
                    float(bottom_right_y)*fit_ratio + pad_top,
                )
            )
            if False:
                tl_x = int(float(top_left_x)*fit_ratio + pad_left)
                tl_y = int(float(top_left_y)*fit_ratio + pad_top)
                br_x = int(float(bottom_right_x)*fit_ratio + pad_left)
                br_y = int(float(bottom_right_y)*fit_ratio + pad_top)
                pil_image = Image.fromarray(img_resized)

                box_predicted = [tl_x, tl_y, br_x, br_y]

                rect_d = ImageDraw.Draw(pil_image)
                rect_d.rectangle(box_predicted, outline=(0, 255, 0), width=2)
                pil_image.show()
                sys.exit(0)

        else:
            targets.append(
                (
                    float(top_left_x) / w,
                    float(top_left_y) / h,
                    float(bottom_right_x) / w,
                    float(bottom_right_y) / h,
                )
            )


    # Convert the list to numpy array, split to train and test dataset
    (x_train), (y_train) = (
        np.asarray(images[: int(len(images) * 0.8)]),
        np.asarray(targets[: int(len(targets) * 0.8)]),
    )
    (x_test), (y_test) = (
        np.asarray(images[int(len(images) * 0.8) :]),
        np.asarray(targets[int(len(targets) * 0.8) :]),
    )    

    #print('x_test[:3]',x_test[:3])
    #sys.exit()

    # 3
    """
    ## Display patches for an input image
    """

    patch_size = 32  # Size of the patches to be extracted from the input images

    plt.figure(figsize=(4, 4))
    plt.imshow(x_train[0].astype("uint8"))
    plt.axis("off")

    patches = Patches(patch_size)(np.expand_dims(x_train[0], axis=0))
    print(f"Image size: {image_size} X {image_size}")
    print(f"Patch size: {patch_size} X {patch_size}")
    print(f"{patches.shape[1]} patches per image \n{patches.shape[-1]} elements per patch")


    n = int(np.sqrt(patches.shape[1]))
    plt.figure(figsize=(4, 4))
    for i, patch in enumerate(patches[0]):
        ax = plt.subplot(n, n, i + 1)
        patch_img = ops.reshape(patch, (patch_size, patch_size, 3))
        plt.imshow(ops.convert_to_numpy(patch_img).astype("uint8"))
        plt.axis("off")    

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

    #vit_object_detector.summary(expand_nested=True)      
    #sys.exit(0)

    # Train model
    history = run_experiment(
        vit_object_detector, learning_rate, weight_decay, batch_size, num_epochs,x_train,y_train
    )

    def plot_history(item):
        plt.plot(history.history[item], label=item)
        plt.plot(history.history["val_" + item], label="val_" + item)
        plt.xlabel("Epochs")
        plt.ylabel(item)
        plt.title("Train and Validation {} Over Epochs".format(item), fontsize=14)
        plt.legend()
        plt.grid()
        plt.show()


    plot_history("loss")

    #5

    """
    ## Evaluate the model
    """

    import matplotlib.patches as patches

    # Saves the model in current path
    vit_object_detector.save("vit_object_detector.keras")


    if resize_true==True:
        print('Cannot Evaluate the model, Then exit!! Try predict.py')
        sys.exit(0)
