import matplotlib.pyplot as plt
import matplotlib.image as img

import skimage 
import cv2
import uuid

from skimage.transform import rescale, resize, downscale_local_mean

import os
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log
from mrcnn.config import Config

class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']


class TestConfig(Config):
    NAME = "test"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = len(class_names)
    
def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

def resize_padded(img, new_shape, fill_cval=None, order=1):
    import numpy as np
    from skimage.transform import resize
    if fill_cval is None:
        fill_cval = np.max(img)
    ratio = np.min([n / i for n, i in zip(new_shape, img.shape)])
    interm_shape = np.rint([s * ratio for s in img.shape]).astype(np.int)
    interm_img = resize(img, interm_shape, order=order, cval=fill_cval)

    new_img = np.empty(new_shape, dtype=interm_img.dtype)
    new_img.fill(fill_cval)

    pad = [(n - s) >> 1 for n, s in zip(new_shape, interm_shape)]
    new_img[[slice(p, -p, None) if 0 != p else slice(None, None, None) 
             for p in pad]] = interm_img

    return new_img

ROOT_DIR = os.path.abspath("./")
sys.path.append(ROOT_DIR)
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)
    
# Device to load the neural network on.
# Useful if you're training a model on the same 
# machine, in which case use CPU and leave the
# GPU for training.
DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0

# Inspect the model in training or inference modes
# values: 'inference' or 'training'
# TODO: code for 'training' test mode not ready yet
TEST_MODE = "inference"

# Create model in inference mode
with tf.device(DEVICE):
   # model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=TestConfig())
# Set weights file path
weights_path = COCO_MODEL_PATH
    
# Load weights
print("Loading weights ", weights_path)
model.load_weights(weights_path, by_name=True)

# when you want to exclude several layers in original mask-rcnn, use following command
# model.load_weights(weights_path, by_name=True, exclude=[ "mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])


image = skimage.io.imread('./samples/test.jpg')

fill_cval = np.max(image) # white
fill_cval = 0; # black

image_rescaled = rescale(image,1.6,anti_aliasing=False)
image_resized = resize_padded(image_rescaled, (1024, 1024, 3),  fill_cval=fill_cval)
results = model.detect([image_resized], verbose=1)
print(results[0]['class_ids'])
plt.imshow(results[0]['masks'][:,:,0])
plt.imshow(image)


