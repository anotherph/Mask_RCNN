# .py version of inspect_model.ipynb
# deepfashion2 dataset 에 대한 inference 가 가능한 코드 

import os
import sys
import random
import math
import re
import time
import numpy as np
#import tensorflow as tf
import tensorflow.compat.v1 as tf
import matplotlib
import skimage 
#import matplotlib.pyplot as plt
#import matplotlib.patches as patches

from matplotlib import pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log
from mrcnn.config import Config
from samples.coco import coco

from PIL import Image

from keras.backend import manual_variable_initialization 
manual_variable_initialization(True)

%matplotlib inline

class_names = ['short_sleeved_shirt', 'long_sleeved_shirt', 'short_sleeved_outwear', 'long_sleeved_outwear', 'vest', 'sling', 
               'shorts', 'trousers', 'skirt', 'short_sleeved_dress', 'long_sleeved_dress',
               'vest_dress', 'sling_dress']

class TestConfig(Config):
     NAME = "test"
     GPU_COUNT = 1
     IMAGES_PER_GPU = 1
     NUM_CLASSES = 1 + 13

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_deepfashion2_0071.h5")
# Download COCO trained weights from Releases if needed
#if not os.path.exists(COCO_MODEL_PATH):
#    utils.download_trained_weights(COCO_MODEL_PATH)

# Path to Shapes trained weights
#SHAPES_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_shapes.h5")

#config=coco.CocoConfig()
#COCO_DIR ='/media/jekim/Samsung_T5/COCO2017'

# class InferenceConfig(config.__class__):
#     # Run detection on one image at a time
#     GPU_COUNT = 1
#     IMAGES_PER_GPU = 1

# config = InferenceConfig()
# config.display()

DEVICE = "/cpu:0"
TEST_MODE = "inference" 

def get_ax(rows=1, cols=1, size=16):
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

# if config.NAME == 'shapes':
#     dataset = shapes.ShapesDataset()
#     dataset.load_shapes(500, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
# elif config.NAME == "coco":
#     dataset = coco.CocoDataset()
#     dataset.load_coco(COCO_DIR, "val")

# Must call before using the dataset
#dataset.prepare()

#print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))

with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=TestConfig())
    
# if config.NAME == "shapes":
#     weights_path = SHAPES_MODEL_PATH
# elif config.NAME == "coco":
#     weights_path = COCO_MODEL_PATH
    
#print("Loading weights ", weights_path)
print("Loading weights ", COCO_MODEL_PATH)

#model.load_weights(weights_path, by_name=True)
tf.keras.Model.load_weights(model.keras_model, COCO_MODEL_PATH, by_name=True)

#image_id = random.choice(dataset.image_ids)
#image_id = 300
#image, image_meta, gt_class_id, gt_bbox, gt_mask =\
#    modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
#info = dataset.image_info[image_id]
#print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id, 
#                                       dataset.image_reference(image_id)))
    
#image = skimage.io.imread('/home/jekim/workspace/Deepfashion2_Training/Deepfashion2_Training/dataset_temp/test/image/000138.jpg')
#image = skimage.io.imread('/home/jekim/workspace/Deepfashion2_Training/Deepfashion2_Training/test2.jpg')
image = skimage.io.imread('/home/jekim/workspace/Deep-Fashion-Analysis-ECCV2018/pics/test_16.jpg')

# Run object detection
results = model.detect([image], verbose=1)

# Display results
ax = get_ax(1)
r = results[0]
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                            class_names, r['scores'], ax=ax,
                            title="Predictions")

class_result=np.array(r['class_ids']-1)
for i in class_result:
    print(class_names[i])

# plt.imshow(r['masks'][:,:,0])

#log("gt_class_id", gt_class_id)
#log("gt_bbox", gt_bbox)
#log("gt_mask", gt_mask)

# ######

# # Generate RPN trainig targets
# # target_rpn_match is 1 for positive anchors, -1 for negative anchors
# # and 0 for neutral anchors.
# target_rpn_match, target_rpn_bbox = modellib.build_rpn_targets(
#     image.shape, model.anchors, gt_class_id, gt_bbox, model.config)
# log("target_rpn_match", target_rpn_match)
# log("target_rpn_bbox", target_rpn_bbox)

# positive_anchor_ix = np.where(target_rpn_match[:] == 1)[0]
# negative_anchor_ix = np.where(target_rpn_match[:] == -1)[0]
# neutral_anchor_ix = np.where(target_rpn_match[:] == 0)[0]
# positive_anchors = model.anchors[positive_anchor_ix]
# negative_anchors = model.anchors[negative_anchor_ix]
# neutral_anchors = model.anchors[neutral_anchor_ix]
# log("positive_anchors", positive_anchors)
# log("negative_anchors", negative_anchors)
# log("neutral anchors", neutral_anchors)

# # Apply refinement deltas to positive anchors
# refined_anchors = utils.apply_box_deltas(
#     positive_anchors,
#     target_rpn_bbox[:positive_anchors.shape[0]] * model.config.RPN_BBOX_STD_DEV)
# log("refined_anchors", refined_anchors, )

# # Display positive anchors before refinement (dotted) and
# # after refinement (solid).
# visualize.draw_boxes(image, boxes=positive_anchors, refined_boxes=refined_anchors, ax=get_ax())

# # Get input and output to classifier and mask heads.
# mrcnn = model.run_graph([image], [
#     ("proposals", model.keras_model.get_layer("ROI").output),
#     ("probs", model.keras_model.get_layer("mrcnn_class").output),
#     ("deltas", model.keras_model.get_layer("mrcnn_bbox").output),
#     ("masks", model.keras_model.get_layer("mrcnn_mask").output),
#     ("detections", model.keras_model.get_layer("mrcnn_detection").output),
# ])

# class_names = ['short_sleeved_shirt', 'long_sleeved_shirt', 'short_sleeved_outwear', 'long_sleeved_outwear', 'vest', 'sling', 
#                'shorts', 'trousers', 'skirt', 'short_sleeved_dress', 'long_sleeved_dress',
#                'vest_dress', 'sling_dress']

# class TestConfig(Config):
#      NAME = "test"
#      GPU_COUNT = 1
#      IMAGES_PER_GPU = 1
#      NUM_CLASSES = 1 + 13

# #rcnn = MaskRCNN(mode='inference', model_dir='/home/link/Desktop/final_100', config=TestConfig())
# rcnn = MaskRCNN(mode='inference', model_dir='./logs/deepfashion220220315T1402',config=TestConfig())
# #rcnn.load_weights('./logs/deepfashion220220315T1402/mask_rcnn_deepfashion2_0015.h5', by_name=True)
# #rcnn.load_weights('mask_rcnn_deepfashion2_0100.h5', by_name=True)
# rcnn.load_weights('mask_rcnn_coco.h5', by_name=True) # It doesn't work: the size of layer (mrcnn_bbox_fc_1) is mismatched
# img = skimage.io.imread('./dataset_temp/test/image/000100.jpg')
# results = rcnn.detect([img], verbose=1)
# r = results[0]
# visualize.display_instances(img, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
# mask = r['masks']
# mask1 = mask.astype(int)
# img[:,:,2] = img[:,:,1] * mask1[:,:,2]
# skimage.io.imsave("detecteded.jpg",img[:,:,2])



