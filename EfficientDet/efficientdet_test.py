# Author: Zylo117

"""
Simple Inference Script of EfficientDet-Pytorch
"""
from operator import index
from textwrap import indent
import time
import torch
from torch.backends import cudnn
from matplotlib import colors

from backbone import EfficientDetBackbone
import cv2
import numpy as np

import os
import gc
import tqdm

from efficientdet.utils import BBoxTransform, ClipBoxes
import pandas as pd

from utils.utils import preprocess, invert_affine, postprocess, STANDARD_COLORS, standard_to_bgr, get_index_label, plot_one_box
gc.collect()
compound_coef = 6
force_input_size = None  # set None to use default size
# img_path = "dataset/train/0a0f8edf69eef0639bc2b30ce0cf09d5.jpg"

# replace this part with your project's anchor config
anchor_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]

threshold = 0.2
iou_threshold = 0.2

use_cuda = True
use_float16 = False
cudnn.fastest = True
cudnn.benchmark = True

obj_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', '', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
            'cow', 'elephant', 'bear', 'zebra', 'giraffe', '', 'backpack', 'umbrella', '', '', 'handbag', 'tie',
            'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', '', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
            'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
            'cake', 'chair', 'couch', 'potted plant', 'bed', '', 'dining table', '', '', 'toilet', '', 'tv',
            'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', '', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush']

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)

os.chdir("C:\github\petfinder\Yet-Another-EfficientDet-Pytorch")

color_list = standard_to_bgr(STANDARD_COLORS)
# tf bilinear interpolation is different from any other's, just make do
input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size

model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),
                             ratios=anchor_ratios, scales=anchor_scales)

                             
model.load_state_dict(torch.load(f'weights/efficientdet-d{compound_coef}.pth', map_location=device))
model.requires_grad_(False)
model.eval()

# img_path = "dataset/resize_train/0a0f8edf69eef0639bc2b30ce0cf09d5.jpg"
global df, image_one_page_pos, image_one_page_class
df = pd.read_csv("dataset/train.csv")

image_one_page_pos = [] # 이미지 한 장
image_one_page_class = []
# add col "point" : list of (xmin,ymin,xmax,ymax)
# add col "classes" : list of detected classes

# IMG_NAMES = os.listdir("dataset/resize_train")


RESIZE_PATH = "C:/github/petfinder/Yet-Another-EfficientDet-Pytorch/dataset/resize_train"
# RESIZE_PATH = "C:/github/petfinder/Yet-Another-EfficientDet-Pytorch/dataset/train"

# IMG_NAMES = os.listdir("dataset/train")
IMG_NAMES = os.listdir(RESIZE_PATH)
# ori_imgs, framed_imgs, framed_metas = preprocess(img_path, max_size=input_size)



framed_imgs_list = []
framed_metas_list = []
ori_imgs_list = []
out_list = []
x_list = []


FILE_NUMS = 240

# ~ 1000


# issue -> batch를 구현
FILE_NUM_RANGE = range(0,9912+1,240)


# IMG_FOLODER = "dataset/resize_train/"
# for name in tqdm.tqdm(IMG_NAMES[START_IDX: START_IDX + FILE_NUMS]):

box = []
images = []
FIRST_CHECK = True

for batch_num in tqdm.tqdm(FILE_NUM_RANGE):
    img = IMG_NAMES[batch_num: batch_num + FILE_NUMS]
    # images = [cv2.imread(RESIZE_PATH+file) for file in img]


    for batch_num in tqdm.tqdm(FILE_NUM_RANGE):
        img = IMG_NAMES[batch_num: batch_num + FILE_NUMS]
        for file in img:
            images.append(cv2.imread(RESIZE_PATH+"/"+file))

    # images = [cv2.imread(file) for file in glob.glob("path/to/files/*.png")]
    
    # img = IMG_FOLODER + name

    
    ori_imgs, framed_imgs, framed_metas = preprocess(img, max_size=input_size,img_folder=RESIZE_PATH)
    ori_imgs_list.append(ori_imgs)
    framed_imgs_list.append(framed_imgs)
    framed_metas_list.append(framed_metas)



# framed_imgs : resized 된 이미지

    if use_cuda:
        x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
    else:
        x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)            

    x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2) #batch, channel, width, height

    
    if FIRST_CHECK:
        FIRST_CHECK = False
        x_list = x
    else:
        x_list = torch.cat((x_list, x),dim=0)
        

print(x_list.shape)
print(x.shape)

# model start : inference

if use_cuda:
    model = model.cuda()
if use_float16:
    model = model.half()

with torch.no_grad():

    for x in x_list:
        features, regression, classification, anchors = model(x)

        regressBoxes = BBoxTransform()
        clipBoxes = ClipBoxes()

        out = postprocess(x,
                        anchors, regression, classification,
                        regressBoxes, clipBoxes,
                        threshold, iou_threshold)

        out_list.append(out)




def display(preds, imgs, imshow=True, imwrite=False):
    
    for i in range(len(imgs)):

        # i번째 사진에서
        if len(preds[i]['rois']) == 0:
            continue

        imgs[i] = imgs[i].copy()

        
        points = [] 
        classes = []
        # j개의 탐지된 objection에서
        for j in range(len(preds[i]['rois'])):
            x1, y1, x2, y2 = preds[i]['rois'][j].astype(np.int)
            obj = obj_list[preds[i]['class_ids'][j]]
            target_class = ["cat","dog"]
            
            print(x1,y1,x2,y2)
            if obj in target_class:
                score = float(preds[i]['scores'][j])
                plot_one_box(imgs[i], [x1, y1, x2, y2], label=obj,score=score,color=color_list[get_index_label(obj, obj_list)])
                # 탐지된 object의 좌표 획득
                points.append((x1, y1, x2, y2))
                classes.append(obj)

            else: pass
                # preds[i]['rois'][j] = (0,0,0,0)
                
        image_one_page_pos.append(points)
        image_one_page_class.append(classes)

        if imshow:
            cv2.imshow('img', imgs[i])
            cv2.waitKey(0)

        if imwrite:
            cv2.imwrite(f'test/img_inferred_d{compound_coef}_this_repo_{i}.jpg', imgs[i])


# for idx in tqdm(range(len(x_list))):
#     x = x_list[idx]
#     ori_imgs_list[idx]
#     framed_imgs = framed_imgs_list[idx]
#     framed_metas = framed_metas_list[idx]



# print("="*100)
# print(len(out_list))
# print("="*100)
for framed_metas, ori_imgs,out in zip(framed_metas_list, ori_imgs_list,out_list):
    out = invert_affine(framed_metas, out)
    display(out, ori_imgs, imshow=False, imwrite=True)


df["points"] = pd.Series(image_one_page_pos)
df["classes"] = pd.Series(image_one_page_class)


print('running speed test...')
with torch.no_grad():
    print('test1: model inferring and postprocessing')
    print('inferring image for 10 times...')
    t1 = time.time()
    for _ in range(10):
        _, regression, classification, anchors = model(x)

        out = postprocess(x,
                        anchors, regression, classification,
                        regressBoxes, clipBoxes,
                        threshold, iou_threshold)
        out = invert_affine(framed_metas, out)

    t2 = time.time()
    tact_time = (t2 - t1) / 10
    print(f'{tact_time} seconds, {1 / tact_time} FPS, @batch_size 1')



gc.collect()

df.to_csv("dataset/df_pos_class.csv",index=False)
    # uncomment this if you want a extreme fps test
    # print('test2: model inferring only')
    # print('inferring images for batch_size 32 for 10 times...')
    # t1 = time.time()
    # x = torch.cat([x] * 32, 0)
    # for _ in range(10):
    #     _, regression, classification, anchors = model(x)
    #
    # t2 = time.time()
    # tact_time = (t2 - t1) / 10
    # print(f'{tact_time} seconds, {32 / tact_time} FPS, @batch_size 32')
