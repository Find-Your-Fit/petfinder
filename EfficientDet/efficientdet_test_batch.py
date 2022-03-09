# Author: Zylo117

"""
Simple Inference Script of EfficientDet-Pytorch
"""

# RuntimeError: cuDNN error: CUDNN_STATUS_ALLOC_FAILED
import time
import torch
from torch.backends import cudnn
from matplotlib import colors

from backbone import EfficientDetBackbone
import cv2
import numpy as np
import os
import pandas as pd
import tqdm
import csv 

from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess, STANDARD_COLORS, standard_to_bgr, get_index_label, plot_one_box


compound_coef = 6
force_input_size = None  # set None to use default size
device = 'cuda' if torch.cuda.is_available() else 'cpu'


ROOT_PATH = "C:\JupyterNotebook\GitHub\petfinder\EfficientDet"
os.chdir(ROOT_PATH)
# img_path = 'test/img.png'
# img_path = os.listdir("dataset/train")
img_path = os.listdir("dataset/resize_padding_train")

# replace this part with your project's anchor config``
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


color_list = standard_to_bgr(STANDARD_COLORS)
# tf bilinear interpolation is different from any other's, just make do
input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size

BATCH_SIZE = 4
for START in range(0,9912,BATCH_SIZE):
    # import gc
    # gc.collect()
    # torch.cuda.empty_cache()
    print("batch start")
    print(START,"/", int(9912/BATCH_SIZE))
    END = START+BATCH_SIZE
    if END > 9913:
        END = 9913
    
        
    framed_imgs, framed_metas = preprocess(img_path,START,END,"dataset/train/", max_size=input_size)

    if use_cuda:
        x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
    else:
        x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

    x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)

    model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),
                                ratios=anchor_ratios, scales=anchor_scales)
    model.load_state_dict(torch.load(f'weights/efficientdet-d{compound_coef}.pth', map_location=device))
    model.requires_grad_(False)
    model.eval()

    if use_cuda:
        model = model.cuda()
    if use_float16:
        model = model.half()

    with torch.no_grad():
        features, regression, classification, anchors = model(x)

        regressBoxes = BBoxTransform()
        clipBoxes = ClipBoxes()

        out = postprocess(x,
                        anchors, regression, classification,
                        regressBoxes, clipBoxes,
                        threshold, iou_threshold)

    def display(preds, imshow=True, imwrite=False): #imgs

        col_class, col_points = [], []
        for i in tqdm.tqdm(range(len(preds))):
            if len(preds[i]['rois']) == 0:
                continue
            

            row_class,row_points = [], []
            # imgs[i] = imgs[i].copy()

            for j in range(len(preds[i]['rois'])):
                x1, y1, x2, y2 = preds[i]['rois'][j].astype(np.int)
                obj = obj_list[preds[i]['class_ids'][j]]
                score = float(preds[i]['scores'][j])
                # plot_one_box(imgs[i], [x1, y1, x2, y2], label=obj,score=score,color=color_list[get_index_label(obj, obj_list)])

                if obj in ["cat","dog"]:
                    row_class.append(obj)
                    row_points.append((x1, y1, x2, y2))


            if imshow:
                pass
                # cv2.imshow('img', imgs[i])
                # cv2.waitKey(0)

            if imwrite:
                cv2.imwrite(f'test/img_inferred_d{compound_coef}_this_repo_{i}.jpg', imgs[i])
        col_class.append(row_class)
        col_points.append(row_points)

        return col_points,col_class

    out = invert_affine(framed_metas, out)

    series_points, series_class = display(out, imshow=False, imwrite=False) #ori_imgs


    
    # list_name = ["가방", "스타크래프트", "핸드폰", "손전등", "스위치"] 
    with open(f"dataset/points.csv", 'w') as file: 
        writer = csv.writer(file) 
        writer.writerow(series_points)

    # with open(f"dataset/classes_backup_{START}.csv", 'w') as file: 
    with open(f"dataset/classes.csv", 'w') as file: 
        writer = csv.writer(file) 
        writer.writerow(series_class)
    print(START,END)
    print("batch END")

train_csv = pd.read_csv("dataset/train.csv")
train_csv["box"] = pd.Series(series_points)
train_csv["class"] = pd.Series(series_class)

# print('running speed test...')
# with torch.no_grad():
#     print('test1: model inferring and postprocessing')
#     print('inferring image for 10 times...')
#     t1 = time.time()
#     for _ in range(10):
#         _, regression, classification, anchors = model(x)

#         out = postprocess(x,
#                           anchors, regression, classification,
#                           regressBoxes, clipBoxes,
#                           threshold, iou_threshold)
#         out = invert_affine(framed_metas, out)

#     t2 = time.time()
#     tact_time = (t2 - t1) / 10
#     print(f'{tact_time} seconds, {1 / tact_time} FPS, @batch_size 1')

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