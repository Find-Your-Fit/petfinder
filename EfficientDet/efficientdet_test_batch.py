# Author: Zylo117

"""
Simple Inference Script of EfficientDet-Pytorch
"""

# RuntimeError: cuDNN error: CUDNN_STATUS_ALLOC_FAILED
# PYTORCH_CUDA_ALLOC_CONF

# solution
# model.to(device)
# batch size 1000 to 32

# RuntimeError: CUDA out of memory. Tried to allocate 1.17 GiB (GPU 0; 8.00 GiB total capacity; 3.44 GiB already allocated; 1.14 GiB free; 4.86 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF

# solution

# batch size 32 -> 2
# input data -> cpu()
# del input data
# gc.collect()
# torch.cuda.empty_cache()

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
import gc

from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess, STANDARD_COLORS, standard_to_bgr, get_index_label, plot_one_box


compound_coef = 6
force_input_size = None 
device = 'cuda' if torch.cuda.is_available() else 'cpu'


ROOT_PATH = "C:\github\petfinder\EfficientDet"
os.chdir(ROOT_PATH)
img_path = os.listdir("dataset/train")

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
input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size

BATCH_SIZE = 1

model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),
                            ratios=anchor_ratios, scales=anchor_scales)
model.load_state_dict(torch.load(f'weights/efficientdet-d{compound_coef}.pth', map_location=device))
model.requires_grad_(False)
model.eval()

def display(preds, imshow=True, imwrite=False): #imgs

    col_class, col_points, col_scores = [], [], []
    for i in range(len(preds)):
        # if len(preds[i]['rois']) == 0:
        #     continue
        

        row_class,row_points,row_scores = [], [],[]

        for j in range(len(preds[i]['rois'])):
            x1, y1, x2, y2 = preds[i]['rois'][j].astype(np.int64)
            obj = obj_list[preds[i]['class_ids'][j]]
            score = float(preds[i]['scores'][j])


            if obj in ["cat","dog"]:
                row_class.append(obj)
                row_points.append((x1, y1, x2, y2))
                row_scores.append(score)
        
        
        col_class.append(row_class)
        col_points.append(row_points)
        col_scores.append(row_scores)

    return col_points,col_class,col_scores


model = model.to(device)
# if use_cuda:
#     model = model.cuda()
# if use_float16:
#     model = model.half()

train_csv = pd.read_csv("dataset/train.csv")

for START in tqdm.tqdm(range(0,9912,BATCH_SIZE)):
    
    gc.collect()
    torch.cuda.empty_cache()

    END = START+BATCH_SIZE
    if END > 9913:
        END = 9913
    
        
    framed_imgs, framed_metas, img_ids = preprocess(img_path,START,END,"dataset/train/", max_size=input_size)

    if use_cuda:
        x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
    else:
        x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

    x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)

    with torch.no_grad():
        features, regression, classification, anchors = model(x)
 

        regressBoxes = BBoxTransform()
        clipBoxes = ClipBoxes()

        out = postprocess(x.cpu().numpy(),
                        anchors, regression, classification,
                        regressBoxes, clipBoxes,
                        threshold, iou_threshold)

    del x
    

    out = invert_affine(framed_metas, out)

    series_points, series_class,series_scores = display(out, imshow=False, imwrite=False) #ori_imgse

    save_file = (img_ids,series_class,series_points,series_scores)


    # with open("dataset/objmeta.csv",'w') as file:
    #     file.write(img_ids)
    #     file.write(series_points)
    #     file.write(series_class)

    # import pickle
    # with open('data_dict.pkl','wb') as f:
    #     pickle.dump(save_file,f)







    for i,id in enumerate(img_ids):
        id = id.split(".")[0]
        
        train_csv.loc[train_csv["Id"] == id]["box"] = [np.nan if not series_points[i] else series_points[i]]
        train_csv.loc[train_csv["Id"] == id]["classes"] = [np.nan if not series_class[i] else series_class[i]]
        train_csv.loc[train_csv["Id"] == id]["scores"] = [np.nan if not series_scores[i] else series_scores[i]]
    
train_csv.to_csv("train_obj.csv",index=False)
