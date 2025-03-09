#!/usr/bin/env Python
# coding=utf-8
import os
import json
import matplotlib.pyplot as plt
import numpy as np
import cv2
from tqdm import tqdm
from pycocotools.coco import COCO



def show_skelenton(img,kpts,color = (255,128,128),thr = 0.01):
    kpts = np.array(kpts).reshape(-1, 3)
    skelenton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9], [8, 10],
                    [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
    for sk in skelenton:

        pos1 = (int(kpts[sk[0] - 1, 0]), int(kpts[sk[0] - 1, 1]))
        pos2 = (int(kpts[sk[1] - 1, 0]), int(kpts[sk[1] - 1, 1]))
        if pos1[0]>0 and pos1[1] >0 and pos2[0] >0 and pos2[1] > 0 and kpts[sk[0] - 1, 2] > thr and kpts[sk[1] - 1, 0] > thr:
            cv2.line(img, pos1, pos2, color, 2, 8)
    return img


RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
CYAN = (255, 255, 0)
YELLOW = (0, 255, 255)
ORANGE = (0, 165, 255)
PURPLE = (255, 0, 255)

color_list = [RED, GREEN, BLUE, CYAN, YELLOW, ORANGE, PURPLE]

coco_json_path = '/home/Mos/OtherDisks/TiPlus7100/Datasets/COCO/coco-pose/annotations/person_keypoints_val2017.json'
coco_img_path = '/home/Mos/OtherDisks/TiPlus7100/Datasets/COCO/val2017'
save_plt_path = '/home/Mos/OtherDisks/TiPlus7100/Datasets/COCO/val2017_visualization'

if not os.path.exists(save_plt_path):
    os.makedirs(save_plt_path)

coco = COCO(coco_json_path)

img_ids = coco.getImgIds()
for img_idx in tqdm(img_ids):
    img_name = str(img_idx).zfill(12) + '.jpg'
    img_path = coco_img_path+'/'+ img_name
    img = cv2.imread(img_path)
    annIds = coco.getAnnIds(imgIds=img_idx, iscrowd=False)
    objs = coco.loadAnns(annIds)
    for person_id,obj in enumerate(objs):
        keypoints = obj['keypoints']
        color = color_list[person_id % len(color_list)]
        img = show_skelenton(img,keypoints,color=color)
    save_path = save_plt_path+'/'+img_name
    cv2.imwrite(save_path,img)



