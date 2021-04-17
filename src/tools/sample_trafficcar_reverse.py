#将cam1-cam7的图像标注放到一个文件夹下，并按顺序重命名
import pickle
import json
import numpy as np
import cv2
DATASET_PATH = '/home/ubuntu/xwp/datasets/multi_view_dataset/new'
DEBUG = False
# VAL_PATH = DATA_PATH + 'training/label_val/'
import os
from tqdm.contrib import tzip
import shutil

TRAIN_NUM = 800
VAL_NUM = 100
TEST_NUM = 100
CAM_SETS = ['cam{}'.format(str(i)) for i in range(1,35)]

CAM_NUM = ''
OUT_PATH = os.path.join(DATASET_PATH, 'cam_sample')
OUT_ANN_PATH = os.path.join(OUT_PATH, 'label_2')

count = 0
pointer = 0
ann_list = os.listdir(OUT_ANN_PATH)
ann_list.sort(key=lambda x:int(x[:-4]))

for ann in ann_list:
    cam_name = CAM_SETS[pointer]
    count += 1
    if count > 1000:
        count = 1
        pointer += 1
    ann_ori_path = os.path.join(OUT_ANN_PATH, ann)
    ann_dst_path = os.path.join(DATASET_PATH,cam_name,'label_2' ,'{:06d}.txt'.format(count))
    shutil.copyfile(ann_ori_path, ann_dst_path)
    
print('finished!')





    

    
