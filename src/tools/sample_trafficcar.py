#将cam1-cam7的图像标注放到一个文件夹下，并按顺序重命名
import pickle
import json
import numpy as np
import cv2
DATASET_PATH = '../../data/traffic_car/'
DEBUG = False
# VAL_PATH = DATA_PATH + 'training/label_val/'
import os
from tqdm.contrib import tzip
import shutil

TRAIN_NUM = 6000
VAL_NUM = 2000
TEST_NUM = 2000
CAM_SETS = ['cam1','cam2','cam3','cam4','cam5','cam6','cam7']

CAM_NUM = ''
OUT_PATH = os.path.join(DATASET_PATH, 'cam_sample')
OUT_IMG_PATH = os.path.join(OUT_PATH, 'image_2')
OUT_ANN_PATH = os.path.join(OUT_PATH, 'label_2')
OUT_CALIB_PATH = os.path.join(OUT_PATH, 'calib')
count = 0
for path in [OUT_IMG_PATH,OUT_CALIB_PATH,OUT_ANN_PATH]:
    if not os.path.exists(path):
        os.mkdir(OUT_PATH)
    
for cam in CAM_SETS:
    CAM_PATH = os.path.join(DATASET_PATH, cam)
    IMG_PATH = os.path.join(CAM_PATH, 'image_2')
    ANN_PATH = os.path.join(CAM_PATH, 'label_2')
    # CALIB_PATH = os.path.join(CAM_PATH, 'calib')
    # CALIB_FILE = os.path.join(CALIB_PATH,'000000.txt')
    # shutil.move(CALIB_FILE, OUT_CALIB_PATH)

    img_list = os.listdir(IMG_PATH)
    ann_list = os.listdir(ANN_PATH)
    print('moving {} imgs/anns from {} to {}'.format(len(img_list),IMG_PATH,OUT_PATH))
    print('with index start with *{}*'.format(count+1))

    for img, ann in tzip(img_list,ann_list):
        count += 1
        ann_ori_path = os.path.join(ANN_PATH, ann)
        ann_dst_path = os.path.join(OUT_IMG_PATH, '{:06d}.txt'.format(count))
        img_ori_path = os.path.join(IMG_PATH, img)
        img_dst_path = os.path.join(OUT_IMG_PATH, '{:06d}.png'.format(count))
        shutil.move(ann_ori_path, ann_dst_path)
        shutil.move(img_ori_path, img_dst_path)



    

    
