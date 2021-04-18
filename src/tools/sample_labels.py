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

CAM_SETS = ['cam{}'.format(str(i)) for i in range(1,35)]

CAM_NUM = ''
ANN_PATH = os.path.join(DATASET_PATH, 'results')
ann_list = os.listdir(ANN_PATH)
ann_list.sort(key=lambda x:int(x[:-6]))
source_index = (901,1000)
target_index = (901,1000)

for cam in CAM_SETS:
    OUT_PATH = os.path.join(DATASET_PATH,cam )
    OUT_ANN_PATH = os.path.join(OUT_PATH, 'label_test')
    for path in [OUT_PATH,OUT_ANN_PATH]:
        if not os.path.exists(path):
            os.mkdir(path)
        
    #CAM_PATH = os.path.join(DATASET_PATH, 'cam_sample')

    # CALIB_PATH = os.path.join(CAM_PATH, 'calib')
    # CALIB_FILE = os.path.join(CALIB_PATH,'000000.txt')
    # shutil.move(CALIB_FILE, OUT_CALIB_PATH)

count = 0
for ann in ann_list:
    border = False
    if int(ann[3:-4]) != 0:
        cam_num = int(ann[:3])+1
    else:
        border = True
        cam_num = int(ann[:3])
    ann_ori_path = os.path.join(ANN_PATH, ann)
    ann_dst_path = os.path.join(DATASET_PATH,'cam{}'.format(cam_num),'label_test', '001000.txt'  if border else '{:06d}.txt'.format(int(ann[3:-4])))
    shutil.copyfile(ann_ori_path, ann_dst_path)
    count += 1
        # else:
        #     if source_index[1] - source_index[0] > count:
        #         raise RuntimeError('number transform error! check result!')

print('finished!')



    

    
