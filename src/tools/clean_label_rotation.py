import json
import numpy as np
import cv2

def read_clib(calib_path): #change 3x3 to 3x4
  f = open(calib_path, 'r')
  for i, line in enumerate(f):
    if i == 0:
      calib = np.array(line[:-1].split(' ')[1:], dtype=np.float32)
      calib = calib.reshape(3, 3)
      calib = np.concatenate([calib,np.zeros((3,1),dtype=np.float32)],axis=1)
      return calib

cats = ['Car','DontCare']
cat_ids = {cat: i + 1 for i, cat in enumerate(cats)}

DATASET_PATH = '/home/ubuntu/xwp/datasets/multi_view_dataset/new/'
DEBUG = False
# VAL_PATH = DATA_PATH + 'training/label_val/'
import os
SPLITS = ['3dop', 'subcnn'] 
import _init_paths
from utils.ddd_utils import compute_box_3d, project_to_image, alpha2rot_y
from utils.ddd_utils import draw_box_3d, unproject_2d_to_3d, draw_box_2d

ann_dir = DATASET_PATH + 'global_label_2/'
output_dir = DATASET_PATH + 'global_label_new/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
ann_list = os.listdir(ann_dir)
ann_list.sort(key=lambda x:int(x[:-4]))
#calib_dir = DATASET_PATH + 'calib/'
# splits = ['trainval', 'test']
#calib_path = calib_dir + '000000.txt'
#calib = read_clib(calib_path)

border_x_min = 0
border_x_max = 168
border_y_min = 0
border_y_max = 540

for label_file in ann_list:
    ann_path = ann_dir + '{}'.format(label_file)
    out_path = output_dir + '{}'.format(label_file)
    f = open(out_path, 'w')
    anns = open(ann_path, 'r')
    for ann_ind, txt in enumerate(anns):
        tmp = txt[:-1].split(' ')
        cat_id = cat_ids[tmp[0]]
        truncated = int(float(tmp[1]))
        occluded = int(tmp[2])
        alpha = float(tmp[3])
        bbox = [float(tmp[4]), float(tmp[5]), float(tmp[6]), float(tmp[7])]
        dim = [float(tmp[8]), float(tmp[9]), float(tmp[10])]
        location = [float(tmp[11]), float(tmp[12]), float(tmp[13])]
        rotation_y = float(tmp[14])
        car_id = int(tmp[15])
        rotation = rotation_y * np.pi / 180
        #box_3d = compute_box_3d(dim, location, rotation_y)
        #box_2d = project_to_image(box_3d, calib)
        #bbox = (np.min(box_2d[:,0]), np.min(box_2d[:,1]), np.max(box_2d[:,0]), np.max(box_2d[:,1]))

        txt="{} {} {} {} {} {} {} {} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {} {}\n".format('Car', truncated, occluded, alpha, bbox[0], bbox[1], bbox[2], bbox[3], dim[0], dim[1],
                            dim[2],location[0], location[1], location[2],  rotation,car_id)
        f.write(txt)
        #f.write('\n')
    f.close()

print('finish!')