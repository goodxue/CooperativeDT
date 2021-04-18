import glob,pdb
import os
import sys
import json
import argparse
import logging
import time
import csv
import cv2
import pycocotools.coco as coco
import random
try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

from utils import *
import src.lib.utils.bbox_utils as bu

def get_vehicle_list(cam_gt,cam_trans):
    #vehicles_loc_list_1 = []
    vehicles_list_v = []
    id_list = []
    for ann_ind, txt in enumerate(cam_gt):
        tmp = txt[:-1].split(' ')
        #cat_id = cat_ids[tmp[0]]
        truncated = int(float(tmp[1]))
        occluded = int(tmp[2])
        alpha = float(tmp[3])
        bbox = [float(tmp[4]), float(tmp[5]), float(tmp[6]), float(tmp[7])]
        dim = [float(tmp[8]), float(tmp[9]), float(tmp[10])]
        location = [float(tmp[11]), float(tmp[12]), float(tmp[13])]
        rotation_y = float(tmp[14])
        #Id = int(tmp[15])
        score = float(tmp[15])

        cord_p = np.zeros((1,4))
        cord_p[0][0] = location[2]
        cord_p[0][1] = location[0]
        cord_p[0][2] = -location[1]
        cord_p[0][3] = 1

        rotation_y = rotation_y * 180 /np.pi
        cam_matrix = ClientSideBoundingBoxes.get_matrix(cam_trans)
        cam_to_world = np.dot(cam_matrix,np.transpose(cord_p))
        ry_cam2world = (rotation_y - 90 + cam_trans.rotation.yaw ) * np.pi / 180
        #vehicles_loc_list_1.append(vehicle_matrix)
        #vehicles_list_1.append({'bbox':bbox,'dim':dim,'location':location,'rotation':rotation_y,'id':Id})
        tv1 = CamVehicle(cam_to_world[1][0],-cam_to_world[2][0],cam_to_world[0][0],*dim,ry_cam2world,score=score)
        vehicles_list_v.append(tv1)
        #id_list.append(Id)
    
    return vehicles_list_v#,id_list

def get_ids(file_name,id_list=None):
    if id_list == None:
        id_list = []
    anns = open(file_name, 'r')
    for ann_ind, txt in enumerate(anns):
        tmp = txt[:-1].split(' ')
        #cat_id = cat_ids[tmp[0]]
        truncated = int(float(tmp[1]))
        occluded = int(tmp[2])
        alpha = float(tmp[3])
        bbox = [float(tmp[4]), float(tmp[5]), float(tmp[6]), float(tmp[7])]
        dim = [float(tmp[8]), float(tmp[9]), float(tmp[10])]
        location = [float(tmp[11]), float(tmp[12]), float(tmp[13])]
        rotation_y = float(tmp[14])
        car_id = int(tmp[15])
        if car_id not in id_list:
            id_list.append(car_id)
    return id_list
    


    


if __name__ == "__main__":
    FILTER_GLOBAL = True
    NUM_CAM = 1
    dataset_path = '/home/ubuntu/xwp/datasets/multi_view_dataset/new'
    cam_set = ['cam1']
    camset_path = [ os.path.join(dataset_path,cam_name) for cam_name in cam_set
        #'/home/ubuntu/xwp/datasets/multi_view_dataset/new/cam1/label_test'
        # '',
        # ''
    ]
    cam_path = [os.path.join(path,'label_test') for path in camset_path]
    cam_transform = {
        'cam1': Transform(location=Location(x=-98, y=-130, z=4),rotation=Rotation(pitch=0, yaw=20, roll=0))
        # 'cam2': Transform(location=Location(x=1, y=-1, z=4),rotation=Rotation(pitch=-90, yaw=-180, roll=0)),
        # 'cam3': Transform(location=Location(x=1, y=-1, z=4),rotation=Rotation(pitch=-90, yaw=-180, roll=0))
    }
    outdir_path = '/home/ubuntu/xwp/datasets/multi_view_dataset/new/fuse_cam1'

    anns = open('/home/ubuntu/xwp/datasets/multi_view_dataset/new/cam1/label_2/000901.txt','r')
    vehicles = get_vehicle_list(anns,cam_transform['cam1'])
    #box_main_list = [v.compute_box_3d() for v in vehicles]
    for car in vehicles:
        print(' {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {} {}'.format(car.height,car.width,car.length,car.z,car.x,-car.y,car.rotation_y,car.score))


