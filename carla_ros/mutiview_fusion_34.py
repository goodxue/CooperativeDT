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
    for ann_ind, txt in enumerate(camgt):
        tmp = txt.split(' ')
        #cat_id = cat_ids[tmp[0]]
        truncated = int(float(tmp[1]))
        occluded = int(tmp[2])
        alpha = float(tmp[3])
        bbox = [float(tmp[4]), float(tmp[5]), float(tmp[6]), float(tmp[7])]
        dim = [float(tmp[8]), float(tmp[9]), float(tmp[10])]
        location = [float(tmp[11]), float(tmp[12]), float(tmp[13])]
        rotation_y = float(tmp[14])
        Id = int(tmp[15])

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
        tv1 = CamVehicle(cam_to_world[1][0],-cam_to_world[2][0],cam_to_world[0][0],*dim,ry_cam2world,Id)
        vehicles_list_v.append(tv1)
    
    return vehicles_list_v
    


if __name__ == "__main__":
    NUM_CAM = 3
    cam_path = [
        '',
        '',
        ''
    ]
    cam_transform = {
        'cam1': Transform(location=Location(x=1, y=-1, z=4),rotation=Rotation(pitch=-90, yaw=-180, roll=0)),
        'cam2': Transform(location=Location(x=1, y=-1, z=4),rotation=Rotation(pitch=-90, yaw=-180, roll=0)),
        'cam3': Transform(location=Location(x=1, y=-1, z=4),rotation=Rotation(pitch=-90, yaw=-180, roll=0))
    }
    output_path = ''

    if len(cam_path) != NUM_CAM:
        raise RuntimeError('expect {} cam path but got {}'.format(NUM_CAM,len(cam_path)))

    detect_main_list = os.listdir(cam_path[0])
    to_fuse_detectlist = []
    for i in range(1,NUM_CAM):
        to_fuse_detectlist.append(os.listdir(cam_path[i]))
    
    for pred in detect_main_list:
        anns = open(pred, 'r')
        f = open(os.path.join(output_path,pred),'w')
        vehicles = get_vehicel_list(anns)
        box_main_list = [v.compute_box_3d for v in vehicles]

        for i in range(1,NUM_CAM):
            if pred not in to_fuse_detectlist[i-1]:
                raise RuntimeError('{} not in {}'.format(pred,cam_path[i]))
            anns2 = open(os.path.join(cam_path[i],pred))
            vehicles2 = get_vehicle_list(anns2)
            box_to_fuse_list = [v.compute_box_3d for v in vehicels2]
            box_main_list = bu.box3d_matching(box_main_list,box_to_fuse_list,iou_threshold=0.1,fusion=bu.box_mean_fusion)

        for car in box_main_list:
            f.write('{} 0.0 0 0 0 0 0 0'.format('Car'))
            f.write(' {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {} {}'.format(car.x,car.y,car.z,car.height,car.width,car.length,car.rotation_y,car.id))
            f.write('\n')
        f.close()


            

    