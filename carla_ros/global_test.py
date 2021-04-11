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


if __name__ == "__main__":
    #获取相机坐标信息
    sensors_definition_file = '/home/ubuntu/xwp/CenterNet/carla_ros/dataset.json'
    if not os.path.exists(sensors_definition_file):
        raise RuntimeError(
            "Could not read sensor-definition from {}".format(sensors_definition_file))

    with open(sensors_definition_file) as handle:
        json_actors = json.loads(handle.read())
    
    global_sensors = []
    for actor in json_actors["objects"]:
        actor_type = actor["type"].split('.')[0]
        if actor_type == "sensor":
            global_sensors.append(actor)
        else:
            continue
    #print(global_sensors)
    cam_loc_dict = {}
    cam_point_dict = {}
    
    for sensor_spec in global_sensors:
        try:
            sensor_names = []
            sensor_type = str(sensor_spec.pop("type"))
            sensor_id = str(sensor_spec.pop("id"))

            sensor_name = sensor_type + "/" + sensor_id
            if sensor_name in sensor_names:
                raise NameError
            sensor_names.append(sensor_name)
            spawn_point = sensor_spec.pop("spawn_point")
            #print(spawn_point)
            point = Transform(location=Location(x=spawn_point.pop("x"), y=-spawn_point.pop("y"), z=spawn_point.pop("z")),
                rotation=Rotation(pitch=-spawn_point.pop("pitch", 0.0), yaw=-spawn_point.pop("yaw", 0.0), roll=spawn_point.pop("roll", 0.0)))
            cam_loc_dict[sensor_id] = ClientSideBoundingBoxes.get_matrix(point)
            cam_point_dict[sensor_id] = point
        
        except RuntimeError as e:
            raise RuntimeError("Setting up global sensors failed: {}".format(e))

    gt_file = open('/home/ubuntu/xwp/datasets/multi_view_dataset/new/global_label_2/000001.txt','r')
    vehicles_loc_list = []
    vehicles_list = []
    for ann_ind, txt in enumerate(gt_file):
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
        #rotation_y = rotation_y * 180 /np.pi
        loc = Location(x=location[0],y=location[1],z=location[2])
        rot = Rotation(yaw = rotation_y)
        trans = Transform(location=loc,rotation=rot)
        vehicle_matrix = ClientSideBoundingBoxes.get_matrix(trans)
        vehicles_loc_list.append(vehicle_matrix)
        vehicles_list.append({'bbox':bbox,'dim':dim,'location':location,'rotation':rotation_y,'id':Id})

    translated_vehicles = []
    cam1_world_matrix = cam_loc_dict['cam1']
    cam1_world_invmatrix = np.linalg.inv(cam_loc_dict['cam1'])
    for i, (vehicle_matrix, vehicle) in enumerate(zip(vehicles_loc_list,vehicles_list)):
        cord_p = np.zeros((1,4))
        cord_p[0][0] = vehicle['location'][0] 
        cord_p[0][1] = vehicle['location'][1] 
        cord_p[0][2] = vehicle['location'][2]
        cord_p[0][3] = 1
        
        p_in_cam1 = np.dot(cam1_world_invmatrix, np.transpose(cord_p))
        ry_world2cam1 = (vehicle['rotation'] - cam_point_dict['cam1'].rotation.yaw+90) * np.pi / 180
        tv = CamVehicle(p_in_cam1[1][0],-p_in_cam1[2][0],p_in_cam1[0][0],*vehicle['dim'],ry_world2cam1)
        translated_vehicles.append(tv)

    calib = read_clib('/home/ubuntu/xwp/datasets/multi_view_dataset/346/calib/000000.txt')
    image = cv2.imread('/home/ubuntu/xwp/datasets/multi_view_dataset/new/cam1/image_2/000001.png')
    box3d_list = []
    for vehicle in translated_vehicles:
        box3d_list.append(vehicle.compute_box_3d())
    
    for box3d in box3d_list:
        #print(box_3d)
        box_2d = project_to_image(box3d, calib)
        #print(box_2d)
        image = draw_box_3d(image,box_2d)

    cv2.imshow('img',image)
    cv2.waitKey()