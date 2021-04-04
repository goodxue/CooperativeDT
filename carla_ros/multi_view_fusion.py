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

argparser = argparse.ArgumentParser(
        description=__doc__)
argparser.add_argument(
    '-MV', '--mainview',
    default='cam1',
    type=str,
    help='choose a cam as the fusion center.')

args = argparser.parse_args()


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


        #获取GT文件信息
    camgt_list = []
    camgt_list.append(open('/home/ubuntu/xwp/datasets/multi_view_dataset/346/label_2/000128.txt','r'))
    camgt_list.append(open('/home/ubuntu/xwp/datasets/multi_view_dataset/347/label_2/000128.txt','r'))
    vehicles_cam_list = []

    camgt = camgt_list[0]
    vehicles_loc_list_1 = []
    vehicles_list_1 = []
    for ann_ind, txt in enumerate(camgt):
        tmp = txt.split(' ')
        #cat_id = cat_ids[tmp[0]]
        truncated = int(float(tmp[1]))
        occluded = int(tmp[2])
        alpha = float(tmp[3])
        bbox = [float(tmp[4]), float(tmp[5]), float(tmp[6]), float(tmp[7])]
        dim = [float(tmp[8]), float(tmp[9]), float(tmp[10])]
        location = [float(tmp[11]), float(tmp[12]), float(tmp[13])]
        rotation_y = float(tmp[14]) + random.uniform(-0.5,0.5)
        Id = int(tmp[15])
        rotation_y = rotation_y * 180 /np.pi
        loc = Location(x=location[0],y=location[1],z=location[2])
        rot = Rotation(yaw = rotation_y)
        trans = Transform(location=loc,rotation=rot)
        vehicle_matrix = ClientSideBoundingBoxes.get_matrix(trans)
        vehicles_loc_list_1.append(vehicle_matrix)
        vehicles_list_1.append({'bbox':bbox,'dim':dim,'location':location,'rotation':rotation_y,'id':Id})
    
    # for camgt in camgt_list:
    camgt = camgt_list[1]
    vehicles_loc_list = []
    vehicles_list = []
    vehicle_list_v = []
    for ann_ind, txt in enumerate(camgt):
        tmp = txt.split(' ')
        #cat_id = cat_ids[tmp[0]]
        truncated = int(float(tmp[1]))
        occluded = int(tmp[2])
        alpha = float(tmp[3])
        bbox = [float(tmp[4]), float(tmp[5]), float(tmp[6]), float(tmp[7])]
        dim = [float(tmp[8]), float(tmp[9]), float(tmp[10])]
        location = [float(tmp[11]), float(tmp[12]), float(tmp[13])]
        rotation_y1 = float(tmp[14]) + random.uniform(-0.5,0.5)
        Id = int(tmp[15])
        rotation_y = rotation_y1 * 180 /np.pi
        loc = Location(x=location[0],y=location[1],z=location[2])
        rot = Rotation(yaw = rotation_y)
        trans = Transform(location=loc,rotation=rot)
        vehicle_matrix = ClientSideBoundingBoxes.get_matrix(trans)
        vehicles_loc_list.append(vehicle_matrix)
        vehicles_list.append({'bbox':bbox,'dim':dim,'location':location,'rotation':rotation_y,'id':Id})
        tv1 = CamVehicle(*location,*dim,rotation_y1)
        vehicle_list_v.append(tv1)

    translated_vehicles = []
    cam1_vehicles = []
    translated_points = []
    translated_rotation = []
    cam1_world_invmatrix = np.linalg.inv(cam_loc_dict['cam1'])
    cam2_world_matrix = cam_loc_dict['cam2']
    cam1_world_matrix = cam_loc_dict['cam1']
    # print("cam1: ",cam_loc_dict['cam1'])
    # print("cam2: ",cam2_world_matrix)
    for v1 in vehicles_list_1:
        # if v1['id'] == vehicle['id']:
        #     cam1_v = v1
        #     cord_v1[0][0] = v1['location'][2] + random.random()
        #     cord_v1[0][1] = v1['location'][0] + random.random()
        #     cord_v1[0][2] = -v1['location'][1] + random.random()
        #     cord_v1[0][3] = 1
        tv1 = CamVehicle(v1['location'][0],v1['location'][1],v1['location'][2],*v1['dim'],v1['rotation']*np.pi/180)
        cam1_vehicles.append(tv1)
    print(len(cam1_vehicles))
    for i, (vehicle_matrix, vehicle) in enumerate(zip(vehicles_loc_list,vehicles_list)):
        cord_p = np.zeros((1,4))
        cord_p[0][0] = vehicle['location'][2] + random.uniform(-0.4,0.4)
        cord_p[0][1] = vehicle['location'][0] + random.uniform(-0.4,0.4)
        cord_p[0][2] = -vehicle['location'][1] + random.uniform(-0.4,0.4)
        cord_p[0][3] = 1
        # bb_cord = compute_box_3d(vehicle['dim'],vehicle['location'],vehicle['rotation'])
        # bb_cord = np.hstack((bb_cord,np.array([[1]]*8)))
        #print(cam_loc_dict)

        #cam1
        #cam1_v = None
        #cord_v1 = np.zeros((1,4))
        
        
        #cam1_to_world = np.dot(cam1_world_matrix,np.transpose(cord_v1))
        
        car_cam2_matrix = vehicle_matrix
        #cord_cam1 = np.dot(cam1_world_invmatrix,np.dot(cam2_world_matrix,np.transpose(cord)))
        cam2_to_world = np.dot(cam2_world_matrix,np.transpose(cord_p))
        p_in_cam1 = np.dot(cam1_world_invmatrix, cam2_to_world)
        ry_cam22world = vehicle['rotation'] - 90 + cam_point_dict['cam2'].rotation.yaw
        #ry_cam22world = ry_filter_a(ry_cam22world)
        ry_world2cam1 = (ry_cam22world - cam_point_dict['cam1'].rotation.yaw+90) * np.pi / 180
        #bb_in_cam1 = np.dot(cam1_world_invmatrix,np.dot(cam2_world_matrix,np.dot(vehicle_matrix,np.transpose(bb_cord))))
        #cord_cam1 = np.transpose(np.concatenate([bb_in_cam1[1,:],-bb_in_cam1[2,:],bb_in_cam1[0,:]])) #8x3
        #coed_p = np.concatenate([cord_p[:,1],-cord_p[:,2],cord_p[:,0]])
        #translated_vehicles.append(cord_cam1)
        #p_in_cam1 = np.transpose(np.concatenate([p_in_cam1[1,:],-p_in_cam1[2,:],p_in_cam1[0,:]]))
        
        tv = CamVehicle(p_in_cam1[1][0],-p_in_cam1[2][0],p_in_cam1[0][0],*vehicle['dim'],ry_world2cam1)
        translated_vehicles.append(tv)
        translated_points.append(p_in_cam1)
        translated_rotation.append(ry_world2cam1)
        # print('cam1_world:',cam1_to_world)
        # print('cam2_world:',cam2_to_world)
        # print('cam2ry_2_world',ry_cam22world)
        # print('ry_2_cam1',ry_world2cam1)
        # print('ry_cam1_gt',cam1_v['rotation'])
        # print('cam2_to_cam1',p_in_cam1)
        # print('cam1_gt',cord_v1)
    #print(cam_loc_dict)
    #print(translated_vehicles)
    #print(translated_points)
    calib = read_clib('/home/ubuntu/xwp/datasets/multi_view_dataset/346/calib/000000.txt')
    image = cv2.imread('/home/ubuntu/xwp/datasets/multi_view_dataset/346/image_2/000128.png')
    image2 = cv2.imread('/home/ubuntu/xwp/datasets/multi_view_dataset/347/image_2/000128.png')
    box3d_list = []
    box3d1_list = []
    for vehicle in translated_vehicles:
        box3d_list.append(vehicle.compute_box_3d())
    for v1 in cam1_vehicles:
        box3d1_list.append(v1.compute_box_3d())
    for box3d in box3d_list:
        #print(box_3d)
        box_2d = project_to_image(box3d, calib)
        #print(box_2d)
        image = draw_box_3d(image,box_2d)
        # cv2.imshow('image',image)
        # cv2.waitKey()
    for tv in vehicle_list_v:
        #print(box_3d)
        box_2d = project_to_image(tv.compute_box_3d(), calib)
        #print(box_2d)
        image2 = draw_box_3d(image2,box_2d)
    for box3d in box3d1_list:
        #print(box_3d)
        box_2d = project_to_image(box3d, calib)
        #print(box_2d)
        image = draw_box_3d(image,box_2d,c=(0,255,255))

    fused = bu.box3d_matching(box3d1_list,box3d_list,iou_threshold=0.1)
    #print(fused.shape)

    bird_view = add_bird_view(box3d_list)
    bird_view = add_bird_view(box3d1_list,bird_view=bird_view,lc=(12, 250, 152),lw=1)
    bird_view = cam_bird_view(cam_point_dict['cam1'],cam_point_dict['cam2'],bird_view)
    fused_bird = add_bird_view(fused)
    cv2.imshow('bird',bird_view)
    cv2.imshow('img',image)
    cv2.imshow('img2',image2)
    cv2.imshow('fbird',fused_bird)
    cv2.waitKey()

