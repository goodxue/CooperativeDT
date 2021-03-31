from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import sys
import numpy as np
import cv2

from opts import opts
from detectors.detector_factory import detector_factory
from utils.multiview_utils import *
import json
import logging
import time
import utils.bbox_utils as bbox_utils

image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']
color_def = {'cam1':(255,0,0),'cam2':(0,255,0)}

def demo(opt):
  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  opt.debug = 1#max(opt.debug, 1)

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

  path_dict = {}
  if os.path.isdir(opt.multiview):
    for cam in opt.camn:
      path_tmp = os.path.join(opt.multiview,str(cam),'image_2',opt.img)
      if os.path.exists(path_tmp):
        path_dict['cam{}'.format(len(path_dict)+1)]=path_tmp
      else:
        raise RuntimeError('path to {} is not exist'.format(path_tmp))

  Detector = detector_factory[opt.task]
  detector = Detector(opt)
  result_dict = {}
  print(path_dict)
  for cam,img_name in path_dict.items():
    result_dict[cam]=(detector.run(img_name))

  #process multi-view transform
  #det = result_dict['cam2']
  translated_vehicles = []
  # translated_points = []
  # translated_rotation = []
  cam1_world_invmatrix = np.linalg.inv(cam_loc_dict['cam1'])
  cam2_world_matrix = cam_loc_dict['cam2']
  cam1_world_matrix = cam_loc_dict['cam1']
  dets = result_dict['cam2']['results']
  for cat in dets:
      for i in range(len(dets[cat])):
        if dets[cat][i, -1] > 0.3:
          dim = dets[cat][i, 5:8]
          loc  = dets[cat][i, 8:11]
          rot_y = dets[cat][i, 11]
          loc_carla = [loc[2],loc[0],-loc[1],1]
          cord_p = np.ones((1,4))
          cord_p[0] = np.array(loc_carla)
          loc_world = np.dot(cam2_world_matrix,np.transpose(cord_p))
          loc_cam1 = np.dot(cam1_world_invmatrix, loc_world)
          ry_world = rot_y - 90 + cam_point_dict['cam2'].rotation.yaw
          ry_world2cam1 = ry_world - cam_point_dict['cam1'].rotation.yaw+90
          
          tv = CamVehicle(loc_cam1[1][0],-loc_cam1[2][0],loc_cam1[0][0],*dim,ry_world2cam1)
          translated_vehicles.append(tv)
  box3d_dict={}
  for k in cam_loc_dict.keys():
    box3d_dict[k] = []
  for vehicle in translated_vehicles:
    box3d_dict['cam2'].append(vehicle.compute_box_3d())
  dets_cam1 = result_dict['cam1']['results']
  for cat in dets_cam1:
      for i in range(len(dets_cam1[cat])):
        if dets_cam1[cat][i, -1] > 0.3:
          dim = dets_cam1[cat][i, 5:8]
          loc  = dets_cam1[cat][i, 8:11]
          rot_y = dets_cam1[cat][i, 11]
          rect = compute_box_3d(dim, loc, rot_y)
          box3d_dict['cam1'].append(rect)

  calib = read_clib('/home/ubuntu/xwp/datasets/multi_view_dataset/346/calib/000000.txt')
  image = cv2.imread('/home/ubuntu/xwp/datasets/multi_view_dataset/346/image_2/000128.png')

  bird_view = None
  for key, value in box3d_dict.items():
    color = color_def[key]
    bird_view = add_bird_view(value,bird_view=bird_view,lc=color)
    for box3d in value:
      box_2d = project_to_image(box3d, calib)
      image_f = draw_box_3d(image,box_2d,color)
  
  cv2.imshow('image',image_f)
  cv2.imshow('bird',bird_view)
  cv2.waitKey()
  # else:
  #   if os.path.isdir(opt.demo):
  #     image_names = []
  #     ls = os.listdir(opt.demo)
  #     for file_name in sorted(ls):
  #         ext = file_name[file_name.rfind('.') + 1:].lower()
  #         if ext in image_ext:
  #             image_names.append(os.path.join(opt.demo, file_name))
  #   else:
  #     image_names = [opt.demo]
    
  #   for (image_name) in image_names:
  #     ret = detector.run(image_name)
  #     time_str = ''
  #     for stat in time_stats:
  #       time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
  #     print(time_str)
if __name__ == '__main__':
  opt = opts().init()
  demo(opt)
