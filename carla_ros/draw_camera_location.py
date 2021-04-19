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
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('-n',default='1',type=str,)
    args = argparser.parse_args()
    FILTER_GLOBAL = True
    NUM_CAM = 3
    dataset_path = '/home/ubuntu/xwp/datasets/multi_view_dataset/new'
    cam_set = ['cam{}'.format(args.n),'cam10','cam18']
    print('processing: ',cam_set)
    camset_path = [ os.path.join(dataset_path,cam_name) for cam_name in cam_set
        #'/home/ubuntu/xwp/datasets/multi_view_dataset/new/cam1/label_test'
        # '',
        # ''
    ]
    cam_path = [os.path.join(path,'label_test') for path in camset_path]

    cam_transform = {}
    sensors_definition_file = '/home/ubuntu/xwp/CenterNet/carla_ros/dataset.json'
    if not os.path.exists(sensors_definition_file):
        raise RuntimeError(
            "Could not read sensor-definition from {}".format(sensors_definition_file))
    with open(sensors_definition_file) as handle:
        json_actors = json.loads(handle.read())
    global_sensors = []
    for actor in json_actors["objects"]:
        global_sensors.append(actor)
    for sensor_spec in global_sensors:
        sensor_id = str(sensor_spec.pop("id"))
        spawn_point = sensor_spec.pop("spawn_point")
        point = Transform(location=Location(x=spawn_point.pop("x"), y=-spawn_point.pop("y"), z=spawn_point.pop("z")),
                rotation=Rotation(pitch=-spawn_point.pop("pitch", 0.0), yaw=-spawn_point.pop("yaw", 0.0), roll=spawn_point.pop("roll", 0.0)))
        cam_transform[sensor_id] = point
    
    img = cams_bird_view((-80,-140),list(cam_transform.values())[:])
    cv2.imshow('bird',img)
    cv2.waitKey()