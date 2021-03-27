import glob,pdb
import os
import sys
import json
import argparse
import logging
import time
import csv
import cv2
from carla import VehicleLightState as vls
from carla import Transform, Location, Rotation
from carla import ColorConverter
import multiprocessing
#import kitti_util as utils


try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

import random

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

try:
    import queue
except ImportError:
    import Queue as queue

from utils import *

if __name__ =="__main__":
    client = carla.Client('localhost', 2000,1)
    client.set_timeout(10.0)
    # cam_subset=1

    world = client.get_world()
    world_snapshot = world.get_snapshot()
    actual_actor=[world.get_actor(actor_snapshot.id) for actor_snapshot in world_snapshot]
    got_cameras=[actor for actor in actual_actor if actor.type_id.find('rgb')!=-1]

    for camera_rgb in got_cameras:
        tr = camera_rgb.get_transform()
        print("matrix: ",ClientSideBoundingBoxes.get_matrix(tr))
    
    print("finished")