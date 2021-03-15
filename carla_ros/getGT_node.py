#!/usr/bin/env python

import os
import sys
import math
import json
import rospy
from tf.transformations import quaternion_from_euler
from geometry_msgs.msg import Pose
from sensor_msgs.msg import CameraInfo, Image

class GetGroundTruth(object):
    def __init__(self,cameras):
        rospy.init_node('get_groundtruth_node', anonymous=True)
        self.objects_definition_file = rospy.get_param('~objects_definition_file')
        self.spawn_sensors_only = rospy.get_param('~spawn_sensors_only', None)

        self.publisher_list = []

        id_list = [camera.id for camera in cameras]


        for camera_id in id_list"
            camera_publisher = rospy.Publisher('/carla/{}/image'.format(camera_id),
                                                        Image,
                                                        queue_size=1)
            self.publisher_list.append(camera_publisher)
        


if __name__ == '__main__':
    client = carla.Client('localhost', 2001)
    client.set_timeout(10.0)
    world = client.get_world()
    world_snapshot = world.get_snapshot()
    cameras=[actor for actor in actual_actor if actor.type_id.find('camera')!=-1]
    rospy.info("find {} camera sensors!".format(len(cameras)))
    getGT_node = GetGroundTruth(cameras)

    try:
        while not rospy.core.is_shutdown():
            world_snapshot = world.get_snapshot()
            actual_actor=[world.get_actor(actor_snapshot.id) for actor_snapshot in world_snapshot]
            vehicles=[actor for actor in actual_actor if actor.type_id.find('vehicle')!=-1]
        
