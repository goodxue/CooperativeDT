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

import utils as ut

def Spawn_the_vehicles(world,client,spawn_points):
    #roi (xmin,ymin,xmax,ymax)
    #logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    synchronous_master = False
    random.seed(args.seed if args.seed is not None else int(time.time()))
    traffic_manager = client.get_trafficmanager(args.tm_port)
    traffic_manager.set_global_distance_to_leading_vehicle(1.0)
    if args.hybrid:
        traffic_manager.set_hybrid_physics_mode(True)
    if args.seed is not None:
        traffic_manager.set_random_device_seed(args.seed)


    if args.sync:
        settings = world.get_settings()
        traffic_manager.set_synchronous_mode(True)
        if not settings.synchronous_mode:
            synchronous_master = True
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05
            world.apply_settings(settings)
        else:
            synchronous_master = False

    vehicles_list = []
    walkers_list = []
    all_id = []

    blueprints = world.get_blueprint_library().filter(args.filterv)
    blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels'))>=4]

    # import pdb; pdb.set_trace()
    # if args.safe:
    #     blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
        # blueprints = [x for x in blueprints if not x.id.endswith('isetta')]
        # blueprints = [x for x in blueprints if not x.id.endswith('carlacola')]
        # blueprints = [x for x in blueprints if not x.id.endswith('cybertruck')]
        # blueprints = [x for x in blueprints if not x.id.endswith('t2')]

    blueprints = sorted(blueprints, key=lambda bp: bp.id)

    # spawn_points = world.get_map().get_spawn_points()
    # filtered_points = [point for point in spawn_points if roi[0]<point.location.x<roi[2] and roi[1]<point.location.y<roi[3]]

    # number_of_spawn_points = len(filtered_points)
    # filtered_points = sorted(filtered_points,key=lambda point: point.location.x)


    # @todo cannot import these directly.
    SpawnActor = carla.command.SpawnActor
    SetAutopilot = carla.command.SetAutopilot
    SetVehicleLightState = carla.command.SetVehicleLightState
    FutureActor = carla.command.FutureActor

    # --------------
    # Spawn vehicles
    # --------------
    batch = []
    for n, transform in enumerate(spawn_points):
        while n > len(blueprints):
            n = n - len(blueprints)
        blueprint = blueprints[n]

        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        if blueprint.has_attribute('driver_id'):
            driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
            blueprint.set_attribute('driver_id', driver_id)
        blueprint.set_attribute('role_name', 'autopilot')

        # prepare the light state of the cars to spawn
        # light_state = vls.NONE
        # if args.car_lights_on:
        #     light_state = vls.Position | vls.LowBeam | vls.LowBeam

        # spawn the cars and set their autopilot and light state all together
        batch.append(SpawnActor(blueprint, transform))
            # .then(SetAutopilot(FutureActor, True, traffic_manager.get_port()))
            # .then(SetVehicleLightState(FutureActor, light_state)))

    for response in client.apply_batch_sync(batch, synchronous_master):
        if response.error:
            logging.error(response.error)
        else:
            vehicles_list.append(response.actor_id)
    
    return vehicles_list

if __name__ == '__main__':
    random.seed(args.seed if args.seed is not None else int(time.time()))
    img_size = np.asarray([960,540],dtype=np.int)
    actor_list = []
    rgb_list = []
    client = carla.Client('localhost', 2000,1)
    client.set_timeout(60.0)
    world = client.get_world()
    world.set_weather(getattr(carla.WeatherParameters, "ClearNoon"))

    # Spawn sensors
    sensors_definition_file = '/home/ubuntu/xwp/CenterNet/carla_ros/dataset.json'
    if not os.path.exists(sensors_definition_file):
        raise RuntimeError(
            "Could not read sensor-definition from {}".format(sensors_definition_file))

    with open(sensors_definition_file) as handle:
        json_actors = json.loads(handle.read())
    
    global_sensors = []
    for actor in json_actors["objects"]:
        global_sensors.append(actor)
    
    camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
    sizex = str(960)
    sizey = str(540)
    fovcfg = str(90)
    camera_bp.set_attribute('image_size_x',sizex )
    camera_bp.set_attribute('image_size_y', sizey)
    camera_bp.set_attribute('fov', fovcfg)
    semseg_bp = world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
    semseg_bp.set_attribute('image_size_x', sizex)
    semseg_bp.set_attribute('image_size_y', sizey)
    semseg_bp.set_attribute('fov', fovcfg)

    for sensor_spec in global_sensors:
        try:
            sensor_names = []
            sensor_type = str('sensor.camera.rgb')
            sensor_id = str(sensor_spec.pop("id"))

            sensor_name = sensor_type + "/" + sensor_id
            if sensor_name in sensor_names:
                raise NameError
            sensor_names.append(sensor_name)

            spawn_point = sensor_spec.pop("spawn_point")
            point = Transform(Location(x=spawn_point.pop("x"), y=-spawn_point.pop("y"), z=spawn_point.pop("z")),
                 Rotation(pitch=-spawn_point.pop("pitch", 0.0), yaw=-spawn_point.pop("yaw", 0.0), roll=spawn_point.pop("roll", 0.0)))
            # camera_bp.set_attribute('sensor_id',str(sensor_id))

            camera_rgb = world.spawn_actor(camera_bp,point)
            #camera_seg = world.spawn_actor(semseg_bp,point)
            camera_rgb.sensor_name = sensor_id
            #camera_seg.sensor_name = sensor_id + "_seg"  
            actor_list.append(camera_rgb)
            #actor_list.append(camera_seg)
            rgb_list .append(camera_rgb)
            print('spawned: ',camera_rgb.sensor_name)
        except RuntimeError as e:
            raise RuntimeError("Setting up global sensors failed: {}".format(e))
    # Spawn finished!

   #  Read spawn points json
    points_list = []
    transform_list = []
    spawnpoints_definition_file = '/home/ubuntu/xwp/CenterNet/carla_ros/points.json'
    if not os.path.exists(spawnpoints_definition_file):
    raise RuntimeError(
        "Could not read points-definition from {}".format(spawnpoints_definition_file))
    
    with open(sensors_definition_file) as handle:
        json_points = json.loads(handle.read())
    
    for point in json_points["points"]:
        points_list.append(actor)
    
    for point in points_list:
        trans = Transform(Location(x=point.pop("x"), y=point.pop("y"), z=point.pop("z")),
            Rotation(pitch=point.pop("pitch", 0.0), yaw=point.pop("yaw", 0.0), roll=point.pop("roll", 0.0)))
        transform_list.append(trans)
    # Finished!
    vehicles_actor = Spawn_the_vehicles(world,client,transform_list)
