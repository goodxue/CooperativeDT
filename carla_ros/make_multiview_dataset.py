#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
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
    import pygame
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

try:
    import queue
except ImportError:
    import Queue as queue

img=[]
VIEW_WIDTH = 1920//2
VIEW_HEIGHT = 1080//2
# VIEW_WIDTH = 1242
# VIEW_HEIGHT = 375
VIEW_FOV = 90
BB_COLOR = (255, 0, 0)
car_num=[260,180,100]
all_weather=[carla.WeatherParameters(cloudiness=60,sun_altitude_angle=-90,sun_azimuth_angle=150,
                fog_density=20,fog_falloff=2,wetness=30),
                carla.WeatherParameters(cloudiness=60,sun_altitude_angle=0,sun_azimuth_angle=150,
                fog_density=22,fog_falloff=2,wetness=40),
                carla.WeatherParameters(cloudiness=60,sun_altitude_angle=2,sun_azimuth_angle=90,
                fog_density=10,fog_falloff=0,wetness=20),
                carla.WeatherParameters(cloudiness=60,sun_altitude_angle=20,sun_azimuth_angle=120,
                fog_density=0,fog_falloff=0,wetness=0),
                carla.WeatherParameters(cloudiness=60,sun_altitude_angle=40,sun_azimuth_angle=180,
                fog_density=0,fog_falloff=0,wetness=0),
                carla.WeatherParameters(cloudiness=60,sun_altitude_angle=80,sun_azimuth_angle=280,
                fog_density=0,fog_falloff=0,wetness=0)]


class CarlaSyncMode(object):
    """
    Context manager to synchronize output from different sensors. Synchronous
    mode is enabled as long as we are inside this context

        with CarlaSyncMode(world, sensors) as sync_mode:
            while True:
                data = sync_mode.tick(timeout=1.0)

    """

    def __init__(self, world, *sensors, **kwargs):
        self.world = world
        self.sensors = sensors
        self.frame = None
        self.delta_seconds = 1.0 / kwargs.get('fps', 20)
        self._queues = []
        self._settings = None

    def __enter__(self):
        self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=True,
            fixed_delta_seconds=self.delta_seconds))

        def make_queue(register_event):
            q = queue.Queue()
            register_event(q.put)
            self._queues.append(q)

        make_queue(self.world.on_tick)
        for sensor in self.sensors:
            make_queue(sensor.listen)
        return self

    def tick(self, timeout):
        self.frame = self.world.tick()
        data = [self._retrieve_data(q, timeout) for q in self._queues]
        assert all(x.frame == self.frame for x in data)
        return data

    def __exit__(self, *args, **kwargs):
        self.world.apply_settings(self._settings)

    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.frame:
                return data

def Spawn_the_vehicles(world,client,car_num):
    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument(
        '-n', '--number-of-vehicles',
        metavar='N',
        default=260,
        type=int,
        help='number of vehicles (default: 10)')
    argparser.add_argument(
        '-w', '--number-of-walkers',
        metavar='W',
        default=0,
        type=int,
        help='number of walkers (default: 50)')
    argparser.add_argument(
        '--safe',
        action='store_true',
        help='avoid spawning vehicles prone to accidents')
    argparser.add_argument(
        '--filterv',
        metavar='PATTERN',
        default='vehicle.*',
        help='vehicles filter (default: "vehicle.*")')
    argparser.add_argument(
        '--filterw',
        metavar='PATTERN',
        default='walker.pedestrian.*',
        help='pedestrians filter (default: "walker.pedestrian.*")')
    argparser.add_argument(
        '--tm-port',
        metavar='P',
        default=8000,
        type=int,
        help='port to communicate with TM (default: 8000)')
    argparser.add_argument(
        '--sync',
        action='store_true',
        help='Synchronous mode execution')
    argparser.add_argument(
        '--hybrid',
        action='store_true',
        help='Enanble')
    argparser.add_argument(
        '-s', '--seed',
        metavar='S',
        type=int,
        help='Random device seed')
    argparser.add_argument(
        '--car-lights-on',
        action='store_true',
        default=False,
        help='Enanble car lights')
    args = argparser.parse_args()

    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    vehicles_list = []
    walkers_list = []
    all_id = []
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

    blueprints = world.get_blueprint_library().filter(args.filterv)
    blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels'))>=4]

    # import pdb; pdb.set_trace()
    if args.safe:
        blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
        # blueprints = [x for x in blueprints if not x.id.endswith('isetta')]
        # blueprints = [x for x in blueprints if not x.id.endswith('carlacola')]
        # blueprints = [x for x in blueprints if not x.id.endswith('cybertruck')]
        # blueprints = [x for x in blueprints if not x.id.endswith('t2')]

    blueprints = sorted(blueprints, key=lambda bp: bp.id)

    spawn_points = world.get_map().get_spawn_points()
    number_of_spawn_points = len(spawn_points)
    args.number_of_vehicles=car_num[0]
    if args.number_of_vehicles < number_of_spawn_points:
        random.shuffle(spawn_points)
    elif args.number_of_vehicles > number_of_spawn_points:
        msg = 'requested %d vehicles, but could only find %d spawn points'
        logging.warning(msg, args.number_of_vehicles, number_of_spawn_points)
        args.number_of_vehicles = number_of_spawn_points

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
        if n >= args.number_of_vehicles:
            break
        blueprint = random.choice(blueprints)
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        if blueprint.has_attribute('driver_id'):
            driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
            blueprint.set_attribute('driver_id', driver_id)
        blueprint.set_attribute('role_name', 'autopilot')

        # prepare the light state of the cars to spawn
        light_state = vls.NONE
        if args.car_lights_on:
            light_state = vls.Position | vls.LowBeam | vls.LowBeam

        # spawn the cars and set their autopilot and light state all together
        batch.append(SpawnActor(blueprint, transform)
            .then(SetAutopilot(FutureActor, True, traffic_manager.get_port()))
            .then(SetVehicleLightState(FutureActor, light_state)))

    for response in client.apply_batch_sync(batch, synchronous_master):
        if response.error:
            logging.error(response.error)
        else:
            vehicles_list.append(response.actor_id)

def draw_image(surface, image, blend=False):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    if blend:
        image_surface.set_alpha(100)
    surface.blit(image_surface, (0, 0))


def get_font():
    fonts = [x for x in pygame.font.get_fonts()]
    default_font = 'ubuntumono'
    font = default_font if default_font in fonts else fonts[0]
    font = pygame.font.match_font(font)
    return pygame.font.Font(font, 14)


def should_quit():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_ESCAPE:
                return True
    return False

def camera_coordinate(camera_bp):
    #Build the intrinsics projection matrix:
    # intrinsics = [[Fx,  0, image_w/2],
    #      [ 0, Fy, image_h/2],
    #      [ 0,  0,         1]]
    image_w = int(camera_bp.attributes["image_size_x"])
    image_h = int(camera_bp.attributes["image_size_y"])
    fov = float(camera_bp.attributes["fov"])
    focal = image_w / (2.0 * np.tan(fov * np.pi / 360.0))
    # In this case Fx and Fy are the same since the pixel aspect ratio is 1
    intrinsics = np.identity(3)
    intrinsics[0, 0] = intrinsics[1, 1] = focal
    intrinsics[0, 2] = image_w / 2.0
    intrinsics[1, 2] = image_h / 2.0
    # return intrinsics,extrinsics
    return intrinsics

# ==============================================================================
# -- ClientSideBoundingBoxes ---------------------------------------------------
# ==============================================================================


class ClientSideBoundingBoxes(object):
    """
    This is a module responsible for creating 3D bounding boxes and drawing them
    client-side on pygame surface.
    """

    @staticmethod
    def get_bounding_boxes(vehicles, camera):
        """
        Creates 3D bounding boxes based on carla vehicle list and camera.
        """

        bounding_boxes = [ClientSideBoundingBoxes.get_bounding_box(vehicle, camera) for vehicle in vehicles]
        # filter objects behind camera
        bounding_boxes = [bb for bb in bounding_boxes if all(bb[:, 2] > 0)]
        return bounding_boxes

    @staticmethod
    def draw_bounding_boxes(display, bounding_boxes):
        """
        Draws bounding boxes on pygame display.
        """

        bb_surface = pygame.Surface((VIEW_WIDTH, VIEW_HEIGHT))
        bb_surface.set_colorkey((0, 0, 0))
        for bbox in bounding_boxes:
            points = [(int(bbox[i, 0]), int(bbox[i, 1])) for i in range(8)]
            # pdb.set_trace()
            # draw lines
            # base
            pygame.draw.line(bb_surface, BB_COLOR, points[0], points[1])
            pygame.draw.line(bb_surface, BB_COLOR, points[0], points[1])
            pygame.draw.line(bb_surface, BB_COLOR, points[1], points[2])
            pygame.draw.line(bb_surface, BB_COLOR, points[2], points[3])
            pygame.draw.line(bb_surface, BB_COLOR, points[3], points[0])
            # top
            pygame.draw.line(bb_surface, BB_COLOR, points[4], points[5])
            pygame.draw.line(bb_surface, BB_COLOR, points[5], points[6])
            pygame.draw.line(bb_surface, BB_COLOR, points[6], points[7])
            pygame.draw.line(bb_surface, BB_COLOR, points[7], points[4])
            # base-top
            pygame.draw.line(bb_surface, BB_COLOR, points[0], points[4])
            pygame.draw.line(bb_surface, BB_COLOR, points[1], points[5])
            pygame.draw.line(bb_surface, BB_COLOR, points[2], points[6])
            pygame.draw.line(bb_surface, BB_COLOR, points[3], points[7])
        display.blit(bb_surface, (0, 0))

    @staticmethod
    def get_bounding_box(vehicle, camera):
        """
        Returns 3D bounding box for a vehicle based on camera view.
        """

        bb_cords = ClientSideBoundingBoxes._create_bb_points(vehicle)
        cords_x_y_z = ClientSideBoundingBoxes._vehicle_to_sensor(bb_cords, vehicle, camera)[:3, :]
        cords_y_minus_z_x = np.concatenate([cords_x_y_z[1, :], -cords_x_y_z[2, :], cords_x_y_z[0, :]])
        bbox = np.transpose(np.dot(camera_coordinate(camera), cords_y_minus_z_x))
        camera_bbox = np.concatenate([bbox[:, 0] / bbox[:, 2], bbox[:, 1] / bbox[:, 2], bbox[:, 2]], axis=1)
        return camera_bbox

    @staticmethod
    def _create_bb_points(vehicle):
        """
        Returns 3D bounding box for a vehicle.
        """

        cords = np.zeros((8, 4))
        extent = vehicle.bounding_box.extent
        cords[0, :] = np.array([extent.x, extent.y, -extent.z, 1])
        cords[1, :] = np.array([-extent.x, extent.y, -extent.z, 1])
        cords[2, :] = np.array([-extent.x, -extent.y, -extent.z, 1])
        cords[3, :] = np.array([extent.x, -extent.y, -extent.z, 1])
        cords[4, :] = np.array([extent.x, extent.y, extent.z, 1])
        cords[5, :] = np.array([-extent.x, extent.y, extent.z, 1])
        cords[6, :] = np.array([-extent.x, -extent.y, extent.z, 1])
        cords[7, :] = np.array([extent.x, -extent.y, extent.z, 1])
        return cords

    @staticmethod
    def _vehicle_to_sensor(cords, vehicle, sensor):
        """
        Transforms coordinates of a vehicle bounding box to sensor.
        """

        world_cord = ClientSideBoundingBoxes._vehicle_to_world(cords, vehicle)
        sensor_cord = ClientSideBoundingBoxes._world_to_sensor(world_cord, sensor)
        return sensor_cord

    @staticmethod
    def _vehicle_to_world(cords, vehicle):
        """
        Transforms coordinates of a vehicle bounding box to world.
        """
        bb_transform = carla.Transform(vehicle.bounding_box.location)
        bb_vehicle_matrix = ClientSideBoundingBoxes.get_matrix(bb_transform)
        vehicle_world_matrix = ClientSideBoundingBoxes.get_matrix(vehicle.get_transform())
        bb_world_matrix = np.dot(vehicle_world_matrix, bb_vehicle_matrix)
        world_cords = np.dot(bb_world_matrix, np.transpose(cords))
        return world_cords

    @staticmethod
    def _world_to_sensor(cords, sensor):
        """
        Transforms world coordinates to sensor.
        """

        sensor_world_matrix = ClientSideBoundingBoxes.get_matrix(sensor.get_transform())
        world_sensor_matrix = np.linalg.inv(sensor_world_matrix)
        sensor_cords = np.dot(world_sensor_matrix, cords)
        return sensor_cords

    @staticmethod
    def get_matrix(transform):
        """
        Creates matrix from carla transform.
        """
        rotation = transform.rotation
        location = transform.location
        c_y = np.cos(np.radians(rotation.yaw))
        s_y = np.sin(np.radians(rotation.yaw))
        c_r = np.cos(np.radians(rotation.roll))
        s_r = np.sin(np.radians(rotation.roll))
        c_p = np.cos(np.radians(rotation.pitch))
        s_p = np.sin(np.radians(rotation.pitch))
        matrix = np.matrix(np.identity(4))
        matrix[0, 3] = location.x
        matrix[1, 3] = location.y
        matrix[2, 3] = location.z
        matrix[0, 0] = c_p * c_y
        matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
        matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
        matrix[1, 0] = s_y * c_p
        matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
        matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
        matrix[2, 0] = s_p
        matrix[2, 1] = -c_p * s_r
        matrix[2, 2] = c_p * c_r
        return matrix

def main(cam_type,cam_subset,weather_num,car_num_id):
    actor_list = []
    pygame.init()

    display = pygame.display.set_mode((VIEW_WIDTH, VIEW_HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)
    font = get_font()
    clock = pygame.time.Clock()

    client = carla.Client('localhost', 2000,1)
    client.set_timeout(10.0)
    # cam_subset=1

    world = client.load_world('Town05')
    # weather = carla.WeatherParameters(
    #     cloudiness=0.0,
    #     precipitation=0.0,
    #     sun_altitude_angle=50.0)

    
    world.set_weather(all_weather[weather_num])

    
    
    # 加入其他车辆
    Spawn_the_vehicles(world,client,car_num[2])
    ###########相机参数
    # cam_type='cam1'
    
    # with open("/media/wuminghu/hard/hard/carla/cam.txt") as csv_file:
    #     reader = csv.reader(csv_file, delimiter=' ')
    #     for line, row in enumerate(reader):
    #         if row[0] in cam_type:
    #             w_x,w_y,w_yaw,cam_H,cam_pitch=[float(i) for i in row[1:]]
    #             break
    ###########相机参数
    # w_x,w_y,w_yaw,cam_H,cam_pitch=-116,100,100,4,90

    sensors_definition_file = 'dataset.json'
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
            point = Transform(Location(x=spawn_point.pop("x"), y=spawn_point.pop("y"), z=spawn_point.pop("z")),
                 Rotation(pitch=spawn_point.pop("pitch", 0.0), yaw=spawn_point.pop("yaw", 0.0), roll=spawn_point.pop("roll", 0.0)))
            camera_bp = world.get_blueprint_library().find(sensor_type)
            camera_bp.set_attribute('image_size_x', str(sensor_spec.pop("image_size_x")))
            camera_bp.set_attribute('image_size_y', str(sensor_spec.pop("image_size_y")))
            camera_bp.set_attribute('fov', str(sensor_spec.pop("fov")))

            camera_rgb = world.spawn_actor(camera_bp,point)
            actor_list.append(camera_rgb)
        except RuntimeError as e:
            raise RuntimeError("Setting up global sensors failed: {}".format(e))
            


    try:

        out_path="~/xwp/datasets/multi_view_dataset"
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        

        for camera_rgb in actor_list:
            calib_files=open("%s/%s/calib/%.6d.txt"%(out_path,camera_rgb.id,0), "w")
            K=camera_coordinate(camera_rgb)

            txt="P2: {} {} {} {} {} {} {} {} {}\n".format(  K[0][0],K[0][1],K[0][2],
                                                        K[1][0],K[1][1],K[1][2],
                                                        K[2][0],K[2][1],K[2][2])
            calib_files.write(txt)
            calib_files.close()

        ######################################################file
        with CarlaSyncMode(world, actor_list, fps=20) as sync_mode:
            count=0
            k=0
            while count<5000:
            # while count<100:
                count+=1
                if should_quit():
                    return
                clock.tick()
                # Advance the simulation and wait for the data.
                #snapshot, image_rgb,image_semseg = sync_mode.tick(timeout=2.0)
                blobs = sync_mode.tick(timeout=2.0)
                if count%10!=0:
                    continue
                k+=1

                # img.append(image_rgb)
                #image=image_rgb
                # image1=image_semseg.convert(ColorConverter.CityScapesPalette)
                # import pdb; pdb.set_trace()
                images = blobs[1:]
                snapshot = blobs[0]
                fps = round(1.0 / snapshot.timestamp.delta_seconds)

                
                world_snapshot = world.get_snapshot()

                actual_actor=[world.get_actor(actor_snapshot.id) for actor_snapshot in world_snapshot]
                got_vehicles=[actor for actor in actual_actor if actor.type_id.find('vehicle')!=-1]

                vehicles_list = []
                for camera_rgb in actor_list:
                    vehicles=[vehicle for vehicle in got_vehicles if vehicle.get_transform().location.distance(camera_rgb.get_transform().location)<80]
                    vehicles_list.append(vehicles)
                # debug = world.debug
                # for vehicle in vehicles:
                #     debug.draw_box(carla.BoundingBox(vehicle.get_transform().location+vehicle.bounding_box.location, vehicle.bounding_box.extent), vehicle.get_transform().rotation, 0.05, carla.Color(255,0,0,0), life_time=0.05)
                # import pdb; pdb.set_trace()
                
                for i,(camera_rgb, vehicles,image) in enumerate(zip(actor_list,vehicles_list,images)):
                    v=[] #filtered_vehicles
                    label_files=open("%s/%s/label_2/%.6d.txt"%(out_path,camera_rgb.id,k), "w")
                    car_2d_bbox=[]
                    for car in vehicles:
                        extent = car.bounding_box.extent
                        ###############location
                        car_location =car.bounding_box.location
                        # car_location1=car.get_transform().location
                        cords = np.zeros((1, 4))
                        cords[0, :]=np.array([car_location.x,car_location.y,car_location.z, 1])
                        cords_x_y_z = ClientSideBoundingBoxes._vehicle_to_sensor(cords, car, camera_rgb)[:3, :]
                        cords_y_minus_z_x = np.concatenate([cords_x_y_z[1, :], -cords_x_y_z[2, :], cords_x_y_z[0, :]])
                        ###############location
                        bbox = np.transpose(np.dot(camera_coordinate(camera_rgb), cords_y_minus_z_x))
                        camera_bbox = (np.concatenate([bbox[:, 0] / bbox[:, 2], bbox[:, 1] / bbox[:, 2], bbox[:, 2]], axis=1))[0]
                        # pdb.set_trace()
                        if camera_bbox[:,0]<0 or camera_bbox[:,1]<0 or camera_bbox[:,2]<0:
                            continue
                        
                        xmin,ymin,xmax,ymax=0,0,0,0
                        bboxe = ClientSideBoundingBoxes.get_bounding_boxes([car], camera_rgb)
                        if len(bboxe)==0:
                            continue
                        bboxe=bboxe[0]
                        t_points = [(int(bboxe[i, 0]), int(bboxe[i, 1])) for i in range(8)]
                        width_x=[int(bboxe[i, 0]) for i in range(8)]
                        high_y=[int(bboxe[i, 1]) for i in range(8)]
                        xmin,ymin,xmax,ymax=min(width_x),min(high_y),max(width_x),max(high_y)
                        x_cen=(xmin+xmax)/2
                        y_cen=(ymin+ymax)/2
                        if x_cen<0 or y_cen<0 or x_cen>VIEW_WIDTH or y_cen>VIEW_HEIGHT:
                            continue
                        car_type, truncated, occluded, alpha= 'Car', 0, 0,0
                        dh, dw,dl=extent.z*2,extent.y*2,extent.x*2
                        cords_y_minus_z_x=np.array(cords_y_minus_z_x)
                        ly, lz,lx=cords_y_minus_z_x[0][0],cords_y_minus_z_x[1][0],cords_y_minus_z_x[2][0]
                        lz=lz+dh
                        ry=(car.get_transform().rotation.yaw-camera_rgb.get_transform().rotation.yaw+90)*np.pi/180
                        check_box=False
                        for one_box in car_2d_bbox:
                            xmi,ymi,xma,yma=one_box
                            if xmin>xmi and ymin>ymi and xmax<xma and ymax<yma:
                                check_box=True
                        if check_box or np.sqrt(ly**2+lx**2)<3:
                            continue
                        car_2d_bbox.append([xmin,ymin,xmax,ymax])
                        v.append(car)
                        if ry>np.pi:
                            ry-=np.pi
                        if ry<-np.pi:
                            ry+=np.pi
                        # pdb.set_trace()
                        txt="{} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}\n".format(car_type, truncated, occluded, alpha, xmin, ymin, xmax, ymax, dh, dw,
                            dl,ly, lz, lx,  ry,car.id)
                        label_files.write(txt)
                    label_files.close()
                    print(cam_type,cam_subset,k,len(v))

                    #bounding_boxes = ClientSideBoundingBoxes.get_bounding_boxes(v, camera_rgb) 
                    ######################################################file
                    # pygame.image.save(display,'%s/image_1/%06d.png' % (out_path,k))       
                    image.save_to_disk('%s/%s/image_2/%06d.png' % (out_path,camera_rgb.id,k))           


    finally:

        print('destroying actors.')
        for actor in actor_list:
            actor.destroy()

        pygame.quit()

        return
def open_serve():
    os.system("bash run_carla.sh")

def multipro():
    all_cam=['cam16']
    # all_cam=['cam12']
    # finish_cam=['cam1']
    # finish_sub=[1,2,3,4,5,6,7,8]
    for cam_id in range(len(all_cam)):
        cam_subset_id=1
        for weather_num_id in range(len(all_weather)):
            for num_id in range(len(car_num)):
                # pdb.set_trace()
                # if (all_cam[cam_id] in finish_cam ) and (cam_subset_id in finish_sub):
                #     cam_subset_id+=1
                #     continue
                # os.system("bash run_carla.sh")
                if cam_subset_id not in [7,11]:
                    print(cam_subset_id)
                    cam_subset_id+=1
                    continue
                p1 = multiprocessing.Process(target=open_serve,args=())
                p2 = multiprocessing.Process(target=main,args=(all_cam[cam_id],cam_subset_id,weather_num_id,num_id))
                p1.start()
                time.sleep(5)
                p2.start()

                p2.join()

                p1.terminate()
                p1.join()

                p2.close()
                p1.close()
                cam_subset_id+=1
    print('done.')
if __name__ == '__main__':
    multipro()
    # all_cam=['cam3']
    # cam_id=0
    # cam_subset_id=1
    # weather_num_id=4
    # num_id=2
    # main(all_cam[cam_id],cam_subset=cam_subset_id,weather_num=weather_num_id,car_num_id=num_id)

    

