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
#from utils import *
VIEW_WIDTH = 1920//2
VIEW_HEIGHT = 1080//2
# VIEW_WIDTH = 1242
# VIEW_HEIGHT = 375
VIEW_FOV = 90
BB_COLOR = (255, 0, 0)
car_num=[260,180,100]
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

def Spawn_the_vehicles(world,client,spawn_points):
    #roi (xmin,ymin,xmax,ymax)
    #logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    synchronous_master = False
    random.seed(int(time.time()))
    #traffic_manager = client.get_trafficmanager(args.tm_port)
    #traffic_manager.set_global_distance_to_leading_vehicle(1.0)
    # if args.hybrid:
    #     traffic_manager.set_hybrid_physics_mode(True)
    # if args.seed is not None:
    #     traffic_manager.set_random_device_seed(args.seed)


    # if args.sync:
    #     settings = world.get_settings()
    #     traffic_manager.set_synchronous_mode(True)
    #     if not settings.synchronous_mode:
    #         synchronous_master = True
    #         settings.synchronous_mode = True
    #         settings.fixed_delta_seconds = 0.05
    #         world.apply_settings(settings)
    #     else:
    #         synchronous_master = False

    vehicles_list = []
    walkers_list = []
    all_id = []

    blueprints = world.get_blueprint_library().filter('vehicle.*')
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
        while n >= len(blueprints):
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
    #random.seed(args.seed if args.seed is not None else int(time.time()))
    img_size = np.asarray([960,540],dtype=np.int)
    actor_list = []
    rgb_list = []
    client = carla.Client('localhost', 2000,1)
    client.set_timeout(10.0)
    world = client.get_world()
    world.set_weather(getattr(carla.WeatherParameters, "ClearNoon"))

    # Spawn sensors
    sensors_definition_file = '/home/ubuntu/xwp/CenterNet/carla_ros/dataset_test.json'
    if not os.path.exists(sensors_definition_file):
        raise RuntimeError(
            "Could not read sensor-definition from {}".format(sensors_definition_file))

    CAM_SET=['cam3','cam20','cam34']
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
        raise RuntimeError("Could not read points-definition from {}".format(spawnpoints_definition_file))
    
    with open(spawnpoints_definition_file) as handle:
        json_points = json.loads(handle.read())
    
    for point in json_points["points"]:
        points_list.append(point)
    
    for point in points_list:
        trans = Transform(Location(x=point.pop("x"), y=-point.pop("y"), z=point.pop("z")),
            Rotation(pitch=-point.pop("pitch", 0.0), yaw=-point.pop("yaw", 0.0), roll=point.pop("roll", 0.0)))
        transform_list.append(trans)
    # Finished!
    vehicles_actor = Spawn_the_vehicles(world,client,transform_list)

    out_path="/home/ubuntu/xwp/datasets/multi_view_dataset/crowd_test2"
    if not os.path.exists(out_path):
        os.makedirs(out_path)


    with CarlaSyncMode(world, *actor_list, fps=20) as sync_mode:
        count=0
        k=0
        while count<20:
        # while count<100:
            count+=1
            #clock.tick()
            # Advance the simulation and wait for the data.
            #snapshot, image_rgb,image_semseg = sync_mode.tick(timeout=2.0)
            blobs = sync_mode.tick(timeout=2.0)
            if count%20!=0:
                continue
            k+=1
            print(k)

            # img.append(image_rgb)
            #image=image_rgb
            # image1=image_semseg.convert(ColorConverter.CityScapesPalette)
            # import pdb; pdb.set_trace()
            all_images = blobs[1:]
            images = all_images
            #images = all_images[::2]
            #semseg_images = all_images[1::2]
            snapshot = blobs[0]
            fps = round(1.0 / snapshot.timestamp.delta_seconds)

            
            world_snapshot = world.get_snapshot()

            actual_actor=[world.get_actor(actor_snapshot.id) for actor_snapshot in world_snapshot]
            got_vehicles=[actor for actor in actual_actor if actor.type_id.find('vehicle')!=-1]

            vehicles_list = []
            for camera_rgb in rgb_list:
                vehicles=[vehicle for vehicle in got_vehicles if vehicle.get_transform().location.distance(camera_rgb.get_transform().location)<65]
                vehicles_list.append(vehicles)
                cam_path = out_path
                if not os.path.exists(cam_path):
                    os.makedirs(cam_path)
                if not os.path.exists(cam_path+"/label_2"):
                    os.makedirs(cam_path+"/label_2")
                    os.makedirs(cam_path+"/image_2")
                if not os.path.exists(cam_path+"/calib"):
                    os.makedirs(cam_path+"/calib")
            # debug = world.debug
            # for vehicle in vehicles:
            #     debug.draw_box(carla.BoundingBox(vehicle.get_transform().location+vehicle.bounding_box.location, vehicle.bounding_box.extent), vehicle.get_transform().rotation, 0.05, carla.Color(255,0,0,0), life_time=0.05)
            # import pdb; pdb.set_trace()
            all_vehicles_list = []
            
            for i,(camera_rgb, vehicles,image) in enumerate(zip(rgb_list,vehicles_list,images)):
                v=[] #filtered_vehicles

                # Convert semseg to cv.image
                # semseg = np.frombuffer(osemseg.raw_data, dtype=np.dtype("uint8"))
                # semseg = np.reshape(semseg, (osemseg.height, osemseg.width, 4))
                # semseg = semseg[:, :, :3]
                # semseg = semseg[:, :, ::-1]
                # BGR
                # Done

                label_files=open("{}/label_2/{}{:0>4d}.txt".format(out_path,camera_rgb.sensor_name[3:],k), "w")
                car_2d_bbox=[]
                for car in vehicles:
                    extent = car.bounding_box.extent
                    ###############location
                    car_location =car.bounding_box.location
                    # car_location1=car.get_transform().location
                    cords = np.zeros((1, 4))
                    cords[0, :]=np.array([car_location.x,car_location.y,car_location.z, 1])
                    cords_x_y_z = ut.ClientSideBoundingBoxes._vehicle_to_sensor(cords, car, camera_rgb)[:3, :]
                    cords_y_minus_z_x = np.concatenate([cords_x_y_z[1, :], -cords_x_y_z[2, :], cords_x_y_z[0, :]])
                    ###############location
                    bbox = np.transpose(np.dot(camera_coordinate(camera_rgb), cords_y_minus_z_x))
                    camera_bbox = (np.concatenate([bbox[:, 0] / bbox[:, 2], bbox[:, 1] / bbox[:, 2], bbox[:, 2]], axis=1))
                    # pdb.set_trace()
                    camera_bbox_e = camera_bbox[0]
                    # if camera_bbox_e[:,0]<0 or camera_bbox_e[:,1]<0 or camera_bbox_e[:,2]<0:
                    #     continue
                    if camera_bbox_e[:,2]<0:
                        continue
                    
                    #bboxe = camera_bbox
                    xmin,ymin,xmax,ymax=0,0,0,0
                    bboxe = ut.ClientSideBoundingBoxes.get_bounding_boxes([car], camera_rgb)
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
                    ofs = 3
                    for one_box in car_2d_bbox:
                        xmi,ymi,xma,yma=one_box
                        if xmin>xmi-ofs and ymin>ymi-ofs and xmax<xma+ofs and ymax<yma+ofs:
                            check_box=True
                    if check_box or np.sqrt(ly**2+lx**2)<3:
                        continue

                    bbox_crop = tuple(max(0, b) for b in [xmin,ymin,xmax,ymax])
                    bbox_crop = (min(img_size[0], bbox_crop[0]),
                                min(img_size[1], bbox_crop[1]),
                                min(img_size[0], bbox_crop[2]),
                                min(img_size[1], bbox_crop[3]))
                    # Use segment image to determine whether the vehicle is occluded.
                    # See https://carla.readthedocs.io/en/0.9.11/ref_sensors/#semantic-segmentation-camera
                    # print('seg: ',semseg[int((ymin+ymax)/2),int((xmin+xmax)/2),0])
                    # print('x: ',int((ymin+ymax)/2),'y: ',int((xmin+xmax)/2))
                    # if semseg[int((ymin+ymax)/2),int((xmin+xmax)/2),0] != 10:
                    #     continue

                    car_2d_bbox.append(bbox_crop)
                    
                    v.append(car)
                    if car not in all_vehicles_list:
                        all_vehicles_list.append(car)
                    if ry>np.pi:
                        ry-=np.pi
                    if ry<-np.pi:
                        ry+=np.pi
                    # pdb.set_trace()
                    txt="{} {} {} {} {} {} {} {} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {} {}\n".format(car_type, truncated, occluded, alpha, xmin, ymin, xmax, ymax, dh, dw,
                        dl,ly, lz, lx,  ry,car.id)
                    label_files.write(txt)
                label_files.close()
                #print(cam_type,cam_subset,k,len(v))

                #bounding_boxes = ClientSideBoundingBoxes.get_bounding_boxes(v, camera_rgb) 
                ######################################################file
                # pygame.image.save(display,'%s/image_1/%06d.png' % (out_path,k))       
                image.save_to_disk('{}/image_2/{}{:0>4d}.png'.format (out_path,camera_rgb.sensor_name[3:],k))        
            
            global_file_path = out_path + '/global_label_2'
            if not os.path.exists(global_file_path):
                os.makedirs(global_file_path)
            global_vehicle_file = open("{}/{:0>6d}.txt".format(global_file_path,k), "w")
            for vehicle in all_vehicles_list:
                extent = vehicle.bounding_box.extent
                car_type, truncated, occluded, alpha,xmin,ymin,xmax,ymax= 'Car', 0, 0,0,0,0,0,0
                dh, dw,dl=extent.z*2,extent.y*2,extent.x*2
                location  = vehicle.get_transform().location
                ly,lz,lx = location.x,location.y,location.z
                ry = (vehicle.get_transform().rotation.yaw) * np.pi /180
                
                txt="{} {} {} {} {} {} {} {} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {} {}\n".format(car_type, truncated, occluded, alpha, xmin, ymin, xmax, ymax, dh, dw,
                            dl,ly, lz, lx,  ry,vehicle.id)
                global_vehicle_file.write(txt)
            global_vehicle_file.close()


    calib_files=open("{}/calib/{:0>6d}.txt".format(out_path,0), "w")
    K=camera_coordinate(rgb_list[0])

    txt="P2: {} {} {} {} {} {} {} {} {}\n".format(  K[0][0],K[0][1],K[0][2],
                                                K[1][0],K[1][1],K[1][2],
                                                K[2][0],K[2][1],K[2][2])
    calib_files.write(txt)
    calib_files.close()


    print('destroying actors.')
    for actor in actor_list:
        actor.destroy()
    
    for vehicle in vehicles_actor:
        vehicle.destory()
