import os
import sys
import math
import json
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

VIEW_WIDTH = 1920//2
VIEW_HEIGHT = 1080//2
actor_list = []

def setup_sensors(world, sensors, attached_vehicle_id=None):
    sensor_names = []
    for sensor_spec in sensors:
        try:
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

            camera_rgb = world.spawn_actor(camera_bp,camera_point)
            actor_list.append(camera_rgb)


        except NameError:
            rospy.logerr("Sensor rolename '{}' is only allowed to be used once.".format(
                sensor_spec['id']))
            continue

def main(sensors_definition_file):
    if not os.path.exists(sensors_definition_file):
        raise RuntimeError(
            "Could not read sensor-definition from {}".format(sensosr_definition_file))

    with open(sensors_definition_file) as handle:
        json_actors = json.loads(handle.read())
    
    global_sensors = []
    for actor in json_actors["objects"]:
        actor_type = actor["type"].split('.')[0]
        if actor_type == "sensor":
            global_sensors.append(actor)
        else:
            continue
    
    try:
        client = carla.Client('localhost', 2000,1)
        client.set_timeout(10.0)
        world = client.get_world()

        self.setup_sensors(world, global_sensors)
        except RuntimeError as e:
            raise RuntimeError("Setting up global sensors failed: {}".format(e))

        while True:
            if args.sync and synchronous_master:
                world.tick()
            else:
                world.wait_for_tick()

    finally:

        print('destroying actors.')
        for actor in actor_list:
            actor.destroy()
    