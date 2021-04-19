import numpy as np
import cv2
import os
from utils import *
import json
import argparse

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


def _bbox_inside(box1, box2):
  #coco box
  return box1[0] > box2[0] and box1[0] + box1[2] < box2[0] + box2[2] and \
         box1[1] > box2[1] and box1[1] + box1[3] < box2[1] + box2[3]

def _rot_y2alpha(rot_y, x, cx, fx):
    """
    Get rotation_y by alpha + theta - 180
    alpha : Observation angle of object, ranging [-pi..pi]
    x : Object center x to the camera center (x-W/2), in pixels
    rotation_y : Rotation ry around Y-axis in camera coordinates [-pi..pi]
    """
    ray = np.arctan2(x - cx, fx)
    alpha = rot_y - ray
    if alpha > np.pi:
      alpha -= 2 * np.pi
    if alpha < -np.pi:
      alpha += 2 * np.pi
    return alpha, ray

def compute_box_3d(dim, location, rotation_y):
  # dim: 3
  # location: 3
  # rotation_y: 1
  # return: 8 x 3
  c, s = np.cos(rotation_y), np.sin(rotation_y)
  R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
  l, w, h = dim[2], dim[1], dim[0]
  x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
  y_corners = [0,0,0,0,-h,-h,-h,-h]
  z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]

  corners = np.array([x_corners, y_corners, z_corners], dtype=np.float32)
  corners_3d = np.dot(R, corners) 
  corners_3d = corners_3d + np.array(location, dtype=np.float32).reshape(3, 1)
  return corners_3d.transpose(1, 0)

def project_to_image(pts_3d, P):
  # pts_3d: n x 3
  # P: 3 x 4
  # return: n x 2
#   pts_3d_homo = np.concatenate(
#     [pts_3d, np.ones((pts_3d.shape[0], 1), dtype=np.float32)], axis=1)
  pts_2d = np.dot(P, pts_3d.transpose(1, 0)).transpose(1, 0)
  pts_2d = pts_2d[:, :2] / pts_2d[:, 2:]
  # import pdb; pdb.set_trace()
  return pts_2d

def draw_box_3d(image, corners, c=(0, 0, 255)):
  face_idx = [[0,1,5,4],
              [1,2,6, 5],
              [2,3,7,6],
              [3,0,4,7]]
  for ind_f in range(3, -1, -1):
    f = face_idx[ind_f]
    for j in range(4):
      cv2.line(image, (corners[f[j], 0], corners[f[j], 1]),
               (corners[f[(j+1)%4], 0], corners[f[(j+1)%4], 1]), c, 1, lineType=cv2.LINE_AA)
    if ind_f == 0:
      cv2.line(image, (corners[f[0], 0], corners[f[0], 1]),
               (corners[f[2], 0], corners[f[2], 1]), c, 1, lineType=cv2.LINE_AA)
      cv2.line(image, (corners[f[1], 0], corners[f[1], 1]),
               (corners[f[3], 0], corners[f[3], 1]), c, 1, lineType=cv2.LINE_AA)
  return image

def draw_box_2d(image, corners, c=(255,0,0)):
  face_idx = [[0,1,2,1],
              [2,1,2,3],
              [2,3,0,3],
              [0,3,0,1]]
  for ind_f in range(4):
    f = face_idx[ind_f]
    cv2.line(image, (corners[f[0]],corners[f[1]]),
                (corners[f[2]], corners[f[3]]), c, 1, lineType=cv2.LINE_AA)
  return image


def read_clib(calib_path):
  f = open(calib_path, 'r')
  for i, line in enumerate(f):
    if i == 0:
      calib = np.array(line[:-1].split(' ')[1:], dtype=np.float32)
      calib = calib.reshape(3, 3)
      return calib

def _bbox_to_coco_bbox(bbox):
  return [float(bbox[0]), float(bbox[1]),
          float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1])]


if __name__ == '__main__':
  argparser = argparse.ArgumentParser(
        description=__doc__)
  argparser.add_argument(
    '-R', '--orientation',
    action='store_true',
    help='if camera has its local orientation')
  argparser.add_argument(
    '-id',
    type=str,
    default="cam1",
    help='which camera id')
  args = argparser.parse_args()

  cats = ['Pedestrian', 'Car', 'Cyclist', 'Van', 'Truck',  'Person_sitting',
        'Tram', 'Misc', 'DontCare']
  cat_ids = {cat: i + 1 for i, cat in enumerate(cats)}
  
  IMG_H = 540
  IMG_W = 960

  #image = cv2.imread('/home/ubuntu/xwp/CenterNet/data/traffic_car/cam_sample/image_2/000053.png')
  image = cv2.imread('/home/ubuntu/xwp/datasets/multi_view_dataset/new/cam_sample/image_2/031973.png')
  calib = read_clib('./test_code/000000.txt')
  #anns = open('/home/ubuntu/xwp/CenterNet/data/traffic_car/cam_sample/label_2/000052.txt', 'r')
  anns = open('/home/ubuntu/xwp/datasets/multi_view_dataset/new/cam32/label_test/000973.txt', 'r')
  ori_anns = []
  ret = {'images': [], 'annotations': [], "categories": []}

  if  args.orientation:
    sensors_definition_file = '/home/ubuntu/xwp/CenterNet/carla_ros/dataset.json'
    if not os.path.exists(sensors_definition_file):
      raise RuntimeError(
          "Could not read sensor-definition from {}".format(sensors_definition_file))
    with open(sensors_definition_file) as handle:
      json_actors = json.loads(handle.read())

    for actor in json_actors["objects"]:
      actor_id = actor["id"]
      if actor_id == args.id:
        thecamera = actor
    spawn_point = thecamera.pop("spawn_point")
    rotation = Rotation(spawn_point.pop('yaw'),spawn_point.pop('roll'),spawn_point.pop('pitch'))
    location = Location(spawn_point.pop('x'),spawn_point.pop('y'),spawn_point.pop('z'))
    transform = Transform(rotation,location)
    sensor_world_matrix = get_matrix(transform)
    world_sensor_matrix = np.linalg.inv(sensor_world_matrix)

  for ann_ind, txt in enumerate(anns):
    tmp = txt[:-1].split(' ') #为了去掉末尾的\n
    cat_id = cat_ids[tmp[0]]
    truncated = int(float(tmp[1]))
    occluded = int(tmp[2])
    alpha = float(tmp[3])
    bbox = [float(tmp[4]), float(tmp[5]), float(tmp[6]), float(tmp[7])]
    dim = [float(tmp[8]), float(tmp[9]), float(tmp[10])]
    location = [float(tmp[11]), float(tmp[12]), float(tmp[13])]
    rotation_y = float(tmp[14])
    
    box_3d = compute_box_3d(dim, location, rotation_y)
    if  args.orientation:
      box_3d = np.dot(world_sensor_matrix,box_3d)
    box_2d = project_to_image(box_3d, calib)
    img_size = np.asarray([IMG_W,IMG_H],dtype=np.int)

    #犯错了，一开始用了location[0]，这里应该是像素坐标，应该用box2d的
    alpha,ray = _rot_y2alpha(rotation_y, box_2d[:,0][0:4].sum()/4, 
                                 calib[0, 2], calib[0, 0])

    bbox = (np.min(box_2d[:,0]), np.min(box_2d[:,1]), np.max(box_2d[:,0]), np.max(box_2d[:,1]))
    bbox_crop = tuple(max(0, b) for b in bbox)
    bbox_crop = (min(img_size[0], bbox_crop[0]),
                  min(img_size[0], bbox_crop[1]),
                  min(img_size[0], bbox_crop[2]),
                  min(img_size[1], bbox_crop[3]))
    # Detect if a cropped box is empty.
    if bbox_crop[0] >= bbox_crop[2] or bbox_crop[1] >= bbox_crop[3]:
      continue
    if location[2] < 2.0:
      continue

    ann = {#'image_id': image_id,
            #'id': int(len(ret['annotations']) + 1),
            'category_id': cat_id,
            'dim': dim,
            'bbox': _bbox_to_coco_bbox(bbox_crop),
            'depth': location[2],
            'alpha': alpha,
            'truncated': truncated,
            'occluded': occluded,
            'location': location,
            'rotation_y': rotation_y}
    ori_anns.append(ann)
    #ret['annotations'].append(ann)
    #box_3d = compute_box_3d(dim, location, rotation_y)
    #box_2d = project_to_image(box_3d, calib)
    #print('box_2d', box_2d)
    #print('bbox_crop',bbox_crop)
    #print('alpha: ',alpha)
    # print('alpha in degree: ',alpha * 180 / np.pi)
    # print('ray in defree: ',ray * 180 / np.pi)
    # #print(box_2d[:,0:3].sum()/4)
    # image = draw_box_3d(image, box_2d)
    # image = draw_box_2d(image, bbox_crop)
    # cv2.imshow('image', image)

    # cv2.waitKey()

  # Filter out bounding boxes outside the image
  visable_anns = []
  for i in range(len(ori_anns)):
    vis = True
    for j in range(len(ori_anns)):
      if ori_anns[i]['depth']  > \
          ori_anns[j]['depth']  and \
        _bbox_inside(ori_anns[i]['bbox'], ori_anns[j]['bbox']):
        vis = False
        break
    if vis:
      visable_anns.append(ori_anns[i])
    else:
      pass

  for ann in visable_anns:
    ret['annotations'].append(ann)
  
  print('len(ori_anns): ',len(ori_anns))
  print('len(vis_ann):',len(visable_anns))

  for ann in visable_anns:
    dim = ann['dim']
    location = ann['location']
    rotation_y = ann['rotation_y']
    box_3d = compute_box_3d(dim,location,rotation_y)
    box_2d = project_to_image(box_3d,calib)
    box_crop = list(ann['bbox'])
    print('ann[\'bbox\' ]',ann['bbox'])
    box_crop[2] = box_crop[0] + box_crop[2]
    box_crop[3] = box_crop[1] + box_crop[3]
    box_crop = [int(x) for x  in box_crop]
    image = draw_box_3d(image,box_2d)
    image = draw_box_2d(image, box_crop)
    print(bbox_crop)
    cv2.imshow('image',image)
    cv2.waitKey()

#TUDO:
#filter out the invisiable 3dbbox which in front of the camera with small distance(1m-2m), which covers more than a half image.
