from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import json
import numpy as np
import cv2
DATASET_PATH = '/home/ubuntu/xwp/datasets/multi_view_dataset/'
DEBUG = False
# VAL_PATH = DATA_PATH + 'training/label_val/'
import os
SPLITS = ['3dop', 'subcnn'] 
import _init_paths
from utils.ddd_utils import compute_box_3d, project_to_image, alpha2rot_y
from utils.ddd_utils import draw_box_3d, unproject_2d_to_3d, draw_box_2d

'''
#Values    Name      Description
----------------------------------------------------------------------------
   1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                     'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                     'Misc' or 'DontCare'
   1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                     truncated refers to the object leaving image boundaries
   1    occluded     Integer (0,1,2,3) indicating occlusion state:
                     0 = fully visible, 1 = partly occluded
                     2 = largely occluded, 3 = unknown
   1    alpha        Observation angle of object, ranging [-pi..pi]
   4    bbox         2D bounding box of object in the image (0-based index):
                     contains left, top, right, bottom pixel coordinates
   3    dimensions   3D object dimensions: height, width, length (in meters)
   3    location     3D object location x,y,z in camera coordinates (in meters)
   1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
   1    score        Only for results: Float, indicating confidence in
                     detection, needed for p/r curves, higher is better.
'''
def _bbox_inside(box1, box2):
  #coco box
  return box1[0] > box2[0] and box1[0] + box1[2] < box2[0] + box2[2] and \
         box1[1] > box2[1] and box1[1] + box1[3] < box2[1] + box2[3]

def _bbox_to_coco_bbox(bbox):
  return [float(bbox[0]), float(bbox[1]),
          float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1])]

def _rot_y2alpha(rot_y, x, cx, fx):
    """
    Get rotation_y by alpha + theta - 180
    alpha : Observation angle of object, ranging [-pi..pi]
    x : Object center x to the camera center (x-W/2), in pixels
    rotation_y : Rotation ry around Y-axis in camera coordinates [-pi..pi]
    """
    alpha = rot_y - np.arctan2(x - cx, fx)
    if alpha > np.pi:
      alpha -= 2 * np.pi
    if alpha < -np.pi:
      alpha += 2 * np.pi
    return alpha

def read_clib(calib_path): #change 3x3 to 3x4
  f = open(calib_path, 'r')
  for i, line in enumerate(f):
    if i == 0:
      calib = np.array(line[:-1].split(' ')[1:], dtype=np.float32)
      calib = calib.reshape(3, 3)
      calib = np.concatenate([calib,np.zeros((3,1),dtype=np.float32)],axis=1)
      return calib

# cats = ['Pedestrian', 'Car', 'Cyclist', 'Van', 'Truck',  'Person_sitting',
#         'Tram', 'Misc', 'DontCare']
cats = ['Car','DontCare']

cat_ids = {cat: i + 1 for i, cat in enumerate(cats)}
# cat_info = [{"name": "pedestrian", "id": 1}, {"name": "vehicle", "id": 2}]
F = 721
H = 384 # 375
W = 1248 # 1242
EXT = [45.75, -0.34, 0.005]
CALIB = np.array([[F, 0, W / 2, EXT[0]], [0, F, H / 2, EXT[1]], 
                  [0, 0, 1, EXT[2]]], dtype=np.float32)

#一个cam共10000张图，选8000张训练，1000张验证，1000张测试
#cam_sample共70000张图，每个10000中6000训练，2000验证，2000测试
TRAIN_NUM = 400
VAL_NUM = 50
TEST_NUM = 50
TRAIN_SETS = ['cam_sample']
TEST_SETS = ['cam7']
IMG_H = 540
IMG_W = 960
TRAIN_SAMPLE = [range(1,TRAIN_NUM+1),range(501,TRAIN_NUM+501)]
VAL_SAMPLE = [range(TRAIN_NUM+1,TRAIN_NUM+1+VAL_NUM),range(TRAIN_NUM+501,TRAIN_NUM+501+VAL_NUM)]
TEST_SAMPLE = [range(TRAIN_NUM+1+VAL_NUM,501),range(TRAIN_NUM+501+VAL_NUM,1001)]

cat_info = []
for i, cat in enumerate(cats):
  cat_info.append({'name': cat, 'id': i + 1})

for CAM in TRAIN_SETS:
  DATA_PATH = DATASET_PATH + '{}/'.format(CAM)
  image_set_path = DATA_PATH + 'image_2/'
  ann_dir = DATA_PATH + 'label_2/'
  calib_dir = DATA_PATH + 'calib/'
  # splits = ['trainval', 'test']
  calib_path = calib_dir + '000000.txt'
  calib = read_clib(calib_path)
  newlabel_dir = DATA_PATH + 'label_new/'
  if not os.path.exists(newlabel_dir):
    os.mkdir(newlabel_dir)

  splits = {'train':TRAIN_SAMPLE, 'val':VAL_SAMPLE,'test':TEST_SAMPLE}
  for split in splits:
    ret = {'images': [], 'annotations': [], "categories": cat_info}
    image_set = []
    for aset in splits[split]:
        image_set += [str(i).rjust(6,'0') for i in aset]
    #image_set = open(image_set_path + '{}.txt'.format(split), 'r')
    image_to_id = {}
    for line in image_set:
      image_id = int(line)
      
      
      image_info = {'file_name': '{}.png'.format(line),
                    'id': int(image_id),
                    'calib': calib.tolist()}
      ret['images'].append(image_info)
      # if split == 'test':
      #   continue
      ann_path = ann_dir + '{}.txt'.format(line)
      # if split == 'val':
      #   os.system('cp {} {}/'.format(ann_path, VAL_PATH))
      anns = open(ann_path, 'r')
      
      if DEBUG:
        image = cv2.imread(
          DATA_PATH + 'images/trainval/' + image_info['file_name'])

      out_path = os.path.join(newlabel_dir, '{:06d}.txt'.format(image_id))
      f = open(out_path, 'w')
      ori_anns = []
      for ann_ind, txt in enumerate(anns):
        tmp = txt[:-1].split(' ')
        cat_id = cat_ids[tmp[0]]
        truncated = int(float(tmp[1]))
        occluded = int(tmp[2])
        alpha = float(tmp[3])
        bbox = [float(tmp[4]), float(tmp[5]), float(tmp[6]), float(tmp[7])]
        dim = [float(tmp[8]), float(tmp[9]), float(tmp[10])]
        location = [float(tmp[11]), float(tmp[12]), float(tmp[13])]
        rotation_y = float(tmp[14])

        # #由于label中存在没有在图像像素范围内的框，因此当投影到图片上的3D框顶点超出个数大于6则忽略这个标签
        box_3d = compute_box_3d(dim, location, rotation_y)
        box_2d = project_to_image(box_3d, calib)
        img_size = np.asarray([IMG_W,IMG_H],dtype=np.int)
        # inds = np.greater(box_2d,img_size)
        # out_num = (inds[:,0] * inds[:,1]).sum()
        # if out_num > 6:
        #   continue

        #根据3d框中心以及内参矩阵计算alpha,注意location是3dbox底面中心（但是只用x，所以不用计算）
        alpha = _rot_y2alpha(rotation_y, box_2d[:,0][0:4].sum()/4, 
                                 calib[0, 2], calib[0, 0])
        
        #计算像素坐标系下的2dbbox，就是3dbbox每个轴最小的值组成的框。裁剪到图像上。
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

        ann = {'image_id': image_id,
               'id': int(len(ret['annotations']) + 1),
               'category_id': cat_id,
               'dim': dim,
               'bbox': _bbox_to_coco_bbox(bbox_crop),
               'depth': location[2],
               'alpha': float(alpha),
               'truncated': truncated,
               'occluded': occluded,
               'location': location,
               'rotation_y': rotation_y}
        ori_anns.append(ann)
        # Filter out bounding boxes outside the image
        visable_anns = []
        for i in range(len(ori_anns)):
          vis = True
          for j in range(len(ori_anns)):
            if ori_anns[i]['depth'] > \
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
        
        f.write('{} 0.0 0'.format('Car'))
        f.write(' {:.2f}'.format(ann['alpha']))
        for tmp in bbox:
          f.write(' {:.2f}'.format(tmp))
        for tmp in dim:
          f.write(' {:.2f}'.format(tmp))
        for tmp in location:
          f.write(' {:.2f}'.format(tmp))
        f.write(' {:.2f}'.format(rotation_y))
        f.write('\n')


        if DEBUG and tmp[0] != 'DontCare':
          #box_3d = compute_box_3d(dim, location, rotation_y)
          #box_2d = project_to_image(box_3d, calib)
          # print('box_2d', box_2d)
          image = draw_box_3d(image, box_2d)
          image = draw_box_2d(image, bbox_crop)
          x = (bbox[0] + bbox[2]) / 2
          '''
          print('rot_y, alpha2rot_y, dlt', tmp[0], 
                rotation_y, alpha2rot_y(alpha, x, calib[0, 2], calib[0, 0]),
                np.cos(
                  rotation_y - alpha2rot_y(alpha, x, calib[0, 2], calib[0, 0])))
          '''
          depth = np.array([location[2]], dtype=np.float32)
          pt_2d = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2],
                            dtype=np.float32)
          pt_3d = unproject_2d_to_3d(pt_2d, depth, calib)
          pt_3d[1] += dim[0] / 2
          print('pt_3d', pt_3d)
          print('location', location)
      f.close()
      if DEBUG:
        cv2.imshow('image', image)
        cv2.waitKey()
    #1数据集文件格式
    #2投影到图片，超出的截去
    #3划分训练测试集

    print("# images: ", len(ret['images']))
    print("# annotations: ", len(ret['annotations']))
    # import pdb; pdb.set_trace()
    if not os.path.exists('{}annotations'.format(DATA_PATH)):
      os.mkdir('{}annotations'.format(DATA_PATH))
    out_path = '{}annotations/multiview_{}.json'.format(DATA_PATH,split)
    json.dump(ret, open(out_path, 'w'))
  
