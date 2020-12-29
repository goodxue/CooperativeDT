import numpy as np
import cv2
import os

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
               (corners[f[(j+1)%4], 0], corners[f[(j+1)%4], 1]), c, 2, lineType=cv2.LINE_AA)
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


if __name__ == '__main__':
  cats = ['Pedestrian', 'Car', 'Cyclist', 'Van', 'Truck',  'Person_sitting',
        'Tram', 'Misc', 'DontCare']
  cat_ids = {cat: i + 1 for i, cat in enumerate(cats)}
  
  IMG_H = 540
  IMG_W = 960

  image = cv2.imread('.\\test_code\\000023.png')
  calib = read_clib('.\\test_code\\000000.txt')
  anns = open('.\\test_code\\000023.txt', 'r')
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
    box_2d = project_to_image(box_3d, calib)
    img_size = np.asarray([IMG_W,IMG_H],dtype=np.int)
    alpha = _rot_y2alpha(rotation_y, location[0], 
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

    ann = {#'image_id': image_id,
            #'id': int(len(ret['annotations']) + 1),
            'category_id': cat_id,
            'dim': dim,
            #'bbox': _bbox_to_coco_bbox(bbox_crop),
            'depth': location[2],
            'alpha': alpha,
            'truncated': truncated,
            'occluded': occluded,
            'location': location,
            'rotation_y': rotation_y}
    #ret['annotations'].append(ann)
    #box_3d = compute_box_3d(dim, location, rotation_y)
    #box_2d = project_to_image(box_3d, calib)
    #print('box_2d', box_2d)
    print('bbox_crop',bbox_crop)
    #print('alpha: ',alpha)
    print('alpha in degree: ',alpha * 180 / np.pi)
    image = draw_box_3d(image, box_2d)
    image = draw_box_2d(image, bbox_crop)
    cv2.imshow('image', image)

    cv2.waitKey()