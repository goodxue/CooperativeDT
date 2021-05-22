import json
import numpy as np
#import cv2

cats = ['Car','DontCare']
cat_ids = {'DontCare':0,'Car':2}

DATASET_PATH = '/home/ubuntu/xwp/datasets/multi_view_dataset/new/fuse_test/cam4+cam16+cam20+cam21/'
DEBUG = False
# VAL_PATH = DATA_PATH + 'training/label_val/'
import os

ann_dir = DATASET_PATH + 'label_test_trans/'
output_dir = DATASET_PATH + 'label_test_tracking/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
ann_list = os.listdir(ann_dir)
ann_list.sort(key=lambda x:int(x[:-4]))
#calib_dir = DATASET_PATH + 'calib/'
# splits = ['trainval', 'test']
#calib_path = calib_dir + '000000.txt'
#calib = read_clib(calib_path)
out_path = output_dir + '{}'.format('0000.txt')
f = open(out_path, 'w')

for label_file in ann_list:
    ann_path = ann_dir + '{}'.format(label_file)   
    anns = open(ann_path, 'r')
    frame = int(label_file[:-4])-901
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
        score = float(tmp[15])
        #rotation = rotation_y * np.pi / 180
        #box_3d = compute_box_3d(dim, location, rotation_y)
        #box_2d = project_to_image(box_3d, calib)
        #bbox = (np.min(box_2d[:,0]), np.min(box_2d[:,1]), np.max(box_2d[:,0]), np.max(box_2d[:,1]))

        # txt="{} {} {} {} {} {} {} {} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {} {}\n".format('Car', truncated, occluded, alpha, bbox[0], bbox[1], bbox[2], bbox[3], dim[0], dim[1],
        #                     dim[2],location[0], location[1], location[2],  rotation,car_id)
        txt="{},{},{},{},{},{},{},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{},{} \n".format(frame,2,0,0,0,0,score,
                                dim[0],dim[1],dim[2],location[1],-location[2],location[0],rotation_y,0)
        f.write(txt)
        #f.write('\n')
f.close()

print('finish!')