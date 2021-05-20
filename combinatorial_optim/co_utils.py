import glob
import os
import sys
import json
import numpy as np
#import fov_utils as fu
from fov_utils import FOV, _polygon_contains_point

class Rotation(object):
    def __init__(self,yaw=0,roll=0,pitch=0):
        self.yaw = yaw
        self.roll = roll
        self.pitch = pitch

class Location(object):
    def __init__(self,x,y,z):
        self.x = x
        self.y = y
        self.z = z

class Transform(object):
    def __init__(self,rotation,location):
        self.rotation = rotation
        self.location = location
    

class CamVehicle(object):
    def __init__(self,x,y,z,dh,dw,dl,ry,cid=-1,score=0):
        self.x = x
        self.y = y
        self.z = z
        self.height = dh
        self.width = dw
        self.length = dl
        self.rotation_y = ry
        self.id = cid
        self.score = score
    
    @classmethod
    def by_location(self,location,dh,dw,dl,ry):
        return self(location.x,location.y,location.z,dh,dw,dl,ry)
    
    @classmethod
    def by_box3d(self,box3d):
        point1 = box3d[0,:]
        point5 = box3d[4,:]
        x = (point1[0]+point5[0])/2
        h = point1[1]-point5[1] 
        y = h+point5[1]
        z = (point1[2] - point5[2])/2
        return self((x,y,z,h,2*point1[0]))

    def compute_box_3d(self):
        # dim: 3
        # location: 3
        # rotation_y: 1
        # return: 8 x 3
        c, s = np.cos(self.rotation_y), np.sin(self.rotation_y)
        R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
        l, w, h = self.length, self.width, self.height
        x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
        y_corners = [0,0,0,0,-h,-h,-h,-h]
        z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]

        corners = np.array([x_corners, y_corners, z_corners], dtype=np.float32)
        corners_3d = np.dot(R, corners) 
        corners_3d = corners_3d + np.array([self.x,self.y,self.z], dtype=np.float32).reshape(3, 1)
        return corners_3d.transpose(1, 0)


def get_sensors_transform_dict(definition_file):
    if not os.path.exists(sensors_definition_file):
        raise RuntimeError(
            "Could not read sensor-definition from {}".format(sensors_definition_file))
    with open(sensors_definition_file) as handle:
        json_actors = json.loads(handle.read())

    cam_transform = {}
    global_sensors = []
    for actor in json_actors["objects"]:
        global_sensors.append(actor)
    for sensor_spec in global_sensors:
        sensor_id = str(sensor_spec.pop("id"))
        spawn_point = sensor_spec.pop("spawn_point")
        point = Transform(location=Location(x=spawn_point.pop("x"), y=-spawn_point.pop("y"), z=spawn_point.pop("z")),
                rotation=Rotation(pitch=-spawn_point.pop("pitch", 0.0), yaw=-spawn_point.pop("yaw", 0.0), roll=spawn_point.pop("roll", 0.0)))
        cam_transform[sensor_id] = point
    
    return cam_transform

def load_test_data(root_dir,num):
    pass


#不用管gt和pred的意思，就是用贪心算法将两个array中最近的距离关联
def match_pairs(gt_label,pred_label,fusion_function=None,distance_threshold_sq=1,*args):
    true_preds = np.empty((0, 8))
    unmatched1 = np.empty((0, 8))
    unmatched2 = np.empty((0, 8))
    #corresponding_gt = np.empty((0, 7))
    #result_score = np.empty((0, 2))
    # Initialize matching loop
    match_incomplete = True
    while match_incomplete and gt_label.shape[0] > 0:
        match_incomplete = False
        for gt_idx, single_gt_label in enumerate(gt_label):
            # Check is any prediction is in range
            distance_sq_array = (single_gt_label[0] - pred_label[:, 0])**2 + (single_gt_label[1] - pred_label[:, 1])**2
            # If there is a prediction in range, pick closest
            if np.any(distance_sq_array < distance_threshold_sq):
                min_idx = np.argmin(distance_sq_array)
                # Store true prediction
                if fusion_function == None:
                    true_preds = np.vstack((true_preds, pred_label[min_idx, :].reshape(-1, 1).T))
                else:
                    temp = ((single_gt_label+pred_label[min_idx,:])/2).reshape(-1, 1).T
                    true_preds = np.vstack((true_preds, temp))
                    #true_preds = fusion_function(pred_label[min_idx, :],gt_label[gt_idx],args)

                #corresponding_gt = np.vstack((corresponding_gt, gt_label[gt_idx]))
                # Store score for mAP
                #result_score = np.vstack((result_score, np.array([[1, pred_label[min_idx, 7]]])))
                # Remove prediction and gt then reset loop
                pred_label = np.delete(pred_label, obj=min_idx, axis=0)
                gt_label = np.delete(gt_label, obj=gt_idx, axis=0)
                match_incomplete = True
                break

    # If there were any false detections, add them.
    if pred_label.shape[0] > 0:
        unmatched2 = np.vstack((unmatched2,pred_label[:]))
    if gt_label.shape[0] > 0:
        #print(gt_label)
        unmatched1 = np.vstack((unmatched1,gt_label[:]))
        # false_positives = np.zeros((pred_label.shape[0], 2))
        # false_positives[:, 1] = pred_label[:, 7]
        # result_score = np.vstack((result_score, false_positives))
    return true_preds, unmatched1, unmatched2


def matching_and_fusion(pred1,pred2,fusion_fuction=None):
    ret = []
    for ind,(frame_det1,frame_det2) in enumerate(zip(pred1,pred2)):
        matched, unmatched1, unmatched2 = match_pairs(frame_det1.astype(np.float), frame_det2.astype(np.float),fusion_fuction)
        ret.append(np.vstack((matched,unmatched1,unmatched2)))
    return ret

def filt_gt_labels(gt_list1,gt_list2):
    if len(gt_list1) != len(gt_list2):
        raise RuntimeError("The length of two gt list does not match! check the dataloader")
    ret = []
    for ind, (gt1,gt2) in enumerate(zip(gt_list1,gt_list2)):
        all_gt = np.vstack((gt1,gt2))
        ret.append(all_gt[np.unique(all_gt[:,7].astype(np.int),return_index=True)[1]])
    return ret

def filt_gt_labels_tuple(*gt_list_tuple):
    if len(gt_list_tuple) == 0:
        raise RuntimeError("input should be more than 1!")
    
    ret = gt_list_tuple[0]
    for gt_list in gt_list_tuple[1:]:
        ret_temp = []
        for ind,(gt1,gt2) in enumerate(zip(ret,gt_list)):
            all_gt = np.vstack((gt1,gt2))
            ret_temp.append(all_gt[np.unique(all_gt[:,7].astype(np.int),return_index=True)[1]]) #去重
        ret = ret_temp
    return ret

def fov_match_and_fusion(pred1,pred2,point1,point2,trust_first=True,fusion_fuction=None):
    ret = []
    fov = FOV(point1).caculate_iou(FOV(point2))
    for ind,(frame_det1,frame_det2) in enumerate(zip(pred1,pred2)):
        trust_frame = frame_det1 if trust_first else frame_det2
        matched, unmatched1, unmatched2 = match_pairs(frame_det1.astype(np.float), frame_det2.astype(np.float))
        unmatched = np.vstack((unmatched1,unmatched2))
        delete_index = []
        #print(ind,':',len(unmatched))
        for ind_un,car_point in enumerate(unmatched):
            if _polygon_contains_point(fov,(car_point[0],car_point[1])):
                if car_point.tolist() in trust_frame.tolist():
                    continue
                else:
                    delete_index.append(ind_un)
        tmp = np.delete(unmatched,obj=delete_index,axis=0)
        ret.append(np.vstack((matched,tmp)))
        #ret.append(unmatched)
        #print(len(tmp))
    return ret
