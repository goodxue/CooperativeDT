import os
import time
import glob
import numpy as np
import argparse
import json
from detection_evaluation.nuscenes_eval_core import NuScenesEval
from detection_evaluation.label_parser import LabelParser
import co_utils as cu
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

def parse_args():
    parser = argparse.ArgumentParser(description='arg parser')
    # parser.add_argument('--pred_labels', type=str, required=True,
    #                     help='Prediction labels data path')
    # parser.add_argument('--gt_labels', type=str, required=True,
    #                     help='Ground Truth labels data path')
    parser.add_argument('--format', type=str, default='class truncated occluded alpha bbox_xmin bbox_ymin bbox_xmax bbox_ymax h w l x y z r score')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    NuScenesEval(args.pred_labels, args.gt_labels, args.format)


if __name__ == '__main__':
    args = parse_args()
    file_parsing = LabelParser(args.format)
    FUSE_NUM = 2

    pred_files_list = []
    dataset_path = '/home/ubuntu/xwp/datasets/multi_view_dataset/new' #数据集根目录
    gt_global_label_dir = '/home/ubuntu/xwp/datasets/multi_view_dataset/new/global_label_new' #全部gt的世界坐标label文件夹
    camset_path = [ os.path.join(dataset_path,"cam{}".format(cam_num),'label_test_trans') for cam_num in range(1,35)] #每一个相机的test txt文件夹
    gtset_path = [ os.path.join(dataset_path,"cam{}".format(cam_num),'global_filtered') for cam_num in range(1,35)] #每一个相机的test txt文件夹

    cam_test_list = [] #所有相机单独检测的世界坐标 len=34,len(cam_test_list[0])=100 type(cam_test_list[0]) =np.ndarray shape = N*9(score) / N*8(gt)
    cam_gt_list = []
    load_start_time = time.time()
    for i,pred_path in enumerate(camset_path):
        pred_file_list = glob.glob(pred_path + "/*")
        pred_file_list.sort()
        if len(pred_file_list) != 100:
            print(len(pred_file_list))
            raise RuntimeError("can\'t read 100 files in cam{}. check the prediction file!".format(i+1))
        frame_test_list = []
        for pred_fn in pred_file_list:
            predictions = file_parsing.parse_label(pred_fn, prediction=True)
            frame_test_list.append(predictions[:,1:])
        cam_test_list.append(frame_test_list)
    # test sample 加载完成

    cam_transform = []
    sensors_definition_file = '/home/ubuntu/xwp/CenterNet/carla_ros/dataset.json'
    if not os.path.exists(sensors_definition_file):
        raise RuntimeError(
            "Could not read sensor-definition from {}".format(sensors_definition_file))
    with open(sensors_definition_file) as handle:
        json_actors = json.loads(handle.read())
    global_sensors = []
    metric_map = []
    metric_ate = []
    for actor in json_actors["objects"]:
        global_sensors.append(actor)
    for sensor_spec in global_sensors:
        sensor_id = str(sensor_spec.pop("id"))
        spawn_point = sensor_spec.pop("spawn_point")
        point = cu.Transform(location=cu.Location(x=spawn_point.pop("x"), y=-spawn_point.pop("y"), z=spawn_point.pop("z")),
                rotation=cu.Rotation(pitch=-spawn_point.pop("pitch", 0.0), yaw=-spawn_point.pop("yaw", 0.0), roll=spawn_point.pop("roll", 0.0)))
        cam_transform.append(point)
        metric_map.append(float(sensor_spec.pop("mAP")))
        metric_ate.append(float(sensor_spec.pop("ATE")))

    # #加载gt
    # cam_gt_list = []
    # gt_file_list = glob.glob(gt_global_label_dir + "/*")
    # gt_file_list.sort()
    # for gt_fn in gt_file_list:
    #     if int(gt_fn[-10:-4]) < 901:
    #         continue
    #     gts = file_parsing.parse_label(gt_fn, prediction=False)
    #     cam_gt_list.append(gts[:,1:])
    # #
    for i,gt_path in enumerate(gtset_path):
        gt_file_list = glob.glob(gt_path + "/*")
        gt_file_list.sort()
        frame_gt_list = []
        for gt_fn in gt_file_list:
            if int(gt_fn[-10:-4]) < 901:
                continue
            gts = file_parsing.parse_label(gt_fn, prediction=False)
            frame_gt_list.append(gts[:,1:])
        cam_gt_list.append(frame_gt_list)

    load_time = time.time() - load_start_time
    print("load time: ",load_time)

    #遍历融合
    
    filt_start_time = time.time()
    fused_gt = cu.filt_gt_labels_tuple(*cam_gt_list)
    #ret = cu.filt_gt_labels(cam_gt_list[0],cam_gt_list[1])
    #filt_time = time.time() - filt_start_time
    #print("filt gt for 1 iter, time: ",filt_time)
    filt_start_time = time.time()
    X = cu.matching_and_fusion_tuple(*cam_test_list)
    data = np.stack([X[1][:,0],X[1][:,1]]).transpose()
    result = DBSCAN(eps = 1.6,min_samples=1).fit(data)
    y_pred = result.labels_ #每个元素的标签，同一聚类下的元素标签相同
    label_set = set(y_pred)

    filtered_preds = np.empty((0, 8))
    for lb in label_set:
        filter_n = np.asarray([lb])
        objs_in_cluster = X[1][np.in1d(y_pred, filter_n)]
        filtered_obj = cu.mean_fusion(objs_in_cluster)
        filtered_preds = np.vstack((filtered_preds, filtered_obj))

    #print(X[0][:,0:2])
    #print(fused_gt[0][0,:])
    #print(y_pred)
    print("gt_num: ",len(fused_gt[1]))
    #print(len(X[0]))
    #print(X[0][:,0].shape)
    #print(X[0].shape)
    #print(data.shape)
    print("cluster_num: ",len(label_set)-(1 if -1 in y_pred else 0))

    Eval = NuScenesEval('', '', args.format)
    mAP_temp = Eval.my_evaluate([filtered_preds],[fused_gt[1]])
    print('Evaluation: ',mAP_temp)
    
    fused_data = cu.matching_and_fusion(fused_data,cam_test_list[k])
    Eval = NuScenesEval('', '', args.format)
    mAP_temp = Eval.my_evaluate(fused_data[1],fused_gt[1])
    print('Evaluation2: ',mAP_temp)

    plt.scatter(data[:,0], data[:,1], marker='o',c=y_pred)
    #plt.show()
    max_map = 0
    max_i,max_j = 0,0
    # for i in range(34):
    #     for j in range(i+1,34):
    #         fused_data = cu.matching_and_fusion(cam_test_list[i],cam_test_list[j]) #融合
    #         for k in range(j+1,34):
    #             fused_data = cu.matching_and_fusion(fused_data,cam_test_list[k]) #融合
    #             for z in range(k+1,34):
    #                 Eval = NuScenesEval('', '', args.format)
    #                 fused_data = cu.matching_and_fusion(fused_data,cam_test_list[z]) #融合
    #                 fused_gt = cu.filt_gt_labels_tuple(cam_gt_list[i],cam_gt_list[j],cam_gt_list[k],cam_gt_list[z])
    #                 #评估
    #                 mAP_temp = Eval.my_evaluate(fused_data,fused_gt)
    #                 if mAP_temp > max_map:
    #                     max_map = mAP_temp
    #                     max_i,max_j = i,j
    #                     print('temp max mAP: {}..........   time: ##   i: {}   j: {}  k:{}  z:{}'.format(max_map,i,j,k,z))
                #print(mAP_temp)
    # max_map = 0
    # max_i,max_j = 0,0
    # fused_gt = cu.filt_gt_labels_tuple(*cam_gt_list)
    # for i in range(34):
    #     for j in range(i+1,34):
    #         fused_data = cu.matching_and_fusion(cam_test_list[i],cam_test_list[j]) #融合
    #         Eval = NuScenesEval('', '', args.format)
    #         #fused_gt = cu.filt_gt_labels_tuple(cam_gt_list[i],cam_gt_list[j],cam_gt_list[k])
    #         #评估
    #         mAP_temp = Eval.my_evaluate(fused_data,fused_gt)
    #         if mAP_temp > max_map:
    #             max_map = mAP_temp
    #             max_i,max_j = i,j
    #             print('temp max mAP: {}..........   time: ##   i: {}   j: {}  '.format(max_map,i,j))
                #print(mAP_temp)
    # max_map = 0
    # max_i,max_j = 0,0
    # fused_gt = cu.filt_gt_labels_tuple(*cam_gt_list)
    # for i in range(34):
    #     for j in range(i+1,34):
    #         fused_data = cu.matching_and_fusion(cam_test_list[i],cam_test_list[j]) #融合
    #         for k in range(j+1,34):
    #             Eval = NuScenesEval('', '', args.format)
    #             fused_data1 = cu.matching_and_fusion(fused_data,cam_test_list[k]) #融合
    #             #fused_gt = cu.filt_gt_labels_tuple(cam_gt_list[i],cam_gt_list[j],cam_gt_list[k])
    #             #评估
    #             mAP_temp = Eval.my_evaluate(fused_data1,fused_gt)
    #             if mAP_temp > max_map:
    #                 max_map = mAP_temp
    #                 max_i,max_j = i,j
    #                 print('temp max mAP: {}..........   time: ##   i: {}   j: {}  k:{} '.format(max_map,i,j,k))
    #             #print(mAP_temp)
    # for i in [1]:
    #     for j in [26]:
    #         #fused_data = cu.matching_and_fusion(cam_test_list[i],cam_test_list[j],fusion_fuction=1) #融合
    #         fused_data = cu.fov_match_and_fusion(cam_test_list[i],cam_test_list[j],cam_transform[i],cam_transform[j],metric_map[i]>=metric_map[j])
    #         for k in [10]:
    #             fused_data = cu.fov_match_and_fusion(fused_data,cam_test_list[k],cam_transform[i],cam_transform[k],metric_map[i]>=metric_map[k])
    #             #fused_data = cu.matching_and_fusion(fused_data,cam_test_list[k],fusion_fuction=1) #融合
    #             for z in [18]:
    #                 fused_data = cu.fov_match_and_fusion(fused_data,cam_test_list[z],cam_transform[i],cam_transform[z],metric_map[i]>=metric_map[z])
    #                 #fused_data = cu.matching_and_fusion(fused_data,cam_test_list[z],fusion_fuction=1) #融合
    #                 for thre in [0,0.3,0.5]:
    #                     Eval = NuScenesEval('', '', args.format,score_threshold=thre)
                        
    #                     #fused_data = cu.fov_match_and_fusion(cam_test_list[i],cam_test_list[j],cam_transform[i],cam_transform[j],metric_map[i]>=metric_map[j])
    #                     fused_gt = cu.filt_gt_labels_tuple(cam_gt_list[i],cam_gt_list[j],cam_gt_list[k],cam_gt_list[z])
    #                     #评估
    #                     mAP_temp = Eval.my_evaluate(fused_data,fused_gt)
    #                     # if mAP_temp > max_map:
    #                     #     max_map = mAP_temp
    #                     #     max_i,max_j = i,j
    #                     print('temp max mAP: {}..........   time: ##   i: {}   j: {}  k:{} '.format(mAP_temp,i,j,k))
                #print(mAP_temp)
    def get_gt_size(gt):
        ret = 0
        for i in gt:
            ret += len(i)
        return ret
    
    
    
    # for i in range(34):
    #     for j in range(i+1,34):
    #         fused_data = cu.matching_and_fusion(cam_test_list[i],cam_test_list[j],fusion_fuction=1) #融合
    #         #fused_data = cu.fov_match_and_fusion(cam_test_list[i],cam_test_list[j],cam_transform[i],cam_transform[j],metric_map[i]>=metric_map[j]) #融合
    #         Eval = NuScenesEval('', '', args.format,score_threshold=0)
    #         #fused_gt = cu.filt_gt_labels_tuple(cam_gt_list[i],cam_gt_list[j])
    #         #评估
    #         mAP_temp = Eval.my_evaluate(fused_data,fused_gt,)
    #         if mAP_temp > max_map:
    #             max_map = mAP_temp
    #             max_i,max_j = i,j
    #             print('temp max mAP: {}..........   time: ##   i: {}   j: {}  '.format(max_map,i,j))
            #print(mAP_temp)
    # for i in [7]:
    #     for j in [18]:
    #         for thre in [0]:
    #             fused_data = cu.matching_and_fusion(cam_test_list[i],cam_test_list[j],fusion_fuction=None) #融合
    #             #fused_data = cu.fov_match_and_fusion(cam_test_list[i],cam_test_list[j],cam_transform[i],cam_transform[j],metric_map[i]>=metric_map[j]) #融合
    #             Eval = NuScenesEval('', '', args.format,score_threshold=thre)
    #             fused_gt = cu.filt_gt_labels_tuple(cam_gt_list[i],cam_gt_list[j])
    #             #评估
    #             mAP_temp = Eval.my_evaluate(fused_data,fused_gt)
    #             # if mAP_temp > max_map:
    #             #     max_map = mAP_temp
    #             #     max_i,max_j = i,j
    #             print('temp max mAP: {}..........   time: ##   i: {}   j: {}  thre:{}'.format(mAP_temp,i,j,thre))
    #             print('gt size:',get_gt_size(fused_gt))
    #             print(mAP_temp)
    filt_time = time.time() - filt_start_time
    print('finished!,used {} s'.format(filt_time))


    

# 1.将所有test读取列表中
#循环2、3
# 2.融合
# 3.评估，迭代