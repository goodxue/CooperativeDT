import os
import time
import glob
import numpy as np
import argparse
from detection_evaluation.nuscenes_eval_core import NuScenesEval
from detection_evaluation.label_parser import LabelParser
import co_utils as cu
import json

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
    from sko import SA

    def func_co(x):
        x.sort()
        Eval = NuScenesEval('', '', args.format)
        fused_data = cu.matching_and_fusion(cam_test_list[x[0]],cam_test_list[x[1]])
        fused_gt = cu.filt_gt_labels_tuple(cam_gt_list[x[0]],cam_gt_list[x[1]])
        mAP_temp = Eval.my_evaluate(fused_data,fused_gt)
        return 1- mAP_temp

    def fuse_constellation(x):
        #根据x的维度进行融合
        x.sort()
        size_n = x.shape[0]
        main_cam = x[0]
        gt_list = []
        gt_list.append(cam_gt_list[main_cam])
        for i in x[1:]:
            fused_data = cu.matching_and_fusion(cam_test_list[main_cam],cam_test_list[i]) #融合
            gt_list.append(cam_gt_list[i])
        fused_gt = cu.filt_gt_labels_tuple(*gt_list)
        Eval = NuScenesEval('', '', args.format)
        mAP_temp = Eval.my_evaluate(fused_data,fused_gt)
        return 1- mAP_temp

    filt_start_time = time.time()
    matrix = cu.fov_matrix(cam_transform)
    for i in range(0,34):
        matrix[i][i] = 0
    sl = matrix.flatten()
    idx = np. argpartition(sl, -15)[-15:]
    print(idx)
    
    # x0 = SA.get_new_constellation(np.array([0,1]))
    # sa = SA.SA_CO(func=fuse_constellation, x0=x0, T_max=1, T_min=0.4, L=40, max_stay_counter=10)
    # best_x, best_y = sa.run()
    # print('best_x:', best_x, 'best_y', 1-best_y)
    
    # filt_start_time = time.time()
    # ret = cu.filt_gt_labels(cam_gt_list[0],cam_gt_list[1])
    # filt_time = time.time() - filt_start_time
    # print("filt gt for 1 iter, time: ",filt_time)

    # max_map = 0
    # max_i,max_j = 0,0
    # for i in range(34):
    #     for j in range(i+1,34):
    #         fused_data = cu.matching_and_fusion(cam_test_list[i],cam_test_list[j]) #融合
    #         for k in range(j+1,34):
    #             Eval = NuScenesEval('', '', args.format)
    #             fused_data = cu.matching_and_fusion(fused_data,cam_test_list[k]) #融合
    #             fused_gt = cu.filt_gt_labels_tuple(cam_gt_list[i],cam_gt_list[j],cam_gt_list[k])
    #             #评估
    #             mAP_temp = Eval.my_evaluate(fused_data,fused_gt)
    #             if mAP_temp > max_map:
    #                 max_map = mAP_temp
    #                 max_i,max_j = i,j
    #                 print('temp max mAP: {}..........   time: ##   i: {}   j: {}  k:{} '.format(max_map,i,j,k))
    #             #print(mAP_temp)
    
    filt_time = time.time() - filt_start_time
    print('finished!,used {} s'.format(filt_time))


    

# 1.将所有test读取列表中
#循环2、3
# 2.融合
# 3.评估，迭代