NUM_CAM=8
cd /home/ubuntu/xwp/CenterNet
python /home/ubuntu/xwp/CenterNet/carla_ros/multiview_fusion_34_2.py -n ${NUM_CAM}
cp /home/ubuntu/xwp/datasets/multi_view_dataset/crowd_test2/label_test_trans/000001.txt /home/ubuntu/xwp/datasets/multi_view_dataset/crowd_test2/label_test_trans/000002.txt
cp /home/ubuntu/xwp/datasets/multi_view_dataset/crowd_test2/global_filtered/000001.txt /home/ubuntu/xwp/datasets/multi_view_dataset/crowd_test2/global_filtered/000002.txt

cd /home/ubuntu/xwp/3d_lidar_detection_evaluation

python nuscenes_eval.py --pred_labels /home/ubuntu/xwp/datasets/multi_view_dataset/crowd_test2/label_test_trans --gt_labels /home/ubuntu/xwp/datasets/multi_view_dataset/crowd_test2/global_filtered --format "class truncated occluded alpha bbox_xmin bbox_ymin bbox_xmax bbox_ymax h w l x y z r score" 
#python nuscenes_eval.py --pred_labels /home/ubuntu/xwp/datasets/multi_view_dataset/new/fuse_test/cam1cam18/label_test_trans --gt_labels /home/ubuntu/xwp/datasets/multi_view_dataset/new/fuse_test/cam1cam18/global_filtered --format "class truncated occluded alpha bbox_xmin bbox_ymin bbox_xmax bbox_ymax h w l x y z r score" 
