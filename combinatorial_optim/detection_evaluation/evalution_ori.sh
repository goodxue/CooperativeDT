CAM_NUM=1
#cd /home/ubuntu/xwp/CenterNet
#python /home/ubuntu/xwp/CenterNet/carla_ros/multiview_fusion_34.py -n ${NUM_CAM}

#cd /home/ubuntu/xwp/3d_lidar_detection_evaluation
python nuscenes_eval.py --pred_labels /home/ubuntu/xwp/datasets/multi_view_dataset/new/fuse_test/cam1+cam10+cam18/label_test_trans/   --gt_labels /home/ubuntu/xwp/datasets/multi_view_dataset/new/global_label_new --format "class truncated occluded alpha bbox_xmin bbox_ymin bbox_xmax bbox_ymax h w l x y z r score" 
