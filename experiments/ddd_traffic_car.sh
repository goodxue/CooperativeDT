cd src
# train
python main.py ddd --exp_id traffic_car_cam1 --dataset traffic_car  --cam cam4 --batch_size 32 --master_batch 8 --num_epochs 90 --lr_step 83,88 --gpus 0,1,2,3 --lr 1.25e-4 --dim_weight 1 --resume
# test
#python test.py ddd --exp_id 3dop --dataset kitti --kitti_split 3dop --resume
cd ..
