cd src
# train
python main.py ddd --exp_id traffic_car_cam1 --dataset traffic_car  --cam cam1 --batch_size 128 --master_batch 112 --num_epochs 20 --lr_step 15,30 --gpus 0,1,2,3
# test
#python test.py ddd --exp_id 3dop --dataset kitti --kitti_split 3dop --resume
cd ..