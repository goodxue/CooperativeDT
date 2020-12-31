cd src
# train
python main.py ddd --exp_id traffic_car_cam1 --dataset traffic_car  --cam cam6 --batch_size 32 --master_batch 8 --num_epochs 98 --lr_step 92,96 --gpus 0,1,2,3 --lr 1.25e-4 --dim_weight 1 --resume
# test
#python test.py ddd --exp_id traffic_car_cam1 --dataset traffic_car --resume --cam cam1
cd ..
