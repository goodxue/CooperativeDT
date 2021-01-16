cd src
# train
python main.py ddd --exp_id traffic_car_cam1 --dataset traffic_car  --cam cam2 --batch_size 32 --master_batch 8 --num_epochs 110 --lr_step 105,108 --gpus 0,1,2,3 --lr 1.25e-5 --dim_weight 1 --resume
# test
#python test.py ddd --exp_id traffic_car_cam1 --dataset traffic_car --resume --cam cam1
cd ..
