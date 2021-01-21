cd src
# train
python main.py ddd --exp_id traffic_car_total --dataset traffic_car  --cam cam_sample --batch_size 32 --master_batch 8 --num_epochs 20 --lr_step 14,18 --gpus 0,1,2,3 --lr 1.25e-5 --dim_weight 1 --resume
# test
#python test.py ddd --exp_id traffic_car_total --dataset traffic_car --resume --cam cam_sample
cd ..
