cd src
# train
python main.py ddd --exp_id multiview1 --dataset multiview  --cam cam_sample --batch_size 32 --master_batch 8 --num_epochs 35 --lr_step 33,34  --gpus 0,1,2,3 --lr 1.25e-5 --resume
# test
#python test.py ddd --exp_id traffic_car_total --dataset traffic_car --resume --cam cam_sample
cd ..
