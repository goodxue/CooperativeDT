cd src
# train
python main.py ddd --exp_id multiview_34 --dataset multiview  --cam cam_sample --batch_size 32 --master_batch 8 --num_epochs 50 --lr_step 30,40  --gpus 0,1,2,3 --lr 1.25e-5
# test
#python test.py ddd --exp_id traffic_car_total --dataset traffic_car --resume --cam cam_sample
cd ..
