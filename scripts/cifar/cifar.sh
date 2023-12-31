# CUDA_VISIBLE_DEVICE='0, 1, 2, 3' python -m torch.distributed.launch --nproc_per_node 4 main.py
python RCN_Main.py \
    --storage_folder 'cifar_25w10s' \
    --DLC True \
    --local_rank 0 \
    --dataset 'cifar_fs' \
    --network 'ResNet18' \
    --pretrain False \
    --train_flag True \
    --test_method 'inc' \
    --test_model 'base' \
    --sim_metric 'euc' \
    --epoch 80 \
    --tasks 200 \
    --n-way 25 \
    --n-shot 10 \
    --n-query 15 \
    --use_auto_scheduler False \
    --batch_size_test 100 \
    --optimizer 'sgd' \
    --lr 0.03 \
    --wd 1e-4 \
    --momentum 0.9 \
    --nesterov True \
    --scheduler 'SLR' \
    --steps 20 \
    --gamma 0.5
