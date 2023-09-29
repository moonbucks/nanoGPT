# simplerun
torchrun --nproc-per-node=1 simple_model.py --batch_size 12 --hidden_size 16384 --n_layer 4 --train_iters 50  2>&1 | tee test.log

## tp
#for b in 20 #{1..16}
#do
#    torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr=node0 --master_port=1234 train.py --batch_size=$b --gradient_accumulation_steps=1 --dataset=random 2>&1 | tee test.log #results/${b}.log
#done

#for b in {1..16}
#do
#    CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr=mew1 --master_port=1234 train.py --batch_size=$b --gradient_accumulation_steps=1 2>&1 | tee results/${b}.log
#done
#
#for b in 20 24 28 32 36 40 44 48 52 56 60 64 
#do
#    CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr=mew1 --master_port=1234 train.py --batch_size=$b --gradient_accumulation_steps=1 2>&1 | tee results/${b}.log
#done
