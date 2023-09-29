# tp
for b in 10 #{1..16}
do
    torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr=mew1 --master_port=1234 train.py --batch_size=$b --gradient_accumulation_steps=1 --max_iters=10 2>&1 | tee test.log #results/${b}.log
done

#for b in {1..16}
#do
#    CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr=mew1 --master_port=1234 train.py --batch_size=$b --gradient_accumulation_steps=1 2>&1 | tee results/${b}.log
#done
#
#for b in 20 24 28 32 36 40 44 48 52 56 60 64 
#do
#    CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr=mew1 --master_port=1234 train.py --batch_size=$b --gradient_accumulation_steps=1 2>&1 | tee results/${b}.log
#done
