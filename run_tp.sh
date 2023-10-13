CUDA_VISIBLE_DEVICES=2,3 torchrun --standalone --nproc_per_node=2 --rdzv_id=102 --rdzv_endpoint="localhost:5969" tp_train.py
#CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --use-env --nproc-per-node=2 --nnodes=1 --node-rank=0 --master-addr=mew1 --master-port=5969 tp_train.py 
