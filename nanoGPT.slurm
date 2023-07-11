#!/bin/bash

#SBATCH --job-name=NanoGPT-trainer

#SBATCH --ntasks=8

#SBATCH --nodes=8

#SBATCH --gpus-per-task=8

#SBATCH --cpus-per-task=96

#SBATCH --partition=train 



nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip
export LOGLEVEL=INFO
# Enable for A100
export FI_PROVIDER="efa"

# debugging flags (optional)
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=WARN
export PYTHONFAULTHANDLER=1

export LD_LIBRARY_PATH=/opt/amazon/efa/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH
export NCCL_DEBUG=INFO
export NCCL_BLOCKING_WAIT=1
export CUDA_LAUNCH_BLOCKING=1

# on your cluster you might need these:
# set the network interface
export NCCL_SOCKET_IFNAME="ens"
export FI_EFA_USE_DEVICE_RDMA=1
srun torchrun --nnodes=8 --nproc_per_node 8 --rdzv_id 101 --rdzv_backend c10d --rdzv_endpoint "$head_node_ip:29500" ./tp_train.py