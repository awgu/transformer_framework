#!/bin/bash

#SBATCH --job-name=transformer-trainer
#SBATCH --partition=train
#SBATCH --ntasks=2
#SBATCH --nodes=2
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=16
#SBATCH --exclude=a100-st-p4d24xlarge-6,a100-st-p4d24xlarge-7,a100-st-p4d24xlarge-28,a100-st-p4d24xlarge-48,a100-st-p4d24xlarge-16,a100-st-p4d24xlarge-58,a100-st-p4d24xlarge-56,a100-st-p4d24xlarge-5,a100-st-p4d24xlarge-8,a100-st-p4d24xlarge-57

##SBATCH --nodelist=a100-st-p4d24xlarge-17,a100-st-p4d24xlarge-18

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip
# Enable for A100
export LOGLEVEL=INFO
export FI_PROVIDER="efa"
export FI_EFA_USE_DEVICE_RDMA=1
export NCCL_ALGO=ring

# debugging flags (optional)
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=WARN
export PYTHONFAULTHANDLER=1

export LD_LIBRARY_PATH=/opt/amazon/efa/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH

echo $LD_LIBRARY_PATH
export CUDA_LAUNCH_BLOCKING=0

# on your cluster you might need these:
# set the network interface
export NCCL_SOCKET_IFNAME="eth0,en,eth,em,bond"
dcgmi profile --pause
srun torchrun --nnodes 2 --nproc_per_node 8 --rdzv_id 101 --rdzv_backend c10d --rdzv_endpoint "$head_node_ip:29500" ./main_training.py --model deepvit --batch-size 105 --use-nonrecursive --bucket-size 3000000000
dcgmi profile --resume
