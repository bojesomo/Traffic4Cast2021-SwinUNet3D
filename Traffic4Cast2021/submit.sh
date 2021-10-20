#!/bin/bash -l
# SLURM SUBMIT SCRIPT
#SBATCH --account=kuex0005
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --ntasks-per-node=4
#SBATCH --time=2-00:00:00
#SBATCH --job-name=T4C
#SBATCH --partition=gpu
#SBATCH --output=t4c.%j.out
#SBATCH --error=t4c.%j.err
# auto resubmit job 90 seconds before training ends due to wall time
# SBATCH --signal=SIGUSR1@90


# debugging flags (optional)
#export NCCL_DEBUG=INFO
#export NCCL_IB_DISABLE=1
#export NCCL_P2P_DISABLE=1
#export PYTHONFAULTHANDLER=1

# on your cluster you might need these:
# set the network interface
export NCCL_SOCKET_IFNAME=^docker0,lo

# might need the latest CUDA
module purge
module load cuda/11.3 miniconda/3 gcc/9.3

# activate conda env
conda activate ai4ex

# run script from above
for i in $(lsof /dev/nvidia0 | grep python | awk '{print $2}' | sort -u); do kill -9 $i; done

srun python Traffic4Cast2021/main1.py --nodes 1 --gpus 4 --precision 16 --batch-size 5 --epochs 100 --mlp_ratio 1 --stages 4 --patch_size 4 --dropout 0.0 --start_filters 192 --sampling-step 1 --decode_depth 1 --use_neck --lr 1e-4 --optimizer lamb --merge_type both --mix_features --city_category TEMPORAL --memory_efficient
# srun python Traffic4Cast2021/main1.py --nodes 1 --gpus 4 --precision 16 --batch-size 5 --epochs 100 --mlp_ratio 1 --stages 4 --patch_size 4 --dropout 0.0 --start_filters 192 --sampling-step 1 --decode_depth 1 --use_neck --lr 1e-4 --optimizer lamb --merge_type both --mix_features --city_category TEMPORAL --memory_efficient --name TEMPORAL_real_swinunet3d_141848694 --time-code 20210913T135845