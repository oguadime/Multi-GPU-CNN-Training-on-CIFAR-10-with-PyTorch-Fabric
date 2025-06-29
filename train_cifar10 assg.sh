#!/bin/bash
#SBATCH --job-name="cifar10-multi-gpu"
#SBATCH --output="output/%j.out"
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=20G
#SBATCH --gpu-bind=closest
#SBATCH --account=becs-delta-gpu
#SBATCH --time=00:30:00

# Load your environment as needed:
module load anaconda3_gpu
conda activate mynewenv

srun python -u cifar10-fabric-tb assg.py \
     --accelerator gpu \
     --devices 2 \
     --num_nodes 1 \
     --strategy ddp \
     --precision bf16-mixed \
     --batch-size 64 \
     --num-workers 4 \
     --lr 1e-3 \
     --gamma 0.7 \
     --epochs 10


    
     

