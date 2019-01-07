#!/bin/bash
#SBATCH --gres=gpu:1              # Number of GPUs (per node)
#SBATCH --mem=2048M               # memory (per node)
#SBATCH --time=0-12:00            # time (DD-HH:MM)
#SBATCH --n-tasks=1
#SBATCH --cpus-per-task=6
#SBATCH --array=0-6
#SBATCH --job-name=fMNISTbasic
#SBATCH --error=./fMNISTbasic_%j.err
#SBATCH --output=./fMNISTbasic_%j.out
module load cuda/8.0.44
module load cudnn/7.0
source $BRAIN_ENV
python3 train_model.py --epochs 5000 --experiment_name basic_training --device "gpu:0" --batch_size 2048

