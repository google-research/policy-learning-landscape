#!/bin/bash
#SBATCH --gres=gpu:1              # Number of GPUs (per node)
#SBATCH --mem=16384M               # memory (per node)
#SBATCH --time=0-10:00            # time (DD-HH:MM)
#SBATCH --n-tasks=1
#SBATCH --cpus-per-task=6
#SBATCH --array=0
#SBATCH --job-name=hess_fMNISTbasic
#SBATCH --error=./hess_fMNISTbasic_%j.err
#SBATCH --output=./hess_fMNISTbasic_%j.out

[ -z "$WEIGHTS" ] && echo "Please provide WEIGHTS" && exit 1;
module load cuda/8.0.44
module load cudnn/7.0
source $BRAIN_ENV

python3 analyze_hessian.py --device "gpu:0" --save_directory basic_training --load_directory $WEIGHTS

