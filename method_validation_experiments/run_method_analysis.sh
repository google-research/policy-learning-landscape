#!/bin/bash
#SBATCH --gres=gpu:1              # Number of GPUs (per node)
#SBATCH --mem=16384M               # memory (per node)
#SBATCH --time=0-10:00            # time (DD-HH:MM)
#SBATCH --n-tasks=1
#SBATCH --cpus-per-task=2
#SBATCH --array=0-3
#SBATCH --job-name=meth_fMNISTbasic
#SBATCH --error=./meth_fMNISTbasic_%j.err
#SBATCH --output=./meth_fMNISTbasic_%j.out

[ -z "$WEIGHTS" ] && echo "Please provide WEIGHTS" && exit 1;
module load cuda/8.0.44
module load cudnn/7.0
source $BRAIN_ENV

python3 analyze_method.py --device "gpu:0" \
--save_directory $SCRATCH/basic_training --load_directory $WEIGHTS --step_size 0.01 --n_samples 1000
echo "done 0.01"
python3 analyze_method.py --device "gpu:0" \
--save_directory $SCRATCH/basic_training --load_directory $WEIGHTS --step_size 0.1 --n_samples 1000
echo "done 0.1"
python3 analyze_method.py --device "gpu:0" \
--save_directory $SCRATCH/basic_training --load_directory $WEIGHTS --step_size 0.5 --n_samples 1000
echo "done 0.5"
python3 analyze_method.py --device "gpu:0" \
--save_directory $SCRATCH/basic_training --load_directory $WEIGHTS --step_size 1.0 --n_samples 1000
echo "done 1.0"
