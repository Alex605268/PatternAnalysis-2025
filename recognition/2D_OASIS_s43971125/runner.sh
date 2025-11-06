#!/bin/bash
#SBATCH --partition=a100
#SBATCH --job-name=TrainingTest
#SBATCH --output=slurm_logs/ModuleTest_%j.log
#SBATCH --error=slurm_logs/ModuleTest_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:a100:1


# Source conda setup script
source ~/miniconda3/etc/profile.d/conda.sh

# Activate your environment
conda activate torch  # replace with your environment name

# Run the Python script
python train.py
