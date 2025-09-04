#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --partition=AMD7-A100-T,gpgpu,gpgpuB,a16gpu
#SBATCH --mail-type=ALL 
#SBATCH --mail-user=si324 
#SBATCH --output=/vol/bitbucket/si324/rf-detr-wildfire/src/images/compare_trial_17/logs/trial_17-%j.out
#SBATCH --error=/vol/bitbucket/si324/rf-detr-wildfire/src/images/compare_trial_17/logs/trial_17-%j.err
 
export PATH=/vol/bitbucket/${USER}/rf-detr-wildfire/.venv/bin:$PATH
source /vol/bitbucket/${USER}/rf-detr-wildfire/.venv/bin/activate
source /vol/cuda/12.0.0/setup.sh
/usr/bin/nvidia-smi

# Go to the project folder and run the training script
cd /vol/bitbucket/${USER}/rf-detr-wildfire

# Create logs directory if it doesn't exist
mkdir -p /vol/bitbucket/${USER}/rf-detr-wildfire/src/images/compare_trial_17/logs

python src/images/eval/compare_trial_17.py

