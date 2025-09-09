#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --mail-type=ALL 
#SBATCH --mail-user=si324 
#SBATCH --output=/vol/bitbucket/si324/rf-detr-wildfire/src/images/inference_time_comparison/logs/GPU_inference_time_comparison-%j.out
#SBATCH --error=/vol/bitbucket/si324/rf-detr-wildfire/src/images/inference_time_comparison/logs/GPU_inference_time_comparison-%j.err
 
export PATH=/vol/bitbucket/${USER}/rf-detr-wildfire/.venv/bin:$PATH
source /vol/bitbucket/${USER}/rf-detr-wildfire/.venv/bin/activate
source /vol/cuda/12.0.0/setup.sh
/usr/bin/nvidia-smi

# Go to the project folder and run the training script
cd /vol/bitbucket/${USER}/rf-detr-wildfire

# Create logs directory if it doesn't exist
mkdir -p /vol/bitbucket/${USER}/rf-detr-wildfire/src/images/inference_time_comparison/logs

python src/images/utils/inference_time_compare_GPU.py

