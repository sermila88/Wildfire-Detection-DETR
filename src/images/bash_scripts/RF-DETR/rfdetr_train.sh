#!/bin/bash
#SBATCH --job-name=RF-DETR_initial_training
#SBATCH --partition=gpgpu,gpgpuB,AMD7-A100-T
#SBATCH --gres=gpu:1                  
#SBATCH --mem=32G                           
#SBATCH --mail-type=ALL               
#SBATCH --mail-user=si324           
#SBATCH --output=/vol/bitbucket/si324/rf-detr-wildfire/src/images/outputs/RF-DETR_initial_training/logs/%x-%j.out
#SBATCH --error=/vol/bitbucket/si324/rf-detr-wildfire/src/images/outputs/RF-DETR_initial_training/logs/%x-%j.err

echo "🔍 RF-DETR Initial Training"
echo "🕐 Started at: $(date)"
echo "🖥️  Node: $(hostname)"

# Environment setup
export PATH=/vol/bitbucket/${USER}/rf-detr-wildfire/.venv/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /vol/bitbucket/${USER}/rf-detr-wildfire/.venv/bin/activate
source /vol/cuda/12.0.0/setup.sh
/usr/bin/nvidia-smi
echo " GPU Model: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"

# Go to the project folder 
cd /vol/bitbucket/${USER}/rf-detr-wildfire

# Ensure logs directory exists
mkdir -p src/images/outputs/RF-DETR_initial_training/logs

python src/images/train/rfdetr_train.py 

