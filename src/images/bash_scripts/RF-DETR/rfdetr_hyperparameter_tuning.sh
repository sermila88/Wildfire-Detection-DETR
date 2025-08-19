#!/bin/bash
#SBATCH --job-name=RF-DETR_hyperparameter_tuning
#SBATCH --partition=gpgpuB,gpgpu,AMD7-A100-T
#SBATCH --gres=gpu:1                  
#SBATCH --mem=32G                           
#SBATCH --mail-type=ALL               
#SBATCH --mail-user=si324           
#SBATCH --output=/vol/bitbucket/si324/rf-detr-wildfire/src/images/outputs/RF-DETR_hyperparameter_tuning/logs/%x-%j.out
#SBATCH --error=/vol/bitbucket/si324/rf-detr-wildfire/src/images/outputs/RF-DETR_hyperparameter_tuning/logs/%x-%j.err

echo "🔍 RF-DETR Hyperparameter Tuning"
echo "🕐 Started at: $(date)"
echo "🖥️  Node: $(hostname)"

# Environment setup
export PATH=/vol/bitbucket/${USER}/rf-detr-wildfire/.venv/bin:$PATH
source /vol/bitbucket/${USER}/rf-detr-wildfire/.venv/bin/activate
source /vol/cuda/12.0.0/setup.sh
/usr/bin/nvidia-smi
echo "🎮 GPU Model: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"

# Move to project directory
cd /vol/bitbucket/${USER}/rf-detr-wildfire

# Ensure logs directory exists
mkdir -p src/images/outputs/RF-DETR_hyperparameter_tuning/logs

# Tell PyTorch we're using 1 GPU, not a distributed cluster
export MASTER_ADDR=localhost
export MASTER_PORT=12355
export WORLD_SIZE=1
export LOCAL_RANK=0

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run the tuning script
python src/images/hyperparameter_tuning/rf_detr_hyperparameter_tuning.py

echo "✅ Finished at: $(date)"
