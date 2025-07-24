#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --mail-type=ALL 
#SBATCH --mail-user=si324 
 
export PATH=/vol/bitbucket/${USER}/rf-detr-wildfire/.venv/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /vol/bitbucket/${USER}/rf-detr-wildfire/.venv/bin/activate
source /vol/cuda/12.0.0/setup.sh
/usr/bin/nvidia-smi

# Go to the project folder and run the training script
cd /vol/bitbucket/${USER}/rf-detr-wildfire
python train/train.py 