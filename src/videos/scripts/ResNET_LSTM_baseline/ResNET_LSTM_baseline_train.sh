#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --mail-type=ALL 
#SBATCH --mail-user=si324 
 
export PATH=/vol/bitbucket/${USER}/rf-detr-wildfire/.venv/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Activate virtual environment
source /vol/bitbucket/${USER}/rf-detr-wildfire/.venv/bin/activate
source /vol/cuda/12.0.0/setup.sh
/usr/bin/nvidia-smi

# Go to the project folder and run the training script
cd /vol/bitbucket/${USER}/rf-detr-wildfire


python src/videos/train/Baseline_LSTM/LSTM_baseline_train.py \
  --data_dir /path/to/data \
  --batch_size 16 \
  --img_size 112 \
  --learning_rate 1e-5 \
  --max_epochs 50 \
  --wandb_project ResNET_LSTM_baseline
