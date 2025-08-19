#!/bin/bash
#SBATCH --job-name=resnet_lstm_baseline_train
#SBATCH --output=/vol/bitbucket/si324/rf-detr-wildfire/src/videos/train_outputs/ResNET_LSTM_baseline/logs/%j.out
#SBATCH --error=/vol/bitbucket/si324/rf-detr-wildfire/src/videos/train_outputs/ResNET_LSTM_baseline/logs/%j.err
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpgpuB,gpgpu,AMD7-A100-T
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

# Create all necessary directories
mkdir -p /vol/bitbucket/${USER}/rf-detr-wildfire/src/videos/train_outputs/ResNET_LSTM_baseline
mkdir -p /vol/bitbucket/${USER}/rf-detr-wildfire/src/videos/train_outputs/ResNET_LSTM_baseline/logs
mkdir -p /vol/bitbucket/${USER}/rf-detr-wildfire/src/videos/train_outputs/ResNET_LSTM_baseline/checkpoints

# Run training
python src/videos/train/Baseline_LSTM/LSTM_baseline_train.py \
  --data_dir /vol/bitbucket/si324/rf-detr-wildfire/src/videos/data \
  --output_dir /vol/bitbucket/si324/rf-detr-wildfire/src/videos/train_outputs/ResNET_LSTM_baseline \
  --batch_size 16 \
  --img_size 112 \
  --learning_rate 1e-5 \
  --max_epochs 50 \
  --wandb_project ResNET_LSTM_baseline \
  --num_workers 8
