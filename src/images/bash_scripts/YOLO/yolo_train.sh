#!/bin/bash
#SBATCH --job-name=YOLO_baseline
#SBATCH --partition=gpgpu,AMD7-A100-T,gpgpuB
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --mail-type=ALL 
#SBATCH --mail-user=si324 
#SBATCH --output=/vol/bitbucket/si324/rf-detr-wildfire/src/images/outputs/YOLO_baseline/logs/%x-%j.out
#SBATCH --error=/vol/bitbucket/si324/rf-detr-wildfire/src/images/outputs/YOLO_baseline/logs/%x-%j.err

# Activate virtual environment
export PATH=/vol/bitbucket/${USER}/rf-detr-wildfire/.venv/bin:$PATH
source /vol/bitbucket/${USER}/rf-detr-wildfire/.venv/bin/activate
source /vol/cuda/12.0.0/setup.sh
/usr/bin/nvidia-smi

# Go to project root
cd /vol/bitbucket/${USER}/rf-detr-wildfire

# Redirect cache directories to project directory to avoid home directory quota issues
export WANDB_CACHE_DIR=/vol/bitbucket/${USER}/rf-detr-wildfire/.cache/wandb
export XDG_CONFIG_HOME=/vol/bitbucket/${USER}/rf-detr-wildfire/.config
mkdir -p "$WANDB_CACHE_DIR" "$XDG_CONFIG_HOME"

echo "🚀 Starting YOLO Baseline Training"
echo "📅 Started at: $(date)"
echo "🖥️  GPU Info:"
echo " GPU Model: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"

# Run YOLO training 
python src/images/train/yolo_train.py \
    --model_weights yolov8x.pt \
    --data_config src/images/data/pyro25img/images/data.yaml \
    --epochs 100 \
    --img_size 640 \
    --batch_size 16 \
    --devices 0 \
    --project src/images/outputs/YOLO_baseline/training_outputs





