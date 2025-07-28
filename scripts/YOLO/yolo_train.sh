#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --mail-type=ALL 
#SBATCH --mail-user=si324 
#SBATCH --job-name=yolo_baseline
#SBATCH --output=outputs/yolo_baseline_v1/logs/slurm-%j.out
#SBATCH --error=outputs/yolo_baseline_v1/logs/slurm-%j.err

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

echo "🚀 Starting YOLO Baseline Training for Wildfire Smoke Detection"
echo "📅 Started at: $(date)"
echo "🖥️  GPU Info:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

# Run YOLO training with optimal settings for GPU
python train/yolo_train.py \
    --model_weights yolov8x.pt \
    --data_config data/pyro25img/images/data.yaml \
    --epochs 100 \
    --img_size 640 \
    --batch_size 16 \
    --devices 0 \
    --project outputs/yolo_baseline_v1

echo "✅ Training completed at: $(date)"
echo "📁 Experiment: yolo_baseline_v1"
echo "📂 Results organized in: outputs/yolo_baseline_v1/"
echo "📋 SLURM logs: outputs/yolo_baseline_v1/logs/"
echo "🏆 Model weights: outputs/yolo_baseline_v1/checkpoints/"
echo "📈 Training plots: outputs/yolo_baseline_v1/plots/"



