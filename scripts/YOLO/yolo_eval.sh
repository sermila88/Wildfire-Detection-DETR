#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=si324
#SBATCH --job-name=yolo_eval
#SBATCH --output=outputs/yolo_baseline_v1_IoU=0.01/logs/eval-%j.out
#SBATCH --error=outputs/yolo_baseline_v1_IoU=0.01/logs/eval-%j.err

# Ensure logs dir exists
mkdir -p outputs/yolo_baseline_v1_IoU=0.01/logs

# Activate virtual environment
export PATH=/vol/bitbucket/${USER}/rf-detr-wildfire/.venv/bin:$PATH
source /vol/bitbucket/${USER}/rf-detr-wildfire/.venv/bin/activate

# Go to project root
cd /vol/bitbucket/${USER}/rf-detr-wildfire

# Set cache and config dirs to avoid home quota issues
export WANDB_CACHE_DIR=/vol/bitbucket/${USER}/rf-detr-wildfire/.cache/wandb
export XDG_CONFIG_HOME=/vol/bitbucket/${USER}/rf-detr-wildfire/.config
mkdir -p "$WANDB_CACHE_DIR" "$XDG_CONFIG_HOME"

echo "🔍 Starting Evaluation for YOLO Baseline"
echo "📅 Started at: $(date)"

# Run evaluation
python eval/pyronear_eval.py

echo "✅ Evaluation completed at: $(date)"
echo "📁 Results saved to: outputs/yolo_baseline_v1_IoU=0.01/eval_results/"
