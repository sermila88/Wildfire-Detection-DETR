#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=si324
#SBATCH --output=/vol/bitbucket/si324/rf-detr-wildfire/outputs/rf_detr_hyperparameter_tuning_v2/logs/eval_trial003_%j.out
#SBATCH --error=/vol/bitbucket/si324/rf-detr-wildfire/outputs/rf_detr_hyperparameter_tuning_v2/logs/eval_trial003_%j.err
#SBATCH --job-name=eval_trial003


export PATH=/vol/bitbucket/${USER}/rf-detr-wildfire/.venv/bin:$PATH
source /vol/bitbucket/${USER}/rf-detr-wildfire/.venv/bin/activate
source /vol/cuda/12.0.0/setup.sh
/usr/bin/nvidia-smi

# Create logs directory if it doesn't exist
mkdir -p /vol/bitbucket/si324/rf-detr-wildfire/outputs/rf_detr_hyperparameter_tuning_v2/logs

# Change to the directory with the eval script
cd /vol/bitbucket/si324/rf-detr-wildfire/eval

# Evaluate trial_003
BASE_DIR="/vol/bitbucket/si324/rf-detr-wildfire/outputs/rf_detr_hyperparameter_tuning_v2"

echo "Starting evaluation of Trial 003 at $(date)"
echo "Checkpoint: $BASE_DIR/trial_003/checkpoints/checkpoint_best_ema.pth"
echo "Resolution: 1120"

python rfdetr_initial_hyperparameter_tuning_eval.py \
    --checkpoint "$BASE_DIR/trial_003/checkpoints/checkpoint_best_ema.pth" \
    --resolution 1120 \
    --experiment_name "trial_003_eval_IoU_0.01"

echo "Evaluation complete at $(date)!"