#!/bin/bash
#SBATCH --job-name=rfdetr_test_eval_trial_6
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --mail-type=ALL 
#SBATCH --mail-user=si324 
#SBATCH --output=/vol/bitbucket/si324/rf-detr-wildfire/outputs/rfdetr_test_eval_final_hparam_tun_trial_6/logs/eval-%j.out
#SBATCH --error=/vol/bitbucket/si324/rf-detr-wildfire/outputs/rfdetr_test_eval_final_hparam_tun_trial_6/logs/eval-%j.err

# Activate environment
export PATH=/vol/bitbucket/${USER}/rf-detr-wildfire/.venv/bin:$PATH
source /vol/bitbucket/${USER}/rf-detr-wildfire/.venv/bin/activate
source /vol/cuda/12.0.0/setup.sh
/usr/bin/nvidia-smi

# Go to the project folder 
cd /vol/bitbucket/${USER}/rf-detr-wildfire

# Define paths
CHECKPOINT="/vol/bitbucket/si324/rf-detr-wildfire/outputs/rf_detr_hyperparameter_tuning_v3/trial_006/checkpoints/checkpoint_best_ema.pth"
RESOLUTION=1232
OUTPUT_DIR="/vol/bitbucket/si324/rf-detr-wildfire/outputs/rfdetr_test_eval_final_hparam_tun_trial_6"

# Create output directory
mkdir -p ${OUTPUT_DIR}
mkdir -p /vol/bitbucket/si324/rf-detr-wildfire/outputs/rfdetr_test_eval_final_hparam_tun_trial_6/logs/

# Run evaluation with SAFE output location
python eval/rfdetr_hparameter_tuning_eval.py \
  --checkpoint ${CHECKPOINT} \
  --resolution ${RESOLUTION} \
  --output_dir ${OUTPUT_DIR}

echo "✅ Evaluation complete! Results in ${OUTPUT_DIR}"