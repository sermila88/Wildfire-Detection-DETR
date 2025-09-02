#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --mail-type=ALL 
#SBATCH --mail-user=si324 
#SBATCH --output=/vol/bitbucket/si324/rf-detr-wildfire/src/images/compare_eval_results_hparam_tuning_w_best_hparams/logs/eval_compare_hparam_tuning_NMS-%j.out
#SBATCH --error=/vol/bitbucket/si324/rf-detr-wildfire/src/images/compare_eval_results_hparam_tuning_w_best_hparams/logs/eval_compare_hparam_tuning_NMS-%j.err
 
export PATH=/vol/bitbucket/${USER}/rf-detr-wildfire/.venv/bin:$PATH
source /vol/bitbucket/${USER}/rf-detr-wildfire/.venv/bin/activate
source /vol/cuda/12.0.0/setup.sh
/usr/bin/nvidia-smi

# Go to the project folder and run the training script
cd /vol/bitbucket/${USER}/rf-detr-wildfire

# Create logs directory if it doesn't exist
mkdir -p /vol/bitbucket/${USER}/rf-detr-wildfire/src/images/compare_eval_results_hparam_tuning_w_best_hparams/logs

python src/images/eval/eval_compare_hparam_tuning.py

