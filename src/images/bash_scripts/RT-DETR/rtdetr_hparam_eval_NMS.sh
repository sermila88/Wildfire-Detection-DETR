#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --partition=AMD7-A100-T,gpgpu,gpgpuB,a16gpu
#SBATCH --mail-type=ALL
#SBATCH --mail-user=si324
#SBATCH --output=/vol/bitbucket/si324/rf-detr-wildfire/src/images/eval_results_hparam_tuning_NMS/logs/hparam_eval_nms-%j.out
#SBATCH --error=/vol/bitbucket/si324/rf-detr-wildfire/src/images/eval_results_hparam_tuning_NMS/logs/hparam_eval_nms-%j.err

export PATH=/vol/bitbucket/${USER}/rf-detr-wildfire/.venv/bin:$PATH
source /vol/bitbucket/${USER}/rf-detr-wildfire/.venv/bin/activate
source /vol/cuda/12.0.0/setup.sh
/usr/bin/nvidia-smi

# Go to the project folder
cd /vol/bitbucket/${USER}/rf-detr-wildfire

# Create logs directory if it doesn't exist
mkdir -p /vol/bitbucket/${USER}/rf-detr-wildfire/src/images/eval_results_hparam_tuning_NMS/logs

# Run the hyperparameter re-evaluation with NMS
python src/images/hyperparameter_tuning/rt_detr_hparam_tuning_eval_val.py