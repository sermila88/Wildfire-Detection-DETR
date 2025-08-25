#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --mail-type=ALL 
#SBATCH --mail-user=si324 
#SBATCH --output=/vol/bitbucket/si324/rf-detr-wildfire/src/images/eval_results/Ground_truth_BB/logs/%x-%j.out
#SBATCH --error=/vol/bitbucket/si324/rf-detr-wildfire/src/images/eval_results/Ground_truth_BB/logs/%x-%j.err

 
export PATH=/vol/bitbucket/${USER}/rf-detr-wildfire/.venv/bin:$PATH
source /vol/bitbucket/${USER}/rf-detr-wildfire/.venv/bin/activate
source /vol/cuda/12.0.0/setup.sh
/usr/bin/nvidia-smi

# Create logs directory if it doesn't exist
mkdir -p /vol/bitbucket/si324/rf-detr-wildfire/src/images/eval_results/Ground_truth_BB/logs

cd /vol/bitbucket/si324/rf-detr-wildfire
python src/images/utils/YOLO/yolo_ground_truth_bb.py