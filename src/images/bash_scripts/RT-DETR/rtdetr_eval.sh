#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --mail-type=ALL 
#SBATCH --mail-user=si324 
#SBATCH --output=/vol/bitbucket/${USER}/rf-detr-wildfire/outputs/rtdetr_smoke_detection_v1_IoU=0.01/logs/rtdetr_eval_%j.out
#SBATCH --error=/vol/bitbucket/${USER}/rf-detr-wildfire/outputs/rtdetr_smoke_detection_v1_IoU=0.01/logs/rtdetr_eval_%j.err

 
export PATH=/vol/bitbucket/${USER}/rf-detr-wildfire/.venv/bin:$PATH
source /vol/bitbucket/${USER}/rf-detr-wildfire/.venv/bin/activate
source /vol/cuda/12.0.0/setup.sh
/usr/bin/nvidia-smi

# Go to the project folder and run the RT-DETR evaluation script
cd /vol/bitbucket/${USER}/rf-detr-wildfire
python eval/rtdetr_eval.py 