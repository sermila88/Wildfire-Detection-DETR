#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --mail-type=ALL 
#SBATCH --mail-user=si324 
#SBATCH --output=/vol/bitbucket/si324/rf-detr-wildfire/outputs/bounding_boxes/rfdetr_smoke_detection_v1_IoU=0.1/rfdetr_pred_bb-%j.out
#SBATCH --error=/vol/bitbucket/si324/rf-detr-wildfire/outputs/bounding_boxes/rfdetr_smoke_detection_v1_IoU=0.1/rfdetr_pred_bb-%j.err
 
export PATH=/vol/bitbucket/${USER}/rf-detr-wildfire/.venv/bin:$PATH
source /vol/bitbucket/${USER}/rf-detr-wildfire/.venv/bin/activate
source /vol/cuda/12.0.0/setup.sh
/usr/bin/nvidia-smi

# Run the script
python /vol/bitbucket/${USER}/rf-detr-wildfire/utils/RF-DETR/rfdetr_generate_pred_bounding_boxes.py

