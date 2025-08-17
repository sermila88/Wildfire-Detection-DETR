#!/bin/bash
#SBATCH --job-name=yolo_gt_labeled_bb
#SBATCH --output=/vol/bitbucket/si324/rf-detr-wildfire/src/videos/bounding_boxes/only_labeled_GT_BB/logs/yolo_gt_labeled_bb_%j.out
#SBATCH --error=/vol/bitbucket/si324/rf-detr-wildfire/src/videos/bounding_boxes/only_labeled_GT_BB/logs/yolo_gt_labeled_bb_%j.err
#SBATCH --cpus-per-task=32
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mail-type=ALL 
#SBATCH --mail-user=si324 

export PATH=/vol/bitbucket/${USER}/rf-detr-wildfire/.venv/bin:$PATH
source /vol/bitbucket/${USER}/rf-detr-wildfire/.venv/bin/activate
source /vol/cuda/12.0.0/setup.sh
/usr/bin/nvidia-smi

mkdir -p /vol/bitbucket/si324/rf-detr-wildfire/src/videos/bounding_boxes/only_labeled_GT_BB
mkdir -p /vol/bitbucket/si324/rf-detr-wildfire/src/videos/bounding_boxes/only_labeled_GT_BB/logs

# Run the script with parallel processing
echo "Starting bounding box drawing with $SLURM_CPUS_PER_TASK workers..."
python src/videos/utils/yolo_only_labeled_GT_BB.py \
    --input_dir /vol/bitbucket/si324/rf-detr-wildfire/data/pyrodata/data \
    --workers $SLURM_CPUS_PER_TASK \
    --splits train val test

