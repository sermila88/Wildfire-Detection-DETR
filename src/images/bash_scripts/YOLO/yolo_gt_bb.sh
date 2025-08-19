#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --mail-type=ALL 
#SBATCH --mail-user=si324 
 
export PATH=/vol/bitbucket/${USER}/rf-detr-wildfire/.venv/bin:$PATH
source /vol/bitbucket/${USER}/rf-detr-wildfire/.venv/bin/activate
source /vol/cuda/12.0.0/setup.sh
/usr/bin/nvidia-smi

# Set up paths
PROJECT_ROOT="/vol/bitbucket/si324/rf-detr-wildfire"
IMAGES_DIR="${PROJECT_ROOT}/data/pyro25img/images"
LABELS_DIR="${PROJECT_ROOT}/data/pyro25img/labels"

# Configuration
EXPERIMENT_NAME="yolo_baseline_v1"

echo "🔧 Configuration:"
echo "   📁 Project Root: ${PROJECT_ROOT}"
echo "   🖼️  Images Dir: ${IMAGES_DIR}"
echo "   🏷️  Labels Dir: ${LABELS_DIR}"
echo "   🎯 Experiment: ${EXPERIMENT_NAME}"
echo "   📊 Processing: train, id, test"

# Process each split
for SPLIT in "train" "valid" "test"; do
    echo ""
    echo "🚀 Processing ${SPLIT} split..."
    
    python ${PROJECT_ROOT}/utils/YOLO/yolo_ground_truth_bb.py \
        --experiment_name "${EXPERIMENT_NAME}" \
        --images_dir "${IMAGES_DIR}" \
        --labels_dir "${LABELS_DIR}" \
        --split "${SPLIT}" \
        --class_names "smoke"
    
    exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "✅ ${SPLIT} split completed successfully!"
    else
        echo "❌ ${SPLIT} split failed with exit code: $exit_code"
        break
    fi
done

if [ $exit_code -eq 0 ]; then
    echo ""
    echo "🎉 ALL SPLITS COMPLETED SUCCESSFULLY!"
    echo "📁 Check outputs in:"
    echo "   📊 Train: bounding_boxes/${EXPERIMENT_NAME}/yolo_ground_truth_bb/train/"
    echo "   📊 Valid:   bounding_boxes/${EXPERIMENT_NAME}/yolo_ground_truth_bb/valid/"
    echo "   📊 Test:  bounding_boxes/${EXPERIMENT_NAME}/yolo_ground_truth_bb/test/"
else
    echo "❌ Ground Truth Generation failed"
fi

echo "🏁 Finished at: $(date)"
exit $exit_code