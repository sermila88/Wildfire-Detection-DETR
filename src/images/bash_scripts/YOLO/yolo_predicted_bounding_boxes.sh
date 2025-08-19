#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --mail-type=ALL 
#SBATCH --mail-user=si324 
#SBATCH --job-name=yolo_pred_bb
#SBATCH --output=outputs/bounding_boxes/yolo_pred_bb_%j.out
#SBATCH --error=outputs/bounding_boxes/yolo_pred_bb_%j.err

echo "🎯 Starting YOLO Predicted Bounding Box Generation with TP/FP/FN/TN Breakdown"
echo "📅 Started at: $(date)"
echo "💻 Running on: $(hostname)"
echo "👤 User: ${USER}"

# Setup environment
export PATH=/vol/bitbucket/si324/rf-detr-wildfire/.venv/bin:$PATH
source /vol/bitbucket/si324/rf-detr-wildfire/.venv/bin/activate
source /vol/cuda/12.0.0/setup.sh

# Check GPU availability
echo "🔍 GPU Information:"
/usr/bin/nvidia-smi

# Create logs directory if it doesn't exist
mkdir -p /vol/bitbucket/si324/rf-detr-wildfire/logs

# Set up paths
PROJECT_ROOT="/vol/bitbucket/si324/rf-detr-wildfire"
IMAGES_DIR="${PROJECT_ROOT}/data/pyro25img/images"
GT_LABELS_DIR="${PROJECT_ROOT}/data/pyro25img/labels"
PRED_LABELS_DIR="${PROJECT_ROOT}/models/yolo_baseline_v1/test_preds/train/labels"

# Configuration
EXPERIMENT_NAME="yolo_baseline_v1_IoU=0.01"
SPLIT="test"
IMG_CONF_THRESHOLD="0.1"
OBJ_CONF_THRESHOLD="0.2"
IOU_THRESHOLD="0.01"

echo "🔧 Configuration:"
echo "   📁 Project Root: ${PROJECT_ROOT}"
echo "   🖼️  Images Dir: ${IMAGES_DIR}"
echo "   🏷️  GT Labels Dir: ${GT_LABELS_DIR}"
echo "   🔮 Pred Labels Dir: ${PRED_LABELS_DIR}"
echo "   🎯 Experiment: ${EXPERIMENT_NAME}"
echo "   📊 Split: ${SPLIT}"
echo "   🎚️  Image-Level Confidence Threshold: ${IMG_CONF_THRESHOLD}"
echo "   🎚️  Object-Level Confidence Threshold: ${OBJ_CONF_THRESHOLD}"
echo "   🎯 IoU Threshold: ${IOU_THRESHOLD}"

# Check if prediction directory exists
if [ ! -d "${PRED_LABELS_DIR}" ]; then
    echo "❌ Prediction labels directory not found: ${PRED_LABELS_DIR}"
    echo "   Please make sure you have run YOLO inference first!"
    exit 1
fi

echo ""
echo "🚀 Running YOLO Predicted Bounding Box Generation..."

python ${PROJECT_ROOT}/utils/YOLO/yolo_predict_bounding_boxes.py \
    --experiment_name "${EXPERIMENT_NAME}" \
    --images_dir "${IMAGES_DIR}" \
    --gt_labels_dir "${GT_LABELS_DIR}" \
    --pred_labels_dir "${PRED_LABELS_DIR}" \
    --split "${SPLIT}" \
    --img_conf_th "${IMG_CONF_THRESHOLD}" \
    --obj_conf_th "${OBJ_CONF_THRESHOLD}" \
    --iou_th "${IOU_THRESHOLD}"

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo ""
    echo "🎉 YOLO Predicted Bounding Box Generation completed successfully!"
    echo ""
    echo "📁 Check outputs in:"
    echo "   🎯 Image-level TP: bounding_boxes/${EXPERIMENT_NAME}/yolo_predicted_bb_breakdown/image_level/TP/"
    echo "   🎯 Image-level FP: bounding_boxes/${EXPERIMENT_NAME}/yolo_predicted_bb_breakdown/image_level/FP/"
    echo "   🎯 Image-level FN: bounding_boxes/${EXPERIMENT_NAME}/yolo_predicted_bb_breakdown/image_level/FN/"
    echo "   🎯 Image-level TN: bounding_boxes/${EXPERIMENT_NAME}/yolo_predicted_bb_breakdown/image_level/TN/"
    echo ""
    echo "   📦 Object-level TP: bounding_boxes/${EXPERIMENT_NAME}/yolo_predicted_bb_breakdown/object_level/TP/"
    echo "   📦 Object-level FP: bounding_boxes/${EXPERIMENT_NAME}/yolo_predicted_bb_breakdown/object_level/FP/"
    echo "   📦 Object-level FN: bounding_boxes/${EXPERIMENT_NAME}/yolo_predicted_bb_breakdown/object_level/FN/"
    echo ""
    echo "   📈 Statistics: bounding_boxes/${EXPERIMENT_NAME}/yolo_predicted_bb_breakdown/test_prediction_breakdown_stats.json"
    echo ""
    echo "   - Green boxes = True Positive detections (correct)"
    echo "   - Red boxes = False Positive detections (wrong)"
    echo "   - Blue circles = False Negative ground truth (missed)"
    echo "   - Yellow boxes = Ground truth annotations"
    echo ""
    echo "📊 Check the JSON stats file to see counts for each category!"
else
    echo "❌ YOLO Predicted Bounding Box Generation failed with exit code: $exit_code"
fi

echo "🏁 Finished at: $(date)"
exit $exit_code