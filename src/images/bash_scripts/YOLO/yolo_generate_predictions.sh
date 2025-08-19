#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=si324
#SBATCH --job-name=yolo_predict
#SBATCH --output=outputs/yolo_baseline_v1/logs/predict-%j.out
#SBATCH --error=outputs/yolo_baseline_v1/logs/predict-%j.err

# Activate virtual environment
export PATH=/vol/bitbucket/${USER}/rf-detr-wildfire/.venv/bin:$PATH
source /vol/bitbucket/${USER}/rf-detr-wildfire/.venv/bin/activate
source /vol/cuda/12.0.0/setup.sh
/usr/bin/nvidia-smi

# Go to project root
cd /vol/bitbucket/${USER}/rf-detr-wildfire

echo "🚀 Starting YOLO Predictions for Baseline Evaluation"
echo "📅 Started at: $(date)"
echo "🖥️  GPU Info:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

# Create predictions output directory
mkdir -p models/yolo_baseline_v1/test_preds

# Run YOLO predictions using PyroNear methodology
python eval/yolo_generate_predictions.py \
    --data_directory "data/pyro25img/images/test" \
    --model_directory "outputs/yolo_baseline_v1/train" \
    --project "yolo_baseline_v1"

echo "✅ Predictions completed at: $(date)"
echo "📁 Experiment: yolo_baseline_v1"
echo "📂 Predictions saved to: models/yolo_baseline_v1/test_preds/"
echo "📋 SLURM logs: outputs/yolo_baseline_v1/logs/"

# Show what was generated
echo "📊 Generated prediction files:"
if [ -d "models/yolo_baseline_v1/test_preds" ]; then
    find models/yolo_baseline_v1/test_preds -name "*.txt" | wc -l | xargs echo "   Total .txt files:"
    ls -la models/yolo_baseline_v1/test_preds/ | head -10
else
    echo "   ⚠️  No predictions directory found"
fi

echo "🔍 Run evaluation with these paths:"
echo "   GT_FOLDER = 'data/pyro25img/images/test'"
echo "   PRED_FOLDER = 'models/yolo_baseline_v1/test_preds/[MODEL_NAME]/labels'"