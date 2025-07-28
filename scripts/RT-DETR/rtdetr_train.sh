#!/bin/bash
#SBATCH --job-name=rtdetr_smoke_training
#SBATCH --output=/vol/bitbucket/si324/rf-detr-wildfire/outputs/rtdetr_smoke_detection_v1/logs/rtdetr_training_%j.out
#SBATCH --error=/vol/bitbucket/si324/rf-detr-wildfire/outputs/rtdetr_smoke_detection_v1/logs/rtdetr_training_%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8

source /vol/bitbucket/${USER}/rf-detr-wildfire/.venv/bin/activate
source /vol/cuda/12.0.0/setup.sh
/usr/bin/nvidia-smi

# Change to project directory  
cd /vol/bitbucket/si324/rf-detr-wildfire

# Create logs directory for this experiment
mkdir -p outputs/rtdetr_smoke_detection_v1/logs

# Create outputs directory
mkdir -p outputs/rtdetr_smoke_detection_v1

echo ""
echo "🎯 RT-DETR Training Configuration:"
echo "   🏗️  Model: RT-DETR-L (balanced speed/accuracy)"
echo "   📐 Resolution: 1280px (optimized for tiny smoke detection)"
echo "   📦 Batch size: 8 (adaptive to GPU memory)"
echo "   🔧 Learning rate: 0.0001 (conservative for fine-tuning)"
echo "   ⚡ Early stopping: 15 epochs patience"
echo "   📈 AMP: Enabled for memory efficiency"
echo "   💾 Checkpoints: Every 10 epochs"

echo ""
echo "🚀 Starting RT-DETR training..."

# Run RT-DETR training
python train/train_rtdetr.py

echo ""
echo "✅ RT-DETR training completed at: $(date)"
echo "📁 Results location:"
echo "   📊 Training results: outputs/rtdetr_smoke_detection_v1/training/"
echo "   💾 Best model: outputs/rtdetr_smoke_detection_v1/training/weights/best.pt"
echo "   📈 Metrics: outputs/rtdetr_smoke_detection_v1/training/results.png"
echo ""
