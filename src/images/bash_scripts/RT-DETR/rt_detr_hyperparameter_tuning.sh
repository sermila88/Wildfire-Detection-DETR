#!/bin/bash
#SBATCH --job-name=rt_detr_hyperparameter_tuning
#SBATCH --output=outputs/rtdetr_official_hyperparameter_tuning_v1/logs/rt_detr_hyperparameter_tuning_%j.out
#SBATCH --error=outputs/rtdetr_official_hyperparameter_tuning_v1/logs/rt_detr_hyperparameter_tuning_%j.err
#SBATCH --gres=gpu:1
#SBATCH --partition=gpgpu,a16gpu,gpgpuB
#SBATCH --mem=16G


echo "🔥 RT-DETR Hyperparameter Tuning (Official Implementation)"
echo "🕐 Job started at: $(date)"
echo "🖥️  Running on node: $(hostname)"

# Create logs directory for this experiment
mkdir -p outputs/rtdetr_hyperparameter_tuning_v1/logs

echo "🔧 GPU info:"
nvidia-smi

source /vol/bitbucket/${USER}/rf-detr-wildfire/.venv/bin/activate
source /vol/cuda/12.0.0/setup.sh
/usr/bin/nvidia-smi

# Change to project directory  
cd /vol/bitbucket/si324/rf-detr-wildfire

# Set Ultralytics cache to project directory to avoid disk quota issues
export ULTRALYTICS_CACHE_DIR=/vol/bitbucket/si324/rf-detr-wildfire/.cache

# Create outputs directory
mkdir -p outputs/rtdetr_hyperparameter_tuning_v1

echo ""
echo "🎯 RT-DETR Optimization Configuration:"
echo "   📊 Trials: 20 systematic trials with TPE sampler + Median pruner"
echo "   🏗️  Model: RT-DETR-X (54.8 mAP, best accuracy for tiny objects)"
echo "   🔧 Parameters: lr0 (5e-5 to 5e-4), weight_decay, batch_size, epochs (36-108)"
echo "   ⚖️  Loss weights: cls (1.0-4.0), bbox (2.0-8.0), giou (1.0-3.0)"
echo "   🎨 Augmentation: mixup (0.0-0.2), mosaic (0.8-1.0), copy_paste (0.1)"
echo "   📐 Resolution range: 640-1024px (RT-DETR standards)"
echo "   📦 Batch size: 4-32 (official effective batch size range)"
echo "   🚀 Optimizer: AdamW (primary) + SGD with cosine LR scheduling"
echo "   ⚡ Early stopping: 20-50 patience with median pruning"
echo "   📈 Evaluation: mAP50-95 (COCO standard metric)"
echo "   🛠️  Schedule: 3x-9x epochs (36-108, based on 6x=72 official)"
echo "   🎯 Focal loss: alpha (0.2-0.3), gamma (1.5-2.5)"
echo "   📊 Logging: Complete trial metadata, checkpoints, metrics, backups"
echo "   💾 Storage: ALL trials preserved with full resumability"
echo "   🔄 Backups: Study database + trial states backed up continuously"
echo "   ⏱️  Expected duration: 60-96 hours (comprehensive evaluation)"
echo "   🛡️  Resumability: Enterprise-grade state preservation for 4-day runs"
echo "   📋 Standards: Following official RT-DETR paper & repository guidelines"

echo ""
echo "🚀 Starting RT-DETR hyperparameter optimization "

# Run RT-DETR hyperparameter optimization
python train/rt_detr_hyperparameter_tuning.py

echo ""
echo "✅ RT-DETR hyperparameter optimization completed at: $(date)"
echo "📁 Results location:"
echo "   📊 Results: outputs/rtdetr_hyperparameter_tuning_v1/comprehensive_final_results.json"
echo "   📂 Individual trials: outputs/rtdetr_hyperparameter_tuning_v1/trial_*/"
echo "   💾 Optuna database: outputs/rtdetr_hyperparameter_tuning_v1/study.db"
echo "   🔄 Database backups: outputs/rtdetr_hyperparameter_tuning_v1/backups/"
echo "   📈 Per-trial data: checkpoints/, plots/, metrics/, logs/, backups/"
echo "   🏆 Best model: outputs/rtdetr_hyperparameter_tuning_v1/trial_XXX/checkpoints/weights/best.pt"
echo ""
echo "🎯 Check comprehensive_final_results.json for optimal RT-DETR hyperparameters!"
echo "🚀 Use the best trial parameters for production wildfire detection model."
echo "🛡️  All data preserved for analysis, resumption, and model deployment!"

