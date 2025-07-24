#!/bin/bash
#SBATCH --job-name=rf_detr_hyperparameter_tuning
#SBATCH --output=outputs/hyperparameter_tuning_v2/logs/rf_detr_hyperparameter_tuning_%j.out
#SBATCH --error=outputs/hyperparameter_tuning_v2/logs/rf_detr_hyperparameter_tuning_%j.err
#SBATCH --gres=gpu:1
#SBATCH --partition=gpgpu,a16gpu,gpgpuB
#SBATCH --mem=40G
#SBATCH --time=96:00:00
#SBATCH --cpus-per-task=8

echo "🔍 RF-DETR Hyperparameter Tuning"
echo "🕐 Job started at: $(date)"
echo "🖥️  Running on node: $(hostname)"

# Create logs directory for this experiment
mkdir -p outputs/hyperparameter_tuning_v2/logs

echo "🔧 GPU info:"
nvidia-smi

source /vol/bitbucket/${USER}/rf-detr-wildfire/.venv/bin/activate
source /vol/cuda/12.0.0/setup.sh
/usr/bin/nvidia-smi

# Change to project directory  
cd /vol/bitbucket/si324/rf-detr-wildfire

# Create outputs directory
mkdir -p outputs/hyperparameter_tuning_v2

echo ""
echo "🎯 RF-DETR Optimization Configuration:"
echo "   📊 Trials: 12 systematic trials with TPE sampler"
echo "   🔧 Parameters: lr (1e-5 to 1e-3), lr_encoder, batch_size, epochs (25-100), weight_decay, resolution"
echo "   📐 Resolution range: 896-1232px (optimized for tiny smoke detection)"
echo "   📦 Effective batch size: 16 (adaptive gradient accumulation)"
echo "   ⚡ Early stopping: Enabled (using EMA for decisions)"
echo "   📈 EMA weights: Enabled for better model stability"
echo "   📊 Evaluation: Binary smoke presence accuracy on validation set"
echo "   🚀 Logging: All checkpoints, metrics, plots, optimizer states saved"
echo "   💾 Storage: ALL trials preserved with complete resumability data"
echo "   🔄 Backups: Study database backed up every 3 trials"
echo "   ⏱️  Expected duration: 20-80 hours (comprehensive logging + early stopping)"
echo "   🛡️  Resumability: Complete state preservation for 4-day limit recovery"

echo ""
echo "🚀 Starting RF-DETR hyperparameter optimization..."

# Run RF-DETR hyperparameter optimization
python train/rf_detr_hyperparameter_tuning.py

echo ""
echo "✅ RF-DETR hyperparameter optimization completed at: $(date)"
echo "📁 Results location:"
echo "   📊 Results: outputs/hyperparameter_tuning_v2/final_results.json"
echo "   📂 Individual trials: outputs/hyperparameter_tuning_v2/trial_*/"
echo "   💾 Optuna database: outputs/hyperparameter_tuning_v2/study.db"
echo "   🔄 Database backups: outputs/hyperparameter_tuning_v2/backups/"
echo "   📈 Per-trial data: checkpoints/, plots/, metrics/, optimizer_states/, logs/"
echo ""
echo "🎯 Check the comprehensive results file for optimal hyperparameters!"
echo "🚀 Use the recommended parameters to train your production model."
echo "🛡️  All data preserved for analysis and potential resumption!" 