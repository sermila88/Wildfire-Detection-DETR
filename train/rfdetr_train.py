"""
RF-DETR Training Script for Wildfire Smoke Detection from Roboflow 
"""

from rfdetr.detr import RFDETRBase
import torch
import os

os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12356" 
os.environ["LOCAL_RANK"] = "0"

# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================
# Modify this  for different experiments
EXPERIMENT_NAME = "rfdetr_smoke_detection_v1_1120_high_res"  # Change this for different experiments

# Dataset path (COCO format)
dataset_path = "/vol/bitbucket/si324/rf-detr-wildfire/data/pyro25img/images"

# ============================================================================
# OUTPUT DIRECTORY SETUP
# ============================================================================
# Create organized output structure
project_root = "/vol/bitbucket/si324/rf-detr-wildfire"
outputs_root = os.path.join(project_root, "outputs")
experiment_dir = os.path.join(outputs_root, EXPERIMENT_NAME)

# Create subdirectories for organized output
checkpoints_dir = os.path.join(experiment_dir, "checkpoints")
plots_dir = os.path.join(experiment_dir, "plots") 
logs_dir = os.path.join(experiment_dir, "logs")
eval_results_dir = os.path.join(experiment_dir, "eval_results")

# Create all necessary directories
os.makedirs(checkpoints_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)
os.makedirs(eval_results_dir, exist_ok=True)

print(f"🎯 Experiment: {EXPERIMENT_NAME}")
print(f"📁 Output directory: {experiment_dir}")
print(f"  ├── checkpoints/   → {checkpoints_dir}")
print(f"  ├── plots/         → {plots_dir}")
print(f"  ├── logs/          → {logs_dir}")
print(f"  └── eval_results/  → {eval_results_dir}")

# ============================================================================
# TRAINING
# ============================================================================
# Clear GPU memory and initialize model  
torch.cuda.empty_cache()
model = RFDETRBase(resolution=1120)

# Train the model with outputs going to checkpoints directory
print(f"\n🚀 Starting training...")
model.train(
    dataset_dir=dataset_path,
    epochs=100,
    batch_size=4,           
    grad_accum_steps=16,     
    lr=1e-4,
    device="cuda",      
    use_amp=True,  # Mixed precision 
    ddp=False,
    output_dir=checkpoints_dir  
)

print(f"\n Training completed")
print(f" Checkpoints saved to: {checkpoints_dir}")
print(f" To evaluate, use experiment name: {EXPERIMENT_NAME}")
