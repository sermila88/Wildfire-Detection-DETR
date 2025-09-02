"""
RF-DETR Training Script with best hyperparameters 
"""

from rfdetr.detr import RFDETRBase
import torch
import os
import time
start_time = time.time()

os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12356" 
os.environ["LOCAL_RANK"] = "0"

# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================
# Modify this  for different experiments
EXPERIMENT_NAME = "RF-DETR_train_with_best_hparameters"  # Change this for different experiments

# Dataset path (COCO format)
dataset_path = "/vol/bitbucket/si324/rf-detr-wildfire/src/images/data/pyro25img/images"

# ============================================================================
# OUTPUT DIRECTORY SETUP
# ============================================================================
# Create organized output structure
project_root = "/vol/bitbucket/si324/rf-detr-wildfire"
outputs_root = os.path.join(project_root, "src/images/outputs")
experiment_dir = os.path.join(outputs_root, EXPERIMENT_NAME)

# Create subdirectories for checkpoints and logs
checkpoints_dir = os.path.join(experiment_dir, "checkpoints")
logs_dir = os.path.join(experiment_dir, "logs")

# Create all necessary directories
os.makedirs(checkpoints_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)

print(f"🎯 Experiment: {EXPERIMENT_NAME}")
print(f"📁 Output directory: {experiment_dir}")
print(f"  ├── checkpoints/   → {checkpoints_dir}")
print(f"  ├── logs/          → {logs_dir}")

# ============================================================================
# TRAINING
# ============================================================================
# Clear GPU memory and initialize model  
torch.cuda.empty_cache()
model = RFDETRBase(resolution=896)

# Train the model with outputs going to checkpoints directory
print(f"\n🚀 Starting training...")
model.train(
    dataset_dir=dataset_path,
    epochs=30,
    batch_size=4,           
    grad_accum_steps=4,     
    lr=3.78e-05,
    lr_encoder=2.41e-05,
    weight_decay=1.09e-04,
    gradient_checkpointing=False,
    use_ema=True,
    device="cuda",      
    use_amp=True, # Mixed precision 
    ddp=False,
    output_dir=checkpoints_dir,
    wandb=True,
    project="RF-DETR_train_with_best_hparameters",
    name=EXPERIMENT_NAME
)

print(f"\n Training completed")
print(f" Checkpoints saved to: {checkpoints_dir}")
print(f" To evaluate, use experiment name: {EXPERIMENT_NAME}")

# Log training time
end_time = time.time()
duration = end_time - start_time
hours = int(duration // 3600)
minutes = int((duration % 3600) // 60)
seconds = int(duration % 60)

print(f"\n Training Duration: {hours}h {minutes}m {seconds}s")
print(f" Finished at: {time.strftime('%Y-%m-%d %H:%M:%S')}")