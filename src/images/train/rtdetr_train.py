#!/usr/bin/env python3
"""
RT-DETR Training Script for Wildfire Smoke Detection using Ultralytics 
"""

import os
import torch
from pathlib import Path
from ultralytics import RTDETR
import wandb

# Set cache directory for Ultralytics models to prevent downloads to project root
cache_dir = Path.home() / ".ultralytics_cache"
cache_dir.mkdir(exist_ok=True)
os.environ['ULTRALYTICS_CACHE_DIR'] = str(cache_dir)

# ============================================================================
# EXPERIMENT CONFIGURATION  
# ============================================================================
EXPERIMENT_NAME = "RT-DETR_initial_training"
DATASET_PATH = "/vol/bitbucket/si324/rf-detr-wildfire/src/images/data/pyro25img/images"

# Directory structure
project_root = "/vol/bitbucket/si324/rf-detr-wildfire"
outputs_root = os.path.join(project_root, "src/images/outputs")
experiment_dir = os.path.join(outputs_root, EXPERIMENT_NAME)
checkpoints_dir = os.path.join(experiment_dir, "checkpoints")
plots_dir = os.path.join(experiment_dir, "plots")
logs_dir = os.path.join(experiment_dir, "logs")
eval_results_dir = os.path.join(experiment_dir, "eval_results")

# ============================================================================
# TRAINING SETUP
# ============================================================================
def main():
    print("🔥 RT-DETR Training for Wildfire Smoke Detection")
    print(f"🎯 Experiment: {EXPERIMENT_NAME}")
    print(f"📁 Dataset: {DATASET_PATH}")
    print(f"💾 Experiment directory: {experiment_dir}")
    
    # Create directory structure
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(eval_results_dir, exist_ok=True)
    
    # Load RT-DETR model (pre-trained on COCO)
    # RT-DETR-X: Higher accuracy (54.8 mAP) better for small smoke detection

    # Initialize W&B 
    wandb.init(
        project="RT-DETR_initial_training",
        name=EXPERIMENT_NAME,
        config={
            "epochs": 50,
            "batch_size": 4,
            "learning_rate": 0.0005,
            "architecture": "RT-DETR-X",
            "dataset": "pyro25img",
            "resolution": 728,
        }
    )
    
    # Change to checkpoints directory so downloaded models go there
    original_dir = os.getcwd()
    os.chdir(checkpoints_dir)
    
    model = RTDETR("rtdetr-x.pt")  # RT-DETR-X for best accuracy on small objects
    
    # Change back to original directory
    os.chdir(original_dir)
    
    print("🚀 Starting RT-DETR training...")
    print("📋 Using COCO dataset format with high resolution for tiny objects")
    
    # Train the model - ultralytics can handle COCO format directly
    results = model.train(
        data=f"{DATASET_PATH}/data.yaml",
        epochs=50,
        imgsz=728,         
        batch=8,          
        lr0=0.0005,        
        patience=20,       
        save_period=-1,    
        project=experiment_dir,
        name="checkpoints",
        exist_ok=True,
        verbose=True,
        device="cuda",
        amp=True,             
        optimizer="AdamW", 
        cos_lr=True,      
        warmup_epochs=3,  
    )
    
    print(f"RT-DETR training completed!")
    print(f"Checkpoints saved to: {checkpoints_dir}/")
    print(f"Best model: {checkpoints_dir}/weights/best.pt")
    print(f"Training plots: {checkpoints_dir}/results.png")
    
    
    return results

if __name__ == "__main__":
    main() 