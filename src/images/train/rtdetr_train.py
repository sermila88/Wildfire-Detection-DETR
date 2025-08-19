#!/usr/bin/env python3
"""
RT-DETR Training Script for Wildfire Smoke Detection using Ultralytics 
"""

import os
import torch
from pathlib import Path
from ultralytics import RTDETR

# Set cache directory for Ultralytics models to prevent downloads to project root
cache_dir = Path.home() / ".ultralytics_cache"
cache_dir.mkdir(exist_ok=True)
os.environ['ULTRALYTICS_CACHE_DIR'] = str(cache_dir)

# ============================================================================
# EXPERIMENT CONFIGURATION  
# ============================================================================
EXPERIMENT_NAME = "rtdetr_smoke_detection_v1"
DATASET_PATH = "/vol/bitbucket/si324/rf-detr-wildfire/data/pyro25img"

# Directory structure
project_root = "/vol/bitbucket/si324/rf-detr-wildfire"
outputs_root = os.path.join(project_root, "outputs")
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
    # RT-DETR-X: Higher accuracy (54.8 mAP) better for tiny smoke detection
    # Alternative: "rtdetr-l.pt" (53.1 mAP, faster), "rtdetr-x.pt" (54.8 mAP, better accuracy)
    
    # Change to checkpoints directory so downloaded models go there
    original_dir = os.getcwd()
    os.chdir(checkpoints_dir)
    
    model = RTDETR("rtdetr-x.pt")  # RT-DETR-X for best accuracy on tiny objects
    
    # Change back to original directory
    os.chdir(original_dir)
    
    print("🚀 Starting RT-DETR training...")
    print("📋 Using COCO dataset format with high resolution for tiny objects")
    
    # Train the model - ultralytics can handle COCO format directly
    results = model.train(
        data=f"{DATASET_PATH}/data.yaml",
        epochs=100,
        imgsz=896,         
        batch=4,          
        lr0=0.0001,        
        patience=15,       
        save_period=10,    
        project=experiment_dir,
        name="checkpoints",
        exist_ok=True,
        verbose=True,
        device="cuda",
        amp=True,          
        freeze=None,       
        optimizer="AdamW", 
        cos_lr=True,      
        warmup_epochs=3,  
        box=7.5,           
        cls=0.5,         
        dfl=1.5,          
    )
    
    print(f"✅ RT-DETR training completed!")
    print(f"📁 Checkpoints saved to: {checkpoints_dir}/")
    print(f"💾 Best model: {checkpoints_dir}/weights/best.pt")
    print(f"📊 Training plots: {checkpoints_dir}/results.png")
    
    # Print final metrics
    if results and hasattr(results, 'results_dict'):
        metrics = results.results_dict
        print(f"\n📈 Final Training Metrics:")
        print(f"   mAP50: {metrics.get('metrics/mAP50(B)', 'N/A'):.3f}")
        print(f"   mAP50-95: {metrics.get('metrics/mAP50-95(B)', 'N/A'):.3f}")
    
    return results

if __name__ == "__main__":
    main() 