#!/usr/bin/env python3
"""
RT-DETR Training Script for Wildfire Smoke Detection using Ultralytics 
"""

import os
import torch
from ultralytics import RTDETR

# ============================================================================
# EXPERIMENT CONFIGURATION  
# ============================================================================
EXPERIMENT_NAME = "rtdetr_smoke_detection_v1"
DATASET_PATH = "/vol/bitbucket/si324/rf-detr-wildfire/data/pyro25img/images"
OUTPUT_DIR = f"/vol/bitbucket/si324/rf-detr-wildfire/outputs/{EXPERIMENT_NAME}"

# ============================================================================
# TRAINING SETUP
# ============================================================================
def main():
    print("🔥 RT-DETR Training for Wildfire Smoke Detection")
    print(f"🎯 Experiment: {EXPERIMENT_NAME}")
    print(f"📁 Dataset: {DATASET_PATH}")
    print(f"💾 Output: {OUTPUT_DIR}")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load RT-DETR model (pre-trained on COCO)
    # RT-DETR-X: Higher accuracy (54.8 mAP) better for tiny smoke detection
    # Alternative: "rtdetr-l.pt" (53.1 mAP, faster), "rtdetr-x.pt" (54.8 mAP, better accuracy)
    model = RTDETR("rtdetr-x.pt")  # RT-DETR-X for best accuracy on tiny objects
    
    print("🚀 Starting RT-DETR training...")
    print("📋 Using COCO dataset format with high resolution for tiny objects")
    
    # Train the model - ultralytics can handle COCO format directly
    results = model.train(
        data=f"{DATASET_PATH}/data.yaml",
        epochs=100,
        imgsz=1280,        
        batch=8,           
        lr0=0.0001,        
        patience=15,       
        save_period=10,    
        project=OUTPUT_DIR,
        name="training",
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
    print(f"📁 Results saved to: {OUTPUT_DIR}/training/")
    print(f"💾 Best model: {OUTPUT_DIR}/training/weights/best.pt")
    print(f"📊 Training plots: {OUTPUT_DIR}/training/results.png")
    
    # Print final metrics
    if hasattr(results, 'results_dict'):
        metrics = results.results_dict
        print(f"\n📈 Final Training Metrics:")
        print(f"   mAP50: {metrics.get('metrics/mAP50(B)', 'N/A'):.3f}")
        print(f"   mAP50-95: {metrics.get('metrics/mAP50-95(B)', 'N/A'):.3f}")
    
    return results

if __name__ == "__main__":
    main() 