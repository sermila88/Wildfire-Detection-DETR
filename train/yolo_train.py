import argparse
from ultralytics import YOLO
import wandb
import os
import getpass
import shutil
from pathlib import Path
import datetime

# Set Ultralytics cache directory before importing YOLO
user = getpass.getuser()
ultralytics_cache = f"/vol/bitbucket/{user}/rf-detr-wildfire/.cache/ultralytics"
os.makedirs(ultralytics_cache, exist_ok=True)
os.environ['ULTRALYTICS_CACHE_DIR'] = ultralytics_cache

# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================
# Modify this for different experiments
EXPERIMENT_NAME = "yolo_baseline_v1"  # Change this for different experiments

# Dataset path
dataset_path = "/vol/bitbucket/si324/rf-detr-wildfire/data/pyro25img/images"
os.environ["ULTRALYTICS_HUB_DIR"] = f"/vol/bitbucket/{user}/rf-detr-wildfire/outputs/{EXPERIMENT_NAME}/.ultralytics_cache"
os.environ["WANDB_DIR"] = os.path.join(experiment_dir, "wandb")

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

def train_model(model_weights, data_config, epochs=100, img_size=640, batch_size=16, devices=None, project=experiment_dir):
    
    wandb.init(project="wildfire-smoke-baseline", config={
        "experiment_name": EXPERIMENT_NAME,
        "model_weights": model_weights,
        "data_config": data_config,
        "epochs": epochs,
        "img_size": img_size,
        "batch_size": batch_size,
        "devices": devices
    })
    
    print(f"\n🚀 Starting YOLO training...")
    print(f"📦 Model: {model_weights}")
    print(f"📁 Dataset: {data_config}")
    print(f"🔄 Epochs: {epochs}, Batch: {batch_size}, Image size: {img_size}")

    # hyperparameters

    hyperparameters = {
     "lr0": 0.01,
     "lrf": 0.01,
     "momentum": 0.937,
     "weight_decay": 0.0005,
     "warmup_epochs": 3.0,
     "warmup_momentum": 0.8,
     "warmup_bias_lr": 0.1,
     "box": 7.5,
     "cls": 0.5,
     "dfl": 1.5,
     "pose": 12.0,
     "kobj": 2.0,
     "hsv_h": 0.015,
     "hsv_s": 0.7,
     "hsv_v": 0.4,
     "degrees": 0.0,
     "translate": 0.1,
     "scale": 0.5,
     "shear": 0.0,
     "perspective": 0.0,
     "flipud": 0.0,
     "fliplr": 0.5,
     "bgr": 0.0,
     "mosaic": 1.0,
     "mixup": 0.0,
     "copy_paste": 0.0,
     "copy_paste_mode": "flip",
     "auto_augment": "randaugment",
     "erasing": 0.4,
     "crop_fraction": 1.0
    }

    model = YOLO(model_weights)
    
    # Train with optimized hyperparameters
    results = model.train(
        data=data_config, 
        epochs=epochs, 
        imgsz=img_size, 
        batch=batch_size, 
        device=devices, 
        project=experiment_dir,  
        name="",
        exist_ok=True,
        **hyperparameters  # Apply all hyperparameters
    )

    
    print(f"\n🗂️  Organizing outputs...")
    
    # Move model weights to checkpoints/
    weights_src = os.path.join(experiment_dir, "weights")
    if os.path.exists(weights_src):
        for weight_file in os.listdir(weights_src):
            src = os.path.join(weights_src, weight_file)
            dst = os.path.join(checkpoints_dir, weight_file)
            shutil.move(src, dst)
            print(f"   📦 Moved {weight_file} → checkpoints/")

    # Copy best.pt to top-level for convenience
    best_ckpt = os.path.join(checkpoints_dir, "best.pt")
    if os.path.exists(best_ckpt):
        final_ckpt = os.path.join(experiment_dir, "best.pt")
        shutil.copy(best_ckpt, final_ckpt)
        print(f"   🏁 Copied best.pt → {final_ckpt}")
        
    # Move plots and visualizations to plots/
    plot_files = [
        "results.png", "confusion_matrix.png", "confusion_matrix_normalized.png",
        "labels.jpg", "labels_correlogram.jpg", "train_batch0.jpg", 
        "val_batch0_labels.jpg", "val_batch0_pred.jpg", "PR_curve.png", "F1_curve.png"
    ]
    for plot_file in plot_files:
        src = os.path.join(experiment_dir, plot_file)
        if os.path.exists(src):
            dst = os.path.join(plots_dir, plot_file)
            shutil.move(src, dst)
            print(f"   📊 Moved {plot_file} → plots/")

    print(f"\n✅ Training completed!")
    print(f"📁 Checkpoints saved to: {checkpoints_dir}")
    print(f"📊 Plots saved to: {plots_dir}")
    print(f"🔍 To evaluate, use experiment name: {EXPERIMENT_NAME}")

    # Save args.yaml if it was created
    args_yaml = os.path.join(experiment_dir, "args.yaml")
    if os.path.exists(args_yaml):
        shutil.move(args_yaml, os.path.join(logs_dir, "args.yaml"))
        print(f"   📝 Moved args.yaml → logs/")

    wandb.finish()
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a YOLO model.')
    parser.add_argument('--model_weights', type=str, default='yolov8x.pt', help='Path to the pretrained model weights (yolov8x.pt for best baseline).')
    parser.add_argument('--data_config', type=str, default=f'{dataset_path}/data.yaml', help='Path to dataset YAML config file.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--img_size', type=int, default=640, help='Image size for training.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training.')
    parser.add_argument('--devices', type=str, default=None, help='GPUs to use for training (e.g., "0", "0,1").')
    parser.add_argument('--project', type=str, default=None, help='Project directory (uses experiment config by default).')

    args = parser.parse_args()

    devices = [int(d) for d in args.devices.split(',')] if args.devices else None

    # Save log file with run config
    log_file = os.path.join(logs_dir, f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    with open(log_file, 'w') as f:
        f.write(f"Model: {args.model_weights}\nDataset: {args.data_config}\nEpochs: {args.epochs}\n")
        f.write(f"Batch size: {args.batch_size}, Image size: {args.img_size}, Devices: {'CPU' if args.devices is None else args.devices}\n")
    print(f"   📝 Saved run config → {log_file}")

    print("🚀 Starting YOLO Training for Wildfire Smoke Detection")
    print(f"   📦 Model: {args.model_weights}")
    print(f"   📁 Dataset: {args.data_config}")
    print(f"   🔄 Epochs: {args.epochs}")
    print(f"   📏 Image Size: {args.img_size}")
    print(f"   📦 Batch Size: {args.batch_size}")
    print(f"   💻 Device: {'CPU' if devices is None else f'GPU {devices}'}")
    print("-" * 50)

    results = train_model(
        model_weights=args.model_weights,
        data_config=args.data_config,
        epochs=args.epochs,
        img_size=args.img_size,
        batch_size=args.batch_size,
        devices=devices,
        project=args.project
    )
    
    print("\n🎯 Training Summary:")
    print(f"   📁 Experiment: {EXPERIMENT_NAME}")
    print(f"   📂 Results in: {experiment_dir}")
    print(f"   🏆 Model weights: {checkpoints_dir}")
    print(f"   📈 Training plots: {plots_dir}")
    print(f"   📋 Logs: {logs_dir}")
    print(f"   📊 Ready for evaluation!")