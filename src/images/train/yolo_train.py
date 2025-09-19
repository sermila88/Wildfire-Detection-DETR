"""
Baseline YOLOv8 Training Script
"""

import argparse
from ultralytics import YOLO
import wandb

def train_model(model_weights, data_config, epochs=100, img_size=640, batch_size=16, devices=None, project="runs/train"):
    wandb.init(project="YOLO_baseline", config={
        "model_weights": model_weights,
        "data_config": data_config,
        "epochs": epochs,
        "img_size": img_size,
        "batch_size": batch_size,
        "devices": devices
    })
    name = wandb.run.name

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
    results = model.train(
        data=data_config, 
        epochs=epochs, 
        imgsz=img_size, 
        batch=batch_size, 
        device=devices, 
        project=project,
        name=name,
        **hyperparameters
    )

    path_weights = f"{project}/{name}/weights/best.pt"
    print(f"Training completed. Best model weights saved at: {path_weights}")

    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a YOLO model.')
    parser.add_argument('--model_weights', type=str, default='yolov5s.pt', help='Path to the pretrained model weights.')
    parser.add_argument('--data_config', type=str, required=True, help='Path to dataset YAML config file.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--img_size', type=int, default=640, help='Image size for training.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training.')
    parser.add_argument('--devices', type=str, default=None, help='GPUs to use for training (e.g., "0", "0,1").')
    parser.add_argument('--project', type=str, default='runs/train', help='Project directory to save results.')

    args = parser.parse_args()

    devices = [int(d) for d in args.devices.split(',')] if args.devices else None

    train_model(
        model_weights=args.model_weights,
        data_config=args.data_config,
        epochs=args.epochs,
        img_size=args.img_size,
        batch_size=args.batch_size,
        devices=devices,
        project=args.project
    )