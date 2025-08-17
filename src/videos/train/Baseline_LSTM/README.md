# Training Utilities for Videos

A script to train a temporal fire detection model using PyTorch Lightning.

## What it does

1. **Loads image sequences** from folders under a given data directory.
2. **Applies transforms** (resize, normalize, or custom pipeline).
3. **Extracts features** with a frozen ResNet50 backbone.
4. **Classifies** sequences via an LSTM + classifier head.
5. **Logs** metrics to Weights & Biases and saves best checkpoints.

## Quick start

```bash
python train.py \
  --data_dir /path/to/data \
  --batch_size 16 \
  --img_size 112 \
  --learning_rate 1e-5 \
  --max_epochs 50 \
  --wandb_project fire_detection
```

## Requirements

See `requirements.txt` for:

```
torch, torchvision, pytorch-lightning,
torchmetrics, wandb, Pillow, numpy
```
