# Experiment Output Structure

This project uses an organized experiment structure to keep all outputs clean and manageable.

## 📁 Directory Structure

```
outputs/
├── {EXPERIMENT_NAME}/
│   ├── checkpoints/     # Model weights and training checkpoints
│   ├── plots/          # All visualization plots (mAP, confusion matrix, etc.)
│   ├── logs/           # Training logs and metrics
│   └── eval_results/   # Evaluation results (JSON and summary files)

bounding_boxes/         # Generated bounding box visualizations
├── Ground_truth/       # Ground truth annotations (splits: train/valid/test)
└── predictions/        # Model predictions organized by experiment
    └── {EXPERIMENT_NAME}/
        ├── train/
        ├── valid/
        └── test/

utils/                  # Utility scripts and helpers 
├── generate_GT.py      # Generate ground truth bounding box images
└── generate_predictions.py  # Generate prediction bounding box images
```

## 🎯 How to Use

### 1. Training

In `train/train.py`, modify the experiment name:

```python
EXPERIMENT_NAME = "rfdetr_smoke_detection_v1"  # Change this for different experiments
```

Run training:
```bash
python train/train.py
```

This will create:
- `outputs/rfdetr_smoke_detection_v1/checkpoints/` - Contains model weights
- Directory structure for this experiment

### 2. Evaluation

In `eval/eval_final.py`, use the **same experiment name**:

```python
EXPERIMENT_NAME = "rfdetr_smoke_detection_v1"  # Must match training!
```

Run evaluation:
```bash
python eval/eval_final.py
```

This will create:
- `outputs/rfdetr_smoke_detection_v1/eval_results/` - JSON results and summary
- `outputs/rfdetr_smoke_detection_v1/plots/` - Visualization plots

## 🔄 Multiple Experiments

To run different experiments:

1. **Change experiment name** in both `train.py` and `eval_final.py`
2. **Train the model** with the new name
3. **Evaluate** using the same name

Example:
```python
# Experiment 1
EXPERIMENT_NAME = "rfdetr_smoke_detection_v1"

# Experiment 2 
EXPERIMENT_NAME = "smoke_detection_lowres_v2"

# Experiment 3
EXPERIMENT_NAME = "smoke_detection_augmented_v3"
```

## 📊 Output Files

### Training Outputs
- `checkpoints/checkpoint_best.pth` - Best model weights
- `checkpoints/checkpoint_best_ema.pth` - EMA-smoothed weights 
- `checkpoints/metrics_plot.png` - Training metrics plot

### Evaluation Outputs  
- `eval_results/evaluation_results.json` - Complete metrics in JSON format
- `eval_results/evaluation_summary.txt` - Human-readable summary
- `plots/map_plot.png` - mAP visualization
- `plots/confusion_matrix.png` - Confusion matrix plot

