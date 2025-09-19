# Evaluation Framework for Wildfire Smoke Detection Models

## Overview
This directory contains the complete evaluation pipeline for comparing **YOLOv8**, **RT-DETR**, and **RF-DETR** models on wildfire smoke detection tasks.  
The framework computes both object-level and image-level metrics across different experimental configurations.

---

## Directory Structure
```
eval/
├── Comparative_eval/       # Multi-model comparison scripts
├── RF-DETR/                # RF-DETR specific evaluation
├── RT-DETR/                # RT-DETR specific evaluation
└── YOLOv8/                 # YOLOv8 specific evaluation
```

---

## Evaluation Methodology

### Metrics Computed
- **Object-level**: Precision, Recall, F1-Score (IoU threshold = 0.1)
- **Image-level**: Precision, Recall, F1-Score, Accuracy
- **Confidence threshold optimization**: Evaluates across range [0.1, 0.9] with 0.05 steps
- **Confusion matrices**: Both object and image level

### Key Features
- Unified prediction generation: all models generate predictions at `conf=0.01` initially
- NMS application: IoU threshold of 0.01 for RT-DETR and RF-DETR to match YOLOv8
- Visualization outputs: bounding boxes with TP/FP/FN classification
- Threshold-specific predictions: saved for each confidence threshold

---

## Comparative Evaluation Scripts

### 1. `eval_compare.py`
**Purpose**: Baseline comparison without NMS or hyperparameter tuning  
- Evaluates initial training checkpoints  
- Object-level metrics only  
- Generates comparison visualizations  

### 2. `eval_compare_NMS.py`
**Purpose**: Evaluation with Non-Maximum Suppression applied  
- Applies NMS to RT-DETR and RF-DETR predictions (IoU=0.01)  
- Ensures fair comparison with YOLOv8's built-in NMS  
- Object-level metrics  

### 3. `eval_compare_hparam_tuning.py`
**Purpose**: Evaluation of hyperparameter-tuned models  
- Uses best checkpoints from hyperparameter optimization  
- Includes NMS for transformer models  
- Object-level metrics  

### 4. `final_comparison.py`
**Purpose**: Comprehensive evaluation with both metric levels  
- **Object-level**: individual smoke plume detection  
- **Image-level**: classifying the entire image (caught smoke in the right place or not)  
- Lower IoU threshold (0.01) for increased sensitivity  

---

## Usage

### Evaluation
```bash
# Initial baseline evaluation
python Comparative_eval/eval_compare.py

# With NMS applied
python Comparative_eval/eval_compare_NMS.py

# Hyperparameter-tuned models
python Comparative_eval/eval_compare_hparam_tuning.py

# Final comprehensive evaluation
python Comparative_eval/final_comparison.py
```

---

## Output Structure
Each evaluation script generates results in the following format:

```
eval_results/
└── {model_name}/
    ├── {model}_predictions/         # Raw predictions
    ├── preds_by_threshold/          # Filtered by confidence
    ├── plots/
    │   ├── metrics.png              # P/R/F1 vs threshold
    │   └── conf_matrix.png          # Confusion matrix
    ├── predicted_bounding_boxes/    # Visualization images
    │   ├── TP/
    │   ├── FP/
    │   └── FN/
    ├── evaluation_summary.json      # Detailed results
    └── evaluation_summary.txt       # Human-readable summary
```

---

## Individual Model Evaluation

### YOLOv8
- `pyronear_eval.py`: evaluates YOLOv8 on Pyronear dataset  
- `yolo_generate_predictions.py`: generates YOLO-format predictions  
- `pyronear_utils.py`: utility functions for IoU, format conversion  

### RT-DETR
- `rtdetr_eval.py`: RT-DETR-specific evaluation pipeline  

### RF-DETR
- `rfdetr_eval.py`: RF-DETR-specific evaluation pipeline  

---

## Configuration

Key parameters in scripts:

```python
OUTPUT_BASE_DIR = "/path/to/results"
TEST_IMAGES_DIR = "/path/to/test/images"
GT_FOLDER = "/path/to/ground/truth/labels"
IOU_THRESHOLD = 0.1  # or 0.01 for final_comparison
conf_range = np.arange(0.1, 0.9, 0.05)
```

Model paths:

```python
models = {
    "YOLO_baseline": {
        "type": "YOLO",
        "path": "path/to/yolo/best.pt"
    },
    "RF-DETR": {
        "type": "RF-DETR", 
        "path": "path/to/rfdetr/checkpoint_best_ema.pth"
    },
    "RT-DETR": {
        "type": "RT-DETR",
        "path": "path/to/rtdetr/best.pt"
    }
}
```

---

## Dependencies
- PyTorch  
- Ultralytics (YOLOv8, RT-DETR)  
- OpenCV  
- NumPy, Matplotlib, Seaborn  
- Supervision  
- tqdm  
