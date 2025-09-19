# Utils

This directory contains **utility scripts** for preprocessing, dataset conversion, bounding box visualization, and inference benchmarking.  
Most scripts are lightweight helpers that mirror the logic used in evaluation and training pipelines.
YOLO annotations are converted to COCO JSON format to ensure compatibility with RF-DETR.

---

## Directory Structure
```
utils/
├── inference_comparison/                 # Scripts for inference time benchmarking
│   ├── inference_time_comparison.py
│   └── inference_time_comparison_with_NMS.py
│
├── RF-DETR/
│   ├── rfdetr_generate_GT.py             # Visualize ground truth bounding boxes (COCO)
│   └── rfdetr_generate_pred_bounding_boxes.py  # Predicted bounding boxes with TP/FP/FN/TN
│
├── RT-DETR/
│   └── rtdetr_generate_pred_bounding_boxes.py  # Predicted bounding boxes with TP/FP/FN/TN
│
├── YOLO/
│   ├── yolo_ground_truth_bb.py           # Visualize YOLO-format ground truth bounding boxes
│   └── yolo_predict_bounding_boxes.py    # Visualize YOLO predictions with TP/FP/FN/TN
│
├── convert_yolo_to_coco.py               # Convert YOLO annotations → COCO format (core script)
├── data_breakdown.py                     # Dataset split statistics (smoke vs no-smoke)
└── pyro_img_data_breakdown.py            # Extended breakdown for PyroNear image dataset
```

---

## Key Script: `convert_yolo_to_coco.py`

**Purpose**: Converts **YOLO-format annotations** into **COCO JSON format** for evaluation pipelines.  
This ensures consistency across models (YOLO, RF-DETR, RT-DETR) that rely on COCO-style datasets.

- Supports **train / valid / test** splits  
- Handles images **with and without labels**  
- Counts empty/no-smoke images vs annotated smoke images  
- Produces `_annotations.coco.json` in each split folder  

**Example Usage:**
```bash
python utils/convert_yolo_to_coco.py
```

**Output:**
- Each split folder (`images/train`, `images/valid`, `images/test`) gets a file:
  ```
  _annotations.coco.json
  ```
- JSON contains:
  - `images`: metadata for all images  
  - `annotations`: bounding box annotations (absolute pixel coordinates)  
  - `categories`: smoke class  

---

## Other Utilities 

- **inference_time_comparison.py / inference_time_comparison_with_NMS.py**  
  Benchmark inference latency & FPS for YOLO, RF-DETR, and RT-DETR on CPU.  
  With and without NMS applied.

- **RF-DETR / RT-DETR visualization scripts**  
  Generate predicted bounding boxes with TP/FP/FN/TN breakdowns.  
  Mirror evaluation logic exactly for visual inspection.

- **YOLO visualization scripts**  
  - `yolo_ground_truth_bb.py`: draws YOLO-format ground truth bounding boxes.  
  - `yolo_predict_bounding_boxes.py`: draws YOLO predictions, categorized by TP/FP/FN/TN.

- **data_breakdown.py / pyro_img_data_breakdown.py**  
  Create dataset distribution tables (smoke vs no-smoke images per split).  
  Save results in LaTeX-friendly format for inclusion in reports.


