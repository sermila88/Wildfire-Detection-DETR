# Wildfire Smoke Detection Framework

This repository provides a complete pipeline for training, evaluating, and comparing object detection models for **early wildfire smoke detection**.  
It supports **YOLOv8**, **RT-DETR**, and **RF-DETR** models, with utilities for dataset conversion, hyperparameter tuning, evaluation, and visualization.  
The framework extends the evaluation methodology introduced in [PyroNear](https://pyronear.org) replicating their YOLOv8 baseline and adapting their evaluation logic for wildfire smoke detection models.

---

## Directory Structure
```bash
src/images
├── data/ # Datasets (YOLO/COCO formatted)
├── train/ # Training scripts for YOLO, RT-DETR, RF-DETR
├── eval/ # Evaluation pipelines (object-level & image-level)
├── hyperparameter_tuning/ # Optuna-based tuning for RT-DETR & RF-DETR
└── utils/ # Dataset conversion, bounding box visualization, inference comparison
```
---

## Datasets (`data/`)

This repository supports multiple wildfire smoke datasets used for evaluation and comparison:  

- **pyro25img/** – Main dataset (YOLO format: `images/` + `labels/`), used for training and primary evaluation.  
- **Fuego/** – Test set derived from the Fuego project, annotated from the HPWREN camera network with wildfire data from Cal Fire [Govil et al., 2020; Team Fuego, 2020].  
- **Nemo/** – Validation split from the Nemo dataset, which provides multi-level smoke intensity labels (low, mid, high) [Yazdi et al., 2022].  
- **AI For Mankind/** – Test set developed by AI For Mankind, annotated with the help of volunteers [AI For Mankind, 2020; AI For Mankind, 2025].  
- **SmokeFrames/** – Subsampled by PyroNear from the SmokeNet ProjectX video dataset [De Schaetzen et al., 2020].  

These external datasets provide diverse testing conditions with different regions, camera perspectives, and smoke characteristics, enabling assessment of model generalisation beyond the PyroNear test set.

Each dataset has YOLO annotations, own `data.yaml` (for RT-DETR training) and COCO annotations (`_annotations.coco.json`) for RF-DETR

---

## Training (`train/`)

Scripts for model training:

- `yolo_train.py` – Train YOLOv8 models on smoke datasets  
- `rtdetr_train.py` – Train RT-DETR models (Ultralytics)  
- `rfdetr_train.py` – Train RF-DETR models (Roboflow)

Training logs and checkpoints are saved under `outputs/{MODEL}/...`.

---

## Evaluation (`eval/`)

Comprehensive evaluation pipelines for comparing models at both **object-level** (individual smoke plume bounding boxes) and **image-level** (smoke/no-smoke classification with spatial overlap):

- `Comparative_eval/`
  - `eval_compare.py` – Baseline comparison  
  - `eval_compare_NMS.py` – Comparison with NMS applied to DETRs  
  - `eval_compare_hparam_tuning.py` – Comparison after hyperparameter tuning  
  - `final_comparison.py` – Full object + image-level evaluation (IoU = 0.01)

- **YOLOv8/**: `pyronear_eval.py`, `yolo_generate_predictions.py`  
- **RT-DETR/**: `rtdetr_eval.py`  
- **RF-DETR/**: `rfdetr_eval.py`

Outputs include:
- Precision, Recall, F1-score (IoU thresholds 0.01 & 0.1)  
- Confidence-threshold sweeps  
- Confusion matrices (object + image-level)  
- Bounding box visualizations (TP/FP/FN/TN)  
- JSON + human-readable summaries  

---

## Hyperparameter Tuning (`hyperparameter_tuning/`)

Optuna-based tuning scripts with automatic checkpointing, metric logging, and visualization:

- **RF-DETR/**:  
  - `rf_detr_hyperparameter_tuning.py`  
  - `rf_detr_hparam_eval_val.py`  
  - `rf_detr_hparam_visualisations.py`

- **RT-DETR/**:  
  - `rt_detr_hyperparameter_tuning.py`  
  - `rt_detr_hparam_eval_val.py`  
  - `rt_detr_hparam_visualisations.py`

---

## Utilities (`utils/`)

Helper scripts for dataset preparation, visualization, and inference comparison:

- **Dataset conversion**
  - `convert_yolo_to_coco.py`  
    Converts YOLO-format annotations (`class x_center y_center w h`) into **COCO JSON format**,  
    required for training and evaluating **RF-DETR**.  
    Ensures all images (including empty-label “no-smoke” images) are included.

- **Dataset breakdown**
  - `pyro_img_data_breakdown.py` – Counts smoke vs no-smoke images per split and generates summary tables.

- **Bounding box visualization**
  - `YOLO/yolo_ground_truth_bb.py` – Visualize YOLO ground-truth bounding boxes  
  - `YOLO/yolo_predict_bounding_boxes.py` – Visualize YOLO predictions with TP/FP/FN breakdown  
  - `RF-DETR/rfdetr_generate_GT.py` – Generate RF-DETR ground-truth bounding boxes (COCO format)  
  - `RF-DETR/rfdetr_generate_pred_bounding_boxes.py` – RF-DETR predictions visualization  
  - `RT-DETR/rtdetr_generate_pred_bounding_boxes.py` – RT-DETR predictions visualization  

- **Inference comparison**
  - `inference_comparison/inference_time_comparison.py` – CPU inference time per model (latency, FPS)  
  - `inference_comparison/inference_time_with_NMS.py` – Same as above, with NMS applied  

---

## Dependencies

- Python 3.10+  
- PyTorch  
- Ultralytics (YOLOv8, RT-DETR)  
- [RF-DETR](https://github.com/roboflow/rf-detr)  
- OpenCV, NumPy, Matplotlib, Seaborn  
- Supervision  
- tqdm, Optuna, psutil  

---

## Citation

If you use this framework or datasets, please cite:  

```bibtex
@misc{lostanlen2024scrappingwebearlywildfire,
      title={Scrapping The Web For Early Wildfire Detection: A New Annotated Dataset of Images and Videos of Smoke Plumes In-the-wild}, 
      author={Mateo Lostanlen and Nicolas Isla and Jose Guillen and Felix Veith and Cristian Buc and Valentin Barriere},
      year={2024},
      eprint={2402.05349},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2402.05349}, 
}
```
### Model Repositories Referenced
- Ultralytics. [YOLOv8](https://github.com/ultralytics/ultralytics). GitHub.  
- Ultralytics. [RT-DETR: Real-Time Detection Transformer](https://github.com/bharath5673/RT-DETR). GitHub.  
- Roboflow. [RF-DETR: Roboflow’s Detection Transformer](https://github.com/roboflow/rf-detr). GitHub.  

Additional external datasets referenced in this repository:

Govil K, Welch ML, Ball JT, Pennypacker CR. Preliminary Results from a Wildfire Detection System Using Deep Learning on Remote Camera Images. Remote Sensing. 2020;12(1):166. https://doi.org/10.3390/rs12010166

FUEGO Project. Firecam: FUEGO Wildfire Detection Repository. GitHub, 2020. https://github.com/fuego-dev/firecam

Yazdi A, Qin H, Jordan C, Yang L, Yan F. NEMO: An Open-Source Transformer-Supercharged Benchmark for Fine-Grained Wildfire Smoke Detection. Remote Sensing. 2022;14(16):3979. https://doi.org/10.3390/rs14163979

NEMO Dataset. GitHub, 2022. https://github.com/SayBender/Nemo

AI For Mankind. Open Wildfire Smoke Datasets. GitHub, 2020. https://github.com/aiformankind/wildfire-smoke-dataset

AI For Mankind. AI For Mankind. Website, 2025. https://aiformankind.org/

De Schaetzen R, Menoni R, Chang C, Chen Y, Hasani D. Smoke Detection Model on the ALERTWildfire Camera Network. ProjectX Report, 2020. https://rdesc.dev/project_x_final.pdf