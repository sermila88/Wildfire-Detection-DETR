# Training Scripts for Wildfire Smoke Detection Models

This directory contains the training pipelines for the three models evaluated in this project: **RF-DETR**, **RT-DETR**, and **YOLOv8**.  
Each script is designed to train models on the **PyroNear wildfire smoke dataset** (COCO-style annotations).

This training of YOLOv8 replicates the training methodology introduced by [Pyronear](https://pyronear.org).

---

## Directory Structure
```
train/
├── rfdetr_train.py    # RF-DETR training script
├── rtdetr_train.py    # RT-DETR training script
└── yolo_train.py      # YOLOv8 training script
```

---

## RF-DETR Training

**Script:** `rfdetr_train.py`  
**Description:** Training pipeline for RF-DETR using Roboflow’s implementation.  
Includes support for checkpointing, mixed precision, W&B logging, and experiment directory management.

### Example Run
```bash
python src/images/train/rfdetr_train.py
```

This will:
- Train RF-DETR on the PyroNear dataset in COCO format.  
- Save checkpoints and logs under:  
  ```
  src/images/outputs/RF-DETR_initial_training/
      ├── checkpoints/
      └── logs/
  ```

---

## RT-DETR Training

**Script:** `rtdetr_train.py`  
**Description:** Training pipeline for RT-DETR using Ultralytics.  
Configured for **RT-DETR-X** (best accuracy for small smoke plumes).  
Includes W&B integration and custom cache handling.

### Example Run
```bash
python src/images/train/rtdetr_train.py
```

This will:
- Train RT-DETR-X for 50 epochs at resolution 728.  
- Save checkpoints and logs under:  
  ```
  src/images/outputs/RT-DETR_initial_training/
      ├── checkpoints/
      └── logs/
  ```
- Best weights will be located at:
  ```
  src/images/outputs/RT-DETR_initial_training/checkpoints/weights/best.pt
  ```

---

## YOLOv8 Training

**Script:** `yolo_train.py`  
**Description:** Baseline YOLOv8 training pipeline with custom hyperparameters and W&B logging.  
Replication Pyronear’s YOLO training workflow for baseline comparison

### Example Run
```bash
python src/images/train/yolo_train.py \
    --model_weights yolov8x.pt \
    --data_config src/images/data/pyro25img/images/data.yaml \
    --epochs 100 \
    --img_size 640 \
    --batch_size 16 \
    --devices 0 \
    --project src/images/outputs/YOLO_baseline/training_outputs
```

This will:
- Train YOLOv8-X for 100 epochs on the PyroNear dataset.  
- Save outputs to:
  ```
  src/images/outputs/YOLO_baseline/training_outputs/
  ```

---

## Dependencies
All training scripts require:  
- PyTorch  
- Ultralytics (YOLOv8, RT-DETR)  
- RF-DETR (Roboflow implementation)  
- NumPy, OpenCV  
- Weights & Biases (`wandb`)  

---

## Citation
If you use this work, please cite the PyroNear paper:

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
