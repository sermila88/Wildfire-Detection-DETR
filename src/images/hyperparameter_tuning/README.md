# Hyperparameter Tuning for Wildfire Smoke Detection Models

This directory contains **Optuna-based hyperparameter optimization pipelines** for **RF-DETR** and **RT-DETR**.  
Both frameworks aim to maximize **object-level F1 score** by sweeping over confidence thresholds and applying **Non-Maximum Suppression (NMS)** during evaluation.

---

## Directory Structure
```
hyperparameter_tuning/
├── RF-DETR/
│   ├── rf_detr_hparam_tuning_eval_val.py      # Re-evaluate tuned RF-DETR trials with NMS
│   ├── rf_detr_hparam_visualisations.py       # Generate plots/tables from results
│   └── rf_detr_hyperparameter_tuning.py       # Optuna tuning pipeline for RF-DETR
│
└── RT-DETR/
    ├── rt_detr_hparam_tuning_eval_val.py      # Re-evaluate tuned RT-DETR trials with NMS
    ├── rt_detr_hparam_visualisations.py       # Generate plots/tables from results
    └── rt_detr_hyperparameter_tuning.py       # Optuna tuning pipeline for RT-DETR
```

---

## RF-DETR Hyperparameter Tuning

**Scripts:**
- `rf_detr_hyperparameter_tuning.py`  
  Runs Optuna optimization over resolution, batch size, learning rate, weight decay, and epochs.  
  Uses **threshold sweep evaluation** to determine best F1 score per trial.
- `rf_detr_hparam_tuning_eval_val.py`  
  Re-evaluates all RF-DETR trials with **NMS** applied (IoU=0.01).  
  Produces per-trial summaries, confusion counts, and final ranking.
- `rf_detr_hparam_visualisations.py`  
  Creates optimization history plots and **Top-5 configuration tables**.

**Example Run:**
```bash
# Run hyperparameter search
python hyperparameter_tuning/RF-DETR/rf_detr_hyperparameter_tuning.py

# Re-evaluate tuned models with NMS
python hyperparameter_tuning/RF-DETR/rf_detr_hparam_tuning_eval_val.py

# Generate visualizations
python hyperparameter_tuning/RF-DETR/rf_detr_hparam_visualisations.py
```

**Outputs:**
```
src/images/outputs/RF-DETR_hyperparameter_tuning/
    ├── trial_000/
    │   ├── checkpoints/
    │   ├── results.json
    │   └── summary.txt
    ├── logs/
    ├── experiment_config.json
    └── final_summary.json
```

---

## RT-DETR Hyperparameter Tuning

**Scripts:**
- `rt_detr_hyperparameter_tuning.py`  
  Runs Optuna optimization following **official RT-DETR standards** (epochs, image size, learning rate, augmentation, loss weights).  
  Includes full W&B logging and CSV metric parsing.
- `rt_detr_hparam_tuning_eval_val.py`  
  Re-evaluates tuned RT-DETR trials with **NMS applied** (IoU=0.01).  
  Produces per-trial reports and aggregated results.
- `rt_detr_hparam_visualisations.py`  
  Generates **optimization history plots** and Top-5 results tables.

**Example Run:**
```bash
# Run hyperparameter search
python hyperparameter_tuning/RT-DETR/rt_detr_hyperparameter_tuning.py

# Re-evaluate tuned models with NMS
python hyperparameter_tuning/RT-DETR/rt_detr_hparam_tuning_eval_val.py

# Generate visualizations
python hyperparameter_tuning/RT-DETR/rt_detr_hparam_visualisations.py
```

**Outputs:**
```
src/images/outputs/RT-DETR_hyperparameter_tuning/
    ├── trial_000/
    │   ├── checkpoints/
    │   ├── results.json
    │   └── summary.txt
    ├── logs/
    ├── experiment_metadata.json
    └── final_summary.json
```

---

## Methodology

- **Optimization target:** Best **object-level F1 score** at optimal confidence threshold.  
- **Evaluation:** IoU threshold = 0.1, NMS IoU = 0.01.  
- **Confidence thresholds tested:** 0.10 → 0.90 (step = 0.05).  
- **Trial results:** Each trial saves hyperparameters, performance metrics, and visual plots.  
- **Visualization:** Plots show optimization progress and **Top-5 configurations** for both RF-DETR and RT-DETR.

---

## Dependencies
- PyTorch  
- Ultralytics (RT-DETR)  
- RF-DETR (Roboflow implementation)  
- Optuna  
- NumPy, Pandas, OpenCV, Matplotlib  
- Supervision  
- tqdm  
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
