import optuna
import os
import json
import shutil
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from datetime import datetime
from pathlib import Path
from ultralytics import RTDETR
import torch
import supervision as sv
from PIL import Image
from tqdm import tqdm
import gc
import pandas as pd
from scipy.ndimage import uniform_filter1d
import wandb
import logging
import re
from ultralytics.utils.metrics import box_iou 

# Force single GPU training
os.environ['RANK'] = '-1'
os.environ['LOCAL_RANK'] = '-1'
os.environ['WORLD_SIZE'] = '1'

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# Set cache directory for Ultralytics models to prevent downloads to project root
cache_dir = Path.home() / ".ultralytics_cache"  
cache_dir.mkdir(exist_ok=True)
os.environ['ULTRALYTICS_CACHE_DIR'] = str(cache_dir)  

# Enable TF32 for faster training
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Config - Following official RT-DETR standards
EXPERIMENT_NAME = "RT-DETR_hyperparameter_tuning"
PROJECT_ROOT = "/vol/bitbucket/si324/rf-detr-wildfire"
DATASET_PATH = f"{PROJECT_ROOT}/src/images/data/pyro25img/images"
OUTPUT_DIR = f"{PROJECT_ROOT}/src/images/outputs/{EXPERIMENT_NAME}"

# Validation setup (same as RF-DETR)
VALIDATION_DIR = f"{DATASET_PATH}/valid"
VALIDATION_ANNOTATIONS = f"{VALIDATION_DIR}/_annotations.coco.json"

# Evaluation parameters 
IOU_THRESHOLD = 0.1  
CONFIDENCE_THRESHOLDS = np.round(np.linspace(0.10, 0.90, 17), 2)

# Training configuration
N_TRIALS = 25  # Match RF-DETR
EFFECTIVE_BATCH_SIZE = 16  # Target for gradient accumulation


def box_iou_numpy(box1: np.ndarray, box2: np.ndarray, eps: float = 1e-7):
    """Calculate IoU - numpy version for evaluation"""
    if box1.ndim == 1:
        box1 = box1.reshape(1, 4)
    if box2.ndim == 1:
        box2 = box2.reshape(1, 4)
    
    (a1, a2), (b1, b2) = np.split(box1, 2, 1), np.split(box2, 2, 1)
    inter = ((np.minimum(a2, b2[:, None, :]) - np.maximum(a1, b1[:, None, :])).clip(0).prod(2))
    return inter / ((a2 - a1).prod(1) + (b2 - b1).prod(1)[:, None] - inter + eps)

def cache_predictions_rtdetr(model, val_dataset, min_conf=0.01):
    """Cache predictions from RT-DETR (adapted from RF-DETR)"""
    print(f"    Caching predictions from {len(val_dataset)} validation images...")
    all_predictions = []
    
    start_time = time.time()
    
    with torch.inference_mode():
        for path, _, annotations in tqdm(val_dataset, desc="Inference", leave=False):
            # RT-DETR prediction
            results = model.predict(path, conf=min_conf, verbose=False)
            
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confidences = results[0].boxes.conf.cpu().numpy()
            else:
                boxes = np.empty((0, 4), dtype=float)
                confidences = np.empty(0, dtype=float)
            
            all_predictions.append({
                'path': path,
                'annotations': annotations,
                'boxes': boxes,
                'confidences': confidences
            })
    
    inference_time = time.time() - start_time
    print(f"    Inference complete in {inference_time:.1f} seconds")
    return all_predictions


def evaluate_at_threshold(cached_predictions, confidence_threshold):
    """
    Evaluate cached predictions at specific threshold 
    """
    tp = fp = fn = 0
    
    for pred_data in cached_predictions:
        # Filter by confidence 
        boxes_all = pred_data['boxes']
        if boxes_all.size == 0:
            filtered_boxes = np.empty((0, 4), dtype=float)
        else:
            filtered_boxes = boxes_all[pred_data['confidences'] >= confidence_threshold]

        # Get ground truth boxes
        gt_boxes = np.array(pred_data['annotations'].xyxy)
        gt_matches = np.zeros(len(gt_boxes), dtype=bool) if gt_boxes.size > 0 else np.array([])

        # Process each prediction 
        for pred_box in filtered_boxes:
            if gt_boxes.size > 0: 
                # Find the best match by IoU
                iou_values = [box_iou_numpy(pred_box, gt_box)[0, 0] for gt_box in gt_boxes]
                max_iou = max(iou_values)
                best_match_idx = np.argmax(iou_values)
                
                # Check for a valid and unique match
                if max_iou > IOU_THRESHOLD and not gt_matches[best_match_idx]:
                    tp += 1
                    gt_matches[best_match_idx] = True
                else:
                    fp += 1
            else:
                fp += 1  #If no GT boxes, prediction is FP
        
        # Count unmatched GT boxes as FN
        if gt_boxes.size > 0:  
            fn += len(gt_boxes) - np.sum(gt_matches)
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * ( precision * recall ) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return f1, precision, recall, int(tp), int(fp), int(fn)

def find_best_threshold(cached_predictions, confidence_thresholds):
    """
    Find best confidence threshold for this model
    """
    results = []
    
    for conf in confidence_thresholds:
        f1, prec, rec, tp, fp, fn = evaluate_at_threshold(cached_predictions, conf)
        results.append({
            'confidence': conf,
            'f1_score': f1,
            'precision': prec,
            'recall': rec,
            'tp': tp,
            'fp': fp,
            'fn': fn
        })
    
    # Find best
    best_idx = np.argmax([r['f1_score'] for r in results])
    return results[best_idx], results

def parse_ultralytics_metrics(trial_dir):
    """Parse RT-DETR training metrics from CSV"""
    metrics = {
        'epochs': [],
        'train_loss': [],
        'val_loss': [],
        'mAP50': [],
        'mAP50_95': []
    }
    
    results_csv = os.path.join(trial_dir, "checkpoints", "results.csv")
    if os.path.exists(results_csv):
        df = pd.read_csv(results_csv)
        df.columns = [col.strip() for col in df.columns]
        
        if 'epoch' in df.columns:
            metrics['epochs'] = df['epoch'].tolist()
        if 'train/box_loss' in df.columns:
            metrics['train_loss'] = df['train/box_loss'].tolist()
        if 'val/box_loss' in df.columns:
            metrics['val_loss'] = df['val/box_loss'].tolist()
        if 'metrics/mAP50(B)' in df.columns:
            metrics['mAP50'] = df['metrics/mAP50(B)'].tolist()
        if 'metrics/mAP50-95(B)' in df.columns:
            metrics['mAP50_95'] = df['metrics/mAP50-95(B)'].tolist()
    
    return metrics

def plot_training_curves(trial_dir, metrics, experiment_name=EXPERIMENT_NAME):
    """Generate comprehensive training plots"""
    if len(metrics['epochs']) < 2:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Training Loss
    if metrics['train_loss']:
        axes[0, 0].plot(metrics['epochs'], metrics['train_loss'], 'b-', linewidth=2)
        axes[0, 0].set_title('Training Loss', fontsize=12, weight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Mark minimum
        if metrics['train_loss']:
            min_loss = min(metrics['train_loss'])
            min_epoch = metrics['epochs'][metrics['train_loss'].index(min_loss)]
            axes[0, 0].scatter(min_epoch, min_loss, color='red', s=100, zorder=5)
            axes[0, 0].text(min_epoch, min_loss, f' Min: {min_loss:.4f}', fontsize=9)
    
    # 2. mAP 
    if metrics['mAP50']:
        axes[0, 1].plot(metrics['epochs'][:len(metrics['mAP50'])], metrics['mAP50'], 'g-', linewidth=2)
        axes[0, 1].set_title('Validation mAP (RT-DETR Internal)', fontsize=12, weight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('mAP')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].text(0.5, 0.98, '(Not used for optimization)', 
                       transform=axes[0, 1].transAxes, ha='center', 
                       fontsize=10, color='red', weight='bold')
    else:
        axes[0, 1].text(0.5, 0.5, 'mAP not available', ha='center', va='center')
        axes[0, 1].set_title('Validation mAP')
    
    # 3. Loss trajectory
    if metrics['train_loss']:
        axes[1, 0].plot(metrics['epochs'], metrics['train_loss'], 'b-', alpha=0.7, linewidth=2)
        
        # Add smoothed line
        if len(metrics['train_loss']) > 5:
            smoothed = uniform_filter1d(metrics['train_loss'], size=5, mode='nearest')
            axes[1, 0].plot(metrics['epochs'], smoothed, 'r-', linewidth=2, label='Smoothed')
        
        axes[1, 0].set_title('Loss Trajectory', fontsize=12, weight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Training Summary
    axes[1, 1].axis('off')
    summary_text = "TRAINING SUMMARY\n" + "="*20 + "\n\n"
    
    if metrics['epochs']:
        summary_text += f"Total Epochs: {max(metrics['epochs'])}\n\n"
    
    if metrics['train_loss']:
        summary_text += f"Initial Loss: {metrics['train_loss'][0]:.4f}\n"
        summary_text += f"Final Loss: {metrics['train_loss'][-1]:.4f}\n"
        summary_text += f"Min Loss: {min(metrics['train_loss']):.4f}\n"
        summary_text += f"Loss Reduction: {(metrics['train_loss'][0] - metrics['train_loss'][-1]) / metrics['train_loss'][0] * 100:.1f}%\n\n"
    
    if metrics['mAP50']:
        summary_text += f"Final mAP50: {metrics['mAP50'][-1]:.4f}\n"
    
    axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes,
                   fontsize=11, verticalalignment='top', family='monospace')
    
    plt.suptitle(f'{EXPERIMENT_NAME} Training Analysis', fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(trial_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()

def plot_threshold_analysis(trial_dir, threshold_results, best_result, experiment_name=EXPERIMENT_NAME):
    """Plot confidence threshold sweep"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    confs = [r['confidence'] for r in threshold_results]
    f1_scores = [r['f1_score'] for r in threshold_results]
    precisions = [r['precision'] for r in threshold_results]
    recalls = [r['recall'] for r in threshold_results]
    
    # F1 vs threshold
    ax1.plot(confs, f1_scores, 'b-', marker='o', linewidth=2, markersize=8, label='F1 Score')
    ax1.scatter(best_result['confidence'], best_result['f1_score'], 
               color='red', s=200, marker='*', zorder=5)
    ax1.text(best_result['confidence'], best_result['f1_score'] + 0.02,
            f"Best: {best_result['f1_score']:.3f}\n@ conf={best_result['confidence']:.1f}",
            ha='center', fontsize=10, weight='bold')
    ax1.set_xlabel('Confidence Threshold', fontsize=12)
    ax1.set_ylabel('F1 Score', fontsize=12)
    ax1.set_title('F1 Score vs Confidence Threshold', fontsize=12, weight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])
    
    # Precision/Recall vs threshold
    ax2.plot(confs, precisions, 'orange', marker='^', linewidth=2, markersize=7, label='Precision')
    ax2.plot(confs, recalls, 'green', marker='v', linewidth=2, markersize=7, label='Recall')
    ax2.axvline(x=best_result['confidence'], color='red', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Confidence Threshold', fontsize=12)
    ax2.set_ylabel('Score', fontsize=12)
    ax2.set_title('Precision/Recall vs Confidence Threshold', fontsize=12, weight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    plt.suptitle(f'{EXPERIMENT_NAME} Threshold Analysis - Best F1: {best_result["f1_score"]:.4f} @ {best_result["confidence"]:.1f}',
                fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(trial_dir, 'threshold_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()

def objective(trial):
    """
    Optuna objective function for RT-DETR hyperparameter optimization.
    Based on official RT-DETR paper and repository recommendations with full logging.
    """
    start_time = datetime.now()
    trial_start_time = start_time.isoformat()

    trial_start = time.time()
    
    try:
        epochs = trial.suggest_int("epochs", 36, 72, step=12)  # 3x to 9x schedule
        batch = trial.suggest_categorical("batch", [4, 8, 16, 32])
        # Learning rate: Official RT-DETR uses 1e-4 base LR
        lr0 = trial.suggest_float("lr0", 5e-5, 5e-4, log=True)
        # Image size: Official RT-DETR primarily uses 640x640
        imgsz = trial.suggest_categorical("imgsz", [640, 896, 1024])
        # Optimizer: Official RT-DETR uses AdamW
        optimizer = trial.suggest_categorical("optimizer", ["AdamW", "SGD"])
        # Weight decay: Important for transformer models
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
        # Patience for early stopping
        patience = trial.suggest_int("patience", 20, 50, step=10)
        # Loss function weights (RT-DETR specific)
        cls_loss = trial.suggest_float("cls", 1.0, 4.0)
        bbox_loss = trial.suggest_float("bbox", 2.0, 8.0)
        giou_loss = trial.suggest_float("giou", 1.0, 3.0)
        # Focal loss alpha and gamma (for classification)
        focal_alpha = trial.suggest_float("focal_alpha", 0.2, 0.3)
        focal_gamma = trial.suggest_float("focal_gamma", 1.5, 2.5)
        # Learning rate schedule
        cos_lr = trial.suggest_categorical("cos_lr", [True, False])
        warmup_epochs = trial.suggest_int("warmup_epochs", 1, 5)
        # Augmentation parameters 
        mixup = trial.suggest_float("mixup", 0.0, 0.2)
        mosaic = trial.suggest_float("mosaic", 0.8, 1.0)
        dropout = trial.suggest_float("dropout", 0.0, 0.2)

        # Create trial directory 
        trial_dir = os.path.join(OUTPUT_DIR, f"trial_{trial.number:03d}")
        os.makedirs(trial_dir, exist_ok=True)
        os.makedirs(os.path.join(trial_dir, "checkpoints"), exist_ok=True)

        # Collect all hyperparameters
        hp = {
            'epochs': epochs,
            'batch_size': batch,
            'lr0': lr0,
            'imgsz': imgsz,
            'optimizer': optimizer,
            'weight_decay': weight_decay,
            'patience': patience,
            'cls': cls_loss,
            'bbox': bbox_loss,
            'giou': giou_loss,
            'focal_alpha': focal_alpha,
            'focal_gamma': focal_gamma,
            'cos_lr': cos_lr,
            'warmup_epochs': warmup_epochs,
            'mixup': mixup,
            'mosaic': mosaic,
            'dropout': dropout,
        }

        # Save hyperparameters
        with open(os.path.join(trial_dir, "hyperparameters.json"), 'w') as f:
            json.dump(hp, f, indent=2, cls=NumpyEncoder)
        
        print(f"\n🎯 Trial {trial.number}: lr0={lr0:.2e}, imgsz={imgsz}, batch={batch}, epochs={epochs}")
        print(f"   Loss weights - cls: {cls_loss:.2f}, bbox: {bbox_loss:.2f}, giou: {giou_loss:.2f}")
        print(f"   📁 Trial directory: {trial_dir}")

        # Change to checkpoints dir for model downloads
        original_dir = os.getcwd()  # type: ignore
        os.chdir(os.path.join(trial_dir, "checkpoints"))  # type: ignore
        
        # Use RT-DETR-X for best performance (as per official benchmarks)
        model = RTDETR("rtdetr-x.pt")
        os.chdir(original_dir)  # type: ignore

        # Train with official RT-DETR style hyperparameters
        training_start = datetime.now()
        
        results = model.train(
            data=f"{DATASET_PATH}/data.yaml",
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            lr0=lr0,
            weight_decay=weight_decay,
            patience=patience,
            save_period=max(5, epochs // 10),  # Save more frequently for long runs
            project=trial_dir,
            name="checkpoints",
            exist_ok=True,
            verbose=True,
            device="cuda" if torch.cuda.is_available() else "cpu",
            amp=True,
            freeze=None,
            optimizer=optimizer,
            cos_lr=cos_lr,
            warmup_epochs=warmup_epochs,
            workers=8,
            nbs=16,
   
            # RT-DETR specific loss weights
            cls=cls_loss,
            box=bbox_loss,  # Combined bbox loss weight
            dfl=1.5,  # DFL loss (fixed based on paper)
            
            # Augmentation (important for detection)
            mixup=mixup,
            mosaic=mosaic,
            copy_paste=0.1,  # Standard for detection
            
            # Advanced training settings
            close_mosaic=max(1, epochs // 10),  # Close mosaic in last 10% of training
            mask_ratio=1,  # For detection
            
            # Validation settings
            val=True,
            plots=True,
            save_json=True,  # For detailed evaluation
        )
        
        training_time = (datetime.now() - training_start).total_seconds()

        # Parse training metrics
        training_metrics = parse_ultralytics_metrics(trial_dir)
        plot_training_curves(trial_dir, training_metrics)

        # Load best checkpoint for evaluation
        checkpoint_path = os.path.join(trial_dir, "checkpoints", "weights", "best.pt")
        if not os.path.exists(checkpoint_path):
            checkpoint_path = os.path.join(trial_dir, "checkpoints", "weights", "last.pt")

        print(f"  Loading checkpoint for evaluation...")
        eval_model = RTDETR(checkpoint_path)

        # Load validation dataset
        val_ds = sv.DetectionDataset.from_coco(
            images_directory_path=VALIDATION_DIR,
            annotations_path=VALIDATION_ANNOTATIONS
        )
        print(f"  Loaded {len(val_ds)} validation images")

        # Cache predictions and find best threshold
        cached_predictions = cache_predictions_rtdetr(eval_model, val_ds, min_conf=0.01)
        best_result, all_threshold_results = find_best_threshold(cached_predictions, CONFIDENCE_THRESHOLDS)

        print(f"  Best F1: {best_result['f1_score']:.4f} @ confidence={best_result['confidence']:.1f}")
        print(f"  Precision: {best_result['precision']:.4f}, Recall: {best_result['recall']:.4f}")

        # Plot threshold analysis
        plot_threshold_analysis(trial_dir, all_threshold_results, best_result)

        score = best_result['f1_score']  # THIS IS WHAT WE OPTIMIZE!

        # Clean up eval model
        del eval_model
        torch.cuda.empty_cache()

        # Save comprehensive results 
        results = {
            "trial_number": trial.number,
            "hyperparameters": hp,
            "training_time_hours": training_time / 3600,
            "total_trial_time_hours": (time.time() - trial_start) / 3600,
            "best_result": best_result,
            "all_threshold_results": all_threshold_results,
            "training_metrics": training_metrics,
            "timestamp": datetime.now().isoformat(),
            "hardware": {
                "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU",
                "cuda_version": torch.version.cuda,
                "torch_version": torch.__version__,
            }
        }

        with open(os.path.join(trial_dir, "results.json"), 'w') as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)

        # Save summary (match RF-DETR format)
        with open(os.path.join(trial_dir, "summary.txt"), 'w') as f:
            f.write(f"TRIAL {trial.number} SUMMARY\n")
            f.write("="*60 + "\n\n")
            
            f.write("HYPERPARAMETERS:\n")
            for k, v in hp.items():
                f.write(f"  {k}: {v}\n")
            
            f.write(f"\nTRAINING:\n")
            f.write(f"  Training time: {training_time/3600:.1f} hours\n")
            if training_metrics['train_loss']:
                f.write(f"  Initial loss: {training_metrics['train_loss'][0]:.4f}\n")
                f.write(f"  Final loss: {training_metrics['train_loss'][-1]:.4f}\n")
            
            f.write(f"\nFINAL RESULTS:\n")
            f.write(f"  Best F1 Score: {best_result['f1_score']:.4f}\n")
            f.write(f"  Best Confidence: {best_result['confidence']:.1f}\n")
            f.write(f"  Precision: {best_result['precision']:.4f}\n")
            f.write(f"  Recall: {best_result['recall']:.4f}\n")
            f.write(f"  TP={best_result['tp']}, FP={best_result['fp']}, FN={best_result['fn']}\n")

        logging.info(f"Trial {trial.number}: F1={best_result['f1_score']:.4f} @ conf={best_result['confidence']:.1f}")

        # GPU cleanup
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        return score

    except Exception as e:
        print(f"❌ Trial {trial.number} failed: {str(e)}")
        
        # Save failure information with full context
        failure_info = {
            'trial_number': trial.number,
            'hyperparameters': trial.params,
            'error': str(e),
            'error_type': type(e).__name__,
            'timestamp': datetime.now().isoformat(),
            'partial_training_time': (datetime.now() - start_time).total_seconds() / 3600,
            'status': 'failed'
        }
        
        trial_dir = os.path.join(OUTPUT_DIR, f"trial_{trial.number:03d}")
        
        with open(os.path.join(trial_dir, "failure_info.json"), "w") as f:
            json.dump(failure_info, f, indent=2)
        
        # Cleanup 
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        raise optuna.TrialPruned(f"Trial failed: {str(e)}")


def main():
    """
    Run RT-DETR hyperparameter optimization following official best practices with full resumability.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Setup logging 
    os.makedirs(os.path.join(OUTPUT_DIR, "logs"), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(OUTPUT_DIR, "logs", "optimization.log")),
            logging.StreamHandler()
        ]
    )
    
    print("\n" + "="*60)
    print("RT-DETR HYPERPARAMETER OPTIMIZATION")
    print("="*60)
    print(f"📁 Experiment: {EXPERIMENT_NAME}")
    print(f"🎯 Target: F1 Score (matching RF-DETR)")
    print(f"📊 Method: Smart threshold sweep per model")
    print(f"🔍 IoU: {IOU_THRESHOLD}")
    print(f"🔢 Trials: {N_TRIALS}")
    print("="*60 + "\n")
    
    # Save global experiment metadata
    experiment_metadata = {
        'experiment_name': EXPERIMENT_NAME,
        'start_time': datetime.now().isoformat(),
        'dataset_path': DATASET_PATH,
        'output_dir': OUTPUT_DIR,
        'total_planned_trials': N_TRIALS,
        'expected_duration_days': 4,
        'model_variant': 'rtdetr-x',
        'optimization_objective': 'F1_score_with_threshold_sweep',
        'gpu_info': {
            'device_name': torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU',
            'memory_total_gb': torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0
        },
        'system_info': {
            'pytorch_version': torch.__version__,
            'cuda_version': getattr(torch.version, 'cuda', None) if torch.cuda.is_available() else None,
            'ultralytics_available': True,
        }
    }
    
    with open(os.path.join(OUTPUT_DIR, "experiment_metadata.json"), 'w') as f:
        json.dump(experiment_metadata, f, indent=2)
    
    # Create Optuna study with appropriate sampler
    study = optuna.create_study(
        direction="maximize",  # Maximize F1 score
        study_name="rtdetr_official_tuning",
        storage=f"sqlite:///{OUTPUT_DIR}/study.db",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(
            seed=42,
            n_startup_trials=5,  # Random trials before TPE
            n_ei_candidates=24,  # More candidates for better optimization
        ),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=10,
            interval_steps=5,
        )
    )
    
    try:
        # Run optimization - more trials for thorough search
        study.optimize(objective, n_trials=N_TRIALS, timeout=None)
        
        print(f"\n🏆 OPTIMIZATION COMPLETE!")
        print(f"🥇 Best F1 Score: {study.best_value:.4f}")
        print(f"🔧 Best hyperparameters:")
        for param, value in study.best_params.items():
            print(f"   {param}: {value}")
        
        # Save comprehensive results with analysis
        final_results = {
            'experiment_info': experiment_metadata,
            'completion_time': datetime.now().isoformat(),
            'best_trial': {
                'score': study.best_value,
                'hyperparameters': study.best_params,
                'trial_number': study.best_trial.number,
            },
            'optimization_summary': {
                'total_trials': len(study.trials),
                'completed_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
                'failed_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]),
                'pruned_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
                'total_optimization_hours': sum(t.duration.total_seconds() for t in study.trials if t.duration) / 3600,
            },
            'all_trials': [{
                'number': t.number,
                'value': t.value,
                'params': t.params,
                'state': str(t.state),
                'duration_hours': t.duration.total_seconds() / 3600 if t.duration else None,
            } for t in study.trials]
        }
        
        with open(os.path.join(OUTPUT_DIR, "final_summary.json"), "w") as f:
            json.dump(final_results, f, indent=2)
        
        print(f"\n💾 Complete results saved to: {OUTPUT_DIR}/final_summary.json")
        print(f"📁 Best model directory: {OUTPUT_DIR}/trial_{study.best_trial.number:03d}/")
        print(f"📊 Study database: {OUTPUT_DIR}/study.db")
        
        # Print optimization insights
        print(f"\n📈 Optimization Insights:")
        print(f"   Best score achieved in trial {study.best_trial.number}")
        total_time = sum(t.duration.total_seconds() for t in study.trials if t.duration) / 3600
        print(f"   Total optimization time: {total_time:.1f} hours ({total_time/24:.1f} days)")
        
        
        return study.best_params
        
    except KeyboardInterrupt:
        print(f"\n Optimization interrupted by keyboard")
        print(f"💾 Current results saved to: {OUTPUT_DIR}/")
        
        # Save interrupted state with full context
        interrupted_results = {
            'experiment_info': experiment_metadata,
            'interruption_time': datetime.now().isoformat(),
            'status': 'interrupted',
            'completed_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            'total_planned_trials': N_TRIALS,
            'current_best_score': study.best_value if study.best_value else None,
            'current_best_params': study.best_params if study.best_params else None,
            'partial_optimization_hours': sum(t.duration.total_seconds() for t in study.trials if t.duration) / 3600,
            'resume_instructions': "Use the same script to resume from existing study.db - Optuna will automatically continue"
        }
        
        with open(os.path.join(OUTPUT_DIR, "interrupted_state.json"), "w") as f:
            json.dump(interrupted_results, f, indent=2)
        
        print(f"💾 Interrupted state saved. Resume by re-running the same script.")
        return None
        
    except Exception as e:
        print(f"\n❌ Optimization failed: {str(e)}")
        print(f"💾 Partial results saved to: {OUTPUT_DIR}/")
        
        return None


if __name__ == "__main__":
    main() 