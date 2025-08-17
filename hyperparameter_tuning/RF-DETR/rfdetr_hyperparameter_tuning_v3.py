"""
===========================================================
RF-DETR Hyperparameter Tuning for Wildfire Smoke Detection
OPTIMIZATION: Object-Level F1 Score 
===========================================================
"""

import optuna
import torch
import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from datetime import datetime
from pathlib import Path
from rfdetr import RFDETRBase
import supervision as sv
from PIL import Image
from tqdm import tqdm
import gc
import re
import logging
from scipy.ndimage import uniform_filter1d 

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_id = torch.cuda.current_device()
    print(f"🚀 Using GPU {gpu_id}: {gpu_name}")
else:
    print("⚠️ No GPU detected, falling back to CPU")


# ============================================================================
# CONFIGURATION
# ============================================================================
EXPERIMENT_NAME = "rf_detr_hyperparameter_tuning_v3.1"
PROJECT_ROOT = "/vol/bitbucket/si324/rf-detr-wildfire"
DATASET_DIR = f"{PROJECT_ROOT}/data/pyro25img/images"

# VALIDATION SET ONLY!
VALIDATION_DIR = f"{DATASET_DIR}/valid"
VALIDATION_ANNOTATIONS = f"{VALIDATION_DIR}/_annotations.coco.json"

OUTPUT_DIR = f"{PROJECT_ROOT}/outputs/{EXPERIMENT_NAME}"

# Evaluation parameters
IOU_THRESHOLD = 0.01  
CONFIDENCE_THRESHOLDS = np.arange(0.1, 0.9, 0.05)  # For threshold sweep

# Training configuration
N_TRIALS = 10  
EFFECTIVE_BATCH_SIZE = 16  # RF-DETR recommended

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "logs"), exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(OUTPUT_DIR, "logs", "optimization.log")),
        logging.StreamHandler()
    ]
)

# ============================================================================
# CORE EVALUATION FUNCTIONS
# ============================================================================
def box_iou(box1: np.ndarray, box2: np.ndarray, eps: float = 1e-7):
    """Calculate IoU - exact copy from eval script"""
    if box1.ndim == 1:
        box1 = box1.reshape(1, 4)
    if box2.ndim == 1:
        box2 = box2.reshape(1, 4)
    
    (a1, a2), (b1, b2) = np.split(box1, 2, 1), np.split(box2, 2, 1)
    inter = ((np.minimum(a2, b2[:, None, :]) - np.maximum(a1, b1[:, None, :])).clip(0).prod(2))
    return inter / ((a2 - a1).prod(1) + (b2 - b1).prod(1)[:, None] - inter + eps)

def cache_predictions(model, val_dataset, min_conf=0.01):
    """
    Run inference ONCE with very low threshold and cache results
    """
    print(f"    Caching predictions from {len(val_dataset)} validation images...")
    all_predictions = []
    
    start_time = time.time()
    for path, _, annotations in tqdm(val_dataset, desc="Inference", leave=False):
        with Image.open(path) as img:
            img_rgb = img.convert("RGB")
            preds = model.predict(img_rgb, threshold=min_conf)

            # --- SAFETY GUARD: must have confidences and lengths must match ---
            assert hasattr(preds, 'confidence') and preds.confidence is not None \
                and len(preds.confidence) == len(preds.xyxy), \
                f"Missing/mismatched confidences for {os.path.basename(path)}; cannot sweep thresholds safely."

            boxes = np.array(preds.xyxy, dtype=float)
            confidences = np.array(preds.confidence, dtype=float)
            
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
    Evaluate cached predictions at specific threshold (FAST!)
    """
    obj_tp = obj_fp = obj_fn = 0
    
    for pred_data in cached_predictions:
        # Filter by confidence 
        boxes_all = pred_data['boxes']
        if boxes_all.size == 0:
            filtered_boxes = np.empty((0, 4), dtype=float)
        else:
            filtered_boxes = boxes_all[pred_data['confidences'] >= confidence_threshold]

        
        # Object-level evaluation
        gt = np.array(pred_data['annotations'].xyxy)
        
        if gt.size == 0:
            obj_fp += len(filtered_boxes)
        else:
            matched = np.zeros(len(gt), dtype=bool)
            for pb in filtered_boxes:
                ious = []
                for gt_box in gt:
                    iou_val = box_iou(pb, gt_box)[0, 0]
                    ious.append(iou_val)
                
                if ious:
                    max_iou = float(np.max(ious))
                    idx = int(np.argmax(ious))
                    if max_iou > IOU_THRESHOLD and not matched[idx]:
                        obj_tp += 1
                        matched[idx] = True
                    else:
                        obj_fp += 1
            obj_fn += int((~matched).sum())
    
    # Calculate metrics
    precision = obj_tp / (obj_tp + obj_fp) if (obj_tp + obj_fp) > 0 else 0.0
    recall = obj_tp / (obj_tp + obj_fn) if (obj_tp + obj_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return f1, precision, recall, obj_tp, obj_fp, obj_fn

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

# ============================================================================
# TRAINING ANALYSIS
# ============================================================================
def parse_training_log(checkpoint_dir):
    """Extract training metrics from RF-DETR log"""
    log_file = os.path.join(checkpoint_dir, "log.txt")
    metrics = {
        'epochs': [],
        'train_loss': [],
        'val_loss': [],
        'mAP': []  # RF-DETR tracks this internally
    }
    
    if not os.path.exists(log_file):
        return metrics
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        # Parse epoch and loss
        if 'Epoch' in line:
            try:
                # RF-DETR format: "Epoch [10/50]... loss: 0.234"
                epoch_match = re.search(r'Epoch\s*\[(\d+)/\d+\]', line)
                if epoch_match:
                    epoch = int(epoch_match.group(1))
                    
                    # Extract loss
                    loss_match = re.search(r'loss:\s*([\d.]+)', line)
                    if loss_match:
                        loss = float(loss_match.group(1))
                        metrics['epochs'].append(epoch)
                        metrics['train_loss'].append(loss)
                    
                    # Extract mAP if present
                    map_match = re.search(r'mAP[@\d.]*:\s*([\d.]+)', line)
                    if map_match:
                        metrics['mAP'].append(float(map_match.group(1)))
            except Exception as e:
                pass
    
    return metrics

def plot_training_curves(trial_dir, metrics):
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
    
    # 2. mAP (if tracked internally by RF-DETR)
    if metrics['mAP']:
        axes[0, 1].plot(metrics['epochs'][:len(metrics['mAP'])], metrics['mAP'], 'g-', linewidth=2)
        axes[0, 1].set_title('Validation mAP (RF-DETR Internal)', fontsize=12, weight='bold')
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
            from scipy.ndimage import uniform_filter1d
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
    
    if metrics['mAP']:
        summary_text += f"Final mAP: {metrics['mAP'][-1]:.4f}\n"
        summary_text += f"(Note: We optimize F1, not mAP)"
    
    axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes,
                   fontsize=11, verticalalignment='top', family='monospace')
    
    plt.suptitle('RF-DETR Training Analysis', fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(trial_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()

def plot_threshold_analysis(trial_dir, threshold_results, best_result):
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
    
    plt.suptitle(f'Threshold Analysis - Best F1: {best_result["f1_score"]:.4f} @ {best_result["confidence"]:.1f}',
                fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(trial_dir, 'threshold_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()

# ============================================================================
# MAIN OBJECTIVE FUNCTION
# ============================================================================
def objective(trial):
    """
    Optuna objective with smart threshold selection and comprehensive analysis
    """
    
    # Hyperparameters for wildfire smoke detection
    hp = {
        'epochs': trial.suggest_int('epochs', 20, 40, step=10),
        'batch_size': trial.suggest_categorical('batch_size', [2, 4, 8, 16]),
        'lr': trial.suggest_float('lr', 5e-5, 5e-4, log=True),
        'lr_encoder': trial.suggest_float('lr_encoder', 5e-6, 1e-4, log=True),
        'resolution': trial.suggest_categorical('resolution', [896, 1120, 1232]),
        'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True),
        'gradient_checkpointing': trial.suggest_categorical('gradient_checkpointing', [True, False]),
        'use_ema': trial.suggest_categorical('use_ema', [True, False]),
    }
    
    # Calculate gradient accumulation
    hp['grad_accum_steps'] = max(1, EFFECTIVE_BATCH_SIZE // hp['batch_size'])
    
    # Create trial directory
    trial_dir = os.path.join(OUTPUT_DIR, f"trial_{trial.number:03d}")
    os.makedirs(trial_dir, exist_ok=True)
    os.makedirs(os.path.join(trial_dir, "checkpoints"), exist_ok=True)
    
    trial_start = time.time()
    
    print(f"\n{'='*60}")
    print(f"TRIAL {trial.number}")
    print(f"  Resolution: {hp['resolution']}px")
    print(f"  Batch: {hp['batch_size']} × {hp['grad_accum_steps']} = {EFFECTIVE_BATCH_SIZE}")
    print(f"  LR: {hp['lr']:.2e} (encoder: {hp['lr_encoder']:.2e})")
    print(f"  Weight Decay: {hp['weight_decay']:.2e}")
    print(f"  Epochs: {hp['epochs']}")
    
    # Save hyperparameters
    with open(os.path.join(trial_dir, "hyperparameters.json"), 'w') as f:
        json.dump(hp, f, indent=2)
    
    try:
        # Load validation dataset
        val_ds = sv.DetectionDataset.from_coco(
            images_directory_path=VALIDATION_DIR,
            annotations_path=VALIDATION_ANNOTATIONS
        )
        print(f"  Loaded {len(val_ds)} validation images")
        
        # Initialize and train model
        model = RFDETRBase(resolution=hp['resolution'])
        
        print(f"  Training for {hp['epochs']} epochs...")
        train_start = time.time()
        
        model.train(
            dataset_dir=DATASET_DIR,
            output_dir=os.path.join(trial_dir, "checkpoints"),
            epochs=hp['epochs'],
            batch_size=hp['batch_size'],
            grad_accum_steps=hp['grad_accum_steps'],
            lr=hp['lr'],
            lr_encoder=hp['lr_encoder'],
            resolution=hp['resolution'],
            weight_decay=hp['weight_decay'],
            gradient_checkpointing=hp['gradient_checkpointing'],
            use_ema=hp['use_ema'],
            device="cuda",
            early_stopping=True,
            early_stopping_patience=10,
            early_stopping_min_delta=0.001,
            early_stopping_use_ema=True,
            save_period=10,
            verbose=False,
            plots=False,
            tensorboard=False,
            wandb=True
        )
        
        train_time = time.time() - train_start
        print(f"  Training complete in {train_time/3600:.1f} hours")
        
        # Parse and plot training metrics
        training_metrics = parse_training_log(os.path.join(trial_dir, "checkpoints"))
        plot_training_curves(trial_dir, training_metrics)
        
        # Load best checkpoint for evaluation
        checkpoint_path = os.path.join(trial_dir, "checkpoints", "checkpoint_best_ema.pth")
        if not os.path.exists(checkpoint_path):
            checkpoint_path = os.path.join(trial_dir, "checkpoints", "checkpoint_best_total.pth")
        
        print(f"  Loading checkpoint for evaluation...")
        eval_model = RFDETRBase(pretrain_weights=checkpoint_path, num_classes=1, resolution=hp['resolution'])
        
        # SMART EVALUATION: Cache predictions once
        cached_predictions = cache_predictions(eval_model, val_ds, min_conf=0.01)
        
        # Find best confidence threshold
        print(f"  Finding optimal confidence threshold...")
        best_result, all_threshold_results = find_best_threshold(cached_predictions, CONFIDENCE_THRESHOLDS)
        
        print(f"  Best F1: {best_result['f1_score']:.4f} @ confidence={best_result['confidence']:.1f}")
        print(f"  Precision: {best_result['precision']:.4f}, Recall: {best_result['recall']:.4f}")
        
        # Plot threshold analysis
        plot_threshold_analysis(trial_dir, all_threshold_results, best_result)
        
        # Clean up
        del eval_model
        del model
        torch.cuda.empty_cache()
        gc.collect()
        
        # Calculate total trial time
        trial_time = time.time() - trial_start
        
        # Save comprehensive results
        results = {
            "trial_number": trial.number,
            "hyperparameters": hp,
            "training_time_hours": train_time / 3600,
            "total_trial_time_hours": trial_time / 3600,
            "best_result": best_result,
            "all_threshold_results": all_threshold_results,
            "training_metrics": training_metrics,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(os.path.join(trial_dir, "results.json"), 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save human-readable summary
        with open(os.path.join(trial_dir, "summary.txt"), 'w') as f:
            f.write(f"TRIAL {trial.number} SUMMARY\n")
            f.write("="*60 + "\n\n")
            
            f.write("HYPERPARAMETERS:\n")
            for k, v in hp.items():
                f.write(f"  {k}: {v}\n")
            
            f.write(f"\nTRAINING:\n")
            f.write(f"  Training time: {train_time/3600:.1f} hours\n")
            if training_metrics['train_loss']:
                f.write(f"  Initial loss: {training_metrics['train_loss'][0]:.4f}\n")
                f.write(f"  Final loss: {training_metrics['train_loss'][-1]:.4f}\n")
                f.write(f"  Min loss: {min(training_metrics['train_loss']):.4f}\n")
            
            f.write(f"\nOBJECT-LEVEL RESULTS (Optimization Target):\n")
            f.write(f"  Best F1 Score: {best_result['f1_score']:.4f} \n")
            f.write(f"  Best Confidence: {best_result['confidence']:.1f}\n")
            f.write(f"  Precision: {best_result['precision']:.4f}\n")
            f.write(f"  Recall: {best_result['recall']:.4f}\n")
            f.write(f"  TP={best_result['tp']}, FP={best_result['fp']}, FN={best_result['fn']}\n")
            
            f.write(f"\nEVALUATION:\n")
            f.write(f"  IoU threshold: {IOU_THRESHOLD}\n")
            f.write(f"  Thresholds tested: {len(all_threshold_results)}\n")
            f.write(f"  Total trial time: {trial_time/3600:.1f} hours\n")
        
        logging.info(f"Trial {trial.number}: F1={best_result['f1_score']:.4f} @ conf={best_result['confidence']:.1f}")
        
        with open(os.path.join(trial_dir, "gpu_info.json"), "w") as f:
            json.dump({
                "gpu_id": gpu_id,
                "gpu_name": gpu_name,
                "cuda_version": torch.version.cuda,
                "torch_version": torch.__version__
            }, f, indent=2)

        return best_result['f1_score']  # RETURN BEST F1 THIS MODEL CAN ACHIEVE
        
    except Exception as e:
        error_msg = f"ERROR in trial {trial.number}: {str(e)}"
        print(f"  {error_msg}")
        logging.error(error_msg)
        
        with open(os.path.join(trial_dir, "error.txt"), 'w') as f:
            f.write(f"Error: {str(e)}\n\n")
            import traceback
            f.write(traceback.format_exc())
        
        # CLEANUP 
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        
        torch.cuda.empty_cache()  
        gc.collect()              
        
        raise optuna.TrialPruned()

# ============================================================================
# OPTIMIZATION SUMMARY PLOTS
# ============================================================================
def create_optimization_summary(study, output_dir):
    """Create comprehensive optimization summary plots"""
    
    trials = [t for t in study.trials if t.value is not None]
    if len(trials) < 2:
        return
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Extract data
    trial_nums = [t.number for t in trials]
    f1_scores = [t.value for t in trials]
    resolutions = [t.params.get('resolution', 0) for t in trials]
    lrs = [t.params.get('lr', 0) for t in trials]
    batch_sizes = [t.params.get('batch_size', 0) for t in trials]
    
    # 1. F1 Scores by Trial
    ax1 = fig.add_subplot(gs[0, 0])
    colors = ['red' if t.number == study.best_trial.number else 'blue' for t in trials]
    ax1.bar(trial_nums, f1_scores, color=colors)
    ax1.axhline(y=max(f1_scores), color='r', linestyle='--', alpha=0.3)
    ax1.set_xlabel('Trial Number')
    ax1.set_ylabel('Object-Level F1 Score')
    ax1.set_title('F1 Scores by Trial', fontsize=12, weight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. F1 vs Resolution
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(resolutions, f1_scores, s=150, alpha=0.7, c=colors)
    ax2.set_xlabel('Resolution (pixels)')
    ax2.set_ylabel('Object-Level F1 Score')
    ax2.set_title('F1 Score vs Resolution', fontsize=12, weight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. F1 vs Learning Rate
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.scatter(lrs, f1_scores, s=150, alpha=0.7, c=colors)
    ax3.set_xlabel('Learning Rate')
    ax3.set_ylabel('Object-Level F1 Score')
    ax3.set_xscale('log')
    ax3.set_title('F1 Score vs Learning Rate', fontsize=12, weight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. F1 vs Batch Size
    ax4 = fig.add_subplot(gs[1, 0])
    for bs in set(batch_sizes):
        bs_f1s = [f1 for f1, b in zip(f1_scores, batch_sizes) if b == bs]
        ax4.scatter([bs]*len(bs_f1s), bs_f1s, s=100, alpha=0.7, label=f'BS={bs}')
    ax4.set_xlabel('Batch Size')
    ax4.set_ylabel('Object-Level F1 Score')
    ax4.set_title('F1 Score vs Batch Size', fontsize=12, weight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Optimization History
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(trial_nums, f1_scores, 'b-', marker='o', markersize=8, linewidth=2)
    ax5.scatter(study.best_trial.number, study.best_trial.value, 
               color='red', s=200, marker='*', zorder=5)
    ax5.set_xlabel('Trial Number')
    ax5.set_ylabel('Object-Level F1 Score')
    ax5.set_title('Optimization Progress', fontsize=12, weight='bold')
    ax5.grid(True, alpha=0.3)
    
    # 6. Parameter Importance
    ax6 = fig.add_subplot(gs[1, 2])
    if len(trials) >= 5:
        try:
            importance = optuna.importance.get_param_importances(study)
            params = list(importance.keys())
            values = list(importance.values())
            ax6.barh(params, values, color='teal')
            ax6.set_xlabel('Importance')
            ax6.set_title('Hyperparameter Importance', fontsize=12, weight='bold')
        except:
            ax6.text(0.5, 0.5, 'Not enough trials', ha='center', va='center')
    else:
        ax6.text(0.5, 0.5, f'Need ≥5 trials\n(have {len(trials)})', ha='center', va='center')
    
    # 7. Best Confidence Thresholds
    ax7 = fig.add_subplot(gs[2, 0])
    best_confs = []
    for t in trials:
        try:
            results_file = os.path.join(output_dir, f"trial_{t.number:03d}", "results.json")
            if os.path.exists(results_file):
                with open(results_file, 'r') as f:
                    data = json.load(f)
                    best_confs.append(data['best_result']['confidence'])
        except:
            pass
    
    if best_confs:
        ax7.hist(best_confs, bins=8, color='green', alpha=0.7, edgecolor='black')
        ax7.set_xlabel('Best Confidence Threshold')
        ax7.set_ylabel('Count')
        ax7.set_title('Distribution of Best Thresholds', fontsize=12, weight='bold')
        ax7.grid(True, alpha=0.3, axis='y')
    
    # 8. Trial Duration
    ax8 = fig.add_subplot(gs[2, 1])
    durations = []
    for t in trials:
        try:
            results_file = os.path.join(output_dir, f"trial_{t.number:03d}", "results.json")
            if os.path.exists(results_file):
                with open(results_file, 'r') as f:
                    data = json.load(f)
                    durations.append(data.get('total_trial_time_hours', 0))
        except:
            durations.append(0)
    
    if durations:
        ax8.bar(trial_nums, durations, color='purple', alpha=0.7)
        ax8.set_xlabel('Trial Number')
        ax8.set_ylabel('Duration (hours)')
        ax8.set_title('Trial Duration', fontsize=12, weight='bold')
        ax8.grid(True, alpha=0.3, axis='y')
    
    # 9. Best Trial Summary
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    
    summary_text = "BEST TRIAL SUMMARY\n" + "="*25 + "\n\n"
    summary_text += f"Trial Number: {study.best_trial.number}\n"
    summary_text += f"F1 Score: {study.best_trial.value:.4f}\n\n"
    summary_text += "Parameters:\n"
    for param, value in study.best_trial.params.items():
        if isinstance(value, float):
            summary_text += f"  {param}: {value:.2e}\n"
        else:
            summary_text += f"  {param}: {value}\n"
    
    ax9.text(0.1, 0.9, summary_text, transform=ax9.transAxes,
            fontsize=10, verticalalignment='top', family='monospace')
    
    fig.suptitle(f'Hyperparameter Optimization Summary - Best F1: {study.best_trial.value:.4f}',
                fontsize=16, weight='bold')
    
    plt.savefig(os.path.join(output_dir, 'optimization_summary.png'), dpi=150, bbox_inches='tight')
    plt.close()

# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    """Run hyperparameter optimization for wildfire smoke detection"""
    
    print("\n" + "="*60)
    print("RF-DETR HYPERPARAMETER OPTIMIZATION")
    print("="*60)
    print(f"📁 Experiment: {EXPERIMENT_NAME}")
    print(f"🎯 Target: OBJECT-LEVEL F1 (reduce duplicates)")
    print(f"📊 Method: Smart threshold sweep per model")
    print(f"📂 Data: Validation")
    print(f"🔍 IoU: {IOU_THRESHOLD}")
    print(f"🔢 Trials: {N_TRIALS}")
    print("="*60 + "\n")
    
    # Save experiment configuration
    config = {
        "experiment_name": EXPERIMENT_NAME,
        "optimization_metric": "object_level_f1_with_best_threshold",
        "methodology": "Smart threshold sweep - each model evaluated at its best",
        "iou_threshold": IOU_THRESHOLD,
        "confidence_thresholds_tested": CONFIDENCE_THRESHOLDS.tolist(),
        "validation_set": VALIDATION_DIR,
        "n_trials": N_TRIALS,
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "start_time": datetime.now().isoformat()
    }
    
    with open(os.path.join(OUTPUT_DIR, "experiment_config.json"), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Create Optuna study
    study = optuna.create_study(
        direction="maximize",
        study_name="rfdetr_wildfire_optimization",
        storage=f"sqlite:///{OUTPUT_DIR}/study.db",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=3)
    )
    
    # Run optimization
    try:
        study.optimize(objective, n_trials=N_TRIALS)
        
        # Get results
        best_trial = study.best_trial
        
        print("\n" + "="*60)
        print("OPTIMIZATION COMPLETE!")
        print("="*60)
        print(f"🏆 Best F1: {best_trial.value:.4f}")
        print(f"📁 Best Model: {OUTPUT_DIR}/trial_{best_trial.number:03d}/")
        
        # Load best trial's detailed results
        best_results_file = os.path.join(OUTPUT_DIR, f"trial_{best_trial.number:03d}", "results.json")
        if os.path.exists(best_results_file):
            with open(best_results_file, 'r') as f:
                best_data = json.load(f)
                print(f"🎯 Best Confidence: {best_data['best_result']['confidence']:.1f}")
                print(f"📊 Precision: {best_data['best_result']['precision']:.4f}")
                print(f"📊 Recall: {best_data['best_result']['recall']:.4f}")
        
        print("\n🔧 Best Hyperparameters:")
        for param, value in best_trial.params.items():
            if isinstance(value, float):
                print(f"   {param}: {value:.2e}")
            else:
                print(f"   {param}: {value}")
        
        # Create comprehensive summary
        create_optimization_summary(study, OUTPUT_DIR)
        
        # Save final summary
        summary = {
            "best_trial": best_trial.number,
            "best_f1": best_trial.value,
            "best_params": best_trial.params,
            "total_trials": len(study.trials),
            "successful_trials": len([t for t in study.trials if t.value is not None]),
            "failed_trials": len([t for t in study.trials if t.value is None]),
            "completion_time": datetime.now().isoformat(),
            "experiment_config": config
        }
        
        with open(os.path.join(OUTPUT_DIR, "final_summary.json"), 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n💾 All results saved to: {OUTPUT_DIR}")
        print("\n✅ Next step: Evaluate on TEST set:")
        print(f"   python rfdetr_hyperparameter_tuning_eval.py \\")
        print(f"     --checkpoint {OUTPUT_DIR}/trial_{best_trial.number:03d}/checkpoints/checkpoint_best_ema.pth \\")
        print(f"     --resolution {best_trial.params['resolution']} \\")
        print(f"     --experiment_name {EXPERIMENT_NAME}_best_TEST_eval")
        
    except KeyboardInterrupt:
        print("\n⚠️  Optimization interrupted")
        print(f"   Partial results saved to: {OUTPUT_DIR}")
        
    except Exception as e:
        print(f"\n❌ Optimization failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()