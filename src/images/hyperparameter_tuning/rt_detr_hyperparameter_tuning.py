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
VALIDATION_DIR = f"{DATASET_PATH}/images/valid"
VALIDATION_ANNOTATIONS = f"{VALIDATION_DIR}/_annotations.coco.json"

class Logger:
    """Comprehensive logging for RT-DETR hyperparameter tuning with full resumability."""
    
    def __init__(self, trial_dir):
        self.trial_dir = trial_dir
        self.checkpoints_dir = os.path.join(trial_dir, "checkpoints")
        self.plots_dir = os.path.join(trial_dir, "plots")
        self.metrics_dir = os.path.join(trial_dir, "metrics")
        self.logs_dir = os.path.join(trial_dir, "logs")
        self.backups_dir = os.path.join(trial_dir, "backups")
        
        # Create all directories
        for dir_path in [self.checkpoints_dir, self.plots_dir, self.metrics_dir, 
                        self.logs_dir, self.backups_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        self.metrics_history = {
            'epoch': [], 'train_loss': [], 'val_loss': [], 'map50': [], 'map50_95': [],
            'precision': [], 'recall': [], 'learning_rate': [], 'training_time': [], 'memory_usage': []
        }
        
    def log_epoch_metrics(self, epoch, metrics):
        """Log comprehensive metrics for each epoch."""
        timestamp = datetime.now().isoformat()
        
        # Update metrics history
        for key, value in metrics.items():
            if key in self.metrics_history:
                self.metrics_history[key].append(value)
        
        # Save epoch-specific metrics
        epoch_file = os.path.join(self.metrics_dir, f"epoch_{epoch:03d}.json")
        epoch_data = {
            'epoch': epoch,
            'timestamp': timestamp,
            'metrics': metrics,
            'gpu_memory_mb': torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0,
            'system_memory_gb': shutil.disk_usage(self.trial_dir).free / 1024**3
        }
        
        with open(epoch_file, 'w') as f:
            json.dump(epoch_data, f, indent=2)
    
    def save_trial_metadata(self, hyperparameters, start_time, status, additional_info=None):
        """Save comprehensive trial metadata for resumability."""
        metadata = {
            'trial_info': {
                'hyperparameters': hyperparameters,
                'start_time': start_time,
                'status': status,
                'last_update': datetime.now().isoformat(),
            },
            'system_info': {
                'pytorch_version': torch.__version__,
                'cuda_version': getattr(torch.version, 'cuda', None) if torch.cuda.is_available() else None,
                'gpu_name': torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU',
                'gpu_memory_gb': torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0,
            },
            'paths': {
                'checkpoints': self.checkpoints_dir,
                'plots': self.plots_dir,
                'metrics': self.metrics_dir,
                'logs': self.logs_dir,
                'backups': self.backups_dir,
            },
            'metrics_history': self.metrics_history,
        }
        
        if additional_info:
            metadata.update(additional_info)
        
        metadata_file = os.path.join(self.trial_dir, "trial_metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def backup_trial_state(self, trial_number, epoch=None):
        """Create comprehensive backup of trial state."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"trial_{trial_number:03d}_backup_{timestamp}"
        
        if epoch is not None:
            backup_name += f"_epoch_{epoch:03d}"
        
        backup_path = os.path.join(self.backups_dir, backup_name)
        os.makedirs(backup_path, exist_ok=True)
        
        # Backup all important files
        files_to_backup = [
            (os.path.join(self.trial_dir, "trial_metadata.json"), "trial_metadata.json"),
            (os.path.join(self.trial_dir, "trial_results.json"), "trial_results.json"),
        ]
        
        # Backup latest checkpoint if exists
        if os.path.exists(self.checkpoints_dir):
            checkpoint_files = [f for f in os.listdir(self.checkpoints_dir) if f.endswith('.pt')]
            if checkpoint_files:
                latest_checkpoint = max(checkpoint_files, key=lambda x: os.path.getctime(os.path.join(self.checkpoints_dir, x)))
                files_to_backup.append((
                    os.path.join(self.checkpoints_dir, latest_checkpoint),
                    f"latest_checkpoint_{latest_checkpoint}"
                ))
        
        # Backup metrics directory
        if os.path.exists(self.metrics_dir):
            shutil.copytree(self.metrics_dir, os.path.join(backup_path, "metrics"), dirs_exist_ok=True)
        
        # Copy individual files
        for src, dst in files_to_backup:
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(backup_path, dst))
        
        # Save backup manifest
        backup_manifest = {
            'backup_timestamp': timestamp,
            'trial_number': trial_number,
            'epoch': epoch,
            'files_backed_up': [dst for src, dst in files_to_backup if os.path.exists(src)],
            'backup_size_mb': sum(os.path.getsize(os.path.join(backup_path, f)) 
                                for f in os.listdir(backup_path) if os.path.isfile(os.path.join(backup_path, f))) / 1024**2
        }
        
        with open(os.path.join(backup_path, "backup_manifest.json"), 'w') as f:
            json.dump(backup_manifest, f, indent=2)
        
        return backup_path

def backup_study_database():
    """Backup the Optuna study database."""
    backups_dir = os.path.join(OUTPUT_DIR, "backups")
    os.makedirs(backups_dir, exist_ok=True)
    
    study_db_path = os.path.join(OUTPUT_DIR, "study.db")
    if os.path.exists(study_db_path):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(backups_dir, f"study_backup_{timestamp}.db")
        shutil.copy2(study_db_path, backup_path)
        
        # Keep only last 10 backups
        backups = sorted([f for f in os.listdir(backups_dir) if f.startswith("study_backup_")], reverse=True)
        for old_backup in backups[10:]:
            os.remove(os.path.join(backups_dir, old_backup))

def objective(trial):
    """
    Optuna objective function for RT-DETR hyperparameter optimization.
    Based on official RT-DETR paper and repository recommendations with full logging.
    """
    start_time = datetime.now()
    trial_start_time = start_time.isoformat()
    
    try:
        # Official RT-DETR hyperparameter ranges (from paper and repo)
        # Epochs: Official models trained for 72 epochs (6x schedule)
        epochs = trial.suggest_int("epochs", 36, 108, step=12)  # 3x to 9x schedule
        
        # Batch size: Official uses effective batch size of 16-32
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
        
        # Augmentation parameters (important for small object detection)
        mixup = trial.suggest_float("mixup", 0.0, 0.2)
        mosaic = trial.suggest_float("mosaic", 0.8, 1.0)
        dropout = trial.suggest_float("dropout", 0.0, 0.2)

        # Create trial directory and logger
        trial_dir = os.path.join(OUTPUT_DIR, f"trial_{trial.number:03d}")
        logger = Logger(trial_dir)
        
        # Save initial trial metadata
        trial_params = dict(trial.params)
        trial_params.update({
            'trial_number': trial.number,
            'estimated_training_hours': epochs * 0.5,  # Rough estimate
            'effective_batch_size': batch,
        })
        
        logger.save_trial_metadata(trial_params, trial_start_time, "starting")
        
        print(f"\n🎯 Trial {trial.number}: lr0={lr0:.2e}, imgsz={imgsz}, batch={batch}, epochs={epochs}")
        print(f"   Loss weights - cls: {cls_loss:.2f}, bbox: {bbox_loss:.2f}, giou: {giou_loss:.2f}")
        print(f"   📁 Trial directory: {trial_dir}")

        # Change to checkpoints dir for model downloads
        original_dir = os.getcwd()  # type: ignore
        os.chdir(logger.checkpoints_dir)  # type: ignore
        
        # Use RT-DETR-X for best performance (as per official benchmarks)
        model = RTDETR("rtdetr-x.pt")
        os.chdir(original_dir)  # type: ignore

        # Update status
        logger.save_trial_metadata(trial_params, trial_start_time, "training")

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

        def compute_overlap_metric(pred_boxes, gt_boxes, iou_threshold=0.3):
            """Custom metric: proportion of predictions with IoU > threshold"""
            if len(pred_boxes) == 0 or len(gt_boxes) == 0:
                return 0.0
            ious = bbox_iou(torch.tensor(pred_boxes), torch.tensor(gt_boxes), iou_type='iou')
            max_ious = ious.max(dim=1)[0]
            return (max_ious > iou_threshold).float().mean().item()

        # Fallback mAP if overlap metric fails
        score = 0.0
        metrics = {}

        try:
            if hasattr(results, 'results_dict'):
                metrics = results.results_dict
                score = metrics.get('metrics/mAP50-95(B)', 0.0)

            print(f"✅ Trial {trial.number}: mAP50-95 = {score:.4f}")

            # Custom overlap metric (optional, tweak as needed)
            pred_boxes = results.boxes.xyxy.cpu().numpy() if hasattr(results, 'boxes') else []
            gt_boxes = results.data['labels'] if hasattr(results, 'data') else []
            overlap_score = compute_overlap_metric(pred_boxes, gt_boxes)
            print(f"   🔍 IoU@0.5 overlap score = {overlap_score:.3f}")

            # Optional: replace mAP score with overlap for optimization
            score = overlap_score

        except Exception as e:
            print(f"⚠️ Metric extraction failed: {e}")
            score = 0.0

        # Save comprehensive trial results
        trial_results = {
            'trial_number': trial.number,
            'timing': {
                'start_time': trial_start_time,
                'end_time': datetime.now().isoformat(),
                'training_hours': training_time / 3600,
                'total_hours': (datetime.now() - start_time).total_seconds() / 3600,
            },
            'hyperparameters': {
                'epochs': epochs,
                'batch_size': batch,
                'learning_rate': lr0,
                'weight_decay': weight_decay,
                'image_size': imgsz,
                'optimizer': optimizer,
                'loss_weights': {
                    'cls': cls_loss,
                    'bbox': bbox_loss,
                    'giou': giou_loss,
                },
                'focal_loss': {
                    'alpha': focal_alpha,
                    'gamma': focal_gamma,
                },
                'augmentation': {
                    'mixup': mixup,
                    'mosaic': mosaic,
                    'dropout': dropout,
                },
                'schedule': {
                    'cosine_lr': cos_lr,
                    'warmup_epochs': warmup_epochs,
                    'patience': patience,
                }
            },
            'results': {
                'mAP50_95': score,
                'all_metrics': metrics,
            },
            'system_info': {
                'gpu_memory_peak_gb': torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0,
                'model_variant': 'rtdetr-x',
                'checkpoint_files': [f for f in os.listdir(logger.checkpoints_dir) if f.endswith('.pt')],
            },
            'status': 'completed'
        }
        
        with open(os.path.join(trial_dir, "trial_results.json"), "w") as f:
            json.dump(trial_results, f, indent=2)

        # Final comprehensive backup
        backup_path = logger.backup_trial_state(trial.number, epochs)
        print(f"💾 Trial {trial.number} backup saved: {backup_path}")
        
        # Update trial metadata
        logger.save_trial_metadata(trial_params, trial_start_time, "completed", {
            'final_score': score,
            'backup_path': backup_path,
        })

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
        logger = Logger(trial_dir)
        
        with open(os.path.join(trial_dir, "failure_info.json"), "w") as f:
            json.dump(failure_info, f, indent=2)
        
        # Save partial backup even on failure
        try:
            backup_path = logger.backup_trial_state(trial.number)
            print(f"💾 Failed trial {trial.number} backup saved: {backup_path}")
        except:
            pass
        
        # Cleanup on failure
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        raise optuna.TrialPruned(f"Trial failed: {str(e)}")


def main():
    """
    Run RT-DETR hyperparameter optimization following official best practices with full resumability.
    """
    print("🚀 Official RT-DETR Hyperparameter Optimization with Comprehensive Logging")
    print(f"🎯 Experiment: {EXPERIMENT_NAME}")
    print(f"📁 Output: {OUTPUT_DIR}")
    print(f"🔧 Dataset: {DATASET_PATH}")
    print("📊 Following official RT-DETR paper recommendations")
    print("💾 Full resumability and state preservation for multi-day training")
    
    # Save global experiment metadata
    experiment_metadata = {
        'experiment_name': EXPERIMENT_NAME,
        'start_time': datetime.now().isoformat(),
        'dataset_path': DATASET_PATH,
        'output_dir': OUTPUT_DIR,
        'total_planned_trials': 20,
        'expected_duration_days': 4,
        'model_variant': 'rtdetr-x',
        'optimization_objective': 'mAP50-95_maximization',
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
        direction="maximize",  # Maximize mAP50-95
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
    
    # Initial backup
    backup_study_database()
    
    try:
        # Run optimization - more trials for thorough search
        study.optimize(objective, n_trials=20, timeout=None)
        
        print(f"\n🏆 OPTIMIZATION COMPLETE!")
        print(f"🥇 Best mAP50-95: {study.best_value:.4f}")
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
        
        with open(os.path.join(OUTPUT_DIR, "comprehensive_final_results.json"), "w") as f:
            json.dump(final_results, f, indent=2)
        
        print(f"\n💾 Complete results saved to: {OUTPUT_DIR}/comprehensive_final_results.json")
        print(f"📁 Best model directory: {OUTPUT_DIR}/trial_{study.best_trial.number:03d}/")
        print(f"📊 Study database: {OUTPUT_DIR}/study.db")
        print(f"🔄 Study database backups: {OUTPUT_DIR}/backups/")
        
        # Print optimization insights
        print(f"\n📈 Optimization Insights:")
        print(f"   Best score achieved in trial {study.best_trial.number}")
        total_time = sum(t.duration.total_seconds() for t in study.trials if t.duration) / 3600
        print(f"   Total optimization time: {total_time:.1f} hours ({total_time/24:.1f} days)")
        
        # Final backup
        backup_study_database()
        
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
            'total_planned_trials': 20,
            'current_best_score': study.best_value if study.best_value else None,
            'current_best_params': study.best_params if study.best_params else None,
            'partial_optimization_hours': sum(t.duration.total_seconds() for t in study.trials if t.duration) / 3600,
            'resume_instructions': "Use the same script to resume from existing study.db - Optuna will automatically continue"
        }
        
        with open(os.path.join(OUTPUT_DIR, "interrupted_state.json"), "w") as f:
            json.dump(interrupted_results, f, indent=2)
        
        # Emergency backup
        backup_study_database()
        
        print(f"💾 Interrupted state saved. Resume by re-running the same script.")
        return None
        
    except Exception as e:
        print(f"\n❌ Optimization failed: {str(e)}")
        print(f"💾 Partial results saved to: {OUTPUT_DIR}/")
        
        # Emergency backup
        backup_study_database()
        
        return None


if __name__ == "__main__":
    main() 