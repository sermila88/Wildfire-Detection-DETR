import optuna
import torch
import torch.distributed
import os
import sys
import json
import time
import shutil
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from pathlib import Path
from rfdetr.detr import RFDETRBase
import supervision as sv
from PIL import Image

# Import the PyroNear evaluation function
from eval.pyronear_eval import evaluate_model_pyronear_style

# Set cache directory
cache_dir = Path.home() / ".ultralytics_cache"
cache_dir.mkdir(exist_ok=True)
os.environ['ULTRALYTICS_CACHE_DIR'] = str(cache_dir)

# Config
EXPERIMENT_NAME = "rfdetr_pyronear_tuning_v1"
DATASET_DIR = "/vol/bitbucket/si324/rf-detr-wildfire/data/pyro25img/images"
VALIDATION_DIR = "/vol/bitbucket/si324/rf-detr-wildfire/data/pyro25img/images/valid"
OUTPUT_DIR = f"/vol/bitbucket/si324/rf-detr-wildfire/outputs/{EXPERIMENT_NAME}"

# Distributed training setup 
os.environ.update({
    "RANK": "0", "WORLD_SIZE": "1", "MASTER_ADDR": "localhost", 
    "MASTER_PORT": "12357", "LOCAL_RANK": "0"
})

class PyroNearLogger:
    """Logger adapted for PyroNear-style evaluation metrics."""
    
    def __init__(self, trial_dir):
        self.trial_dir = trial_dir
        self.checkpoints_dir = os.path.join(trial_dir, "checkpoints")
        self.plots_dir = os.path.join(trial_dir, "plots")
        self.metrics_dir = os.path.join(trial_dir, "metrics")
        self.predictions_dir = os.path.join(trial_dir, "predictions")  # For YOLO format outputs
        self.logs_dir = os.path.join(trial_dir, "logs")
        
        # Create all directories
        for dir_path in [self.checkpoints_dir, self.plots_dir, self.metrics_dir, 
                        self.predictions_dir, self.logs_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        self.metrics_history = {
            'epoch': [], 'loss': [], 'val_loss': [], 'val_f1_score': [],
            'val_precision': [], 'val_recall': [], 'best_confidence_threshold': [],
            'learning_rate': [], 'training_time': [], 'memory_usage': []
        }
        
    def save_pyronear_plots(self):
        """Generate PyroNear-style evaluation plots."""
        if len(self.metrics_history['epoch']) < 2:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('RF-DETR Training Progress (PyroNear-style Evaluation)', fontsize=16)
        
        epochs = self.metrics_history['epoch']
        
        # Loss curves
        if self.metrics_history['loss']:
            axes[0, 0].plot(epochs, self.metrics_history['loss'], 'b-', label='Training Loss')
            if self.metrics_history['val_loss']:
                axes[0, 0].plot(epochs, self.metrics_history['val_loss'], 'r-', label='Validation Loss')
            axes[0, 0].set_title('Loss Curves')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # F1 Score (Primary metric)
        if self.metrics_history['val_f1_score']:
            axes[0, 1].plot(epochs, self.metrics_history['val_f1_score'], 'g-', label='F1 Score', linewidth=2)
            axes[0, 1].set_title('F1 Score (Primary Metric)')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('F1 Score')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Precision and Recall
        if self.metrics_history['val_precision'] and self.metrics_history['val_recall']:
            axes[0, 2].plot(epochs, self.metrics_history['val_precision'], 'purple', label='Precision')
            axes[0, 2].plot(epochs, self.metrics_history['val_recall'], 'orange', label='Recall')
            axes[0, 2].set_title('Precision and Recall')
            axes[0, 2].set_xlabel('Epoch')
            axes[0, 2].set_ylabel('Score')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
        
        # Best confidence thresholds over time
        if self.metrics_history['best_confidence_threshold']:
            axes[1, 0].plot(epochs, self.metrics_history['best_confidence_threshold'], 'brown', marker='o')
            axes[1, 0].set_title('Best Confidence Threshold')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Confidence Threshold')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Learning rate
        if self.metrics_history['learning_rate']:
            axes[1, 1].plot(epochs, self.metrics_history['learning_rate'], 'm-')
            axes[1, 1].set_title('Learning Rate Schedule')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_yscale('log')
            axes[1, 1].grid(True, alpha=0.3)
        
        # Performance summary
        if self.metrics_history['val_f1_score']:
            best_f1 = max(self.metrics_history['val_f1_score'])
            best_epoch = epochs[self.metrics_history['val_f1_score'].index(best_f1)]
            best_precision = self.metrics_history['val_precision'][self.metrics_history['val_f1_score'].index(best_f1)]
            best_recall = self.metrics_history['val_recall'][self.metrics_history['val_f1_score'].index(best_f1)]
            
            axes[1, 2].text(0.1, 0.8, f'Best F1 Score: {best_f1:.4f}', transform=axes[1, 2].transAxes, fontsize=12, weight='bold')
            axes[1, 2].text(0.1, 0.7, f'Best Epoch: {best_epoch}', transform=axes[1, 2].transAxes, fontsize=12)
            axes[1, 2].text(0.1, 0.6, f'Precision: {best_precision:.4f}', transform=axes[1, 2].transAxes, fontsize=12)
            axes[1, 2].text(0.1, 0.5, f'Recall: {best_recall:.4f}', transform=axes[1, 2].transAxes, fontsize=12)
            axes[1, 2].text(0.1, 0.4, f'Total Epochs: {len(epochs)}', transform=axes[1, 2].transAxes, fontsize=12)
            axes[1, 2].set_title('Best Performance Summary')
            axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        # Save plots
        plots_file = os.path.join(self.plots_dir, f'pyronear_training_progress_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        plt.savefig(plots_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Also save latest
        latest_plots = os.path.join(self.plots_dir, 'latest_pyronear_progress.png')
        shutil.copy2(plots_file, latest_plots)

def objective_pyronear(trial):
    """Optuna objective function adapted for PyroNear-style evaluation."""
    
    # Suggest hyperparameters (adapted for smoke detection)
    epochs = trial.suggest_int("epochs", 30, 80, step=10)  # Smoke detection may need fewer epochs
    batch_size = trial.suggest_categorical("batch_size", [4, 8, 16])  # Smaller batches for fine details
    lr = trial.suggest_float("lr", 5e-5, 5e-4, log=True)  # Lower LR for fine-tuning on smoke
    lr_encoder = trial.suggest_float("lr_encoder", 1e-5, 1e-4, log=True)
    resolution = trial.suggest_categorical("resolution", [672, 896, 1120])  # Higher res for small smoke
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    
    # Maintain effective batch size of 16
    grad_accum_steps = max(1, 16 // batch_size)
    
    # Create trial directory and logger
    trial_dir = os.path.join(OUTPUT_DIR, f"trial_{trial.number:03d}")
    logger = PyroNearLogger(trial_dir)
    
    start_time = datetime.now().isoformat()
    trial_params = dict(trial.params)
    trial_params['grad_accum_steps'] = grad_accum_steps
    trial_params['effective_batch_size'] = batch_size * grad_accum_steps
    trial_params['evaluation_method'] = 'pyronear_style'
    
    print(f"\n Trial {trial.number}: lr={lr:.2e}, resolution={resolution}, batch={batch_size}")
    print(f" Trial directory: {trial_dir}")
    print(f" Using PyroNear-style evaluation (Precision/Recall/F1)")
    
    try:
        # Train model
        torch.cuda.empty_cache()
        model = RFDETRBase(resolution=resolution)
        
        training_start = time.time()
        
        model.train(
            dataset_dir=DATASET_DIR,
            output_dir=logger.checkpoints_dir,
            epochs=epochs,
            batch_size=batch_size,
            grad_accum_steps=grad_accum_steps,
            lr=lr,
            lr_encoder=lr_encoder,
            resolution=resolution,
            weight_decay=weight_decay,
            device="cuda",
            use_ema=True,
            gradient_checkpointing=True,
            early_stopping=True,
            early_stopping_patience=10,  # More patience for smoke detection
            early_stopping_min_delta=0.005,  # Smaller delta for F1 improvements
            early_stopping_use_ema=True,
            tensorboard=True,
            wandb=False,
            save_period=10,  # Save less frequently
            plots=True,
            verbose=True
        )
        
        # Evaluate model using PyroNear-style metrics
        f1_score, eval_results = evaluate_model_pyronear_style(
            logger.checkpoints_dir, resolution, logger
        )
        
        training_time = time.time() - training_start
        
        # Update logger with final metrics
        final_metrics = {
            'final_f1_score': f1_score,
            'final_precision': eval_results['best_precision'],
            'final_recall': eval_results['best_recall'],
            'best_confidence_threshold': eval_results['best_confidence_threshold'],
            'training_hours': training_time/3600,
            'epochs_completed': epochs,
            'trial_number': trial.number
        }
        
        logger.metrics_history['val_f1_score'].append(f1_score)
        logger.metrics_history['val_precision'].append(eval_results['best_precision'])
        logger.metrics_history['val_recall'].append(eval_results['best_recall'])
        logger.metrics_history['best_confidence_threshold'].append(eval_results['best_confidence_threshold'])
        
        logger.save_pyronear_plots()
        
        # Save comprehensive results
        results = {
            **trial_params,
            "f1_score": f1_score,
            "precision": eval_results['best_precision'],
            "recall": eval_results['best_recall'],
            "best_confidence_threshold": eval_results['best_confidence_threshold'],
            "training_hours": training_time/3600,
            "start_time": start_time,
            "end_time": datetime.now().isoformat(),
            "status": "completed",
            "evaluation_method": "pyronear_style",
            "spatial_iou_threshold": 0.1
        }
        
        with open(os.path.join(trial_dir, "pyronear_final_results.json"), "w") as f:
            json.dump(results, f, indent=2)
        
        print(f" Trial {trial.number}: F1={f1_score:.4f}, P={eval_results['best_precision']:.4f}, R={eval_results['best_recall']:.4f}")
        print(f" Best confidence threshold: {eval_results['best_confidence_threshold']:.3f}")
        print(f"  Training time: {training_time/3600:.1f}h")
        print(f" All data saved to: {trial_dir}")
        
        # Cleanup
        del model
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        
        return f1_score  # Optimize for F1 score instead of mAP
        
    except Exception as e:
        print(f" Trial {trial.number} failed: {e}")
        
        # Save failure information
        failure_info = {
            **trial_params,
            "status": "failed",
            "error_message": str(e),
            "start_time": start_time,
            "failure_time": datetime.now().isoformat()
        }
        
        with open(os.path.join(trial_dir, "failure_info.json"), "w") as f:
            json.dump(failure_info, f, indent=2)
        
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
            
        raise optuna.TrialPruned(f"Training failed: {e}")

def main():
    """Run RF-DETR hyperparameter optimization with PyroNear-style evaluation."""
    
    print("🚀 RF-DETR Hyperparameter Optimization (PyroNear-style)")
    print(f"🎯 Experiment: {EXPERIMENT_NAME}")
    print("📊 Optimizing for F1 Score (Precision/Recall balance)")
    print("🔥 Wildfire smoke detection evaluation")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save experiment metadata
    experiment_metadata = {
        'experiment_name': EXPERIMENT_NAME,
        'evaluation_method': 'pyronear_style',
        'primary_metric': 'f1_score',
        'spatial_iou_threshold': 0.1,
        'start_time': datetime.now().isoformat(),
        'dataset_dir': DATASET_DIR,
        'validation_dir': VALIDATION_DIR,
        'output_dir': OUTPUT_DIR,
        'total_planned_trials': 15,  # More trials for PyroNear evaluation
        'objective': 'maximize_f1_score_for_wildfire_detection'
    }
    
    with open(os.path.join(OUTPUT_DIR, "experiment_metadata.json"), 'w') as f:
        json.dump(experiment_metadata, f, indent=2)
    
    # Create study
    study = optuna.create_study(
        direction="maximize",  # Maximize F1 score
        study_name="rf_detr_pyronear_tuning",
        storage=f"sqlite:///{OUTPUT_DIR}/study.db",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    try:
        # Run optimization
        study.optimize(objective_pyronear, n_trials=15)
        
        # Print results
        print(f"\n🏆 PYRONEAR-STYLE OPTIMIZATION COMPLETE!")
        print(f"🥇 Best F1 Score: {study.best_value:.4f}")
        print(f"🔧 Best params:")
        for param, value in study.best_params.items():
            print(f"   {param}: {value}")
        
        # Save final results
        final_results = {
            "experiment_metadata": experiment_metadata,
            "completion_time": datetime.now().isoformat(),
            "best_f1_score": study.best_value,
            "best_params": study.best_params,
            "best_trial": study.best_trial.number,
            "total_trials": len(study.trials),
            "evaluation_method": "pyronear_style",
            "optimization_target": "f1_score"
        }
        
        with open(os.path.join(OUTPUT_DIR, "pyronear_final_results.json"), "w") as f:
            json.dump(final_results, f, indent=2)
        
        print(f"\n💾 Results saved to: {OUTPUT_DIR}/pyronear_final_results.json")
        print(f"📁 Best model: {OUTPUT_DIR}/trial_{study.best_trial.number:03d}/")
        
        return study.best_params
        
    except KeyboardInterrupt:
        print(f"\n⚠️  OPTIMIZATION INTERRUPTED")
        return None

if __name__ == "__main__":
    main()