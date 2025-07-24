import optuna
import torch
import torch.distributed
import os
import json
import time
import shutil
import pickle
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless servers
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from rfdetr.detr import RFDETRBase
import supervision as sv
from PIL import Image

# Config
EXPERIMENT_NAME = "hyperparameter_tuning_v2"  # Change this for different tuning experiments
DATASET_DIR = "/vol/bitbucket/si324/rf-detr-wildfire/data/pyro25img/images"
VALIDATION_DIR = "/vol/bitbucket/si324/rf-detr-wildfire/data/pyro25img/images/valid"
OUTPUT_DIR = f"/vol/bitbucket/si324/rf-detr-wildfire/outputs/{EXPERIMENT_NAME}"

# Distributed training setup 
os.environ.update({
    "RANK": "0", "WORLD_SIZE": "1", "MASTER_ADDR": "localhost", 
    "MASTER_PORT": "12357", "LOCAL_RANK": "0"
})

class Logger:
    """Comprehensive logging for hyperparameter tuning with resumability."""
    
    def __init__(self, trial_dir):
        self.trial_dir = trial_dir
        self.checkpoints_dir = os.path.join(trial_dir, "checkpoints")
        self.plots_dir = os.path.join(trial_dir, "plots")
        self.metrics_dir = os.path.join(trial_dir, "metrics")
        self.optimizer_dir = os.path.join(trial_dir, "optimizer_states")
        self.logs_dir = os.path.join(trial_dir, "logs")
        
        # Create all directories
        for dir_path in [self.checkpoints_dir, self.plots_dir, self.metrics_dir, 
                        self.optimizer_dir, self.logs_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        self.metrics_history = {
            'epoch': [], 'loss': [], 'val_loss': [], 'val_accuracy': [],
            'learning_rate': [], 'training_time': [], 'memory_usage': []
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
            'gpu_memory_mb': torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
        }
        
        with open(epoch_file, 'w') as f:
            json.dump(epoch_data, f, indent=2)
    
    def save_checkpoint_comprehensive(self, model, optimizer, epoch, metrics, is_best=False):
        """Save comprehensive checkpoint with all necessary information."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Model checkpoint
        model_file = f"model_epoch_{epoch:03d}.pth"
        if is_best:
            model_file = f"best_model_epoch_{epoch:03d}.pth"
        
        checkpoint_path = os.path.join(self.checkpoints_dir, model_file)
        
        # Save model state
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'metrics': metrics,
            'timestamp': timestamp,
            'model_config': getattr(model, 'config', {}),
        }, checkpoint_path)
        
        # Save optimizer state separately (can be large)
        optimizer_file = os.path.join(self.optimizer_dir, f"optimizer_epoch_{epoch:03d}.pkl")
        with open(optimizer_file, 'wb') as f:
            pickle.dump({
                'epoch': epoch,
                'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
                'lr_scheduler_state': None,  # Add if using lr_scheduler
                'timestamp': timestamp
            }, f)
        
        return checkpoint_path
    
    def save_loss_curves(self):
        """Generate and save comprehensive loss curves and metrics plots."""
        if len(self.metrics_history['epoch']) < 2:
            return
        
        # Create multi-plot figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Training Progress', fontsize=16)
        
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
        
        # Validation accuracy
        if self.metrics_history['val_accuracy']:
            axes[0, 1].plot(epochs, self.metrics_history['val_accuracy'], 'g-', label='Validation Accuracy')
            axes[0, 1].set_title('Validation Accuracy')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Learning rate
        if self.metrics_history['learning_rate']:
            axes[0, 2].plot(epochs, self.metrics_history['learning_rate'], 'm-')
            axes[0, 2].set_title('Learning Rate Schedule')
            axes[0, 2].set_xlabel('Epoch')
            axes[0, 2].set_ylabel('Learning Rate')
            axes[0, 2].set_yscale('log')
            axes[0, 2].grid(True, alpha=0.3)
        
        # Training time per epoch
        if self.metrics_history['training_time']:
            axes[1, 0].plot(epochs, self.metrics_history['training_time'], 'c-')
            axes[1, 0].set_title('Training Time per Epoch')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Time (minutes)')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Memory usage
        if self.metrics_history['memory_usage']:
            axes[1, 1].plot(epochs, self.metrics_history['memory_usage'], 'orange')
            axes[1, 1].set_title('GPU Memory Usage')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Memory (MB)')
            axes[1, 1].grid(True, alpha=0.3)
        
        # Performance summary
        if self.metrics_history['val_accuracy']:
            best_acc = max(self.metrics_history['val_accuracy'])
            best_epoch = epochs[self.metrics_history['val_accuracy'].index(best_acc)]
            axes[1, 2].text(0.1, 0.8, f'Best Accuracy: {best_acc:.4f}', transform=axes[1, 2].transAxes, fontsize=12)
            axes[1, 2].text(0.1, 0.6, f'Best Epoch: {best_epoch}', transform=axes[1, 2].transAxes, fontsize=12)
            axes[1, 2].text(0.1, 0.4, f'Total Epochs: {len(epochs)}', transform=axes[1, 2].transAxes, fontsize=12)
            axes[1, 2].set_title('Training Summary')
            axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        # Save plots
        plots_file = os.path.join(self.plots_dir, f'training_progress_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        plt.savefig(plots_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Also save latest as overwrite
        latest_plots = os.path.join(self.plots_dir, 'latest_training_progress.png')
        shutil.copy2(plots_file, latest_plots)
    
    def save_trial_metadata(self, trial_params, start_time, current_status="running"):
        """Save comprehensive trial metadata."""
        metadata = {
            'trial_params': trial_params,
            'start_time': start_time,
            'current_time': datetime.now().isoformat(),
            'status': current_status,
            'total_checkpoints': len([f for f in os.listdir(self.checkpoints_dir) if f.endswith('.pth')]) if os.path.exists(self.checkpoints_dir) else 0,
            'total_optimizer_states': len([f for f in os.listdir(self.optimizer_dir) if f.endswith('.pkl')]) if os.path.exists(self.optimizer_dir) else 0,
            'metrics_files': len([f for f in os.listdir(self.metrics_dir) if f.endswith('.json')]) if os.path.exists(self.metrics_dir) else 0,
            'system_info': {
                'gpu_name': torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU',
                'gpu_memory_total': torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0,
                'python_version': os.sys.version,
                'pytorch_version': torch.__version__
            }
        }
        
        metadata_file = os.path.join(self.trial_dir, 'trial_metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

def backup_study_database():
    """Create backup of the study database."""
    backup_dir = os.path.join(OUTPUT_DIR, "backups")
    os.makedirs(backup_dir, exist_ok=True)
    
    study_db = os.path.join(OUTPUT_DIR, "study.db")
    if os.path.exists(study_db):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = os.path.join(backup_dir, f"study_backup_{timestamp}.db")
        shutil.copy2(study_db, backup_file)
        
        # Keep only last 10 backups
        backups = sorted([f for f in os.listdir(backup_dir) if f.startswith("study_backup_")])
        while len(backups) > 10:
            os.remove(os.path.join(backup_dir, backups.pop(0)))

def objective(trial):
    """Optuna objective function - one trial of hyperparameter optimization."""
    
    # Suggest hyperparameters
    epochs = trial.suggest_int("epochs", 25, 100, step=25)
    batch_size = trial.suggest_categorical("batch_size", [2, 4, 8, 16])
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    lr_encoder = trial.suggest_float("lr_encoder", 5e-6, 5e-4, log=True)
    resolution = trial.suggest_categorical("resolution", [896, 1120, 1232])
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    
    # Maintain effective batch size of 16
    grad_accum_steps = max(1, 16 // batch_size)
    
    # Create trial directory and logger
    trial_dir = os.path.join(OUTPUT_DIR, f"trial_{trial.number:03d}")
    logger = Logger(trial_dir)
    
    start_time = datetime.now().isoformat()
    trial_params = dict(trial.params)
    trial_params['grad_accum_steps'] = grad_accum_steps
    trial_params['effective_batch_size'] = batch_size * grad_accum_steps
    
    # Save initial trial metadata
    logger.save_trial_metadata(trial_params, start_time, "starting")
    
    print(f"\n🎯 Trial {trial.number}: lr={lr:.2e}, resolution={resolution}, batch={batch_size}")
    print(f"📁 Trial directory: {trial_dir}")
    
    try:
        # Train model with comprehensive logging
        torch.cuda.empty_cache()
        model = RFDETRBase(resolution=resolution)  # RF-DETR auto-detects classes from dataset
        
        training_start = time.time()
        
        # Create custom training loop with periodic saves
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
            early_stopping_patience=8,
            early_stopping_min_delta=0.001,
            early_stopping_use_ema=True,
            tensorboard=True,  # Enable for better logging
            wandb=False,
            save_period=5,  # Save every 5 epochs instead of just at end
            plots=True,  # Enable plot generation
            verbose=True
        )
        
        # Evaluate model
        score = evaluate_model(logger.checkpoints_dir, resolution, logger)
        training_time = time.time() - training_start
        
        # Final comprehensive save
        final_metrics = {
            'final_score': score,
            'training_hours': training_time/3600,
            'epochs_completed': epochs,
            'trial_number': trial.number
        }
        
        logger.log_epoch_metrics(epochs, final_metrics)
        logger.save_loss_curves()
        logger.save_trial_metadata(trial_params, start_time, "completed")
        
        # Save comprehensive results
        results = {
            **trial_params, 
            "score": score, 
            "training_hours": training_time/3600,
            "start_time": start_time,
            "end_time": datetime.now().isoformat(),
            "status": "completed",
            "checkpoints_saved": len([f for f in os.listdir(logger.checkpoints_dir) if f.endswith('.pth')]),
            "final_gpu_memory_mb": torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
        }
        
        with open(os.path.join(trial_dir, "final_results.json"), "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"✅ Trial {trial.number}: score={score:.4f}, time={training_time/3600:.1f}h")
        print(f"💾 All data saved to: {trial_dir}")
        
        # Backup study database periodically
        if trial.number % 3 == 0:  # Every 3 trials
            backup_study_database()
        
        # GPU cleanup before returning
        del model
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # CRITICAL: Cleanup distributed environment for next trial
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        
        return score
        
    except Exception as e:
        print(f"❌ Trial {trial.number} failed: {e}")
        
        # Save failure information
        failure_info = {
            **trial_params,
            "status": "failed",
            "error_message": str(e),
            "start_time": start_time,
            "failure_time": datetime.now().isoformat()
        }
        
        logger.save_trial_metadata(trial_params, start_time, "failed")
        
        with open(os.path.join(trial_dir, "failure_info.json"), "w") as f:
            json.dump(failure_info, f, indent=2)
        
        # CRITICAL: Cleanup distributed environment even on failure
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
            
        # Don't remove failed trial - keep for debugging
        raise optuna.TrialPruned(f"Training failed: {e}")

def evaluate_model(checkpoints_dir, resolution, logger):
    """Enhanced evaluation with comprehensive logging."""
    
    # Find best checkpoint (prefer EMA)
    import glob
    ema_checkpoints = glob.glob(os.path.join(checkpoints_dir, "*_ema.pth"))
    regular_checkpoints = glob.glob(os.path.join(checkpoints_dir, "*.pth"))
    
    if ema_checkpoints:
        checkpoint = max(ema_checkpoints, key=os.path.getctime)
        checkpoint_type = "EMA"
    elif regular_checkpoints:
        checkpoint = max(regular_checkpoints, key=os.path.getctime)
        checkpoint_type = "Regular"
    else:
        return 0.0
    
    print(f"📊 Evaluating {checkpoint_type} checkpoint: {os.path.basename(checkpoint)}")
    
    # Load model and dataset
    model = RFDETRBase(pretrain_weights=checkpoint, resolution=resolution)  
    ds = sv.DetectionDataset.from_coco(
        images_directory_path=VALIDATION_DIR,
        annotations_path=os.path.join(VALIDATION_DIR, "_annotations.coco.json")
    )
    
    # More comprehensive evaluation
    correct = 0
    total = min(100, len(ds))  # Slightly more thorough evaluation
    evaluation_details = []
    
    eval_start = time.time()
    
    for i in range(total):
        try:
            path, image, annotations = ds[i]
            image = Image.open(path)
            detections = model.predict(image, threshold=0.3)
            
            has_smoke = len(annotations.xyxy) > 0
            predicted_smoke = len(detections.xyxy) > 0
            
            is_correct = has_smoke == predicted_smoke
            if is_correct:
                correct += 1
            
            evaluation_details.append({
                'image_idx': i,
                'has_smoke_gt': has_smoke,
                'predicted_smoke': predicted_smoke,
                'correct': is_correct,
                'num_gt_boxes': len(annotations.xyxy),
                'num_pred_boxes': len(detections.xyxy),
                'confidence_scores': detections.confidence.tolist() if len(detections.confidence) > 0 else []
            })
            
        except Exception as e:
            evaluation_details.append({
                'image_idx': i,
                'error': str(e)
            })
            continue
    
    eval_time = time.time() - eval_start
    accuracy = correct / total if total > 0 else 0.0
    
    # Save detailed evaluation results
    eval_results = {
        'accuracy': accuracy,
        'correct_predictions': correct,
        'total_evaluated': total,
        'evaluation_time_seconds': eval_time,
        'checkpoint_used': checkpoint,
        'checkpoint_type': checkpoint_type,
        'evaluation_details': evaluation_details,
        'evaluation_timestamp': datetime.now().isoformat()
    }
    
    eval_file = os.path.join(logger.trial_dir, "evaluation_results.json")
    with open(eval_file, 'w') as f:
        json.dump(eval_results, f, indent=2)
    
    print(f"📊 Evaluation complete: {accuracy:.4f} accuracy ({correct}/{total}) in {eval_time:.1f}s")
    
    # Cleanup model and distributed environment
    del model
    torch.cuda.empty_cache()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    
    return accuracy

def main():
    """Run RF-DETR hyperparameter optimization with comprehensive logging."""
    
    print("🚀 RF-DETR Hyperparameter Optimization with Comprehensive Logging")
    print(f"🎯 Experiment: {EXPERIMENT_NAME}")
    print("📊 12 trials with complete checkpoint and metadata preservation")
    print("💾 All data will be preserved for resumability")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save global experiment metadata
    experiment_metadata = {
        'experiment_name': EXPERIMENT_NAME,
        'start_time': datetime.now().isoformat(),
        'dataset_dir': DATASET_DIR,
        'validation_dir': VALIDATION_DIR,
        'output_dir': OUTPUT_DIR,
        'total_planned_trials': 12,
        'gpu_info': {
            'device_name': torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU',
            'memory_total_gb': torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0
        },
        'system_info': {
            'pytorch_version': torch.__version__,
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None
        }
    }
    
    with open(os.path.join(OUTPUT_DIR, "experiment_metadata.json"), 'w') as f:
        json.dump(experiment_metadata, f, indent=2)
    
    # Create study
    study = optuna.create_study(
        direction="maximize",
        study_name="rf_detr_tuning",
        storage=f"sqlite:///{OUTPUT_DIR}/study.db",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    # Initial backup
    backup_study_database()
    
    try:
        # Run optimization
        study.optimize(objective, n_trials=12)
        
        # Print results
        print(f"\n🏆 OPTIMIZATION COMPLETE!")
        print(f"🥇 Best score: {study.best_value:.4f}")
        print(f"🔧 Best params:")
        for param, value in study.best_params.items():
            print(f"   {param}: {value}")
        
        # Save comprehensive final results
        final_results = {
            "experiment_metadata": experiment_metadata,
            "completion_time": datetime.now().isoformat(),
            "best_score": study.best_value,
            "best_params": study.best_params,
            "best_trial": study.best_trial.number,
            "total_trials": len(study.trials),
            "completed_trials": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            "failed_trials": len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]),
            "all_trial_results": [
                {
                    'trial_number': t.number,
                    'value': t.value,
                    'params': t.params,
                    'state': str(t.state)
                } for t in study.trials
            ]
        }
        
        with open(os.path.join(OUTPUT_DIR, "comprehensive_final_results.json"), "w") as f:
            json.dump(final_results, f, indent=2)
        
        print(f"\n💾 Complete results saved to: {OUTPUT_DIR}/comprehensive_final_results.json")
        print(f"📁 Best model and all data: {OUTPUT_DIR}/trial_{study.best_trial.number:03d}/")
        print(f"🔄 Study database backups: {OUTPUT_DIR}/backups/")
        
        # Final backup
        backup_study_database()
        
        return study.best_params
        
    except KeyboardInterrupt:
        print(f"\n⚠️  OPTIMIZATION INTERRUPTED - Saving current state...")
        
        # Save interrupted state
        interrupted_results = {
            "experiment_metadata": experiment_metadata,
            "interruption_time": datetime.now().isoformat(),
            "status": "interrupted",
            "completed_trials": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            "total_planned_trials": 12,
            "current_best_score": study.best_value if study.best_value else None,
            "current_best_params": study.best_params if study.best_params else None,
            "resume_instructions": "Use optuna dashboard or modify script to resume from existing study.db"
        }
        
        with open(os.path.join(OUTPUT_DIR, "interrupted_state.json"), "w") as f:
            json.dump(interrupted_results, f, indent=2)
        
        # Emergency backup
        backup_study_database()
        
        print(f"💾 Interrupted state saved. Resume instructions in: {OUTPUT_DIR}/interrupted_state.json")
        return None

if __name__ == "__main__":
    main()