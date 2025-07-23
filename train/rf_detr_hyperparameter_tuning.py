import optuna
import torch
import os
import json
import time
import shutil
from rfdetr.detr import RFDETRBase
import supervision as sv
from PIL import Image

# Config
EXPERIMENT_NAME = "hyperparameter_tuning_v1"  # Change this for different tuning experiments
DATASET_DIR = "/vol/bitbucket/si324/rf-detr-wildfire/data/pyro25img/images"
VALIDATION_DIR = "/vol/bitbucket/si324/rf-detr-wildfire/data/pyro25img/images/valid"
OUTPUT_DIR = f"/vol/bitbucket/si324/rf-detr-wildfire/outputs/{EXPERIMENT_NAME}"

# Distributed training setup 
os.environ.update({
    "RANK": "0", "WORLD_SIZE": "1", "MASTER_ADDR": "localhost", 
    "MASTER_PORT": "12357", "LOCAL_RANK": "0"
})

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
    
    # Create trial directory
    trial_dir = os.path.join(OUTPUT_DIR, f"trial_{trial.number:03d}")
    checkpoints_dir = os.path.join(trial_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    # Save trial config
    config = {
        "trial": trial.number,
        "hyperparameters": dict(trial.params),
        "effective_batch_size": batch_size * grad_accum_steps
    }
    with open(os.path.join(trial_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"\n🎯 Trial {trial.number}: lr={lr:.2e}, resolution={resolution}, batch={batch_size}")
    
    try:
        # Train model
        torch.cuda.empty_cache()
        model = RFDETRBase(resolution=resolution)  # RF-DETR auto-detects classes from dataset
        
        start_time = time.time()
        model.train(
            dataset_dir=DATASET_DIR,
            output_dir=checkpoints_dir,
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
            tensorboard=False,
            wandb=False
        )
        
        # Evaluate model
        score = evaluate_model(checkpoints_dir, resolution)
        training_time = time.time() - start_time
        
        # Save results
        results = {**config, "score": score, "training_hours": training_time/3600}
        with open(os.path.join(trial_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"✅ Trial {trial.number}: score={score:.4f}, time={training_time/3600:.1f}h")
        
        # Clean up - keep only current best
        cleanup_if_not_best(trial, score, trial_dir)
        
        # GPU cleanup before returning
        del model
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        return score
        
    except Exception as e:
        print(f"❌ Trial {trial.number} failed: {e}")
        # Remove failed trial
        if os.path.exists(trial_dir):
            shutil.rmtree(trial_dir)
        raise optuna.TrialPruned(f"Training failed: {e}")

def evaluate_model(checkpoints_dir, resolution):
    """Simple evaluation using validation accuracy."""
    
    # Find best checkpoint (prefer EMA)
    import glob
    ema_checkpoints = glob.glob(os.path.join(checkpoints_dir, "*_ema.pth"))
    regular_checkpoints = glob.glob(os.path.join(checkpoints_dir, "*.pth"))
    
    if ema_checkpoints:
        checkpoint = max(ema_checkpoints, key=os.path.getctime)
    elif regular_checkpoints:
        checkpoint = max(regular_checkpoints, key=os.path.getctime)
    else:
        return 0.0
    
    # Load model and dataset
    model = RFDETRBase(pretrain_weights=checkpoint, resolution=resolution)  
    ds = sv.DetectionDataset.from_coco(
        images_directory_path=VALIDATION_DIR,
        annotations_path=os.path.join(VALIDATION_DIR, "_annotations.coco.json")
    )
    
    # Evaluate on subset (for speed)
    correct = 0
    total = min(50, len(ds))  # Quick evaluation
    
    for i in range(total):
        try:
            path, image, annotations = ds[i]
            image = Image.open(path)
            detections = model.predict(image, threshold=0.3)
            
            has_smoke = len(annotations.xyxy) > 0
            predicted_smoke = len(detections.xyxy) > 0
            
            if has_smoke == predicted_smoke:
                correct += 1
        except:
            continue
    
    return correct / total if total > 0 else 0.0

def cleanup_if_not_best(trial, score, trial_dir):
    """Remove checkpoints if this trial is not the current best."""
    
    try:
        # Get current study
        study = optuna.load_study(
            study_name="rf_detr_tuning",
            storage=f"sqlite:///{OUTPUT_DIR}/study.db"
        )
        
        # Find completed trials with valid scores
        completed = []
        for t in study.trials:
            if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None:
                completed.append(t)
        
        if len(completed) <= 1:
            return  # Keep first trial
        
        # Get best score
        scores = [t.value for t in completed]
        best_score = max(scores)
        
        if score < best_score:
            # Not the best - remove checkpoints but keep config
            checkpoints_dir = os.path.join(trial_dir, "checkpoints")
            if os.path.exists(checkpoints_dir):
                shutil.rmtree(checkpoints_dir)
                print(f"   🗑️  Removed checkpoints (score {score:.4f} < best {best_score:.4f})")
        else:
            # New best - remove previous best's checkpoints
            for t in completed:
                if t.number != trial.number and t.value == best_score:
                    prev_dir = os.path.join(OUTPUT_DIR, f"trial_{t.number:03d}", "checkpoints")
                    if os.path.exists(prev_dir):
                        shutil.rmtree(prev_dir)
                        print(f"   🔄 New best! Removed previous best trial_{t.number:03d}")
                        break
                    
    except Exception as e:
        print(f"   ⚠️  Cleanup error: {e}")

def main():
    """Run RF-DETR hyperparameter optimization."""
    
    print("🚀 RF-DETR Hyperparameter Optimization")
    print(f"🎯 Experiment: {EXPERIMENT_NAME}")
    print("📊 12 trials with early stopping and dynamic cleanup")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Create study
    study = optuna.create_study(
        direction="maximize",
        study_name="rf_detr_tuning",
        storage=f"sqlite:///{OUTPUT_DIR}/study.db",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    # Run optimization
    study.optimize(objective, n_trials=12)
    
    # Print results
    print(f"\n🏆 OPTIMIZATION COMPLETE!")
    print(f"🥇 Best score: {study.best_value:.4f}")
    print(f"🔧 Best params:")
    for param, value in study.best_params.items():
        print(f"   {param}: {value}")
    
    # Save final results
    results = {
        "best_score": study.best_value,
        "best_params": study.best_params,
        "best_trial": study.best_trial.number,
        "total_trials": len(study.trials)
    }
    
    with open(os.path.join(OUTPUT_DIR, "final_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {OUTPUT_DIR}/final_results.json")
    print(f"📁 Best model: {OUTPUT_DIR}/trial_{study.best_trial.number:03d}/")
    
    return study.best_params

if __name__ == "__main__":
    main()