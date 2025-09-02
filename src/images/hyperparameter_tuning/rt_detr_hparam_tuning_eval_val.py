"""
RT-DETR Hyperparameter Tuning Re-evaluation with NMS
"""

import os
import json
import glob
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
import cv2
from PIL import Image
import supervision as sv
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

# ============================================================================
# CONFIGURATION
# ============================================================================
EXPERIMENT_DIR = "/vol/bitbucket/si324/rf-detr-wildfire/src/images/outputs/RT-DETR_hyperparameter_tuning"
OUTPUT_DIR = "/vol/bitbucket/si324/rf-detr-wildfire/src/images/eval_results_hparam_tuning_NMS/RT-DETR_hparam_tuning_NMS"

# Validation dataset (matching original hyperparameter tuning)
DATASET_DIR = "/vol/bitbucket/si324/rf-detr-wildfire/src/images/data/pyro25img/images"
VALIDATION_DIR = f"{DATASET_DIR}/valid"
VALIDATION_ANNOTATIONS = f"{VALIDATION_DIR}/_annotations.coco.json"

# Evaluation parameters 
IOU_THRESHOLD = 0.1
NMS_IOU_THRESHOLD = 0.01
CONFIDENCE_THRESHOLDS = np.round(np.linspace(0.10, 0.90, 17), 2)

# Create output directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "trial_results"), exist_ok=True)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def box_iou(box1: np.ndarray, box2: np.ndarray, eps: float = 1e-7):
    """Calculate IoU"""
    if box1.ndim == 1:
        box1 = box1.reshape(1, 4)
    if box2.ndim == 1:
        box2 = box2.reshape(1, 4)
    
    (a1, a2), (b1, b2) = np.split(box1, 2, 1), np.split(box2, 2, 1)
    inter = ((np.minimum(a2, b2[:, None, :]) - np.maximum(a1, b1[:, None, :])).clip(0).prod(2))
    return inter / ((a2 - a1).prod(1) + (b2 - b1).prod(1)[:, None] - inter + eps)

def apply_nms_to_boxes(boxes, scores, iou_threshold=0.01):
    """Apply Non-Maximum Suppression"""
    if len(boxes) == 0:
        return np.array([])
    
    boxes_list = boxes.tolist()
    scores_list = scores.tolist()
    
    indices = cv2.dnn.NMSBoxes(
        bboxes=boxes_list,
        scores=scores_list,
        score_threshold=0.01,
        nms_threshold=iou_threshold
    )
    
    if len(indices) > 0:
        return indices.flatten()
    return np.array([])

# ============================================================================
# EVALUATION FUNCTIONS WITH NMS
# ============================================================================

def cache_predictions_with_nms(model, val_dataset, min_conf=0.01):
    """Run inference with NMS applied and cache results"""
    print(f"    Caching predictions with NMS from {len(val_dataset)} validation images...")
    all_predictions = []
    
    start_time = datetime.now()
    with torch.inference_mode():
        for path, _, annotations in tqdm(val_dataset, desc="Inference", leave=False):
            with Image.open(path) as img:
                img_rgb = img.convert("RGB")
                results = model.predict(source=img_rgb, conf=min_conf, verbose=False)
                preds = sv.Detections.from_ultralytics(results[0])

                if hasattr(preds, 'xyxy') and len(preds.xyxy) > 0:
                    boxes = np.array(preds.xyxy, dtype=float)
                    confidences = np.array(preds.confidence, dtype=float) if hasattr(preds, 'confidence') else np.ones(len(boxes))
                    
                    # Apply NMS
                    nms_indices = apply_nms_to_boxes(boxes, confidences, NMS_IOU_THRESHOLD)
                    
                    if len(nms_indices) > 0:
                        boxes = boxes[nms_indices]
                        confidences = confidences[nms_indices]
                    else:
                        boxes = np.empty((0, 4), dtype=float)
                        confidences = np.empty(0, dtype=float)
                else:
                    boxes = np.empty((0, 4), dtype=float)
                    confidences = np.empty(0, dtype=float)
                
            all_predictions.append({
                'path': path,
                'annotations': annotations,
                'boxes': boxes,
                'confidences': confidences
            })
    
    inference_time = (datetime.now() - start_time).total_seconds()
    print(f"    Inference complete in {inference_time:.1f} seconds")
    
    return all_predictions

def evaluate_at_threshold(cached_predictions, confidence_threshold):
    """Evaluate cached predictions at specific threshold"""
    tp = fp = fn = 0
    
    for pred_data in cached_predictions:
        boxes_all = pred_data['boxes']
        if boxes_all.size == 0:
            filtered_boxes = np.empty((0, 4), dtype=float)
        else:
            filtered_boxes = boxes_all[pred_data['confidences'] >= confidence_threshold]

        gt_boxes = np.array(pred_data['annotations'].xyxy)
        gt_matches = np.zeros(len(gt_boxes), dtype=bool) if gt_boxes.size > 0 else np.array([])

        for pred_box in filtered_boxes:
            if gt_boxes.size > 0:
                iou_values = [box_iou(pred_box, gt_box)[0, 0] for gt_box in gt_boxes]
                max_iou = max(iou_values)
                best_match_idx = np.argmax(iou_values)
                
                if max_iou > IOU_THRESHOLD and not gt_matches[best_match_idx]:
                    tp += 1
                    gt_matches[best_match_idx] = True
                else:
                    fp += 1
            else:
                fp += 1
        
        if gt_boxes.size > 0:
            fn += len(gt_boxes) - np.sum(gt_matches)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'confidence': confidence_threshold,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn)
    }

def find_best_threshold(cached_predictions, confidence_thresholds):
    """Find best confidence threshold"""
    results = []
    
    for conf in confidence_thresholds:
        result = evaluate_at_threshold(cached_predictions, conf)
        results.append(result)
    
    best_idx = np.argmax([r['f1_score'] for r in results])
    return results[best_idx], results

def reevaluate_trial(trial_num, trial_dir, val_dataset):
    """Re-evaluate a single trial with NMS"""
    print(f"\n{'='*50}")
    print(f"Re-evaluating Trial {trial_num:03d}")
    print(f"{'='*50}")
    
    hp_file = os.path.join(trial_dir, "hyperparameters.json")
    if not os.path.exists(hp_file):
        print(f"  No hyperparameters file found")
        return None
    
    with open(hp_file, 'r') as f:
        hyperparams = json.load(f)
    
    checkpoint_path = os.path.join(trial_dir, "checkpoints", "weights", "best.pt")
    
    if not os.path.exists(checkpoint_path):
        print(f"  No checkpoint found")
        return None
    
    print(f"  Loading: {os.path.basename(checkpoint_path)}")
    print(f"  Image size: {hyperparams.get('imgsz', 640)}")
    print(f"  Batch size: {hyperparams.get('batch_size', 8)}")
    print(f"  LR: {hyperparams.get('lr0', 1e-4):.2e}")
    
    try:
        model = RTDETR(checkpoint_path)
        
        cached_predictions = cache_predictions_with_nms(model, val_dataset, min_conf=0.01)
        
        print(f"  📏 Evaluating at {len(CONFIDENCE_THRESHOLDS)} thresholds...")
        best_result, all_results = find_best_threshold(cached_predictions, CONFIDENCE_THRESHOLDS)
        
        print(f"  ✅ Best F1: {best_result['f1_score']:.4f} @ conf={best_result['confidence']:.2f}")
        print(f"     Precision: {best_result['precision']:.4f}")
        print(f"     Recall: {best_result['recall']:.4f}")
        
        del model
        torch.cuda.empty_cache()
        
        return {
            'trial_number': trial_num,
            'hyperparameters': hyperparams,
            'best_result': best_result,
            'all_threshold_results': all_results,
            'checkpoint_path': checkpoint_path
        }
        
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return None

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("\n" + "="*60)
    print("RT-DETR HYPERPARAMETER RE-EVALUATION WITH NMS")
    print("="*60)
    print(f"📂 Experiment dir: {EXPERIMENT_DIR}")
    print(f"📁 Output dir: {OUTPUT_DIR}")
    print(f"🎯 IoU threshold: {IOU_THRESHOLD}")
    print(f"🔧 NMS IoU: {NMS_IOU_THRESHOLD}")
    print("="*60 + "\n")
    
    # Load validation dataset
    print("Loading validation dataset...")
    val_ds = sv.DetectionDataset.from_coco(
        images_directory_path=VALIDATION_DIR,
        annotations_path=VALIDATION_ANNOTATIONS
    )
    print(f"Loaded {len(val_ds)} validation images\n")
    
    # Find all trial directories
    trial_dirs = sorted(glob.glob(os.path.join(EXPERIMENT_DIR, "trial_*")))
    print(f"Found {len(trial_dirs)} trials to re-evaluate\n")
    
    # Re-evaluate all trials
    all_results = []
    for trial_dir in trial_dirs:
        try:
            trial_num = int(os.path.basename(trial_dir).split('_')[1])

            # Check if trial already evaluated
            trial_output_file = os.path.join(OUTPUT_DIR, "trial_results", f"trial_{trial_num:03d}", "results.json")
            if os.path.exists(trial_output_file):
                print(f"  ✓ Trial {trial_num:03d} already evaluated, loading cached results...")
                with open(trial_output_file, 'r') as f:
                    result = json.load(f)
                all_results.append(result)
                continue
                
            result = reevaluate_trial(trial_num, trial_dir, val_ds)
            
            if result:
                all_results.append(result)
                
                # Save individual trial result
                trial_output_dir = os.path.join(OUTPUT_DIR, "trial_results", f"trial_{trial_num:03d}")
                os.makedirs(trial_output_dir, exist_ok=True)
                
                with open(os.path.join(trial_output_dir, "results.json"), 'w') as f:
                    json.dump(result, f, indent=2, cls=NumpyEncoder)
                
                # Save summary
                with open(os.path.join(trial_output_dir, "summary.txt"), 'w') as f:
                    f.write(f"TRIAL {trial_num} RE-EVALUATION SUMMARY (WITH NMS)\n")
                    f.write("="*60 + "\n\n")
                    f.write("HYPERPARAMETERS:\n")
                    for k, v in result['hyperparameters'].items():
                        f.write(f"  {k}: {v}\n")
                    f.write(f"\nBEST RESULTS:\n")
                    f.write(f"  F1 Score: {result['best_result']['f1_score']:.4f}\n")
                    f.write(f"  Confidence: {result['best_result']['confidence']:.2f}\n")
                    f.write(f"  Precision: {result['best_result']['precision']:.4f}\n")
                    f.write(f"  Recall: {result['best_result']['recall']:.4f}\n")
                    f.write(f"  TP={result['best_result']['tp']}, ")
                    f.write(f"FP={result['best_result']['fp']}, ")
                    f.write(f"FN={result['best_result']['fn']}\n")
                    
        except Exception as e:
            print(f"Error processing {trial_dir}: {e}")
            continue
    
    # Save combined results
    combined_file = os.path.join(OUTPUT_DIR, "all_trials_results.json")
    with open(combined_file, 'w') as f:
        json.dump(all_results, f, indent=2, cls=NumpyEncoder)
    
    print(f"\n{'='*60}")
    print(f"Re-evaluation complete!")
    print(f"   Successful: {len(all_results)}/{len(trial_dirs)}")
    print(f"{'='*60}\n")
    
    if all_results:

        # Generate final report
        best_trial = max(all_results, key=lambda x: x['best_result']['f1_score'])
        
        with open(os.path.join(OUTPUT_DIR, "final_summary.json"), 'w') as f:
            summary = {
                "best_trial": best_trial['trial_number'],
                "best_f1": best_trial['best_result']['f1_score'],
                "best_params": best_trial['hyperparameters'],
                "best_confidence": best_trial['best_result']['confidence'],
                "total_trials": len(all_results),
                "timestamp": datetime.now().isoformat()
            }
            json.dump(summary, f, indent=2, cls=NumpyEncoder)
        
        with open(os.path.join(OUTPUT_DIR, "final_summary.txt"), 'w') as f:
            f.write("RT-DETR HYPERPARAMETER TUNING FINAL RESULTS (WITH NMS)\n")
            f.write("="*60 + "\n\n")
            f.write(f"BEST CONFIGURATION: Trial {best_trial['trial_number']:03d}\n")
            f.write(f"  F1 Score: {best_trial['best_result']['f1_score']:.4f}\n")
            f.write(f"  Precision: {best_trial['best_result']['precision']:.4f}\n")
            f.write(f"  Recall: {best_trial['best_result']['recall']:.4f}\n")
            f.write(f"  Optimal Threshold: {best_trial['best_result']['confidence']:.2f}\n\n")
            f.write("HYPERPARAMETERS:\n")
            for param, value in best_trial['hyperparameters'].items():
                if isinstance(value, float) and value < 0.001:
                    f.write(f"  {param}: {value:.2e}\n")
                else:
                    f.write(f"  {param}: {value}\n")
            f.write(f"\nCheckpoint: {best_trial['checkpoint_path']}\n")
            
            # Top 5 trials
            sorted_results = sorted(all_results, key=lambda x: x['best_result']['f1_score'], reverse=True)
            f.write("\nTOP 5 TRIALS:\n")
            for i, trial in enumerate(sorted_results[:5], 1):
                f.write(f"{i}. Trial {trial['trial_number']:03d}: ")
                f.write(f"F1={trial['best_result']['f1_score']:.4f}, ")
                f.write(f"τ={trial['best_result']['confidence']:.2f}\n")
        
        print(f"\n🏆 BEST CONFIGURATION:")
        print(f"   Trial: {best_trial['trial_number']:03d}")
        print(f"   F1 Score: {best_trial['best_result']['f1_score']:.4f}")
        print(f"   Checkpoint: {best_trial['checkpoint_path']}")
        print(f"\n All outputs saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()