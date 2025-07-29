"""
RF-DETR Evaluation Script for Wildfire Smoke Detection
Uses PyroNear methodology for baseline comparison
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import torch
from torchvision.ops import box_iou
import supervision as sv
from rfdetr import RFDETRBase

# ============================================================================
# CONFIGURATION
# ============================================================================
EXPERIMENT_NAME = "rfdetr_smoke_detection_v1"
PROJECT_ROOT = "/vol/bitbucket/si324/rf-detr-wildfire"

# Dataset paths
DATASET_PATH = f"{PROJECT_ROOT}/data/pyro25img/images/test"
ANNOTATIONS_PATH = f"{DATASET_PATH}/_annotations.coco.json"

# Output paths
EXPERIMENT_DIR = f"{PROJECT_ROOT}/outputs/{EXPERIMENT_NAME}"
CHECKPOINTS_DIR = f"{EXPERIMENT_DIR}/checkpoints"
EVAL_RESULTS_DIR = f"{EXPERIMENT_DIR}/eval_results"
PLOTS_DIR = f"{EXPERIMENT_DIR}/plots"

# Evaluation parameters (PyroNear methodology)
CONFIDENCE_THRESHOLDS = np.arange(0.1, 0.9, 0.05)
SPATIAL_IOU_THRESHOLD = 0.1  # PyroNear baseline

# Create output directories
os.makedirs(EVAL_RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

print(f"🎯 RF-DETR Wildfire Smoke Detection Evaluation")
print(f"📁 Experiment: {EXPERIMENT_NAME}")

# ============================================================================
# LOAD YOLO BASELINE FOR COMPARISON
# ============================================================================
def load_yolo_baseline():
    """Load YOLO baseline results for comparison"""
    yolo_results_path = f"{PROJECT_ROOT}/outputs/yolo_baseline_v1/eval_results/summary_results.json"
    
    try:
        with open(yolo_results_path, 'r') as f:
            yolo_data = json.load(f)
        
        baseline = {
            'f1_score': yolo_data['best_f1_score'],
            'precision': yolo_data['best_precision'],
            'recall': yolo_data['best_recall'],
            'accuracy': yolo_data['best_accuracy']
        }
        
        print(f"✅ Loaded YOLO baseline: F1={baseline['f1_score']:.4f}")
        return baseline
        
    except FileNotFoundError:
        print(f"⚠️  YOLO baseline file not found, using fallback values")
        return {
            'f1_score': 0.688,
            'precision': 0.743,
            'recall': 0.641,
            'accuracy': 0.587
        }

# ============================================================================
# MODEL INITIALIZATION
# ============================================================================
def load_model():
    """Load RF-DETR model from checkpoint"""
    checkpoint_path = f"{CHECKPOINTS_DIR}/checkpoint_best_ema.pth"
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        exit(1)
    
    print(f"✅ Loading RF-DETR from: {checkpoint_path}")
    return RFDETRBase(pretrain_weights=checkpoint_path)

# ============================================================================
# DATASET LOADING
# ============================================================================
def load_dataset():

    """Load all test images (both annotated and non-annotated)"""
    if not os.path.exists(ANNOTATIONS_PATH):
        print(f"❌ Annotations not found: {ANNOTATIONS_PATH}")
        exit(1)
    
    # Load COCO dataset for annotated images
    coco_ds = sv.DetectionDataset.from_coco(
        images_directory_path=DATASET_PATH,
        annotations_path=ANNOTATIONS_PATH
    )
    
    # Get all test images
    all_test_images = [f for f in os.listdir(DATASET_PATH) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    coco_images = {os.path.basename(path) for path, _, _ in coco_ds}
    non_coco_images = [img for img in all_test_images if img not in coco_images]
    
    print(f"📊 Dataset composition:")
    print(f"  Annotated images (COCO): {len(coco_ds)}")
    print(f"  Non-annotated images: {len(non_coco_images)}")
    print(f"  Total test images: {len(all_test_images)}")
    
    # Create combined dataset
    combined_dataset = []
    
    # Add annotated images
    for path, image, annotations in coco_ds:
        combined_dataset.append((path, image, annotations))
    
    # Add non-annotated images with empty annotations
    for img_name in non_coco_images:
        img_path = os.path.join(DATASET_PATH, img_name)
        if os.path.exists(img_path):
            image = Image.open(img_path)
            empty_annotations = sv.Detections.empty()
            combined_dataset.append((img_path, image, empty_annotations))
    
    print(f"📊 Combined dataset: {len(combined_dataset)} images")
    return combined_dataset


# ============================================================================
# PYRONEAR-STYLE EVALUATION
# ============================================================================
def has_spatial_overlap(predictions, ground_truth):
    """
    Check if predictions have spatial overlap with ground truth
    Uses PyroNear's 0.1 IoU threshold for spatial matching
    """
    if len(predictions.xyxy) == 0 or len(ground_truth.xyxy) == 0:
        return False
    
    # Convert to torch tensors for IoU calculation
    pred_boxes = torch.tensor(predictions.xyxy, dtype=torch.float32)
    gt_boxes = torch.tensor(ground_truth.xyxy, dtype=torch.float32)
    
    # Calculate IoU matrix and check for sufficient overlap
    ious = box_iou(pred_boxes, gt_boxes)
    return (ious > SPATIAL_IOU_THRESHOLD).any().item()

def evaluate_at_confidence(model, dataset, confidence_threshold):
    """
    Evaluate model at a single confidence threshold
    Returns TP, FP, FN, TN counts 
    """
    tp = fp = fn = tn = 0
    
    for path, image, annotations in dataset:
        # Get model predictions
        predictions = model.predict(image, threshold=confidence_threshold)
        
        # Check for smoke presence
        has_smoke_gt = len(annotations.xyxy) > 0
        has_smoke_pred = len(predictions.xyxy) > 0

        if has_smoke_gt:
            # Ground truth has smoke - check for spatial overlap
            spatial_match = has_spatial_overlap(predictions, annotations) if has_smoke_pred else False
            
            if spatial_match:
                tp += 1  # True Positive: Correct detection with spatial overlap
            else:
                fn += 1  # False Negative: Missed detection or no prediction
        else:
            # Ground truth has no smoke (empty scene)
            if has_smoke_pred:
                fp += 1  # False Positive: False alarm on empty scene
            else:
                tn += 1  # True Negative: Correctly identified empty scene

    total_images = len(dataset)
    return tp, fp, fn, tn, total_images

def calculate_metrics(tp, fp, fn, tn, total_images):
    """Calculate evaluation metrics"""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / total_images if total_images > 0 else 0.0
    
    return precision, recall, f1_score, accuracy

def evaluate_model(model, dataset):
    """
    Evaluate RF-DETR across multiple confidence thresholds
    Uses PyroNear methodology for direct comparison
    """
    print(f"\n🔥 Running evaluation across {len(CONFIDENCE_THRESHOLDS)} confidence thresholds...")
    
    results = []
    
    for conf_threshold in tqdm(CONFIDENCE_THRESHOLDS, desc="Evaluating"):
        # Evaluate at this confidence threshold
        tp, fp, fn, tn, total_images = evaluate_at_confidence(model, dataset, conf_threshold)
        
        # Calculate metrics
        precision, recall, f1_score, accuracy = calculate_metrics(tp, fp, fn, tn, total_images)
        
        # Store results
        result = {
            'confidence_threshold': float(conf_threshold),
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'accuracy': accuracy,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn
        }
        results.append(result)
        
        print(f"Conf {conf_threshold:.2f}: P={precision:.3f}, R={recall:.3f}, F1={f1_score:.3f}, Acc={accuracy:.3f}")
    
    return results

# ============================================================================
# VISUALIZATION
# ============================================================================
def create_evaluation_plot(results):
    """Create evaluation metrics plot"""
    conf_vals = [r['confidence_threshold'] for r in results]
    f1_scores = [r['f1_score'] for r in results]
    precisions = [r['precision'] for r in results]
    recalls = [r['recall'] for r in results]
    accuracies = [r['accuracy'] for r in results]
    
    # Find best F1 score
    best_idx = np.argmax(f1_scores)
    best_conf = conf_vals[best_idx]
    best_f1 = f1_scores[best_idx]
    
    # Create plot
    plt.figure(figsize=(12, 8))
    plt.plot(conf_vals, f1_scores, 'b-o', label='F1 Score', linewidth=2)
    plt.plot(conf_vals, precisions, 'g--', label='Precision')
    plt.plot(conf_vals, recalls, 'r-.', label='Recall')
    plt.plot(conf_vals, accuracies, 'm:', label='Spatial Accuracy')
    
    # Highlight best point
    plt.scatter(best_conf, best_f1, color='blue', s=100, edgecolor='black', zorder=5)
    plt.text(best_conf, best_f1, f'Best F1: {best_f1:.3f}\n@conf={best_conf:.2f}', 
             fontsize=10, ha='center', va='bottom')
    
    plt.title('RF-DETR Performance vs Confidence Threshold', fontsize=14)
    plt.xlabel('Confidence Threshold')
    plt.ylabel('Metric Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plot_path = f"{PLOTS_DIR}/evaluation_metrics.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📈 Plot saved: {plot_path}")
    return plot_path

def create_confusion_matrix(model, dataset, best_confidence):
    """Create confusion matrix visualization"""
    predictions_list = []
    targets_list = []
    
    for path, image, annotations in tqdm(dataset, desc="Generating confusion matrix"):
        predictions = model.predict(image, threshold=best_confidence)
        
        # Normalize class IDs to 0 (single smoke class)
        if len(predictions.class_id) > 0:
            predictions.class_id = np.zeros(len(predictions.class_id), dtype=int)
        if len(annotations.class_id) > 0:
            annotations.class_id = np.zeros(len(annotations.class_id), dtype=int)
        
        predictions_list.append(predictions)
        targets_list.append(annotations)
    
    # Create and save confusion matrix
    conf_matrix = sv.ConfusionMatrix.from_detections(
        predictions=predictions_list,
        targets=targets_list,
        classes=["smoke"]
    )
    
    conf_matrix.plot()
    matrix_path = f"{PLOTS_DIR}/confusion_matrix.png"
    plt.savefig(matrix_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"🎯 Confusion matrix saved: {matrix_path}")
    return matrix_path

# ============================================================================
# RESULTS SAVING
# ============================================================================
def save_results(results, yolo_baseline):
    """Save comprehensive evaluation results"""
    # Find best result
    best_result = max(results, key=lambda x: x['f1_score'])
    
    # Calculate improvement over YOLO baseline
    improvement = best_result['f1_score'] - yolo_baseline['f1_score']
    improvement_percent = (improvement / yolo_baseline['f1_score']) * 100
    
    # Compile comprehensive results
    evaluation_data = {
        "experiment_info": {
            "experiment_name": EXPERIMENT_NAME,
            "spatial_iou_threshold": SPATIAL_IOU_THRESHOLD,
            "dataset_path": DATASET_PATH
        },
        "best_results": {
            "confidence_threshold": best_result['confidence_threshold'],
            "f1_score": best_result['f1_score'],
            "precision": best_result['precision'],
            "recall": best_result['recall'],
            "accuracy": best_result['accuracy']
        },
         "confusion_matrix": {  
            "true_positives": int(best_result["tp"]),
            "true_negatives": int(best_result["tn"]),
            "false_positives": int(best_result["fp"]),
            "false_negatives": int(best_result["fn"])
        },
        "baseline_comparison": {
            "yolo_baseline": yolo_baseline,
            "rfdetr_results": {
                "f1_score": best_result['f1_score'],
                "precision": best_result['precision'],
                "recall": best_result['recall'],
                "accuracy": best_result['accuracy']
            },
            "improvement": {
                "f1_absolute": improvement,
                "f1_relative_percent": improvement_percent
            }
        },
        "detailed_results": results
    }
    
    # Save JSON results
    results_path = f"{EVAL_RESULTS_DIR}/evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump(evaluation_data, f, indent=2)
    
    # Save comparison summary
    summary_path = f"{EVAL_RESULTS_DIR}/comparison_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("RF-DETR vs YOLO Baseline Comparison\n")
        f.write("=" * 40 + "\n\n")
        f.write("YOLO Baseline Results:\n")
        f.write(f"  F1 Score:  {yolo_baseline['f1_score']:.4f}\n")
        f.write(f"  Precision: {yolo_baseline['precision']:.4f}\n")
        f.write(f"  Recall:    {yolo_baseline['recall']:.4f}\n")
        f.write(f"  Accuracy:  {yolo_baseline['accuracy']:.4f}\n\n")
        f.write("RF-DETR Results:\n")
        f.write(f"  F1 Score:  {best_result['f1_score']:.4f}\n")
        f.write(f"  Precision: {best_result['precision']:.4f}\n")
        f.write(f"  Recall:    {best_result['recall']:.4f}\n")
        f.write(f"  Accuracy:  {best_result['accuracy']:.4f}\n\n")
        f.write("Improvement:\n")
        f.write(f"  F1 Score: {improvement:+.4f} ({improvement_percent:+.1f}%)\n")
    
    print(f"💾 Results saved:")
    print(f"  📄 Detailed: {results_path}")
    print(f"  📊 Summary: {summary_path}")
    
    return best_result, improvement_percent

# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    """Main evaluation pipeline"""
    print("🚀 Starting RF-DETR evaluation...")
    
    # Load components
    yolo_baseline = load_yolo_baseline()
    model = load_model()
    dataset = load_dataset()
    
    # Run evaluation
    results = evaluate_model(model, dataset)
    
    # Create visualizations
    create_evaluation_plot(results)
    
    # Find best result for confusion matrix
    best_result = max(results, key=lambda x: x['f1_score'])
    create_confusion_matrix(model, dataset, best_result['confidence_threshold'])
    
    # Save results
    best_result, improvement_percent = save_results(results, yolo_baseline)
    
    # Print final summary
    print(f"\n🏆 EVALUATION COMPLETE!")
    print(f"   RF-DETR F1 Score: {best_result['f1_score']:.4f}")
    print(f"   YOLO Baseline F1: {yolo_baseline['f1_score']:.4f}")
    print(f"   Improvement: {improvement_percent:+.1f}%")
    print(f"   Best Confidence: {best_result['confidence_threshold']:.2f}")
    print(f"🔢 Confusion Matrix - TP: {best_result['tp']}, TN: {best_result['tn']}, FP: {best_result['fp']}, FN: {best_result['fn']}")
    print(f"\n✅ All results saved to: {EVAL_RESULTS_DIR}")

if __name__ == "__main__":
    main()