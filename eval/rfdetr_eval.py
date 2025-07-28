import os  
import supervision as sv 
from tqdm import tqdm  
from supervision.metrics import MeanAveragePrecision 
from PIL import Image  
import numpy as np  
from rfdetr.detr import RFDETRBase  
import matplotlib.pyplot as plt 
import json 
from torchvision.ops import box_iou 
import torch

# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================
EXPERIMENT_NAME = "rfdetr_smoke_detection_v1"  # Change this to match your training experiment

# Dataset paths 
project_root = "/vol/bitbucket/si324/rf-detr-wildfire"
dataset_path = "/vol/bitbucket/si324/rf-detr-wildfire/data/pyro25img/images/test" 
annotations_path = f"{dataset_path}/_annotations.coco.json"

# ============================================================================
# OUTPUT DIRECTORY SETUP
# ============================================================================
outputs_root = os.path.join(project_root, "outputs")
experiment_dir = os.path.join(outputs_root, EXPERIMENT_NAME)
checkpoints_dir = os.path.join(experiment_dir, "checkpoints")
plots_dir = os.path.join(experiment_dir, "plots") 
eval_results_dir = os.path.join(experiment_dir, "eval_results")

# Create directories
os.makedirs(eval_results_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

print(f"🎯 Evaluating Experiment: {EXPERIMENT_NAME}")
print(f"📁 Using checkpoint from: {checkpoints_dir}")
print(f"📁 Saving evaluation results to: {eval_results_dir}")

# ============================================================================
# MODEL INITIALIZATION
# ============================================================================
checkpoint_path = os.path.join(checkpoints_dir, "checkpoint_best_ema.pth")

if not os.path.exists(checkpoint_path):
    print(f"❌ ERROR: Checkpoint not found at {checkpoint_path}")
    exit(1)

print(f"✅ Loading model from: {checkpoint_path}")
model = RFDETRBase(pretrain_weights=checkpoint_path)

# ============================================================================
# LOAD DATASET 
# ============================================================================
# Load COCO dataset - images WITH annotations
ds = sv.DetectionDataset.from_coco(
    images_directory_path=dataset_path,
    annotations_path=annotations_path
)

print(f"📊 Dataset loaded: {len(ds)} images with annotations")

# ============================================================================
# EVALUATION FUNCTION
# ============================================================================
def evaluate_rfdetr(model, dataset, confidence_thresholds):
    """
    Evaluate RF-DETR using PyroNear methodology:
    - Test multiple confidence thresholds
    - Use 0.1 IoU threshold for spatial matching 
    - Calculate Precision, Recall, F1, and Spatial Accuracy
    """
    
    results_per_threshold = []
    
    for conf_threshold in tqdm(confidence_thresholds, desc="Testing confidence thresholds"):
        tp, fp, fn, total_correct = 0, 0, 0, 0
        total_images = len(dataset)
        
        for i, (path, image_pil, annotations) in enumerate(dataset):
            # Get predictions from RF-DETR
            detections = model.predict(image_pil, threshold=conf_threshold)
            
            # Ground truth: does the image have smoke?
            has_smoke_gt = len(annotations.xyxy) > 0 if annotations.xyxy is not None else False
            
            # Predictions: did we detect smoke?
            has_smoke_pred = len(detections.xyxy) > 0 if detections.xyxy is not None else False
            
            # PyroNear spatial matching with 0.1 IoU threshold
            spatial_match = False
            if has_smoke_gt and has_smoke_pred:
                pred_boxes = torch.tensor(detections.xyxy, dtype=torch.float32)
                gt_boxes = torch.tensor(annotations.xyxy, dtype=torch.float32)
                
                if len(pred_boxes) > 0 and len(gt_boxes) > 0:
                    ious = box_iou(pred_boxes, gt_boxes)
                    spatial_match = (ious > 0.1).any().item()  # 0.1 threshold
            
            # Classification 
            if has_smoke_gt and has_smoke_pred and spatial_match:
                tp += 1  # Correct detection with spatial overlap
                total_correct += 1
            elif has_smoke_gt and has_smoke_pred and not spatial_match:
                fp += 1  # Detection but wrong location
            elif has_smoke_gt and not has_smoke_pred:
                fn += 1  # Missed detection
            elif not has_smoke_gt and has_smoke_pred:
                fp += 1  # False alarm
            elif not has_smoke_gt and not has_smoke_pred:
                total_correct += 1  # True negative 
        
        # Calculate evaluation metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        spatial_accuracy = total_correct / total_images if total_images > 0 else 0.0
        
        results_per_threshold.append({
            'confidence_threshold': conf_threshold,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'spatial_accuracy': spatial_accuracy,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'total_evaluated': total_images
        })
        
        print(f"Conf {conf_threshold:.2f}: P={precision:.3f}, R={recall:.3f}, F1={f1_score:.3f}, Acc={spatial_accuracy:.3f}")
    
    return results_per_threshold

# ============================================================================
# RUN PYRONEAR-STYLE EVALUATION
# ============================================================================
print(f"\n🔥 RF-DETR EVALUATION")
print("- Multiple confidence thresholds")
print("- 0.1 IoU threshold for spatial matching")
print("- Precision, Recall, F1, Spatial Accuracy metrics")

# Test same confidence range as PyroNear
confidence_thresholds = np.arange(0.1, 0.9, 0.05)
print(f"📊 Testing {len(confidence_thresholds)} confidence thresholds: {confidence_thresholds[0]:.2f} to {confidence_thresholds[-1]:.2f}")

# Run evaluation
eval_results = evaluate_rfdetr(model, ds, confidence_thresholds)

# Find best F1 score 
best_result = max(eval_results, key=lambda x: x['f1_score'])
best_f1 = best_result['f1_score']
best_conf = best_result['confidence_threshold']

print(f"\n🏆 BEST RESULTS:")
print(f"   Best F1 Score: {best_f1:.4f}")
print(f"   Best Confidence: {best_conf:.3f}")
print(f"   Precision: {best_result['precision']:.4f}")
print(f"   Recall: {best_result['recall']:.4f}")
print(f"   Spatial Accuracy: {best_result['spatial_accuracy']:.4f}")

# ============================================================================
# SAVE PLOTS 
# ============================================================================
# Plot evaluation metrics vs confidence threshold
conf_vals = [r['confidence_threshold'] for r in eval_results]
f1_scores = [r['f1_score'] for r in eval_results]
precisions = [r['precision'] for r in eval_results]
recalls = [r['recall'] for r in eval_results]
accuracies = [r['spatial_accuracy'] for r in eval_results]

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

eval_plot_path = os.path.join(plots_dir, "evaluation_metrics.png")
plt.savefig(eval_plot_path, dpi=150, bbox_inches='tight')
plt.close()

# Get predictions at best confidence threshold for confusion matrix
predictions_best = []
targets_best = []

for path, image_pil, annotations in tqdm(ds, desc="Getting predictions for visualization"):
    detections = model.predict(image_pil, threshold=best_conf)
    
    # Normalize class IDs to 0 (single class)
    if len(detections.class_id) > 0:
        detections.class_id = np.zeros(len(detections.class_id), dtype=int)
    if len(annotations.class_id) > 0:
        annotations.class_id = np.zeros(len(annotations.class_id), dtype=int)
    
    predictions_best.append(detections)
    targets_best.append(annotations)

# Save confusion matrix
conf_matrix = sv.ConfusionMatrix.from_detections(
    predictions=predictions_best,
    targets=targets_best,
    classes=["smoke"]
)
conf_matrix_path = os.path.join(plots_dir, "confusion_matrix.png")
conf_matrix.plot()
plt.savefig(conf_matrix_path, dpi=150, bbox_inches='tight')
plt.close()

print(f"🖼️ Plots saved:")
print(f"  📈 Evaluation metrics: {eval_plot_path}")
print(f"  🎯 Confusion matrix: {conf_matrix_path}")

# ============================================================================
# SAVE RESULTS 
# ============================================================================

# Load YOLO baseline results for comparison
yolo_results_path = "/vol/bitbucket/si324/rf-detr-wildfire/outputs/yolo_baseline_v1/eval_results/summary_results.json"
try:
    with open(yolo_results_path, 'r') as f:
        yolo_baseline = json.load(f)
    yolo_f1_baseline = yolo_baseline['best_f1_score']
    print(f"📊 Loaded YOLO baseline F1: {yolo_f1_baseline:.4f}")
except FileNotFoundError:
    print(f"⚠️  YOLO baseline file not found, using hardcoded value")
    yolo_f1_baseline = 0.688

results = {
    "experiment_info": {
        "experiment_name": EXPERIMENT_NAME,
        "checkpoint_used": checkpoint_path,
        "dataset_path": dataset_path
    },
    "methodology": {
        "spatial_iou_threshold": 0.1,
        "confidence_thresholds_tested": confidence_thresholds.tolist(),
    },
    "best_results": {
        "best_confidence_threshold": float(best_conf),
        "best_f1_score": float(best_f1),
        "best_precision": float(best_result['precision']),
        "best_recall": float(best_result['recall']),
        "best_spatial_accuracy": float(best_result['spatial_accuracy'])
    },
    "comparison_with_yolo_baseline": {
    "yolo_f1_baseline": float(yolo_f1_baseline),  # Use loaded value
    "rfdetr_f1": float(best_f1),
    "improvement": float(best_f1 - yolo_f1_baseline),
    "relative_improvement_percent": float((best_f1 - yolo_f1_baseline) / yolo_f1_baseline * 100)
}
}

# Save comprehensive results
results_path = os.path.join(eval_results_dir, "evaluation_results.json")
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)

# Save PyroNear comparison summary
comparison_path = os.path.join(eval_results_dir, "pyronear_comparison.txt")
with open(comparison_path, 'w') as f:
    f.write(f"RF-DETR vs YOLO Baseline (PyroNear Methodology)\n")
    f.write(f"="*50 + "\n\n")
    f.write(f"YOLO Baseline (from your evaluation):\n")
    f.write(f"  F1 Score: 0.688\n")
    f.write(f"  Precision: 0.743\n") 
    f.write(f"  Recall: 0.641\n")
    f.write(f"  Accuracy: 0.587\n\n")
    f.write(f"RF-DETR Results:\n")
    f.write(f"  F1 Score: {best_f1:.4f}\n")
    f.write(f"  Precision: {best_result['precision']:.4f}\n")
    f.write(f"  Recall: {best_result['recall']:.4f}\n")
    f.write(f"  Accuracy: {best_result['spatial_accuracy']:.4f}\n\n")
    f.write(f"IMPROVEMENT:\n")
    f.write(f"  F1 Score: {best_f1 - 0.688:+.4f} ({(best_f1 - 0.688)/0.688*100:+.1f}%)\n")

print(f"\n💾 Results saved:")
print(f"  📄 Detailed: {results_path}")
print(f"  📊 Comparison: {comparison_path}")

print(f"\n🎯 FINAL SUMMARY:")
print(f"   RF-DETR F1 Score: {best_f1:.4f}")
print(f"   YOLO Baseline F1: 0.688")
print(f"   Improvement: {best_f1 - 0.688:+.4f} ({(best_f1 - 0.688)/0.688*100:+.1f}%)")
print(f"\n✅ Evaluation complete using PyroNear methodology for direct comparison!")