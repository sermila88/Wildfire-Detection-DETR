# Import libraries 
import os  
import supervision as sv 
from tqdm import tqdm  
from supervision.metrics import MeanAveragePrecision 
from PIL import Image  
import numpy as np  
from ultralytics import RTDETR
import matplotlib.pyplot as plt 
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score  
import json 
from torchvision.ops import box_iou 
import torch

# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================
# MUST MATCH the experiment name used in training!
EXPERIMENT_NAME = "rtdetr_smoke_detection_v1"  # Change this to match your training experiment

# Dataset paths
project_root = "/vol/bitbucket/si324/rf-detr-wildfire"
dataset_path = "/vol/bitbucket/si324/rf-detr-wildfire/data/pyro25img/images/test" # Test images directory
annotations_path = f"{dataset_path}/_annotations.coco.json" # COCO format annotations

# ============================================================================
# OUTPUT DIRECTORY SETUP
# ============================================================================
# Use same organized structure as training
outputs_root = os.path.join(project_root, "outputs")
experiment_dir = os.path.join(outputs_root, EXPERIMENT_NAME)

# Define subdirectories
checkpoints_dir = os.path.join(experiment_dir, "checkpoints")
plots_dir = os.path.join(experiment_dir, "plots") 
logs_dir = os.path.join(experiment_dir, "logs")
eval_results_dir = os.path.join(experiment_dir, "eval_results")

# Create evaluation results directory if it doesn't exist
os.makedirs(eval_results_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

print(f"🎯 Evaluating Experiment: {EXPERIMENT_NAME}")
print(f"📁 Using checkpoint from: {checkpoints_dir}")
print(f"📁 Saving evaluation results to: {eval_results_dir}")
print(f"📁 Saving plots to: {plots_dir}")

# ============================================================================
# MODEL INITIALIZATION
# ============================================================================
# Load model from the experiment's checkpoint directory
checkpoint_path = os.path.join(checkpoints_dir, "weights", "best.pt")

# Check if checkpoint exists
if not os.path.exists(checkpoint_path):
    print(f"❌ ERROR: Checkpoint not found at {checkpoint_path}")
    print(f"   Make sure you've trained the model with experiment name: {EXPERIMENT_NAME}")
    print(f"   Or update EXPERIMENT_NAME to match your trained model.")
    exit(1)

print(f"✅ Loading RT-DETR model from: {checkpoint_path}")
model = RTDETR(checkpoint_path)

# ============================================================================
# DATASET LOADING
# ============================================================================
# Load COCO annotations to identify which images have smoke annotations
with open(annotations_path, 'r') as f:
    coco_data = json.load(f)

# Get all image files in the test directory (smoke + no-smoke)
all_image_files = [f for f in os.listdir(dataset_path) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

# Extract filenames of images that have annotations (smoke images)
annotated_images = {img['file_name'] for img in coco_data['images']}

# Separate images into smoke and no-smoke categories
smoke_images = [f for f in all_image_files if f in annotated_images] # Images with smoke annotations
no_smoke_images = [f for f in all_image_files if f not in annotated_images] # Images without smoke annotations

# Print dataset composition for transparency
print(f"\n📊 Dataset Composition:")
print(f"  Total test images: {len(all_image_files)}")
print(f"  Smoke images (with annotations): {len(smoke_images)}")
print(f"  No-smoke images (without annotations): {len(no_smoke_images)}")

# Load COCO dataset for smoke images (images with annotations)
ds_smoke = sv.DetectionDataset.from_coco(
    images_directory_path=dataset_path,
    annotations_path=annotations_path
)

# Initialize lists to store predictions and ground truth
predictions = []  # Model predictions for all images
targets = []      # Ground truth annotations for all images
image_labels = [] # Track whether each image contains smoke

# Evaluation parameters 
DETECTION_THRESHOLD = 0.2  # Low threshold since smoke boundaries are subjective
SPATIAL_IOU_THRESHOLD = 0.4  # IoU threshold for spatial correctness 

def ultralytics_to_supervision(results, confidence_threshold=0.2):
    """
    Convert Ultralytics YOLO results to supervision Detections format
    """
    if len(results) == 0 or results[0].boxes is None:
        return sv.Detections.empty()
    
    boxes = results[0].boxes
    if len(boxes) == 0:
        return sv.Detections.empty()
    
    # Filter by confidence
    confidences = boxes.conf.cpu().numpy()
    valid_indices = confidences >= confidence_threshold
    
    if not np.any(valid_indices):
        return sv.Detections.empty()
    
    # Extract valid detections
    xyxy = boxes.xyxy[valid_indices].cpu().numpy()
    confidence = confidences[valid_indices]
    class_id = boxes.cls[valid_indices].cpu().numpy().astype(int)
    
    return sv.Detections(
        xyxy=xyxy,
        confidence=confidence,
        class_id=class_id
    )

print(f"\n🔍 Evaluating smoke images (threshold={DETECTION_THRESHOLD}, spatial_iou={SPATIAL_IOU_THRESHOLD})...")
# Process all smoke images (those with annotations)
for path, image, annotations in tqdm(ds_smoke):
    # Run model inference using Ultralytics
    results = model.predict(path, conf=DETECTION_THRESHOLD, verbose=False)
    
    # Convert to supervision format
    detections = ultralytics_to_supervision(results, DETECTION_THRESHOLD)
    
    predictions.append(detections)  # Store model predictions
    targets.append(annotations)     # Store ground truth annotations
    image_labels.append("smoke")    # Mark as smoke image

print(f"🔍 Evaluating no-smoke images...")
# Process all no-smoke images (those without annotations)
for img_file in tqdm(no_smoke_images):
    img_path = os.path.join(dataset_path, img_file)  # Build full image path
    
    # Run model inference using Ultralytics
    results = model.predict(img_path, conf=DETECTION_THRESHOLD, verbose=False)
    
    # Convert to supervision format
    detections = ultralytics_to_supervision(results, DETECTION_THRESHOLD)
    
    # Create empty annotations for no-smoke images (no ground truth boxes)
    empty_annotations = sv.Detections.empty()
    
    predictions.append(detections)      # Store model predictions
    targets.append(empty_annotations)   # Store empty ground truth
    image_labels.append("no_smoke")     # Mark as no-smoke image

# Normalize all class IDs to 0 (single class: smoke detection)
for det in predictions:
    if len(det.class_id) > 0:
        det.class_id = np.zeros(len(det.class_id), dtype=int)  # Set all detections to class 0

for ann in targets:
    if len(ann.class_id) > 0:
        ann.class_id = np.zeros(len(ann.class_id), dtype=int)  # Set all annotations to class 0

# ============================================================================
# REFERENCE METRICS (mAP)
# ============================================================================
print("\n📊 Computing mAP (Mean Average Precision) for reference...")
map_metric = MeanAveragePrecision()
map_result = map_metric.update(predictions, targets).compute()
print("mAP Results (object detection metrics for reference):")
print(f"  mAP@0.50:0.95: {map_result.map50_95:.4f}")  # mAP across IoU thresholds 0.5-0.95
print(f"  mAP@0.50: {map_result.map50:.4f}")          # mAP at IoU threshold 0.5
print(f"  mAP@0.75: {map_result.map75:.4f}")          # mAP at IoU threshold 0.75

# Save mAP plot for reference
map_plot_path = os.path.join(plots_dir, "map_plot.png")
map_result.plot()  # Generate mAP plot
plt.savefig(map_plot_path, dpi=150, bbox_inches='tight')  # Save plot to file
plt.close()  # Close plot to free memory
print(f"🖼️ mAP plot saved to: {map_plot_path}")

# Generate and save confusion matrix to plots directory
conf_matrix = sv.ConfusionMatrix.from_detections(
    predictions=predictions,
    targets=targets,
    classes=["smoke"]
)
conf_matrix_path = os.path.join(plots_dir, "conf_matrix.png")
conf_matrix.plot()  # Generate confusion matrix plot
plt.savefig(conf_matrix_path, dpi=150, bbox_inches='tight')  # Save plot to file
plt.close()  # Close plot to free memory
print(f"🖼️ Confusion matrix saved to: {conf_matrix_path}")

# ============================================================================
# SPATIAL OVERLAP FUNCTION
# ============================================================================
def has_spatial_overlap(pred_detections, gt_detections, iou_threshold=SPATIAL_IOU_THRESHOLD):
    """
    Check if any prediction overlaps with ground truth with IoU >= threshold
    This ensures we only count detections that are spatially correct (not boxing trees as smoke)
    """
    if len(pred_detections.xyxy) == 0 or len(gt_detections.xyxy) == 0:
        return False  # No overlap if either is empty
    
    # Convert to torch tensors for IoU calculation
    pred_boxes = torch.tensor(pred_detections.xyxy, dtype=torch.float32)
    gt_boxes = torch.tensor(gt_detections.xyxy, dtype=torch.float32)
    
    # Calculate IoU matrix between all prediction and ground truth boxes
    iou_matrix = box_iou(pred_boxes, gt_boxes)
    
    # Return True if any prediction has sufficient overlap with any GT box
    return torch.max(iou_matrix) >= iou_threshold

# ============================================================================
#  EVALUATION: SPATIALLY-AWARE BINARY CLASSIFICATION
# ============================================================================
print(f"\n🔥 PRIMARY EVALUATION: Spatially-Aware Binary Image Classification")
print(f"   (Pyronear methodology + spatial validation)")

# Initialize binary classification labels
y_true_binary = []  # Ground truth: 1=smoke image, 0=no-smoke image  
y_pred_binary = []  # Prediction: 1=correct smoke detection, 0=no correct detection

# Convert object detection results to spatially-aware binary classification
for i, (pred, target, label) in enumerate(zip(predictions, targets, image_labels)):
    # Ground truth label: 1 if image contains smoke, 0 if no smoke
    y_true_binary.append(1 if label == "smoke" else 0)
    
    if label == "smoke":
        # For smoke images: check if detection spatially overlaps with ground truth
        # Only count as positive if model detects smoke in approximately correct location
        has_correct_detection = has_spatial_overlap(pred, target, SPATIAL_IOU_THRESHOLD)
        y_pred_binary.append(1 if has_correct_detection else 0)
    else:
        # For no-smoke images: any detection is incorrect (false positive)
        # Even if model detects something, it should be 0 since there's no smoke
        y_pred_binary.append(1 if len(pred.xyxy) > 0 else 0)

# Calculate primary classification metrics (as used in Pyronear paper)
binary_precision = precision_score(y_true_binary, y_pred_binary, zero_division="warn")
binary_recall = recall_score(y_true_binary, y_pred_binary, zero_division="warn")
binary_f1 = f1_score(y_true_binary, y_pred_binary, zero_division="warn")
binary_accuracy = accuracy_score(y_true_binary, y_pred_binary)

# Display primary metrics (main results following Pyronear methodology)
print(f"\n✅ PRIMARY METRICS (Spatially-Aware Image-level Classification):")
print(f"   Precision: {binary_precision:.4f} - Detected smoke images that actually contain smoke in correct location")
print(f"   Recall:    {binary_recall:.4f} - Smoke images correctly detected with proper spatial overlap") 
print(f"   F1 Score:  {binary_f1:.4f} - Harmonic mean of spatially-aware precision and recall")
print(f"   Accuracy:  {binary_accuracy:.4f} - Overall correct spatially-aware classifications")

# Calculate and display confusion matrix components for detailed analysis
TP = sum(1 for gt, pred in zip(y_true_binary, y_pred_binary) if gt == 1 and pred == 1)  # True Positives
TN = sum(1 for gt, pred in zip(y_true_binary, y_pred_binary) if gt == 0 and pred == 0)  # True Negatives  
FP = sum(1 for gt, pred in zip(y_true_binary, y_pred_binary) if gt == 0 and pred == 1)  # False Positives
FN = sum(1 for gt, pred in zip(y_true_binary, y_pred_binary) if gt == 1 and pred == 0)  # False Negatives

print(f"\n📈 DETAILED BREAKDOWN:")
print(f"   True Positives (TP):  {TP} - Smoke images with spatially correct detections")
print(f"   True Negatives (TN):  {TN} - No-smoke images correctly identified as no-smoke") 
print(f"   False Positives (FP): {FP} - No-smoke images with false smoke detections")
print(f"   False Negatives (FN): {FN} - Smoke images missed or with spatially incorrect detections")

# ============================================================================
# SAVE RESULTS
# ============================================================================
# Compile all results into a structured dictionary for saving
results = {
    "experiment_info": {
        "experiment_name": EXPERIMENT_NAME,
        "checkpoint_used": checkpoint_path,
        "dataset_path": dataset_path,
        "model_type": "RT-DETR"
    },
    "evaluation_settings": {
        "detection_threshold": DETECTION_THRESHOLD,
        "spatial_iou_threshold": SPATIAL_IOU_THRESHOLD,
        "evaluation_approach": "spatially_aware_binary_classification"
    },
    "dataset_composition": {
        "total_images": len(all_image_files),
        "smoke_images": len(smoke_images), 
        "no_smoke_images": len(no_smoke_images)
    },
    "primary_metrics": {  
        "precision": float(binary_precision),
        "recall": float(binary_recall),
        "f1_score": float(binary_f1),
        "accuracy": float(binary_accuracy)
    },
    "confusion_matrix": {  
        "true_positives": int(TP),
        "true_negatives": int(TN), 
        "false_positives": int(FP),
        "false_negatives": int(FN)
    },
    "reference_metrics": { 
        "mAP_50_95": float(map_result.map50_95),
        "mAP_50": float(map_result.map50),
        "mAP_75": float(map_result.map75)
    }
}

# Output directory for evaluation results
EVAL_RESULTS_DIR = os.path.join(project_root, "outputs", EXPERIMENT_NAME, "eval_results")
os.makedirs(EVAL_RESULTS_DIR, exist_ok=True)

# Save results to JSON file in eval_results directory
results_path = os.path.join(EVAL_RESULTS_DIR, "evaluation_results.json")
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)

# Save a summary report 
summary_path = os.path.join(EVAL_RESULTS_DIR, "evaluation_summary.txt")
with open(summary_path, 'w') as f:
    f.write(f"RT-DETR Experiment: {EXPERIMENT_NAME}\n")
    f.write(f"="*50 + "\n\n")
    f.write(f"PRIMARY METRICS:\n")
    f.write(f"  Precision: {binary_precision:.4f}\n")
    f.write(f"  Recall:    {binary_recall:.4f}\n") 
    f.write(f"  F1 Score:  {binary_f1:.4f}\n")
    f.write(f"  Accuracy:  {binary_accuracy:.4f}\n\n")
    f.write(f"CONFUSION MATRIX:\n")
    f.write(f"  TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}\n\n")
    f.write(f"REFERENCE METRICS:\n")
    f.write(f"  mAP@0.50:0.95: {map_result.map50_95:.4f}\n")
    f.write(f"  mAP@0.50: {map_result.map50:.4f}\n")

print(f"\n💾 Complete results saved to:")
print(f"  📄 JSON: {results_path}")
print(f"  📄 Summary: {summary_path}")
print(f"  🖼️ Plots: {plots_dir}")
print(f"\n🎯 SUMMARY:")
print(f"   Experiment: {EXPERIMENT_NAME}")
print(f"   F1 Score: {binary_f1:.4f} ")
print(f"   This evaluation ensures detections overlap with ground truth locations.")
print(f"   (IoU >= {SPATIAL_IOU_THRESHOLD}).") 