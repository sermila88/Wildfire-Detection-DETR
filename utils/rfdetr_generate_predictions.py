import os
from PIL import Image
from tqdm import tqdm
import supervision as sv
from rfdetr.detr import RFDETRBase
import numpy as np
import shutil

# --- Configuration ---
EXPERIMENT_NAME = "rfdetr_smoke_detection_v1"  # Change this to match your experiment
DATASET_ROOT = "/vol/bitbucket/si324/rf-detr-wildfire/data/pyro25img/images"
OUTPUT_ROOT = f"/vol/bitbucket/si324/rf-detr-wildfire/bounding_boxes/Pyro25Images/predicted_bboxes/{EXPERIMENT_NAME}"
MODEL_PATH = f"/vol/bitbucket/si324/rf-detr-wildfire/outputs/{EXPERIMENT_NAME}/checkpoints/checkpoint_best_ema.pth"

# --- Load fine-tuned model ---
model = RFDETRBase(pretrain_weights=MODEL_PATH)

print(f"🎯 Generating predictions for experiment: {EXPERIMENT_NAME}")
print(f"📁 Output directory: {OUTPUT_ROOT}")

splits = ["train", "valid", "test"]  

for split in splits:
    print(f"Predicting {split}...")

    images_dir = os.path.join(DATASET_ROOT, split)
    annotations_path = os.path.join(DATASET_ROOT, split, "_annotations.coco.json")
    output_dir = os.path.join(OUTPUT_ROOT, split)
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset in COCO format using supervision
    dataset = sv.DetectionDataset.from_coco(images_dir, annotations_path)

    thickness = sv.calculate_optimal_line_thickness((640, 640))
    text_scale = sv.calculate_optimal_text_scale((640, 640))
    bbox_annotator = sv.BoxAnnotator(thickness=thickness)
    label_annotator = sv.LabelAnnotator(
        text_color=sv.Color.BLACK,
        text_scale=text_scale,
        text_thickness=thickness,
        smart_position=True
    )

    # Loop through each image in the dataset 
    for path, image, _ in tqdm(dataset):
        # Load original image directly with PIL to preserve color space
        original_pil = Image.open(path)
        
        # Convert PIL to numpy array for supervision (ensure RGB)
        if original_pil.mode != 'RGB':
            original_pil = original_pil.convert('RGB')
        image_array = np.array(original_pil)
        
        # Predict bounding boxes on the image using RF-DETR (use PIL image for model)
        detections = model.predict(original_pil, threshold=0.1)
        
        # Create labels with confidence scores (skip if no detections)
        if (detections.class_id is not None and detections.confidence is not None and 
            len(detections.class_id) > 0):
            labels = [
                f"{dataset.classes[class_id]} {confidence:.2f}"
                for class_id, confidence in zip(detections.class_id, detections.confidence)
            ]
        else:
            labels = []

        # Annotate the image using supervision
        annotated_image = bbox_annotator.annotate(image_array.copy(), detections)
        annotated_image = label_annotator.annotate(annotated_image, detections, labels)

        # Convert back to PIL Image maintaining color integrity
        annotated_pil = Image.fromarray(annotated_image.astype(np.uint8), mode='RGB')
        
        # Save with original filename
        filename = os.path.basename(path)
        output_path = os.path.join(output_dir, filename)
        annotated_pil.save(output_path, quality=95)  # High quality JPEG

print(f"✅ Predicted bounding boxes generated for {EXPERIMENT_NAME} ")

print(f"\n🔍 Creating evaluation breakdown folders...")

def has_spatial_overlap(predictions, ground_truth, iou_threshold=0.1):
    """Check if predictions have spatial overlap with ground truth"""
    if len(predictions.xyxy) == 0 or len(ground_truth.xyxy) == 0:
        return False
    
    import torch
    from torchvision.ops import box_iou
    
    pred_boxes = torch.tensor(predictions.xyxy, dtype=torch.float32)
    gt_boxes = torch.tensor(ground_truth.xyxy, dtype=torch.float32)
    
    ious = box_iou(pred_boxes, gt_boxes)
    return (ious > iou_threshold).any().item()

# Create metrics breakdown directory
metrics_dir = os.path.join(OUTPUT_ROOT, "metrics_breakdown")
tp_dir = os.path.join(metrics_dir, "TP_true_positives")
fp_dir = os.path.join(metrics_dir, "FP_false_positives") 
fn_dir = os.path.join(metrics_dir, "FN_false_negatives")
tn_dir = os.path.join(metrics_dir, "TN_true_negatives")

for dir_path in [tp_dir, fp_dir, fn_dir, tn_dir]:
    os.makedirs(dir_path, exist_ok=True)

# Only evaluate on test set for breakdown
test_split = "test"
print(f"Creating breakdown for {test_split} set...")

images_dir = os.path.join(DATASET_ROOT, test_split)
annotations_path = os.path.join(DATASET_ROOT, test_split, "_annotations.coco.json")
predicted_images_dir = os.path.join(OUTPUT_ROOT, test_split)

# Load test dataset
test_dataset = sv.DetectionDataset.from_coco(images_dir, annotations_path)

# Also check for images not in COCO dataset (potential TNs)
all_test_images = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
coco_images = {os.path.basename(path) for path, _, _ in test_dataset}
non_coco_images = [img for img in all_test_images if img not in coco_images]

print(f"Found {len(non_coco_images)} additional images not in COCO dataset (potential empty scenes)")
print(f"Processing {len(test_dataset)} annotated test images + {len(non_coco_images)} non-annotated images...")

# Track counts for summary
tp_count = fp_count = fn_count = tn_count = 0

# Process annotated images (from COCO dataset)
for path, image, annotations in tqdm(test_dataset, desc="Processing annotated images"):
    # Get predicted image filename  
    filename = os.path.basename(path)
    predicted_image_path = os.path.join(predicted_images_dir, filename)
    
    # Skip if predicted image doesn't exist
    if not os.path.exists(predicted_image_path):
        continue
    
    # Get model predictions (re-run prediction to get detections object)
    original_pil = Image.open(path)
    if original_pil.mode != 'RGB':
        original_pil = original_pil.convert('RGB')
    
    detections = model.predict(original_pil, threshold=0.1)
    
    # Determine ground truth and prediction status
    has_smoke_gt = len(annotations.xyxy) > 0
    has_smoke_pred = len(detections.xyxy) > 0
    
    # Classify the prediction result
    if has_smoke_gt:
        # Ground truth has smoke
        if has_smoke_pred:
            # Check for spatial overlap
            spatial_match = has_spatial_overlap(detections, annotations)
            if spatial_match:
                # True Positive: Smoke detected with correct spatial overlap
                destination = tp_dir
                tp_count += 1
                category = "TP"
            else:
                # False Negative: Smoke present but not detected correctly
                destination = fn_dir
                fn_count += 1
                category = "FN"
        else:
            # False Negative: Smoke present but not detected at all
            destination = fn_dir
            fn_count += 1
            category = "FN"
    else:
        # Ground truth has no smoke (empty scene)
        if has_smoke_pred:
            # False Positive: No smoke but model detected something
            destination = fp_dir
            fp_count += 1
            category = "FP"
        else:
            # True Negative: No smoke and correctly identified as empty
            destination = tn_dir
            tn_count += 1
            category = "TN"
    
    # Copy the predicted/annotated image to appropriate category folder
    destination_path = os.path.join(destination, filename)
    shutil.copy2(predicted_image_path, destination_path)

# Process non-COCO images (likely empty scenes)
for filename in tqdm(non_coco_images, desc="Processing non-annotated images"):
    image_path = os.path.join(images_dir, filename)
    predicted_image_path = os.path.join(predicted_images_dir, filename)
    
    if not os.path.exists(predicted_image_path):
        continue
    
    # Get model predictions for non-annotated image
    original_pil = Image.open(image_path)
    if original_pil.mode != 'RGB':
        original_pil = original_pil.convert('RGB')
    
    detections = model.predict(original_pil, threshold=0.1)
    has_smoke_pred = len(detections.xyxy) > 0
    
    # These are empty scenes (no ground truth annotations)
    if has_smoke_pred:
        # False Positive: Model detected smoke in empty scene
        destination = fp_dir
        fp_count += 1
    else:
        # True Negative: Correctly identified empty scene
        destination = tn_dir
        tn_count += 1
    
    # Copy the predicted image
    destination_path = os.path.join(destination, filename)
    shutil.copy2(predicted_image_path, destination_path)

# Print summary
total_images = tp_count + fp_count + fn_count + tn_count
print(f"\n📊 Evaluation Breakdown Summary:")
print(f"   True Positives (TP):  {tp_count:3d} - Smoke correctly detected with spatial overlap")
print(f"   False Positives (FP): {fp_count:3d} - False alarms on empty scenes")
print(f"   False Negatives (FN): {fn_count:3d} - Missed smoke detections")
print(f"   True Negatives (TN):  {tn_count:3d} - Empty scenes correctly identified")
print(f"   Total Images:         {total_images:3d}")

# Initialize metrics variables
precision = 0
recall = 0
f1 = 0

# Calculate and display metrics
if (tp_count + fp_count) > 0:
    precision = tp_count / (tp_count + fp_count)
    print(f"   Precision: {precision:.3f}")

if (tp_count + fn_count) > 0:
    recall = tp_count / (tp_count + fn_count)
    print(f"   Recall: {recall:.3f}")

if (precision + recall) > 0:
    f1 = 2 * (precision * recall) / (precision + recall)
    print(f"   F1 Score: {f1:.3f}")

accuracy = (tp_count + tn_count) / total_images if total_images > 0 else 0
print(f"   Accuracy: {accuracy:.3f}")

print(f"\n✅ Breakdown images saved to:")
print(f"   📁 {metrics_dir}")
print(f"   📁 TP: {tp_dir}")
print(f"   📁 FP: {fp_dir}") 
print(f"   📁 FN: {fn_dir}")
print(f"   📁 TN: {tn_dir}")

# Create a summary text file
summary_path = os.path.join(metrics_dir, "breakdown_summary.txt")
with open(summary_path, 'w') as f:
    f.write(f"RF-DETR Evaluation Breakdown - {EXPERIMENT_NAME}\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Dataset Composition:\n")
    f.write(f"  Annotated images (COCO): {len(test_dataset)}\n")
    f.write(f"  Non-annotated images: {len(non_coco_images)}\n")
    f.write(f"  Total processed: {total_images}\n\n")
    f.write(f"Results:\n")
    f.write(f"  True Positives (TP):  {tp_count:3d} - Smoke correctly detected\n")
    f.write(f"  False Positives (FP): {fp_count:3d} - False alarms on empty scenes\n") 
    f.write(f"  False Negatives (FN): {fn_count:3d} - Missed smoke detections\n")
    f.write(f"  True Negatives (TN):  {tn_count:3d} - Empty scenes correctly identified\n\n")
    f.write(f"Metrics:\n")
    f.write(f"  Precision: {precision:.3f}\n")
    f.write(f"  Recall: {recall:.3f}\n") 
    f.write(f"  F1 Score: {f1:.3f}\n")
    f.write(f"  Accuracy: {accuracy:.3f}\n")

print(f"📄 Summary saved: {summary_path}")



