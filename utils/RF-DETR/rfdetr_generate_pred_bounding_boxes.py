"""
RF-DETR Predicted Bounding Box Visualizer with TP/FP/FN/TN Breakdown
Mirrors the RF-DETR evaluation logic exactly for consistent visualization
"""

import os
import cv2
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from PIL import Image
import supervision as sv
from rfdetr import RFDETRBase
from tqdm import tqdm

# ============================================================================
# CONFIGURATION - MUST MATCH YOUR EVAL SCRIPT
# ============================================================================
PROJECT_ROOT = "/vol/bitbucket/si324/rf-detr-wildfire"
EXPERIMENT_NAME = "rfdetr_smoke_detection_v1_IoU=0.1"  # Update to match eval

# Dataset paths
DATASET_PATH = f"{PROJECT_ROOT}/data/pyro25img/images/test"
ANNOTATIONS_PATH = f"{DATASET_PATH}/_annotations.coco.json"

# Model checkpoint
CHECKPOINTS_DIR = f"{PROJECT_ROOT}/outputs/{EXPERIMENT_NAME}/checkpoints"

# Load evaluation results to get best thresholds
EVAL_RESULTS_PATH = f"{PROJECT_ROOT}/outputs/{EXPERIMENT_NAME}/eval_results/evaluation_results.json"

def load_eval_thresholds():
    """Load best thresholds from evaluation results"""
    try:
        with open(EVAL_RESULTS_PATH, 'r') as f:
            eval_data = json.load(f)
        
        iou_threshold = eval_data['experiment_info']['iou_threshold']
        img_conf_threshold = eval_data['image_best_results']['confidence_threshold']
        obj_conf_threshold = eval_data['object_best_results']['confidence_threshold']
        
        return iou_threshold, img_conf_threshold, obj_conf_threshold
    
    except FileNotFoundError:
        print(f" Evaluation results not found at: {EVAL_RESULTS_PATH}")
        print(f"   Using default thresholds")
        return 0.1, 0.25, 0.30
    except KeyError as e:
        print(f" Missing key in evaluation results: {e}")
        print(f"   Using default thresholds")
        return 0.1, 0.25, 0.30

# Load thresholds from evaluation results
IOU_THRESHOLD, IMG_CONF_THRESHOLD, OBJ_CONF_THRESHOLD = load_eval_thresholds()

# Output directories
OUTPUT_ROOT = f"{PROJECT_ROOT}/bounding_boxes/{EXPERIMENT_NAME}/rfdetr_predicted_bb_breakdown"

# ============================================================================
# PYRONEAR UTILITY FUNCTIONS 
# ============================================================================
def xywh2xyxy(x):
    """Convert bounding box format from center to top-left corner"""
    y = np.zeros_like(x)
    y[0] = x[0] - x[2] / 2  # x_min
    y[1] = x[1] - x[3] / 2  # y_min
    y[2] = x[0] + x[2] / 2  # x_max
    y[3] = x[1] + x[3] / 2  # y_max
    return y

def box_iou(box1: np.ndarray, box2: np.ndarray, eps: float = 1e-7):
    """Calculate IoU - exact copy from eval script"""
    if box1.ndim == 1:
        box1 = box1.reshape(1, 4)
    if box2.ndim == 1:
        box2 = box2.reshape(1, 4)

    (a1, a2), (b1, b2) = np.split(box1, 2, 1), np.split(box2, 2, 1)
    inter = (
        (np.minimum(a2, b2[:, None, :]) - np.maximum(a1, b1[:, None, :]))
        .clip(0)
        .prod(2)
    )
    return inter / ((a2 - a1).prod(1) + (b2 - b1).prod(1)[:, None] - inter + eps)

# ============================================================================
# EVALUATION LOGIC 
# ============================================================================
def has_spatial_overlap(predictions, ground_truth, iou_th=IOU_THRESHOLD):
    """Check spatial overlap - exact copy from eval"""
    if len(predictions.xyxy) == 0 or len(ground_truth.xyxy) == 0:
        return False
    
    for pred_box in predictions.xyxy:
        for gt_box in ground_truth.xyxy:
            iou_matrix = box_iou(pred_box, gt_box)
            iou = iou_matrix[0, 0]
            if iou > iou_th:
                return True
    return False

def evaluate_image_classification(predictions, annotations, conf_th, iou_th=IOU_THRESHOLD):
    """Image-level classification - exact logic from eval"""
    has_smoke_gt = len(annotations.xyxy) > 0
    has_smoke_pred = len(predictions.xyxy) > 0
    
    spatial_match = False
    if has_smoke_gt and has_smoke_pred:
        spatial_match = has_spatial_overlap(predictions, annotations, iou_th)
    
    if has_smoke_gt:
        return 'TP' if spatial_match else 'FN'
    else:
        return 'FP' if has_smoke_pred else 'TN'

def evaluate_object_classifications(predictions, annotations, conf_th, iou_th=IOU_THRESHOLD):
    """Object-level classification - exact logic from eval"""
    gt = np.array(annotations.xyxy)
    pr = np.array(predictions.xyxy)
    
    obj_classifications = []
    
    # No GT boxes - all predictions are FP
    if gt.size == 0:
        for i in range(len(pr)):
            obj_classifications.append(('FP', i, -1, predictions.confidence[i] if hasattr(predictions, 'confidence') else 1.0, 0.0))
        return obj_classifications
    
    # Match predictions to GT
    matched = np.zeros(len(gt), dtype=bool)
    
    for i, pb in enumerate(pr):
        ious = []
        for gt_box in gt:
            iou_matrix = box_iou(pb, gt_box)
            iou_val = iou_matrix[0, 0]
            ious.append(iou_val)
        
        if len(ious) > 0:
            max_iou = float(np.max(ious))
            idx = int(np.argmax(ious))
            
            if max_iou > iou_th and not matched[idx]:
                obj_classifications.append(('TP', i, idx, predictions.confidence[i] if hasattr(predictions, 'confidence') else 1.0, max_iou))
                matched[idx] = True
            else:
                obj_classifications.append(('FP', i, -1, predictions.confidence[i] if hasattr(predictions, 'confidence') else 1.0, max_iou))
    
    # Mark unmatched GT as FN
    for i, is_matched in enumerate(matched):
        if not is_matched:
            obj_classifications.append(('FN', -1, i, 0.0, 0.0))
    
    return obj_classifications

# ============================================================================
# VISUALIZATION
# ============================================================================
def draw_bounding_boxes(image_path, predictions, annotations, output_path,
                        obj_classifications, img_classification, conf_th, evaluation_type='image'):
    """Draw bounding boxes with classifications"""
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    img_height, img_width = image.shape[:2]
    output_image = image.copy()
    
    # Colors
    colors = {
        'TP': (0, 255, 0),      # Green
        'FP': (0, 0, 255),      # Red
        'FN': (255, 0, 0),      # Blue
        'GT': (0, 255, 255),    # Yellow
    }
    
    # Draw ground truth boxes (yellow)
    for i, gt_box in enumerate(annotations.xyxy):
        x1, y1, x2, y2 = gt_box.astype(int)
        cv2.rectangle(output_image, (x1, y1), (x2, y2), colors['GT'], 3)
        cv2.putText(output_image, f'GT_{i}', (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors['GT'], 2)

    # Draw predictions with classifications
    if evaluation_type == 'image':
        # IMAGE-LEVEL: Only show boxes that determine the image classification
        if img_classification == 'TP':
            # Show ONLY the prediction(s) that have IoU > threshold (these make it TP)
            for i, pred_box in enumerate(predictions.xyxy):
                x1, y1, x2, y2 = pred_box.astype(int)
                conf = predictions.confidence[i] if hasattr(predictions, 'confidence') else 1.0
                
                # Check if this prediction overlaps with any GT
                best_iou = 0.0
                for gt_box in annotations.xyxy:
                    iou = box_iou(pred_box, gt_box)[0, 0]
                    best_iou = max(best_iou, iou)
                
                if best_iou > IOU_THRESHOLD:
                    # ONLY show predictions that contribute to making the image TP
                    color = colors['TP']
                    label = f'TP: {conf:.2f} IoU:{best_iou:.2f}'
                    
                    cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 4)
                    label_y = y2 + 25
                    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(output_image, (x1-2, label_y-text_h-5), 
                                (x1+text_w+5, label_y+5), (0, 0, 0), -1)
                    cv2.putText(output_image, label, (x1, label_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                # Don't draw predictions with IoU <= threshold for TP images
                
        elif img_classification == 'FP':
            # No GT, ALL predictions contribute to FP - show them all
            for i, pred_box in enumerate(predictions.xyxy):
                x1, y1, x2, y2 = pred_box.astype(int)
                conf = predictions.confidence[i] if hasattr(predictions, 'confidence') else 1.0
                color = colors['FP']
                
                cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 4)
                label = f'FP: {conf:.2f}'
                label_y = y2 + 25
                (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(output_image, (x1-2, label_y-text_h-5), 
                            (x1+text_w+5, label_y+5), (0, 0, 0), -1)
                cv2.putText(output_image, label, (x1, label_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        elif img_classification == 'FN':
            # Has GT but no predictions match - show blue FN circles only
            # Don't draw any prediction boxes (they don't contribute to FN classification)
            pass
        
        elif img_classification == 'TN':
            # No GT and no predictions - nothing to draw except maybe a note
            pass
        
        # Draw FN markers (blue circles) for unmatched GT boxes (for FN and TP images)
        if img_classification in ['FN', 'TP']:
            matched_gt = set()
            if img_classification == 'TP':
                # Find which GTs were matched
                for i, pred_box in enumerate(predictions.xyxy):
                    for j, gt_box in enumerate(annotations.xyxy):
                        iou = box_iou(pred_box, gt_box)[0, 0]
                        if iou > IOU_THRESHOLD:
                            matched_gt.add(j)
            
            # Draw FN circles for unmatched GTs
            for j, gt_box in enumerate(annotations.xyxy):
                if j not in matched_gt:
                    x1, y1 = gt_box[:2].astype(int)
                    cv2.circle(output_image, (x1-20, y1-20), 15, colors['FN'], -1)
                    cv2.circle(output_image, (x1-20, y1-20), 15, (255, 255, 255), 2)
                    cv2.putText(output_image, 'FN', (x1-35, y1-15), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
    elif obj_classifications:  # OBJECT-LEVEL 
        # Map predictions to their classifications
        pred_map = {}
        for classification, pred_idx, gt_idx, confidence, iou in obj_classifications:
            if pred_idx >= 0:
                pred_map[pred_idx] = (classification, confidence, iou)
        
        # Draw predictions
        for i, pred_box in enumerate(predictions.xyxy):
            if i in pred_map:
                classification, conf, iou = pred_map[i]
                color = colors[classification]
                x1, y1, x2, y2 = pred_box.astype(int)
                
                cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 4)
                
                label = f'{classification}: {conf:.2f} IoU:{iou:.2f}'
                label_y = y2 + 25
                
                # Background for text
                (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(output_image, (x1-2, label_y-text_h-5), 
                            (x1+text_w+5, label_y+5), (0, 0, 0), -1)
                cv2.putText(output_image, label, (x1, label_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw FN markers
        for classification, pred_idx, gt_idx, confidence, iou in obj_classifications:
            if classification == 'FN' and gt_idx >= 0:
                gt_box = annotations.xyxy[gt_idx]
                x1, y1 = gt_box[:2].astype(int)
                
                cv2.circle(output_image, (x1-20, y1-20), 15, colors['FN'], -1)
                cv2.circle(output_image, (x1-20, y1-20), 15, (255, 255, 255), 2)
                cv2.putText(output_image, 'FN', (x1-50, y1-25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors['FN'], 2)
    
    # Add header
    header = f"{evaluation_type.upper()}: {img_classification} | GT: {len(annotations.xyxy)} | Pred@{conf_th:.2f}: {len(predictions.xyxy)}"
    (text_w, text_h), _ = cv2.getTextSize(header, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.rectangle(output_image, (5, 5), (text_w + 20, text_h + 30), (0, 0, 0), -1)
    cv2.rectangle(output_image, (5, 5), (text_w + 20, text_h + 30), (255, 255, 255), 2)
    cv2.putText(output_image, header, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Add legend
    legend_y = img_height - 100
    legend_items = [
        ("GT (Yellow)", colors['GT']),
        ("TP (Green)", colors['TP']),
        ("FP (Red)", colors['FP']),
        ("FN (Blue)", colors['FN'])
    ]
    
    for i, (text, color) in enumerate(legend_items):
        y_pos = legend_y + (i * 22)
        cv2.rectangle(output_image, (10, y_pos-15), (40, y_pos-5), color, -1)
        cv2.putText(output_image, text, (50, y_pos-8), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    cv2.imwrite(output_path, output_image)
    
    return {
        'image_path': image_path,
        'output_path': output_path,
        'img_classification': img_classification,
        'num_gt': len(annotations.xyxy),
        'num_pred': len(predictions.xyxy),
        'confidence_threshold': conf_th
    }

# ============================================================================
# MAIN VISUALIZATION PIPELINE
# ============================================================================
def generate_visualizations():
    """Generate all visualizations matching eval logic exactly"""
    
    print(f"🎯 RF-DETR Bounding Box Visualizer")
    print(f"📁 Experiment: {EXPERIMENT_NAME}")
    print(f"🎚️ Image-Level Confidence: {IMG_CONF_THRESHOLD}")
    print(f"🎚️ Object-Level Confidence: {OBJ_CONF_THRESHOLD}")
    print(f"🎯 IoU Threshold: {IOU_THRESHOLD}")
    
    # Create output directories
    img_dirs = {
        'TP': os.path.join(OUTPUT_ROOT, 'image_level', 'TP'),
        'FP': os.path.join(OUTPUT_ROOT, 'image_level', 'FP'),
        'FN': os.path.join(OUTPUT_ROOT, 'image_level', 'FN'),
        'TN': os.path.join(OUTPUT_ROOT, 'image_level', 'TN')
    }
    
    obj_dirs = {
        'TP': os.path.join(OUTPUT_ROOT, 'object_level', 'TP'),
        'FP': os.path.join(OUTPUT_ROOT, 'object_level', 'FP'),
        'FN': os.path.join(OUTPUT_ROOT, 'object_level', 'FN')
    }
    
    for dirs_dict in [img_dirs, obj_dirs]:
        for dir_path in dirs_dict.values():
            os.makedirs(dir_path, exist_ok=True)
    
    # Load model
    checkpoint_path = f"{CHECKPOINTS_DIR}/checkpoint_best_ema.pth"
    if not os.path.exists(checkpoint_path):
        print(f" Checkpoint not found: {checkpoint_path}")
        return
    
    print(f" Loading RF-DETR from: {checkpoint_path}")
    model = RFDETRBase(pretrain_weights=checkpoint_path, num_classes=1)
    
    # Load dataset
    if not os.path.exists(ANNOTATIONS_PATH):
        print(f"❌ Annotations not found: {ANNOTATIONS_PATH}")
        return
    
    coco_ds = sv.DetectionDataset.from_coco(
        images_directory_path=DATASET_PATH,
        annotations_path=ANNOTATIONS_PATH
    )
    
    # Get all test images (including non-annotated)
    all_test_images = [f for f in os.listdir(DATASET_PATH) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    coco_images = {os.path.basename(path) for path, _, _ in coco_ds}
    non_coco_images = [img for img in all_test_images if img not in coco_images]
    
    # Create combined dataset
    combined_dataset = []
    for path, image, annotations in coco_ds:
        combined_dataset.append((path, annotations))
    
    for img_name in non_coco_images:
        img_path = os.path.join(DATASET_PATH, img_name)
        if os.path.exists(img_path):
            combined_dataset.append((img_path, sv.Detections.empty()))

    print(f"📊 Processing {len(combined_dataset)} images")
    
    # Statistics
    stats = {
        'image_level': {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0},
        'object_level': {'TP': 0, 'FP': 0, 'FN': 0}
    }
    
    # Process each image
    for image_path, annotations in tqdm(combined_dataset, desc="Generating visualizations"):
        filename = os.path.basename(image_path).replace('.jpg', '').replace('.jpeg', '').replace('.png', '')
        
        # Get predictions at IMAGE-LEVEL threshold
        with Image.open(image_path) as img:
            img_rgb = img.convert("RGB")
            img_predictions = model.predict(img_rgb, threshold=IMG_CONF_THRESHOLD)
        
        # Image-level classification
        img_classification = evaluate_image_classification(img_predictions, annotations, IMG_CONF_THRESHOLD)
        img_obj_classifications = evaluate_object_classifications(img_predictions, annotations, IMG_CONF_THRESHOLD)
        
        # Generate image-level visualization
        img_output_path = os.path.join(img_dirs[img_classification], 
                                       f"{filename}_img_th{IMG_CONF_THRESHOLD:.2f}.jpg")
        
        draw_bounding_boxes(image_path, img_predictions, annotations, img_output_path,
                          img_obj_classifications, img_classification, IMG_CONF_THRESHOLD, 'image')
        
        stats['image_level'][img_classification] += 1
        
        # Get predictions at OBJECT-LEVEL threshold
        with Image.open(image_path) as img:
            img_rgb = img.convert("RGB")
            obj_predictions = model.predict(img_rgb, threshold=OBJ_CONF_THRESHOLD)
        
        # Object-level classification
        obj_classifications = evaluate_object_classifications(obj_predictions, annotations, OBJ_CONF_THRESHOLD)
        
        # Count object-level stats
        for obj_class, _, _, _, _ in obj_classifications:
            if obj_class in stats['object_level']:
                stats['object_level'][obj_class] += 1
        
        # Generate object-level visualizations (one per classification)
        for obj_class, pred_idx, gt_idx, conf, iou in obj_classifications:
            if obj_class in obj_dirs:
                if pred_idx >= 0:
                    obj_filename = f"{filename}_{obj_class}_pred{pred_idx}_conf{conf:.2f}_iou{iou:.2f}.jpg"
                else:
                    obj_filename = f"{filename}_{obj_class}_gt{gt_idx}.jpg"
                
                obj_output_path = os.path.join(obj_dirs[obj_class], obj_filename)
                
                # For visualization, show the specific classification
                vis_classifications = [(obj_class, pred_idx, gt_idx, conf, iou)]
                
                draw_bounding_boxes(image_path, obj_predictions, annotations, obj_output_path,
                                  vis_classifications, obj_class, OBJ_CONF_THRESHOLD, 'object')
    
    # Save summary
    summary = {
        'experiment_name': EXPERIMENT_NAME,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'thresholds_used': {
            'iou_threshold': IOU_THRESHOLD,
            'image_confidence_threshold': IMG_CONF_THRESHOLD,
            'object_confidence_threshold': OBJ_CONF_THRESHOLD,
            'loaded_from': EVAL_RESULTS_PATH if os.path.exists(EVAL_RESULTS_PATH) else 'default_values'
        },
        'total_images': len(combined_dataset),
        'image_level_counts': stats['image_level'],
        'object_level_counts': stats['object_level'],
        'image_level_confusion_matrix': {
            'true_positives': stats['image_level']['TP'],
            'true_negatives': stats['image_level']['TN'],
            'false_positives': stats['image_level']['FP'],
            'false_negatives': stats['image_level']['FN']
        },
        'object_level_confusion_matrix': {
            'true_positives': stats['object_level']['TP'],
            'false_positives': stats['object_level']['FP'],
            'false_negatives': stats['object_level']['FN'],
            'true_negatives': 'N/A'
        },
        'output_directories': {
            'image_level': img_dirs,
            'object_level': obj_dirs
        }
    }
    
    summary_path = os.path.join(OUTPUT_ROOT, 'visualization_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n Visualization Complete!")
    print(f" Thresholds Used:")
    print(f"   IoU: {IOU_THRESHOLD}")
    print(f"   Image Conf: {IMG_CONF_THRESHOLD:.3f}")
    print(f"   Object Conf: {OBJ_CONF_THRESHOLD:.3f}")
    print(f"   Source: {EVAL_RESULTS_PATH if os.path.exists(EVAL_RESULTS_PATH) else 'default values'}")
    print(f"\n Image-Level Results (conf={IMG_CONF_THRESHOLD:.3f}):")
    for classification, count in stats['image_level'].items():
        print(f"   {classification}: {count}")
    print(f" Object-Level Results (conf={OBJ_CONF_THRESHOLD:.3f}):")
    for classification, count in stats['object_level'].items():
        print(f"   {classification}: {count}")
    print(f" Summary saved to: {summary_path}")
    print(f" Outputs in: {OUTPUT_ROOT}")

if __name__ == "__main__":
    generate_visualizations()