"""
YOLO Predicted Bounding Box Visualizer with TP/FP/FN/TN Breakdown - FIXED VISUALIZATION VERSION
Generates predicted bounding box visualizations categorized by metrics 
"""

import os
import cv2
import numpy as np
from pathlib import Path
import argparse
from typing import List, Tuple, Dict
import json
from datetime import datetime
import numpy as np

#Pyronear utils functions
def xywh2xyxy(x):
    """Function to convert bounding box format from center to top-left corner"""
    y = np.zeros_like(x)
    y[0] = x[0] - x[2] / 2  # x_min
    y[1] = x[1] - x[3] / 2  # y_min
    y[2] = x[0] + x[2] / 2  # x_max
    y[3] = x[1] + x[3] / 2  # y_max
    return y

def box_iou(box1: np.ndarray, box2: np.ndarray, eps: float = 1e-7):
    # Ensure box1 and box2 are in the shape (N, 4) even if N is 1
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

    # IoU = inter / (area1 + area2 - inter)
    return inter / ((a2 - a1).prod(1) + (b2 - b1).prod(1)[:, None] - inter + eps)

def xywh_to_xyxy_absolute(x_center, y_center, width, height, img_width, img_height):
    """Convert YOLO format to absolute coordinates"""
    x_center_abs = x_center * img_width
    y_center_abs = y_center * img_height
    width_abs = width * img_width
    height_abs = height * img_height
    
    x1 = int(x_center_abs - width_abs / 2)
    y1 = int(y_center_abs - height_abs / 2)
    x2 = int(x_center_abs + width_abs / 2)
    y2 = int(y_center_abs + height_abs / 2)
    
    return x1, y1, x2, y2

def load_yolo_annotations(label_path: str) -> List[Tuple[int, float, float, float, float]]:
    """Load YOLO format annotations"""
    annotations = []
    if os.path.isfile(label_path) and os.path.getsize(label_path) > 0:
        with open(label_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) == 5:  # GT format
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    annotations.append((class_id, x_center, y_center, width, height, 1.0))
                elif len(parts) == 6:  # Prediction format
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    confidence = float(parts[5])
                    annotations.append((class_id, x_center, y_center, width, height, confidence))
    return annotations

def evaluate_image_classification(gt_boxes, pred_boxes, conf_th=0.1, iou_th=0.1):
    """
    Evaluate image-level classification for a single image
    Exactly matches the evaluation logic from your eval script
    """
    # Convert to xyxy format for IoU calculation
    gt_xyxy = [xywh2xyxy(np.array(box[1:5])) for box in gt_boxes]
    
    # Track GT matches for spatial overlap check
    gt_matches = np.zeros(len(gt_boxes), dtype=bool)
    
    # Check each prediction against GT boxes
    if pred_boxes:  # If predictions exist
        for pred_box in pred_boxes:
            confidence = pred_box[5]
            if confidence < conf_th:
                continue
                
            pred_xyxy = xywh2xyxy(np.array(pred_box[1:5]))
            
            if gt_xyxy:
                # Find best IoU match - exactly like eval script
                iou_values = []
                for gt_box in gt_xyxy:
                    iou_matrix = box_iou(pred_xyxy, gt_box)
                    iou_scalar = float(iou_matrix[0, 0]) if iou_matrix.size > 0 else 0.0
                    iou_values.append(iou_scalar)
                
                if iou_values:
                    max_iou = max(iou_values)
                    best_match_idx = np.argmax(iou_values)
                    
                    if max_iou > iou_th and not gt_matches[best_match_idx]:
                        gt_matches[best_match_idx] = True
    
    # Image-level classification logic - exactly matching eval script
    has_smoke_gt = len(gt_boxes) > 0
    has_smoke_pred = len([p for p in pred_boxes if p[5] >= conf_th]) > 0
    spatial_match = np.sum(gt_matches) > 0 if has_smoke_gt and has_smoke_pred else False
    
    if has_smoke_gt:
        return 'TP' if spatial_match else 'FN'
    else:
        return 'FP' if has_smoke_pred else 'TN'

def evaluate_object_classifications(gt_boxes, pred_boxes, conf_th=0.2, iou_th=0.1):
    """
    Evaluate object-level classifications for a single image
    Exactly matches the evaluation logic from your eval script
    """
    # Filter predictions by confidence
    valid_preds = [box for box in pred_boxes if box[5] >= conf_th]
    
    # Convert to xyxy format for IoU calculation
    gt_xyxy = [xywh2xyxy(np.array(box[1:5])) for box in gt_boxes]
    pred_xyxy = [xywh2xyxy(np.array(box[1:5])) for box in valid_preds]
    
    # Object-level evaluation
    gt_matches = np.zeros(len(gt_boxes), dtype=bool)
    obj_classifications = []
    
    # Check each valid prediction
    for i, pred_box in enumerate(valid_preds):
        pred_xyxy_box = pred_xyxy[i]
        confidence = pred_box[5]
        
        if gt_xyxy:
            # Find best IoU match - exactly like eval script
            iou_values = []
            for gt_box in gt_xyxy:
                iou_matrix = box_iou(pred_xyxy_box, gt_box)
                iou_scalar = float(iou_matrix[0, 0]) if iou_matrix.size > 0 else 0.0
                iou_values.append(iou_scalar)
            
            if iou_values:
                max_iou = max(iou_values)
                best_match_idx = np.argmax(iou_values)
                
                if max_iou > iou_th and not gt_matches[best_match_idx]:
                    obj_classifications.append(('TP', i, best_match_idx, confidence, max_iou))
                    gt_matches[best_match_idx] = True
                else:
                    obj_classifications.append(('FP', i, -1, confidence, max_iou))
            else:
                obj_classifications.append(('FP', i, -1, confidence, 0.0))
        else:
            obj_classifications.append(('FP', i, -1, confidence, 0.0))
    
    # Count FN (unmatched GT boxes)
    for i, matched in enumerate(gt_matches):
        if not matched:
            obj_classifications.append(('FN', -1, i, 0.0, 0.0))
    
    return obj_classifications

def draw_predictions_with_classification(image_path: str, gt_path: str, pred_path: str, 
                                       output_path: str, conf_th: float,
                                       obj_classifications: List = None, 
                                       img_classification: str = '',
                                       evaluation_type: str = 'image') -> Dict:
    """Draw predictions with TP/FP/FN classifications - CLEANEST VERSION"""
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    img_height, img_width = image.shape[:2]
    output_image = image.copy()
    
    # Load annotations
    gt_boxes = load_yolo_annotations(gt_path)
    pred_boxes = load_yolo_annotations(pred_path)
    
    # Filter predictions by confidence threshold used for this evaluation
    valid_preds = [box for box in pred_boxes if box[5] >= conf_th]
    
    # Clean, distinct colors
    colors = {
        'TP': (0, 255, 0),      # Bright Green - True Positive
        'FP': (0, 0, 255),      # Bright Red - False Positive  
        'FN': (255, 0, 0),      # Bright Blue - False Negative
        'GT': (0, 255, 255),    # Bright Yellow - Ground Truth
    }
    
    # Draw ground truth boxes (yellow)
    for i, (class_id, x_center, y_center, width, height, _) in enumerate(gt_boxes):
        x1, y1, x2, y2 = xywh_to_xyxy_absolute(x_center, y_center, width, height, img_width, img_height)
        cv2.rectangle(output_image, (x1, y1), (x2, y2), colors['GT'], 3)
        cv2.putText(output_image, f'GT_{i}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors['GT'], 2)
    
    # Draw ONLY properly classified predictions
    if obj_classifications:
        # Create mapping for quick lookup
        pred_classification_map = {}
        for classification, pred_idx, gt_idx, confidence, iou in obj_classifications:
            if pred_idx >= 0:  # Valid prediction index
                pred_classification_map[pred_idx] = (classification, confidence, iou)
        
        # Draw all predictions above threshold with their correct colors
        for i, pred_box in enumerate(valid_preds):
            x_center, y_center, width, height, confidence = pred_box[1:6]
            x1, y1, x2, y2 = xywh_to_xyxy_absolute(x_center, y_center, width, height, img_width, img_height)
            
            # Only draw if properly classified
            if i in pred_classification_map:
                classification, conf, iou = pred_classification_map[i]
                color = colors[classification]
                
                # Draw thick, visible box
                cv2.rectangle(output_image, (x1-2, y1-2), (x2+2, y2+2), color, 4)
                
                # Clear label with classification, confidence AND IoU
                label = f'{classification}: {conf:.2f} IoU:{iou:.2f}'
                
                # Position labels with black background for readability
                label_y = y2 + 25 + (i * 20)
                (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(output_image, (x1-2, label_y-text_h-5), (x1+text_w+5, label_y+5), (0, 0, 0), -1)
                cv2.putText(output_image, label, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw FN markers on unmatched GT boxes
        for classification, pred_idx, gt_idx, confidence, iou in obj_classifications:
            if classification == 'FN' and gt_idx >= 0:
                gt_box = gt_boxes[gt_idx]
                x_center, y_center, width, height = gt_box[1:5]
                x1, y1, x2, y2 = xywh_to_xyxy_absolute(x_center, y_center, width, height, img_width, img_height)
                
                # Draw large, visible FN marker
                cv2.circle(output_image, (x1-20, y1-20), 15, colors['FN'], -1)
                cv2.circle(output_image, (x1-20, y1-20), 15, (255, 255, 255), 2)  # White border
                cv2.putText(output_image, 'FN', (x1-50, y1-25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors['FN'], 2)
    
    # Add clear header with better contrast
    stats_text = f"{evaluation_type.upper()}: {img_classification} | GT: {len(gt_boxes)} | Pred@{conf_th}: {len(valid_preds)}"
    
    # Large background box for header
    (text_w, text_h), _ = cv2.getTextSize(stats_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.rectangle(output_image, (5, 5), (text_w + 20, text_h + 30), (0, 0, 0), -1)
    cv2.rectangle(output_image, (5, 5), (text_w + 20, text_h + 30), (255, 255, 255), 2)
    cv2.putText(output_image, stats_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Add clean legend in bottom corner
    legend_y_start = img_height - 100
    legend_items = [
        ("GT (Yellow)", colors['GT']),
        ("TP (Green)", colors['TP']),
        ("FP (Red)", colors['FP']),
        ("FN (Blue)", colors['FN'])
    ]
    
    for i, (text, color) in enumerate(legend_items):
        y_pos = legend_y_start + (i * 22)
        cv2.rectangle(output_image, (10, y_pos-15), (40, y_pos-5), color, -1)
        cv2.putText(output_image, text, (50, y_pos-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(output_image, text, (50, y_pos-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Save output
    cv2.imwrite(output_path, output_image)
    
    return {
        'image_path': image_path,
        'gt_path': gt_path,
        'pred_path': pred_path,
        'output_path': output_path,
        'img_classification': img_classification,
        'obj_classifications': obj_classifications,
        'num_gt_boxes': len(gt_boxes),
        'num_pred_boxes': len(valid_preds),
        'confidence_threshold': conf_th,
        'evaluation_type': evaluation_type
    }

def generate_prediction_visualizations(experiment_name: str, 
                                     images_dir: str,
                                     gt_labels_dir: str,
                                     pred_labels_dir: str,
                                     split: str = 'test',
                                     img_conf_th: float = 0.1,
                                     obj_conf_th: float = 0.2,
                                     iou_th: float = 0.1):  
    """Generate prediction visualizations categorized by TP/FP/FN/TN"""
    
    print(f"🎯 Generating CLEAN Prediction Visualizations for {experiment_name}")
    print(f"📁 Images: {images_dir}")
    print(f"🏷️  GT Labels: {gt_labels_dir}")
    print(f"🔮 Pred Labels: {pred_labels_dir}")
    print(f"📊 Split: {split}")
    print(f"🎚️  Image-Level Confidence Threshold: {img_conf_th}")
    print(f"🎚️  Object-Level Confidence Threshold: {obj_conf_th}")
    print(f"🎯 IoU Threshold: {iou_th}")
    
    # Create output directories
    output_root = f"bounding_boxes/{experiment_name}/yolo_predicted_bb_breakdown"
    
    # Image-level directories
    img_level_dirs = {
        'TP': os.path.join(output_root, 'image_level', 'TP'),
        'FP': os.path.join(output_root, 'image_level', 'FP'), 
        'FN': os.path.join(output_root, 'image_level', 'FN'),
        'TN': os.path.join(output_root, 'image_level', 'TN')
    }
    
    # Object-level directories (no TN for object-level)
    obj_level_dirs = {
        'TP': os.path.join(output_root, 'object_level', 'TP'),
        'FP': os.path.join(output_root, 'object_level', 'FP'),
        'FN': os.path.join(output_root, 'object_level', 'FN')
    }
    
    # Create all directories
    for dirs_dict in [img_level_dirs, obj_level_dirs]:
        for dir_path in dirs_dict.values():
            os.makedirs(dir_path, exist_ok=True)
    
    # Get all images - Match eval code exactly
    test_images_dir = os.path.join(images_dir, split)
    split_gt_dir = os.path.join(gt_labels_dir, split)
    
    if not os.path.exists(test_images_dir):
        print(f"❌ Images directory not found: {test_images_dir}")
        return
    
    # Match evaluation code's image discovery logic
    all_image_files = [f for f in os.listdir(test_images_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    all_filenames = [os.path.splitext(f)[0] for f in all_image_files]
    
    print(f"📊 Found {len(all_filenames)} images to process")
    
    # Statistics
    stats = {
        'image_level': {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0},
        'object_level': {'TP': 0, 'FP': 0, 'FN': 0},
        'processed_images': 0,
        'detailed_results': []
    }
    
    # Process each image
    for filename in sorted(all_filenames):
        image_path = os.path.join(test_images_dir, f"{filename}.jpg")
        if not os.path.exists(image_path):
            # Try other extensions
            for ext in ['.jpeg', '.png']:
                alt_path = os.path.join(test_images_dir, f"{filename}{ext}")
                if os.path.exists(alt_path):
                    image_path = alt_path
                    break
            else:
                continue  # Skip if image not found
        
        gt_file = os.path.join(split_gt_dir, f"{filename}.txt")
        pred_file = os.path.join(pred_labels_dir, f"{filename}.txt")
        
        # Load annotations
        gt_boxes = load_yolo_annotations(gt_file)
        pred_boxes = load_yolo_annotations(pred_file)
        
        # IMAGE-LEVEL EVALUATION (using image-level threshold)
        img_classification = evaluate_image_classification(gt_boxes, pred_boxes, img_conf_th, iou_th)
        
        # OBJECT-LEVEL EVALUATION (using object-level threshold)
        obj_classifications = evaluate_object_classifications(gt_boxes, pred_boxes, obj_conf_th, iou_th)
        
        # Generate IMAGE-LEVEL visualization
        img_output_path = os.path.join(img_level_dirs[img_classification], f"{filename}_img_th{img_conf_th}.jpg")
        
        # Generate IMAGE-LEVEL object classifications using IMAGE-LEVEL threshold
        img_level_obj_classifications = evaluate_object_classifications(gt_boxes, pred_boxes, img_conf_th, iou_th)
        
        img_result = draw_predictions_with_classification(
            image_path, gt_file, pred_file, img_output_path,
            img_conf_th, img_level_obj_classifications, img_classification, 'image'
        )

        if img_result:
            stats['detailed_results'].append(img_result)
            stats['processed_images'] += 1
            stats['image_level'][img_classification] += 1
            
            # Count object-level classifications
            for obj_class, _, _, _, _ in obj_classifications:
                if obj_class in stats['object_level']:
                    stats['object_level'][obj_class] += 1
            
            # Generate OBJECT-LEVEL visualizations - one image per object instance
            for obj_class, pred_idx, gt_idx, conf, iou in obj_classifications:
                if obj_class in obj_level_dirs:  # TP, FP, or FN
                    # Create unique filename for each object instance
                    if pred_idx >= 0:
                        obj_filename = f"{filename}_{obj_class}_pred{pred_idx}_conf{conf:.2f}_iou{iou:.2f}.jpg"
                    else:
                        obj_filename = f"{filename}_{obj_class}_gt{gt_idx}.jpg"
                    
                    obj_output_path = os.path.join(obj_level_dirs[obj_class], obj_filename)
                    
                    # For FN: show ALL predictions to see what caused the miss
                    # For TP/FP: focus on single object as before
                    if obj_class == 'FN':
                        # Show all object classifications to see FP predictions that interfered
                        visualization_data = obj_classifications
                        title = f"FN GT_{gt_idx} - All Predictions Shown"
                    else:
                        # Focus on single object for TP/FP
                        visualization_data = [(obj_class, pred_idx, gt_idx, conf, iou)]
                        title = f"{obj_class} Object (Conf: {conf:.2f}, IoU: {iou:.2f})"
                    
                    draw_predictions_with_classification(
                        image_path, gt_file, pred_file, obj_output_path,
                        obj_conf_th, visualization_data, title, 'object'
                    )
                
        if stats['processed_images'] % 50 == 0:
            print(f"⏳ Processed {stats['processed_images']}/{len(all_filenames)} images...")
    
    # Save statistics
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(convert_numpy_types(item) for item in obj)
        return obj
    
    summary = {
        'experiment_name': experiment_name,
        'split': split,
        'image_confidence_threshold': float(img_conf_th),
        'object_confidence_threshold': float(obj_conf_th),
        'iou_threshold': float(iou_th),
        'generation_timestamp': datetime.now().isoformat(),
        'total_images_found': int(len(all_filenames)),
        'processed_images': int(stats['processed_images']),
        'image_level_counts': {k: int(v) for k, v in stats['image_level'].items()},
        'object_level_counts': {k: int(v) for k, v in stats['object_level'].items()},
        'output_directories': {
            'image_level': img_level_dirs,
            'object_level': obj_level_dirs
        },
        'detailed_results': convert_numpy_types(stats['detailed_results'])
    }
    
    stats_file = os.path.join(output_root, f'{split}_prediction_breakdown_stats.json')
    with open(stats_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ CLEAN Prediction Visualization Complete!")
    print(f"📊 Total Images: {len(all_filenames)}")
    print(f"✅ Processed: {stats['processed_images']}")
    print(f"\n📈 Image-Level Results (conf_th={img_conf_th}):")
    for classification, count in stats['image_level'].items():
        print(f"   {classification}: {count}")
    print(f"\n📈 Object-Level Results (conf_th={obj_conf_th}):")
    for classification, count in stats['object_level'].items():
        print(f"   {classification}: {count}")
    
    # Verification: Show object-level folder distribution
    print(f"\n🗂️ Object-Level Folder Distribution:")
    total_obj_files = 0
    for obj_type in ['TP', 'FP', 'FN']:
        folder_path = obj_level_dirs[obj_type]
        if os.path.exists(folder_path):
            count = len([f for f in os.listdir(folder_path) if f.endswith('.jpg')])
            total_obj_files += count
            print(f"   {obj_type} folder: {count} images")
    
    print(f"   Total object-level images: {total_obj_files}")
    print(f"   Expected sum: {sum(stats['object_level'].values())}")
    
    if total_obj_files != sum(stats['object_level'].values()):
        print(f"   ⚠️  WARNING: Folder count mismatch detected!")
    else:
        print(f"   ✅ Folder counts match perfectly!")
    
    print(f"\n🎨 Clean Color Legend:")
    print(f"   🟡 Yellow boxes = Ground Truth (GT)")
    print(f"   🟢 Green boxes = True Positive (TP) - Correct detections")
    print(f"   🔴 Red boxes = False Positive (FP) - Incorrect detections")  
    print(f"   🔵 Blue circles = False Negative (FN) - Missed detections")
    print(f"   📊 All boxes show: Classification: Confidence IoU:Value")
    
    print(f"\n💾 Output Directories:")
    print(f"   📁 Image-level: {output_root}/image_level/")
    print(f"   📁 Object-level: {output_root}/object_level/")
    print(f"📈 Stats saved to: {stats_file}")

def main():
    parser = argparse.ArgumentParser(description='Generate CLEAN YOLO Prediction Bounding Box Visualizations')
    parser.add_argument('--experiment_name', type=str, required=True,
                        help='Name of the experiment')
    parser.add_argument('--images_dir', type=str, required=True,
                        help='Path to images directory')
    parser.add_argument('--gt_labels_dir', type=str, required=True,
                        help='Path to ground truth labels directory')
    parser.add_argument('--pred_labels_dir', type=str, required=True,
                        help='Path to prediction labels directory')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'],
                        help='Which split to process')
    parser.add_argument('--img_conf_th', type=float, default=0.1,
                        help='Image-level confidence threshold')
    parser.add_argument('--obj_conf_th', type=float, default=0.2,
                        help='Object-level confidence threshold')
    parser.add_argument('--iou_th', type=float, default=0.1,
                        help='IoU threshold for matching predictions to ground truth')
    
    args = parser.parse_args()
    
    generate_prediction_visualizations(
        experiment_name=args.experiment_name,
        images_dir=args.images_dir,
        gt_labels_dir=args.gt_labels_dir,
        pred_labels_dir=args.pred_labels_dir,
        split=args.split,
        img_conf_th=args.img_conf_th,
        obj_conf_th=args.obj_conf_th,
        iou_th=args.iou_th
    )

if __name__ == "__main__":
    main()