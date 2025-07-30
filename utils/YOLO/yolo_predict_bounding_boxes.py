"""
YOLO Predicted Bounding Box Visualizer with TP/FP/FN/TN Breakdown
Generates predicted bounding box visualizations categorized by performance
"""

import os
import cv2
import numpy as np
from pathlib import Path
import argparse
from typing import List, Tuple, Dict
import json
from datetime import datetime

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

def xywh2xyxy(box):
    """Convert from xywh to xyxy format for IoU calculation"""
    x, y, w, h = box
    return np.array([x - w/2, y - w/2, x + w/2, y + h/2])

def box_iou(box1, box2):
    """Calculate IoU between two boxes"""
    # Get intersection coordinates
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    # Calculate intersection and union areas
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def load_yolo_annotations(label_path: str) -> List[Tuple[int, float, float, float, float]]:
    """Load YOLO format annotations"""
    annotations = []
    if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
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

def evaluate_image_predictions(gt_boxes, pred_boxes, conf_th=0.1, iou_th=0.1):
    """
    Evaluate predictions for a single image
    Returns object-level and image-level classifications
    """
    # Filter predictions by confidence
    valid_preds = [box for box in pred_boxes if box[5] >= conf_th]
    
    # Convert to xyxy format for IoU calculation
    gt_xyxy = [xywh2xyxy(np.array(box[1:5])) for box in gt_boxes]
    pred_xyxy = [xywh2xyxy(np.array(box[1:5])) for box in valid_preds]
    
    # Object-level evaluation
    gt_matches = np.zeros(len(gt_boxes), dtype=bool)
    obj_classifications = []
    
    for i, pred_box in enumerate(valid_preds):
        pred_xyxy_box = pred_xyxy[i]
        confidence = pred_box[5]
        
        if gt_xyxy:
            # Find best IoU match
            iou_values = [box_iou(pred_xyxy_box, gt_box) for gt_box in gt_xyxy]
            max_iou = max(iou_values)
            best_match_idx = np.argmax(iou_values)
            
            if max_iou > iou_th and not gt_matches[best_match_idx]:
                obj_classifications.append(('TP', i, best_match_idx, confidence, max_iou))
                gt_matches[best_match_idx] = True
            else:
                obj_classifications.append(('FP', i, -1, confidence, max_iou))
        else:
            obj_classifications.append(('FP', i, -1, confidence, 0.0))
    
    # Count FN (unmatched GT boxes)
    for i, matched in enumerate(gt_matches):
        if not matched:
            obj_classifications.append(('FN', -1, i, 0.0, 0.0))
    
    # Image-level evaluation
    has_smoke_gt = len(gt_boxes) > 0
    has_smoke_pred = len(valid_preds) > 0
    spatial_match = np.sum(gt_matches) > 0 if has_smoke_gt and has_smoke_pred else False
    
    if has_smoke_gt:
        img_class = 'TP' if spatial_match else 'FN'
    else:
        img_class = 'FP' if has_smoke_pred else 'TN'
    
    return obj_classifications, img_class

def draw_predictions_with_classification(image_path: str, gt_path: str, pred_path: str, 
                                       output_path: str, conf_th: float = 0.1,
                                       obj_classifications: List = None, 
                                       img_classification: str = '') -> Dict:
    """Draw predictions with TP/FP/FN classifications"""
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    img_height, img_width = image.shape[:2]
    output_image = image.copy()
    
    # Load annotations
    gt_boxes = load_yolo_annotations(gt_path)
    pred_boxes = load_yolo_annotations(pred_path)
    
    # Colors for different classifications
    colors = {
        'TP': (0, 255, 0),    # Green - True Positive
        'FP': (0, 0, 255),    # Red - False Positive  
        'FN': (255, 0, 0),    # Blue - False Negative
        'GT': (0, 255, 255),  # Yellow - Ground Truth
    }
    
    # Draw ground truth boxes first (in yellow)
    for i, (class_id, x_center, y_center, width, height, _) in enumerate(gt_boxes):
        x1, y1, x2, y2 = xywh_to_xyxy_absolute(x_center, y_center, width, height, img_width, img_height)
        cv2.rectangle(output_image, (x1, y1), (x2, y2), colors['GT'], 2)
        cv2.putText(output_image, f'GT_{i}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['GT'], 1)
    
    # Draw predictions with classifications
    if obj_classifications:
        for classification, pred_idx, gt_idx, confidence, iou in obj_classifications:
            if pred_idx >= 0:  # Valid prediction index
                pred_box = pred_boxes[pred_idx]
                x_center, y_center, width, height = pred_box[1:5]
                x1, y1, x2, y2 = xywh_to_xyxy_absolute(x_center, y_center, width, height, img_width, img_height)
                
                color = colors.get(classification, (128, 128, 128))
                cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 2)
                
                # Label with classification and confidence
                label = f'{classification}: {confidence:.2f}'
                if iou > 0:
                    label += f' IoU:{iou:.2f}'
                
                cv2.putText(output_image, label, (x1, y2+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            elif classification == 'FN' and gt_idx >= 0:
                # Draw FN marker on unmatched GT box
                gt_box = gt_boxes[gt_idx]
                x_center, y_center, width, height = gt_box[1:5]
                x1, y1, x2, y2 = xywh_to_xyxy_absolute(x_center, y_center, width, height, img_width, img_height)
                
                # Draw FN marker
                cv2.circle(output_image, (x1-10, y1-10), 8, colors['FN'], -1)
                cv2.putText(output_image, 'FN', (x1-25, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors['FN'], 1)
    
    # Add image classification and stats
    stats_text = f"Image: {img_classification} | GT: {len(gt_boxes)} | Pred: {len([p for p in pred_boxes if p[5] >= conf_th])}"
    cv2.putText(output_image, stats_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(output_image, stats_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    
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
        'num_pred_boxes': len([p for p in pred_boxes if p[5] >= conf_th])
    }

def generate_prediction_visualizations(experiment_name: str, 
                                     images_dir: str,
                                     gt_labels_dir: str,
                                     pred_labels_dir: str,
                                     split: str = 'test',
                                     conf_th: float = 0.1):
    """Generate prediction visualizations categorized by TP/FP/FN/TN"""
    
    print(f"🎯 Generating Prediction Visualizations for {experiment_name}")
    print(f"📁 Images: {images_dir}")
    print(f"🏷️  GT Labels: {gt_labels_dir}")
    print(f"🔮 Pred Labels: {pred_labels_dir}")
    print(f"📊 Split: {split}")
    print(f"🎚️  Confidence Threshold: {conf_th}")
    
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
    
    # Get all images
    split_images_dir = os.path.join(images_dir, split)
    split_gt_dir = os.path.join(gt_labels_dir, split)
    
    if not os.path.exists(split_images_dir):
        print(f"❌ Images directory not found: {split_images_dir}")
        return
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(split_images_dir).glob(f'*{ext}'))
        image_files.extend(Path(split_images_dir).glob(f'*{ext.upper()}'))
    
    print(f"📊 Found {len(image_files)} images to process")
    
    # Statistics
    stats = {
        'image_level': {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0},
        'object_level': {'TP': 0, 'FP': 0, 'FN': 0},
        'processed_images': 0,
        'detailed_results': []
    }
    
    # Process each image
    for image_file in sorted(image_files):
        image_name = image_file.stem
        gt_file = os.path.join(split_gt_dir, f"{image_name}.txt")
        pred_file = os.path.join(pred_labels_dir, f"{image_name}.txt")
        
        # Skip if no prediction file
        if not os.path.exists(pred_file):
            continue
        
        # Load annotations
        gt_boxes = load_yolo_annotations(gt_file)
        pred_boxes = load_yolo_annotations(pred_file)
        
        # Evaluate predictions
        obj_classifications, img_classification = evaluate_image_predictions(
            gt_boxes, pred_boxes, conf_th
        )
        
        # Generate visualizations for image-level
        img_output_path = os.path.join(img_level_dirs[img_classification], f"{image_name}_img.jpg")
        
        result = draw_predictions_with_classification(
            str(image_file), gt_file, pred_file, img_output_path,
            conf_th, obj_classifications, img_classification
        )
        
        if result:
            stats['detailed_results'].append(result)
            stats['processed_images'] += 1
            stats['image_level'][img_classification] += 1
            
            # Count object-level classifications
            for obj_class, _, _, _, _ in obj_classifications:
                if obj_class in stats['object_level']:
                    stats['object_level'][obj_class] += 1
            
            # Also save to object-level directories if it has predictions/GT
            if img_classification in ['TP', 'FP', 'FN']:  # Skip TN for object-level
                obj_output_path = os.path.join(obj_level_dirs[img_classification], f"{image_name}_obj.jpg")
                draw_predictions_with_classification(
                    str(image_file), gt_file, pred_file, obj_output_path,
                    conf_th, obj_classifications, img_classification
                )
        
        if stats['processed_images'] % 50 == 0:
            print(f"⏳ Processed {stats['processed_images']}/{len(image_files)} images...")
    
    # Save statistics (convert numpy types to native Python types for JSON serialization)
    def convert_numpy_types(obj):
        """Convert numpy types to native Python types for JSON serialization"""
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
        'confidence_threshold': float(conf_th),
        'generation_timestamp': datetime.now().isoformat(),
        'total_images_found': int(len(image_files)),
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
    
    print(f"\n✅ Prediction Visualization Complete!")
    print(f"📊 Total Images: {len(image_files)}")
    print(f"✅ Processed: {stats['processed_images']}")
    print(f"\n📈 Image-Level Results:")
    for classification, count in stats['image_level'].items():
        print(f"   {classification}: {count}")
    print(f"\n📈 Object-Level Results:")
    for classification, count in stats['object_level'].items():
        print(f"   {classification}: {count}")
    print(f"\n💾 Output Directories:")
    print(f"   📁 Image-level: {output_root}/image_level/")
    print(f"   📁 Object-level: {output_root}/object_level/")
    print(f"📈 Stats saved to: {stats_file}")

def main():
    parser = argparse.ArgumentParser(description='Generate YOLO Prediction Bounding Box Visualizations with TP/FP/FN/TN Breakdown')
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
    parser.add_argument('--conf_th', type=float, default=0.1,
                        help='Confidence threshold for predictions')
    
    args = parser.parse_args()
    
    generate_prediction_visualizations(
        experiment_name=args.experiment_name,
        images_dir=args.images_dir,
        gt_labels_dir=args.gt_labels_dir,
        pred_labels_dir=args.pred_labels_dir,
        split=args.split,
        conf_th=args.conf_th
    )

if __name__ == "__main__":
    main()