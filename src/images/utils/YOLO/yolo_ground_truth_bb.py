"""
YOLO Ground Truth Bounding Box Visualizer
Generates ground truth bounding box visualizations from YOLO format annotations
"""

import os
import cv2
import numpy as np
from pathlib import Path
import argparse
from typing import List, Tuple
import json
from datetime import datetime

def xywh_to_xyxy_absolute(x_center, y_center, width, height, img_width, img_height):
    """
    Convert YOLO format (normalized x_center, y_center, width, height) to absolute coordinates
    """
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
    """
    Load YOLO format annotations from a text file
    Returns list of (class_id, x_center, y_center, width, height)
    """
    annotations = []
    if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
        with open(label_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    annotations.append((class_id, x_center, y_center, width, height))
    return annotations

def draw_ground_truth_boxes(image_path: str, label_path: str, output_path: str, 
                           class_names: dict = None) -> dict:
    """
    Draw ground truth bounding boxes on image and save result
    Returns stats about the image
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Could not load image: {image_path}")
        return None
    
    img_height, img_width = image.shape[:2]
    
    # Load annotations
    annotations = load_yolo_annotations(label_path)
    
    # Create output image (copy original)
    output_image = image.copy()
    
    # Class names (default to 'smoke' for class 0)
    if class_names is None:
        class_names = {0: 'smoke'}
    
    # Colors for different classes (BGR format)
    class_colors = {
        0: (0, 255, 0),    # Green for smoke
        1: (255, 0, 0),    # Blue for fire
        2: (0, 0, 255),    # Red for other
    }
    
    # Draw bounding boxes
    box_count = 0
    for class_id, x_center, y_center, width, height in annotations:
        # Convert to absolute coordinates
        x1, y1, x2, y2 = xywh_to_xyxy_absolute(x_center, y_center, width, height, 
                                                img_width, img_height)
        
        # Get color and class name
        color = class_colors.get(class_id, (128, 128, 128))  # Gray for unknown classes
        class_name = class_names.get(class_id, f'class_{class_id}')
        
        # Draw bounding box
        cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 2)
        
        # Draw class label with background
        label = f'GT: {class_name}'
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 1
        
        # Get text size for background rectangle
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
        
        # Draw background rectangle for text
        cv2.rectangle(output_image, 
                     (x1, y1 - text_height - baseline - 5), 
                     (x1 + text_width + 5, y1), 
                     color, -1)
        
        # Draw text
        cv2.putText(output_image, label, 
                   (x1 + 2, y1 - baseline - 2), 
                   font, font_scale, (255, 255, 255), font_thickness)
        
        box_count += 1
    
    # Add image info text
    info_text = f"GT Boxes: {box_count}"
    cv2.putText(output_image, info_text, 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(output_image, info_text, 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
    
    # Save output image
    cv2.imwrite(output_path, output_image)
    
    # Return stats
    return {
        'image_path': image_path,
        'label_path': label_path,
        'output_path': output_path,
        'num_boxes': box_count,
        'image_size': (img_width, img_height),
        'has_annotations': box_count > 0
    }

def generate_ground_truth_visualizations(experiment_name: str, 
                                        images_dir: str,
                                        labels_dir: str,
                                        split: str = 'test',
                                        class_names: dict = None):
    """
    Generate ground truth bounding box visualizations for all images in a split
    """
    print(f"🎯 Generating Ground Truth Visualizations for {experiment_name}")
    print(f"📁 Images: {images_dir}")
    print(f"🏷️  Labels: {labels_dir}")
    
    # Create output directory
    output_root = f"bounding_boxes/{experiment_name}/yolo_ground_truth_bb"
    output_dir = os.path.join(output_root, split)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"💾 Output: {output_dir}")
    
    # Get all images in the split
    split_images_dir = os.path.join(images_dir, split)
    split_labels_dir = os.path.join(labels_dir, split)
    
    if not os.path.exists(split_images_dir):
        print(f"❌ Images directory not found: {split_images_dir}")
        return
    
    if not os.path.exists(split_labels_dir):
        print(f"❌ Labels directory not found: {split_labels_dir}")
        return
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(split_images_dir).glob(f'*{ext}'))
        image_files.extend(Path(split_images_dir).glob(f'*{ext.upper()}'))
    
    print(f"📊 Found {len(image_files)} images to process")
    
    # Process each image
    stats = []
    processed = 0
    with_annotations = 0
    without_annotations = 0
    
    for image_file in sorted(image_files):
        # Get corresponding label file
        image_name = image_file.stem
        label_file = os.path.join(split_labels_dir, f"{image_name}.txt")
        
        # Generate output path
        output_file = os.path.join(output_dir, f"{image_name}_gt.jpg")
        
        # Process image
        result = draw_ground_truth_boxes(str(image_file), label_file, output_file, class_names)
        
        if result:
            stats.append(result)
            processed += 1
            
            if result['has_annotations']:
                with_annotations += 1
            else:
                without_annotations += 1
            
            if processed % 50 == 0:
                print(f"⏳ Processed {processed}/{len(image_files)} images...")
    
    # Save statistics
    summary_stats = {
        'experiment_name': experiment_name,
        'split': split,
        'generation_timestamp': datetime.now().isoformat(),
        'total_images': len(image_files),
        'processed_images': processed,
        'images_with_annotations': with_annotations,
        'images_without_annotations': without_annotations,
        'images_dir': images_dir,
        'labels_dir': labels_dir,
        'output_dir': output_dir,
        'detailed_stats': stats
    }
    
    stats_file = os.path.join(output_root, f'{split}_gt_stats.json')
    with open(stats_file, 'w') as f:
        json.dump(summary_stats, f, indent=2)
    
    print(f"\n✅ Ground Truth Visualization Complete!")
    print(f"📊 Total Images: {len(image_files)}")
    print(f"✅ Processed: {processed}")
    print(f"🎯 With Annotations: {with_annotations}")
    print(f"🔘 Without Annotations: {without_annotations}")
    print(f"💾 Output Directory: {output_dir}")
    print(f"📈 Stats saved to: {stats_file}")

def main():
    parser = argparse.ArgumentParser(description='Generate YOLO Ground Truth Bounding Box Visualizations')
    parser.add_argument('--experiment_name', type=str, required=True,
                        help='Name of the experiment (creates subfolder)')
    parser.add_argument('--images_dir', type=str, required=True,
                        help='Path to images directory (should contain train/valid/test subdirs)')
    parser.add_argument('--labels_dir', type=str, required=True,
                        help='Path to labels directory (should contain train/valid/test subdirs)')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'valid', 'test'],
                        help='Which split to process (default: test)')
    parser.add_argument('--class_names', type=str, nargs='+', default=['smoke'],
                        help='Class names in order (default: smoke)')
    
    args = parser.parse_args()
    
    # Convert class names list to dict
    class_names_dict = {i: name for i, name in enumerate(args.class_names)}
    
    # Generate visualizations
    generate_ground_truth_visualizations(
        experiment_name=args.experiment_name,
        images_dir=args.images_dir,
        labels_dir=args.labels_dir,
        split=args.split,
        class_names=class_names_dict
    )

if __name__ == "__main__":
    main()