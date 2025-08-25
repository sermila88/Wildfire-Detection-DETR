#!/usr/bin/env python3
"""
YOLO Ground Truth Bounding Box Visualizer
Script to visualize YOLO format ground truth bounding boxes on test images
"""

import os
import cv2
import shutil
from pathlib import Path

# Configuration
BASE_PATH = "/vol/bitbucket/si324/rf-detr-wildfire/src/images"
INPUT_IMAGES = f"{BASE_PATH}/data/pyro25img/images/test"
INPUT_LABELS = f"{BASE_PATH}/data/pyro25img/labels/test"
OUTPUT_BASE = f"{BASE_PATH}/eval_results/Ground_truth_BB"

# Output directories
OUTPUT_ALL = f"{OUTPUT_BASE}/Ground_truth_bounding_boxes"
OUTPUT_SMOKE = f"{OUTPUT_BASE}/Ground_truth_bounding_boxes_breakdown/GT_BB_smoke"
OUTPUT_NO_SMOKE = f"{OUTPUT_BASE}/Ground_truth_bounding_boxes_breakdown/GT_BB_no_smoke"

# Visualization settings
BOX_COLOR = (0, 255, 0)  # Green in BGR
BOX_THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
FONT_COLOR = (255, 255, 255)  # White
FONT_THICKNESS = 2


def create_output_directories():
    """Create all necessary output directories"""
    for directory in [OUTPUT_ALL, OUTPUT_SMOKE, OUTPUT_NO_SMOKE]:
        os.makedirs(directory, exist_ok=True)
    print(f" Created output directories")


def load_yolo_annotations(label_path):
    """
    Load YOLO annotations from a text file
    Returns list of [class_id, x_center, y_center, width, height] (normalized)
    """
    annotations = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    annotations.append([float(x) for x in parts])
    return annotations


def draw_bounding_boxes(image, annotations):
    """
    Draw bounding boxes on image
    Args:
        image: OpenCV image
        annotations: List of [class_id, x_center, y_center, width, height] (normalized)
    Returns:
        Image with drawn bounding boxes
    """
    img_height, img_width = image.shape[:2]
    output_image = image.copy()
    
    # If no annotations, add "GT: no smoke" text
    if not annotations:
        text = "GT: no smoke"
        # Add background rectangle for better visibility
        cv2.rectangle(output_image, (10, 10), (200, 40), (0, 0, 0), -1)
        cv2.putText(output_image, text, (15, 30), 
                   FONT, FONT_SCALE, FONT_COLOR, FONT_THICKNESS)
        return output_image
    
    # Draw bounding boxes for annotations
    for ann in annotations:
        class_id, x_center, y_center, width, height = ann
        
        # Convert normalized coordinates to pixel coordinates
        x_center_px = x_center * img_width
        y_center_px = y_center * img_height
        width_px = width * img_width
        height_px = height * img_height
        
        # Calculate corner points
        x1 = int(x_center_px - width_px / 2)
        y1 = int(y_center_px - height_px / 2)
        x2 = int(x_center_px + width_px / 2)
        y2 = int(y_center_px + height_px / 2)
        
        # Draw rectangle
        cv2.rectangle(output_image, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS)
        
        # Add label with "GT:" prefix
        label = "GT: smoke" if int(class_id) == 0 else f"GT: class_{int(class_id)}"
        label_y = max(y1 - 5, 20)  # Ensure label is visible
        cv2.putText(output_image, label, (x1, label_y), 
                   FONT, FONT_SCALE, FONT_COLOR, FONT_THICKNESS)
    
    return output_image


def process_test_images():
    """Process all test images and generate visualizations"""
    # Get all image files
    image_files = list(Path(INPUT_IMAGES).glob("*.jpg"))
    image_files.extend(Path(INPUT_IMAGES).glob("*.png"))
    
    print(f" Found {len(image_files)} test images")
    
    # Statistics
    total_processed = 0
    images_with_smoke = 0
    images_without_smoke = 0
    
    # Process each image
    for img_path in sorted(image_files):
        # Get corresponding label file
        label_name = img_path.stem + ".txt"
        label_path = Path(INPUT_LABELS) / label_name
        
        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            print(f" Could not load: {img_path.name}")
            continue
        
        # Load annotations
        annotations = load_yolo_annotations(label_path)
        
        # Draw bounding boxes
        output_image = draw_bounding_boxes(image, annotations)
        
        # Save to main output folder
        output_filename = f"{img_path.stem}_gt.jpg"
        cv2.imwrite(f"{OUTPUT_ALL}/{output_filename}", output_image)
        
        # Save to breakdown folders based on presence of smoke
        if annotations:  # Has annotations (smoke detected)
            cv2.imwrite(f"{OUTPUT_SMOKE}/{output_filename}", output_image)
            images_with_smoke += 1
        else:  # No annotations (no smoke)
            cv2.imwrite(f"{OUTPUT_NO_SMOKE}/{output_filename}", output_image)
            images_without_smoke += 1
        
        total_processed += 1
        
        # Progress update
        if total_processed % 100 == 0:
            print(f"⏳ Processed {total_processed}/{len(image_files)} images...")
    
    # Final statistics
    print("\n" + "="*50)
    print(" PROCESSING COMPLETE")
    print("="*50)
    print(f" Total images processed: {total_processed}")
    print(f" Images with smoke: {images_with_smoke}")
    print(f" Images without smoke: {images_without_smoke}")
    print(f"\n Output locations:")
    print(f"   All images: {OUTPUT_ALL}")
    print(f"   With smoke: {OUTPUT_SMOKE}")
    print(f"   Without smoke: {OUTPUT_NO_SMOKE}")


def main():
    """Main execution function"""
    print("🎯 YOLO Ground Truth Bounding Box Visualizer")
    print("="*50)
    
    # Create output directories
    create_output_directories()
    
    # Process all test images
    process_test_images()
    
    print("\n✨ Done!")


if __name__ == "__main__":
    main()