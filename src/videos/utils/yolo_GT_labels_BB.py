#!/usr/bin/env python3
"""
yolo_GT_labels_BB.py
Draw ground truth bounding boxes from YOLO labels on wildfire dataset images.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# Configuration
COLORS = {
    0: (0, 0, 255),    # Red for wildfire smoke
    1: (255, 0, 0)     # Blue for no fire
}
LABELS = {
    0: "wildfire smoke",
    1: "no fire"
}
BOX_THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
FONT_THICKNESS = 2


def yolo_to_xyxy(x_center, y_center, width, height, img_width, img_height):
    """Convert YOLO normalized format to pixel coordinates."""
    x1 = int((x_center - width/2) * img_width)
    y1 = int((y_center - height/2) * img_height)
    x2 = int((x_center + width/2) * img_width)
    y2 = int((y_center + height/2) * img_height)
    return x1, y1, x2, y2


def process_image(args):
    """Process a single image with its labels."""
    img_path, label_path, output_path = args
    
    # Read image
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Warning: Could not read {img_path}")
        return False
    
    h, w = img.shape[:2]
    
    # Read labels if they exist
    if label_path.exists():
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        # Draw each bounding box
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                # Convert to pixel coordinates
                x1, y1, x2, y2 = yolo_to_xyxy(x_center, y_center, width, height, w, h)
                
                # Get color and label
                color = COLORS.get(class_id, (0, 255, 0))  # Green for unknown classes
                label = LABELS.get(class_id, f"class_{class_id}")
                
                # Draw bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), color, BOX_THICKNESS)
                
                # Draw label background
                label_size = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICKNESS)[0]
                label_y = y1 - 10 if y1 - 10 > label_size[1] else y1 + label_size[1] + 10
                cv2.rectangle(img, 
                            (x1, label_y - label_size[1] - 3),
                            (x1 + label_size[0] + 3, label_y + 3),
                            color, -1)
                
                # Draw label text
                cv2.putText(img, label, (x1 + 2, label_y), 
                           FONT, FONT_SCALE, (255, 255, 255), FONT_THICKNESS)
    
    # Save annotated image
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), img)
    return True


def main():
    parser = argparse.ArgumentParser(description='Draw GT bounding boxes on wildfire dataset')
    parser.add_argument('--input_dir', type=str, 
                       default='/vol/bitbucket/si324/rf-detr-wildfire/data/pyrodata/data',
                       help='Input dataset directory')
    parser.add_argument('--output_dir', type=str,
                       default='/vol/bitbucket/si324/rf-detr-wildfire/src/videos/bounding_boxes/ground_truth',
                       help='Output directory for annotated images')
    parser.add_argument('--workers', type=int, default=multiprocessing.cpu_count(),
                       help='Number of parallel workers')
    parser.add_argument('--splits', nargs='+', default=['train', 'val', 'test'],
                       help='Dataset splits to process')
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    # Collect all image-label pairs
    tasks = []
    
    for split in args.splits:
        split_dir = input_dir / split
        if not split_dir.exists():
            print(f"Warning: {split_dir} does not exist, skipping...")
            continue
        
        print(f"\nProcessing {split} split...")
        
        # Find all scene directories
        scene_dirs = [d for d in split_dir.iterdir() if d.is_dir()]
        
        for scene_dir in scene_dirs:
            # Find all jpg images
            img_files = list(scene_dir.glob('*.jpg'))
            
            for img_path in img_files:
                # Construct label path
                img_name = img_path.stem
                label_path = scene_dir / 'labels' / f'{img_name}.txt'
                
                # Construct output path (maintaining directory structure)
                relative_path = img_path.relative_to(input_dir)
                output_path = output_dir / relative_path
                
                tasks.append((img_path, label_path, output_path))
    
    print(f"\nTotal images to process: {len(tasks)}")
    
    # Process images in parallel
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        results = list(tqdm(
            executor.map(process_image, tasks),
            total=len(tasks),
            desc="Drawing bounding boxes"
        ))
    
    # Report results
    successful = sum(results)
    print(f"\nCompleted: {successful}/{len(tasks)} images processed successfully")
    print(f"Output saved to: {output_dir}")


if __name__ == '__main__':
    main()