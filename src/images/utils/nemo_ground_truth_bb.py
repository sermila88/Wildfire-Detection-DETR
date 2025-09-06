import os
import cv2
import json
import numpy as np
from pathlib import Path
import shutil
from tqdm import tqdm
import glob

# Configuration
NEMO_IMAGES_DIR = "/vol/bitbucket/si324/rf-detr-wildfire/src/images/data/nemo_val/img"
NEMO_ANNOTATIONS_DIR = "/vol/bitbucket/si324/rf-detr-wildfire/src/images/data/nemo_val/ann"
OUTPUT_BASE_DIR = "/vol/bitbucket/si324/rf-detr-wildfire/src/images/Nemo_ground_truth_BB"

# Color mapping for smoke densities (BGR format for OpenCV)
COLORS = {
    'high': (0, 0, 255),      # Red
    'mid': (128, 0, 128),     # Purple
    'low': (0, 165, 255)      # Orange
}

def create_output_dirs():
    """Create output directory structure"""
    dirs = [
        f"{OUTPUT_BASE_DIR}/ALL_NEMO",
        f"{OUTPUT_BASE_DIR}/Breakdown/high",
        f"{OUTPUT_BASE_DIR}/Breakdown/mid",
        f"{OUTPUT_BASE_DIR}/Breakdown/low",
        f"{OUTPUT_BASE_DIR}/Breakdown/mixed"
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    return dirs

def read_nemo_json_annotations(json_path):
    """Read NEMO JSON format annotations"""
    boxes = []
    
    if not os.path.exists(json_path):
        return boxes, None
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Get image dimensions
    img_dims = data.get('size', {})
    
    # Process each object
    for obj in data.get('objects', []):
        class_title = obj.get('classTitle', '').lower()
        
        # Determine density from class title
        if 'high' in class_title:
            density = 'high'
        elif 'mid' in class_title:
            density = 'mid'
        elif 'low' in class_title:
            density = 'low'
        else:
            # Default to low if unknown
            density = 'low'
        
        # Get bounding box coordinates
        points = obj.get('points', {}).get('exterior', [])
        if len(points) == 2:
            x1, y1 = points[0]
            x2, y2 = points[1]
            
            boxes.append({
                'density': density,
                'x1': min(x1, x2),
                'y1': min(y1, y2),
                'x2': max(x1, x2),
                'y2': max(y1, y2),
                'class_title': class_title
            })
    
    return boxes, img_dims

def draw_boxes(image, boxes):
    """Draw bounding boxes on image with appropriate colors"""
    annotated_img = image.copy()
    
    for box in boxes:
        # Get color based on density
        color = COLORS[box['density']]
        
        # Draw rectangle
        cv2.rectangle(annotated_img, 
                     (box['x1'], box['y1']), 
                     (box['x2'], box['y2']), 
                     color, 2)
        
        # Add label with background for visibility
        label = box['density'].upper()
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        
        # Draw label background
        cv2.rectangle(annotated_img,
                     (box['x1'], box['y1'] - label_size[1] - 5),
                     (box['x1'] + label_size[0], box['y1']),
                     color, -1)
        
        # Draw label text
        cv2.putText(annotated_img, label, 
                   (box['x1'], box['y1'] - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return annotated_img

def get_density_category(boxes):
    """Determine if image has single density type or mixed"""
    densities = set([box['density'] for box in boxes])
    
    if len(densities) == 0:
        return None
    elif len(densities) == 1:
        return list(densities)[0]
    else:
        return 'mixed'

def process_nemo_dataset():
    """Process all NEMO images and create visualizations"""
    
    # Create output directories
    create_output_dirs()
    
    # Get all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(NEMO_IMAGES_DIR, ext)))
    
    print(f"Found {len(image_files)} images to process")
    
    # Statistics
    stats = {'high': 0, 'mid': 0, 'low': 0, 'mixed': 0, 'no_annotations': 0}
    
    # Process each image
    for img_path in tqdm(image_files, desc="Processing images"):
        img_filename = os.path.basename(img_path)  # Get full filename with extension
        img_name = Path(img_path).stem
        
        # IMPORTANT: The JSON file is named as image_name.jpg.json
        json_path = os.path.join(NEMO_ANNOTATIONS_DIR, f"{img_filename}.json")
        
        # Read image
        image = cv2.imread(img_path)
        if image is None:
            print(f"Could not read image: {img_path}")
            continue
        
        # Read annotations
        boxes, img_dims = read_nemo_json_annotations(json_path)
        
        if not boxes:
            stats['no_annotations'] += 1
            # Skip images without annotations
            continue
        
        # Create visualization
        img_with_boxes = draw_boxes(image, boxes)
        
        # Save to ALL_NEMO
        all_output_path = f"{OUTPUT_BASE_DIR}/ALL_NEMO/{img_name}_annotated.jpg"
        cv2.imwrite(all_output_path, img_with_boxes)
        
        # Determine category and save to breakdown
        category = get_density_category(boxes)
        if category:
            breakdown_output_path = f"{OUTPUT_BASE_DIR}/Breakdown/{category}/{img_name}_annotated.jpg"
            cv2.imwrite(breakdown_output_path, img_with_boxes)
            stats[category] += 1
    
    print("\nProcessing complete!")
    print(f"Output saved to: {OUTPUT_BASE_DIR}")
    
    # Print statistics
    print("\nDataset Statistics:")
    for category in ['high', 'mid', 'low', 'mixed']:
        print(f"  {category}: {stats[category]} images")
    print(f"  No annotations: {stats['no_annotations']} images")
    print(f"  Total processed: {sum([stats['high'], stats['mid'], stats['low'], stats['mixed']])} images")

def create_legend():
    """Create a legend image showing color coding"""
    legend = np.ones((150, 350, 3), dtype=np.uint8) * 255
    
    # Add title
    cv2.putText(legend, "NEMO Smoke Density Color Legend", 
               (10, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    y_offset = 50
    for density, color in COLORS.items():
        # Draw colored box
        cv2.rectangle(legend, (20, y_offset), (50, y_offset + 25), color, -1)
        cv2.rectangle(legend, (20, y_offset), (50, y_offset + 25), (0, 0, 0), 2)
        
        # Add text
        text = f"{density.upper()} density smoke"
        cv2.putText(legend, text, 
                   (60, y_offset + 18),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        y_offset += 35
    
    cv2.imwrite(f"{OUTPUT_BASE_DIR}/color_legend.jpg", legend)
    print("Color legend saved")

if __name__ == "__main__":
    process_nemo_dataset()
    create_legend()