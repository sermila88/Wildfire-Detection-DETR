"""
BOUNDING_BOX_GT.py - Final Data Inspection and Visualization for Video Dataset
Inspects YOLO annotations, draws bounding boxes, and organizes by class
"""

import os
import cv2
import glob
import shutil
from collections import defaultdict
from datetime import datetime
import numpy as np

# Configuration
BASE_PATH = "/vol/bitbucket/si324/rf-detr-wildfire/src/videos/data"
OUTPUT_PATH = "/vol/bitbucket/si324/rf-detr-wildfire/src/videos/FINAL_GT_BREAKDOWN"
SPLITS = ["train", "val", "test"]

# Colors for bounding boxes (BGR format for OpenCV)
COLORS = {
    0: (0, 0, 255),    # RED for class 0 (smoke)
    1: (128, 0, 128)   # PURPLE for class 1 (if exists)
}

def parse_yolo_label(label_path, img_width, img_height):
    """Parse YOLO format label file and convert to pixel coordinates."""
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    
    try:
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1]) * img_width
                    y_center = float(parts[2]) * img_height
                    width = float(parts[3]) * img_width
                    height = float(parts[4]) * img_height
                    
                    x1 = int(x_center - width/2)
                    y1 = int(y_center - height/2)
                    x2 = int(x_center + width/2)
                    y2 = int(y_center + height/2)
                    
                    boxes.append((class_id, x1, y1, x2, y2))
    except Exception as e:
        print(f"Error parsing {label_path}: {e}")
    
    return boxes

def draw_boxes_on_frame(img_path, label_path, output_path):
    """Draw bounding boxes on a single frame."""
    img = cv2.imread(img_path)
    if img is None:
        return None
    
    height, width = img.shape[:2]
    boxes = parse_yolo_label(label_path, width, height)
    
    classes_found = set()
    for class_id, x1, y1, x2, y2 in boxes:
        classes_found.add(class_id)
        color = COLORS.get(class_id, (255, 255, 255))  # White for unknown classes
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Add class label
        label_text = f"Class {class_id}"
        cv2.putText(img, label_text, (x1, y1-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, img)
    
    return classes_found

def process_video_folder(video_path, output_base):
    """Process a single video folder and return class information."""
    video_name = os.path.basename(video_path)
    frames = sorted(glob.glob(os.path.join(video_path, "*.jpg")))
    
    if not frames:
        return None, 0
    
    all_classes = set()
    frames_with_annotations = 0
    
    # Sample frames to visualize (first, middle, last with annotations)
    frames_to_viz = []
    
    for frame_path in frames:
        frame_name = os.path.basename(frame_path)
        label_path = os.path.join(video_path, "labels", 
                                 frame_name.replace(".jpg", ".txt"))
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                content = f.read().strip()
                if content:
                    frames_with_annotations += 1
                    # Get first character (class) from each line
                    for line in content.split('\n'):
                        if line.strip():
                            all_classes.add(int(line[0]))
                    
                    # Add to visualization list (limit to 3 frames per video)
                    frames_to_viz.append((frame_path, label_path))
    
    # Visualize selected frames if there are annotations
    if all_classes and frames_to_viz:
        # Determine output folder based on classes found
        if 1 in all_classes:
            class_folder = "1"
        elif 0 in all_classes:
            class_folder = "0"
        else:
            class_folder = "unknown"
        
        # Draw boxes on selected frames
        for frame_path, label_path in frames_to_viz:
            frame_name = os.path.basename(frame_path)
            output_path = os.path.join(output_base, class_folder, 
                                      video_name, frame_name)
            draw_boxes_on_frame(frame_path, label_path, output_path)
    
    return all_classes, frames_with_annotations

def main():
    """Main processing function."""
    print("=" * 60)
    print("FINAL GT BREAKDOWN - Video Dataset Inspection")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    # Statistics storage
    stats = defaultdict(lambda: defaultdict(list))
    
    # Process each split
    for split in SPLITS:
        print(f"\nProcessing {split}...")
        split_path = os.path.join(BASE_PATH, split)
        
        if not os.path.exists(split_path):
            print(f"  {split} folder not found, skipping...")
            continue
        
        output_split = os.path.join(OUTPUT_PATH, split)
        os.makedirs(output_split, exist_ok=True)
        
        # Create class folders
        for class_id in ["0", "1"]:
            os.makedirs(os.path.join(output_split, class_id), exist_ok=True)
        
        # Process each video folder
        video_folders = [d for d in glob.glob(os.path.join(split_path, "*")) 
                        if os.path.isdir(d)]
        
        for video_path in video_folders:
            video_name = os.path.basename(video_path)
            classes_found, frames_annotated = process_video_folder(video_path, output_split)
            
            if classes_found:
                if 1 in classes_found:
                    stats[split]["class_1"].append(video_name)
                if 0 in classes_found:
                    stats[split]["class_0"].append(video_name)
                
                print(f"  {video_name}: Classes {sorted(classes_found)}, "
                     f"{frames_annotated} frames annotated")
    
    # Generate summary report
    report_path = os.path.join(OUTPUT_PATH, "summary_report.txt")
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("FINAL GT BREAKDOWN - SUMMARY REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        
        # Overall statistics
        f.write("OVERALL STATISTICS\n")
        f.write("-" * 40 + "\n")
        
        total_class_0 = 0
        total_class_1 = 0
        
        for split in SPLITS:
            class_0_count = len(stats[split]["class_0"])
            class_1_count = len(stats[split]["class_1"])
            total_videos = len(set(stats[split]["class_0"] + stats[split]["class_1"]))
            
            f.write(f"\n{split.upper()}:\n")
            f.write(f"  Total videos with annotations: {total_videos}\n")
            f.write(f"  Videos with class 0 (smoke): {class_0_count}\n")
            f.write(f"  Videos with class 1: {class_1_count}\n")
            
            total_class_0 += class_0_count
            total_class_1 += class_1_count
        
        f.write(f"\nTOTAL ACROSS ALL SPLITS:\n")
        f.write(f"  Videos with class 0: {total_class_0}\n")
        f.write(f"  Videos with class 1: {total_class_1}\n")
        
        # Detailed folder listings
        f.write("\n" + "=" * 80 + "\n")
        f.write("DETAILED FOLDER LISTINGS\n")
        f.write("=" * 80 + "\n")
        
        for split in SPLITS:
            f.write(f"\n{split.upper()} SPLIT:\n")
            f.write("-" * 40 + "\n")
            
            # Class 0 folders
            if stats[split]["class_0"]:
                f.write(f"\nFolders with Class 0 (smoke) [{len(stats[split]['class_0'])} folders]:\n")
                for folder in sorted(stats[split]["class_0"]):
                    f.write(f"  - {folder}\n")
            
            # Class 1 folders
            if stats[split]["class_1"]:
                f.write(f"\nFolders with Class 1 [{len(stats[split]['class_1'])} folders]:\n")
                for folder in sorted(stats[split]["class_1"]):
                    f.write(f"  - {folder}\n")
    
    print("\n" + "=" * 60)
    print(f"Processing complete!")
    print(f"Output saved to: {OUTPUT_PATH}")
    print(f"Summary report: {report_path}")
    print("=" * 60)

    # Print summary to console
    print("\nQUICK SUMMARY:")
    for split in SPLITS:
        print(f"{split}: {len(stats[split]['class_0'])} class_0, "
              f"{len(stats[split]['class_1'])} class_1")

if __name__ == "__main__":
    main()