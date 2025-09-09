import os
import shutil
from pathlib import Path
import cv2
import numpy as np

def draw_yolo_boxes(img_path, label_path):
    """Draw YOLO format bounding boxes on image"""
    # Read image
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    
    h, w = img.shape[:2]
    
    # Read label file and draw boxes
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                # Parse YOLO format: class_id x_center y_center width height
                x_center = float(parts[1]) * w
                y_center = float(parts[2]) * h
                box_width = float(parts[3]) * w
                box_height = float(parts[4]) * h
                
                # Convert to corner coordinates
                x1 = int(x_center - box_width / 2)
                y1 = int(y_center - box_height / 2)
                x2 = int(x_center + box_width / 2)
                y2 = int(y_center + box_height / 2)
                
                # Draw rectangle (red color, thickness 2)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                # Add label
                cv2.putText(img, 'smoke', (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return img

def count_and_organize_dataset(dataset_name, images_path, labels_path, output_base):
    """Count smoke/no-smoke images and organize them into folders with GT boxes drawn"""
    
    # Create output directories
    smoke_dir = output_base / "smoke"
    no_smoke_dir = output_base / "no_smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    no_smoke_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all images
    image_files = list(Path(images_path).glob("**/*.jpg")) + list(Path(images_path).glob("**/*.png"))
    
    smoke_count = 0
    no_smoke_count = 0
    
    for img_path in image_files:
        # Get corresponding label file
        img_name = img_path.stem
        label_file = Path(labels_path) / "test" / f"{img_name}.txt"
        
        # Check if label file exists and is not empty
        has_smoke = False
        if label_file.exists():
            with open(label_file, 'r') as f:
                content = f.read().strip()
                if content:  # File has content = has smoke
                    has_smoke = True
        
        if has_smoke:
            smoke_count += 1
            # Draw bounding boxes and save
            img_with_boxes = draw_yolo_boxes(img_path, label_file)
            if img_with_boxes is not None:
                output_path = smoke_dir / img_path.name
                cv2.imwrite(str(output_path), img_with_boxes)
            else:
                # If drawing failed, just copy original
                shutil.copy2(img_path, smoke_dir / img_path.name)
        else:
            no_smoke_count += 1
            # Just copy the image without modifications
            shutil.copy2(img_path, no_smoke_dir / img_path.name)
    
    total = smoke_count + no_smoke_count
    smoke_pct = (smoke_count / total * 100) if total > 0 else 0
    
    return total, smoke_count, no_smoke_count, smoke_pct

def main():
    base_path = Path("/vol/bitbucket/si324/rf-detr-wildfire/src/images")
    data_path = base_path / "data"
    output_path = base_path / "pyro_readme_data_breakdown"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Dataset configurations
    datasets = {
        "2019 Smoke Data": {
            "images": data_path / "2019_Smoke" / "images",
            "labels": data_path / "2019_Smoke" / "labels",
            "output": output_path / "2019SmokeDataBB"
        },
        "AI4Mankind": {
            "images": data_path / "AI4Mankind" / "images",
            "labels": data_path / "AI4Mankind" / "labels",
            "output": output_path / "AI4MankindBB"
        },
        "SmokeFrames": {
            "images": data_path / "SmokeFrames" / "images",
            "labels": data_path / "SmokeFrames" / "labels",
            "output": output_path / "SmokeFramesBB"
        }
    }
    
    # Prepare table data
    table_lines = []
    table_lines.append("Dataset          | Total Images | Smoke | No Smoke | Smoke %")
    table_lines.append("-----------------|-------------|-------|----------|--------")
    
    # Process each dataset
    for dataset_name, paths in datasets.items():
        print(f"Processing {dataset_name}...")
        total, smoke, no_smoke, smoke_pct = count_and_organize_dataset(
            dataset_name,
            paths["images"],
            paths["labels"],
            paths["output"]
        )
        
        # Format numbers with commas
        total_str = f"{total:,}"
        smoke_str = f"{smoke:,}"
        no_smoke_str = f"{no_smoke:,}"
        smoke_pct_str = f"{smoke_pct:.1f}%"
        
        # Add to table (left-aligned with padding)
        table_lines.append(f"{dataset_name:<16} | {total_str:<11} | {smoke_str:<5} | {no_smoke_str:<8} | {smoke_pct_str:<6}")
    
    # Add Nemo (all smoke, no processing needed)
    nemo_total = 250
    table_lines.append(f"{'Nemo':<16} | {'250':<11} | {'250':<5} | {'0':<8} | {'100.0%':<6}")
    
    # Write table to file
    output_file = output_path / "dataset_breakdown.txt"
    with open(output_file, 'w') as f:
        f.write('\n'.join(table_lines))
    
    print(f"\nTable written to: {output_file}")
    print("\nDataset Breakdown:")
    print('\n'.join(table_lines))
    
    print("\nImages organized in:")
    for dataset_name, paths in datasets.items():
        print(f"  {paths['output']}/")
        print(f"    ├── smoke/       (images with GT bounding boxes drawn)")
        print(f"    └── no_smoke/    (images without smoke)")

if __name__ == "__main__":
    main()