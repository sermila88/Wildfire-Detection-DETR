import os
import glob
from pathlib import Path
from datetime import datetime

# Paths
BASE_DIR = "/vol/bitbucket/si324/rf-detr-wildfire/src/images/data/pyro25img"
OUTPUT_DIR = "/vol/bitbucket/si324/rf-detr-wildfire/src/images/PyroImg_data_breakdown"

def count_smoke_images(split):
    """Count smoke vs no-smoke images for a given split"""
    
    images_dir = os.path.join(BASE_DIR, "images", split)
    labels_dir = os.path.join(BASE_DIR, "labels", split)
    
    # Get all image files
    image_files = glob.glob(os.path.join(images_dir, "*.jpg"))
    image_files.extend(glob.glob(os.path.join(images_dir, "*.png")))
    
    smoke_count = 0
    no_smoke_count = 0
    
    for img_path in image_files:
        img_name = Path(img_path).stem
        label_path = os.path.join(labels_dir, f"{img_name}.txt")
        
        # Check if label file exists and has content
        if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
            smoke_count += 1
        else:
            no_smoke_count += 1
    
    total = smoke_count + no_smoke_count
    smoke_pct = (smoke_count / total * 100) if total > 0 else 0
    
    return smoke_count, no_smoke_count, total, smoke_pct

def main():
    """Create dataset breakdown table"""
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Analyzing PyroNear dataset...")
    
    # Collect data
    splits = ['train', 'valid', 'test']
    data = {}
    
    for split in splits:
        data[split] = count_smoke_images(split)
    
    # Calculate totals
    total_smoke = sum(d[0] for d in data.values())
    total_no_smoke = sum(d[1] for d in data.values())
    total_images = sum(d[2] for d in data.values())
    total_pct = (total_smoke / total_images * 100) if total_images > 0 else 0
    
    # Save formatted table
    output_file = os.path.join(OUTPUT_DIR, "dataset_breakdown.txt")
    
    with open(output_file, 'w') as f:
        # Write LaTeX-friendly table format
        f.write("PyroNear Dataset Distribution\n")
        f.write("=" * 60 + "\n")
        f.write(f"{'Split':<10} {'Smoke':<10} {'No Smoke':<12} {'Total':<10} {'Smoke %':<10}\n")
        f.write("-" * 60 + "\n")
        
        for split in splits:
            smoke, no_smoke, total, pct = data[split]
            f.write(f"{split.capitalize():<10} {smoke:<10} {no_smoke:<12} {total:<10} {pct:<10.1f}\n")
        
        f.write("-" * 60 + "\n")
        f.write(f"{'Total':<10} {total_smoke:<10} {total_no_smoke:<12} {total_images:<10} {total_pct:<10.1f}\n")
        f.write("=" * 60 + "\n")
    
    # Print to console
    print(f"\n{'Split':<10} {'Smoke':<10} {'No Smoke':<12} {'Total':<10} {'Smoke %':<10}")
    print("-" * 60)
    for split in splits:
        smoke, no_smoke, total, pct = data[split]
        print(f"{split.capitalize():<10} {smoke:<10} {no_smoke:<12} {total:<10} {pct:<10.1f}")
    print("-" * 60)
    print(f"{'Total':<10} {total_smoke:<10} {total_no_smoke:<12} {total_images:<10} {total_pct:<10.1f}")
    
    print(f"\n✓ Table saved to: {output_file}")

if __name__ == "__main__":
    main()