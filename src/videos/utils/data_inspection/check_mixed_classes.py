"""
check_mixed_classes.py - Quick check for folders with both class 0 and 1
"""

import os
import glob
from collections import defaultdict

BASE_PATH = "/vol/bitbucket/si324/rf-detr-wildfire/src/videos/data"
SPLITS = ["train", "val", "test"]

def check_video_folder(video_path):
    """Check what classes are present in a video folder."""
    video_name = os.path.basename(video_path)
    labels_dir = os.path.join(video_path, "labels")
    
    if not os.path.exists(labels_dir):
        return video_name, set()
    
    classes_found = set()
    
    # Check all label files
    label_files = glob.glob(os.path.join(labels_dir, "*.txt"))
    
    for label_path in label_files:
        try:
            with open(label_path, 'r') as f:
                content = f.read().strip()
                if content:
                    # Get class from first character of each line
                    for line in content.split('\n'):
                        if line.strip():
                            classes_found.add(int(line[0]))
        except:
            pass
    
    return video_name, classes_found

def main():
    print("=" * 60)
    print("CHECKING FOR MIXED CLASS FOLDERS")
    print("=" * 60)
    
    results = defaultdict(lambda: {"class_0_only": [], "class_1_only": [], "mixed": [], "no_labels": []})
    
    for split in SPLITS:
        print(f"\nChecking {split}...")
        split_path = os.path.join(BASE_PATH, split)
        
        if not os.path.exists(split_path):
            print(f"  {split} not found")
            continue
        
        video_folders = [d for d in glob.glob(os.path.join(split_path, "*")) 
                        if os.path.isdir(d)]
        
        for video_path in video_folders:
            video_name, classes = check_video_folder(video_path)
            
            if not classes:
                results[split]["no_labels"].append(video_name)
            elif 0 in classes and 1 in classes:
                results[split]["mixed"].append(video_name)
                print(f" MIXED: {video_name} has both class 0 and 1")
            elif 0 in classes:
                results[split]["class_0_only"].append(video_name)
            elif 1 in classes:
                results[split]["class_1_only"].append(video_name)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for split in SPLITS:
        print(f"\n{split.upper()}:")
        print(f"  Class 0 only: {len(results[split]['class_0_only'])}")
        print(f"  Class 1 only: {len(results[split]['class_1_only'])}")
        print(f"  MIXED (0 & 1): {len(results[split]['mixed'])}")
        print(f"  No labels: {len(results[split]['no_labels'])}")
        
        if results[split]["mixed"]:
            print(f"\n  Mixed folders in {split}:")
            for folder in results[split]["mixed"]:
                print(f"    - {folder}")
    
    # Check if ANY mixed folders exist
    total_mixed = sum(len(results[split]["mixed"]) for split in SPLITS)
    
    print("\n" + "=" * 60)
    if total_mixed > 0:
        print(f" Found {total_mixed} folders with BOTH class 0 and 1!")
    else:
        print("✅ No mixed class folders found - all folders have only one class")
    print("=" * 60)

if __name__ == "__main__":
    main()