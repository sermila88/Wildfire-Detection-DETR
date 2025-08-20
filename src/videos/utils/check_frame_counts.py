# check_frame_counts.py
import os
import glob
from collections import Counter

def check_frame_counts(data_dir):
    """Check frame count distribution across all videos."""
    
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(data_dir, split)
        if not os.path.exists(split_dir):
            continue
            
        frame_counts = []
        video_folders = [d for d in glob.glob(os.path.join(split_dir, "*")) 
                        if os.path.isdir(d)]
        
        for video_folder in video_folders:
            frames = glob.glob(os.path.join(video_folder, "*.jpg"))
            frame_counts.append(len(frames))
        
        if frame_counts:
            print(f"\n{split.upper()} SET:")
            print(f"  Total videos: {len(frame_counts)}")
            print(f"  Min frames: {min(frame_counts)}")
            print(f"  Max frames: {max(frame_counts)}")
            print(f"  Mean frames: {sum(frame_counts)/len(frame_counts):.1f}")
            print(f"  Videos with <10 frames: {sum(1 for c in frame_counts if c < 10)}")
            print(f"  Videos with <5 frames: {sum(1 for c in frame_counts if c < 5)}")
            
            # Show distribution
            counter = Counter(frame_counts)
            print(f"  Most common lengths: {counter.most_common(5)}")

if __name__ == "__main__":
    check_frame_counts("/vol/bitbucket/si324/rf-detr-wildfire/src/videos/data")