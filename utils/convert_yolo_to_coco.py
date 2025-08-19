import os
import json
from PIL import Image

def convert_split_to_coco(image_dir, label_dir, output_path, categories):
    coco = {
        "info": {
            "description": "Wildfire Smoke Dataset",
            "version": "1.0",
            "year": 2025
        },
        "licenses": [],
        "categories": [{"id": i, "name": name, "supercategory": "object"} for i, name in enumerate(categories)],
        "images": [],
        "annotations": [],
    }

    image_id = 0
    annotation_id = 0
    no_annotations = 0

    # Process all images in the directory
    for fname in sorted(os.listdir(image_dir)):
        if not fname.endswith(".jpg"):
            continue

        img_path = os.path.join(image_dir, fname)
        lbl_path = os.path.join(label_dir, fname.replace(".jpg", ".txt"))

        # Get image dimensions
        with Image.open(img_path) as img:
            width, height = img.size

        # Add ALL images to the dataset 
        coco["images"].append({
            "id": image_id,
            "file_name": fname,
            "width": width,
            "height": height
        })

        # Check if corresponding label file exists
        if os.path.exists(lbl_path):
            # Read all lines first to check if file is empty
            with open(lbl_path, "r") as f:
                lines = f.readlines()
            
            # Process only if file has content (not empty)
            if lines:
                for line in lines:
                    parts = line.strip().split()
                    
                    # Skip malformed lines (YOLO format should have exactly 5 values)
                    if len(parts) != 5:
                        continue

                    # Parse YOLO format: class_id center_x center_y width height 
                    class_id, x, y, w, h = map(float, parts)

                    # Convert to COCO format: absolute pixel coordinates
                    abs_w = w * width
                    abs_h = h * height
                    abs_x = (x * width) - (abs_w / 2)
                    abs_y = (y * height) - (abs_h / 2)

                    # Add annotation in COCO format
                    coco["annotations"].append({
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": int(class_id),
                        "bbox": [abs_x, abs_y, abs_w, abs_h],
                        "area": abs_w * abs_h,
                        "iscrowd": 0,
                        "segmentation": []
                    })
                    annotation_id += 1

            else:
                no_annotations += 1  # Empty label file
        else:
            no_annotations += 1  # No label file

        image_id += 1

    # Save the COCO JSON file
    with open(output_path, "w") as f:
        json.dump(coco, f, indent=2)

    print(f"✅ Saved {output_path}")
    print(f"📸 Total images processed: {image_id}")
    print(f"📊 Images without annotations (background images): {no_annotations}")
    print(f"🔥 Images with smoke annotations: {image_id - no_annotations}")


if __name__ == "__main__":
    base = "data/pyro25img"
    categories = ["smoke"]
    splits = ["train", "valid", "test"]

    for split in splits:
        image_dir = os.path.join(base, "images", split)
        label_dir = os.path.join(base, "labels", split)
        output_path = os.path.join(image_dir, "_annotations.coco.json")

        convert_split_to_coco(
            image_dir=image_dir,
            label_dir=label_dir,
            output_path=output_path,
            categories=categories
        )

