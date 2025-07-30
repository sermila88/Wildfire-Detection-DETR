import os
from PIL import Image
from tqdm import tqdm
import supervision as sv
import numpy as np

# Path to dataset root 
DATASET_ROOT = "/vol/bitbucket/si324/rf-detr-wildfire/data/pyro25img"
OUTPUT_ROOT = "/vol/bitbucket/si324/rf-detr-wildfire/bounding_boxes/Pyro25Images/Ground_truth"

splits = ["train", "valid", "test"]

for split in splits:
    images_dir = os.path.join(DATASET_ROOT, "images", split)
    annotations_path = os.path.join(images_dir, "_annotations.coco.json")
    output_dir = os.path.join(OUTPUT_ROOT, split)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Processing {split}...")

    # Check if annotation file exists
    if not os.path.exists(annotations_path):
        print(f"⚠️ Annotation file not found: {annotations_path}")
        continue

    dataset = sv.DetectionDataset.from_coco(images_dir, annotations_path)
    thickness = sv.calculate_optimal_line_thickness((640, 640))  
    text_scale = sv.calculate_optimal_text_scale((640, 640))
    bbox_annotator = sv.BoxAnnotator(thickness=thickness)
    label_annotator = sv.LabelAnnotator(
        text_color=sv.Color.BLACK,
        text_scale=text_scale,
        text_thickness=thickness,
        smart_position=True
    )

    for path, image, annotations in tqdm(dataset):
        # Load original image directly with PIL to preserve color space
        original_pil = Image.open(path)
        
        # Convert PIL to numpy array for supervision (ensure RGB)
        if original_pil.mode != 'RGB':
            original_pil = original_pil.convert('RGB')
        image_array = np.array(original_pil)
        
        # Create labels for annotations (skip if no annotations)
        if annotations.class_id is not None and len(annotations.class_id) > 0:
            labels = [
                f"{dataset.classes[class_id]}"
                for class_id in annotations.class_id
            ]
        else:
            labels = []
        
        # Annotate the image using supervision
        annotated_image = bbox_annotator.annotate(image_array.copy(), annotations)
        annotated_image = label_annotator.annotate(annotated_image, annotations, labels)

        # Convert back to PIL Image maintaining color integrity
        annotated_pil = Image.fromarray(annotated_image.astype(np.uint8), mode='RGB')
        
        # Save with original filename
        filename = os.path.basename(path)
        output_path = os.path.join(output_dir, filename)
        annotated_pil.save(output_path, quality=95)  # High quality JPEG
        
print(f"✅ Grount truth bounding boxes generated for {DATASET_ROOT} ")