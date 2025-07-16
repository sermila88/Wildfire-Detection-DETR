import os
from PIL import Image
from tqdm import tqdm
import supervision as sv

# Path to your dataset root (adjust if different)
DATASET_ROOT = "/vol/bitbucket/si324/rf-detr-wildfire/rf-initial-test-data/WildfireSmoke.v1-raw.coco"
OUTPUT_ROOT = "/vol/bitbucket/si324/rf-detr-wildfire/bounding_boxes/annotated_ground_truth"

splits = ["train", "valid", "test"]

for split in splits:
    images_dir = os.path.join(DATASET_ROOT, split)
    annotations_path = os.path.join(DATASET_ROOT, split, "_annotations.coco.json")
    output_dir = os.path.join(OUTPUT_ROOT, split)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Processing {split}...")

    dataset = sv.DetectionDataset.from_coco(images_dir, annotations_path)
    thickness = sv.calculate_optimal_line_thickness((640, 640))  # You can tweak this if needed
    text_scale = sv.calculate_optimal_text_scale((640, 640))
    bbox_annotator = sv.BoxAnnotator(thickness=thickness)
    label_annotator = sv.LabelAnnotator(
        text_color=sv.Color.BLACK,
        text_scale=text_scale,
        text_thickness=thickness,
        smart_position=True
    )

    for path, image, annotations in tqdm(dataset):
        labels = [
            f"{dataset.classes[class_id]}"
            for class_id in annotations.class_id
        ]
        annotated_image = bbox_annotator.annotate(image.copy(), annotations)
        annotated_image = label_annotator.annotate(annotated_image, annotations, labels)

        filename = os.path.basename(path)
        Image.fromarray(annotated_image).save(os.path.join(output_dir, filename))

