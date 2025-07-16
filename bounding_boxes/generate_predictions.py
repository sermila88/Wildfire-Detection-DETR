import os
from PIL import Image
from tqdm import tqdm
import supervision as sv
from rfdetr import RFDETRBase

# --- Paths ---
DATASET_ROOT = "/vol/bitbucket/si324/rf-detr-wildfire/rf-initial-test-data/WildfireSmoke.v1-raw.coco"
OUTPUT_ROOT = "/vol/bitbucket/si324/rf-detr-wildfire/bounding_boxes/predicted_outputs"
MODEL_PATH = "/vol/bitbucket/si324/rf-detr-wildfire/output/checkpoint_best_ema.pth"

# --- Load fine-tuned model ---
model = RFDETRBase(pretrain_weights="/vol/bitbucket/si324/rf-detr-wildfire/output/checkpoint_best_ema.pth")

splits = ["train", "valid", "test"]  

for split in splits:
    print(f"Predicting {split}...")

    images_dir = os.path.join(DATASET_ROOT, split)
    annotations_path = os.path.join(DATASET_ROOT, split, "_annotations.coco.json")
    output_dir = os.path.join(OUTPUT_ROOT, split)
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset in COCO format using supervision
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

    # Loop through each image in the dataset 
    for path, image, _ in tqdm(dataset):
        # Predict bounding boxes on the image using RF-DET
        detections = model.predict(image, threshold=0.5)
        labels = [
            f"{dataset.classes[class_id]} {confidence:.2f}"
            for class_id, confidence in zip(detections.class_id, detections.confidence)
        ]

        annotated_image = bbox_annotator.annotate(image.copy(), detections)
        annotated_image = label_annotator.annotate(annotated_image, detections, labels)

        filename = os.path.basename(path)
        Image.fromarray(annotated_image).save(os.path.join(output_dir, filename))

