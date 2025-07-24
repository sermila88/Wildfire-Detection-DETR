import os
from PIL import Image
from tqdm import tqdm
import supervision as sv
from rfdetr.detr import RFDETRBase
import numpy as np

# --- Configuration ---
EXPERIMENT_NAME = "rfdetr_smoke_detection_v1"  # Change this to match your experiment
DATASET_ROOT = "/vol/bitbucket/si324/rf-detr-wildfire/data/pyro25img/images"
OUTPUT_ROOT = f"/vol/bitbucket/si324/rf-detr-wildfire/bounding_boxes/Pyro25Images/predicted_bboxes/{EXPERIMENT_NAME}"
MODEL_PATH = f"/vol/bitbucket/si324/rf-detr-wildfire/outputs/{EXPERIMENT_NAME}/checkpoints/checkpoint_best_ema.pth"

# --- Load fine-tuned model ---
model = RFDETRBase(pretrain_weights=MODEL_PATH)

print(f"🎯 Generating predictions for experiment: {EXPERIMENT_NAME}")
print(f"📁 Output directory: {OUTPUT_ROOT}")

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
        # Load original image directly with PIL to preserve color space
        original_pil = Image.open(path)
        
        # Convert PIL to numpy array for supervision (ensure RGB)
        if original_pil.mode != 'RGB':
            original_pil = original_pil.convert('RGB')
        image_array = np.array(original_pil)
        
        # Predict bounding boxes on the image using RF-DETR (use PIL image for model)
        detections = model.predict(original_pil, threshold=0.5)
        
        # Create labels with confidence scores (skip if no detections)
        if (detections.class_id is not None and detections.confidence is not None and 
            len(detections.class_id) > 0):
            labels = [
                f"{dataset.classes[class_id]} {confidence:.2f}"
                for class_id, confidence in zip(detections.class_id, detections.confidence)
            ]
        else:
            labels = []

        # Annotate the image using supervision
        annotated_image = bbox_annotator.annotate(image_array.copy(), detections)
        annotated_image = label_annotator.annotate(annotated_image, detections, labels)

        # Convert back to PIL Image maintaining color integrity
        annotated_pil = Image.fromarray(annotated_image.astype(np.uint8), mode='RGB')
        
        # Save with original filename
        filename = os.path.basename(path)
        output_path = os.path.join(output_dir, filename)
        annotated_pil.save(output_path, quality=95)  # High quality JPEG

print(f"✅ Predicted bounding boxes generated for {EXPERIMENT_NAME} ")



