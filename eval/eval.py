import os
import supervision as sv
from tqdm import tqdm
from supervision.metrics import MeanAveragePrecision
from PIL import Image
import numpy as np
from rfdetr import RFDETRBase
import matplotlib.pyplot as plt
from torchvision.ops import box_iou
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

model = RFDETRBase(pretrain_weights="/vol/bitbucket/si324/rf-detr-wildfire/output/img_data_train_output/checkpoint_best_ema.pth")  

# Test set location 
dataset_path = "/vol/bitbucket/si324/rf-detr-wildfire/pyro25img/images/test"

# Output directory for plots
output_dir = "/vol/bitbucket/si324/rf-detr-wildfire/eval/img_data_train_output"
os.makedirs(output_dir, exist_ok=True)

# Load test dataset (COCO format)
ds = sv.DetectionDataset.from_coco(
    images_directory_path=dataset_path,
    annotations_path=f"{dataset_path}/_annotations.coco.json"
)

# Predict and compare
predictions, targets = [], []
for path, image, annotations in tqdm(ds):
    image = Image.open(path)
    detections = model.predict(image, threshold=0.5)
    predictions.append(detections)
    targets.append(annotations)

for det in predictions:
    if len(det.class_id) > 0:
        det.class_id = np.zeros(len(det.class_id), dtype=int)

for ann in targets:
    if len(ann.class_id) > 0:
        ann.class_id = np.zeros(len(ann.class_id), dtype=int)
    
# mAP
map_metric = MeanAveragePrecision()
map_result = map_metric.update(predictions, targets).compute()
print("📊 mAP result:")
print(map_result)

map_plot_path = os.path.join(output_dir, "map_plot.png")
map_result.plot()
plt.savefig(map_plot_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"🖼️ mAP plot saved to: {map_plot_path}")

# Confusion Matrix
conf_matrix = sv.ConfusionMatrix.from_detections(
    predictions=predictions,
    targets=targets,
    classes=["smoke"]
)
conf_matrix_path = os.path.join(output_dir, "conf_matrix.png")
conf_matrix.plot()
plt.savefig(conf_matrix_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"🖼️ Confusion matrix saved to: {conf_matrix_path}")


# Collect matching presence/absence labels for classification metrics
# TP: any prediction that overlaps a ground-truth box (IOU >= 0.5)

# Initialize lists to hold the GT (y_true) and predicted labels (y_pred)
y_true = []
y_pred = []

# Loop through all predicted and ground-truth detection pairs
for pred, target in zip(predictions, targets):
    if len(target.xyxy) == 0 and len(pred.xyxy) == 0:
        continue
    elif len(target.xyxy) == 0:
        # No ground truth, but predictions - false positives
        y_true.extend([0] * len(pred.xyxy))
        y_pred.extend([1] * len(pred.xyxy))
    elif len(pred.xyxy) == 0:
        # Ground truth exists, no predictions - false negatives
        y_true.extend([1] * len(target.xyxy))
        y_pred.extend([0] * len(target.xyxy))
    else:
        # Both exist - compute IoU matching
        target_boxes = torch.tensor(target.xyxy, dtype=torch.float32)
        pred_boxes = torch.tensor(pred.xyxy, dtype=torch.float32)
        iou_matrix = box_iou(target_boxes, pred_boxes).numpy()
        
        # For each ground truth box
        for i in range(len(target.xyxy)):
            max_iou = np.max(iou_matrix[i]) if iou_matrix.size > 0 else 0
            y_true.append(1)
            y_pred.append(1 if max_iou >= 0.5 else 0)
        
        # For each predicted box that doesn't match any GT
        for j in range(len(pred.xyxy)):
            max_iou = np.max(iou_matrix[:, j]) if iou_matrix.size > 0 else 0
            if max_iou < 0.5:
                y_true.append(0)
                y_pred.append(1)

# Classification metrics
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)
acc = accuracy_score(y_true, y_pred)

print(f"✅ Precision: {precision:.4f}")
print(f"✅ Recall:    {recall:.4f}")
print(f"✅ F1 Score:  {f1:.4f}")
print(f"✅ Accuracy:  {acc:.4f}")


