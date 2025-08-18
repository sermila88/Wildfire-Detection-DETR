"""
RF-DETR Evaluation Script for Wildfire Smoke Detection
Uses PyroNear methodology for baseline comparison
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import supervision as sv
from rfdetr import RFDETRBase
import seaborn as sns
from datetime import datetime
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, required=True)
parser.add_argument('--resolution', type=int, required=True)
parser.add_argument('--output_dir', type=str, required=True)
args = parser.parse_args()

# ============================================================================
# CONFIGURATION
# ============================================================================
EXPERIMENT_NAME = "rfdetr_hyperparameter_tuning_v3_eval"
PROJECT_ROOT = "/vol/bitbucket/si324/rf-detr-wildfire"

# Dataset paths
DATASET_PATH = f"{PROJECT_ROOT}/data/pyro25img/images/test"
ANNOTATIONS_PATH = f"{DATASET_PATH}/_annotations.coco.json"

# Output paths
EXPERIMENT_DIR = f"{PROJECT_ROOT}/outputs/{EXPERIMENT_NAME}/trial_006"
CHECKPOINTS_DIR = f"{EXPERIMENT_DIR}/checkpoints"
EVAL_RESULTS_DIR = f"{args.output_dir}/eval_results"
PLOTS_DIR = f"{args.output_dir}/plots"

# Evaluation parameters 
CONFIDENCE_THRESHOLDS = np.arange(0.1, 0.9, 0.05)
IOU_THRESHOLD = 0.01  

# Create output directories
os.makedirs(EVAL_RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

print(f"🎯 RF-DETR Wildfire Smoke Detection Evaluation")
print(f"📁 Experiment: {EXPERIMENT_NAME}")


# ============================================================================
# PYRONEAR UTILITY FUNCTIONS
# ============================================================================
def xywh2xyxy(x):
    """Function to convert bounding box format from center to top-left corner"""
    y = np.zeros_like(x)
    y[0] = x[0] - x[2] / 2  # x_min
    y[1] = x[1] - x[3] / 2  # y_min
    y[2] = x[0] + x[2] / 2  # x_max
    y[3] = x[1] + x[3] / 2  # y_max
    return y


def box_iou(box1: np.ndarray, box2: np.ndarray, eps: float = 1e-7):
    """
    Calculate intersection-over-union (IoU) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Based on https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py

    Args:
        box1 (np.ndarray): A numpy array of shape (N, 4) representing N bounding boxes.
        box2 (np.ndarray): A numpy array of shape (M, 4) representing M bounding boxes.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (np.ndarray): An NxM numpy array containing the pairwise IoU values for every element in box1 and box2.
    """

    # Ensure box1 and box2 are in the shape (N, 4) even if N is 1
    if box1.ndim == 1:
        box1 = box1.reshape(1, 4)
    if box2.ndim == 1:
        box2 = box2.reshape(1, 4)

    (a1, a2), (b1, b2) = np.split(box1, 2, 1), np.split(box2, 2, 1)
    inter = (
        (np.minimum(a2, b2[:, None, :]) - np.maximum(a1, b1[:, None, :]))
        .clip(0)
        .prod(2)
    )

    # IoU = inter / (area1 + area2 - inter)
    return inter / ((a2 - a1).prod(1) + (b2 - b1).prod(1)[:, None] - inter + eps)


def save_predictions_yolo_format(preds, output_path, img_width, img_height):
    """Save RF-DETR predictions in YOLO format for debugging"""
    with open(output_path, 'w') as f:
        for i, box in enumerate(preds.xyxy):
            x1, y1, x2, y2 = box
            # Convert to YOLO format (normalized center x, y, width, height)
            x_center = ((x1 + x2) / 2) / img_width
            y_center = ((y1 + y2) / 2) / img_height
            width = (x2 - x1) / img_width
            height = (y2 - y1) / img_height
            conf = preds.confidence[i] if hasattr(preds, 'confidence') and preds.confidence is not None else 1.0
            # Format: class_id x_center y_center width height confidence
            f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {conf:.6f}\n")

# ============================================================================
# LOAD YOLO BASELINE FOR COMPARISON
# ============================================================================


def load_yolo_baseline():
    """Load YOLO baseline (image + object level) from saved summary JSON."""
    path = f"{PROJECT_ROOT}/outputs/yolo_baseline_v1_IoU=0.01/eval_results/summary_results.json"
    try:
        with open(path, "r") as f:
            data = json.load(f)

        baseline = {
            "experiment_name": data.get("experiment_name"),
            "iou_threshold": data.get("iou_threshold"),
            "image": {
                "best_threshold": data["image_level_best_threshold"],
                "metrics": data["image_level_metrics"],
                "confusion_matrix": data["image_level_confusion_matrix"],
            },
            "object": {
                "best_threshold": data["object_level_best_threshold"],
                "metrics": data["object_level_metrics"],
                "confusion_matrix": data["object_level_confusion_matrix"],
            },
        }
        print(
            f"✅ Loaded YOLO baseline @IoU={baseline['iou_threshold']}: "
            f"image-F1={baseline['image']['metrics']['f1_score']:.3f} "
            f"(thr={baseline['image']['best_threshold']})"
        )
        return baseline

    except FileNotFoundError:
        print(f"⚠️  YOLO baseline file not found at:\n    {path}")
        return {
            "experiment_name": "yolo_baseline_fallback",
            "iou_threshold": IOU_THRESHOLD,
            "image": {
                "best_threshold": 0.1,
                "metrics": {"f1_score": 0.0, "precision": 0.0, "recall": 0.0, "accuracy": 0.0},
                "confusion_matrix": {"true_positives": 0, "true_negatives": 0, "false_positives": 0, "false_negatives": 0},
            },
            "object": {
                "best_threshold": 0.2,
                "metrics": {"f1_score": 0.0, "precision": 0.0, "recall": 0.0},
                "confusion_matrix": {"true_positives": 0, "false_positives": 0, "false_negatives": 0},
            },
        }

# ============================================================================
# MODEL INITIALIZATION
# ============================================================================
def load_model():
    """Load RF-DETR model from checkpoint"""
    checkpoint_path = args.checkpoint
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        exit(1)
    
    print(f"✅ Loading RF-DETR from: {checkpoint_path}")
    model = RFDETRBase(pretrain_weights=checkpoint_path, num_classes=1, resolution=args.resolution)
    return model

# ============================================================================
# DATASET LOADING
# ============================================================================
def load_dataset():

    """Load all test images (both annotated and non-annotated)"""
    if not os.path.exists(ANNOTATIONS_PATH):
        print(f"❌ Annotations not found: {ANNOTATIONS_PATH}")
        exit(1)
    
    # Load COCO dataset for annotated images
    coco_ds = sv.DetectionDataset.from_coco(
        images_directory_path=DATASET_PATH,
        annotations_path=ANNOTATIONS_PATH
    )
    
    # Get all test images
    all_test_images = [f for f in os.listdir(DATASET_PATH) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    coco_images = {os.path.basename(path) for path, _, _ in coco_ds}
    non_coco_images = [img for img in all_test_images if img not in coco_images]
    
    print(f"📊 Dataset composition:")
    print(f"  Annotated images (COCO): {len(coco_ds)}")
    print(f"  Non-annotated images: {len(non_coco_images)}")
    print(f"  Total test images: {len(all_test_images)}")
    
    # Create combined dataset
    combined_dataset = []
    
    # Add annotated images
    for path, image, annotations in coco_ds:
        combined_dataset.append((path, None, annotations))
    
    # Add non-annotated images with empty annotations
    for img_name in non_coco_images:
        img_path = os.path.join(DATASET_PATH, img_name)
        if os.path.exists(img_path):
            empty_annotations = sv.Detections.empty()
            combined_dataset.append((img_path, None, empty_annotations))
    
    print(f"📊 Combined dataset: {len(combined_dataset)} images")
    return combined_dataset


# ============================================================================
# PYRONEAR-STYLE EVALUATION
# ============================================================================

def has_spatial_overlap(predictions, ground_truth):
    """Check if predictions have spatial overlap with ground truth 
    using PyroNear's IoU"""
    if len(predictions.xyxy) == 0 or len(ground_truth.xyxy) == 0:
        return False
    
    for pred_box in predictions.xyxy:
        for gt_box in ground_truth.xyxy:
            iou_matrix = box_iou(pred_box, gt_box)
            iou = iou_matrix[0, 0]  # Extract single IoU value
            if iou > IOU_THRESHOLD:
                return True
    return False

def _object_tp_fp_fn_for_image(preds, annotations, iou_th=IOU_THRESHOLD):
    """Compute object-level TP/FP/FN for a single image."""
    gt = np.array(annotations.xyxy)
    pr = np.array(preds.xyxy)
    tp = fp = fn = 0

    # When NO GT, every prediction above conf is a FP
    if gt.size == 0:
        return 0, len(pr), 0

    # Match predictions to GT with 1-1 assignment and count FPs otherwise
    matched = np.zeros(len(gt), dtype=bool)

    for pb in pr:
        ious = []
        for gt_box in gt: # Check overlap with ground truth boxes
            iou_matrix = box_iou(pb, gt_box)
            iou_val = iou_matrix[0, 0] # Extract single IoU value
            ious.append(iou_val)

        # Find best matching GT box for this prediction
        max_iou = float(np.max(ious))
        idx = int(np.argmax(ious))

        # If IoU is above threshold and not already matched, count as TP
        if max_iou > iou_th and not matched[idx]:
            tp += 1
            matched[idx] = True

        # If IoU is below threshold or already matched, count as FP
        else:
            fp += 1
 
    # Count unmatched GT boxes as FN
    fn += int((~matched).sum())
    return tp, fp, fn


def _counts_at_conf(model, dataset, conf):
    """
    One sweep over the dataset at a given confidence:
      - returns image-level TP/FP/FN/TN
      - returns object-level TP/FP/FN
    """
    img_tp = img_fp = img_fn = img_tn = 0
    obj_tp = obj_fp = obj_fn = 0

    # Create predictions directory in the correct location
    pred_dir = f"{args.output_dir}/rfdetr_preds_yolo_format/conf_{conf:.2f}"
    os.makedirs(pred_dir, exist_ok=True) 

    for path, _, annotations in dataset:
        # ensure PIL RGB; doesn't alter colors, just guarantees channel order
        with Image.open(path) as _img:
            img = _img.convert("RGB")
            preds = model.predict(img, threshold=conf)

            # Save predictions in YOLO format
            img_name = os.path.basename(path).replace('.jpg', '').replace('.png', '').replace('.jpeg', '')
            pred_path = os.path.join(pred_dir, f"{img_name}.txt")
            save_predictions_yolo_format(preds, pred_path, img.width, img.height)

        # ---- image-level ----
        has_smoke_gt = len(annotations.xyxy) > 0
        has_smoke_pred = len(preds.xyxy) > 0
        spatial_match = False
        if has_smoke_gt and has_smoke_pred:
            spatial_match = has_spatial_overlap(preds, annotations)

        if has_smoke_gt:
            if spatial_match: img_tp += 1
            else:             img_fn += 1
        else:
            if has_smoke_pred: img_fp += 1
            else:              img_tn += 1

        # ---- object-level  ----
        tpo, fpo, fno = _object_tp_fp_fn_for_image(preds, annotations, IOU_THRESHOLD)
        obj_tp += tpo; obj_fp += fpo; obj_fn += fno

    return (img_tp, img_fp, img_fn, img_tn), (obj_tp, obj_fp, obj_fn)


def evaluate_model_both_levels(model, dataset):
    """
    Run image-level and object-level eval in parallel across thresholds.
    Returns (img_results, obj_results) lists with metrics per threshold.
    """
    print(f"\n🔥 Running evaluation across {len(CONFIDENCE_THRESHOLDS)} thresholds (image+object together)...")
    img_results, obj_results = [], []

    for conf in tqdm(CONFIDENCE_THRESHOLDS, desc="Evaluating"):
        (img_tp, img_fp, img_fn, img_tn), (obj_tp, obj_fp, obj_fn) = _counts_at_conf(model, dataset, conf)
        total_images = len(dataset)

        # image-level metrics
        img_prec = img_tp / (img_tp + img_fp) if (img_tp + img_fp) > 0 else 0.0
        img_rec  = img_tp / (img_tp + img_fn) if (img_tp + img_fn) > 0 else 0.0
        img_f1   = 2*img_prec*img_rec / (img_prec + img_rec) if (img_prec + img_rec) > 0 else 0.0
        img_acc  = (img_tp + img_tn) / total_images if total_images > 0 else 0.0

        img_results.append({
            "confidence_threshold": float(conf),
            "precision": img_prec, "recall": img_rec, "f1_score": img_f1, "accuracy": img_acc,
            "tp": img_tp, "fp": img_fp, "fn": img_fn, "tn": img_tn
        })

        # object-level metrics
        obj_prec = obj_tp / (obj_tp + obj_fp) if (obj_tp + obj_fp) > 0 else 0.0
        obj_rec  = obj_tp / (obj_tp + obj_fn) if (obj_tp + obj_fn) > 0 else 0.0
        obj_f1   = 2*obj_prec*obj_rec / (obj_prec + obj_rec) if (obj_prec + obj_rec) > 0 else 0.0

        obj_results.append({
            "confidence_threshold": float(conf),
            "precision": obj_prec, "recall": obj_rec, "f1_score": obj_f1,
            "tp": obj_tp, "fp": obj_fp, "fn": obj_fn
        })

        print(f"Conf {conf:.2f}  |  IMG  P={img_prec:.3f} R={img_rec:.3f} F1={img_f1:.3f} Acc={img_acc:.3f}  ||  "
              f"OBJ  P={obj_prec:.3f} R={obj_rec:.3f} F1={obj_f1:.3f}")

    return img_results, obj_results



# ============================================================================
# VISUALIZATION
# ============================================================================
def create_image_metrics_plot(img_results):
    conf = [r['confidence_threshold'] for r in img_results]
    f1   = [r['f1_score'] for r in img_results]
    prec = [r['precision'] for r in img_results]
    rec  = [r['recall'] for r in img_results]
    acc  = [r['accuracy'] for r in img_results]

    best_idx = int(np.argmax(f1))
    best_conf, best_f1 = conf[best_idx], f1[best_idx]
    best_prec, best_rec, best_acc = prec[best_idx], rec[best_idx], acc[best_idx]

    plt.figure(figsize=(10, 6))
    plt.plot(conf, f1,  label="F1 Score", color="blue", marker="o")
    plt.plot(conf, prec, label="Precision", color="orange", linestyle="--")
    plt.plot(conf, rec,  label="Recall", color="green", linestyle="-.")
    plt.plot(conf, acc,  label="Accuracy", color="red", linestyle=":")
    
    # Scatter points at best threshold
    plt.scatter(best_conf, best_f1, color="blue", s=150, marker='*', edgecolor="black", zorder=6)
    plt.scatter(best_conf, best_prec, color="orange", s=100, edgecolor="k", zorder=5)
    plt.scatter(best_conf, best_rec, color="green", s=100, edgecolor="k", zorder=5)
    plt.scatter(best_conf, best_acc, color="red", s=100, edgecolor="k", zorder=5)
    
    # Add text annotations like YOLO
    plt.text(best_conf + 0.02, best_f1,
            f"Best F1: {best_f1:.2f}",
            fontsize=9, ha='left', va='center', color='blue', weight='bold')
    
    plt.text(best_conf + 0.02, best_prec,
            f"Precision: {best_prec:.2f}",
            fontsize=9, ha='left', va='center', color='orange', weight='bold')
    
    plt.text(best_conf + 0.02, best_rec,
            f"Recall: {best_rec:.2f}",
            fontsize=9, ha='left', va='center', color='green', weight='bold')
    
    plt.text(best_conf + 0.02, best_acc,
            f"Accuracy: {best_acc:.2f}",
            fontsize=9, ha='left', va='center', color='red', weight='bold')
    
    plt.title(f"{EXPERIMENT_NAME} – Image-Level Metrics vs. Confidence Threshold")
    plt.xlabel("Confidence Threshold")
    plt.ylabel("Metric Value")
    plt.legend()
    plt.grid(True)
    out = f"{PLOTS_DIR}/image_metrics.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    return out, img_results[best_idx]


def create_object_metrics_plot(obj_results):
    conf = [r['confidence_threshold'] for r in obj_results]
    f1   = [r['f1_score'] for r in obj_results]
    prec = [r['precision'] for r in obj_results]
    rec  = [r['recall'] for r in obj_results]

    best_idx = int(np.argmax(f1))
    best_conf, best_f1 = conf[best_idx], f1[best_idx]
    best_prec, best_rec = prec[best_idx], rec[best_idx]

    plt.figure(figsize=(10, 6))
    plt.plot(conf, f1,  label="F1 Score", color="blue", marker="o")
    plt.plot(conf, prec, label="Precision", color="orange", linestyle="--")
    plt.plot(conf, rec,  label="Recall", color="green", linestyle="-.")
    
    # Scatter points at best threshold
    plt.scatter(best_conf, best_f1, color="blue", s=150, marker='*', edgecolor="black", zorder=6)
    plt.scatter(best_conf, best_prec, color="orange", s=100, edgecolor="k", zorder=5)
    plt.scatter(best_conf, best_rec, color="green", s=100, edgecolor="k", zorder=5)
    
    # Add text annotations like YOLO
    plt.text(best_conf + 0.02, best_f1,
            f"Best F1: {best_f1:.2f}",
            fontsize=9, ha='left', va='center', color='blue', weight='bold')
    
    plt.text(best_conf + 0.02, best_prec,
            f"Precision: {best_prec:.2f}",
            fontsize=9, ha='left', va='center', color='orange', weight='bold')
    
    plt.text(best_conf + 0.02, best_rec,
            f"Recall: {best_rec:.2f}",
            fontsize=9, ha='left', va='center', color='green', weight='bold')
    
    plt.title(f"{EXPERIMENT_NAME} – Object-Level Metrics vs. Confidence Threshold")
    plt.xlabel("Confidence Threshold")
    plt.ylabel("Metric Value")
    plt.legend()
    plt.grid(True)
    out = f"{PLOTS_DIR}/object_metrics.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    return out, obj_results[best_idx]

def create_image_confusion_matrix_from_row(img_best_row):
    cm = np.array([[img_best_row["tp"], img_best_row["fn"]],
                   [img_best_row["fp"], img_best_row["tn"]]])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Pred Fire", "Pred No Fire"],
                yticklabels=["GT Fire", "GT No Fire"])
    plt.title(f"{EXPERIMENT_NAME} – Image-Level Confusion Matrix (IoU={IOU_THRESHOLD})")
    out = f"{PLOTS_DIR}/image_conf_matrix.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    return out


def create_object_confusion_matrix_from_row(obj_best_row):
    # Create numeric matrix for heatmap
    cm_numeric = np.array([[obj_best_row["tp"], obj_best_row["fn"]],
                          [obj_best_row["fp"], 0]])
    
    # Create annotation matrix with N/A
    cm_labels = np.array([[str(obj_best_row["tp"]), str(obj_best_row["fn"])],
                         [str(obj_best_row["fp"]), "N/A"]])
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_numeric, annot=cm_labels, fmt="", cmap="Oranges",
                xticklabels=["Pred Fire", "Pred No Fire"],
                yticklabels=["GT Fire", "GT No Fire"])
    plt.title(f"{EXPERIMENT_NAME} – Object-Level Confusion Matrix (TN = N/A, IoU={IOU_THRESHOLD})")
    out = f"{PLOTS_DIR}/object_conf_matrix.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    return out

# ============================================================================
# RESULTS SAVING
# ============================================================================
def save_results(img_results, obj_results, yolo_baseline, img_best, obj_best):
    yolo_img = yolo_baseline["image"]["metrics"]
    yolo_obj = yolo_baseline["object"]["metrics"]

    # improvements (image-level comparison is the headline; object-level optional)
    def _pct_increase(new, old):
        return ((new - old) / old * 100.0) if old > 0 else 0.0
    imp_img_pct = _pct_increase(img_best['f1_score'], yolo_img['f1_score'])
    imp_obj_pct = _pct_increase(obj_best['f1_score'], yolo_obj['f1_score'])

    evaluation_data = {
        "experiment_info": {
            "experiment_name": EXPERIMENT_NAME,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "iou_threshold": IOU_THRESHOLD, # this eval's IoU
            "yolo_baseline_iou_threshold": yolo_baseline.get("iou_threshold", None),
            "dataset_path": DATASET_PATH
        },
        "image_best_results": {
            "confidence_threshold": img_best['confidence_threshold'],
            "f1_score": img_best['f1_score'],
            "precision": img_best['precision'],
            "recall": img_best['recall'],
            "accuracy": img_best['accuracy'],
            "confusion_matrix": {
                "true_positives": int(img_best["tp"]),
                "true_negatives": int(img_best["tn"]),
                "false_positives": int(img_best["fp"]),
                "false_negatives": int(img_best["fn"])
            }
        },
        "object_best_results": {
            "confidence_threshold": obj_best['confidence_threshold'],
            "f1_score": obj_best['f1_score'],
            "precision": obj_best['precision'],
            "recall": obj_best['recall'],
            "confusion_matrix": {
                "true_positives": int(obj_best["tp"]),
                "false_positives": int(obj_best["fp"]),
                "false_negatives": int(obj_best["fn"]),
                "true_negatives": "N/A"
            }
        },
        "baseline_comparison": {
            "yolo_baseline": yolo_baseline,
            "improvements_percent": {
                "image_f1": imp_img_pct,
                "object_f1": imp_obj_pct
            }
        },
        "detailed_results": {
            "image_level": img_results,
            "object_level": obj_results
        }
    }
    # Summary results text
    eval_summary_path = f"{EVAL_RESULTS_DIR}/evaluation_summary.txt"
    with open(eval_summary_path, "w") as f:
        f.write(f"Experiment: {EXPERIMENT_NAME}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"IoU threshold (this eval): {IOU_THRESHOLD}\n")
        f.write(f"YOLO baseline IoU: {yolo_baseline.get('iou_threshold', 'N/A')}\n")

        f.write("IMAGE LEVEL PRIMARY METRICS:\n")
        f.write(f"Precision: {img_best['precision']:.4f}\n")
        f.write(f"Recall: {img_best['recall']:.4f}\n")
        f.write(f"F1 Score: {img_best['f1_score']:.4f}\n")
        f.write(f"Accuracy: {img_best['accuracy']:.4f}\n\n")

        f.write("IMAGE LEVEL CONFUSION MATRIX:\n")
        f.write(f"TP: {int(img_best['tp'])}\n")
        f.write(f"TN: {int(img_best['tn'])}\n")
        f.write(f"FP: {int(img_best['fp'])}\n")
        f.write(f"FN: {int(img_best['fn'])}\n\n")

        f.write("OBJECT LEVEL PRIMARY METRICS:\n")
        f.write(f"Precision: {obj_best['precision']:.4f}\n")
        f.write(f"Recall: {obj_best['recall']:.4f}\n")
        f.write(f"F1 Score: {obj_best['f1_score']:.4f}\n")
        f.write("Accuracy: N/A\n\n")  # no TN at object-level → accuracy undefined

        f.write("OBJECT LEVEL CONFUSION MATRIX:\n")
        f.write(f"TP: {int(obj_best['tp'])}\n")
        f.write("TN: N/A\n")
        f.write(f"FP: {int(obj_best['fp'])}\n")
        f.write(f"FN: {int(obj_best['fn'])}\n")


    results_path = f"{EVAL_RESULTS_DIR}/evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump(evaluation_data, f, indent=2)

    summary_path = f"{EVAL_RESULTS_DIR}/comparison_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(f"Experiment: {EXPERIMENT_NAME}\n\n")
        f.write("RF-DETR vs YOLO Baseline \n")
        f.write("=" * 48 + "\n\n")
        f.write(f"IoU threshold (this eval): {IOU_THRESHOLD}\n")
        f.write(f"YOLO baseline IoU: {yolo_baseline.get('iou_threshold', 'N/A')}\n")

        f.write("YOLO Baseline (Image Level):\n")
        f.write(f"  F1: {yolo_img['f1_score']:.4f}  Precision: {yolo_img['precision']:.4f}  "
                f"Recall: {yolo_img['recall']:.4f}  Accuracy: {yolo_img['accuracy']:.4f}\n\n")

        f.write("RF-DETR (Image Level):\n")
        f.write(f"  F1: {img_best['f1_score']:.4f}  Precision: {img_best['precision']:.4f}  "
                f"Recall: {img_best['recall']:.4f}  Accuracy: {img_best['accuracy']:.4f}  "
                f"@conf={img_best['confidence_threshold']:.2f}\n\n")

        f.write("YOLO Baseline (Object Level):\n")
        f.write(f"  F1: {yolo_obj['f1_score']:.4f}  Precision: {yolo_obj['precision']:.4f}  "
                f"Recall: {yolo_obj['recall']:.4f}\n\n")

        f.write("RF-DETR (Object Level):\n")
        f.write(f"  F1: {obj_best['f1_score']:.4f}  Precision: {obj_best['precision']:.4f}  "
                f"Recall: {obj_best['recall']:.4f}  @conf={obj_best['confidence_threshold']:.2f}\n\n")

        f.write("Improvements on YOLO Baseline:\n")
        f.write(f"  Image F1: {imp_img_pct:+.1f}%\n")
        f.write(f"  Object F1: {imp_obj_pct:+.1f}%\n")

    print(f"💾 Results saved:\n  📄 {results_path}\n  📊 {summary_path} 📝 {eval_summary_path}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    """Main evaluation pipeline"""
    print("🚀 Starting RF-DETR evaluation...")
    
    # Load components
    yolo_baseline = load_yolo_baseline()
    model = load_model()
    dataset = load_dataset()
    
    # Run evaluation (both levels together)
    img_results, obj_results = evaluate_model_both_levels(model, dataset)

    # Plots + get best rows
    img_plot, img_best = create_image_metrics_plot(img_results)
    obj_plot, obj_best = create_object_metrics_plot(obj_results)

    # Confusion matrices (from best rows, no extra inference)
    img_cm_path = create_image_confusion_matrix_from_row(img_best)
    obj_cm_path = create_object_confusion_matrix_from_row(obj_best)

    # Save results (both levels) & summaries
    _ = save_results(img_results, obj_results, yolo_baseline, img_best, obj_best)

    # Final summary
    print(f"\n🏆 EVALUATION COMPLETE!")
    print(f"   RF-DETR Image F1:  {img_best['f1_score']:.4f} @ conf {img_best['confidence_threshold']:.2f}")
    print(f"   RF-DETR Object F1: {obj_best['f1_score']:.4f} @ conf {obj_best['confidence_threshold']:.2f}")
    print(f"   YOLO Baseline Image F1:  {yolo_baseline['image']['metrics']['f1_score']:.4f}")
    print(f"   YOLO Baseline Object F1: {yolo_baseline['object']['metrics']['f1_score']:.4f}")
    print(f"   Image CM: {img_cm_path}")
    print(f"   Object CM: {obj_cm_path}")

if __name__ == "__main__":
    main()