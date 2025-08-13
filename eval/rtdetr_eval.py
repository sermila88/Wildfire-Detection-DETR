"""
Real Time Detection Transformer:
RT-DETR Evaluation Script for Wildfire Smoke Detection
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import torch
from torchvision.ops import box_iou
import supervision as sv
from ultralytics import RTDETR
import seaborn as sns

# ============================================================================
# CONFIGURATION
# ============================================================================
EXPERIMENT_NAME = "rtdetr_smoke_detection_v1"
PROJECT_ROOT = "/vol/bitbucket/si324/rf-detr-wildfire"

# Dataset paths
DATASET_PATH = f"{PROJECT_ROOT}/data/pyro25img/images/test"
ANNOTATIONS_PATH = f"{DATASET_PATH}/_annotations.coco.json"

# Output paths
EXPERIMENT_DIR = f"{PROJECT_ROOT}/outputs/{EXPERIMENT_NAME}"
CHECKPOINTS_DIR = f"{EXPERIMENT_DIR}/checkpoints"
EVAL_RESULTS_DIR = f"{EXPERIMENT_DIR}/eval_results"
PLOTS_DIR = f"{EXPERIMENT_DIR}/plots"

# Evaluation parameters (PyroNear methodology)
CONFIDENCE_THRESHOLDS = np.arange(0.1, 0.9, 0.05)
SPATIAL_IOU_THRESHOLD = 0.1  # PyroNear baseline

# Create output directories
os.makedirs(EVAL_RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

print(f"🎯 RT-DETR Wildfire Smoke Detection Evaluation")
print(f"📁 Experiment: {EXPERIMENT_NAME}")

# ============================================================================
# LOAD YOLO BASELINE FOR COMPARISON
# ============================================================================
def _format_iou_tag(iou: float) -> str:
    s = f"{iou:.2f}"
    return s.rstrip("0").rstrip(".")

def load_yolo_baseline(iou_for_baseline: float = SPATIAL_IOU_THRESHOLD):
    """Load YOLO baseline (image + object level) from saved summary JSON."""
    tag = _format_iou_tag(iou_for_baseline)
    path = f"{PROJECT_ROOT}/outputs/yolo_baseline_v1_IoU={tag}/eval_results/summary_results.json"
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
            "iou_threshold": iou_for_baseline,
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
    """Load RT-DETR (Ultralytics) checkpoint"""
    checkpoint_path = f"{CHECKPOINTS_DIR}/best.pt"   # adjust if your file name differs
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        exit(1)
    print(f"✅ Loading RT-DETR from: {checkpoint_path}")
    model = RTDETR(checkpoint_path)
    return model

def predict_detections(model, img_pil, conf):
    """Run RT-DETR and return Supervision Detections."""
    results = model.predict(source=img_pil, conf=float(conf), verbose=False)
    det = sv.Detections.from_ultralytics(results[0])
    if len(det.class_id):
        det.class_id = np.zeros(len(det.class_id), dtype=int)  # single class
    return det

# ============================================================================
# DATASET LOADING
# ============================================================================
def load_dataset():
    """Load all test images (both annotated and non-annotated)"""
    if not os.path.exists(ANNOTATIONS_PATH):
        print(f"❌ Annotations not found: {ANNOTATIONS_PATH}")
        exit(1)

    coco_ds = sv.DetectionDataset.from_coco(
        images_directory_path=DATASET_PATH,
        annotations_path=ANNOTATIONS_PATH
    )

    all_test_images = [f for f in os.listdir(DATASET_PATH)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    coco_images = {os.path.basename(path) for path, _, _ in coco_ds}
    non_coco_images = [img for img in all_test_images if img not in coco_images]

    print(f"📊 Dataset composition:")
    print(f"  Annotated images (COCO): {len(coco_ds)}")
    print(f"  Non-annotated images: {len(non_coco_images)}")
    print(f"  Total test images: {len(all_test_images)}")

    combined_dataset = []
    for path, image, annotations in coco_ds:
        combined_dataset.append((path, None, annotations))
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
    """
    Check if predictions have spatial overlap with ground truth
    Uses PyroNear's 0.1 IoU threshold for spatial matching
    """
    if len(predictions.xyxy) == 0 or len(ground_truth.xyxy) == 0:
        return False
    
    # Convert to torch tensors for IoU calculation
    pred_boxes = torch.tensor(predictions.xyxy, dtype=torch.float32)
    gt_boxes = torch.tensor(ground_truth.xyxy, dtype=torch.float32)
    
    # Calculate IoU matrix and check for sufficient overlap
    ious = box_iou(pred_boxes, gt_boxes)
    return (ious > SPATIAL_IOU_THRESHOLD).any().item()

def _object_tp_fp_fn_for_image(preds, annotations, iou_th=SPATIAL_IOU_THRESHOLD):
    """YOLO-style one-to-one matching for a single image."""
    gt = np.array(annotations.xyxy)
    pr = np.array(preds.xyxy)
    tp = fp = fn = 0

    if gt.size == 0 and pr.size == 0:
        return 0, 0, 0

    matched = np.zeros(len(gt), dtype=bool) if len(gt) else np.array([], dtype=bool)

    for pb in pr:
        if len(gt):
            ious = box_iou(
                torch.tensor(pb).unsqueeze(0).float(),
                torch.tensor(gt).float()
            ).squeeze(0)
            if ious.numel() > 0:
                m, idx = torch.max(ious, dim=0)
                idx = int(idx.item())
                if m.item() > iou_th and not matched[idx]:
                    tp += 1
                    matched[idx] = True
                else:
                    fp += 1
            else:
                fp += 1
        else:
            fp += 1

    if len(gt):
        fn += int((~matched).sum())

    return tp, fp, fn


def _counts_at_conf(model, dataset, conf):
    """One sweep: image-level TP/FP/FN/TN + object-level TP/FP/FN."""
    img_tp = img_fp = img_fn = img_tn = 0
    obj_tp = obj_fp = obj_fn = 0

    for path, _, annotations in dataset:
        with Image.open(path) as _img:
            img = _img.convert("RGB")
            preds = predict_detections(model, img, conf)

        # image-level (PyroNear-style spatial check)
        has_smoke_gt = len(annotations.xyxy) > 0
        has_smoke_pred = len(preds.xyxy) > 0
        spatial_match = False
        if has_smoke_gt and has_smoke_pred:
            pred_boxes = torch.tensor(preds.xyxy, dtype=torch.float32)
            gt_boxes   = torch.tensor(annotations.xyxy, dtype=torch.float32)
            ious = box_iou(pred_boxes, gt_boxes)
            spatial_match = (ious > SPATIAL_IOU_THRESHOLD).any().item()

        if has_smoke_gt:
            if spatial_match: img_tp += 1
            else:             img_fn += 1
        else:
            if has_smoke_pred: img_fp += 1
            else:              img_tn += 1

        # object-level
        tpo, fpo, fno = _object_tp_fp_fn_for_image(preds, annotations, SPATIAL_IOU_THRESHOLD)
        obj_tp += tpo; obj_fp += fpo; obj_fn += fno

    return (img_tp, img_fp, img_fn, img_tn), (obj_tp, obj_fp, obj_fn)


def evaluate_model_both_levels(model, dataset):
    """Run image-level and object-level eval across thresholds."""
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

    plt.figure(figsize=(10, 6))
    plt.plot(conf, f1,  label="F1 Score", marker="o")
    plt.plot(conf, prec, label="Precision", linestyle="--")
    plt.plot(conf, rec,  label="Recall", linestyle="-.")
    plt.plot(conf, acc,  label="Accuracy", linestyle=":")
    plt.scatter(best_conf, best_f1, s=150, marker='*', edgecolor="black", zorder=5)
    plt.scatter(best_conf, prec[best_idx], s=80, edgecolor="k", zorder=5)
    plt.scatter(best_conf, rec[best_idx],  s=80, edgecolor="k", zorder=5)
    plt.scatter(best_conf, acc[best_idx],  s=80, edgecolor="k", zorder=5)
    plt.title(f"{EXPERIMENT_NAME} – Image-Level Metrics vs. Confidence Threshold")
    plt.xlabel("Confidence Threshold"); plt.ylabel("Metric Value"); plt.legend(); plt.grid(True)
    out = f"{PLOTS_DIR}/image_metrics.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    return out, img_results[best_idx]


def create_object_metrics_plot(obj_results):
    conf = [r['confidence_threshold'] for r in obj_results]
    f1   = [r['f1_score'] for r in obj_results]
    prec = [r['precision'] for r in obj_results]
    rec  = [r['recall'] for r in obj_results]

    best_idx = int(np.argmax(f1))
    best_conf, best_f1 = conf[best_idx], f1[best_idx]

    plt.figure(figsize=(10, 6))
    plt.plot(conf, f1,  label="F1 Score", marker="o")
    plt.plot(conf, prec, label="Precision", linestyle="--")
    plt.plot(conf, rec,  label="Recall", linestyle="-.")
    plt.scatter(best_conf, best_f1, s=150, marker='*', edgecolor="black", zorder=5)
    plt.scatter(best_conf, prec[best_idx], s=80, edgecolor="k", zorder=5)
    plt.scatter(best_conf, rec[best_idx],  s=80, edgecolor="k", zorder=5)
    plt.title(f"{EXPERIMENT_NAME} – Object-Level Metrics vs. Confidence Threshold")
    plt.xlabel("Confidence Threshold"); plt.ylabel("Metric Value"); plt.legend(); plt.grid(True)
    out = f"{PLOTS_DIR}/object_metrics.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    return out, obj_results[best_idx]


def create_image_confusion_matrix_from_row(img_best_row):
    cm = np.array([[img_best_row["tp"], img_best_row["fn"]],
                   [img_best_row["fp"], img_best_row["tn"]]])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Pred Fire", "Pred No Fire"],
                yticklabels=["GT Fire", "GT No Fire"])
    plt.title(f"{EXPERIMENT_NAME} – Image-Level Confusion Matrix (IoU={SPATIAL_IOU_THRESHOLD})")
    out = f"{PLOTS_DIR}/image_conf_matrix.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    return out


def create_object_confusion_matrix_from_row(obj_best_row):
    cm_numeric = np.array([[obj_best_row["tp"], obj_best_row["fn"]],
                           [obj_best_row["fp"], 0]])
    cm_labels = np.array([[str(obj_best_row["tp"]), str(obj_best_row["fn"])],
                          [str(obj_best_row["fp"]), "N/A"]])

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_numeric, annot=cm_labels, fmt="", cmap="Oranges",
                xticklabels=["Pred Fire", "Pred No Fire"],
                yticklabels=["GT Fire", "GT No Fire"])
    plt.title(f"{EXPERIMENT_NAME} – Object-Level Confusion Matrix (TN = N/A, IoU={SPATIAL_IOU_THRESHOLD})")
    out = f"{PLOTS_DIR}/object_conf_matrix.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    return out

# ============================================================================
# RESULTS SAVING
# ============================================================================
def save_results(img_results, obj_results, yolo_baseline, img_best, obj_best):
    yolo_img = yolo_baseline["image"]["metrics"]
    yolo_obj = yolo_baseline["object"]["metrics"]

    def _pct_increase(new, old):
        return ((new - old) / old * 100.0) if old > 0 else 0.0
    imp_img_pct = _pct_increase(img_best['f1_score'], yolo_img['f1_score'])
    imp_obj_pct = _pct_increase(obj_best['f1_score'], yolo_obj['f1_score'])

    evaluation_data = {
        "experiment_info": {
            "experiment_name": EXPERIMENT_NAME,
            "spatial_iou_threshold": SPATIAL_IOU_THRESHOLD,
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

    results_path = f"{EVAL_RESULTS_DIR}/evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump(evaluation_data, f, indent=2)

    summary_path = f"{EVAL_RESULTS_DIR}/comparison_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("RT-DETR vs YOLO Baseline (Image & Object Levels)\n")
        f.write("="*48 + "\n\n")
        f.write("YOLO Baseline (Image Level):\n")
        f.write(f"  F1: {yolo_img['f1_score']:.4f}  P: {yolo_img['precision']:.4f}  R: {yolo_img['recall']:.4f}  Acc: {yolo_img['accuracy']:.4f}\n")
        f.write("YOLO Baseline (Object Level):\n")
        f.write(f"  F1: {yolo_obj['f1_score']:.4f}  P: {yolo_obj['precision']:.4f}  R: {yolo_obj['recall']:.4f}\n\n")
        f.write("RT-DETR (Image Level):\n")
        f.write(f"  F1: {img_best['f1_score']:.4f}  P: {img_best['precision']:.4f}  R: {img_best['recall']:.4f}  Acc: {img_best['accuracy']:.4f}  @conf={img_best['confidence_threshold']:.2f}\n")
        f.write("RT-DETR (Object Level):\n")
        f.write(f"  F1: {obj_best['f1_score']:.4f}  P: {obj_best['precision']:.4f}  R: {obj_best['recall']:.4f}  @conf={obj_best['confidence_threshold']:.2f}\n\n")
        f.write("Improvements vs YOLO:\n")
        f.write(f"  Image F1: {imp_img_pct:+.1f}%\n")
        f.write(f"  Object F1: {imp_obj_pct:+.1f}%\n")

    print(f"💾 Results saved:\n  📄 {results_path}\n  📊 {summary_path}")
    return img_best


# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    """Main evaluation pipeline"""
    print("🚀 Starting RT-DETR evaluation...")

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
    print(f"   RT-DETR Image F1:  {img_best['f1_score']:.4f} @ conf {img_best['confidence_threshold']:.2f}")
    print(f"   RT-DETR Object F1: {obj_best['f1_score']:.4f} @ conf {obj_best['confidence_threshold']:.2f}")
    print(f"   YOLO Baseline Image F1:  {yolo_baseline['image']['metrics']['f1_score']:.4f}")
    print(f"   YOLO Baseline Object F1: {yolo_baseline['object']['metrics']['f1_score']:.4f}")
    print(f"   Image CM: {img_cm_path}")
    print(f"   Object CM: {obj_cm_path}")


if __name__ == "__main__":
    main()