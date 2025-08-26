import glob
import os
from pyronear_utils import xywh2xyxy, box_iou
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import subprocess
import json
from datetime import datetime
from PIL import Image
import cv2
from tqdm import tqdm
import supervision as sv
import shutil
import seaborn as sns

# Configuration
OUTPUT_BASE_DIR = "/vol/bitbucket/si324/rf-detr-wildfire/src/images/eval_results"
TEST_IMAGES_DIR = "/vol/bitbucket/si324/rf-detr-wildfire/src/images/data/pyro25img/images/test"
GT_FOLDER = "/vol/bitbucket/si324/rf-detr-wildfire/src/images/data/pyro25img/labels/test"


def generate_yolo_predictions(model_path, output_dir):
    """Generate YOLO predictions ONCE at conf=0.01 same as baseline"""
    cmd = f"yolo predict model={model_path} iou=0.01 conf=0.01 source={TEST_IMAGES_DIR} save=False save_txt save_conf project={output_dir} name=predictions"
    print(f"Running: {cmd}")
    subprocess.call(cmd, shell=True)
    return os.path.join(output_dir, "predictions", "labels")

def load_models(model_type, model_path):
    """Load model once"""
    if model_type == "RF-DETR":
        import sys
        sys.path.append('/vol/bitbucket/si324/rf-detr-wildfire/src/images')
        from rfdetr import RFDETRBase
        print(f"Loading RF-DETR from: {model_path}")
        return RFDETRBase(pretrain_weights=model_path, num_classes=1)
    elif model_type == "RT-DETR":
        from ultralytics import RTDETR
        print(f"Loading RT-DETR from: {model_path}")
        return RTDETR(model_path)
    return None

def generate_rfdetr_predictions(model, image_path, conf_threshold):
    """Generate RF-DETR predictions"""
    with Image.open(image_path) as img:
        img_rgb = img.convert("RGB")
        predictions = model.predict(img_rgb, threshold=conf_threshold)
    
    # Convert to YOLO format
    img_width, img_height = img_rgb.size
    yolo_lines = []
    
    if hasattr(predictions, 'xyxy') and len(predictions.xyxy) > 0:
        for i, box in enumerate(predictions.xyxy):
            x1, y1, x2, y2 = box
            x_center = ((x1 + x2) / 2) / img_width
            y_center = ((y1 + y2) / 2) / img_height
            width = (x2 - x1) / img_width
            height = (y2 - y1) / img_height
            conf = predictions.confidence[i] if hasattr(predictions, 'confidence') else 1.0
            yolo_lines.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {conf:.6f}")
    
    return yolo_lines

def generate_rtdetr_predictions(model, image_path, conf_threshold):
    """Generate RT-DETR predictions"""
    with Image.open(image_path) as img:
        img_rgb = img.convert("RGB")
        results = model.predict(source=img_rgb, conf=conf_threshold, verbose=False)
        predictions = sv.Detections.from_ultralytics(results[0])
    
    # Convert to YOLO format
    img_width, img_height = img_rgb.size
    yolo_lines = []
    
    if hasattr(predictions, 'xyxy') and len(predictions.xyxy) > 0:
        for i, box in enumerate(predictions.xyxy):
            x1, y1, x2, y2 = box
            x_center = ((x1 + x2) / 2) / img_width
            y_center = ((y1 + y2) / 2) / img_height
            width = (x2 - x1) / img_width
            height = (y2 - y1) / img_height
            conf = predictions.confidence[i] if hasattr(predictions, 'confidence') else 1.0
            yolo_lines.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {conf:.6f}")
    
    return yolo_lines

def evaluate_predictions(pred_folder, gt_folder, conf_th=0.1, cat=None):
    nb_fp, nb_tp, nb_fn = 0, 0, 0

    gt_filenames = [
        os.path.splitext(os.path.basename(f))[0]
        for f in glob.glob(os.path.join(gt_folder, "*.txt"))
    ]
    pred_filenames = [
        os.path.splitext(os.path.basename(f))[0]
        for f in glob.glob(os.path.join(pred_folder, "*.txt"))
    ]

    all_filenames = set(gt_filenames + pred_filenames)
    if cat is not None:
        all_filenames = [f for f in all_filenames if cat == f.split("_")[0].lower()]

    for filename in all_filenames:
        gt_file = os.path.join(gt_folder, f"{filename}.txt")
        pred_file = os.path.join(pred_folder, f"{filename}.txt")

        gt_boxes = []
        if os.path.isfile(gt_file) and os.path.getsize(gt_file) > 0:
            with open(gt_file, "r") as f:
                gt_boxes = [
                    xywh2xyxy(np.array(line.strip().split(" ")[1:5]).astype(float))
                    for line in f.readlines()
                ]

        gt_matches = np.zeros(len(gt_boxes), dtype=bool)

        if os.path.isfile(pred_file) and os.path.getsize(pred_file) > 0:
            with open(pred_file, "r") as f:
                pred_boxes = [line.strip().split(" ") for line in f.readlines()]

            for pred_box in pred_boxes:
                try:
                    _, x, y, w, h, conf = map(float, pred_box)
                except:
                    print(f"Error reading {pred_file}")
                    continue
                if conf < conf_th:
                    continue
                pred_box = xywh2xyxy(np.array([x, y, w, h]))

                if gt_boxes:
                    # Find the best match by IoU
                    iou_values = [box_iou(pred_box, gt_box) for gt_box in gt_boxes]
                    max_iou = max(iou_values)
                    best_match_idx = np.argmax(iou_values)

                    # Check for valid and unique match
                    if max_iou > 0.1 and not gt_matches[best_match_idx]:
                        nb_tp += 1
                        gt_matches[best_match_idx] = True
                    else:
                        nb_fp += 1
                else:
                    nb_fp += 1

        if gt_boxes:
            nb_fn += len(gt_boxes) - np.sum(gt_matches)

    precision = nb_tp / (nb_tp + nb_fp) if (nb_tp + nb_fp) > 0 else 0
    recall = nb_tp / (nb_tp + nb_fn) if (nb_tp + nb_fn) > 0 else 0
    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    return {
        "precision": precision, 
        "recall": recall, 
        "f1_score": f1_score,
        "tp": nb_tp,
        "fp": nb_fp, 
        "fn": nb_fn
    }

def save_predictions_by_threshold(pred_dir, output_base_dir, conf_threshold):
    """Save filtered predictions at each threshold for inspection"""
    threshold_dir = os.path.join(output_base_dir, "preds_by_threshold", f"conf_{conf_threshold:.2f}")
    os.makedirs(threshold_dir, exist_ok=True)
    
    # Get all prediction files
    pred_files = glob.glob(os.path.join(pred_dir, "*.txt"))
    
    for pred_file in pred_files:
        filename = os.path.basename(pred_file)
        output_file = os.path.join(threshold_dir, filename)
        
        filtered_lines = []
        if os.path.exists(pred_file) and os.path.getsize(pred_file) > 0:
            with open(pred_file, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) >= 6:
                        conf = float(parts[5])
                        if conf >= conf_threshold:
                            filtered_lines.append(line.strip())
        
        # Write filtered predictions (or empty file if no predictions above threshold)
        with open(output_file, 'w') as f:
            if filtered_lines:
                f.write('\n'.join(filtered_lines))


def evaluate_model(model_name, model_type, model_path, conf_thres_range):
    """Evaluate model - generate predictions ONCE at conf=0.01, filter during eval"""
    print(f"\n{'='*60}")
    print(f"Evaluating {model_name}")
    print(f"{'='*60}")
    
    # Setup directories
    model_output_dir = os.path.join(OUTPUT_BASE_DIR, model_name)
    
    if model_type == "YOLO":
        pred_dir_name = "YOLO_predictions"
    elif model_type == "RF-DETR":
        pred_dir_name = "RF-DETR_preds_yolo_format"
    else:
        pred_dir_name = "RT-DETR_preds_yolo_format"
    
    pred_base_dir = os.path.join(model_output_dir, pred_dir_name)
    plots_dir = os.path.join(model_output_dir, "plots")
    
    os.makedirs(pred_base_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    # Generate predictions ONCE at conf=0.01
    print(f"Generating predictions at conf=0.01...")
    
    if model_type == "YOLO":
        pred_dir = generate_yolo_predictions(model_path, pred_base_dir)
    else:
        pred_dir = os.path.join(pred_base_dir, "predictions")
        os.makedirs(pred_dir, exist_ok=True)
        
        # Load model once
        model = load_models(model_type, model_path)
        
        # Get all test images
        test_images = glob.glob(os.path.join(TEST_IMAGES_DIR, "*.jpg"))
        test_images.extend(glob.glob(os.path.join(TEST_IMAGES_DIR, "*.png")))
        
        for img_path in tqdm(test_images, desc="Generating predictions"):
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            pred_file = os.path.join(pred_dir, f"{img_name}.txt")
            
            # Generate at conf=0.01 to match YOLO baseline
            if model_type == "RF-DETR":
                yolo_lines = generate_rfdetr_predictions(model, img_path, 0.01)
            else:
                yolo_lines = generate_rtdetr_predictions(model, img_path, 0.01)
            
            with open(pred_file, 'w') as f:
                if yolo_lines:
                    f.write('\n'.join(yolo_lines))

    # Evaluate at different thresholds 
    all_results = []
    best_results = None
    best_conf = 0
    
    for conf_threshold in tqdm(conf_thres_range, desc="Evaluating thresholds"):
        results = evaluate_predictions(pred_dir, GT_FOLDER, conf_threshold)
        results['confidence_threshold'] = conf_threshold
        all_results.append(results)

        # Save filtered predictions at this threshold
        save_predictions_by_threshold(pred_dir, pred_base_dir, conf_threshold)
        
        if results['f1_score'] > (best_results['f1_score'] if best_results else 0):
            best_results = results
            best_conf = conf_threshold
    
    # Save results
    save_results(best_results, best_conf, all_results, model_output_dir, model_name, plots_dir)

    # Create confusion matrix 
    if 'tp' in best_results:
        cm_path = create_object_confusion_matrix(best_results, model_name, plots_dir)
        print(f"Confusion matrix saved: {cm_path}")

    # Generate bounding box visualizations at best threshold
    generate_bounding_boxes(model_name, model_type, pred_dir, best_conf)

    # Create visualization summary
    create_visualization_summary(model_name, model_type, pred_dir, best_conf)
    
    return best_results, best_conf

def save_results(best_results, best_conf, all_results, output_dir, model_name, plots_dir):
    """Save evaluation results with plots and summaries"""
    
    # Generate plot
    conf_thresholds = [r['confidence_threshold'] for r in all_results]
    f1_scores = [r['f1_score'] for r in all_results]
    precisions = [r['precision'] for r in all_results]
    recalls = [r['recall'] for r in all_results]
    
    plt.figure(figsize=(10, 6))
    plt.plot(conf_thresholds, f1_scores, label="F1 Score", color="blue", marker="o")
    plt.plot(conf_thresholds, precisions, label="Precision", color="green", linestyle="--")
    plt.plot(conf_thresholds, recalls, label="Recall", color="red", linestyle="-.")
    
    plt.scatter(best_conf, best_results['f1_score'], color="blue", s=150, marker = '*', edgecolor="black", zorder=6)
    plt.scatter(best_conf, best_results['precision'], color="green", s=100, edgecolor="k", zorder=5)
    plt.scatter(best_conf, best_results['recall'], color="red", s=100, edgecolor="k", zorder=5)
    
    plt.text(best_conf + 0.02, best_results['f1_score'],
            f"Best F1: {best_results['f1_score']:.2f}",
            fontsize=9, ha='left', va='center', color='blue', weight='bold')
    
    plt.text(best_conf + 0.02, best_results['precision'],
            f"Precision: {best_results['precision']:.2f}",
            fontsize=9, ha='left', va='center', color='green', weight='bold')
    
    plt.text(best_conf + 0.02, best_results['recall'],
            f"Recall: {best_results['recall']:.2f}",
            fontsize=9, ha='left', va='center', color='red', weight='bold')
    
    plt.title(f"{model_name}: F1 Score, Precision, and Recall vs. Confidence Threshold")
    plt.xlabel("Confidence Threshold")
    plt.ylabel("Metric Value")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, "metrics.png"))
    plt.close()
    
    # Save JSON summary
    summary = {
        "model_name": model_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "best_confidence_threshold": float(best_conf),
        "best_results": {
            "f1_score": float(best_results['f1_score']),
            "precision": float(best_results['precision']),
            "recall": float(best_results['recall']),
            "tp": int(best_results['tp']) if 'tp' in best_results else None,
            "fp": int(best_results['fp']) if 'fp' in best_results else None,
            "fn": int(best_results['fn']) if 'fn' in best_results else None
        },
        "all_results": [{
            "confidence_threshold": float(r['confidence_threshold']),
            "f1_score": float(r['f1_score']),
            "precision": float(r['precision']),
            "recall": float(r['recall']),
            "tp": int(r['tp']) if 'tp' in r else None,
            "fp": int(r['fp']) if 'fp' in r else None,
            "fn": int(r['fn']) if 'fn' in r else None
        } for r in all_results]
    }
    
    with open(os.path.join(output_dir, "evaluation_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save TXT summary
    with open(os.path.join(output_dir, "evaluation_summary.txt"), 'w') as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"{'='*50}\n")
        f.write(f"Best Confidence Threshold: {best_conf:.3f}\n")
        f.write(f"Best F1 Score: {best_results['f1_score']:.4f}\n")
        f.write(f"Best Precision: {best_results['precision']:.4f}\n")
        f.write(f"Best Recall: {best_results['recall']:.4f}\n")
        if 'tp' in best_results:
            f.write(f"\nConfusion Matrix:\n")
            f.write(f"TP: {int(best_results['tp'])}\n")
            f.write(f"FP: {int(best_results['fp'])}\n")
            f.write(f"FN: {int(best_results['fn'])}\n")
            f.write(f"TN: N/A (not applicable for object detection)\n")

def create_object_confusion_matrix(best_results, model_name, plots_dir):
    """Create object-level confusion matrix visualization"""
    # Check if we have the counts
    if 'tp' not in best_results or 'fp' not in best_results or 'fn' not in best_results:
        print("Warning: No counts available for confusion matrix")
        return None
    
    # Create numeric matrix for heatmap
    cm_numeric = np.array([[best_results["tp"], best_results["fn"]],
                          [best_results["fp"], 0]])
    
    # Create annotation matrix with N/A
    cm_labels = np.array([[str(int(best_results["tp"])), str(int(best_results["fn"]))],
                         [str(int(best_results["fp"])), "N/A"]])
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_numeric, annot=cm_labels, fmt="", cmap="Oranges",
                xticklabels=["Pred Fire", "Pred No Fire"],
                yticklabels=["GT Fire", "GT No Fire"])
    plt.title(f"{model_name} – Object-Level Confusion Matrix (TN = N/A, IoU=0.1)")
    out = os.path.join(plots_dir, "object_conf_matrix.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    return out

def classify_image_boxes(pred_file, gt_file, conf_threshold):
    """Mirror exact logic from evaluate_predictions to classify boxes"""
    # Load GT boxes
    gt_boxes = []
    if os.path.isfile(gt_file) and os.path.getsize(gt_file) > 0:
        with open(gt_file, "r") as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    _, x, y, w, h = map(float, parts[:5])
                    gt_boxes.append(xywh2xyxy(np.array([x, y, w, h])))
    
    # Load predictions above threshold with their original data
    pred_data = []
    if os.path.isfile(pred_file) and os.path.getsize(pred_file) > 0:
        with open(pred_file, "r") as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) >= 6:
                    _, x, y, w, h, conf = map(float, parts)
                    if conf >= conf_threshold:
                        pred_box = xywh2xyxy(np.array([x, y, w, h]))
                        pred_data.append({'box': pred_box, 'conf': conf, 'type': None})
    
    # Match predictions to GT 
    gt_matched = [False] * len(gt_boxes)
    
    for pred in pred_data:
        if gt_boxes:
            iou_values = [box_iou(pred['box'], gt_box) for gt_box in gt_boxes]
            max_iou = max(iou_values)
            best_match_idx = np.argmax(iou_values)

            pred['iou'] = max_iou  # Store the IoU value
            
            if max_iou > 0.1 and not gt_matched[best_match_idx]:
                pred['type'] = 'TP'
                gt_matched[best_match_idx] = True
            else:
                pred['type'] = 'FP'
        else:
            pred['type'] = 'FP'
            pred['iou'] = 0.0  # No GT boxes, so IoU is 0
    
    # Determine image classification for folder placement
    has_tp = any(p['type'] == 'TP' for p in pred_data)
    has_fp = any(p['type'] == 'FP' for p in pred_data)
    has_fn = any(not matched for matched in gt_matched)
    
    # Return both the detailed box classifications and overall image classification
    return pred_data, gt_matched, (has_tp, has_fp, has_fn)

def generate_bounding_boxes(model_name, model_type, pred_dir, best_conf):
    """Generate bounding box visualizations at best threshold"""
    print(f"Generating bounding boxes for {model_name} at conf={best_conf:.3f}")
    
    bbox_dir = os.path.join(OUTPUT_BASE_DIR, model_name, f"predicted_bounding_boxes_{model_type}")
    for cls in ['TP', 'FP', 'FN']: 
        os.makedirs(os.path.join(bbox_dir, cls), exist_ok=True)
    
    test_images = glob.glob(os.path.join(TEST_IMAGES_DIR, "*.jpg"))
    test_images.extend(glob.glob(os.path.join(TEST_IMAGES_DIR, "*.png")))
    
    for img_path in tqdm(test_images, desc="Drawing bounding boxes"):
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        gt_file = os.path.join(GT_FOLDER, f"{img_name}.txt")
        pred_file = os.path.join(pred_dir, f"{img_name}.txt")
        
        # Get detailed box classifications
        box_classifications = classify_image_boxes(pred_file, gt_file, best_conf)
        pred_data, gt_matched, (has_tp, has_fp, has_fn) = box_classifications
        
        # Skip if no detections or GT
        if not has_tp and not has_fp and not has_fn:
            continue
        
        # Count each type
        tp_boxes = [i for i, p in enumerate(pred_data) if p['type'] == 'TP']
        fp_boxes = [i for i, p in enumerate(pred_data) if p['type'] == 'FP']
        fn_indices = [i for i, matched in enumerate(gt_matched) if not matched]

        # Save one image per detection
        for idx, tp_idx in enumerate(tp_boxes, 1):
            output_path = os.path.join(bbox_dir, "TP", f"{img_name}_TP_{idx}.jpg")
            draw_bounding_boxes(img_path, pred_file, gt_file, box_classifications, 
                            output_path, model_name, best_conf, "TP", highlight_idx=tp_idx)

        for idx, fp_idx in enumerate(fp_boxes, 1):
            output_path = os.path.join(bbox_dir, "FP", f"{img_name}_FP_{idx}.jpg")
            draw_bounding_boxes(img_path, pred_file, gt_file, box_classifications, 
                            output_path, model_name, best_conf, "FP", highlight_idx=fp_idx)

        for idx, fn_idx in enumerate(fn_indices, 1):
            output_path = os.path.join(bbox_dir, "FN", f"{img_name}_FN_{idx}.jpg")
            draw_bounding_boxes(img_path, pred_file, gt_file, box_classifications, 
                            output_path, model_name, best_conf, "FN", highlight_idx=fn_idx)
        

def create_visualization_summary(model_name, model_type, pred_dir, best_conf):
    """Create a summary file for all bounding box folders"""
    bbox_dir = os.path.join(OUTPUT_BASE_DIR, model_name, f"predicted_bounding_boxes_{model_type}")
    
    summary = {
        "model_name": model_name,
        "best_confidence_threshold": float(best_conf),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "folder_contents": {}
    }
    
    # Count detections in each folder
    for folder_type in ['TP', 'FP', 'FN']:
        folder_path = os.path.join(bbox_dir, folder_type)
        images_in_folder = glob.glob(os.path.join(folder_path, "*.jpg"))
        
        tp_count = fp_count = fn_count = 0
        
        for img_path in images_in_folder:
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            gt_file = os.path.join(GT_FOLDER, f"{img_name}.txt")
            pred_file = os.path.join(pred_dir, f"{img_name}.txt")
            
            box_classifications = classify_image_boxes(pred_file, gt_file, best_conf)
            pred_data, gt_matched, (has_tp, has_fp, has_fn) = box_classifications
            
            if has_tp:
                tp_count += sum(1 for p in pred_data if p['type'] == 'TP')
            if has_fp:
                fp_count += sum(1 for p in pred_data if p['type'] == 'FP')
            if has_fn:
                fn_count += sum(1 for matched in gt_matched if not matched)
        
        summary["folder_contents"][folder_type] = {
            "num_images": len(images_in_folder),
            "total_tp_boxes": tp_count,
            "total_fp_boxes": fp_count,
            "total_fn_boxes": fn_count,
            "note": f"Images in this folder contain at least one {folder_type} detection"
        }
    
    # Save summary as JSON
    summary_path = os.path.join(bbox_dir, "visualization_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Also save as text for easy reading
    summary_txt_path = os.path.join(bbox_dir, "visualization_summary.txt")
    with open(summary_txt_path, 'w') as f:
        f.write(f"Visualization Summary for {model_name}\n")
        f.write(f"{'='*60}\n")
        f.write(f"Generated: {summary['timestamp']}\n")
        f.write(f"Best confidence threshold: {best_conf:.3f}\n\n")
        
        for folder, data in summary["folder_contents"].items():
            f.write(f"{folder} Folder:\n")
            f.write(f"  Images: {data['num_images']}\n")
            f.write(f"  Total TP boxes: {data['total_tp_boxes']}\n")
            f.write(f"  Total FP boxes: {data['total_fp_boxes']}\n")
            f.write(f"  Total FN boxes: {data['total_fn_boxes']}\n")
            f.write(f"  Note: {data['note']}\n\n")
    
    print(f"Visualization summary saved: {summary_path}")

def draw_bounding_boxes(image_path, pred_file, gt_file, box_classifications, 
                        output_path, model_name, conf_threshold, folder_type, highlight_idx=None):
    """Draw bounding boxes with colors based on the TP/FP classification"""
    image = cv2.imread(image_path)
    if image is None:
        return
    
    img_height, img_width = image.shape[:2]
    
    pred_data, gt_matched, (has_tp, has_fp, has_fn) = box_classifications
    
    # Draw GT boxes with different colors for matched (TP) vs unmatched (FN)
    if os.path.isfile(gt_file) and os.path.getsize(gt_file) > 0:
        with open(gt_file, "r") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                parts = line.strip().split()
                if len(parts) >= 5:
                    _, x, y, w, h = map(float, parts[:5])
                    x1 = int((x - w/2) * img_width)
                    y1 = int((y - h/2) * img_height)
                    x2 = int((x + w/2) * img_width)
                    y2 = int((y + h/2) * img_height)
                    
                    # Yellow for matched GT (part of TP), cyan for unmatched GT (FN)
                    if i < len(gt_matched) and gt_matched[i]:
                        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 2)  # Yellow
                        cv2.putText(image, "GT-matched", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    else:
                        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 0), 2)  # Cyan for FN
                        cv2.putText(image, "GT-missed", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    # Draw predictions with their actual TP/FP classification
    if os.path.isfile(pred_file) and os.path.getsize(pred_file) > 0:
        with open(pred_file, "r") as f:
            pred_idx = 0
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) >= 6:
                    _, x, y, w, h, conf = map(float, parts)
                    if conf >= conf_threshold:
                        x1 = int((x - w/2) * img_width)
                        y1 = int((y - h/2) * img_height)
                        x2 = int((x + w/2) * img_width)
                        y2 = int((y + h/2) * img_height)
                        
                        # Color based on actual classification
                        if pred_idx < len(pred_data):
                            # Highlight the prediction box that the image is focused on
                            thickness = 4 if (pred_idx == highlight_idx and folder_type in ['TP', 'FP']) else 2
                            box_type = pred_data[pred_idx]['type']
                            if box_type == 'TP':
                                color = (0, 255, 0)  # Green
                            else:  # FP
                                color = (0, 0, 255)  # Red
                            
                            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
                            iou_val = pred_data[pred_idx]['iou'] if 'iou' in pred_data[pred_idx] else 0.0
                            label = f"{box_type}: IoU={float(iou_val):.2f}"
                            cv2.putText(image, label, (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        pred_idx += 1
    
    # Add header with all info
    header = f"{model_name} | Contains: "
    types = []
    if has_tp: types.append("TP")
    if has_fp: types.append("FP")  
    if has_fn: types.append("FN")
    header += "/".join(types) + f" | Saved in: {folder_type}"
    
    cv2.rectangle(image, (0, 0), (img_width, 35), (0, 0, 0), -1)
    cv2.putText(image, header, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imwrite(output_path, image)

def find_best_conf_threshold(pred_folder, gt_folder, conf_thres_range, cat=None):
    best_conf_thres = 0
    best_f1_score = 0
    best_precision = 0
    best_recall = 0

    for conf_thres in conf_thres_range:
        results = evaluate_predictions(pred_folder, gt_folder, conf_thres, cat)
        if results["f1_score"] > best_f1_score:
            best_conf_thres = conf_thres
            best_f1_score = results["f1_score"]
            best_precision = results["precision"]
            best_recall = results["recall"]

    return best_conf_thres, best_f1_score, best_precision, best_recall


def evaluate_multiple_pred_folders(pred_folders, gt_folder, conf_thres_range, cat=None):
    # Initialize a DataFrame to store the results
    results_df = pd.DataFrame(
        columns=[
            "Prediction Folder",
            "Best Threshold",
            "Best F1 Score",
            "Precision",
            "Recall",
        ]
    )

    for pred_folder in pred_folders:
        best_conf_thres, best_f1_score, best_precision, best_recall = (
            find_best_conf_threshold(pred_folder, gt_folder, conf_thres_range, cat)
        )

        # Use loc to append data to the DataFrame to avoid potential issues
        results_df.loc[len(results_df.index)] = [
            pred_folder.split("/")[7],
            best_conf_thres,
            best_f1_score,
            best_precision,
            best_recall,
        ]

    return results_df


def find_best_conf_threshold_and_plot(
    pred_folder, gt_folder, conf_thres_range, plot=True
):
    f1_scores, precisions, recalls = [], [], []

    for conf_thres in conf_thres_range:
        results = evaluate_predictions(pred_folder, gt_folder, conf_thres)
        f1_scores.append(results["f1_score"])
        precisions.append(results["precision"])
        recalls.append(results["recall"])

    # Find the best confidence threshold
    best_idx = np.argmax(f1_scores)
    best_conf_thres = conf_thres_range[best_idx]
    best_f1_score = f1_scores[best_idx]
    best_precision = precisions[best_idx]
    best_recall = recalls[best_idx]
    # save 
    # save the best recall, precision and f1 score
    
    np.save(f"{pred_folder}/f1_scores.npy", f1_scores)
    np.save(f"{pred_folder}/precisions.npy", precisions)
    np.save(f"{pred_folder}/recalls.npy", recalls)
    np.save(f"{pred_folder}/conf_thres.npy", conf_thres_range)

    if plot:

        # Plotting the metrics
        plt.figure(figsize=(10, 6))
        plt.plot(
            conf_thres_range, f1_scores, label="F1 Score", color="blue", marker="o"
        )
        plt.plot(
            conf_thres_range,
            precisions,
            label="Precision",
            color="green",
            linestyle="--",
        )
        plt.plot(conf_thres_range, recalls, label="Recall", color="red", linestyle="-.")

        # Highlight the best configuration
        plt.scatter(
            best_conf_thres, best_f1_score, color="blue", s=100, edgecolor="k", zorder=5
        )
        plt.scatter(
            best_conf_thres,
            best_precision,
            color="green",
            s=100,
            edgecolor="k",
            zorder=5,
        )
        plt.scatter(
            best_conf_thres, best_recall, color="red", s=100, edgecolor="k", zorder=5
        )

        plt.text(
            best_conf_thres,
            best_f1_score,
            f" Best F1: {best_f1_score:.2f}\n Precision: {best_precision:.2f}\n Recall: {best_recall:.2f}",
            fontsize=9,
            verticalalignment="bottom",
        )

        plt.title("F1 Score, Precision, and Recall vs. Confidence Threshold")
        plt.xlabel("Confidence Threshold")
        plt.ylabel("Metric Value")
        plt.legend()
        plt.grid(True)
        # save in predictions folder
        plt.savefig(f"{pred_folder}/metrics.png")
        # save the list

        plt.show()

    return best_conf_thres, best_f1_score, best_precision, best_recall


def create_comparison_visualizations(all_model_results, all_results_data, output_dir):
    """Create comparison plots for all models"""
    
    # Color schemes - shades of same color per model
    colors = {
        'YOLO_baseline': {'main': '#2E7D32', 'f1': '#388E3C', 'precision': '#4CAF50', 'recall': '#66BB6A'},
        'RF-DETR_initial_training': {'main': '#6A1B9A', 'f1': '#7B1FA2', 'precision': '#8E24AA', 'recall': '#9C27B0'},
        'RT-DETR_initial_training': {'main': '#1565C0', 'f1': '#1976D2', 'precision': '#1E88E5', 'recall': '#2196F3'}
    }
    model_labels = ['YOLO', 'RF-DETR', 'RT-DETR']
    models = list(all_model_results.keys())
    
    # 1. GROUPED BAR CHART - F1, Precision, Recall
    fig, ax = plt.subplots(figsize=(12, 7))
    metrics = ['F1 Score', 'Precision', 'Recall']
    metric_keys = ['best_f1', 'precision', 'recall']
    color_keys = ['f1', 'precision', 'recall']
    x = np.arange(len(models))
    width = 0.25
    
    for i, metric in enumerate(metric_keys):
        values = [all_model_results[m][metric] for m in models]
        positions = x + i * width
        bars = ax.bar(positions, values, width, label=metrics[i])
        
        # Apply gradient colors per model
        for j, bar in enumerate(bars):
            bar.set_color(colors[models[j]][color_keys[i]])
            conf = all_model_results[models[j]]['best_conf']
            height = bar.get_height()
            # Only show conf on F1 bars
            if i == 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                       f'{height:.3f}\n(τ={conf:.2f})', 
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
            else:
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                       f'{height:.3f}', 
                       ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Models', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(model_labels)
    ax.legend(loc='upper right', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.set_ylim([0, 1.08])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_comparison_bars.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. PR CURVES - Clean without isolines
    fig, ax = plt.subplots(figsize=(11, 8))

    for i, (model_name, results_list) in enumerate(all_results_data.items()):
        precisions = [r['precision'] for r in results_list]
        recalls = [r['recall'] for r in results_list]
        
        sorted_indices = np.argsort(recalls)
        recalls_sorted = [recalls[j] for j in sorted_indices]
        precisions_sorted = [precisions[j] for j in sorted_indices]
        
        # Main curve
        ax.plot(recalls_sorted, precisions_sorted, 
                label=model_labels[i], color=colors[model_name]['main'], 
                linewidth=2.5, alpha=0.9)
        
        # Best F1 point
        best_idx = np.argmax([r['f1_score'] for r in results_list])
        ax.scatter(results_list[best_idx]['recall'], 
                results_list[best_idx]['precision'],
                s=300, color=colors[model_name]['main'], marker='*', 
                edgecolors='white', linewidth=2.5, zorder=5,
                label=f'{model_labels[i]} Best (F1={results_list[best_idx]["f1_score"]:.3f})')

    ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax.set_title('Precision-Recall Curves', fontsize=14, fontweight='bold')
    ax.legend(loc='lower left', frameon=True, shadow=True, fontsize=10)
    ax.grid(True, alpha=0.25, linestyle=':', linewidth=0.5)
    ax.set_xlim([0, 1.02])
    ax.set_ylim([0, 1.02])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pr_curves.png"), dpi=300, bbox_inches='tight')
    plt.close()
        
    # 3. DETECTION BREAKDOWN - Grouped bars
    fig, ax = plt.subplots(figsize=(12, 7))

    tp_values = [all_model_results[m].get('tp', 0) for m in models]
    fp_values = [all_model_results[m].get('fp', 0) for m in models]
    fn_values = [all_model_results[m].get('fn', 0) for m in models]

    x = np.arange(len(model_labels))
    width = 0.25

    # Grouped bars
    bars1 = ax.bar(x - width, tp_values, width, label='True Positives', color='#2ECC71')
    bars2 = ax.bar(x, fp_values, width, label='False Positives', color='#E74C3C')
    bars3 = ax.bar(x + width, fn_values, width, label='False Negatives', color='#3498DB')

    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 3,
                    f'{int(height)}', ha='center', va='bottom', fontweight='bold')

    ax.set_xlabel('Models', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Detections', fontsize=12, fontweight='bold')
    ax.set_title('Detection Classification Breakdown', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "detection_breakdown.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison visualizations saved to {output_dir}")

if __name__ == "__main__":
    # Model configurations
    models = {
        "YOLO_baseline": {
            "type": "YOLO",
            "path": "/vol/bitbucket/si324/rf-detr-wildfire/src/images/outputs/YOLO_baseline/training_outputs/eager-flower-1/weights/best.pt"
        },
        "RF-DETR_initial_training": {
            "type": "RF-DETR",
            "path": "/vol/bitbucket/si324/rf-detr-wildfire/src/images/outputs/RF-DETR_initial_training/checkpoints/checkpoint_best_ema.pth"
        },
        "RT-DETR_initial_training": {
            "type": "RT-DETR",
            "path": "/vol/bitbucket/si324/rf-detr-wildfire/src/images/outputs/RT-DETR_initial_training/checkpoints/weights/best.pt"
        }
    }
    
    conf_range = np.arange(0.1, 0.9, 0.05)
    all_model_results = {}
    all_results_data = {}
    
    for model_name, config in models.items():
        best_results, best_conf = evaluate_model(
            model_name, 
            config['type'],
            config['path'],
            conf_range
        )
        
        all_model_results[model_name] = {
            "best_f1": best_results['f1_score'],
            "precision": best_results['precision'],
            "recall": best_results['recall'],
            "best_conf": best_conf,
            "tp": best_results.get('tp', 0),  
            "fp": best_results.get('fp', 0),
            "fn": best_results.get('fn', 0)
        }

        # Load the saved results for PR curves
        summary_path = os.path.join(OUTPUT_BASE_DIR, model_name, "evaluation_summary.json")
        with open(summary_path, 'r') as f:
            summary_data = json.load(f)
            all_results_data[model_name] = summary_data['all_results']

    # Create comparison visualizations
    create_comparison_visualizations(all_model_results, all_results_data, OUTPUT_BASE_DIR)
    
    # Save final comparison
    comparison = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": all_model_results
    }
    
    with open(os.path.join(OUTPUT_BASE_DIR, "final_comparison_summary.json"), 'w') as f:
        json.dump(comparison, f, indent=2)
    
    with open(os.path.join(OUTPUT_BASE_DIR, "final_comparison_summary.txt"), 'w') as f:
        f.write("FINAL MODEL COMPARISON\n")
        f.write("="*60 + "\n\n")
        
        for model_name, results in all_model_results.items():
            f.write(f"{model_name}:\n")
            f.write(f"  Best F1: {results['best_f1']:.4f} @ conf={results['best_conf']:.3f}\n")
            f.write(f"  Precision: {results['precision']:.4f}\n")
            f.write(f"  Recall: {results['recall']:.4f}\n\n")
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("Results saved to:", OUTPUT_BASE_DIR)
    print("\nFinal Results:")
    for model_name, results in all_model_results.items():
        print(f"\n{model_name}:")
        print(f"  F1: {results['best_f1']:.4f} @ conf={results['best_conf']:.3f}")