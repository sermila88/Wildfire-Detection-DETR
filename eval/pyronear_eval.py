import glob
import os
from pyronear_utils import xywh2xyxy, box_iou
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

EXPERIMENT_NAME = "yolo_baseline_v1_IoU=0.1"  # <-- change this per experiment
EVAL_DIR = f"outputs/{EXPERIMENT_NAME}/eval_results"
PLOT_DIR = f"outputs/{EXPERIMENT_NAME}/plots"
os.makedirs(EVAL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

def evaluate_predictions(pred_folder, gt_folder, conf_th=0.1, iou_th=0.1, cat=None):
    # For object-level tracking:
    obj_fp, obj_tp, obj_fn = 0, 0, 0 

    # For image-level tracking:
    img_tp, img_fp, img_fn, img_tn = 0, 0, 0, 0

    # Get all test images (including non-annotated images)
    test_images_dir = "/vol/bitbucket/si324/rf-detr-wildfire/data/pyro25img/images/test"
    # Get all image files
    all_image_files = [f for f in os.listdir(test_images_dir) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    all_filenames = [os.path.splitext(f)[0] for f in all_image_files]
    

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

        # Object-level classification
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
                    
                    # Check for a valid and unique match
                    if max_iou > iou_th and not gt_matches[best_match_idx]:
                        obj_tp += 1
                        gt_matches[best_match_idx] = True
                    else:
                        # Either IoU too low OR GT already matched
                        obj_fp += 1
                else:
                     # No GT boxes in image, so any prediction is wrong
                    obj_fp += 1

        # After processing all predictions, count unmatched GT boxes
        if gt_boxes:
            obj_fn += len(gt_boxes) - np.sum(gt_matches)

        # Check if ground truth has any smoke boxes in this image
        # True if GT file contains smoke annotations, False if empty/no smoke
        has_smoke_gt = len(gt_boxes) > 0
        # Check if prediction file exists and has any predictions
        has_smoke_pred = os.path.isfile(pred_file) and os.path.getsize(pred_file) > 0
        
        # Check if any prediction above threshold overlaps with GT
        spatial_match = False
        if has_smoke_gt and has_smoke_pred:
            spatial_match = np.sum(gt_matches) > 0  # Use existing gt_matches
        
        # Image-level counting
        if has_smoke_gt:
            if spatial_match:
                img_tp += 1
            else:
                img_fn += 1
        else:
            if has_smoke_pred:
                # Check if any pred above threshold
                has_valid_pred = False
                if os.path.isfile(pred_file) and os.path.getsize(pred_file) > 0:
                    with open(pred_file, "r") as f:
                        for line in f.readlines():
                            try:
                                conf = float(line.strip().split()[5])
                                if conf >= conf_th:
                                    has_valid_pred = True
                                    break
                            except:
                                continue
                if has_valid_pred:
                    img_fp += 1
                else:
                    img_tn += 1
            else:
                img_tn += 1

    # Calculate OBJECT-LEVEL metrics 
    obj_precision = obj_tp / (obj_tp + obj_fp) if (obj_tp + obj_fp) > 0 else 0
    obj_recall = obj_tp / (obj_tp + obj_fn) if (obj_tp + obj_fn) > 0 else 0
    obj_f1_score = (
        2 * (obj_precision * obj_recall) / (obj_precision + obj_recall)
        if (obj_precision + obj_recall) > 0
        else 0
    )

    # Calculate IMAGE-LEVEL metrics 
    img_precision = img_tp / (img_tp + img_fp) if (img_tp + img_fp) > 0 else 0
    img_recall = img_tp / (img_tp + img_fn) if (img_tp + img_fn) > 0 else 0  
    img_f1_score = 2 * (img_precision * img_recall) / (img_precision + img_recall) if (img_precision + img_recall) > 0 else 0
    img_accuracy = (img_tp + img_tn) / len(all_filenames) if len(all_filenames) > 0 else 0

    return { #Object-level metrics 
            "obj_precision": obj_precision, 
            "obj_recall": obj_recall, 
            "obj_f1_score": obj_f1_score, 
            "obj_tp": obj_tp, 
            "obj_fp": obj_fp, 
            "obj_fn": obj_fn, 

            # Image-level metrics
            "img_precision": img_precision,
            "img_recall": img_recall, 
            "img_f1_score": img_f1_score,
            "img_accuracy": img_accuracy,
            "img_tp": img_tp,
            "img_fp": img_fp, 
            "img_fn": img_fn,
            "img_tn": img_tn}

# Find the IMAGE-LEVEL best threshold 
def find_best_conf_threshold_img(pred_folder, gt_folder, conf_thres_range, cat=None):
    best_conf_thres_img = 0
    best_f1_score_img = 0
    best_precision_img = 0
    best_recall_img = 0
    best_accuracy_img = 0

    for conf_thres in conf_thres_range:
        results = evaluate_predictions(pred_folder, gt_folder, conf_th=conf_thres, iou_th=0.1, cat=cat)
        if results["img_f1_score"] > best_f1_score_img:
            best_conf_thres_img = conf_thres
            best_f1_score_img = results["img_f1_score"]
            best_precision_img = results["img_precision"]
            best_recall_img = results["img_recall"]
            best_accuracy_img = results["img_accuracy"]

    return best_conf_thres_img, best_f1_score_img, best_precision_img, best_recall_img, best_accuracy_img

# Find the OBJECT-LEVEL best threshold 
def find_best_conf_threshold_obj(pred_folder, gt_folder, conf_thres_range, cat=None):
    best_conf_thres_obj = 0
    best_f1_score_obj = 0
    best_precision_obj = 0
    best_recall_obj = 0

    for conf_thres in conf_thres_range:
        results = evaluate_predictions(pred_folder, gt_folder, conf_th=conf_thres, iou_th=0.1, cat=cat)
        if results["obj_f1_score"] > best_f1_score_obj:
            best_conf_thres_obj = conf_thres
            best_f1_score_obj = results["obj_f1_score"]
            best_precision_obj = results["obj_precision"]
            best_recall_obj = results["obj_recall"]

    return best_conf_thres_obj, best_f1_score_obj, best_precision_obj, best_recall_obj


def evaluate_multiple_pred_folders_img(pred_folders, gt_folder, conf_thres_range, cat=None):
    """Compare multiple prediction folders using IMAGE-LEVEL metrics"""
    results_df = pd.DataFrame(
        columns=[
            "Prediction Folder",
            "Best Threshold",
            "Best F1 Score",
            "Precision",
            "Recall",
            "Accuracy",
        ]
    )

    for pred_folder in pred_folders:
        img_best_conf_thres, img_best_f1_score, img_best_precision, img_best_recall, img_best_accuracy= (
            find_best_conf_threshold_img(pred_folder, gt_folder, conf_thres_range, cat)
        )

        # Use loc to append data to the DataFrame to avoid potential issues
        results_df.loc[len(results_df.index)] = [
            pred_folder.split("/")[7],
            img_best_conf_thres,
            img_best_f1_score,
            img_best_precision,
            img_best_recall,
            img_best_accuracy,
        ]

    return results_df

def evaluate_multiple_pred_folders_obj(pred_folders, gt_folder, conf_thres_range, cat=None):
    """Compare multiple prediction folders using OBJECT-LEVEL metrics"""
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
        obj_best_conf_thres, obj_best_f1_score, obj_best_precision, obj_best_recall = (
            find_best_conf_threshold_obj(pred_folder, gt_folder, conf_thres_range, cat)
        )

        # Use loc to append data to the DataFrame to avoid potential issues
        results_df.loc[len(results_df.index)] = [
            pred_folder.split("/")[7],
            obj_best_conf_thres,
            obj_best_f1_score,
            obj_best_precision,
            obj_best_recall,
        ]

    return results_df


def img_find_best_conf_threshold_and_plot(
    pred_folder, gt_folder, conf_thres_range, plot=True
):
    f1_scores, precisions, recalls, accuracies = [], [], [], []

    for conf_thres in conf_thres_range:
        results = evaluate_predictions(pred_folder, gt_folder, conf_thres, iou_th=0.1)
        f1_scores.append(results["img_f1_score"])
        precisions.append(results["img_precision"])
        recalls.append(results["img_recall"])
        accuracies.append(results["img_accuracy"])

    # Find the best confidence threshold
    best_idx = np.argmax(f1_scores)
    best_conf_thres = conf_thres_range[best_idx]
    best_f1_score = f1_scores[best_idx]
    best_precision = precisions[best_idx]
    best_recall = recalls[best_idx]
    best_accuracy = accuracies[best_idx] 

    # save the best recall, precision and f1 score
    np.save(f"{EVAL_DIR}/img_f1_scores.npy", f1_scores)
    np.save(f"{EVAL_DIR}/img_precisions.npy", precisions)
    np.save(f"{EVAL_DIR}/img_recalls.npy", recalls)
    np.save(f"{EVAL_DIR}/img_conf_thres.npy", conf_thres_range)
    np.save(f"{EVAL_DIR}/img_accuracies.npy", accuracies)


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

        plt.plot(
            conf_thres_range, accuracies, label="Accuracy", color="purple", linestyle=":"
        )
        plt.scatter(
            best_conf_thres, best_accuracy, color="purple", s=100, edgecolor="k", zorder=5
        )

        # Highlight the best configuration
        plt.scatter(
            best_conf_thres, best_f1_score, color="blue", s=150, marker='*', edgecolor="black", zorder=6
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
            best_conf_thres + 0.02, best_f1_score,
            f"Best F1: {best_f1_score:.2f}",
            fontsize=9, ha='left', va='center', color='blue', weight='bold',
        )

        plt.text(
            best_conf_thres + 0.02, best_precision,
            f"Precision at best threshold: {best_precision:.2f}",
            fontsize=9, ha='left', va='center', color='green', weight='bold'
        )

        plt.text(
            best_conf_thres + 0.02, best_recall,
            f"Recall at best threshold: {best_recall:.2f}",
            fontsize=9, ha='left', va='center', color='red', weight='bold',
        )

        plt.text(
            best_conf_thres + 0.02, best_accuracy,
            f"Accuracy at best threshold: {best_accuracy:.2f}",
            fontsize=9, ha='left', va='center', color='purple', weight='bold',
        )

        plt.title(f"{EXPERIMENT_NAME} – Image-Level Metrics vs. Confidence Threshold")
        plt.xlabel("Confidence Threshold")
        plt.ylabel("Metric Value")
        plt.legend()
        plt.grid(True)
        # save the plot in the plots folder (image level)
        plt.savefig(f"{PLOT_DIR}/image_metrics.png")
        # save the list

        plt.show()

    return best_conf_thres, best_f1_score, best_precision, best_recall, best_accuracy

def obj_find_best_conf_threshold_and_plot(
    pred_folder, gt_folder, conf_thres_range, plot=True
):
    obj_f1_scores, obj_precisions, obj_recalls = [], [], []

    for conf_thres in conf_thres_range:
        results = evaluate_predictions(pred_folder, gt_folder, conf_thres, iou_th=0.1)
        obj_f1_scores.append(results["obj_f1_score"])
        obj_precisions.append(results["obj_precision"])
        obj_recalls.append(results["obj_recall"])

    # Find the best confidence threshold
    best_idx = np.argmax(obj_f1_scores)
    best_conf_thres = conf_thres_range[best_idx]
    best_f1_score = obj_f1_scores[best_idx]
    best_precision = obj_precisions[best_idx]
    best_recall = obj_recalls[best_idx]

    # save the best recall, precision and f1 score
    np.save(f"{EVAL_DIR}/obj_f1_scores.npy", obj_f1_scores)
    np.save(f"{EVAL_DIR}/obj_precisions.npy", obj_precisions)
    np.save(f"{EVAL_DIR}/obj_recalls.npy", obj_recalls)
    np.save(f"{EVAL_DIR}/obj_conf_thres.npy", conf_thres_range)


    if plot:

        # Plotting the metrics
        plt.figure(figsize=(10, 6))
        plt.plot(
            conf_thres_range, obj_f1_scores, label="F1 Score", color="blue", marker="o"
        )
        plt.plot(
            conf_thres_range,
            obj_precisions,
            label="Precision",
            color="green",
            linestyle="--",
        )
        plt.plot(conf_thres_range, obj_recalls, label="Recall", color="red", linestyle="-.")

        # Highlight the best configuration
        plt.scatter(
            best_conf_thres, best_f1_score, color="blue", s=150, marker='*', edgecolor="black", zorder=6
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
            best_conf_thres + 0.02, best_f1_score,
            f"Best F1: {best_f1_score:.2f}",
            fontsize=9, ha='left', va='center', color='blue', weight='bold',
        )

        plt.text(
            best_conf_thres + 0.02, best_precision,
            f"Precision at best threshold: {best_precision:.2f}",
            fontsize=9, ha='left', va='center', color='green', weight='bold',
        )

        plt.text(
            best_conf_thres + 0.02, best_recall,
            f"Recall at best threshold: {best_recall:.2f}",
            fontsize=9, ha='left', va='center', color='red', weight='bold',
        )

        plt.title(f"{EXPERIMENT_NAME} – Object-Level Metrics vs. Confidence Threshold")
        plt.xlabel("Confidence Threshold")
        plt.ylabel("Metric Value")
        plt.legend()
        plt.grid(True)
        # save the plot in the plots folder (object level)
        plt.savefig(f"{PLOT_DIR}/object_metrics.png")
        # save the list

        plt.show()

    return best_conf_thres, best_f1_score, best_precision, best_recall



if __name__ == "__main__":
    import json
    from datetime import datetime
    
    print("🚀 Starting PyroNear YOLO Baseline Evaluation")
    print(f"📅 Started at: {datetime.now()}")
    
    # ===== UPDATE THESE PATHS =====
    GT_FOLDER = "/vol/bitbucket/si324/rf-detr-wildfire/data/pyro25img/labels/test"  # GT labels
    PRED_FOLDER = "/vol/bitbucket/si324/rf-detr-wildfire/models/yolo_baseline_v1/test_preds/train/labels"  # Predictions
    
    # Check paths exist
    if not os.path.exists(GT_FOLDER):
        print(f"❌ GT folder not found: {GT_FOLDER}")
        print("   Please update GT_FOLDER path")
        exit(1)
        
    if not os.path.exists(PRED_FOLDER):
        print(f"❌ Predictions folder not found: {PRED_FOLDER}")
        print("   Please update PRED_FOLDER path")
        exit(1)
    
    print(f"✅ GT folder: {GT_FOLDER}")
    print(f"✅ Predictions folder: {PRED_FOLDER}")
    
    # Run evaluation 
    conf_range = np.arange(0.1, 0.9, 0.05)
    print(f"📊 Testing {len(conf_range)} confidence thresholds")

    test_images_dir = "/vol/bitbucket/si324/rf-detr-wildfire/data/pyro25img/images/test"
    # Get all image files
    all_image_files = [f for f in os.listdir(test_images_dir)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    all_filenames = [os.path.splitext(f)[0] for f in all_image_files]

    print(f"📂 Processing {len(all_filenames)} total test images")
    
    try:
        # Get best thresholds for both levels
        img_best_conf, img_best_f1, img_best_precision, img_best_recall, img_best_accuracy = find_best_conf_threshold_img(PRED_FOLDER, GT_FOLDER, conf_range)
        obj_best_conf, obj_best_f1, obj_best_precision, obj_best_recall = find_best_conf_threshold_obj(PRED_FOLDER, GT_FOLDER, conf_range)

        # Plot both
        img_find_best_conf_threshold_and_plot(PRED_FOLDER, GT_FOLDER, conf_range, plot=True)
        obj_find_best_conf_threshold_and_plot(PRED_FOLDER, GT_FOLDER, conf_range, plot=True)

        # Get final detailed results at best threshold for confusion matrix
        iou_th = 0.1
        final_results_img = evaluate_predictions(PRED_FOLDER, GT_FOLDER, img_best_conf, iou_th=0.1)
        final_results_obj = evaluate_predictions(PRED_FOLDER, GT_FOLDER, obj_best_conf, iou_th=0.1)
        
        # Save summary results
        summary = {
            "experiment_name": EXPERIMENT_NAME,
            "iou_threshold": iou_th,
            "image_level_best_threshold": float(img_best_conf),
            "object_level_best_threshold": float(obj_best_conf),

            # IMAGE-LEVEL results 
            "image_level_confusion_matrix": {
                "true_positives": int(final_results_img["img_tp"]),
                "true_negatives": int(final_results_img["img_tn"]),
                "false_positives": int(final_results_img["img_fp"]),
                "false_negatives": int(final_results_img["img_fn"])
            },
            "image_level_metrics": {
                "f1_score": float(final_results_img["img_f1_score"]),
                "precision": float(final_results_img["img_precision"]),
                "recall": float(final_results_img["img_recall"]),
                "accuracy": float(final_results_img["img_accuracy"])
            },

            # OBJECT-LEVEL results using OBJECT-LEVEL best threshold
            "object_level_confusion_matrix": {
                "true_positives": int(final_results_obj["obj_tp"]),
                "false_positives": int(final_results_obj["obj_fp"]),
                "false_negatives": int(final_results_obj["obj_fn"])
            },
            "object_level_metrics": {
                "f1_score": float(final_results_obj["obj_f1_score"]),
                "precision": float(final_results_obj["obj_precision"]),
                "recall": float(final_results_obj["obj_recall"]),
            },         
            "evaluation_timestamp": datetime.now().isoformat(),
            "gt_folder": GT_FOLDER,
            "pred_folder": PRED_FOLDER
        }
        
        with open(f"{EVAL_DIR}/summary_results.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n🎉 EVALUATION COMPLETE!")
        print(f"🏆 Image-Level Best F1: {img_best_f1:.3f} at confidence {img_best_conf:.3f}")
        print(f"🏆 Object-Level Best F1: {obj_best_f1:.3f} at confidence {obj_best_conf:.3f}")
        print(f"📊 Image Metrics - Precision: {img_best_precision:.3f}, Recall: {img_best_recall:.3f}, Accuracy: {img_best_accuracy:.3f}")
        print(f"📊 Object Metrics - Precision: {obj_best_precision:.3f}, Recall: {obj_best_recall:.3f}")
        print(f"🔢 Image Level Confusion Matrix - TP: {final_results_img['img_tp']}, TN: {final_results_img['img_tn']}, FP: {final_results_img['img_fp']}, FN: {final_results_img['img_fn']}")
        print(f"🔢 Object Level Confusion Matrix - TP: {final_results_obj['obj_tp']}, FP: {final_results_obj['obj_fp']}, FN: {final_results_obj['obj_fn']}")
        print(f"📁 Results saved to: {EVAL_DIR}/")
        print(f"📈 Plots saved to: {PLOT_DIR}/")
        print(f"📊 Confusion matrices saved to: {PLOT_DIR}/")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

    import seaborn as sns

    # IMAGE-LEVEL CONFUSION MATRIX
    img_cm = np.array([[final_results_img["img_tp"], final_results_img["img_fn"]],
                   [final_results_img["img_fp"], final_results_img["img_tn"]]])
    plt.figure(figsize=(6, 5))
    sns.heatmap(img_cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Pred Fire", "Pred No Fire"],
                yticklabels=["GT Fire", "GT No Fire"])
    plt.title(f"{EXPERIMENT_NAME} – Image-Level Confusion Matrix")
    plt.savefig(f"{PLOT_DIR}/image_conf_matrix.png")
    plt.close()

    # OBJECT-LEVEL CONFUSION MATRIX
    obj_cm = np.array([[final_results_obj["obj_tp"], final_results_obj["obj_fn"]],
                   [final_results_obj["obj_fp"], 0]])

    # TN for object level is N/A
    labels = obj_cm.astype(object)
    labels[1, 1] = "N/A"

    plt.figure(figsize=(6, 5))
    sns.heatmap(obj_cm, annot=labels, fmt='', cmap="Oranges",
                xticklabels=["Pred Fire", "Pred No Fire"],
                yticklabels=["GT Fire", "GT No Fire"])
    plt.title(f"{EXPERIMENT_NAME} – Object-Level Confusion Matrix")
    plt.savefig(f"{PLOT_DIR}/object_conf_matrix.png")
    plt.close()

    
    print(f"✅ Finished at: {datetime.now()}")