import glob
import os
from pyronear_utils import xywh2xyxy, box_iou
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

EXPERIMENT_NAME = "yolo_baseline_v1"  # <-- change this per experiment
EVAL_DIR = f"outputs/{EXPERIMENT_NAME}/eval_results"
os.makedirs(EVAL_DIR, exist_ok=True)


def evaluate_predictions(pred_folder, gt_folder, conf_th=0.1, cat=None):
    nb_fp, nb_tp, nb_fn = 0, 0, 0
    # Track correct predictions (TP + TN) for spatial accuracy calculation
    total_correct = 0

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
                    # Encontrar la mejor coincidencia por IoU
                    iou_values = [box_iou(pred_box, gt_box) for gt_box in gt_boxes]
                    max_iou = max(iou_values)
                    best_match_idx = np.argmax(iou_values)

                    # Verificar coincidencia válida y única
                    if max_iou > 0.1 and not gt_matches[best_match_idx]:
                        nb_tp += 1
                        gt_matches[best_match_idx] = True
                    else:
                        nb_fp += 1
                else:
                    nb_fp += 1

        if gt_boxes:
            nb_fn += len(gt_boxes) - np.sum(gt_matches)

        # Check if ground truth has any smoke boxes in this image
        # True if GT file contains smoke annotations, False if empty/no smoke
        has_smoke_gt = len(gt_boxes) > 0
        # Check if prediction file exists and has any predictions
        has_smoke_pred = os.path.isfile(pred_file) and os.path.getsize(pred_file) > 0
        # Check if there is spatial overlap between predictions and ground truth
        spatial_match = has_smoke_gt and has_smoke_pred and np.sum(gt_matches) > 0

        # Count correct predictions for accuracy calculation
        if (has_smoke_gt and spatial_match) or (not has_smoke_gt and not has_smoke_pred):
            total_correct += 1

    # Calculate spatial accuracy = (TP + TN) / Total images
    spatial_accuracy = total_correct / len(all_filenames) if len(all_filenames) > 0 else 0

    precision = nb_tp / (nb_tp + nb_fp) if (nb_tp + nb_fp) > 0 else 0
    recall = nb_tp / (nb_tp + nb_fn) if (nb_tp + nb_fn) > 0 else 0
    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    return {"precision": precision, "recall": recall, "f1_score": f1_score, "spatial_accuracy": spatial_accuracy}


def find_best_conf_threshold(pred_folder, gt_folder, conf_thres_range, cat=None):
    best_conf_thres = 0
    best_f1_score = 0
    best_precision = 0
    best_recall = 0
    best_accuracy = 0

    for conf_thres in conf_thres_range:
        results = evaluate_predictions(pred_folder, gt_folder, conf_thres, cat)
        if results["f1_score"] > best_f1_score:
            best_conf_thres = conf_thres
            best_f1_score = results["f1_score"]
            best_precision = results["precision"]
            best_recall = results["recall"]
            best_accuracy = results["spatial_accuracy"]

    return best_conf_thres, best_f1_score, best_precision, best_recall, best_accuracy


def evaluate_multiple_pred_folders(pred_folders, gt_folder, conf_thres_range, cat=None):
    # Initialize a DataFrame to store the results
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
        best_conf_thres, best_f1_score, best_precision, best_recall, best_accuracy= (
            find_best_conf_threshold(pred_folder, gt_folder, conf_thres_range, cat)
        )

        # Use loc to append data to the DataFrame to avoid potential issues
        results_df.loc[len(results_df.index)] = [
            pred_folder.split("/")[7],
            best_conf_thres,
            best_f1_score,
            best_precision,
            best_recall,
            best_accuracy,
        ]

    return results_df


def find_best_conf_threshold_and_plot(
    pred_folder, gt_folder, conf_thres_range, plot=True
):
    f1_scores, precisions, recalls, accuracies = [], [], [], []

    for conf_thres in conf_thres_range:
        results = evaluate_predictions(pred_folder, gt_folder, conf_thres)
        f1_scores.append(results["f1_score"])
        precisions.append(results["precision"])
        recalls.append(results["recall"])
        accuracies.append(results["spatial_accuracy"])

    # Find the best confidence threshold
    best_idx = np.argmax(f1_scores)
    best_conf_thres = conf_thres_range[best_idx]
    best_f1_score = f1_scores[best_idx]
    best_precision = precisions[best_idx]
    best_recall = recalls[best_idx]
    best_accuracy = accuracies[best_idx] 
    # save 
    # save the best recall, precision and f1 score
    
    np.save(f"{EVAL_DIR}/f1_scores.npy", f1_scores)
    np.save(f"{EVAL_DIR}/precisions.npy", precisions)
    np.save(f"{EVAL_DIR}/recalls.npy", recalls)
    np.save(f"{EVAL_DIR}/conf_thres.npy", conf_thres_range)
    np.save(f"{EVAL_DIR}/accuracies.npy", accuracies)


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
            f"Best F1: {best_f1_score:.2f}\n"
            f"Precision: {best_precision:.2f}\n"
            f"Recall: {best_recall:.2f}\n \n"
            f"Accuracy: {best_accuracy:.2f}",
            fontsize=9,
            verticalalignment="bottom",
        )

        plt.title("Evaluation Metrics vs. Confidence Threshold")
        plt.xlabel("Confidence Threshold")
        plt.ylabel("Metric Value")
        plt.legend()
        plt.grid(True)
        # save in predictions folder
        plt.savefig(f"{EVAL_DIR}/metrics.png")
        # save the list

        plt.show()

    return best_conf_thres, best_f1_score, best_precision, best_recall, best_accuracy


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
    
    try:
        best_conf, best_f1, best_precision, best_recall, best_accuracy = find_best_conf_threshold_and_plot(
            PRED_FOLDER, GT_FOLDER, conf_range, plot=True
        )
        
        # Save summary results
        summary = {
            "experiment_name": EXPERIMENT_NAME,
            "best_confidence_threshold": float(best_conf),
            "best_f1_score": float(best_f1),
            "best_precision": float(best_precision),
            "best_recall": float(best_recall),
            "best_accuracy": float(best_accuracy),
            "evaluation_timestamp": datetime.now().isoformat(),
            "gt_folder": GT_FOLDER,
            "pred_folder": PRED_FOLDER
        }
        
        with open(f"{EVAL_DIR}/summary_results.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n🎉 EVALUATION COMPLETE!")
        print(f"🏆 Best F1: {best_f1:.3f} at confidence {best_conf:.3f}")
        print(f"📊 Precision: {best_precision:.3f}, Recall: {best_recall:.3f}, Accuracy: {best_accuracy:.3f}")
        print(f"📁 Results saved to: {EVAL_DIR}/")
        print(f"📈 Plot saved to: {EVAL_DIR}/metrics.png")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    
    print(f"✅ Finished at: {datetime.now()}")