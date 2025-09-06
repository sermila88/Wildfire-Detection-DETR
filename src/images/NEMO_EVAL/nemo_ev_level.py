import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
import cv2
import subprocess
from PIL import Image
import supervision as sv
from datetime import datetime

# Configuration
NEMO_DIR = "/vol/bitbucket/si324/rf-detr-wildfire/src/images/data/nemo_val"  # Parent directory
OUTPUT_DIR = "/vol/bitbucket/si324/rf-detr-wildfire/src/images/outputs/NEMO_evaluation"
TEST_IMAGES_DIR = "/vol/bitbucket/si324/rf-detr-wildfire/src/images/data/nemo_val/img"
CONF_THRESHOLD = 0.01  
IOU_THRESHOLD = 0.01  # As you specified

class NEMOEvaluator:
    def __init__(self, nemo_dir, models_dict, output_dir, conf_threshold=0.01, iou_threshold=0.01):
        self.nemo_dir = nemo_dir
        self.models = models_dict
        self.output_dir = output_dir
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for different outputs
        self.plots_dir = os.path.join(self.output_dir, "plots")
        self.predictions_dir = os.path.join(self.output_dir, "predictions")
        Path(self.plots_dir).mkdir(exist_ok=True)
        Path(self.predictions_dir).mkdir(exist_ok=True)
        
        # Load annotations by severity
        print(f"Loading NEMO annotations from: {self.nemo_dir}")
        self.annotations = self.load_nemo_annotations()
        print(f"Loaded annotations for {len(self.annotations['all'])} images")
        self.results = {}
        
    def load_nemo_annotations(self):
        """Load NEMO annotations organized by severity"""
        ann_dir = os.path.join(self.nemo_dir, "ann")
        img_dir = os.path.join(self.nemo_dir, "img")
        
        if not os.path.exists(ann_dir):
            raise ValueError(f"Annotation directory not found: {ann_dir}")
        if not os.path.exists(img_dir):
            raise ValueError(f"Image directory not found: {img_dir}")
            
        annotations = {
            'low': defaultdict(list),
            'mid': defaultdict(list),
            'high': defaultdict(list),
            'all': defaultdict(list)  # Combined for smoke-presence matching
        }
        
        # FIXED: Look for *.jpg.json files
        ann_files = list(Path(ann_dir).glob("*.jpg.json"))
        if not ann_files:
            ann_files = list(Path(ann_dir).glob("*.png.json"))
        
        print(f"Found {len(ann_files)} annotation files")
        
        for ann_file in tqdm(ann_files, desc="Loading annotations"):
            # FIXED: Remove .json to get image name
            img_name_with_ext = ann_file.name[:-5]  # Remove '.json' from 'image.jpg.json'
            img_name_stem = img_name_with_ext[:-4]  # Remove '.jpg' to get stem
            
            # Check if corresponding image exists
            img_path = os.path.join(img_dir, img_name_with_ext)
            if not os.path.exists(img_path):
                continue
            
            with open(ann_file, 'r') as f:
                data = json.load(f)
            
            img_width = data["size"]["width"]
            img_height = data["size"]["height"]
            
            for obj in data["objects"]:
                class_title = obj.get("classTitle", "").lower()
                
                if "smoke" in class_title:
                    points = obj["points"]["exterior"]
                    x1, y1 = points[0]
                    x2, y2 = points[1]
                    
                    # Normalize to [0,1]
                    box = [
                        x1 / img_width,
                        y1 / img_height,
                        x2 / img_width,
                        y2 / img_height
                    ]
                    
                    # Add to appropriate severity level - use image name without extension
                    if "low" in class_title:
                        annotations['low'][img_name_stem].append(box)
                    elif "mid" in class_title:
                        annotations['mid'][img_name_stem].append(box)
                    elif "high" in class_title:
                        annotations['high'][img_name_stem].append(box)
                    
                    # Add to combined list
                    annotations['all'][img_name_stem].append(box)
        
        # Print statistics
        for severity in ['low', 'mid', 'high', 'all']:
            count = len([k for k, v in annotations[severity].items() if v])
            print(f"  {severity}: {count} images with annotations")
        
        return annotations
    
    def calculate_iou(self, box1, box2):
        """Calculate IoU between two boxes (normalized coordinates)"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def generate_predictions_for_model(self, model_name, model_info):
        """Generate predictions using your existing code"""
        print(f"\nGenerating predictions for {model_name}...")
        
        pred_output_dir = os.path.join(self.predictions_dir, model_name)
        Path(pred_output_dir).mkdir(exist_ok=True)
        
        if model_info["type"] == "YOLO":
            # Use YOLO command line
            cmd = f"yolo predict model={model_info['path']} iou=0.01 conf=0.01 source={TEST_IMAGES_DIR} save=False save_txt save_conf project={pred_output_dir} name=predictions exist_ok=True"
            print(f"Running: {cmd}")
            subprocess.call(cmd, shell=True)
            return os.path.join(pred_output_dir, "predictions", "labels")
            
        elif model_info["type"] in ["RF-DETR", "RT-DETR"]:
            # Create labels directory
            labels_dir = os.path.join(pred_output_dir, "labels")
            Path(labels_dir).mkdir(exist_ok=True)
            
            # Load model once
            if model_info["type"] == "RF-DETR":
                import sys
                sys.path.append('/vol/bitbucket/si324/rf-detr-wildfire/src/images')
                from rfdetr import RFDETRBase
                model = RFDETRBase(pretrain_weights=model_info['path'], num_classes=1)
            else:
                from ultralytics import RTDETR
                model = RTDETR(model_info['path'])
            
            # Process each image
            for img_file in tqdm(Path(TEST_IMAGES_DIR).glob("*.jpg"), desc=f"Predicting with {model_name}"):
                img_name = img_file.stem  # Get name without .jpg
                
                if model_info["type"] == "RF-DETR":
                    predictions = self.generate_rfdetr_predictions(model, str(img_file), 0.01)
                else:
                    predictions = self.generate_rtdetr_predictions(model, str(img_file), 0.01)
                
                # Save predictions
                pred_file = os.path.join(labels_dir, f"{img_name}.txt")
                with open(pred_file, 'w') as f:
                    for line in predictions:
                        f.write(line + '\n')
                        
                # Apply NMS
                self.apply_nms_to_predictions(pred_file, 0.01)
            
            return labels_dir
    
    def generate_rfdetr_predictions(self, model, image_path, conf_threshold):
        """Generate RF-DETR predictions"""
        with Image.open(image_path) as img:
            img_rgb = img.convert("RGB")
            predictions = model.predict(img_rgb, threshold=conf_threshold)
        
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
    
    def generate_rtdetr_predictions(self, model, image_path, conf_threshold):
        """Generate RT-DETR predictions"""
        with Image.open(image_path) as img:
            img_rgb = img.convert("RGB")
            results = model.predict(source=img_rgb, conf=conf_threshold, verbose=False)
            predictions = sv.Detections.from_ultralytics(results[0])
        
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
    
    def apply_nms_to_predictions(self, pred_file, iou_threshold=0.01):
        """Apply NMS to predictions"""
        if not os.path.exists(pred_file) or os.path.getsize(pred_file) == 0:
            return
        
        boxes = []
        scores = []
        lines = []
        
        with open(pred_file, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) >= 6:
                    _, x, y, w, h, conf = map(float, parts)
                    x1 = x - w/2
                    y1 = y - h/2
                    x2 = x + w/2
                    y2 = y + h/2
                    boxes.append([x1, y1, x2, y2])
                    scores.append(conf)
                    lines.append(line.strip())
        
        if not boxes:
            return
        
        indices = cv2.dnn.NMSBoxes(
            bboxes=boxes,
            scores=scores,
            score_threshold=0.01,
            nms_threshold=iou_threshold
        )
        
        with open(pred_file, 'w') as f:
            if len(indices) > 0:
                indices = indices.flatten()
                for i in indices:
                    f.write(lines[i] + '\n')
    
    def load_predictions(self, model_name, pred_dir):
        """Load predictions for a model"""
        predictions = defaultdict(list)
        
        if not os.path.exists(pred_dir):
            print(f"Warning: Predictions directory not found: {pred_dir}")
            return predictions
            
        for pred_file in Path(pred_dir).glob("*.txt"):
            img_name = pred_file.stem  # Image name without extension
            
            if not os.path.exists(pred_file) or os.path.getsize(pred_file) == 0:
                continue
                
            with open(pred_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 6:
                        _, x, y, w, h, conf = map(float, parts)
                        if conf >= self.conf_threshold:
                            # Convert to xyxy normalized
                            x1 = x - w/2
                            y1 = y - h/2
                            x2 = x + w/2
                            y2 = y + h/2
                            predictions[img_name].append([x1, y1, x2, y2, conf])
        
        print(f"  Loaded predictions for {len(predictions)} images")
        return predictions
    
    def evaluate_smoke_presence(self, predictions):
        """
        A. Smoke-Presence Matching (Most Lenient)
        Any detection overlapping ANY smoke annotation = TP
        """
        tp, fp, fn = 0, 0, 0
        
        for img_name in set(list(predictions.keys()) + list(self.annotations['all'].keys())):
            pred_boxes = predictions.get(img_name, [])
            gt_boxes = self.annotations['all'].get(img_name, [])
            
            # Track which GT boxes were matched
            matched_gt = [False] * len(gt_boxes)
            
            # Check each prediction
            for pred in pred_boxes:
                pred_box = pred[:4]
                found_match = False
                
                # Check if it overlaps ANY smoke annotation
                for gt_idx, gt_box in enumerate(gt_boxes):
                    if self.calculate_iou(pred_box, gt_box) >= self.iou_threshold:
                        found_match = True
                        matched_gt[gt_idx] = True
                        break
                
                if found_match:
                    tp += 1
                else:
                    fp += 1
            
            # Count unmatched GT boxes as FN
            fn += matched_gt.count(False)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'tp': tp, 'fp': fp, 'fn': fn,
            'precision': precision, 'recall': recall, 'f1': f1
        }
    
    def evaluate_severity_stratified(self, predictions):
        """
        B. Severity-Stratified Recall
        Report recall separately for each severity level
        """
        results = {}
        
        for severity in ['low', 'mid', 'high']:
            tp, fn = 0, 0
            
            for img_name, gt_boxes in self.annotations[severity].items():
                pred_boxes = predictions.get(img_name, [])
                
                for gt_box in gt_boxes:
                    matched = False
                    for pred in pred_boxes:
                        if self.calculate_iou(pred[:4], gt_box) >= self.iou_threshold:
                            matched = True
                            break
                    
                    if matched:
                        tp += 1
                    else:
                        fn += 1
            
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            results[severity] = {
                'tp': tp,
                'fn': fn,
                'recall': recall,
                'total_annotations': tp + fn
            }
        
        return results
    
    def evaluate_cross_severity(self, predictions):
        """
        D. Cross-Severity Confusion Matrix
        Track which severity level each detection matches
        """
        confusion = {
            'low': {'low': 0, 'mid': 0, 'high': 0, 'missed': 0},
            'mid': {'low': 0, 'mid': 0, 'high': 0, 'missed': 0},
            'high': {'low': 0, 'mid': 0, 'high': 0, 'missed': 0}
        }
        
        for img_name in self.annotations['all'].keys():
            pred_boxes = predictions.get(img_name, [])
            
            # Process each severity level
            for true_severity in ['low', 'mid', 'high']:
                gt_boxes = self.annotations[true_severity].get(img_name, [])
                
                for gt_box in gt_boxes:
                    best_match = None
                    best_iou = 0
                    
                    # Find best matching prediction
                    for pred in pred_boxes:
                        iou = self.calculate_iou(pred[:4], gt_box)
                        if iou > best_iou and iou >= self.iou_threshold:
                            best_iou = iou
                            best_match = pred[:4]
                    
                    if best_match:
                        # Determine which severity this detection best matches
                        detected_as = self.find_detection_severity(best_match, img_name)
                        confusion[true_severity][detected_as] += 1
                    else:
                        confusion[true_severity]['missed'] += 1
        
        return confusion
    
    def find_detection_severity(self, pred_box, img_name):
        """Find which severity level a detection best matches"""
        best_severity = 'low'  # Default
        best_iou = 0
        
        for severity in ['low', 'mid', 'high']:
            gt_boxes = self.annotations[severity].get(img_name, [])
            for gt_box in gt_boxes:
                iou = self.calculate_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_severity = severity
        
        return best_severity
    
    def plot_detection_sensitivity(self, all_results):
        """
        C. Detection Sensitivity Curve
        Plot detection rate vs smoke density level
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        severities = ['low', 'mid', 'high']
        colors = {'YOLO_baseline': 'blue', 'RF-DETR_initial_training_NMS': 'green', 
                  'RT-DETR_HP_best': 'orange'}
        
        # Plot 1: Line plot
        for model_name, results in all_results.items():
            recalls = [results['severity_stratified'][sev]['recall'] * 100 
                      for sev in severities]
            ax1.plot(severities, recalls, marker='o', linewidth=2, 
                    label=model_name.replace('_', ' '), 
                    color=colors.get(model_name, 'gray'))
        
        ax1.set_xlabel('Smoke Density Level', fontsize=12)
        ax1.set_ylabel('Detection Rate (%)', fontsize=12)
        ax1.set_title('Detection Sensitivity Curve', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 100])
        
        # Plot 2: Bar plot for comparison
        x = np.arange(len(severities))
        width = 0.25
        
        for i, (model_name, results) in enumerate(all_results.items()):
            recalls = [results['severity_stratified'][sev]['recall'] * 100 
                      for sev in severities]
            ax2.bar(x + i*width, recalls, width, 
                   label=model_name.replace('_', ' '),
                   color=colors.get(model_name, 'gray'))
        
        ax2.set_xlabel('Smoke Density Level', fontsize=12)
        ax2.set_ylabel('Detection Rate (%)', fontsize=12)
        ax2.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax2.set_xticks(x + width)
        ax2.set_xticklabels(severities)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        output_path = os.path.join(self.plots_dir, 'detection_sensitivity_curve.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_path}")
        plt.close()
        
        return fig
    
    def plot_confusion_matrix(self, confusion, model_name):
        """Plot cross-severity confusion matrix"""
        matrix = []
        
        for true_sev in ['low', 'mid', 'high']:
            row = []
            total = sum(confusion[true_sev].values())
            for detected_as in ['low', 'mid', 'high', 'missed']:
                pct = (confusion[true_sev][detected_as] / total * 100) if total > 0 else 0
                row.append(pct)
            matrix.append(row)
        
        matrix = np.array(matrix)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(matrix, annot=True, fmt='.1f', cmap='YlOrRd',
                   xticklabels=['Low', 'Mid', 'High', 'Missed'],
                   yticklabels=['Low', 'Mid', 'High'],
                   cbar_kws={'label': 'Percentage (%)'},
                   ax=ax)
        
        ax.set_xlabel('Detected As', fontsize=12)
        ax.set_ylabel('True Severity', fontsize=12)
        ax.set_title(f'Cross-Severity Confusion Matrix - {model_name}', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        output_path = os.path.join(self.plots_dir, f'confusion_matrix_{model_name}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_path}")
        plt.close()
        
        return fig
    
    def run_evaluation(self):
        """Run complete evaluation for all models"""
        all_results = {}
        
        for model_name, model_info in self.models.items():
            print(f"\n{'='*60}")
            print(f"Evaluating {model_name}")
            print(f"{'='*60}")
            
            # Generate predictions
            labels_dir = self.generate_predictions_for_model(model_name, model_info)
            
            # Load predictions
            predictions = self.load_predictions(model_name, labels_dir)
            
            # Run all evaluations
            print(f"\nRunning evaluations...")
            results = {
                'smoke_presence': self.evaluate_smoke_presence(predictions),
                'severity_stratified': self.evaluate_severity_stratified(predictions),
                'cross_severity': self.evaluate_cross_severity(predictions)
            }
            
            all_results[model_name] = results
            
            # Plot confusion matrix
            self.plot_confusion_matrix(results['cross_severity'], model_name)
        
        # Plot sensitivity curves for all models
        self.plot_detection_sensitivity(all_results)
        
        # Print comprehensive results
        self.print_results(all_results)
        
        # Save results
        self.save_results_to_csv(all_results)
        
        return all_results
    
    def print_results(self, all_results):
        """Print formatted results"""
        print("\n" + "="*80)
        print("COMPREHENSIVE EVALUATION RESULTS")
        print("="*80)
        
        # A. Smoke Presence Results
        print("\nA. SMOKE PRESENCE MATCHING (Lenient Evaluation)")
        print("-"*60)
        print(f"{'Model':<30} {'TP':<8} {'FP':<8} {'FN':<8} {'Precision':<10} {'Recall':<10} {'F1':<10}")
        print("-"*60)
        
        for model_name, results in all_results.items():
            sp = results['smoke_presence']
            print(f"{model_name:<30} {sp['tp']:<8} {sp['fp']:<8} {sp['fn']:<8} "
                  f"{sp['precision']:<10.3f} {sp['recall']:<10.3f} {sp['f1']:<10.3f}")
        
        # B. Severity-Stratified Recall
        print("\n\nB. SEVERITY-STRATIFIED RECALL")
        print("-"*60)
        print(f"{'Model':<30} {'Low Smoke':<12} {'Mid Smoke':<12} {'High Smoke':<12}")
        print("-"*60)
        
        for model_name, results in all_results.items():
            ss = results['severity_stratified']
            print(f"{model_name:<30} "
                  f"{ss['low']['recall']*100:<11.1f}% "
                  f"{ss['mid']['recall']*100:<11.1f}% "
                  f"{ss['high']['recall']*100:<11.1f}%")
        
        # Find best model for early detection
        best_early = max(all_results.items(), 
                        key=lambda x: x[1]['severity_stratified']['low']['recall'])
        print(f"\n🔥 Best for Early Detection (Low Smoke): {best_early[0]} "
              f"({best_early[1]['severity_stratified']['low']['recall']*100:.1f}% recall)")
        
        # Save results to file
        self.save_results_to_csv(all_results)
    
    def save_results_to_csv(self, all_results):
        """Save results to CSV for paper tables"""
        # Create DataFrame for severity-stratified results
        data = []
        for model_name, results in all_results.items():
            row = {
                'Model': model_name,
                'Low_Recall': results['severity_stratified']['low']['recall'],
                'Mid_Recall': results['severity_stratified']['mid']['recall'],
                'High_Recall': results['severity_stratified']['high']['recall'],
                'Overall_F1': results['smoke_presence']['f1'],
                'Overall_Precision': results['smoke_presence']['precision'],
                'Overall_Recall': results['smoke_presence']['recall']
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        output_path = os.path.join(self.output_dir, 'nemo_evaluation_results.csv')
        df.to_csv(output_path, index=False)
        print(f"\n📊 Results saved to {output_path}")

# Main execution
if __name__ == "__main__":
    models = {
        "YOLO_baseline": {
            "type": "YOLO",
            "path": "/vol/bitbucket/si324/rf-detr-wildfire/src/images/outputs/YOLO_baseline/training_outputs/eager-flower-1/weights/best.pt"
        },
        "RF-DETR_initial_training_NMS": {
            "type": "RF-DETR",
            "path": "/vol/bitbucket/si324/rf-detr-wildfire/src/images/outputs/RF-DETR_initial_training/checkpoints/checkpoint_best_ema.pth"
        },
        "RT-DETR_HP_best": {
            "type": "RT-DETR",
            "path": "/vol/bitbucket/si324/rf-detr-wildfire/src/images/outputs/RT-DETR_hyperparameter_tuning/trial_009/checkpoints/weights/best.pt"
        }
    }
    
    print(f"Starting NEMO Evaluation - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    evaluator = NEMOEvaluator(
        nemo_dir=NEMO_DIR,
        models_dict=models,
        output_dir=OUTPUT_DIR,
        conf_threshold=0.01,
        iou_threshold=0.01
    )
    
    results = evaluator.run_evaluation()
    
    print(f"\n✅ Evaluation complete! All outputs saved to: {OUTPUT_DIR}")