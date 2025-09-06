import matplotlib.pyplot as plt
import numpy as np

# Data from your JSON
data = {
    "YOLO_baseline": {"tp": 59, "fp": 15, "fn": 12},
    "RT-DETR_best_hparams": {"tp": 67, "fp": 19, "fn": 4},
    "RF-DETR_initial_training_best": {"tp": 58, "fp": 10, "fn": 13}
}

# Order and labels to match your plot
model_order = ['YOLO_baseline', 'RT-DETR_best_hparams', 'RF-DETR_initial_training_best']
model_labels = ['YOLOv8', 'RT-DETR', 'RF-DETR']

# Extract values in correct order
tp_values = [data[m]['tp'] for m in model_order]
fp_values = [data[m]['fp'] for m in model_order]
fn_values = [data[m]['fn'] for m in model_order]

# Create plot
fig, ax = plt.subplots(figsize=(12, 7))

x = np.arange(len(model_order))
width = 0.25

# Create bars
bars1 = ax.bar(x - width, tp_values, width, label='True Positives', color='#2ECC71')
bars2 = ax.bar(x, fp_values, width, label='False Positives', color='#E74C3C')
bars3 = ax.bar(x + width, fn_values, width, label='False Negatives', color='#3498DB')

# Add value labels on bars - using smaller offset for closer positioning
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        # Changed from +3 to +1 for closer positioning
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{int(height)}', ha='center', va='bottom', 
                fontsize=11, fontweight='bold')

ax.set_xlabel('Models', fontsize=15, fontweight='bold')
ax.set_ylabel('Number of Detections', fontsize=15, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(model_labels, fontsize=14)
ax.legend(loc='upper right', fontsize=12)
ax.grid(True, alpha=0.3, axis='y')

# Set y-axis limit to give some space at top
ax.set_ylim([0, max(tp_values + fp_values + fn_values) + 5])

plt.tight_layout()
plt.savefig("smokeframes_object_detection_breakdown.png", dpi=300, bbox_inches='tight')
plt.show()