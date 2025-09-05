import json
import numpy as np
import matplotlib.pyplot as plt

# Load the data
with open('/vol/bitbucket/si324/rf-detr-wildfire/src/images/final_comparison/final_comparison_summary.json', 'r') as f:
    data = json.load(f)

model_order = ['YOLO_baseline', 'RT-DETR_best_hparams', 'RF-DETR_initial_training_best']
model_labels = ['YOLOv8', 'RT-DETR', 'RF-DETR']

# Colors
bar_colors = {
    'f1': '#1976D2',      # Blue for F1
    'precision': '#4CAF50', # Green for Precision  
    'recall': '#F44336',    # Red for Recall
    'accuracy': '#9B59B6'   # Purple for Accuracy
}

# Create the plot
fig, ax = plt.subplots(figsize=(12, 7))
metrics = ['F1 Score', 'Precision', 'Recall', 'Accuracy']
metric_keys = ['best_f1', 'precision', 'recall', 'accuracy']
color_keys = ['f1', 'precision', 'recall', 'accuracy']
x = np.arange(len(model_order))
width = 0.20  # Changed from 0.25 to fit 4 bars

for i, (metric, key, color_key) in enumerate(zip(metrics, metric_keys, color_keys)):
    values = [data['models'][m]['image_level'][key] for m in model_order]
    positions = x + (i - 1.5) * width  # Center the 4 bars
    color = bar_colors[color_key]
    bars = ax.bar(positions, values, width, label=metric, color=color)
    
    for j, bar in enumerate(bars):
        conf = data['models'][model_order[j]]['image_level']['best_conf']
        height = bar.get_height()
        # Only show conf on F1 bars
        if i == 0:
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                   f'{height:.2f}\n(τ={conf:.2f})', 
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        else:
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                   f'{height:.2f}', 
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.axhline(y=0.7, color='mediumpurple', linestyle='--', alpha=0.7, linewidth=0.8)

ax.set_xlabel('Models', fontsize=15, fontweight='bold')
ax.set_ylabel('Score', fontsize=15, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(model_labels, fontsize=14)
ax.legend(loc='upper right', frameon=True, shadow=True, fontsize=12)
ax.grid(True, alpha=0.3, linestyle='--', axis='y')
ax.set_ylim([0.60, 1.09])
plt.tight_layout()
plt.savefig('/vol/bitbucket/si324/rf-detr-wildfire/src/images/final_comparison/image_level_model_comparison_bars_with_accuracy.png', dpi=300, bbox_inches='tight')
plt.show()
print("Saved: image_level_model_comparison_bars_with_accuracy.png")