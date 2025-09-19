"""
RF-DETR Hyperparameter Tuning Visualizations
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Configuration 
RESULTS_FILE = "/vol/bitbucket/si324/rf-detr-wildfire/src/images/eval_results_hparam_tuning_NMS/RF-DETR_hparam_tuning_NMS/all_trials_results.json"
OUTPUT_DIR = "/vol/bitbucket/si324/rf-detr-wildfire/src/images/eval_results_hparam_tuning_NMS/RF-DETR_hparam_tuning_NMS"

def create_visualizations(results_file, output_dir):
    """Hyperparameter visualizations"""
    
    os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    trials = sorted(results, key=lambda x: x['trial_number'])
    f1_scores = [t['best_result']['f1_score'] for t in trials]
    best_idx = np.argmax(f1_scores)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Create figure with subplot positions 
    fig = plt.figure(figsize=(14, 5))
    
    # Create subplots with equal heights
    ax1 = plt.subplot2grid((1, 2), (0, 0))
    ax2 = plt.subplot2grid((1, 2), (0, 1))
    
    # 1. Optimization history
    ax1.plot(range(len(f1_scores)), f1_scores, 'o-', color='cornflowerblue', 
             alpha=0.8, linewidth=2, markersize=7, label='Trial F1')
    
    # Add best line and star
    ax1.axhline(y=max(f1_scores), color='tomato', linestyle='--', linewidth=2, alpha=0.8)
    ax1.scatter(best_idx, f1_scores[best_idx], s=200, marker='*', color='tomato', 
                edgecolor='darkred', linewidth=1.5, zorder=5)
    
    # Legend with star
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='cornflowerblue', linewidth=2, marker='o', label='Trial F1'),
        Line2D([0], [0], color='tomato', linewidth=2, linestyle='--', 
               marker='*', markersize=10, label=f'Best: {max(f1_scores):.2f}')
    ]
    ax1.legend(handles=legend_elements, loc='lower right', frameon=True, fancybox=True, shadow=True)
    
    ax1.set_xlabel('Trial Number', fontsize=12)
    ax1.set_ylabel('F1 Score', fontsize=12)
    #ax1.set_title('Hyperparameter Optimization Progress', fontsize=14, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])
    
    # 2. Top 5 table
    ax2.axis('tight')
    ax2.axis('off')
    
    # Add title at the same height as ax1
    #ax2.set_title('Top 5 Configurations', fontsize=14, fontweight='bold', pad=15)
    
    top_5 = sorted(trials, key=lambda x: x['best_result']['f1_score'], reverse=True)[:5]
    
    table_data = [[f"{t['trial_number']}", 
                   f"{t['best_result']['f1_score']:.3f}",
                   f"{t['hyperparameters']['resolution']}",
                   f"{t['hyperparameters']['batch_size']}",
                   f"{t['hyperparameters']['lr']:.2e}"] 
                  for t in top_5]
    
    # Position table to be centered vertically in the subplot
    table = ax2.table(cellText=table_data,
                 colLabels=['Trial', 'F1 Score', 'Resolution', 'Batch Size', 'Learning Rate'],
                 cellLoc='center',
                 loc='center',
                 colWidths=[0.12, 0.18, 0.18, 0.18, 0.28],
                 bbox=[0, 0, 1, 1])  

    table.auto_set_font_size(False)
    table.set_fontsize(12)  
    table.scale(1.2, 2.2)  
    
    # Style the table
    for i in range(5):
        # Header row - dark gray
        table[(0, i)].set_facecolor('#34495E')
        table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Highlight best trial row with paleturquoise
        if i == 0:  # First data row is the best
            for j in range(5):
                table[(1, j)].set_facecolor('paleturquoise')
                table[(1, j)].set_text_props(weight='bold')
    
    # Plot title
    #fig.suptitle('RF-DETR Hyperparameter Tuning Results', fontsize=15, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.90) 
    
    output_path = os.path.join(output_dir, 'visualizations', 'hparam_results.png')
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    create_visualizations(RESULTS_FILE, OUTPUT_DIR)