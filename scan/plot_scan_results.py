"""
Script to plot results from the final_L0 parameter scan.
Loads the latest scan_results_*.json file and generates plots.
"""

import json
import glob
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_latest_results():
    # Find all scan_results_*.json files
    files = glob.glob("scan_results_*.json")
    if not files:
        print("No scan_results_*.json files found!")
        return None, None
    
    # Sort by modification time
    latest_file = max(files, key=os.path.getmtime)
    print(f"Loading results from: {latest_file}")
    
    with open(latest_file, 'r') as f:
        data = json.load(f)
    return data, latest_file

def extract_series(history, metric_name, x_axis='step'):
    """Extract x and y values for a given metric from history."""
    xs = []
    ys = []
    for entry in history:
        if metric_name in entry:
            ys.append(entry[metric_name])
            # Try to find x_axis, default to index if not found
            if x_axis in entry:
                xs.append(entry[x_axis])
            else:
                # If x_axis is not in this specific entry (e.g. validation metrics might log epoch separately)
                # we might need to look for it. 
                # However, wandb.log usually includes the step metric if defined.
                # In our case, training_sparse logs 'step' with training metrics
                # and 'epoch' with validation metrics.
                xs.append(len(xs)) 
    return xs, ys

def plot_results(data, filename):
    if not data:
        return

    # Extract L0 values and sort them
    l0_values = sorted([float(k) for k in data.keys()])
    
    # Setup colors - Blue to Red gradient
    # RdBu_r goes from Blue (0.0) to Red (1.0)
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(l0_values))]
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Parameter Scan Results: final_L0 (Source: {filename})', fontsize=16)
    
    # 1. Train Loss
    ax = axes[0, 0]
    for i, l0 in enumerate(l0_values):
        history = data[str(l0)]['history']
        steps, losses = extract_series(history, 'training/loss', 'step')
        if steps:
            ax.plot(steps, losses, label=f'L0={l0}', color=colors[i], alpha=0.8)
    ax.set_title('Training Loss vs Step')
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.legend(fontsize='small')
    ax.grid(True, alpha=0.3)

    # 2. Train Accuracy
    ax = axes[0, 1]
    for i, l0 in enumerate(l0_values):
        history = data[str(l0)]['history']
        steps, accs = extract_series(history, 'training/accuracy', 'step')
        if steps:
            ax.plot(steps, accs, label=f'L0={l0}', color=colors[i], alpha=0.8)
    ax.set_title('Training Accuracy vs Step')
    ax.set_xlabel('Step')
    ax.set_ylabel('Accuracy')
    ax.grid(True, alpha=0.3)

    # 3. Validation Loss
    ax = axes[1, 0]
    for i, l0 in enumerate(l0_values):
        history = data[str(l0)]['history']
        epochs, losses = extract_series(history, 'validation/loss', 'epoch')
        if epochs:
            ax.plot(epochs, losses, label=f'L0={l0}', color=colors[i], marker='o', markersize=4)
    ax.set_title('Validation Loss vs Epoch')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.grid(True, alpha=0.3)

    # 4. Validation Accuracy
    ax = axes[1, 1]
    for i, l0 in enumerate(l0_values):
        history = data[str(l0)]['history']
        epochs, accs = extract_series(history, 'validation/accuracy', 'epoch')
        if epochs:
            ax.plot(epochs, accs, label=f'L0={l0}', color=colors[i], marker='o', markersize=4)
    ax.set_title('Validation Accuracy vs Epoch')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save plot
    plot_filename = filename.replace('.json', '.png')
    plt.savefig(plot_filename, dpi=150)
    print(f"Plots saved to {plot_filename}")
    
    # Create summary plot (Final Performance vs L0)
    plt.figure(figsize=(10, 6))
    final_accs = []
    final_losses = []
    
    for l0 in l0_values:
        history = data[str(l0)]['history']
        _, accs = extract_series(history, 'validation/accuracy', 'epoch')
        _, losses = extract_series(history, 'validation/loss', 'epoch')
        
        if accs: final_accs.append(accs[-1])
        else: final_accs.append(0)
            
        if losses: final_losses.append(losses[-1])
        else: final_losses.append(0)
            
    plt.plot(l0_values, final_accs, 'o-', label='Final Val Accuracy', color='blue')
    plt.plot(l0_values, final_losses, 's-', label='Final Val Loss', color='red')
    
    plt.title('Final Performance vs Target Sparsity (L0)')
    plt.xlabel('Target L0 (Fraction of Non-zero Weights)')
    plt.ylabel('Metric Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    summary_filename = filename.replace('.json', '_summary.png')
    plt.savefig(summary_filename, dpi=150)
    print(f"Summary plot saved to {summary_filename}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        with open(filename, 'r') as f:
            data = json.load(f)
        plot_results(data, filename)
    else:
        data, filename = load_latest_results()
        if data:
            plot_results(data, filename)
