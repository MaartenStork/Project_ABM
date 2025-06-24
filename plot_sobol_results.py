"""
Visualization script for Sobol sensitivity analysis results.
Creates two types of plots:
1. Bar plots of first-order and total-order sensitivity indices
2. Heatmaps of second-order interactions
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from datetime import datetime
from glob import glob

# Set up the plotting style
plt.style.use('default')

def load_latest_results():
    """Load the most recent Sobol results file"""
    results_dir = 'focused_sensitivity_results'
    files = glob(os.path.join(results_dir, 'sobol_results_*.npy'))
    if not files:
        raise FileNotFoundError("No Sobol results files found!")
    
    latest_file = max(files, key=os.path.getctime)
    print(f"Loading results from {latest_file}")
    return np.load(latest_file, allow_pickle=True)

def plot_sensitivity_indices(Si_all, problem, timestamp):
    """Create bar plots of first-order and total-order sensitivity indices"""
    metrics = ['Fish Population', 'Total Catch', 'Population Stability', 
              'Sustainability', 'Cooperation Level', 'Trust Level']
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 20))
    axes = axes.flatten()
    
    for idx, (metric, ax) in enumerate(zip(metrics, axes)):
        Si = Si_all[idx]
        
        # Data preparation
        indices = pd.DataFrame({
            'Parameter': problem['names'] * 2,
            'Index Type': ['First-order'] * len(problem['names']) + ['Total-order'] * len(problem['names']),
            'Value': np.concatenate([Si['S1'], Si['ST']]),
            'Confidence': np.concatenate([Si['S1_conf'], Si['ST_conf']])
        })
        
        # Create grouped bar plot
        sns.barplot(
            data=indices,
            x='Parameter',
            y='Value',
            hue='Index Type',
            ax=ax,
            capsize=0.1
        )
        
        # Customize plot
        ax.set_title(f'{metric} Sensitivity Indices')
        ax.set_xlabel('Parameters')
        ax.set_ylabel('Sensitivity Index')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'focused_sensitivity_results/sobol_indices_{timestamp}.png', 
                dpi=300, bbox_inches='tight')
    print(f"Saved sensitivity indices plot to focused_sensitivity_results/sobol_indices_{timestamp}.png")

def plot_interactions(Si_all, problem, timestamp):
    """Create heatmaps of second-order interactions"""
    metrics = ['Fish Population', 'Total Catch', 'Population Stability', 
              'Sustainability', 'Cooperation Level', 'Trust Level']
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 20))
    axes = axes.flatten()
    
    for idx, (metric, ax) in enumerate(zip(metrics, axes)):
        Si = Si_all[idx]
        
        # Create heatmap of second-order indices
        sns.heatmap(
            Si['S2'],
            ax=ax,
            xticklabels=problem['names'],
            yticklabels=problem['names'],
            cmap='YlOrRd',
            center=0,
            annot=True,
            fmt='.2f'
        )
        
        ax.set_title(f'{metric} Parameter Interactions')
    
    plt.tight_layout()
    plt.savefig(f'focused_sensitivity_results/sobol_interactions_{timestamp}.png',
                dpi=300, bbox_inches='tight')
    print(f"Saved interactions plot to focused_sensitivity_results/sobol_interactions_{timestamp}.png")

def main():
    # Load results
    Si_all = load_latest_results()
    
    # Define problem for visualization
    problem = {
        'num_vars': 5,
        'names': ['scale', 'imitation_period', 'cooperation_increase', 'q', 'trust_decrease']
    }
    
    # Create timestamp for unique filenames
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create plots
    plot_sensitivity_indices(Si_all, problem, timestamp)
    plot_interactions(Si_all, problem, timestamp)
    
    print("\nVisualization complete!")

if __name__ == '__main__':
    main() 