"""
Enhanced Morris Plot Generator

Creates a visually appealing version of the Morris sensitivity analysis plot
using the saved results data.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text  # For smart label positioning

# Read the analyzed results
df = pd.read_csv('morris_results/morris_analyzed_20250623_233107.csv')

# Set global font sizes - increased all sizes
SMALL_SIZE = 16    # Was 12
MEDIUM_SIZE = 18   # Was 14
BIGGER_SIZE = 22   # Was 16
LARGEST_SIZE = 24  # New size for main elements

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=LARGEST_SIZE)   # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)   # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)   # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)   # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# Create figure with specific size and high DPI
plt.figure(figsize=(14, 12), dpi=300)  # Increased figure size

# Set the style
plt.style.use('default')
sns.set_theme(style="whitegrid")

# Create the scatter plot
scatter = plt.scatter(df['mu_star'], df['sigma'], 
                     alpha=0.6, 
                     s=150,  # Increased point size
                     c=df['mu_star'] * df['sigma'],  # Color by importance (μ* × σ)
                     cmap='viridis')

# Create texts for smart positioning
texts = []
for idx, row in df.iterrows():
    texts.append(plt.text(row['mu_star'], row['sigma'], 
                         row['parameter'],
                         fontsize=MEDIUM_SIZE,  # Increased from 11
                         bbox=dict(facecolor='white', 
                                 edgecolor='none',
                                 alpha=0.7,
                                 pad=3)))  # Increased padding

# Use adjust_text to prevent overlap - increased spacing
adjust_text(texts,
           arrowprops=dict(arrowstyle='-', color='gray', alpha=0.5),
           expand_points=(2.0, 2.0),  # Increased from 1.5
           force_points=(0.8, 0.8),   # Increased from 0.5
           force_text=(0.8, 0.8))     # Added text force

# Add gridlines with lower opacity
plt.grid(True, alpha=0.3, linestyle='--')

# Add colorbar with larger font
cbar = plt.colorbar(scatter, label='Parameter Importance (μ* × σ)')
cbar.ax.tick_params(labelsize=MEDIUM_SIZE)
cbar.set_label('Parameter Importance (μ* × σ)', size=MEDIUM_SIZE)

# Set labels and title with LaTeX formatting - increased sizes
plt.xlabel(r'$\mu^*$ (Mean Absolute Elementary Effect)', fontsize=LARGEST_SIZE)
plt.ylabel(r'$\sigma$ (Standard Deviation of Elementary Effects)', fontsize=LARGEST_SIZE)
# plt.title('Morris Sensitivity Analysis\nParameter Importance and Interactions', 
#           fontsize=BIGGER_SIZE, pad=20)

# # Add regions description with larger font
# plt.text(0.02, 0.98, 
#          'Non-influential\nParameters', 
#          transform=plt.gca().transAxes,
#          fontsize=MEDIUM_SIZE,
#          verticalalignment='top',
#          bbox=dict(facecolor='white', alpha=0.7, pad=3))
# plt.text(0.98, 0.98,
#          'Non-linear/\nInteractive Effects',
#          transform=plt.gca().transAxes,
#          fontsize=MEDIUM_SIZE,
#          horizontalalignment='right',
#          verticalalignment='top',
#          bbox=dict(facecolor='white', alpha=0.7, pad=3))
# plt.text(0.98, 0.02,
#          'Linear Effects',
#          transform=plt.gca().transAxes,
#          fontsize=MEDIUM_SIZE,
#          horizontalalignment='right',
#          bbox=dict(facecolor='white', alpha=0.7, pad=3))

# Adjust layout with more space
plt.tight_layout(pad=1.5)  # Increased padding

# Save with high quality
plt.savefig('morris_results/morris_ee_enhanced.png', 
            dpi=300, 
            bbox_inches='tight',
            facecolor='white',
            edgecolor='none')

print("Enhanced Morris plot saved as 'morris_results/morris_ee_enhanced.png'") 