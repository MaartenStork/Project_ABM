import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

def create_cooperation_scale():
    # Create figure and axis with increased width and height
    fig, ax = plt.subplots(figsize=(15, 3))

    # Create gradient data with high resolution
    x = np.linspace(0, 1, 1000)
    y = np.linspace(0, 1, 200)
    X, Y = np.meshgrid(x, y)

    # Create split gradient
    Z = np.zeros_like(X)
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            if Y[i,j] > (1 - X[i,j]):  # Above diagonal
                Z[i,j] = X[i,j]  # Left to right: dark to light
            else:  # Below diagonal
                Z[i,j] = 1 - X[i,j]  # Left to right: light to dark

    # Create custom blue colormap
    colors = [(0.2, 0.4, 0.8), (0.9, 0.95, 1)]  # Dark blue to light blue
    blue_cmap = LinearSegmentedColormap.from_list('custom_blue', colors)

    # Plot the gradient
    plt.imshow(Z, cmap=blue_cmap, aspect='auto', extent=[0, 1, 0, 1])

    # Add diagonal line
    plt.plot([0, 1], [1, 0], '-', color='white', linewidth=1.5)

    # Add cooperation categories at the bottom
    categories = ['Fully\nnon-cooperative    ', 'Non-\ncooperative', 'Conditionally\ncooperative', 'Cooperative', 'Fully\ncooperative']
    x_positions = np.linspace(0.1, 0.9, 5)
    for x, cat in zip(x_positions, categories):
        plt.text(x, -0.15, cat, ha='center', va='top', fontsize=14)

    # Add scale values at the top
    scale_values = ['1.0', '0.8', '0.6', '0.4', '0.2']
    x_positions = np.linspace(0.1, 0.9, 5)
    for x, val in zip(x_positions, scale_values):
        plt.text(x, 1.1, val, ha='center', va='bottom', fontsize=14)

    # Add "Cooperation" and "Effort" labels - moved further out
    plt.text(-0.25, 0.5, 'Cooperation', rotation=90, va='center', ha='center', fontsize=16)
    plt.text(1.25, 0.5, 'Effort', rotation=-90, va='center', ha='center', fontsize=16)

    # Add "Low" and "High" labels
    plt.text(-0.08, 0.1, 'Low', va='center', ha='right', fontsize=16)
    plt.text(-0.08, 0.9, 'High', va='center', ha='right', fontsize=16)
    plt.text(1.08, 0.1, 'High', va='center', ha='left', fontsize=16)
    plt.text(1.08, 0.9, 'Low', va='center', ha='left', fontsize=16)

    # Remove axes
    plt.axis('off')

    # Adjust layout to prevent text cutoff - increased margins for wider labels
    plt.subplots_adjust(left=0.15, right=0.85, bottom=0.2, top=0.8)

    # Save the figure with more padding for the wider labels
    plt.savefig('cooperation_scale.png', dpi=300, bbox_inches='tight', pad_inches=0.6)
    plt.close()

# Call the function to create the visualization
create_cooperation_scale() 