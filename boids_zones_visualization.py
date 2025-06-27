import matplotlib.pyplot as plt
import numpy as np

def create_boids_zones_visualization():
    # Create figure and axis with a white background
    plt.style.use('default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5),
                                  gridspec_kw={'width_ratios': [1, 1]})
    fig.patch.set_facecolor('white')
    
    # Define radii for the three zones (from largest to smallest)
    radii = [3, 2, 1]
    
    # Create circles with blue colors (from lightest to darkest)
    colors = ['#E6F3FF',  # Light blue for repulsion
             '#4A90E2',   # Medium blue for parallel-orientation
             '#0047AB']   # Dark blue for attraction
    
    # Draw circles from largest to smallest
    for r, color in zip(radii, colors):
        circle = plt.Circle((0, 0), r, fill=True, color=color)
        ax1.add_artist(circle)
    
    # Add central fish agent
    ax1.plot(0, 0, 'ko', markersize=8)
    
    # Set equal aspect ratio and limits
    ax1.set_aspect('equal')
    ax1.set_xlim(-3.5, 3.5)
    ax1.set_ylim(-3.5, 3.5)
    
    # Remove axes and border
    ax1.set_xticks([])
    ax1.set_yticks([])
    for spine in ax1.spines.values():
        spine.set_visible(False)
    
    # Create legend items (order matches the visualization from outside to inside)
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor=colors[0], label='Repulsion zone'),
        plt.Rectangle((0, 0), 1, 1, facecolor=colors[1], label='Parallel-orientation zone'),
        plt.Rectangle((0, 0), 1, 1, facecolor=colors[2], label='Attraction zone'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black',
                  markersize=8, label='Fish agent')
    ]
    
    # Add legend to the right subplot with larger font size
    ax2.legend(handles=legend_elements, loc='center', fontsize=16,
              bbox_to_anchor=(0, 0.5))
    ax2.axis('off')
    
    # Add title
    fig.suptitle('The different zones of the boids rules, visualized', 
                 fontsize=14, y=1.05)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('boids_zones.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    create_boids_zones_visualization() 