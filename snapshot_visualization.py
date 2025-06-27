import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
import os

# Create output directory if it doesn't exist
OUTPUT_DIR = 'simulation_output/snapshots'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_snapshot(agents, time, half_length_area=50):
    """
    Create a snapshot of the underwater environment with agents and a legend.
    
    Args:
        agents: List of agents in the simulation
        time: Current simulation time
        half_length_area: Half the length of the simulation area
    """
    # Create figure with custom layout
    plt.style.use('default')  # Reset to default style
    fig = plt.figure(figsize=(15, 10))  # Reduced overall width
    
    # Create subplot layout with more space for legend
    gs = GridSpec(1, 2, width_ratios=[3, 1], figure=fig)
    gs.update(wspace=0.1)  # Minimal space between subplots
    
    # Main plot (underwater environment)
    ax_main = fig.add_subplot(gs[0])
    ax_main.set_facecolor('lightskyblue')  # Exact blue from simulation
    
    # Add a border around the main plot
    for spine in ax_main.spines.values():
        spine.set_linewidth(2)
        spine.set_color('black')
    
    # Separate agents by type
    fish_agents = []
    fishermen_by_trait = {
        'fully_noncoop': [],
        'noncoop': [],
        'cond_coop': [],
        'coop': [],
        'fully_coop': []
    }
    
    for agent in agents:
        if agent.type == 'fish':
            fish_agents.append(agent)
        elif agent.type == 'fishers':
            fishermen_by_trait[agent.trait].append(agent)
    
    # Plot fish agents in main plot
    if fish_agents:
        fish_x = [ag.x for ag in fish_agents]
        fish_y = [ag.y for ag in fish_agents]
        ax_main.scatter(fish_x, fish_y, marker='^', c='darkgreen', s=100, label='Fish')
    
    # Define colors for cooperation levels
    coop_colors = {
        'fully_coop': '#1a1a1a',      # Very dark grey
        'coop': '#4d4d4d',            # Dark grey
        'cond_coop': '#808080',       # Medium grey
        'noncoop': '#b3b3b3',         # Light grey
        'fully_noncoop': '#e6e6e6'    # Very light grey
    }
    
    # Plot fishermen
    for trait in fishermen_by_trait:
        if fishermen_by_trait[trait]:
            x_positions = [ag.x for ag in fishermen_by_trait[trait]]
            y_positions = [ag.y for ag in fishermen_by_trait[trait]]
            ax_main.scatter(x_positions, y_positions, c=coop_colors[trait], s=200)
    
    # Configure main plot
    ax_main.set_xlim([-half_length_area, half_length_area])
    ax_main.set_ylim([-half_length_area, half_length_area])
    ax_main.grid(True, alpha=0.2)
    ax_main.set_xticks([])
    ax_main.set_yticks([])
    
    # Create legend subplot
    ax_legend = fig.add_subplot(gs[1])
    ax_legend.set_facecolor('white')
    
    # Remove axes from legend
    ax_legend.set_xticks([])
    ax_legend.set_yticks([])
    
    # Add border to legend
    for spine in ax_legend.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1)
        spine.set_color('black')
    
    # Add fish to legend with text underneath
    ax_legend.scatter([0.5], [0.9], marker='^', c='darkgreen', s=200)
    ax_legend.text(0.5, 0.85, 'Fish', fontsize=16, ha='center')
    
    # Add horizontal line separator
    ax_legend.axhline(y=0.8, xmin=0.1, xmax=0.9, color='black', alpha=0.3)
    
    # Add cooperation levels with text underneath each icon
    y_positions = np.linspace(0.7, 0.2, 5)  # Adjusted spacing to fit all items
    traits = ['fully_coop', 'coop', 'cond_coop', 'noncoop', 'fully_noncoop']
    
    for y_pos, trait in zip(y_positions, traits):
        # Add the circle marker
        ax_legend.scatter([0.5], [y_pos], c=coop_colors[trait], s=200)
        # Add the trait name underneath
        trait_name = trait.replace('_', ' ').title()
        ax_legend.text(0.5, y_pos - 0.05, trait_name, 
                      fontsize=16, ha='center', va='top')
    
    # Set legend axis limits to ensure all content is visible
    ax_legend.set_ylim(0.1, 1.0)  # Ensure bottom items are visible
    
    plt.tight_layout()
    return fig

def save_snapshot(agents, time, filename='underwater_snapshot.png', half_length_area=50):
    """
    Create and save a snapshot of the underwater environment.
    
    Args:
        agents: List of agents in the simulation
        time: Current simulation time
        filename: Name of the output file
        half_length_area: Half the length of the simulation area
    """
    output_path = os.path.join(OUTPUT_DIR, filename)
    fig = create_snapshot(agents, time, half_length_area)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Snapshot saved to: {output_path}") 