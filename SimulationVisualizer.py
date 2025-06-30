import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import sys
import importlib
from parameters import *

class SimulationVisualizer:
    def __init__(self):
        """Initialize the simulation visualizer"""
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.ax.set_facecolor('lightskyblue')
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.set_xlim([-half_length_area, half_length_area])
        self.ax.set_ylim([-half_length_area, half_length_area])
        self.ax.grid(False)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        
        # Create empty plots for different agent types that will be updated
        # Fisher plots by trait type
        colors = np.linspace(0, 1, 5)
        mymap = plt.get_cmap("Greys")
        self.fisher_colors = mymap(colors)
        
        self.fisher_plots = {
            'fully_noncoop': self.ax.plot([], [], 'o', color=self.fisher_colors[0], markersize=7.5, label='fully_noncoop')[0],
            'noncoop': self.ax.plot([], [], 'o', color=self.fisher_colors[1], markersize=7.5, label='noncoop')[0],
            'cond_coop': self.ax.plot([], [], 'o', color=self.fisher_colors[2], markersize=7.5, label='conditional_coop')[0],
            'coop': self.ax.plot([], [], 'o', color=self.fisher_colors[3], markersize=7.5, label='coop')[0],
            'fully_coop': self.ax.plot([], [], 'o', color=self.fisher_colors[4], markersize=7.5, label='fully_coop')[0]
        }
        
        # Fish plots (will be created dynamically based on subtypes)
        self.fish_plots = {}
        
        # MPA lines (will be created when needed)
        self.mpa_lines = []
        
        # Set title and legend
        # Get total steps from parameters
        total_steps = getattr(sys.modules['parameters'], 'n', 50)
        # Start at step 0 out of total_steps-1
        self.title = self.ax.set_title(f'Step 0 out of {total_steps-1}')
        self.ax.legend(numpoints=1, loc='center', bbox_to_anchor=(0.5, -0.072), ncol=3, 
                       prop={'size': 11}, facecolor='lightskyblue')
        
        plt.tight_layout()
        
        # Animation control
        self.animation = None
        self.paused = False
        
    def start_visualization(self):
        """Start the visualization animation"""
        plt.ion()  # Enable interactive mode
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
        plt.show()
        
    def _on_key_press(self, event):
        """Handle key press events for controlling the animation"""
        if event.key == ' ':  # Space bar
            self.paused = not self.paused
            if self.paused:
                self.ax.set_title(f'{self.ax.get_title()} (Paused)')
            else:
                self.ax.set_title(self.ax.get_title().replace(' (Paused)', ''))
            plt.draw()
    
    def update_visualization(self, agents, time_value):
        """Update the visualization with new agent positions"""
        if self.paused:
            return
            
        # Update title
        # Get total steps from parameters
        total_steps = getattr(sys.modules['parameters'], 'n', 50)
        # Divide time by 2 to get actual step number (since time increments twice per simulation step)
        self.title.set_text(f'Step {int(time_value/2)} out of {total_steps-1}')
        
        # Get fishers and fish
        fishermen = [ag for ag in agents if ag.type == 'fishers']
        fish = [ag for ag in agents if ag.type == 'fish']
        
        # Update fisher positions
        for trait, plot in self.fisher_plots.items():
            trait_fishers = [ag for ag in fishermen if ag.trait == trait]
            x_pos = [ag.x for ag in trait_fishers]
            y_pos = [ag.y for ag in trait_fishers]
            plot.set_data(x_pos, y_pos)
        
        # Update fish positions
        if fish:
            # Get unique subtypes
            subtypes = list(set([ag.subtype for ag in fish]))
            subtypes.sort()  # For consistent ordering
            
            # Get colormap for fish
            cmap = plt.colormaps.get_cmap('viridis')
            fish_colors = [cmap(i / max(1, len(subtypes) - 1)) for i in range(len(subtypes))]
            
            # Create or update plots for each subtype
            for i, subtype in enumerate(subtypes):
                if subtype not in self.fish_plots:
                    # Create new plot for this subtype
                    self.fish_plots[subtype] = self.ax.plot([], [], '^', color=fish_colors[i], 
                                                          markersize=3, label=f'fish {subtype}')[0]
                
                # Update positions
                subtype_fish = [ag for ag in fish if ag.subtype == subtype]
                x_pos = [ag.x for ag in subtype_fish]
                y_pos = [ag.y for ag in subtype_fish]
                self.fish_plots[subtype].set_data(x_pos, y_pos)
        
        # Draw MPA boundaries
        self._draw_mpa_boundaries(time_value)
        
        # Redraw the figure
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
    
    def _draw_mpa_boundaries(self, time_value):
        """Draw MPA boundaries based on current configuration"""
        # Clear existing MPA lines
        for line in self.mpa_lines:
            line.remove()
        self.mpa_lines = []
        
        # Check if MPA should be shown
        show_single_mpa = (MPA == 'yes' and Type_MPA == 'single' and Both == 'no') or \
                         (MPA == 'no' and Both == 'yes' and Type_MPA == 'single' and time_value <= Time_MPA)
        
        show_spaced_mpa = (MPA == 'yes' and Type_MPA == 'spaced' and Both == 'no') or \
                         (MPA == 'no' and Both == 'yes' and Type_MPA == 'spaced' and time_value <= Time_MPA)
        
        # Draw appropriate MPA boundaries
        if show_single_mpa:
            self.mpa_lines.append(self.ax.vlines(Xa, Ya, Yb, lw=2, color='k'))
            self.mpa_lines.append(self.ax.vlines(Xb, Ya, Yb, lw=2, color='k'))
            self.mpa_lines.append(self.ax.hlines(Ya, Xa, Xb, lw=2, color='k'))
            self.mpa_lines.append(self.ax.hlines(Yb, Xa, Xb, lw=2, color='k'))
        elif show_spaced_mpa:
            # First MPA
            self.mpa_lines.append(self.ax.vlines(Xm, Ym, Yn, lw=2, color='k'))
            self.mpa_lines.append(self.ax.vlines(Xn, Ym, Yn, lw=2, color='k'))
            self.mpa_lines.append(self.ax.hlines(Ym, Xm, Xn, lw=2, color='k'))
            self.mpa_lines.append(self.ax.hlines(Yn, Xm, Xn, lw=2, color='k'))
            # Second MPA
            self.mpa_lines.append(self.ax.vlines(Xp, Yp, Yq, lw=2, color='k'))
            self.mpa_lines.append(self.ax.vlines(Xq, Yp, Yq, lw=2, color='k'))
            self.mpa_lines.append(self.ax.hlines(Yp, Xp, Xq, lw=2, color='k'))
            self.mpa_lines.append(self.ax.hlines(Yq, Xp, Xq, lw=2, color='k'))

    def close(self):
        """Close the visualization"""
        plt.close(self.fig)

# Function to attach visualizer to DynamicCoop during runtime
def attach_visualizer():
    """Attach the visualizer to the DynamicCoop simulation at runtime"""
    # Get reference to DynamicCoop module
    import DynamicCoop
    
    # Create visualizer instance
    visualizer = SimulationVisualizer()
    
    # Store original functions we need to modify
    original_update = DynamicCoop.update_one_unit_time
    original_move_fisher = DynamicCoop.move_fisher
    original_observe = DynamicCoop.observe
    
    # Define a modified observe function that saves frames but doesn't display them
    def dummy_observe():
        # We need to collect data and save frames, but not display them
        fishermen = [ag for ag in DynamicCoop.agents if ag.type == 'fishers']
        fish = [ag for ag in DynamicCoop.agents if ag.type == 'fish']
        
        # Count fish inside MPAs - check coordinates against MPA boundaries
        fish_in_MPAs = 0
        
        # Get MPA parameters from the parameters module
        params = sys.modules['parameters']
        MPA = getattr(params, 'MPA', 'no')
        Both = getattr(params, 'Both', 'no')
        Type_MPA = getattr(params, 'Type_MPA', 'single')
        Time_MPA = getattr(params, 'Time_MPA', 0)
        
        # If any MPA is active
        if any([(MPA == 'yes' and Type_MPA == 'single' and Both == 'no'), 
                (MPA == 'no' and Both == 'yes' and Type_MPA =='single' and DynamicCoop.time1 <= Time_MPA)]):
            # Single MPA
            Xa = getattr(params, 'Xa', 0)
            Xb = getattr(params, 'Xb', 0)
            Ya = getattr(params, 'Ya', 0)
            Yb = getattr(params, 'Yb', 0)
            fish_in_MPAs = sum(1 for ag in fish 
                             if Xa <= ag.x <= Xb and Ya <= ag.y <= Yb)
        elif any([(MPA == 'yes' and Type_MPA == 'spaced' and Both == 'no'), 
                  (MPA == 'no' and Both == 'yes' and Type_MPA =='spaced' and DynamicCoop.time1 <= Time_MPA)]):
            # Two MPAs
            Xm = getattr(params, 'Xm', 0)
            Xn = getattr(params, 'Xn', 0)
            Ym = getattr(params, 'Ym', 0)
            Yn = getattr(params, 'Yn', 0)
            Xp = getattr(params, 'Xp', 0)
            Xq = getattr(params, 'Xq', 0)
            Yp = getattr(params, 'Yp', 0)
            Yq = getattr(params, 'Yq', 0)
            fish_in_MPAs = sum(1 for ag in fish 
                             if (Xm <= ag.x <= Xn and Ym <= ag.y <= Yn) or 
                                (Xp <= ag.x <= Xq and Yp <= ag.y <= Yq))
        
        # Update global data in DynamicCoop module
        DynamicCoop.total_fish_count.append(len(fish))
        DynamicCoop.fish_data_MPA.append(fish_in_MPAs)
        DynamicCoop.fishermen_data3.append(len(fish) - fish_in_MPAs)
        
        # Create a figure and save it for GIF creation, but don't display it
        # We need to create a minimal frame for GIF generation
        plt.ioff()  # Disable interactive mode
        fig, ax = plt.subplots(figsize=(10, 10), num=1000)  # Use a specific figure number
        ax.set_facecolor('lightskyblue')
        
        # Draw fishermen by trait
        if len(fishermen) > 0:
            traits = ['fully_noncoop', 'noncoop', 'cond_coop', 'coop', 'fully_coop']
            colors = np.linspace(0, 1, 5)
            mymap = plt.get_cmap("Greys")
            my_colors = mymap(colors)
            
            for i, trait in enumerate(traits):
                X = [ag.x for ag in fishermen if ag.trait == trait]
                Y = [ag.y for ag in fishermen if ag.trait == trait]
                if X and Y:  # Only plot if there are fishers with this trait
                    ax.plot(X, Y, 'o', color=my_colors[i], markersize=7.5, label=trait)
        
        # Draw fish (simplified to avoid having to get subtypes)
        if fish:
            X = [ag.x for ag in fish]
            Y = [ag.y for ag in fish]
            ax.plot(X, Y, '^', color='blue', markersize=3, label='fish')
        
        # Draw MPA boundaries if needed
        params = sys.modules['parameters']
        half_length_area = getattr(params, 'half_length_area', 50)
        
        if any([(MPA == 'yes' and Type_MPA == 'single' and Both == 'no'), 
                (MPA == 'no' and Both == 'yes' and Type_MPA =='single' and DynamicCoop.time1 <= Time_MPA)]):
            ax.vlines(Xa, Ya, Yb, lw=2, color='k')
            ax.vlines(Xb, Ya, Yb, lw=2, color='k')
            ax.hlines(Ya, Xa, Xb, lw=2, color='k')
            ax.hlines(Yb, Xa, Xb, lw=2, color='k')
        elif any([(MPA == 'yes' and Type_MPA == 'spaced' and Both == 'no'), 
                  (MPA == 'no' and Both == 'yes' and Type_MPA =='spaced' and DynamicCoop.time1 <= Time_MPA)]):
            ax.vlines(Xm, Ym, Yn, lw=2, color='k')
            ax.vlines(Xn, Ym, Yn, lw=2, color='k')
            ax.hlines(Ym, Xm, Xn, lw=2, color='k')
            ax.hlines(Yn, Xm, Xn, lw=2, color='k')
            ax.vlines(Xp, Yp, Yq, lw=2, color='k')
            ax.vlines(Xq, Yp, Yq, lw=2, color='k')
            ax.hlines(Yp, Xp, Xq, lw=2, color='k')
            ax.hlines(Yq, Xp, Xq, lw=2, color='k')
            
        # Set up plot appearance
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim([-half_length_area, half_length_area])
        ax.set_ylim([-half_length_area, half_length_area])
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Get total steps from parameters
        total_steps = getattr(params, 'n', 50)
        ax.set_title(f'Step {int( (DynamicCoop.time1) / 2)} out of {total_steps-1}')
        
        ax.legend(numpoints=1, loc='center', bbox_to_anchor=(0.5, -0.072), 
                 ncol=3, prop={'size': 11}, facecolor='lightskyblue')
        
        # Save the frame but don't display it
        import os
        script_dir = os.path.dirname(os.path.abspath(DynamicCoop.__file__))
        sim_output_dir = os.path.join(script_dir, "simulation_output")
        os.makedirs(sim_output_dir, exist_ok=True)
        fig.savefig(f'{sim_output_dir}/year_{int(DynamicCoop.time1):04d}.png', 
                   bbox_inches='tight', pad_inches=0, dpi=200)
        plt.close(fig)
    
    # Define wrapper function for move_fisher that updates visualization after movement
    def visualized_move_fisher(fisher):
        result = original_move_fisher(fisher)  # Call the original function
        # Only update visualization occasionally to avoid slowdown
        if fisher.num.endswith('1'):  # Only update on the first fisher of each type
            visualizer.update_visualization(DynamicCoop.agents, DynamicCoop.time1)
        return result
    
    # Define new function that calls visualizer
    def visualized_update():
        original_update()  # Call the original function
        visualizer.update_visualization(DynamicCoop.agents, DynamicCoop.time1)
    
    # Replace the functions
    DynamicCoop.update_one_unit_time = visualized_update
    DynamicCoop.move_fisher = visualized_move_fisher
    DynamicCoop.observe = dummy_observe  # Replace observe function
    
    # Show visualizer window
    visualizer.start_visualization()
    
    return visualizer

# Function to run simulation with visualization
def run_with_visualization(experiment_type='both'):
    """Run DynamicCoop simulation with real-time visualization"""
    import DynamicCoop
    importlib.reload(DynamicCoop)
    
    # Attach visualizer
    visualizer = attach_visualizer()
    
    # Run simulation
    DynamicCoop.run_simulation(experiment_type)
    
    # Clean up
    visualizer.close()

if __name__ == "__main__":
    # If run directly, start visualization
    experiment_type = sys.argv[1] if len(sys.argv) > 1 else 'both'
    run_with_visualization(experiment_type)
