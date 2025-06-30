import tkinter as tk
from tkinter import ttk
import importlib
import sys
import os
import matplotlib.pyplot as plt
from types import ModuleType
from SimulationVisualizer import run_with_visualization

class CoopSimulationUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Dynamic Cooperation Simulation Parameters")
        self.root.geometry("750x650")
        
        # Create a notebook (tabbed interface)
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create tabs
        self.basic_tab = ttk.Frame(self.notebook)
        self.fisher_tab = ttk.Frame(self.notebook)
        self.mpa_tab = ttk.Frame(self.notebook)
        self.trust_tab = ttk.Frame(self.notebook)
        self.visual_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.basic_tab, text="Basic Parameters")
        self.notebook.add(self.fisher_tab, text="Fisher Parameters")
        self.notebook.add(self.mpa_tab, text="MPA Settings")
        self.notebook.add(self.trust_tab, text="Trust Parameters")
        self.notebook.add(self.visual_tab, text="Visualization")
        
        # Load current parameter values
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        import parameters as params
        self.params_module = params
        
        # Create UI elements
        self.create_basic_params()
        self.create_fisher_params()
        self.create_mpa_params()
        self.create_trust_params()
        self.create_visual_params()
        
        # Create run button
        self.run_button = tk.Button(root, text="Run Simulation", command=self.run_simulation, 
                                   bg="#4CAF50", fg="white", font=("Arial", 12, "bold"),
                                   padx=20, pady=10)
        self.run_button.pack(pady=20)
    
    def create_basic_params(self):
        # Basic simulation parameters
        frame = ttk.LabelFrame(self.basic_tab, text="Simulation Parameters")
        frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Carrying capacity
        tk.Label(frame, text="Carrying Capacity (K):").grid(row=0, column=0, sticky="w", padx=10, pady=5)
        self.K_var = tk.IntVar(value=self.params_module.K)
        tk.Entry(frame, textvariable=self.K_var, width=10).grid(row=0, column=1, sticky="w", padx=10, pady=5)
        
        # Simulation steps
        tk.Label(frame, text="Simulation Steps (n):").grid(row=1, column=0, sticky="w", padx=10, pady=5)
        self.n_var = tk.IntVar(value=self.params_module.n)
        tk.Entry(frame, textvariable=self.n_var, width=10).grid(row=1, column=1, sticky="w", padx=10, pady=5)
    
    def create_fisher_params(self):
        # Fisher parameters
        frame = ttk.LabelFrame(self.fisher_tab, text="Fisher Parameters")
        frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Number of fishers
        tk.Label(frame, text="Number of Fishers:").grid(row=0, column=0, sticky="w", padx=10, pady=5)
        self.num_fishers_var = tk.IntVar(value=self.params_module.num_fishers)
        tk.Entry(frame, textvariable=self.num_fishers_var, width=10).grid(row=0, column=1, sticky="w", padx=10, pady=5)
        
        # Cooperation scenario distribution
        tk.Label(frame, text="Cooperation Distribution").grid(row=1, column=0, columnspan=2, sticky="w", padx=10, pady=15)
        
        tk.Label(frame, text="Fully Non-cooperative:").grid(row=2, column=0, sticky="w", padx=10, pady=5)
        self.fully_noncoop_var = tk.IntVar(value=self.params_module.fully_noncoop)
        tk.Entry(frame, textvariable=self.fully_noncoop_var, width=10).grid(row=2, column=1, sticky="w", padx=10, pady=5)
        
        tk.Label(frame, text="Non-cooperative:").grid(row=3, column=0, sticky="w", padx=10, pady=5)
        self.noncoop_var = tk.IntVar(value=self.params_module.noncoop)
        tk.Entry(frame, textvariable=self.noncoop_var, width=10).grid(row=3, column=1, sticky="w", padx=10, pady=5)
        
        tk.Label(frame, text="Conditional Cooperative:").grid(row=4, column=0, sticky="w", padx=10, pady=5)
        self.cond_coop_var = tk.IntVar(value=self.params_module.cond_coop)
        tk.Entry(frame, textvariable=self.cond_coop_var, width=10).grid(row=4, column=1, sticky="w", padx=10, pady=5)
        
        tk.Label(frame, text="Cooperative:").grid(row=5, column=0, sticky="w", padx=10, pady=5)
        self.coop_var = tk.IntVar(value=self.params_module.coop)
        tk.Entry(frame, textvariable=self.coop_var, width=10).grid(row=5, column=1, sticky="w", padx=10, pady=5)
        
        tk.Label(frame, text="Fully Cooperative:").grid(row=6, column=0, sticky="w", padx=10, pady=5)
        self.fully_coop_var = tk.IntVar(value=self.params_module.fully_coop)
        tk.Entry(frame, textvariable=self.fully_coop_var, width=10).grid(row=6, column=1, sticky="w", padx=10, pady=5)
        
        # Add validation indicator
        tk.Label(frame, text="Total should equal number of fishers").grid(row=7, column=0, columnspan=2, padx=10, pady=5)
        self.validation_label = tk.Label(frame, text="✓ Valid", fg="green")
        self.validation_label.grid(row=7, column=2, padx=10, pady=5)
        
        # Add callbacks to validate
        self.fully_noncoop_var.trace("w", self.validate_fisher_counts)
        self.noncoop_var.trace("w", self.validate_fisher_counts)
        self.cond_coop_var.trace("w", self.validate_fisher_counts)
        self.coop_var.trace("w", self.validate_fisher_counts)
        self.fully_coop_var.trace("w", self.validate_fisher_counts)
        self.num_fishers_var.trace("w", self.validate_fisher_counts)
        
    def create_mpa_params(self):
        # MPA parameters
        frame = ttk.LabelFrame(self.mpa_tab, text="Marine Protected Area (MPA) Settings")
        frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # MPA presence
        tk.Label(frame, text="MPA Presence:").grid(row=0, column=0, sticky="w", padx=10, pady=5)
        self.mpa_var = tk.StringVar(value=self.params_module.MPA)
        tk.Radiobutton(frame, text="Yes", variable=self.mpa_var, value="yes").grid(row=0, column=1, sticky="w")
        tk.Radiobutton(frame, text="No", variable=self.mpa_var, value="no").grid(row=0, column=2, sticky="w")
        
        # Both (part-time MPA)
        tk.Label(frame, text="Part-time MPA:").grid(row=1, column=0, sticky="w", padx=10, pady=5)
        self.both_var = tk.StringVar(value=self.params_module.Both)
        tk.Radiobutton(frame, text="Yes", variable=self.both_var, value="yes").grid(row=1, column=1, sticky="w")
        tk.Radiobutton(frame, text="No", variable=self.both_var, value="no").grid(row=1, column=2, sticky="w")
        
        # MPA active time
        tk.Label(frame, text="MPA Active Time:").grid(row=2, column=0, sticky="w", padx=10, pady=5)
        self.time_mpa_var = tk.IntVar(value=self.params_module.Time_MPA)
        tk.Entry(frame, textvariable=self.time_mpa_var, width=10).grid(row=2, column=1, sticky="w", padx=10, pady=5)
        
        # MPA type
        tk.Label(frame, text="MPA Configuration:").grid(row=3, column=0, sticky="w", padx=10, pady=5)
        self.type_mpa_var = tk.StringVar(value=self.params_module.Type_MPA)
        tk.Radiobutton(frame, text="Single", variable=self.type_mpa_var, value="single").grid(row=3, column=1, sticky="w")
        tk.Radiobutton(frame, text="Spaced", variable=self.type_mpa_var, value="spaced").grid(row=3, column=2, sticky="w")
        
        # MPA distance
        tk.Label(frame, text="Distance Between MPAs:").grid(row=4, column=0, sticky="w", padx=10, pady=5)
        self.dist_mpa_var = tk.DoubleVar(value=self.params_module.Dist_MPA)
        tk.Entry(frame, textvariable=self.dist_mpa_var, width=10).grid(row=4, column=1, sticky="w", padx=10, pady=5)
        
        # MPA fraction
        tk.Label(frame, text="MPA Coverage Fraction:").grid(row=5, column=0, sticky="w", padx=10, pady=5)
        self.frac_mpa_var = tk.DoubleVar(value=self.params_module.Frac_MPA)
        tk.Entry(frame, textvariable=self.frac_mpa_var, width=10).grid(row=5, column=1, sticky="w", padx=10, pady=5)
    
    def create_trust_params(self):
        # Trust parameters
        frame = ttk.LabelFrame(self.trust_tab, text="Trust and Behavioral Parameters")
        frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Trust parameters
        tk.Label(frame, text="Initial Trust:").grid(row=0, column=0, sticky="w", padx=10, pady=5)
        self.initial_trust_var = tk.DoubleVar(value=self.params_module.initial_trust)
        tk.Entry(frame, textvariable=self.initial_trust_var, width=10).grid(row=0, column=1, sticky="w", padx=10, pady=5)
        
        tk.Label(frame, text="Trust Increase:").grid(row=1, column=0, sticky="w", padx=10, pady=5)
        self.trust_increase_var = tk.DoubleVar(value=self.params_module.trust_increase)
        tk.Entry(frame, textvariable=self.trust_increase_var, width=10).grid(row=1, column=1, sticky="w", padx=10, pady=5)
        
        tk.Label(frame, text="Trust Decrease:").grid(row=2, column=0, sticky="w", padx=10, pady=5)
        self.trust_decrease_var = tk.DoubleVar(value=self.params_module.trust_decrease)
        tk.Entry(frame, textvariable=self.trust_decrease_var, width=10).grid(row=2, column=1, sticky="w", padx=10, pady=5)
        
        tk.Label(frame, text="Trust Radius:").grid(row=3, column=0, sticky="w", padx=10, pady=5)
        self.trust_radius_var = tk.DoubleVar(value=self.params_module.trust_radius)
        tk.Entry(frame, textvariable=self.trust_radius_var, width=10).grid(row=3, column=1, sticky="w", padx=10, pady=5)
        
        tk.Label(frame, text="Trust Memory:").grid(row=4, column=0, sticky="w", padx=10, pady=5)
        self.trust_memory_var = tk.IntVar(value=self.params_module.trust_memory)
        tk.Entry(frame, textvariable=self.trust_memory_var, width=10).grid(row=4, column=1, sticky="w", padx=10, pady=5)
        
        tk.Label(frame, text="Trust Threshold:").grid(row=5, column=0, sticky="w", padx=10, pady=5)
        self.trust_threshold_var = tk.DoubleVar(value=self.params_module.trust_threshold)
        tk.Entry(frame, textvariable=self.trust_threshold_var, width=10).grid(row=5, column=1, sticky="w", padx=10, pady=5)
        
        # Threshold behavior parameters
        tk.Label(frame, text="Fish Density Threshold:").grid(row=0, column=2, sticky="w", padx=10, pady=5)
        self.fish_density_threshold_var = tk.DoubleVar(value=self.params_module.fish_density_threshold)
        tk.Entry(frame, textvariable=self.fish_density_threshold_var, width=10).grid(row=0, column=3, sticky="w", padx=10, pady=5)
        
        tk.Label(frame, text="Threshold Radius:").grid(row=1, column=2, sticky="w", padx=10, pady=5)
        self.threshold_radius_var = tk.DoubleVar(value=self.params_module.threshold_radius)
        tk.Entry(frame, textvariable=self.threshold_radius_var, width=10).grid(row=1, column=3, sticky="w", padx=10, pady=5)
        
        tk.Label(frame, text="Threshold Memory:").grid(row=2, column=2, sticky="w", padx=10, pady=5)
        self.threshold_memory_var = tk.IntVar(value=self.params_module.threshold_memory)
        tk.Entry(frame, textvariable=self.threshold_memory_var, width=10).grid(row=2, column=3, sticky="w", padx=10, pady=5)
        
        tk.Label(frame, text="Cooperation Increase:").grid(row=3, column=2, sticky="w", padx=10, pady=5)
        self.cooperation_increase_var = tk.DoubleVar(value=self.params_module.cooperation_increase)
        tk.Entry(frame, textvariable=self.cooperation_increase_var, width=10).grid(row=3, column=3, sticky="w", padx=10, pady=5)
        
    def create_visual_params(self):
        # Visualization parameters
        frame = ttk.LabelFrame(self.visual_tab, text="Visualization Parameters")
        frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Plot update frequency
        tk.Label(frame, text="Plot Update Frequency:").grid(row=0, column=0, sticky="w", padx=10, pady=5)
        self.plot_update_freq_var = tk.IntVar(value=self.params_module.plot_update_freq)
        tk.Entry(frame, textvariable=self.plot_update_freq_var, width=10).grid(row=0, column=1, sticky="w", padx=10, pady=5)
        tk.Label(frame, text="(update every X steps)").grid(row=0, column=2, sticky="w", pady=5)
        
        # Create radio buttons for visualization options
        self.visualization_var = tk.StringVar(value="none")
        
        # No visualization
        tk.Radiobutton(frame, text="No Live Visualization", 
                      variable=self.visualization_var, value="none").grid(row=1, column=0, columnspan=3, sticky="w", padx=10, pady=5)
        
        # Live plots only
        tk.Radiobutton(frame, text="Show Live Plots (Fish Population & Catch)", 
                      variable=self.visualization_var, value="plots").grid(row=2, column=0, columnspan=3, sticky="w", padx=10, pady=5)
                      
        # Movement visualization only
        tk.Radiobutton(frame, text="Show Fish and Fisher Movement in Real-time", 
                      variable=self.visualization_var, value="movement").grid(row=3, column=0, columnspan=3, sticky="w", padx=10, pady=5)
        
        # Add tooltip or help text
        tk.Label(frame, text="(Press space bar to pause/resume the movement visualization)", 
                 fg="gray").grid(row=4, column=0, columnspan=3, sticky="w", padx=30, pady=2)
    
    def validate_fisher_counts(self, *args):
        try:
            total = (self.fully_noncoop_var.get() + 
                    self.noncoop_var.get() + 
                    self.cond_coop_var.get() + 
                    self.coop_var.get() + 
                    self.fully_coop_var.get())
            
            if total == self.num_fishers_var.get():
                self.validation_label.config(text="✓ Valid", fg="green")
                return True
            else:
                self.validation_label.config(text="✗ Invalid: Sum = " + str(total), fg="red")
                return False
        except:
            self.validation_label.config(text="✗ Input Error", fg="red")
            return False
    
    def update_parameters(self):
        """Update the parameters module with the UI values"""
        # Basic parameters
        self.params_module.K = self.K_var.get()
        self.params_module.n = self.n_var.get()
        
        # Fisher parameters
        self.params_module.num_fishers = self.num_fishers_var.get()
        self.params_module.fully_noncoop = self.fully_noncoop_var.get()
        self.params_module.noncoop = self.noncoop_var.get()
        self.params_module.cond_coop = self.cond_coop_var.get()
        self.params_module.coop = self.coop_var.get()
        self.params_module.fully_coop = self.fully_coop_var.get()
        
        # MPA parameters
        self.params_module.MPA = self.mpa_var.get()
        self.params_module.Both = self.both_var.get()
        self.params_module.Time_MPA = self.time_mpa_var.get()
        self.params_module.Type_MPA = self.type_mpa_var.get()
        self.params_module.Dist_MPA = self.dist_mpa_var.get()
        self.params_module.Frac_MPA = self.frac_mpa_var.get()
        
        # Trust parameters
        self.params_module.initial_trust = self.initial_trust_var.get()
        self.params_module.trust_increase = self.trust_increase_var.get()
        self.params_module.trust_decrease = self.trust_decrease_var.get()
        self.params_module.trust_radius = self.trust_radius_var.get()
        self.params_module.trust_memory = self.trust_memory_var.get()
        self.params_module.trust_threshold = self.trust_threshold_var.get()
        
        # Threshold parameters
        self.params_module.fish_density_threshold = self.fish_density_threshold_var.get()
        self.params_module.threshold_radius = self.threshold_radius_var.get()
        self.params_module.threshold_memory = self.threshold_memory_var.get()
        self.params_module.cooperation_increase = self.cooperation_increase_var.get()
        
        # Visual parameters
        self.params_module.plot_update_freq = self.plot_update_freq_var.get()
    
    def patch_matplotlib(self):
        """Fix matplotlib compatibility issues"""
        # Fix for the colormap issue
        if not hasattr(plt.colormaps, 'get_cmap') and hasattr(plt, 'cm') and hasattr(plt.cm, 'get_cmap'):
            plt.colormaps.get_cmap = plt.cm.get_cmap

    def run_simulation(self):
        """Update parameters and run the simulation"""
        if not self.validate_fisher_counts():
            import tkinter.messagebox as messagebox
            messagebox.showerror("Invalid Configuration", 
                               "The sum of fisher types must equal the total number of fishers.")
            return
            
        # Update parameters
        self.update_parameters()
        
        # Apply matplotlib patches
        self.patch_matplotlib()
        
        # Set visualization flags based on radio button selection
        enable_live_plotting = (self.visualization_var.get() == "plots")
        enable_movement_visualization = (self.visualization_var.get() == "movement")
        
        # Set the module-level flag for live plotting
        if hasattr(self.params_module, 'enable_live_plotting'):
            self.params_module.enable_live_plotting = enable_live_plotting
        else:
            # Add the attribute if it doesn't exist
            setattr(self.params_module, 'enable_live_plotting', enable_live_plotting)
            
        # Hide UI temporarily
        self.root.withdraw()
        
        try:
            # Import DynamicCoop.py
            import DynamicCoop
            # Force reload to ensure it sees updated parameters
            importlib.reload(DynamicCoop)
            
            # Determine which run method to use based on visualization option
            experiment_label = 'both'  # You could make this an option in the UI
            
            if self.visualization_var.get() == "movement":
                # Run with movement visualization
                run_with_visualization(experiment_label)
            else:
                # Run with standard visualization
                DynamicCoop.run_simulation(experiment_label)
        finally:
            # Show UI again
            self.root.deiconify()


if __name__ == "__main__":
    root = tk.Tk()
    app = CoopSimulationUI(root)
    root.mainloop()
