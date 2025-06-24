import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import DynamicCoop as dc
import parameters

from SALib.sample import morris as morris_sample
from SALib.analyze import morris as morris_analyze

def run_model_no_mpa():
    """
    Re-initialize and run the DynamicCoop model for n steps with no MPA,
    and return the final total fish count.
    """
    # Ensure MPA is disabled for this analysis
    parameters.MPA = 'no'
    parameters.Both = 'no'
    
    # Initialize the model
    # Using 'reproduction_rate' experiment as in sens_anl.py for consistency
    dc.initialize('reproduction_rate') 
    
    # Overwrite the reproduction rate for all fish to be a single value from the
    # sensitivity analysis sampler. This makes the population homogeneous for
    # this parameter for each model run.
    if hasattr(parameters, 'reproduction_rate'):
        for agent in dc.agents:
            if agent.type == 'fish':
                setattr(agent, 'reproduction_rate', parameters.reproduction_rate)
    
    # Run the simulation
    for _ in range(parameters.n):
        dc.update_one_unit_time()
        
    # Return the final fish count
    return dc.total_fish_count[-1]


def get_param_names_and_bounds_no_mpa():
    """
    Gathers all numerical parameters from parameters.py, excluding those
    irrelevant for a no-MPA scenario.
    """
    # Add 'reproduction_rate' to the parameters module in memory for the analysis.
    # I am setting a default value of 0.1, with bounds of +/- 50% (0.05 to 0.15).
    if not hasattr(parameters, 'reproduction_rate'):
        setattr(parameters, 'reproduction_rate', 0.1)

    # Gather all numerical parameters
    names = [name for name, val in vars(parameters).items()
             if not name.startswith('_') and isinstance(val, (int, float))]
    
    # Define parameters to exclude for a no-MPA run
    exclude = {
        # Exclude non-parameter variables and fixed scenario params
        'n', 'Area', 'Length_Area', 'half_length_area',
        
        # Exclude derived parameters
        'rad_repulsion_sqr', 'rad_orientation_sqr', 'rad_attraction_sqr', 'r_sqr',
                
        # Exclude all MPA-related parameters
        'Time_MPA', 'Type_MPA', 'Dist_MPA', 'Frac_MPA', 'Half_Length',
        'Xa', 'Xb', 'Ya', 'Yb', 'Xm', 'Xn', 'Ym', 'Yn', 'Xp', 'Xq', 'Yp', 'Yq',
        
        # Exclude plotting parameter
        'plot_update_freq'
    }
    
    names = sorted([p for p in names if p not in exclude])
    
    # Define bounds as ±50% of the default value
    bounds = [[getattr(parameters, p) * 0.5, getattr(parameters, p) * 1.5] for p in names]
    
    return names, bounds


def morris_analysis(param_names, bounds, num_trajectories=20, grid_levels=4):
    """
    Performs a Morris sensitivity analysis.
    """
    problem = {
        'num_vars': len(param_names),
        'names': param_names,
        'bounds': bounds,
        'num_levels': grid_levels
    }

    print(f"Sampling for Morris Analysis: N={num_trajectories} trajectories")
    X = morris_sample.sample(problem, N=num_trajectories, num_levels=grid_levels)
    
    Y = np.zeros(X.shape[0])
    
    # Store default parameter values
    defaults = {p: getattr(parameters, p) for p in param_names}

    print("Running model for each parameter sample...")
    for i, xi in enumerate(tqdm(X, desc='Morris Runs')):
        # Set parameters for the current run
        for name, val in zip(param_names, xi):
            setattr(parameters, name, val)
        
        # Debug: Print first few parameter combinations and results
        if i < 5:
            param_values = {name: getattr(parameters, name) for name in param_names}
            print(f"Run {i}: Parameters = {param_values}")
        
        Y[i] = run_model_no_mpa()
        
        if i < 5:
            print(f"Run {i}: Output = {Y[i]}")

    # Debug: Print summary statistics
    print(f"\nOutput summary:")
    print(f"Min: {np.min(Y)}, Max: {np.max(Y)}, Mean: {np.mean(Y):.2f}, Std: {np.std(Y):.2f}")
    
    # Restore default parameter values
    for name, value in defaults.items():
        setattr(parameters, name, value)
        
    print("Analyzing results...")
    Si = morris_analyze.analyze(problem, X, Y, print_to_console=True, num_levels=grid_levels)
    return Si


def plot_morris(Si, filename='morris_analysis_no_mpa.png'):
    """
    Plots the results of the Morris analysis (mu_star vs. sigma).
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    ax.scatter(Si['sigma'], Si['mu_star'], marker='o', c='k')

    for i, txt in enumerate(Si['names']):
        ax.annotate(txt, (Si['sigma'][i], Si['mu_star'][i]), 
                    xytext=(5,-5), textcoords='offset points')

    ax.set_ylabel(r'$\mu^*$ (mean of elementary effects)')
    ax.set_xlabel(r'$\sigma$ (std. dev. of elementary effects)')
    ax.set_title('Morris Sensitivity Analysis (No MPA Scenario)')
    
    # Add a grid and y=x line for visual reference
    ax.grid(True, linestyle='--', alpha=0.6)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.plot(xlim, xlim, 'r--', alpha=0.8, label=r'$\mu^* = \sigma$')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"\nMorris analysis plot saved to {filename}")
    plt.show()


def main():
    """
    Main function to run the Morris analysis for the no-MPA scenario.
    """
    # --- Test Mode ---
    # Set to True for a quick run with fewer parameters and trajectories.
    # Set to False for the full analysis.
    TEST_MODE = True
    # -----------------

    param_names, bounds = get_param_names_and_bounds_no_mpa()

    if TEST_MODE:
        print("--- RUNNING IN TEST MODE ---")
        # Select only the first 2 parameters for a quick test
        param_names = param_names[:2]
        bounds = bounds[:2]
        # The number of trajectories for optimal sampling must be >= 2.
        # Using 10 (as in sens_anl.py) is a safe value for testing.
        num_trajectories = 10
    else:
        num_trajectories = 20 # Default value

    # Ensure the number of trajectories is even for optimal sampling
    if num_trajectories % 2 != 0:
        num_trajectories += 1
        print(f"Warning: Number of trajectories must be even for optimal sampling. Rounded up to {num_trajectories}.")

    print("--- Running Morris Analysis for No-MPA Scenario ---")
    print(f"Parameters included ({len(param_names)}): {param_names}\n")
    
    Si = morris_analysis(param_names, bounds, num_trajectories=num_trajectories)
    
    if Si:
        plot_morris(Si)


if __name__ == '__main__':
    main() 