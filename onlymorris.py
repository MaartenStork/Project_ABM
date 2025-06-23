"""
Morris Sensitivity Analysis for DynamicCoop Model

This script performs Morris elementary effects screening, excluding reproduction_rate parameter.

Key features:
- Uses 300 timesteps per run (vs 150) to capture more mature system behavior
- Uses statistically significant sample sizes:
  * Morris: 200 trajectories × (N params + 1) runs
- Properly restores parameters after analysis
- Better visualization with sorted results

Usage:
- Full analysis: python3 onlymorris.py
- Quick test: python3 onlymorris.py quick

Results interpretation:
- Morris: Use for initial screening - high μ* and σ indicates important/interactive parameters
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import DynamicCoop as dc
import parameters
import pandas as pd
import os
from datetime import datetime

# For Morris screening
from SALib.sample import morris as morris_sample
from SALib.analyze import morris as morris_analyze


def run_model(n_timesteps=300):
    """
    Re-initialize and run the DynamicCoop model for n_timesteps,
    return final total fish count.
    
    Using 300 timesteps instead of default 150 to capture more mature
    system behavior while maintaining computational efficiency.
    System has inherent stochasticity, so multiple reps handle variability.
    """
    dc.initialize('reproduction_rate')
    
    # If reproduction_rate parameter exists, apply it to all fish for homogeneous analysis
    if hasattr(parameters, 'reproduction_rate'):
        for agent in dc.agents:
            if agent.type == 'fish':
                setattr(agent, 'reproduction_rate', parameters.reproduction_rate)
    
    # Run for more timesteps to capture mature behavior
    for _ in range(n_timesteps):
        dc.update_one_unit_time()
    return dc.total_fish_count[-1]


def get_param_names_and_bounds():
    # Add reproduction_rate as a parameter if it doesn't exist (but we won't analyze it)
    if not hasattr(parameters, 'reproduction_rate'):
        setattr(parameters, 'reproduction_rate', 0.3)  # Set default value
    
    # gather numeric parameters
    names = [name for name, val in vars(parameters).items()
             if not name.startswith('_') and isinstance(val, (int, float))]
    exclude = {
        # Exclude non-parameter variables and fixed scenario params
        'n', 'Area', 'Length_Area', 'half_length_area',
        
        # Exclude derived parameters
        'rad_repulsion_sqr', 'rad_orientation_sqr', 'rad_attraction_sqr', 'r_sqr',
                
        # Exclude all MPA-related parameters
        'Time_MPA', 'Type_MPA', 'Dist_MPA', 'Frac_MPA', 'Half_Length',
        'Xa', 'Xb', 'Ya', 'Yb', 'Xm', 'Xn', 'Ym', 'Yn', 'Xp', 'Xq', 'Yp', 'Yq',
        
        # Exclude plotting parameter
        'plot_update_freq',
        
        # Exclude reproduction_rate from analysis
        'reproduction_rate'
    }
    names = [p for p in names if p not in exclude]
    names.sort()
    
    # Store original values for restoration
    original_values = {p: getattr(parameters, p) for p in names}
    
    # bounds ±50%
    bounds = [[getattr(parameters, p)*0.5, getattr(parameters, p)*1.5] for p in names]
    return names, bounds, original_values


def restore_parameters(param_names, original_values):
    """Restore parameters to their original values after analysis"""
    for name in param_names:
        if name in original_values:
            setattr(parameters, name, original_values[name])


def morris_analysis(param_names, bounds, original_values, num_trajectories=200, grid_levels=10):
    """Morris screening with statistically significant sample size"""
    problem = {'num_vars': len(param_names), 'names': param_names, 'bounds': bounds}
    print(f"Sampling for Morris: trajectories={num_trajectories}, levels={grid_levels}")
    print(f"Total model runs: {num_trajectories * (len(param_names) + 1)}")
    
    # Create timestamp for unique filenames
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create results directory if it doesn't exist
    os.makedirs('morris_results', exist_ok=True)
    
    # Generate Morris trajectories
    X = morris_sample.sample(problem, N=num_trajectories, num_levels=grid_levels, optimal_trajectories=None)
    Y = np.zeros(X.shape[0])
    
    # Run model and collect results
    for i, xi in enumerate(tqdm(X, desc='Morris runs')):
        for name, val in zip(param_names, xi): setattr(parameters, name, val)
        Y[i] = run_model()
    
    # Save raw trajectories and outputs
    raw_df = pd.DataFrame(X, columns=param_names)
    raw_df['output'] = Y
    raw_filename = f'morris_results/morris_raw_data_{timestamp}.csv'
    raw_df.to_csv(raw_filename, index=False)
    print(f"Raw Morris data saved to {raw_filename}")
    
    # Restore original values
    restore_parameters(param_names, original_values)
    
    # Analyze results
    Si = morris_analyze.analyze(problem, X, Y, conf_level=0.95, print_to_console=False)
    
    # Save analyzed results
    results_df = pd.DataFrame({
        'parameter': param_names,
        'mu': Si['mu'],
        'mu_star': np.abs(Si['mu']),  # Absolute mean
        'sigma': Si['sigma'],
        'mu_star_conf': Si['mu_star_conf']
    })
    
    # Sort by mu_star (absolute mean) for easier interpretation
    results_df = results_df.sort_values('mu_star', ascending=False)
    
    analyzed_filename = f'morris_results/morris_analyzed_{timestamp}.csv'
    results_df.to_csv(analyzed_filename, index=False)
    print(f"Analyzed Morris results saved to {analyzed_filename}")
    
    return Si


def plot_morris(Si, param_names, filename='morris_ee.png'):
    mu = np.abs(Si['mu'])        # absolute mean elementary effects
    sigma = Si['sigma']
    fig, ax = plt.subplots(figsize=(10,10))
    ax.scatter(mu, sigma, alpha=0.7, s=50)
    for i, name in enumerate(param_names):
        ax.annotate(name, (mu[i], sigma[i]), xytext=(5,5), textcoords='offset points', 
                   fontsize=9, alpha=0.8)
    ax.set_xlabel(r'$\mu^*$ (mean absolute EE)')
    ax.set_ylabel(r'$\sigma$ (EE standard deviation)')
    ax.set_title('Morris Elementary Effects Screening (200 trajectories)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    print(f"Morris EE plot saved to {filename}")
    plt.show()


def main(quick_test=False):
    param_names, bounds, original_values = get_param_names_and_bounds()
    print(f"Parameters included ({len(param_names)}): {param_names}")
    print(f"reproduction_rate excluded from analysis")
    
    # Create results directory if it doesn't exist
    os.makedirs('morris_results', exist_ok=True)
    
    if quick_test:
        print("\n=== QUICK TEST MODE ===")
        print("Running with reduced sample sizes for testing...")
        
        # Quick Morris Analysis
        print("\n=== Running Quick Morris Analysis ===")
        Si_m = morris_analysis(param_names, bounds, original_values, num_trajectories=10, grid_levels=4)
        plot_morris(Si_m, param_names, filename='morris_results/morris_ee_quick.png')
        
        print("\nQuick test completed. For full statistically significant results, run main(quick_test=False)")
        
    else:
        print("\n=== FULL MORRIS ANALYSIS ===")
        print(f"Total estimated model runs: {200 * (len(param_names) + 1):,}")
        
        # Morris Analysis
        print(f"\n=== Running Morris Analysis ===")
        Si_m = morris_analysis(param_names, bounds, original_values)
        plot_morris(Si_m, param_names, filename='morris_results/morris_ee.png')
    
    # Ensure all parameters are restored to original values
    restore_parameters(param_names, original_values)
    print("\nAll parameters restored to original values.")

if __name__=='__main__':
    import sys
    # Run quick test if 'quick' is passed as argument, otherwise run full analysis
    quick_mode = len(sys.argv) > 1 and sys.argv[1] == 'quick'
    main(quick_test=quick_mode)
