"""
Sensitivity Analysis for DynamicCoop Model

This script performs three types of sensitivity analysis:
1. One-Factor-At-A-Time (OFAT) analysis
2. Sobol sensitivity analysis 
3. Morris elementary effects screening

Usage:
- Full analysis: python3 sens_anl.py
- Quick test: python3 sens_anl.py quick

Results interpretation:
- Morris: Use for initial screening - high μ* and σ indicates important/interactive parameters
- Sobol: Use for quantitative importance ranking - higher S1 = more influential
- OFAT: Use for detailed response curves - shows how output changes with each parameter
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import DynamicCoop as dc
import parameters
import pandas as pd
import os

# For Sobol sampling
from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol as sobol_analyze
# For Morris screening
from SALib.sample import morris as morris_sample
from SALib.analyze import morris as morris_analyze


def run_model(n_timesteps=350):
    """
    Re-initialize and run the DynamicCoop model for n_timesteps,
    return final total fish count.
    
    Using 350 timesteps as specified in the methods section to capture
    mature system behavior and enable proper equilibrium detection.
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
    # Add reproduction_rate as a parameter if it doesn't exist
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
        
        # # Exclude all parameters except our five key ones
        # 'reproduction_rate', 'trust_memory', 'imitation_prob', 'threshold_radius',
        # 'fully_coop', 'trust_radius', 'trust_increase', 'move_fishers', 'K',
        # 'rad_orientation', 'imitation_radius', 'rad_attraction', 'r',
        # 'cond_coop', 'imitation_nudge_factor', 'fish_density_threshold',
        # 'noncoop', 'initial_trust', 'coop', 'num_fishers', 'fully_noncoop',
        # 'rad_repulsion', 'trust_threshold', 'threshold_memory'
    }
    names = [p for p in names if p not in exclude]
    names.sort()
    
    # Store original values for restoration
    original_values = {p: getattr(parameters, p) for p in names}
    
    # bounds ±50%
    bounds = [[getattr(parameters, p)*0.5, getattr(parameters, p)*1.5] for p in names]
    return names, bounds, original_values


def update_parameters(param_names, param_values):
    """Update parameters and handle derived parameters"""
    for name, val in zip(param_names, param_values):
        setattr(parameters, name, val)
    
    # Handle derived parameters that depend on scale
    if 'scale' in param_names:
        scale_val = getattr(parameters, 'scale')
        parameters.rad_repulsion = 0.025 * scale_val
        parameters.rad_orientation = 0.06 * scale_val
        parameters.rad_attraction = 0.1 * scale_val
        parameters.rad_repulsion_sqr = parameters.rad_repulsion ** 2
        parameters.rad_orientation_sqr = parameters.rad_orientation ** 2
        parameters.rad_attraction_sqr = parameters.rad_attraction ** 2


def restore_parameters(param_names, original_values):
    """Restore parameters to their original values after analysis"""
    for name in param_names:
        if name in original_values:
            setattr(parameters, name, original_values[name])
    
    # Restore derived parameters if scale was included
    if 'scale' in param_names and 'scale' in original_values:
        scale_val = original_values['scale']
        parameters.rad_repulsion = 0.025 * scale_val
        parameters.rad_orientation = 0.06 * scale_val
        parameters.rad_attraction = 0.1 * scale_val
        parameters.rad_repulsion_sqr = parameters.rad_repulsion ** 2
        parameters.rad_orientation_sqr = parameters.rad_orientation ** 2
        parameters.rad_attraction_sqr = parameters.rad_attraction ** 2


def ofat_analysis(param_names, original_values, n_points=25, n_reps=600):
    """One-Factor-At-A-Time analysis with statistically significant sample sizes
    
    Using 600 repetitions per parameter value and 25 parameter values per factor
    as specified in the methods section, totaling 15,000 model runs per parameter.
    """
    ofat_results = {}
    for p in param_names:
        x = np.linspace(original_values[p]*0.5, original_values[p]*1.5, n_points)
        means = np.zeros(n_points)
        stds  = np.zeros(n_points)
        for i, xi in enumerate(tqdm(x, desc=f"OFAT {p}")):
            setattr(parameters, p, xi)
            
            # Handle derived parameters that depend on scale
            if p == 'scale':
                parameters.rad_repulsion = 0.025 * xi
                parameters.rad_orientation = 0.06 * xi
                parameters.rad_attraction = 0.1 * xi
                parameters.rad_repulsion_sqr = parameters.rad_repulsion ** 2
                parameters.rad_orientation_sqr = parameters.rad_orientation ** 2
                parameters.rad_attraction_sqr = parameters.rad_attraction ** 2
            
            Y = [run_model() for _ in range(n_reps)]
            means[i] = np.mean(Y)
            stds[i]  = np.std(Y, ddof=1)
        ofat_results[p] = (x, means, stds)
        setattr(parameters, p, original_values[p])  # Restore original value
        
        # Restore derived parameters if we changed scale
        if p == 'scale':
            parameters.rad_repulsion = 0.025 * original_values['scale']
            parameters.rad_orientation = 0.06 * original_values['scale']
            parameters.rad_attraction = 0.1 * original_values['scale']
            parameters.rad_repulsion_sqr = parameters.rad_repulsion ** 2
            parameters.rad_orientation_sqr = parameters.rad_orientation ** 2
            parameters.rad_attraction_sqr = parameters.rad_attraction ** 2
    return ofat_results


def plot_ofat(ofat_results, filename='ofat_full.png'):
    n = len(ofat_results)
    cols = 5  # Changed from 3 to 5 columns
    rows = int(np.ceil(n/cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))  # Adjusted figure size for better fit
    axes = axes.flatten()
    
    # Define the order of parameters for the first row
    first_row_params = ['scale', 'imitation_period', 'cooperation_increase', 'q', 'trust_decrease']
    # Get remaining parameters
    other_params = [p for p in ofat_results.keys() if p not in first_row_params]
    
    # Plot first row parameters
    for i, param in enumerate(first_row_params):
        if param in ofat_results:
            x, m, s = ofat_results[param]
            axes[i].plot(x, m, '-', color='#1f77b4', lw=2)
            axes[i].fill_between(x, m-s, m+s, color='#1f77b4', alpha=0.2)
            axes[i].set_title(param)
            axes[i].set_xlabel(param)
            axes[i].set_ylabel('Final fish')
            axes[i].grid(True, alpha=0.3)
    
    # Plot remaining parameters
    for i, param in enumerate(other_params):
        ax_idx = i + len(first_row_params)
        if ax_idx < len(axes):
            x, m, s = ofat_results[param]
            axes[ax_idx].plot(x, m, '-', color='#1f77b4', lw=2)
            axes[ax_idx].fill_between(x, m-s, m+s, color='#1f77b4', alpha=0.2)
            axes[ax_idx].set_title(param)
            axes[ax_idx].set_xlabel(param)
            axes[ax_idx].set_ylabel('Final fish')
            axes[ax_idx].grid(True, alpha=0.3)
    
    # Hide any unused subplots
    for ax in axes[len(ofat_results):]:
        ax.set_visible(False)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"OFAT results saved to {filename}")
    plt.show()


def sobol_analysis(param_names, bounds, original_values, sample_size=2500, sobol_type='all'):
    """
    Sobol sensitivity analysis with options to run specific Sobol index types
    
    Using Saltelli sampling scheme with N=2500 base sample size as specified in
    the methods section, yielding 150,000 model evaluations for 29 parameters
    (2500 × (2×29 + 2)) for robust estimation of Sobol indices at 95% confidence.
    
    Parameters:
    -----------
    param_names : list
        List of parameter names (29 numerical parameters)
    bounds : list
        Parameter bounds (±50% of baseline values)
    original_values : list
        Original parameter values to restore
    sample_size : int
        Number of samples for Sobol analysis (default 2500)
    sobol_type : str
        Type of Sobol analysis to run:
        - 'all': Run all indices (first-order, second-order, total-order) [default]
        - 'first': Run only first-order indices (S1)
        - 'second': Run only second-order indices (S2) 
        - 'total': Run only total-order indices (ST)
        - 'first_total': Run first-order and total-order (most common combination)
    
    Returns:
    --------
    dict : Sobol indices results
    """
    valid_types = ['all', 'first', 'second', 'total', 'first_total']
    if sobol_type not in valid_types:
        raise ValueError(f"sobol_type must be one of {valid_types}")

    save_file = f'sobol_output_{sobol_type}_{sample_size}.csv'
    if os.path.exists(save_file):
        os.remove(save_file)

    problem = {'num_vars': len(param_names), 'names': param_names, 'bounds': bounds}
    
    # Determine calculation requirements
    calc_second_order = sobol_type in ['all', 'second']
    
    # Calculate total runs based on what we're computing
    if sobol_type == 'first':
        total_runs = sample_size * (len(param_names) + 2)  # Formula for first-order only
    elif sobol_type == 'second':
        total_runs = sample_size * (2*len(param_names) + 2)  # Need second-order
    elif sobol_type == 'total':
        total_runs = sample_size * (len(param_names) + 2)  # Formula for total-order
    elif sobol_type == 'first_total':
        total_runs = sample_size * (len(param_names) + 2)  # Same as first/total
    else:  # 'all'
        total_runs = sample_size * (2*len(param_names) + 2)  # Formula for all indices
    
    print(f"Sampling for Sobol ({sobol_type}): N={sample_size} (Total model runs: {total_runs})")
    
    # Generate samples
    X = sobol_sample.sample(problem, sample_size, calc_second_order=calc_second_order)
    Y = np.zeros(X.shape[0])
    batch_buffer = []
    write_batch_size = 10
    
    # Run model evaluations
    for i, xi in enumerate(tqdm(X, desc=f'Sobol runs ({sobol_type})')):
        update_parameters(param_names, xi)
        Y[i] = run_model()

        # Collect row
        row = dict(zip(param_names, xi))
        row['model_output'] = Y[i]
        batch_buffer.append(row)

        # Write batch to CSV
        if len(batch_buffer) >= write_batch_size:
            df_batch = pd.DataFrame(batch_buffer)
            df_batch.to_csv(save_file, mode='a', header=not os.path.exists(save_file), index=False)
            batch_buffer = []

    # Write any remaining rows
    if batch_buffer:
        df_batch = pd.DataFrame(batch_buffer)
        df_batch.to_csv(save_file, mode='a', header=not os.path.exists(save_file), index=False)
    
    # Restore original values
    restore_parameters(param_names, original_values)
    
    # Analyze results based on type requested
    Si = sobol_analyze.analyze(problem, Y, calc_second_order=calc_second_order, print_to_console=True)
    
    # Filter results based on requested type
    if sobol_type == 'first':
        # Keep only first-order indices
        filtered_Si = {
            'S1': Si['S1'],
            'S1_conf': Si['S1_conf'],
            'names': Si.get('names', param_names)
        }
        print("\n=== FIRST-ORDER SOBOL INDICES ONLY ===")
        for i, name in enumerate(param_names):
            print(f"{name}: S1 = {Si['S1'][i]:.3f} ± {Si['S1_conf'][i]:.3f}")
            
    elif sobol_type == 'second':
        # Keep only second-order indices
        filtered_Si = {
            'S2': Si['S2'],
            'S2_conf': Si['S2_conf'],
            'names': Si.get('names', param_names)
        }
        print("\n=== SECOND-ORDER SOBOL INDICES ONLY ===")
        print("Parameter interactions (S2):")
        for i in range(len(param_names)):
            for j in range(i+1, len(param_names)):
                print(f"{param_names[i]} × {param_names[j]}: S2 = {Si['S2'][i,j]:.3f} ± {Si['S2_conf'][i,j]:.3f}")
                
    elif sobol_type == 'total':
        # Keep only total-order indices  
        filtered_Si = {
            'ST': Si['ST'],
            'ST_conf': Si['ST_conf'],
            'names': Si.get('names', param_names)
        }
        print("\n=== TOTAL-ORDER SOBOL INDICES ONLY ===")
        for i, name in enumerate(param_names):
            print(f"{name}: ST = {Si['ST'][i]:.3f} ± {Si['ST_conf'][i]:.3f}")
            
    elif sobol_type == 'first_total':
        # Keep first-order and total-order indices
        filtered_Si = {
            'S1': Si['S1'],
            'S1_conf': Si['S1_conf'],
            'ST': Si['ST'],
            'ST_conf': Si['ST_conf'],
            'names': Si.get('names', param_names)
        }
        print("\n=== FIRST-ORDER AND TOTAL-ORDER SOBOL INDICES ===")
        for i, name in enumerate(param_names):
            print(f"{name}: S1 = {Si['S1'][i]:.3f} ± {Si['S1_conf'][i]:.3f}, ST = {Si['ST'][i]:.3f} ± {Si['ST_conf'][i]:.3f}")
            
    else:  # 'all'
        filtered_Si = Si
        print("\n=== ALL SOBOL INDICES ===")
        print("First-order indices:")
        for i, name in enumerate(param_names):
            print(f"{name}: S1 = {Si['S1'][i]:.3f} ± {Si['S1_conf'][i]:.3f}")
        print("\nTotal-order indices:")
        for i, name in enumerate(param_names):
            print(f"{name}: ST = {Si['ST'][i]:.3f} ± {Si['ST_conf'][i]:.3f}")
        print("\nTop second-order interactions:")
        interactions = []
        for i in range(len(param_names)):
            for j in range(i+1, len(param_names)):
                interactions.append((param_names[i], param_names[j], Si['S2'][i,j]))
        interactions.sort(key=lambda x: abs(x[2]), reverse=True)
        for p1, p2, s2 in interactions[:5]:  # Show top 5 interactions
            print(f"{p1} × {p2}: S2 = {s2:.3f}")
    
    return filtered_Si


def plot_sobol(Si, param_names, filename='sobol_indices.png', sobol_type='all'):
    """
    Plot Sobol indices based on available data
    
    Parameters:
    -----------
    Si : dict
        Sobol indices results
    param_names : list
        Parameter names
    filename : str
        Output filename
    sobol_type : str
        Type of analysis that was run (determines plotting style)
    """
    
    # Determine what indices are available
    has_first = 'S1' in Si
    has_total = 'ST' in Si  
    has_second = 'S2' in Si
    
    # Create appropriate subplot layout
    n_plots = sum([has_first, has_total, has_second])
    if n_plots == 0:
        print("No Sobol indices to plot!")
        return
        
    fig = plt.figure(figsize=(12, 6*n_plots))
    plot_idx = 1
    
    # First-order indices plot
    if has_first:
        S1 = np.array(Si['S1'])
        S1_conf = np.array(Si['S1_conf'])
        idx = np.argsort(-S1)
        sorted_names = [param_names[i] for i in idx]
        sorted_S1 = S1[idx]
        sorted_conf = S1_conf[idx]
        
        ax1 = plt.subplot(n_plots, 1, plot_idx)
        y_pos = np.arange(len(sorted_names))
        ax1.barh(y_pos, sorted_S1, xerr=sorted_conf, align='center', color='skyblue', ecolor='gray', capsize=4)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(sorted_names)
        ax1.invert_yaxis()
        ax1.set_xlabel('First-order Sobol index')
        ax1.set_title('First-order Sensitivity Indices (S1)')
        ax1.grid(axis='x', linestyle='--', alpha=0.5)
        plot_idx += 1
    
    # Total-order indices plot
    if has_total:
        ST = np.array(Si['ST'])
        ST_conf = np.array(Si['ST_conf'])
        idx_t = np.argsort(-ST)
        sorted_names_t = [param_names[i] for i in idx_t]
        sorted_ST = ST[idx_t]
        sorted_ST_conf = ST_conf[idx_t]
        
        ax2 = plt.subplot(n_plots, 1, plot_idx)
        y_pos = np.arange(len(sorted_names_t))
        ax2.barh(y_pos, sorted_ST, xerr=sorted_ST_conf, align='center', color='lightgreen', ecolor='gray', capsize=4)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(sorted_names_t)
        ax2.invert_yaxis()
        ax2.set_xlabel('Total-order Sobol index')
        ax2.set_title('Total-order Sensitivity Indices (ST)')
        ax2.grid(axis='x', linestyle='--', alpha=0.5)
        plot_idx += 1
    
    # Second-order interactions heatmap
    if has_second:
        ax3 = plt.subplot(n_plots, 1, plot_idx)
        S2 = Si['S2']
        
        # Make the S2 matrix symmetric by copying upper triangle to lower triangle
        S2_symmetric = S2.copy()
        for i in range(len(param_names)):
            for j in range(i):
                S2_symmetric[i,j] = S2_symmetric[j,i]
        
        im = ax3.imshow(S2_symmetric, cmap='YlOrRd', aspect='equal')
        plt.colorbar(im, ax=ax3, label='Second-order Sensitivity Index')
        
        # Add parameter names to axes with better formatting
        ax3.set_xticks(np.arange(len(param_names)))
        ax3.set_yticks(np.arange(len(param_names)))
        ax3.set_xticklabels(param_names, rotation=90, ha='center', fontsize=8)
        ax3.set_yticklabels(param_names, fontsize=8)
        
        ax3.set_title('Second-order Interactions (S2)')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout(pad=3.0, h_pad=1.0)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Sobol plot ({sobol_type}) saved to {filename}")
    plt.show()


def morris_analysis(param_names, bounds, original_values, num_trajectories=1000, grid_levels=10):
    """Morris elementary effects screening analysis
    
    Using 1000 trajectories and 10 grid levels as specified in the methods section.
    This results in 1000 × (29 + 1) = 30,000 total model runs for robust
    statistical significance of μ* and σ rankings at 95% confidence level.
    """
    problem = {'num_vars': len(param_names), 'names': param_names, 'bounds': bounds}
    print(f"Sampling for Morris: trajectories={num_trajectories}, levels={grid_levels}")
    print(f"Total model runs: {num_trajectories * (len(param_names) + 1)}")
    X = morris_sample.sample(problem, N=num_trajectories, num_levels=grid_levels, optimal_trajectories=None)
    Y = np.zeros(X.shape[0])
    for i, xi in enumerate(tqdm(X, desc='Morris runs')):
        update_parameters(param_names, xi)
        Y[i] = run_model()
    
    # Restore original values
    restore_parameters(param_names, original_values)
    
    Si = morris_analyze.analyze(problem, X, Y, conf_level=0.95, print_to_console=False)
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
    ax.set_title('Morris Elementary Effects Screening (1000 trajectories)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    print(f"Morris EE plot saved to {filename}")
    plt.show()


def focused_ofat_analysis(n_points=25, n_reps=600):
    """
    Focused OFAT analysis on the 5 most important parameters identified by Morris screening
    """
    # The 5 key parameters from Morris analysis
    key_params = {
        'scale': 2,
        'imitation_period': 5, 
        'cooperation_increase': 0.2,
        'q': 0.6,
        'trust_decrease': 0.2
    }
    
    print(f"Running focused OFAT on {len(key_params)} key parameters")
    print(f"Points per parameter: {n_points}, Repetitions per point: {n_reps}")
    
    ofat_results = {}
    
    for param_name, original_val in key_params.items():
        print(f"\nAnalyzing {param_name} (original value: {original_val})")
        
        # Use wider ranges for some parameters that might need it
        if param_name == 'scale':
            x = np.linspace(0.5, 4.0, n_points)  # Wider range for scale
        elif param_name == 'imitation_period':
            x = np.linspace(1, 15, n_points)  # Wider range, ensure integers
            x = np.round(x).astype(int)
        elif param_name == 'cooperation_increase':
            x = np.linspace(0.05, 0.5, n_points)  # Wider range
        elif param_name == 'q':
            x = np.linspace(0.2, 1.0, n_points)  # Wider range
        elif param_name == 'trust_decrease':
            x = np.linspace(0.05, 0.5, n_points)  # Wider range
        else:
            x = np.linspace(original_val*0.3, original_val*2.0, n_points)  # Default wider range
        
        means = np.zeros(n_points)
        stds = np.zeros(n_points)
        
        for i, xi in enumerate(tqdm(x, desc=f"OFAT {param_name}")):
            # Set the parameter
            setattr(parameters, param_name, xi)
            
            # Handle derived parameters for scale
            if param_name == 'scale':
                parameters.rad_repulsion = 0.025 * xi
                parameters.rad_orientation = 0.06 * xi
                parameters.rad_attraction = 0.1 * xi
                parameters.rad_repulsion_sqr = parameters.rad_repulsion ** 2
                parameters.rad_orientation_sqr = parameters.rad_orientation ** 2
                parameters.rad_attraction_sqr = parameters.rad_attraction ** 2
            
            # Run multiple repetitions
            Y = [run_model() for _ in range(n_reps)]
            means[i] = np.mean(Y)
            stds[i] = np.std(Y, ddof=1)
        
        ofat_results[param_name] = (x, means, stds)
        
        # Restore original value
        setattr(parameters, param_name, original_val)
        if param_name == 'scale':
            parameters.rad_repulsion = 0.025 * original_val
            parameters.rad_orientation = 0.06 * original_val
            parameters.rad_attraction = 0.1 * original_val
            parameters.rad_repulsion_sqr = parameters.rad_repulsion ** 2
            parameters.rad_orientation_sqr = parameters.rad_orientation ** 2
            parameters.rad_attraction_sqr = parameters.rad_attraction ** 2
        
        # Print summary
        effect_size = max(means) - min(means)
        avg_std = np.mean(stds)
        signal_to_noise = effect_size / avg_std if avg_std > 0 else 0
        print(f"  Effect size: {effect_size:.1f} fish")
        print(f"  Signal-to-noise ratio: {signal_to_noise:.2f}")
    
    return ofat_results


def main(quick_test=False, sobol_type='all'):
    """
    Main sensitivity analysis function
    
    Performs comprehensive sensitivity analysis as described in the methods section:
    - Morris Elementary Effects: 1000 trajectories, 30,000 model runs
    - OFAT Analysis: 600 repetitions per parameter value, 15,000 runs per parameter  
    - Sobol Analysis: 2500 base sample size, up to 150,000 model evaluations
    
    All analyses use 350 timesteps and ±50% parameter ranges around baseline values.
    
    Parameters:
    -----------
    quick_test : bool
        Whether to run quick test with reduced sample sizes (Sobol only)
    sobol_type : str
        Type of Sobol analysis to run:
        - 'all': All indices (first-order, second-order, total-order) [default]
        - 'first': Only first-order indices (S1)
        - 'second': Only second-order indices (S2)
        - 'total': Only total-order indices (ST)  
        - 'first_total': First-order and total-order indices
    """
    param_names, bounds, original_values = get_param_names_and_bounds()
    print(f"Parameters included ({len(param_names)}): {param_names}")
    print(f"reproduction_rate included: {'reproduction_rate' in param_names}")
    print(f"Sobol analysis type: {sobol_type}")
    
    if quick_test:
        print("\n=== QUICK TEST MODE ===")
        print("Running with reduced sample sizes for testing...")
        
        # Quick Sobol Analysis with specified type
        print(f"\n=== Running Quick Sobol Analysis ({sobol_type}) ===")
        Si_sob = sobol_analysis(param_names, bounds, original_values, sample_size=32, sobol_type=sobol_type)
        plot_sobol(Si_sob, param_names, filename=f'sobol_indices_{sobol_type}_quick.png', sobol_type=sobol_type)
        
        # Initialize placeholders for consistency
        Si_morris = None
        ofat_results = None
        
        print(f"\nQuick test completed for {sobol_type} indices. For full statistically significant results, run main(quick_test=False)")
        
    else:
        print("\n=== FULL STATISTICAL ANALYSIS ===")
        print("WARNING: This will take many hours!")
        
        # Calculate estimated runs based on sobol_type using N=2500 as per methods section
        if sobol_type == 'first':
            sobol_runs = 2500 * (len(param_names) + 2)
        elif sobol_type == 'second':
            sobol_runs = 2500 * (2*len(param_names) + 2)
        elif sobol_type == 'total':
            sobol_runs = 2500 * (len(param_names) + 2)
        elif sobol_type == 'first_total':
            sobol_runs = 2500 * (len(param_names) + 2)
        else:  # 'all'
            sobol_runs = 2500 * (2*len(param_names) + 2)  # 150,000 runs for 29 parameters
            
        total_runs_estimated = (
            len(param_names) * 25 * 600 +  # OFAT: 15,000 runs per parameter
            sobol_runs +  # Sobol with specified type
            1000 * (len(param_names) + 1)  # Morris: 30,000 runs total
        )
        print(f"Total estimated model runs: {total_runs_estimated:,}")
        
        # Morris Elementary Effects Analysis
        print(f"\n=== Running Morris Elementary Effects Analysis ===")
        morris_runs = 1000 * (len(param_names) + 1)
        print(f"Estimated runs: {morris_runs:,}")
        Si_morris = morris_analysis(param_names, bounds, original_values, num_trajectories=1000, grid_levels=10)
        plot_morris(Si_morris, param_names, filename='morris_ee_full.png')
        
        # OFAT Analysis
        print(f"\n=== Running OFAT Analysis ===")
        ofat_runs = len(param_names) * 25 * 600
        print(f"Estimated runs: {ofat_runs:,}")
        ofat_results = ofat_analysis(param_names, original_values, n_points=25, n_reps=600)
        plot_ofat(ofat_results, filename='ofat_full_600reps.png')

        # Sobol Analysis with specified type
        print(f"\n=== Running Sobol Analysis ({sobol_type}) ===")
        print(f"Estimated runs: {sobol_runs:,}")
        Si_sob = sobol_analysis(param_names, bounds, original_values, sample_size=2500, sobol_type=sobol_type)
        plot_sobol(Si_sob, param_names, filename=f'sobol_indices_{sobol_type}.png', sobol_type=sobol_type)
    
    # Ensure all parameters are restored to original values
    restore_parameters(param_names, original_values)
    print("\nAll parameters restored to original values.")
    
    if quick_test:
        return Si_sob
    else:
        # Return all analysis results for full analysis
        return {'morris': Si_morris, 'ofat': ofat_results, 'sobol': Si_sob}

if __name__=='__main__':
    import sys
    
    # Parse command line arguments
    quick_mode = False
    sobol_type = 'all'
    
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg == 'quick':
                quick_mode = True
            elif arg in ['all', 'first', 'second', 'total', 'first_total']:
                sobol_type = arg
    
    print(f"Running sensitivity analysis with:")
    print(f"  Quick mode: {quick_mode}")
    print(f"  Sobol type: {sobol_type}")
    print(f"\nUsage: python sens_anl.py [quick] [all|first|second|total|first_total]")
    print(f"Examples:")
    print(f"  python sens_anl.py quick first       # Quick test, first-order only")
    print(f"  python sens_anl.py total            # Full analysis, total-order only")
    print(f"  python sens_anl.py quick first_total # Quick test, first + total order")
    print(f"  python sens_anl.py                  # Full analysis, all indices")
    
    main(quick_test=quick_mode, sobol_type=sobol_type)
