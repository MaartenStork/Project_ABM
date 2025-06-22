"""
Sensitivity Analysis for DynamicCoop Model

This script performs three types of sensitivity analysis:
1. One-Factor-At-A-Time (OFAT) analysis
2. Sobol sensitivity analysis 
3. Morris elementary effects screening

Key improvements:
- Now includes reproduction_rate as a parameter (was missing before)
- Uses statistically significant sample sizes:
  * OFAT: 25 points × 50 reps = 1,250 runs per parameter (~36,250 total runs)
  * Sobol: 2,048 samples × (29 params + 2) = ~63,500 runs
  * Morris: 200 trajectories × (29 params + 1) = ~6,000 runs
- Properly restores parameters after each analysis
- Better visualization with sorted results

TOTAL ESTIMATED RUNS: ~105,750 model runs

Usage:
- Full analysis: python3 sens_anl.py
- Quick test: python3 sens_anl.py quick

WARNING: Full analysis will take 6-12+ hours depending on your machine.
Consider running overnight or on a cluster. You can also comment out specific
analyses in the main() function to run them individually.

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

# For Sobol sampling
from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol as sobol_analyze
# For Morris screening
from SALib.sample import morris as morris_sample
from SALib.analyze import morris as morris_analyze


def run_model():
    """
    Re-initialize and run the DynamicCoop model for n steps,
    return final total fish count.
    """
    dc.initialize('reproduction_rate')
    
    # If reproduction_rate parameter exists, apply it to all fish for homogeneous analysis
    if hasattr(parameters, 'reproduction_rate'):
        for agent in dc.agents:
            if agent.type == 'fish':
                setattr(agent, 'reproduction_rate', parameters.reproduction_rate)
    
    for _ in range(parameters.n):
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
        'plot_update_freq'
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


def ofat_analysis(param_names, original_values, n_points=25, n_reps=50):
    """One-Factor-At-A-Time analysis with statistically significant sample sizes"""
    ofat_results = {}
    for p in param_names:
        x = np.linspace(original_values[p]*0.5, original_values[p]*1.5, n_points)
        means = np.zeros(n_points)
        stds  = np.zeros(n_points)
        for i, xi in enumerate(tqdm(x, desc=f"OFAT {p}")):
            setattr(parameters, p, xi)
            Y = [run_model() for _ in range(n_reps)]
            means[i] = np.mean(Y)
            stds[i]  = np.std(Y, ddof=1)
        ofat_results[p] = (x, means, stds)
        setattr(parameters, p, original_values[p])  # Restore original value
    return ofat_results


def plot_ofat(ofat_results, filename='ofat_full.png'):
    n = len(ofat_results)
    cols = 3
    rows = int(np.ceil(n/cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols,4*rows))
    axes = axes.flatten()
    for ax, (p,(x,m,s)) in zip(axes, ofat_results.items()):
        ax.plot(x, m, '-k', lw=2)
        ax.fill_between(x, m-s, m+s, color='lightgray', alpha=0.5)
        ax.set_title(p)
        ax.set_xlabel(p)
        ax.set_ylabel('Final fish')
        ax.grid(True)
    for ax in axes[n:]: ax.set_visible(False)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    print(f"OFAT results saved to {filename}")
    plt.show()


def sobol_analysis(param_names, bounds, original_values, sample_size=2048):
    """Sobol sensitivity analysis with statistically significant sample size"""
    problem = {'num_vars': len(param_names), 'names': param_names, 'bounds': bounds}
    print(f"Sampling for Sobol: N={sample_size} (Total model runs: {sample_size * (len(param_names) + 2)})")
    X = sobol_sample.sample(problem, sample_size, calc_second_order=False)
    Y = np.zeros(X.shape[0])
    for i, xi in enumerate(tqdm(X, desc='Sobol runs')):
        for name, val in zip(param_names, xi): setattr(parameters, name, val)
        Y[i] = run_model()
    
    # Restore original values
    restore_parameters(param_names, original_values)
    
    Si = sobol_analyze.analyze(problem, Y, calc_second_order=False, print_to_console=False)
    return Si


def plot_sobol(Si, param_names, filename='sobol_indices.png'):
    S1 = np.array(Si['S1'])
    S1_conf = np.array(Si['S1_conf'])
    idx = np.argsort(-S1)
    sorted_names = [param_names[i] for i in idx]
    sorted_S1 = S1[idx]
    sorted_conf = S1_conf[idx]
    fig, ax = plt.subplots(figsize=(12, max(8, len(param_names)*0.35)))
    y_pos = np.arange(len(sorted_names))
    ax.barh(y_pos, sorted_S1, xerr=sorted_conf, align='center', color='skyblue', ecolor='gray', capsize=4)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_names)
    ax.invert_yaxis()
    ax.set_xlabel('First-order Sobol index')
    ax.set_title('Factor Prioritization via Sobol Indices (N=2048)')
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    print(f"Sobol prioritization plot saved to {filename}")
    plt.show()


def morris_analysis(param_names, bounds, original_values, num_trajectories=200, grid_levels=10):
    """Morris screening with statistically significant sample size"""
    problem = {'num_vars': len(param_names), 'names': param_names, 'bounds': bounds}
    print(f"Sampling for Morris: trajectories={num_trajectories}, levels={grid_levels}")
    print(f"Total model runs: {num_trajectories * (len(param_names) + 1)}")
    X = morris_sample.sample(problem, N=num_trajectories, num_levels=grid_levels, optimal_trajectories=None)
    Y = np.zeros(X.shape[0])
    for i, xi in enumerate(tqdm(X, desc='Morris runs')):
        for name, val in zip(param_names, xi): setattr(parameters, name, val)
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
    ax.set_title('Morris Elementary Effects Screening (200 trajectories)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    print(f"Morris EE plot saved to {filename}")
    plt.show()


def main(quick_test=False):
    param_names, bounds, original_values = get_param_names_and_bounds()
    print(f"Parameters included ({len(param_names)}): {param_names}")
    print(f"reproduction_rate included: {'reproduction_rate' in param_names}")
    
    if quick_test:
        print("\n=== QUICK TEST MODE ===")
        print("Running with reduced sample sizes for testing...")
        
        # Quick Morris Analysis (least computationally expensive)
        print("\n=== Running Quick Morris Analysis ===")
        Si_m = morris_analysis(param_names, bounds, original_values, num_trajectories=10, grid_levels=4)
        plot_morris(Si_m, param_names, filename='morris_ee_quick.png')
        
        # Quick Sobol Analysis 
        print("\n=== Running Quick Sobol Analysis ===")
        Si_sob = sobol_analysis(param_names, bounds, original_values, sample_size=32)
        plot_sobol(Si_sob, param_names, filename='sobol_indices_quick.png')
        
        print("\nQuick test completed. For full statistically significant results, run main(quick_test=False)")
        
    else:
        print("\n=== FULL STATISTICAL ANALYSIS ===")
        print("WARNING: This will take many hours (6-12+ hours total)!")
        print("Consider running analyses individually by commenting out others in the code.")
        
        total_runs_estimated = (
            len(param_names) * 25 * 50 +  # OFAT
            2048 * (len(param_names) + 2) +  # Sobol
            200 * (len(param_names) + 1)  # Morris
        )
        print(f"Total estimated model runs: {total_runs_estimated:,}")
        
        # Morris Analysis (least computationally expensive - start here)
        print(f"\n=== Running Morris Analysis ===")
        print(f"Estimated runs: {200 * (len(param_names) + 1):,} (~30-60 minutes)")
        Si_m = morris_analysis(param_names, bounds, original_values, num_trajectories=200, grid_levels=10)
        plot_morris(Si_m, param_names)

        # Sobol Analysis (moderate computational cost)
        print(f"\n=== Running Sobol Analysis ===")
        print(f"Estimated runs: {2048 * (len(param_names) + 2):,} (~2-4 hours)")
        Si_sob = sobol_analysis(param_names, bounds, original_values, sample_size=2048)
        plot_sobol(Si_sob, param_names)
        
        # OFAT Analysis (most computationally expensive - run last)
        print(f"\n=== Running OFAT Analysis ===")
        print(f"Estimated runs: {len(param_names) * 25 * 50:,} (~4-8 hours)")
        ofat_res = ofat_analysis(param_names, original_values, n_points=25, n_reps=50)
        plot_ofat(ofat_res)
    
    # Ensure all parameters are restored to original values
    restore_parameters(param_names, original_values)
    print("\nAll parameters restored to original values.")

if __name__=='__main__':
    import sys
    # Run quick test if 'quick' is passed as argument, otherwise run full analysis
    quick_mode = len(sys.argv) > 1 and sys.argv[1] == 'quick'
    main(quick_test=quick_mode)
