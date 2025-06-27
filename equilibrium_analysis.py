"""
Equilibrium Analysis for DynamicCoop Model

This script systematically explores parameter ranges to identify regions where 
the system reaches stable equilibria rather than crashes or exhibits unstable dynamics.

Focus on 5 key parameters:
1. scale - fish schooling interaction distance multiplier
2. imitation_period - frequency of strategy comparison between fishers  
3. cooperation_increase - magnitude of effort reduction when density drops
4. q - catchability coefficient determining fishing efficiency
5. trust_decrease - rate of trust degradation when observing non-cooperation

Equilibrium criteria:
- Fish population stabilizes (not trending toward 0 or infinity)
- Low coefficient of variation in final 50 timesteps
- System doesn't crash (fish population > 5)
- Reasonable population levels (5 < fish < 1000)
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import DynamicCoop as dc
import parameters
import pandas as pd
import seaborn as sns
from scipy import stats
import os
from datetime import datetime

# Disable visualization in DynamicCoop
dc.VISUALIZATION_ENABLED = False
dc.CREATE_GIF = False

def run_equilibrium_test(param_dict, n_timesteps=800, equilibrium_window=100):
    """
    Run model with given parameters and test for equilibrium
    
    Returns:
    - equilibrium_reached: bool
    - final_mean: float (mean fish population in final window)
    - final_cv: float (coefficient of variation in final window)
    - trend_slope: float (slope of trend in final window)
    - min_fish: float (minimum fish population during run)
    - crash: bool (did population crash to near zero)
    """
    try:
        # Store original parameter values
        original_values = {}
        for param_name in param_dict.keys():
            if hasattr(parameters, param_name):
                original_values[param_name] = getattr(parameters, param_name)
        
        # Set new parameter values
        for param_name, value in param_dict.items():
            if hasattr(parameters, param_name):
                setattr(parameters, param_name, value)
        
        # Handle scale-dependent parameters
        if 'scale' in param_dict:
            scale_val = param_dict['scale']
            parameters.rad_repulsion = 0.025 * scale_val
            parameters.rad_orientation = 0.06 * scale_val
            parameters.rad_attraction = 0.1 * scale_val
            parameters.rad_repulsion_sqr = parameters.rad_repulsion ** 2
            parameters.rad_orientation_sqr = parameters.rad_orientation ** 2
            parameters.rad_attraction_sqr = parameters.rad_attraction ** 2
        
        # Initialize and run model
        dc.initialize('default')
        
        fish_counts = []
        for t in range(n_timesteps):
            dc.update_one_unit_time()
            fish_counts.append(dc.total_fish_count[-1])
        
        # Restore original parameter values
        for param_name, original_value in original_values.items():
            setattr(parameters, param_name, original_value)
        
        # Restore scale-dependent parameters if needed
        if 'scale' in original_values:
            scale_val = original_values['scale']
            parameters.rad_repulsion = 0.025 * scale_val
            parameters.rad_orientation = 0.06 * scale_val
            parameters.rad_attraction = 0.1 * scale_val
            parameters.rad_repulsion_sqr = parameters.rad_repulsion ** 2
            parameters.rad_orientation_sqr = parameters.rad_orientation ** 2
            parameters.rad_attraction_sqr = parameters.rad_attraction ** 2
        
        # Analyze equilibrium
        fish_counts = np.array(fish_counts)
        final_window = fish_counts[-equilibrium_window:]
        
        # Basic metrics
        final_mean = np.mean(final_window)
        final_std = np.std(final_window)
        final_cv = final_std / final_mean if final_mean > 0 else np.inf
        min_fish = np.min(fish_counts)
        
        # Trend analysis
        x = np.arange(len(final_window))
        slope, _, _, _, _ = stats.linregress(x, final_window)
        
        # Crash detection
        crash = min_fish < 5 or final_mean < 5
        
        # Equilibrium criteria
        equilibrium_reached = (
            not crash and
            final_cv < 0.3 and  # CV less than 30%
            abs(slope) < 0.1 and  # Slope less than 0.1 fish/timestep
            5 < final_mean < 1000  # Reasonable population range
        )
        
        return {
            'equilibrium_reached': equilibrium_reached,
            'final_mean': final_mean,
            'final_cv': final_cv,
            'trend_slope': slope,
            'min_fish': min_fish,
            'crash': crash,
            'fish_trajectory': fish_counts
        }
        
    except Exception as e:
        print(f"Error in equilibrium test: {str(e)}")
        return {
            'equilibrium_reached': False,
            'final_mean': 0,
            'final_cv': np.inf,
            'trend_slope': 0,
            'min_fish': 0,
            'crash': True,
            'fish_trajectory': np.zeros(n_timesteps)
        }

def explore_parameter_space(param_ranges, n_samples_per_param=20, n_reps=3):
    """
    Systematically explore parameter space to find equilibrium regions
    """
    print(f"Exploring parameter space with {n_samples_per_param} samples per parameter")
    print(f"Running {n_reps} repetitions per parameter combination")
    
    # Generate parameter grids
    param_grids = {}
    for param_name, (min_val, max_val) in param_ranges.items():
        if param_name == 'imitation_period':
            # Integer parameter
            param_grids[param_name] = np.linspace(min_val, max_val, n_samples_per_param, dtype=int)
        else:
            param_grids[param_name] = np.linspace(min_val, max_val, n_samples_per_param)
    
    # Store results
    results = []
    
    # Test each parameter individually (one-at-a-time)
    for param_name in param_ranges.keys():
        print(f"\nTesting parameter: {param_name}")
        
        for value in tqdm(param_grids[param_name], desc=f"Testing {param_name}"):
            # Create parameter dict with default values except for the one being tested
            param_dict = {
                'scale': 2.0,
                'imitation_period': 5,
                'cooperation_increase': 0.2,
                'q': 0.6,
                'trust_decrease': 0.2
            }
            param_dict[param_name] = value
            
            # Run multiple repetitions
            rep_results = []
            for rep in range(n_reps):
                result = run_equilibrium_test(param_dict)
                rep_results.append(result)
            
            # Aggregate results across repetitions
            equilibrium_count = sum(1 for r in rep_results if r['equilibrium_reached'])
            equilibrium_probability = equilibrium_count / n_reps
            
            mean_final_pop = np.mean([r['final_mean'] for r in rep_results])
            mean_cv = np.mean([r['final_cv'] for r in rep_results if not np.isinf(r['final_cv'])])
            crash_count = sum(1 for r in rep_results if r['crash'])
            crash_probability = crash_count / n_reps
            
            results.append({
                'parameter': param_name,
                'value': value,
                'equilibrium_probability': equilibrium_probability,
                'mean_final_population': mean_final_pop,
                'mean_cv': mean_cv,
                'crash_probability': crash_probability,
                'n_reps': n_reps
            })
    
    return pd.DataFrame(results)

def analyze_pairwise_interactions(param_ranges, key_params, n_samples=10, n_reps=2):
    """
    Analyze pairwise parameter interactions for equilibrium
    """
    print(f"\nAnalyzing pairwise interactions with {n_samples}x{n_samples} grid per pair")
    
    results = []
    param_pairs = [(key_params[i], key_params[j]) for i in range(len(key_params)) 
                   for j in range(i+1, len(key_params))]
    
    for param1, param2 in param_pairs:
        print(f"Testing interaction: {param1} x {param2}")
        
        # Generate parameter grids
        if param1 == 'imitation_period':
            vals1 = np.linspace(param_ranges[param1][0], param_ranges[param1][1], n_samples, dtype=int)
        else:
            vals1 = np.linspace(param_ranges[param1][0], param_ranges[param1][1], n_samples)
            
        if param2 == 'imitation_period':
            vals2 = np.linspace(param_ranges[param2][0], param_ranges[param2][1], n_samples, dtype=int)
        else:
            vals2 = np.linspace(param_ranges[param2][0], param_ranges[param2][1], n_samples)
        
        for val1 in tqdm(vals1, desc=f"{param1} x {param2}"):
            for val2 in vals2:
                # Set parameters
                param_dict = {
                    'scale': 2.0,
                    'imitation_period': 5,
                    'cooperation_increase': 0.2,
                    'q': 0.6,
                    'trust_decrease': 0.2
                }
                param_dict[param1] = val1
                param_dict[param2] = val2
                
                # Run repetitions
                rep_results = []
                for rep in range(n_reps):
                    result = run_equilibrium_test(param_dict)
                    rep_results.append(result)
                
                # Aggregate
                equilibrium_count = sum(1 for r in rep_results if r['equilibrium_reached'])
                equilibrium_probability = equilibrium_count / n_reps
                mean_final_pop = np.mean([r['final_mean'] for r in rep_results])
                
                results.append({
                    'param1': param1,
                    'param2': param2,
                    'val1': val1,
                    'val2': val2,
                    'equilibrium_probability': equilibrium_probability,
                    'mean_final_population': mean_final_pop
                })
    
    return pd.DataFrame(results)

def plot_equilibrium_results(df_univariate, df_pairwise=None):
    """
    Create comprehensive plots of equilibrium analysis results
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create output directory
    os.makedirs('equilibrium_results', exist_ok=True)
    
    # 1. Univariate equilibrium probability plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    params = df_univariate['parameter'].unique()
    for i, param in enumerate(params):
        if i < len(axes):
            param_data = df_univariate[df_univariate['parameter'] == param]
            
            ax = axes[i]
            ax.plot(param_data['value'], param_data['equilibrium_probability'], 'o-', linewidth=2, markersize=6)
            ax.set_xlabel(param)
            ax.set_ylabel('Equilibrium Probability')
            ax.set_title(f'Equilibrium Regions: {param}')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-0.05, 1.05)
            
            # Highlight equilibrium regions (prob > 0.5)
            eq_regions = param_data[param_data['equilibrium_probability'] > 0.5]
            if len(eq_regions) > 0:
                ax.fill_between(eq_regions['value'], 0, eq_regions['equilibrium_probability'], 
                               alpha=0.3, color='green', label='Equilibrium Region')
                ax.legend()
    
    # Hide unused subplots
    for i in range(len(params), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f'equilibrium_results/univariate_equilibrium_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Mean final population plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, param in enumerate(params):
        if i < len(axes):
            param_data = df_univariate[df_univariate['parameter'] == param]
            
            ax = axes[i]
            # Color points by equilibrium probability
            scatter = ax.scatter(param_data['value'], param_data['mean_final_population'], 
                               c=param_data['equilibrium_probability'], cmap='RdYlGn', 
                               s=60, alpha=0.7)
            ax.set_xlabel(param)
            ax.set_ylabel('Mean Final Fish Population')
            ax.set_title(f'Population vs {param}')
            ax.grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=ax, label='Equilibrium Probability')
    
    # Hide unused subplots
    for i in range(len(params), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f'equilibrium_results/population_vs_params_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Pairwise interaction heatmaps (if available)
    if df_pairwise is not None and len(df_pairwise) > 0:
        pairs = df_pairwise[['param1', 'param2']].drop_duplicates()
        n_pairs = len(pairs)
        
        if n_pairs > 0:
            cols = min(3, n_pairs)
            rows = int(np.ceil(n_pairs / cols))
            
            fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
            if n_pairs == 1:
                axes = [axes]
            elif rows == 1:
                axes = axes if hasattr(axes, '__len__') else [axes]
            else:
                axes = axes.flatten()
            
            for i, (_, pair) in enumerate(pairs.iterrows()):
                if i < len(axes):
                    param1, param2 = pair['param1'], pair['param2']
                    pair_data = df_pairwise[(df_pairwise['param1'] == param1) & 
                                          (df_pairwise['param2'] == param2)]
                    
                    # Create pivot table for heatmap
                    pivot = pair_data.pivot(index='val2', columns='val1', values='equilibrium_probability')
                    
                    ax = axes[i]
                    sns.heatmap(pivot, ax=ax, cmap='RdYlGn', vmin=0, vmax=1, 
                               cbar_kws={'label': 'Equilibrium Probability'})
                    ax.set_title(f'{param1} x {param2}')
                    ax.set_xlabel(param1)
                    ax.set_ylabel(param2)
            
            # Hide unused subplots
            for i in range(n_pairs, len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(f'equilibrium_results/pairwise_interactions_{timestamp}.png', dpi=300, bbox_inches='tight')
            plt.show()

def save_equilibrium_ranges(df_univariate, threshold=0.5):
    """
    Extract and save viable parameter ranges where equilibrium probability > threshold
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    equilibrium_ranges = {}
    
    for param in df_univariate['parameter'].unique():
        param_data = df_univariate[df_univariate['parameter'] == param].sort_values('value')
        viable = param_data[param_data['equilibrium_probability'] > threshold]
        
        if len(viable) > 0:
            min_val = viable['value'].min()
            max_val = viable['value'].max()
            mean_prob = viable['equilibrium_probability'].mean()
            
            equilibrium_ranges[param] = {
                'min_viable': min_val,
                'max_viable': max_val,
                'range_width': max_val - min_val,
                'mean_equilibrium_prob': mean_prob,
                'n_viable_points': len(viable)
            }
        else:
            equilibrium_ranges[param] = {
                'min_viable': None,
                'max_viable': None,
                'range_width': 0,
                'mean_equilibrium_prob': 0,
                'n_viable_points': 0
            }
    
    # Save to file
    with open(f'equilibrium_results/viable_ranges_{timestamp}.txt', 'w') as f:
        f.write(f"VIABLE PARAMETER RANGES (Equilibrium Probability > {threshold})\n")
        f.write("="*60 + "\n\n")
        
        for param, ranges in equilibrium_ranges.items():
            f.write(f"{param}:\n")
            if ranges['min_viable'] is not None:
                f.write(f"  Viable range: {ranges['min_viable']:.3f} - {ranges['max_viable']:.3f}\n")
                f.write(f"  Range width: {ranges['range_width']:.3f}\n")
                f.write(f"  Mean equilibrium probability: {ranges['mean_equilibrium_prob']:.3f}\n")
                f.write(f"  Number of viable points: {ranges['n_viable_points']}\n")
            else:
                f.write(f"  No viable range found (no points with equilibrium prob > {threshold})\n")
            f.write("\n")
    
    print(f"Viable parameter ranges saved to equilibrium_results/viable_ranges_{timestamp}.txt")
    return equilibrium_ranges

def main():
    """
    Main equilibrium analysis workflow
    """
    print("=== EQUILIBRIUM ANALYSIS FOR DYNAMICCOOP MODEL ===")
    print("Analyzing parameter ranges for stable equilibria\n")
    
    # Define parameter ranges to explore
    param_ranges = {
        'scale': (0.5, 4.0),
        'imitation_period': (1, 15),
        'cooperation_increase': (0.05, 0.5),
        'q': (0.2, 1.0),
        'trust_decrease': (0.05, 0.5)
    }
    
    print("Parameter ranges to explore:")
    for param, (min_val, max_val) in param_ranges.items():
        print(f"  {param}: {min_val} - {max_val}")
    
    # 1. Univariate analysis
    print("\n=== UNIVARIATE ANALYSIS ===")
    df_univariate = explore_parameter_space(param_ranges, n_samples_per_param=15, n_reps=5)
    
    # 2. Pairwise analysis (optional - computationally expensive)
    print("\n=== PAIRWISE ANALYSIS ===")
    key_params = list(param_ranges.keys())
    df_pairwise = analyze_pairwise_interactions(param_ranges, key_params, n_samples=8, n_reps=3)
    
    # 3. Generate plots
    print("\n=== GENERATING PLOTS ===")
    plot_equilibrium_results(df_univariate, df_pairwise)
    
    # 4. Extract viable ranges
    print("\n=== EXTRACTING VIABLE RANGES ===")
    viable_ranges = save_equilibrium_ranges(df_univariate, threshold=0.6)
    
    # 5. Save raw data
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    df_univariate.to_csv(f'equilibrium_results/univariate_results_{timestamp}.csv', index=False)
    df_pairwise.to_csv(f'equilibrium_results/pairwise_results_{timestamp}.csv', index=False)
    
    print(f"\n=== ANALYSIS COMPLETE ===")
    print(f"Results saved in 'equilibrium_results/' directory")
    print(f"Key findings:")
    
    for param, ranges in viable_ranges.items():
        if ranges['min_viable'] is not None:
            print(f"  {param}: viable range {ranges['min_viable']:.3f} - {ranges['max_viable']:.3f}")
        else:
            print(f"  {param}: NO VIABLE RANGE FOUND")

if __name__ == '__main__':
    main() 