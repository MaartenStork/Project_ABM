"""
Time-Series Sensitivity Analysis for DynamicCoop Model

This improved sensitivity analysis captures the full temporal dynamics
rather than just final values, providing much richer insights into
parameter effects on system behavior over time.

Metrics captured:
1. Final fish count (original metric)
2. Mean fish count over simulation
3. Fish population stability (coefficient of variation)
4. Time to equilibrium
5. Minimum fish count (crash detection)
6. Maximum fish count (explosion detection)
7. Recovery rate after crashes
8. Trend direction (growing/declining/stable)
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import DynamicCoop as dc
import parameters
from scipy import stats
import pandas as pd

# For Sobol sampling
from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol as sobol_analyze

def run_model_timeseries(n_timesteps=1000, burn_in=200):
    """
    Run model and return comprehensive time-series metrics
    """
    dc.initialize('default')
    
    # Store fish counts over time
    fish_counts = []
    
    for t in range(n_timesteps):
        dc.update_one_unit_time()
        fish_counts.append(dc.total_fish_count[-1])
    
    fish_counts = np.array(fish_counts)
    
    # Calculate comprehensive metrics
    metrics = calculate_timeseries_metrics(fish_counts, burn_in)
    
    return metrics

def calculate_timeseries_metrics(fish_counts, burn_in=200):
    """
    Calculate comprehensive time-series metrics from fish population data
    """
    # Full time series metrics
    final_fish = fish_counts[-1]
    mean_fish = np.mean(fish_counts)
    std_fish = np.std(fish_counts)
    cv_fish = std_fish / mean_fish if mean_fish > 0 else np.inf
    min_fish = np.min(fish_counts)
    max_fish = np.max(fish_counts)
    
    # Post burn-in metrics (more stable period)
    if len(fish_counts) > burn_in:
        stable_period = fish_counts[burn_in:]
        mean_stable = np.mean(stable_period)
        cv_stable = np.std(stable_period) / mean_stable if mean_stable > 0 else np.inf
        
        # Trend analysis
        x = np.arange(len(stable_period))
        if len(stable_period) > 10:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, stable_period)
            trend_significant = p_value < 0.05
        else:
            slope, r_value, p_value = 0, 0, 1
            trend_significant = False
    else:
        mean_stable = mean_fish
        cv_stable = cv_fish
        slope, r_value, p_value = 0, 0, 1
        trend_significant = False
    
    # Crash and recovery analysis
    crash_threshold = mean_fish * 0.5  # 50% below mean
    crashes = fish_counts < crash_threshold
    n_crashes = np.sum(crashes)
    
    # Recovery analysis
    if n_crashes > 0:
        crash_indices = np.where(crashes)[0]
        recoveries = 0
        for idx in crash_indices:
            if idx < len(fish_counts) - 20:  # Need 20 steps to assess recovery
                recovery_window = fish_counts[idx+1:idx+21]
                if np.mean(recovery_window) > fish_counts[idx] * 1.2:  # 20% recovery
                    recoveries += 1
        recovery_rate = recoveries / len(crash_indices) if len(crash_indices) > 0 else 0
    else:
        recovery_rate = 1.0  # No crashes to recover from
    
    # Time to equilibrium (when CV stabilizes)
    equilibrium_time = estimate_equilibrium_time(fish_counts)
    
    # Population viability
    viable_timesteps = np.sum(fish_counts >= 10)  # Viable population threshold
    viability_rate = viable_timesteps / len(fish_counts)
    
    return {
        'final_fish': final_fish,
        'mean_fish': mean_fish,
        'cv_fish': cv_fish,
        'min_fish': min_fish,
        'max_fish': max_fish,
        'mean_stable': mean_stable,
        'cv_stable': cv_stable,
        'trend_slope': slope,
        'trend_r_squared': r_value**2,
        'trend_p_value': p_value,
        'trend_significant': trend_significant,
        'n_crashes': n_crashes,
        'recovery_rate': recovery_rate,
        'equilibrium_time': equilibrium_time,
        'viability_rate': viability_rate,
        'fish_trajectory': fish_counts
    }

def estimate_equilibrium_time(fish_counts, window_size=50):
    """
    Estimate when the system reaches equilibrium by finding when
    the coefficient of variation stabilizes
    """
    if len(fish_counts) < window_size * 2:
        return len(fish_counts)
    
    cv_values = []
    for i in range(window_size, len(fish_counts) - window_size):
        window = fish_counts[i-window_size:i+window_size]
        cv = np.std(window) / np.mean(window) if np.mean(window) > 0 else np.inf
        cv_values.append(cv)
    
    # Find when CV becomes stable (small changes)
    cv_values = np.array(cv_values)
    if len(cv_values) < 10:
        return len(fish_counts)
    
    # Look for first point where CV stops changing rapidly
    cv_changes = np.abs(np.diff(cv_values))
    stable_threshold = np.percentile(cv_changes, 25)  # Bottom quartile of changes
    
    stable_points = cv_changes < stable_threshold
    if np.any(stable_points):
        equilibrium_idx = np.where(stable_points)[0][0]
        return equilibrium_idx + window_size
    else:
        return len(fish_counts)

def get_param_names_and_bounds():
    """Get parameter names and bounds, focusing on key parameters"""
    # Focus on the 5 key parameters identified by Morris analysis
    key_params = ['scale', 'imitation_period', 'cooperation_increase', 'q', 'trust_decrease']
    
    # Store original values
    original_values = {p: getattr(parameters, p) for p in key_params}
    
    # Set bounds ±50%
    bounds = [[getattr(parameters, p)*0.5, getattr(parameters, p)*1.5] for p in key_params]
    
    return key_params, bounds, original_values

def update_parameters(param_names, param_values):
    """Update parameters and handle derived parameters"""
    for name, val in zip(param_names, param_values):
        setattr(parameters, name, val)
    
    # Handle scale-dependent parameters
    if 'scale' in param_names:
        scale_idx = param_names.index('scale')
        scale_val = param_values[scale_idx]
        parameters.rad_repulsion = 0.025 * scale_val
        parameters.rad_orientation = 0.06 * scale_val
        parameters.rad_attraction = 0.1 * scale_val
        parameters.rad_repulsion_sqr = parameters.rad_repulsion ** 2
        parameters.rad_orientation_sqr = parameters.rad_orientation ** 2
        parameters.rad_attraction_sqr = parameters.rad_attraction ** 2

def restore_parameters(original_values):
    """Restore parameters to original values"""
    for name, value in original_values.items():
        setattr(parameters, name, value)
    
    # Restore scale-dependent parameters
    if 'scale' in original_values:
        scale_val = original_values['scale']
        parameters.rad_repulsion = 0.025 * scale_val
        parameters.rad_orientation = 0.06 * scale_val
        parameters.rad_attraction = 0.1 * scale_val
        parameters.rad_repulsion_sqr = parameters.rad_repulsion ** 2
        parameters.rad_orientation_sqr = parameters.rad_orientation ** 2
        parameters.rad_attraction_sqr = parameters.rad_attraction ** 2

def timeseries_ofat_analysis(param_names, original_values, n_points=15, n_reps=10):
    """
    OFAT analysis with comprehensive time-series metrics
    """
    print("=== TIME-SERIES OFAT ANALYSIS ===")
    print(f"Parameters: {param_names}")
    print(f"Points per parameter: {n_points}")
    print(f"Repetitions per point: {n_reps}")
    print(f"Total simulations: {len(param_names) * n_points * n_reps}")
    
    results = []
    
    for param in param_names:
        print(f"\nAnalyzing parameter: {param}")
        
        # Create parameter range
        param_range = np.linspace(original_values[param]*0.5, original_values[param]*1.5, n_points)
        
        for value in tqdm(param_range, desc=f"OFAT {param}"):
            # Set parameter value
            setattr(parameters, param, value)
            
            # Handle scale-dependent parameters
            if param == 'scale':
                parameters.rad_repulsion = 0.025 * value
                parameters.rad_orientation = 0.06 * value
                parameters.rad_attraction = 0.1 * value
                parameters.rad_repulsion_sqr = parameters.rad_repulsion ** 2
                parameters.rad_orientation_sqr = parameters.rad_orientation ** 2
                parameters.rad_attraction_sqr = parameters.rad_attraction ** 2
            
            # Run multiple repetitions
            rep_results = []
            for rep in range(n_reps):
                metrics = run_model_timeseries()
                rep_results.append(metrics)
            
            # Aggregate results
            aggregated = aggregate_repetitions(rep_results)
            aggregated['parameter'] = param
            aggregated['value'] = value
            results.append(aggregated)
        
        # Restore original parameter value
        setattr(parameters, param, original_values[param])
        if param == 'scale':
            scale_val = original_values['scale']
            parameters.rad_repulsion = 0.025 * scale_val
            parameters.rad_orientation = 0.06 * scale_val
            parameters.rad_attraction = 0.1 * scale_val
            parameters.rad_repulsion_sqr = parameters.rad_repulsion ** 2
            parameters.rad_orientation_sqr = parameters.rad_orientation ** 2
            parameters.rad_attraction_sqr = parameters.rad_attraction ** 2
    
    return pd.DataFrame(results)

def aggregate_repetitions(rep_results):
    """Aggregate results from multiple repetitions"""
    metrics_to_aggregate = [
        'final_fish', 'mean_fish', 'cv_fish', 'min_fish', 'max_fish',
        'mean_stable', 'cv_stable', 'trend_slope', 'trend_r_squared',
        'n_crashes', 'recovery_rate', 'equilibrium_time', 'viability_rate'
    ]
    
    aggregated = {}
    
    for metric in metrics_to_aggregate:
        values = [r[metric] for r in rep_results if not np.isinf(r[metric])]
        if values:
            aggregated[f'{metric}_mean'] = np.mean(values)
            aggregated[f'{metric}_std'] = np.std(values)
            aggregated[f'{metric}_min'] = np.min(values)
            aggregated[f'{metric}_max'] = np.max(values)
        else:
            aggregated[f'{metric}_mean'] = np.nan
            aggregated[f'{metric}_std'] = np.nan
            aggregated[f'{metric}_min'] = np.nan
            aggregated[f'{metric}_max'] = np.nan
    
    # Special handling for boolean metrics
    trend_significant_rate = np.mean([r['trend_significant'] for r in rep_results])
    aggregated['trend_significant_rate'] = trend_significant_rate
    
    return aggregated

def timeseries_sobol_analysis(param_names, bounds, original_values, sample_size=512):
    """
    Sobol analysis with multiple output metrics
    """
    print("=== TIME-SERIES SOBOL ANALYSIS ===")
    print(f"Sample size: {sample_size}")
    print(f"Total simulations: {sample_size * (len(param_names) + 2)}")
    
    # Define problem for SALib
    problem = {
        'num_vars': len(param_names),
        'names': param_names,
        'bounds': bounds
    }
    
    # Generate samples
    param_values = sobol_sample.sample(problem, sample_size)
    
    # Store results for multiple outputs
    outputs = {
        'final_fish': [],
        'mean_fish': [],
        'cv_stable': [],
        'viability_rate': [],
        'recovery_rate': []
    }
    
    print("Running simulations...")
    for params in tqdm(param_values):
        # Update parameters
        update_parameters(param_names, params)
        
        # Run model
        metrics = run_model_timeseries()
        
        # Store outputs
        outputs['final_fish'].append(metrics['final_fish'])
        outputs['mean_fish'].append(metrics['mean_fish'])
        outputs['cv_stable'].append(metrics['cv_stable'] if not np.isinf(metrics['cv_stable']) else 1.0)
        outputs['viability_rate'].append(metrics['viability_rate'])
        outputs['recovery_rate'].append(metrics['recovery_rate'])
    
    # Restore parameters
    restore_parameters(original_values)
    
    # Analyze sensitivity for each output
    sobol_results = {}
    for output_name, output_values in outputs.items():
        try:
            Si = sobol_analyze.analyze(problem, np.array(output_values))
            sobol_results[output_name] = Si
        except Exception as e:
            print(f"Error analyzing {output_name}: {e}")
            sobol_results[output_name] = None
    
    return sobol_results

def plot_timeseries_ofat(df, output_dir='timeseries_sensitivity'):
    """Create comprehensive OFAT plots for multiple metrics"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Metrics to plot
    metrics = ['final_fish', 'mean_fish', 'cv_stable', 'viability_rate', 'recovery_rate']
    metric_labels = ['Final Fish Count', 'Mean Fish Count', 'Stability (CV)', 'Viability Rate', 'Recovery Rate']
    
    params = df['parameter'].unique()
    
    for metric, label in zip(metrics, metric_labels):
        fig, axes = plt.subplots(1, len(params), figsize=(4*len(params), 4))
        if len(params) == 1:
            axes = [axes]
        
        for i, param in enumerate(params):
            param_data = df[df['parameter'] == param].sort_values('value')
            
            mean_col = f'{metric}_mean'
            std_col = f'{metric}_std'
            
            if mean_col in param_data.columns:
                x = param_data['value']
                y = param_data[mean_col]
                yerr = param_data[std_col] if std_col in param_data.columns else None
                
                axes[i].errorbar(x, y, yerr=yerr, fmt='o-', capsize=5, capthick=2)
                axes[i].set_xlabel(param)
                axes[i].set_ylabel(label)
                axes[i].set_title(f'{param} vs {label}')
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{metric}_ofat.png', dpi=300, bbox_inches='tight')
        plt.show()

def plot_timeseries_sobol(sobol_results, output_dir='timeseries_sensitivity'):
    """Plot Sobol indices for multiple outputs"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Create subplot for each output metric
    valid_results = {k: v for k, v in sobol_results.items() if v is not None}
    
    if not valid_results:
        print("No valid Sobol results to plot")
        return
    
    fig, axes = plt.subplots(len(valid_results), 1, figsize=(10, 4*len(valid_results)))
    if len(valid_results) == 1:
        axes = [axes]
    
    for i, (output_name, Si) in enumerate(valid_results.items()):
        param_names = Si['names'] if 'names' in Si else list(range(len(Si['S1'])))
        
        # Plot first-order indices
        axes[i].barh(param_names, Si['S1'], alpha=0.7, label='First-order (S1)')
        
        # Plot total-order indices if available
        if 'ST' in Si:
            axes[i].barh(param_names, Si['ST'], alpha=0.5, label='Total-order (ST)')
        
        axes[i].set_xlabel('Sensitivity Index')
        axes[i].set_title(f'Sobol Indices for {output_name.replace("_", " ").title()}')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/sobol_multiple_outputs.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main analysis function"""
    print("="*60)
    print("TIME-SERIES SENSITIVITY ANALYSIS")
    print("="*60)
    
    # Get parameters
    param_names, bounds, original_values = get_param_names_and_bounds()
    
    print(f"Analyzing parameters: {param_names}")
    print(f"Original values: {original_values}")
    
    try:
        # Run OFAT analysis
        print("\nRunning OFAT analysis...")
        ofat_df = timeseries_ofat_analysis(param_names, original_values, n_points=15, n_reps=10)
        
        # Plot OFAT results
        plot_timeseries_ofat(ofat_df)
        
        # Save OFAT results
        ofat_df.to_csv('timeseries_sensitivity/ofat_results.csv', index=False)
        
        # Run Sobol analysis
        print("\nRunning Sobol analysis...")
        sobol_results = timeseries_sobol_analysis(param_names, bounds, original_values, sample_size=256)
        
        # Plot Sobol results
        plot_timeseries_sobol(sobol_results)
        
        print("\n=== ANALYSIS COMPLETE ===")
        print("Results saved to timeseries_sensitivity/ directory")
        
        return ofat_df, sobol_results
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return None, None
    
    finally:
        # Always restore parameters
        restore_parameters(original_values)

if __name__ == '__main__':
    results = main() 