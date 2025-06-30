#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Parameter Equilibrium Analyzer

This script systematically tests different parameter combinations to find ranges
where the fish population reaches equilibrium in the ABM simulation.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import os
import itertools
import multiprocessing
import json
import traceback
import sys

print("Starting parameter equilibrium analyzer...")

# Import simulation modules
try:
    from parameters.parameters import BaseParameters, ModelParameters
    print("Successfully imported parameters")
    import dynamic_coop as sim
    print("Successfully imported DynamicCoop")
except Exception as e:
    print(f"Error importing modules: {e}")
    traceback.print_exc()
    sys.exit(1)


BASE_PARAMS = BaseParameters()
MODEL_PARAMS = ModelParameters()


# Define equilibrium criteria
def is_equilibrium(fish_counts, window_size=30, threshold=0.05, warm_up_period=50, extinction_threshold=10, growth_check=True):
    """
    Determine if fish count has reached equilibrium.
    
    Parameters:
    -----------
    fish_counts : list
        List of fish counts over time
    window_size : int
        Size of window to check for stability
    threshold : float
        Maximum allowed coefficient of variation (std/mean) for equilibrium
    warm_up_period : int
        Number of initial time steps to ignore before checking for equilibrium
    extinction_threshold : int
        If population falls below this value, it's considered extinction (not equilibrium)
    growth_check : bool
        Whether to check for continuous growth patterns
        
    Returns:
    --------
    bool
        True if equilibrium is reached, False otherwise
    equilibrium_value : float or None
        The average fish count value at equilibrium, or None if not reached
    time_to_equilibrium : int or None
        The time step when equilibrium was reached, or None if not reached
    status : str
        Description of the system state ('equilibrium', 'extinction', 'growth', 'unstable')
    """
    # Need enough data points after warm-up period
    if len(fish_counts) <= warm_up_period + window_size:
        return False, None, None, "insufficient_data"
    
    # Skip warm-up period
    post_warmup = fish_counts[warm_up_period:]
    
    # Check for extinction
    if np.mean(post_warmup[-window_size:]) < extinction_threshold:
        return False, None, None, "extinction"
    
    # Check for continuous growth
    if growth_check and len(post_warmup) >= window_size * 2:
        # Compare first and second half of the post-warmup period
        first_half_mean = np.mean(post_warmup[:len(post_warmup)//2])
        second_half_mean = np.mean(post_warmup[len(post_warmup)//2:])
        
        # Calculate growth rate
        growth_rate = (second_half_mean / first_half_mean) - 1 if first_half_mean > 0 else 0
        
        # If significant continuous growth (>20% over the period), not equilibrium
        if growth_rate > 0.2:
            return False, None, None, "growth"
    
    # Check for stability in sliding windows
    for i in range(len(post_warmup) - window_size, len(post_warmup) - window_size//2):
        window = post_warmup[i:i + window_size]
        
        # Calculate coefficient of variation (std/mean)
        mean_val = np.mean(window)
        if mean_val <= 0:
            continue  # Skip windows with zero or negative means
            
        cv = np.std(window) / mean_val
        
        # Check for trend by comparing first and second half of window
        first_half = window[:len(window)//2]
        second_half = window[len(window)//2:]
        
        first_mean = np.mean(first_half)
        second_mean = np.mean(second_half)
        
        # Calculate relative change
        trend = abs(second_mean - first_mean) / mean_val if mean_val > 0 else 1
        
        # If variation is low and no significant trend
        if cv < threshold and trend < 0.05:  # Less than 5% change between halves
            # Equilibrium reached
            actual_time = i + warm_up_period
            return True, mean_val, actual_time, "equilibrium"
    
    return False, None, None, "unstable"

def run_simulation_with_params(params):
    """
    Run a simulation with the given parameters and analyze the results.
    
    Parameters:
    -----------
    params : dict
        Dictionary of parameters to set for this simulation run
        
    Returns:
    --------
    dict
        Results including if equilibrium was reached, time to equilibrium,
        equilibrium value, and the parameters used
    """
    print(f"Running simulation with params: {params}")
    
    # Save original parameter values to restore later
    original_params = {}
    for param_name in params:
        if param_name == 'reproduction_rate':
            # For reproduction_rate, we need to handle differently since it's part of fish agents
            original_params[param_name] = params[param_name]  # Just store for reference
        else:
            original_params[param_name] = getattr(BASE_PARAMS, param_name)
            setattr(BASE_PARAMS, param_name, params[param_name])
    
    # Reset the simulation
    try:
        print(f"Running simulation for {BASE_PARAMS.n} time steps...")
        results = sim.run_model(MODEL_PARAMS, BASE_PARAMS, experiment_label='both')
        fish_counts = results.total_fish_count # Start with carrying capacity
        print(f"Simulation complete. Final fish count: {fish_counts[-1]}")
    except Exception as e:
        print(f"Error during simulation: {e}")
        traceback.print_exc()

        # Restore original parameters
        for param_name in original_params:
            if param_name == 'noncoop':
                BASE_PARAMS.noncoop = original_params[param_name]
            elif param_name != 'reproduction_rate':  # Skip reproduction_rate
                setattr(BASE_PARAMS, param_name, original_params[param_name])

        return {
            'params': params,
            'equilibrium_reached': False,
            'error': str(e)
        }

    # Check if equilibrium was reached
    eq_reached, eq_value, eq_time, eq_status = is_equilibrium(
        fish_counts, 
        window_size=global_window_size, 
        warm_up_period=global_warm_up_period
    )
    
    # Restore original parameters
    for param_name in original_params:
        if param_name == 'noncoop':
            BASE_PARAMS.noncoop = original_params[param_name]
        else:
            setattr(BASE_PARAMS, param_name, original_params[param_name])
    
    return {
        'params': params,
        'fish_counts': fish_counts,
        'equilibrium_reached': eq_reached,
        'equilibrium_value': eq_value,
        'time_to_equilibrium': eq_time,
        'status': eq_status
    }

def parameter_sweep(param_ranges, num_processes=4, save_every=50):
    """
    Perform a parameter sweep to find ranges where equilibrium is reached.
    Saves progress incrementally to avoid data loss during long runs.
    
    Parameters:
    -----------
    param_ranges : dict
        Dictionary of parameter names and the values to test
    num_processes : int
        Number of parallel processes to use
    save_every : int
        Save progress every N simulations
        
    Returns:
    --------
    list
        List of simulation results
    """
    # Generate all parameter combinations
    param_names = list(param_ranges.keys())
    param_values = list(param_ranges.values())
    param_combinations = list(itertools.product(*param_values))
    
    # Convert tuples to dictionaries
    param_dicts = [dict(zip(param_names, combo)) for combo in param_combinations]
    
    # Check for existing progress file
    progress_file = "simulation_output/parameter_scan/progress.json"
    completed_results = []
    start_idx = 0
    
    if os.path.exists(progress_file):
        print("Found existing progress file, loading...")
        with open(progress_file, 'r') as f:
            completed_results = json.load(f)
        start_idx = len(completed_results)
        print(f"Resuming from simulation {start_idx}/{len(param_dicts)}")
    
    # Run remaining simulations
    print(f"Running {len(param_dicts) - start_idx} remaining parameter combinations...")
    
    if num_processes > 1:
        # Process in batches to enable saving
        batch_size = num_processes * 5  # Process 5 batches at a time per process
        remaining_params = param_dicts[start_idx:]
        
        for batch_start in range(0, len(remaining_params), batch_size):
            batch_end = min(batch_start + batch_size, len(remaining_params))
            batch = remaining_params[batch_start:batch_end]
            
            print(f"Processing batch {batch_start//batch_size + 1}/{(len(remaining_params)-1)//batch_size + 1}")
            
            with multiprocessing.Pool(num_processes) as pool:
                batch_results = list(tqdm(
                    pool.imap(run_simulation_with_params, batch),
                    total=len(batch),
                    desc=f"Batch {batch_start//batch_size + 1}"
                ))
            
            # Add batch results
            completed_results.extend(batch_results)
            
            # Save progress
            os.makedirs(os.path.dirname(progress_file), exist_ok=True)
            with open(progress_file, 'w') as f:
                json.dump([serialize_result(r) for r in completed_results], f, indent=2)
            
            print(f"Progress saved: {len(completed_results)}/{len(param_dicts)} completed")
    else:
        # Single process with regular saving
        for i, params in enumerate(tqdm(param_dicts[start_idx:], initial=start_idx, total=len(param_dicts))):
            result = run_simulation_with_params(params)
            completed_results.append(result)
            
            # Save every N simulations
            if (i + 1) % save_every == 0:
                os.makedirs(os.path.dirname(progress_file), exist_ok=True)
                with open(progress_file, 'w') as f:
                    json.dump([serialize_result(r) for r in completed_results], f, indent=2)
                print(f"Progress saved: {len(completed_results)}/{len(param_dicts)} completed")
    
    # Final save
    os.makedirs(os.path.dirname(progress_file), exist_ok=True)
    with open(progress_file, 'w') as f:
        json.dump([serialize_result(r) for r in completed_results], f, indent=2)
    
    return completed_results

def serialize_result(result):
    """Helper function to serialize results for JSON saving."""
    serialized = result.copy()
    if 'fish_counts' in serialized:
        serialized['fish_counts'] = [float(x) for x in serialized['fish_counts']]
    return serialized

def analyze_results(results):
    """
    Analyze parameter sweep results and visualize equilibrium regions.
    
    Parameters:
    -----------
    results : list
        List of simulation results from parameter_sweep
        
    Returns:
    --------
    dict
        Dictionary containing analysis results
    """
    # Convert results to DataFrame for easier analysis
    results_df = pd.DataFrame([
        {
            **r['params'],
            'equilibrium_reached': r['equilibrium_reached'],
            'equilibrium_value': r['equilibrium_value'] if r['equilibrium_reached'] else None,
            'time_to_equilibrium': r['time_to_equilibrium'] if r['equilibrium_reached'] else None,
            'status': r.get('status', 'unknown')
        }
        for r in results
    ])
    
    # Calculate percentage of parameters that reached equilibrium
    eq_percentage = results_df['equilibrium_reached'].mean() * 100
    print(f"{eq_percentage:.1f}% of parameter combinations reached equilibrium")
    
    # Count different statuses
    status_counts = results_df['status'].value_counts()
    print("\nStatus distribution:")
    for status, count in status_counts.items():
        percentage = (count / len(results_df)) * 100
        print(f"  {status}: {count} ({percentage:.1f}%)")
    
    # Filter only equilibrium results
    eq_results = results_df[results_df['equilibrium_reached']]
    
    # If no combinations reached equilibrium
    if len(eq_results) == 0:
        print("No parameter combinations reached equilibrium")
        return {
            'equilibrium_percentage': 0,
            'status_counts': status_counts.to_dict(),
            'equilibrium_ranges': {},
            'best_parameters': None
        }
    
    # Find ranges where equilibrium was reached for each parameter
    param_names = list(results[0]['params'].keys())
    equilibrium_ranges = {}
    
    for param in param_names:
        equilibrium_values = eq_results[param].unique()
        equilibrium_ranges[param] = {
            'min': min(equilibrium_values),
            'max': max(equilibrium_values),
            'values': sorted(equilibrium_values.tolist())
        }
    
    # Find parameter combination with highest equilibrium fish count
    best_params = eq_results.loc[eq_results['equilibrium_value'].idxmax()]
    
    print("\nParameter ranges leading to equilibrium:")
    for param, ranges in equilibrium_ranges.items():
        print(f"{param}: {ranges['min']} to {ranges['max']}")
    
    print("\nBest parameter combination:")
    for param in param_names:
        print(f"{param}: {best_params[param]}")
    print(f"Equilibrium value: {best_params['equilibrium_value']}")
    print(f"Time to equilibrium: {best_params['time_to_equilibrium']}")
    
    # Analyze parameter effects on different outcomes
    print("\nParameter effects on simulation outcomes:")
    for param in param_names:
        # Group by parameter value and status
        param_status = results_df.groupby([param, 'status']).size().unstack(fill_value=0)
        print(f"\n{param}:")
        print(param_status)
    
    return {
        'equilibrium_percentage': eq_percentage,
        'status_counts': status_counts.to_dict(),
        'equilibrium_ranges': equilibrium_ranges,
        'best_parameters': best_params.to_dict()
    }

def save_results(results, analysis, output_dir="simulation_output"):
    """
    Save simulation results and analysis to files.
    
    Parameters:
    -----------
    results : list
        List of simulation results
    analysis : dict
        Analysis results from analyze_results
    output_dir : str
        Directory to save results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save raw results
    with open(os.path.join(output_dir, "parameter_sweep_results.json"), "w") as f:
        # Convert ndarray to list for JSON serialization
        json_results = []
        for result in results:
            if 'fish_counts' in result:
                result['fish_counts'] = [float(x) for x in result['fish_counts']]
            json_results.append(result)
        
        json.dump(json_results, f, indent=2)
    
    # Save analysis results
    with open(os.path.join(output_dir, "parameter_analysis.json"), "w") as f:
        json.dump(analysis, f, indent=2)
    
    # Create parameter range plots
    if len(analysis['equilibrium_ranges']) > 0:
        # Create plot for each parameter pair
        param_names = list(analysis['equilibrium_ranges'].keys())
        
        # Create a dataframe with only equilibrium results
        equilibrium_data = [r for r in results if r['equilibrium_reached']]
        eq_df = pd.DataFrame([
            {
                **r['params'],
                'equilibrium_value': r['equilibrium_value']
            }
            for r in equilibrium_data
        ])
        
        # Generate pairwise scatter plots for parameters
        if len(param_names) >= 2:
            fig_dir = os.path.join(output_dir, "parameter_plots")
            os.makedirs(fig_dir, exist_ok=True)
            
            for i, param1 in enumerate(param_names):
                for param2 in param_names[i+1:]:
                    plt.figure(figsize=(8, 6))
                    scatter = plt.scatter(
                        eq_df[param1], 
                        eq_df[param2], 
                        c=eq_df['equilibrium_value'],
                        cmap='viridis', 
                        alpha=0.7,
                        s=50
                    )
                    
                    plt.colorbar(scatter, label='Equilibrium Fish Count')
                    plt.xlabel(param1)
                    plt.ylabel(param2)
                    plt.title(f'Parameter Combinations Leading to Equilibrium')
                    plt.grid(alpha=0.3)
                    
                    plt.savefig(os.path.join(fig_dir, f"{param1}_vs_{param2}.png"), dpi=300)
                    plt.close()

def plot_sample_trajectories(results, num_samples=5, output_dir="simulation_output"):
    """
    Plot fish count trajectories for a sample of parameter combinations.
    
    Parameters:
    -----------
    results : list
        List of simulation results
    num_samples : int
        Number of trajectories to plot
    output_dir : str
        Directory to save plots
    """
    # Create output directory if it doesn't exist
    plot_dir = os.path.join(output_dir, "trajectory_plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    # Filter results that have fish_counts
    results_with_counts = [r for r in results if 'fish_counts' in r]
    
    # Plot equilibrium and non-equilibrium trajectories
    eq_results = [r for r in results_with_counts if r['equilibrium_reached']]
    non_eq_results = [r for r in results_with_counts if not r['equilibrium_reached']]
    
    # Sample from each category
    eq_samples = np.random.choice(
        range(len(eq_results)), 
        min(num_samples, len(eq_results)), 
        replace=False
    ).tolist() if len(eq_results) > 0 else []
    
    non_eq_samples = np.random.choice(
        range(len(non_eq_results)), 
        min(num_samples, len(non_eq_results)), 
        replace=False
    ).tolist() if len(non_eq_results) > 0 else []
    
    # Plot equilibrium trajectories
    if eq_samples:
        plt.figure(figsize=(10, 6))
        for idx in eq_samples:
            result = eq_results[idx]
            time_steps = range(len(result['fish_counts']))
            label = ", ".join(f"{k}={v:.3f}" for k, v in result['params'].items())
            plt.plot(time_steps, result['fish_counts'], label=label)
            
            # Mark equilibrium point if available
            if result['time_to_equilibrium']:
                plt.axvline(x=result['time_to_equilibrium'], linestyle='--', alpha=0.5, color='gray')
                
        plt.xlabel('Time Step')
        plt.ylabel('Fish Count')
        plt.title('Fish Population Trajectories (Equilibrium Reached)')
        plt.grid(alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "equilibrium_trajectories.png"), dpi=300)
        plt.close()
    
    # Plot non-equilibrium trajectories
    if non_eq_samples:
        plt.figure(figsize=(10, 6))
        for idx in non_eq_samples:
            result = non_eq_results[idx]
            time_steps = range(len(result['fish_counts']))
            label = ", ".join(f"{k}={v:.3f}" for k, v in result['params'].items())
            plt.plot(time_steps, result['fish_counts'], label=label)
            
        plt.xlabel('Time Step')
        plt.ylabel('Fish Count')
        plt.title('Fish Population Trajectories (No Equilibrium)')
        plt.grid(alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "non_equilibrium_trajectories.png"), dpi=300)
        plt.close()

# Global variables for equilibrium detection - OPTIMIZED for 350 time steps
global_window_size = 40  # Larger window for more stable detection
global_warm_up_period = 100  # Longer warm-up period for 350 time steps

if __name__ == "__main__":
    # Define parameter ranges to test - OPTIMIZED LONG SIMULATION SWEEP
    # With 10x10x10 = 1000 combinations + 350 time steps = 12-14 hours runtime
    param_ranges = {
        'reproduction_rate': np.linspace(0.65, 0.85, 12),  # 12 values: 0.65 to 0.85
        'trust_increase': np.linspace(0.0001, 0.3, 12),  # 12 values: 0.0001 to 0.3
        'imitation_radius': np.linspace(0.01, 1.5, 12),  # 12 values: 0.01 to 1.5
    }
    
    print(f"Parameter ranges to test: {param_ranges}")
    
    # Calculate total combinations and estimate runtime
    total_combinations = 1
    for param_name, values in param_ranges.items():
        total_combinations *= len(values)
        print(f"  {param_name}: {len(values)} values from {min(values):.4f} to {max(values):.4f}")
    
    print(f"\nTotal parameter combinations: {total_combinations}")
    estimated_hours = total_combinations * 0.7 / 60  # ~42 seconds per sim (350 steps)
    print(f"Estimated runtime: {estimated_hours:.1f} hours ({estimated_hours*60:.0f} minutes)")
    
    # Create a directory for storing parameter scan results
    output_dir = "simulation_output/parameter_scan"
    os.makedirs(output_dir, exist_ok=True)
    
    # Run a minimal test first with just one parameter combination
    print("\nRunning test with a single parameter combination...")
    test_params = {
        'reproduction_rate': param_ranges['reproduction_rate'][0],
        'trust_increase': param_ranges['trust_increase'][0],
        'imitation_radius': param_ranges['imitation_radius'][0]
    }
    print(f"Test parameters: {test_params}")
    
    test_result = run_simulation_with_params(test_params)
    print(f"Test result: {'Equilibrium reached' if test_result.get('equilibrium_reached') else 'No equilibrium'}")
    
    # If the test was successful, continue with the full parameter sweep
    if 'error' not in test_result:
        # Run parameter sweep
        print("\nStarting parameter sweep...")
        results = parameter_sweep(param_ranges, num_processes=4)  # Use 4 processes for speed
        
        # Analyze results
        print("\nAnalyzing results...")
        analysis = analyze_results(results)
        
        # Save results
        save_results(results, analysis, output_dir)
        
        # Plot sample trajectories
        plot_sample_trajectories(results, output_dir=output_dir)
        
        print(f"\nResults saved to {output_dir}")
    else:
        print(f"\nTest failed with error: {test_result.get('error')}")
        print("Skipping full parameter sweep.")
