#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Parameter Equilibrium Analyzer

This script systematically tests different parameter combinations to find ranges
where the fish population reaches equilibrium in the ABM simulation.
"""

import numpy as np
import matplotlib.pyplot as plt
import copy
import pandas as pd
from tqdm import tqdm
import os
import itertools
import multiprocessing
from functools import partial
import json
import traceback
import sys

print("Starting parameter equilibrium analyzer...")

# Import simulation modules
try:
    from parameters import *
    print("Successfully imported parameters")
    import DynamicCoop as sim
    print("Successfully imported DynamicCoop")
except Exception as e:
    print(f"Error importing modules: {e}")
    traceback.print_exc()
    sys.exit(1)

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
        if param_name == 'noncoop':
            original_params[param_name] = sim.noncoop
            sim.noncoop = params[param_name]
        elif param_name == 'reproduction_rate':
            # For reproduction_rate, we need to handle differently since it's part of fish agents
            original_params[param_name] = params[param_name]  # Just store for reference
        else:
            original_params[param_name] = globals()[param_name]
            globals()[param_name] = params[param_name]
    
    # Reset the simulation
    try:
        print("Initializing simulation...")
        sim.initialize('both')
        
        # Run the simulation but capture the fish counts without plotting
        fish_counts = [sim.K]  # Start with carrying capacity
        
        print(f"Running simulation for {sim.n} time steps...")
        for j in range(1, sim.n):
            sim.update_one_unit_time()
            sim.observe()
            fish_counts.append(sim.total_fish_count[-1])
            
            # Print progress occasionally
            if j % 50 == 0:
                print(f"  Step {j}/{sim.n}: Fish count = {fish_counts[-1]}")
                
        print(f"Simulation complete. Final fish count: {fish_counts[-1]}")
    except Exception as e:
        print(f"Error during simulation: {e}")
        traceback.print_exc()
        
        # Restore original parameters
        for param_name in original_params:
            if param_name == 'noncoop':
                sim.noncoop = original_params[param_name]
            elif param_name != 'reproduction_rate':  # Skip reproduction_rate
                globals()[param_name] = original_params[param_name]
                
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
            sim.noncoop = original_params[param_name]
        else:
            globals()[param_name] = original_params[param_name]
    
    return {
        'params': params,
        'fish_counts': fish_counts,
        'equilibrium_reached': eq_reached,
        'equilibrium_value': eq_value,
        'time_to_equilibrium': eq_time,
        'status': eq_status
    }

def parameter_sweep(param_ranges, num_processes=4):
    """
    Perform a parameter sweep to find ranges where equilibrium is reached.
    
    Parameters:
    -----------
    param_ranges : dict
        Dictionary of parameter names and the values to test
    num_processes : int
        Number of parallel processes to use
        
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
    
    # Run simulations in parallel
    print(f"Running {len(param_dicts)} parameter combinations...")
    
    if num_processes > 1:
        with multiprocessing.Pool(num_processes) as pool:
            results = list(tqdm(
                pool.imap(run_simulation_with_params, param_dicts),
                total=len(param_dicts)
            ))
    else:
        results = []
        for params in tqdm(param_dicts):
            results.append(run_simulation_with_params(params))
    
    return results

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

def save_results(results, analysis, output_dir):
    """
    Save parameter sweep results and analysis to files.
    
    Parameters:
    -----------
    results : list
        List of simulation results from parameter_sweep
    analysis : dict
        Analysis results from analyze_results
    output_dir : str
        Directory to save results to
    """
    # Save results to JSON
    results_file = os.path.join(output_dir, "parameter_sweep_results.json")
    
    # Convert results to JSON-serializable format
    json_results = {'results': [], 'analysis': {}}
    
    # Process individual results
    for result in results:
        json_result = {}
        for key, value in result.items():
            # Handle fish_counts (potentially numpy arrays)
            if key == 'fish_counts':
                json_result[key] = [int(count) for count in value]
            # Handle params dictionary
            elif key == 'params':
                json_result[key] = {k: float(v) if isinstance(v, (np.float32, np.float64)) else int(v) if isinstance(v, np.integer) else v 
                                   for k, v in value.items()}
            # Handle other numpy types
            elif isinstance(value, (np.integer, np.floating)):
                json_result[key] = float(value) if isinstance(value, np.floating) else int(value)
            else:
                json_result[key] = value
        json_results['results'].append(json_result)
    
    # Process analysis results
    for key, value in analysis.items():
        if key == 'equilibrium_ranges':
            # Handle equilibrium ranges
            json_results['analysis'][key] = {}
            for param, ranges in value.items():
                json_results['analysis'][key][param] = {
                    'min': float(ranges['min']),
                    'max': float(ranges['max']),
                    'values': [float(v) for v in ranges['values']]
                }
        elif key == 'best_parameters':
            # Handle best parameters
            if value is not None:
                json_results['analysis'][key] = {k: float(v) if isinstance(v, (np.float32, np.float64, float)) 
                                              else int(v) if isinstance(v, (np.integer, int)) else v 
                                              for k, v in value.items()}
            else:
                json_results['analysis'][key] = None
        elif key == 'status_counts':
            # Handle status counts
            json_results['analysis'][key] = {k: int(v) for k, v in value.items()}
        else:
            # Handle other values
            json_results['analysis'][key] = float(value) if isinstance(value, (np.float32, np.float64, float)) else value
    
    try:
        with open(os.path.join(output_dir, "parameter_sweep_results.json"), 'w') as f:
            json.dump(json_results, f, indent=2)
        print(f"Results saved to {os.path.join(output_dir, 'parameter_sweep_results.json')}")
    except Exception as e:
        print(f"Error saving results to JSON: {e}")
        
        # Fallback: save as pickle
        pickle_file = os.path.join(output_dir, "parameter_sweep_results.pkl")
        import pickle
        with open(pickle_file, 'wb') as f:
            pickle.dump({'results': results, 'analysis': analysis}, f)
        print(f"Results saved as pickle to {pickle_file}")
    
    # Create parameter range plots
    if len(analysis['equilibrium_ranges']) > 0:
        # Create plot for each parameter pair
        param_names = list(analysis['equilibrium_ranges'].keys())
        
        # Create a dataframe with only equilibrium results
        equilibrium_data = [r for r in results if r['equilibrium_reached']]
        eq_df = pd.DataFrame([
            {
                **r['params'],
                'equilibrium_value': r['equilibrium_value'],
                'equilibrium_certainty': calculate_equilibrium_certainty(r)
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
                        c=eq_df['equilibrium_certainty'],
                        cmap='viridis', 
                        alpha=0.7,
                        s=50
                    )
                    
                    plt.colorbar(scatter, label='Equilibrium Certainty Score')
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
        # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
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
        # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "non_equilibrium_trajectories.png"), dpi=300)
        plt.close()

# Global variables for equilibrium detection
global_window_size = 20
global_warm_up_period = 30

if __name__ == "__main__":
    # Define parameter ranges to test
    # Using smaller ranges for an initial test
    param_ranges = {
        'rad_repulsion': np.linspace(0.01, 0.1, 2),  # Range for repulsion radius
        'rad_orientation': np.linspace(0.02, 0.15, 2),  # Range for orientation radius
        'imitation_radius': np.linspace(0.1, 0.5, 2),  # Range for imitation radius
        'noncoop': np.array([4]),  # Different values for non-cooperative agents
    }
    
    print(f"Parameter ranges to test: {param_ranges}")
    
    # Create a directory for storing parameter scan results
    output_dir = "simulation_output/parameter_scan"
    os.makedirs(output_dir, exist_ok=True)
    
    # Run a minimal test first with just one parameter combination
    print("\nRunning test with a single parameter combination...")
    test_params = {
        'rad_repulsion': param_ranges['rad_repulsion'][0],
        'rad_orientation': param_ranges['rad_orientation'][0],
        'imitation_radius': param_ranges['imitation_radius'][0],
        'noncoop': param_ranges['noncoop'][0]
    }
    print(f"Test parameters: {test_params}")
    
    test_result = run_simulation_with_params(test_params)
    print(f"Test result: {'Equilibrium reached' if test_result.get('equilibrium_reached') else 'No equilibrium'}")
    
    # If the test was successful, continue with the full parameter sweep
    if 'error' not in test_result:
        # Run parameter sweep
        print("\nStarting parameter sweep...")
        results = parameter_sweep(param_ranges, num_processes=1)  # Use 1 process to avoid conflicts
        
        # Analyze results
        print("\nAnalyzing results...")
        analysis = analyze_results(results)
        
        # Save results
        save_results(results, analysis, output_dir)
        
        # Plot sample trajectories
        plot_sample_trajectories(results, output_dir=output_dir)
        
        # Plot parameter relationships
        plot_parameter_relationships(results, output_dir)
        
        print(f"\nResults saved to {output_dir}")
    else:
        print(f"\nTest failed with error: {test_result.get('error')}")
        print("Skipping full parameter sweep.")
