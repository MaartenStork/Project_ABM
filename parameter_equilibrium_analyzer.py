#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Parameter Equilibrium Analyzer

This script systematically tests different parameter combinations to find ranges
where the fish population reaches equilibrium in the ABM simulation.
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import seaborn as sns
from tqdm import tqdm
from itertools import product
import multiprocessing as mp
import functools
import traceback
import sys

print("Starting parameter equilibrium analyzer...")

# Import simulation modules
try:
    from parameters import *
    print("Successfully imported parameters")
    import DynamicCoop as sim
    from DynamicCoop import agent
    print("Successfully imported DynamicCoop")
except Exception as e:
    print(f"Error importing modules: {e}")
    traceback.print_exc()
    sys.exit(1)

# Define equilibrium criteria
def calculate_equilibrium_certainty(fish_counts, window_size=30, warm_up_period=50):
    """
    Calculate a certainty score for equilibrium detection.
    
    Parameters:
    -----------
    fish_counts : list
        List of fish counts over time
    window_size : int
        Size of sliding window to check for equilibrium
    warm_up_period : int
        Number of time steps to ignore at the beginning
        
    Returns:
    --------
    float
        Certainty score between 0 and 1, where 1 is highest certainty
    """
    # Check if we have enough data
    if len(fish_counts) < warm_up_period + window_size:
        return 0.0
    
    # Skip warm-up period
    post_warmup = fish_counts[warm_up_period:]
    
    # Use the last window for calculation
    window = post_warmup[-window_size:]
    
    # Calculate coefficient of variation
    cv = np.std(window) / np.mean(window) if np.mean(window) > 0 else float('inf')
    
    # Calculate trend by comparing first and second half of window
    first_half = window[:len(window)//2]
    second_half = window[len(window)//2:]
    first_mean = np.mean(first_half)
    second_mean = np.mean(second_half)
    trend = abs(second_mean - first_mean) / first_mean if first_mean > 0 else float('inf')
    
    # Convert CV to a score (lower CV = higher score)
    cv_max = 0.2  # Maximum CV that would still be considered for equilibrium
    cv_score = 1 - min(cv, cv_max) / cv_max
    
    # Convert trend to a score (lower trend = higher score)
    trend_max = 0.2  # Maximum trend that would still be considered for equilibrium
    trend_score = 1 - min(trend, trend_max) / trend_max
    
    # Combine scores (geometric mean)
    certainty = cv_score * trend_score
    
    return max(0.0, min(1.0, certainty))  # Ensure score is between 0 and 1

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
    
    # Check if fish population reaches 500 (unchecked growth)
    if np.max(post_warmup) >= 500:
        return False, None, None, "unchecked_growth"
    
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
    
    # Set parameters for this run
    for param_name, value in params.items():
        if param_name == 'reproduction_rate':
            # reproduction_rate is handled during initialization
            continue
        elif hasattr(sim, param_name):
            setattr(sim, param_name, value)
    
    print(f"Running simulation with params: {params}")
    
    # Initialize simulation
    print("Initializing simulation...")
    
    # Special handling for reproduction_rate
    if 'reproduction_rate' in params:
        # Store the original reproduction rate value
        reproduction_rate_value = params['reproduction_rate']
        
        # Monkey patch the agent initialization to set reproduction_rate
        original_agent_init = agent.__init__
        
        def modified_agent_init(self, **kwargs):
            # Call the original init
            original_agent_init(self, **kwargs)
            # If this is a fish agent being created, set the reproduction rate
            if getattr(self, 'type', None) == 'fish':
                self.reproduction_rate = reproduction_rate_value
        
        # Replace the agent init method temporarily
        agent.__init__ = modified_agent_init
    
    # Initialize with modified parameters
    sim.initialize('both')
    
    # Run simulation
    time_steps = 150
    print(f"Running simulation for {time_steps} time steps...")
    
    fish_counts = []
    for step in range(time_steps):
        sim.update_fish()
        fish_counts.append(sim.total_fish_count[-1])
        
        if step % 50 == 0:
            print(f"  Step {step}/{time_steps}: Fish count = {fish_counts[-1]}")
    
    print(f"Simulation complete. Final fish count: {fish_counts[-1]}")
    
    # Check if equilibrium was reached
    eq_reached, eq_value, eq_time, eq_status = is_equilibrium(
        fish_counts, 
        window_size=global_window_size, 
        warm_up_period=global_warm_up_period
    )
    
    # Restore original parameters
    for param_name in original_params:
        setattr(sim, param_name, original_params[param_name])
    
    # Restore original agent.__init__ method if modified
    if 'reproduction_rate' in params:
        agent.__init__ = original_agent_init
    
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
    param_combinations = list(product(*param_values))
    
    # Convert tuples to dictionaries
    param_dicts = [dict(zip(param_names, combo)) for combo in param_combinations]
    
    # Run simulations in parallel
    print(f"Running {len(param_dicts)} parameter combinations...")
    
    if num_processes > 1:
        with mp.Pool(num_processes) as pool:
            results = list(tqdm(
                pool.imap(run_simulation_with_params, param_dicts),
                total=len(param_dicts)
            ))
    else:
        results = []
        for params in tqdm(param_dicts):
            results.append(run_simulation_with_params(params))
    
    return results

def analyze_results(results, output_dir='simulation_output/parameter_scan'):
    """
    Analyze the results of the parameter sweep and generate visualizations.
    
    Parameters:
    -----------
    results : list
        List of dictionaries containing simulation results
    output_dir : str
        Directory to save output files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract parameters and results
    params_list = []
    eq_values = []
    eq_reached = []
    eq_times = []
    eq_status = []
    fish_counts_list = []
    certainty_scores = []
    
    for result in results:
        if 'error' in result:
            continue
            
        params_list.append(result['params'])
        eq_values.append(result['equilibrium_value'] if result['equilibrium_reached'] else 0)
        eq_reached.append(result['equilibrium_reached'])
        eq_times.append(result['time_to_equilibrium'] if result['equilibrium_reached'] else 0)
        eq_status.append(result['status'])
        fish_counts_list.append(result['fish_counts'])
        
        # Calculate certainty score for this result
        if result['equilibrium_reached']:
            certainty_score = calculate_equilibrium_certainty(result['fish_counts'])
        else:
            certainty_score = 0
        certainty_scores.append(certainty_score)
        
    # Create a DataFrame for easier analysis and CSV export
    data = []
    for i, params in enumerate(params_list):
        row = params.copy()
        row['equilibrium_reached'] = eq_reached[i]
        row['equilibrium_value'] = eq_values[i]
        row['time_to_equilibrium'] = eq_times[i]
        row['status'] = eq_status[i]
        row['certainty_score'] = certainty_scores[i]
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # Save to CSV
    csv_path = os.path.join(output_dir, 'parameter_sweep_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"Results saved to CSV: {csv_path}")
    
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

# Custom JSON encoder for NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

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
    # Save the results to JSON
    save_results_to_json(results, os.path.join(output_dir, 'parameter_sweep_results.json'))
    
    # Save analysis to JSON with custom encoder for NumPy types
    try:
        with open(os.path.join(output_dir, 'analysis_results.json'), 'w') as f:
            json.dump(analysis, f, indent=2, cls=NumpyEncoder)
    except TypeError as e:
        print(f"Warning: Could not save analysis to JSON: {e}")
        print("Falling back to pickle format")
        with open(os.path.join(output_dir, 'analysis_results.pkl'), 'wb') as f:
            pickle.dump(analysis, f)
    
    print("Results saved to files in the output directory.")

def save_results_to_json(results, output_file):
    """
    Save parameter sweep results to a JSON file.
    
    Parameters:
    -----------
    results : list
        List of simulation results from parameter_sweep
    output_file : str
        Path to save the JSON file
    """
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
    
    # Convert results to DataFrame for easier analysis
    results_df = pd.DataFrame([
        {
            **r['params'],
            'equilibrium_reached': r.get('equilibrium_reached', False),
            'equilibrium_value': r.get('equilibrium_value', 0),
            'time_to_equilibrium': r.get('time_to_equilibrium', 0),
            'status': r.get('status', 'unknown'),
            'certainty': calculate_equilibrium_certainty(r.get('fish_counts', []))
        }
        for r in results
    ])
    
    # If we have no results, return
    if len(results_df) == 0:
        print("No results to plot parameter relationships")
        return
    
    # Get parameter names
    param_names = [k for k in results[0]['params'].keys()]
    
    # If we have at least two parameters that vary, create pairwise plots
    varying_params = [p for p in param_names if len(results_df[p].unique()) > 1]
    
    if len(varying_params) >= 2:
        # Create pairwise plots for parameters with equilibrium certainty as color
        for i, param1 in enumerate(varying_params):
            for param2 in varying_params[i+1:]:
                plt.figure(figsize=(10, 8))
                
                # Create scatter plot
                scatter = plt.scatter(
                    results_df[param1],
                    results_df[param2],
                    c=results_df['certainty'],
                    cmap='viridis',
                    s=100,
                    alpha=0.7
                )
                
                # Add colorbar
                cbar = plt.colorbar(scatter)
                cbar.set_label('Equilibrium Certainty')
                
                # Add labels and title
                plt.xlabel(param1)
                plt.ylabel(param2)
                plt.title(f'Parameter Relationship: {param1} vs {param2}')
                
                # Use log scale if parameter values span orders of magnitude
                if max(results_df[param1]) / max(1e-10, min(results_df[param1])) > 10:
                    plt.xscale('log')
                if max(results_df[param2]) / max(1e-10, min(results_df[param2])) > 10:
                    plt.yscale('log')
                
                # Save the plot
                plt.tight_layout()
                plt.savefig(os.path.join(plot_dir, f"{param1}_vs_{param2}.png"), dpi=300)
                plt.close()
    
    # Create individual parameter plots showing effect on equilibrium certainty
    for param in param_names:
        if len(results_df[param].unique()) > 1:
            plt.figure(figsize=(10, 6))
            
            # Group by parameter and calculate mean certainty
            param_effect = results_df.groupby(param)['certainty'].mean().reset_index()
            
            # Plot
            plt.plot(param_effect[param], param_effect['certainty'], 'o-', linewidth=2, markersize=10)
            
            # Add labels and title
            plt.xlabel(param)
            plt.ylabel('Mean Equilibrium Certainty')
            plt.title(f'Effect of {param} on Equilibrium Certainty')
            
            # Use log scale if parameter values span orders of magnitude
            if max(results_df[param]) / max(1e-10, min(results_df[param])) > 10:
                plt.xscale('log')
            
            # Add grid
            plt.grid(alpha=0.3)
            
            # Save the plot
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f"{param}_effect.png"), dpi=300)
            plt.close()
    
    print(f"Parameter relationship plots saved to {plot_dir}")

def plot_parameter_relationships(results, output_dir='simulation_output/parameter_scan'):
    """
    Plot relationships between parameters and equilibrium certainty.
    
    Parameters:
    -----------
    results : list
        List of simulation results
    output_dir : str
        Directory to save plots to
    """
    # Create output directory for plots
    plot_dir = os.path.join(output_dir, 'plots', 'parameter_relationships')
    os.makedirs(plot_dir, exist_ok=True)
    
    # Create a DataFrame from results for easier analysis
    data = []
    for result in results:
        if 'params' in result and 'equilibrium_certainty' in result:
            row = result['params'].copy()
            row['equilibrium_reached'] = result.get('equilibrium_reached', False)
            row['equilibrium_certainty'] = result.get('equilibrium_certainty', 0)
            row['equilibrium_fish_count'] = result.get('equilibrium_fish_count', 0)
            data.append(row)
    
    if not data:
        print("No valid results to plot parameter relationships")
        return
    
    results_df = pd.DataFrame(data)
    
    # Save the DataFrame to CSV for further analysis
    csv_path = os.path.join(output_dir, 'parameter_results.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"Parameter results saved to {csv_path}")
    
    # Get parameters with more than one value
    params = [col for col in results_df.columns if col in ['rad_repulsion', 'reproduction_rate', 
                                                         'imitation_radius', 'rad_orientation', 'noncoop']]
    multi_value_params = [p for p in params if len(results_df[p].unique()) > 1]
    
    # If we have multiple parameters with multiple values, create pairwise plots
    if len(multi_value_params) >= 2:
        # Create pairwise plots for parameters with multiple values
        for i, param1 in enumerate(multi_value_params):
            for param2 in multi_value_params[i+1:]:
                plt.figure(figsize=(10, 8))
                
                # Create scatter plot
                scatter = plt.scatter(results_df[param1], results_df[param2], 
                           c=results_df['equilibrium_certainty'], 
                           cmap='viridis', 
                           s=100, 
                           alpha=0.7)
                
                # Add colorbar
                cbar = plt.colorbar(scatter)
                cbar.set_label('Equilibrium Certainty')
                
                # Set axis labels
                plt.xlabel(param1)
                plt.ylabel(param2)
                
                # Use log scale if parameter ranges span orders of magnitude
                if max(results_df[param1]) / max(1e-10, min(results_df[param1])) > 10:
                    plt.xscale('log')
                if max(results_df[param2]) / max(1e-10, min(results_df[param2])) > 10:
                    plt.yscale('log')
                
                # Add grid
                plt.grid(alpha=0.3)
                
                # Add title
                plt.title(f'Effect of {param1} and {param2} on Equilibrium Certainty')
                
                # Save the plot
                plt.tight_layout()
                plt.savefig(os.path.join(plot_dir, f"{param1}_vs_{param2}.png"), dpi=300)
                plt.close()
    
    # Create individual parameter effect plots
    for param in params:
        if len(results_df[param].unique()) > 1:  # Only plot if parameter has multiple values
            plt.figure(figsize=(10, 6))
            
            # Group by parameter and calculate mean equilibrium certainty
            param_effect = results_df.groupby(param)['equilibrium_certainty'].agg(['mean', 'std', 'count']).reset_index()
            
            # Plot mean with error bars
            plt.errorbar(param_effect[param], param_effect['mean'], 
                       yerr=param_effect['std']/np.sqrt(param_effect['count']),
                       marker='o', linestyle='-', linewidth=2, markersize=8)
            
            # Set axis labels
            plt.xlabel(param)
            plt.ylabel('Mean Equilibrium Certainty')
            plt.title(f'Effect of {param} on Equilibrium Certainty')
            
            # Use log scale if parameter ranges span orders of magnitude
            if max(results_df[param]) / max(1e-10, min(results_df[param])) > 10:
                plt.xscale('log')
            
            # Add grid
            plt.grid(alpha=0.3)
            
            # Save the plot
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f"{param}_effect.png"), dpi=300)
            plt.close()
    
    print(f"Parameter relationship plots saved to {plot_dir}")

def plot_sample_trajectories(results, output_dir='simulation_output/parameter_scan'):
    """
    Plot sample trajectories for equilibrium and non-equilibrium cases.
    
    Parameters:
    -----------
    results : list
        List of simulation results
    output_dir : str
        Directory to save plots to
    """
    # Create output directory for plots
    plot_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    
    # Separate results into equilibrium and non-equilibrium
    eq_results = [r for r in results if r.get('equilibrium_reached', False)]
    non_eq_results = [r for r in results if not r.get('equilibrium_reached', False)]
    
    # Plot sample equilibrium trajectories
    if eq_results:
        # Sample up to 5 equilibrium trajectories
        eq_samples = list(range(min(5, len(eq_results))))
        
        plt.figure(figsize=(10, 6))
        for idx in eq_samples:
            result = eq_results[idx]
            time_steps = range(len(result['fish_counts']))
            label = ", ".join(f"{k}={v:.3f}" for k, v in result['params'].items())
            plt.plot(time_steps, result['fish_counts'], label=label)
            
        plt.xlabel('Time Step')
        plt.ylabel('Fish Count')
        plt.title('Fish Population Trajectories (Equilibrium)')
        plt.grid(alpha=0.3)
        # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "equilibrium_trajectories.png"), dpi=300)
        plt.close()
    
    # Plot sample non-equilibrium trajectories
    if non_eq_results:
        # Sample up to 5 non-equilibrium trajectories
        non_eq_samples = list(range(min(5, len(non_eq_results))))
        
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
    'rad_repulsion': np.logspace(np.log10(0.005), np.log10(0.2), 1),
    'reproduction_rate': np.logspace(np.log10(0.01), np.log10(0.8), 3),
    'imitation_radius': np.logspace(np.log10(0.05), np.log10(0.8), 1),
    'rad_orientation': np.logspace(np.log10(0.01), np.log10(0.2), 1),
    'noncoop': np.array([2])
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
