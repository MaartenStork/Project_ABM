"""
Focused Sensitivity Analysis for DynamicCoop Model

This script performs detailed sensitivity analysis on the five most influential parameters
identified through Morris screening:

1. scale (μ*≈2.1, σ≈13.5)
2. imitation_period (μ*≈1.8, σ≈11.0)
3. cooperation_increase (μ*≈1.6, σ≈11.8)
4. q (catchability coefficient) (μ*≈1.6, σ≈12.6)
5. trust_decrease (μ*≈1.1, σ≈12.9)

Methods:
1. Sobol Analysis (5000 samples)
2. OFAT (One Factor At A Time) Analysis (5000 runs per parameter)

This provides extremely high statistical confidence in the results.
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import DynamicCoop as dc
import parameters
import pandas as pd
import os
from datetime import datetime
import seaborn as sns

# For Sobol analysis
from SALib.sample import saltelli
from SALib.analyze import sobol

# For parallel processing to handle large sample sizes
from multiprocessing import Pool, cpu_count
from functools import partial

# Disable visualization in DynamicCoop
dc.VISUALIZATION_ENABLED = False
dc.CREATE_GIF = False

def run_model(param_values=None, n_timesteps=300):
    """
    Run the DynamicCoop model with specified parameters.
    Returns multiple metrics for more comprehensive analysis.
    """
    try:
        if param_values is not None:
            # Set parameters
            parameters.scale = param_values[0]
            parameters.imitation_period = int(round(param_values[1]))  # Must be integer
            parameters.cooperation_increase = param_values[2]
            parameters.q = param_values[3]
            parameters.trust_decrease = param_values[4]
        
        # Initialize with default fish parameters
        dc.initialize('default')  # Use default experiment for consistent fish params
        
        if not dc.agents:  # Check if initialization failed
            print("Warning: No agents initialized")
            return {
                'final_fish': 0,
                'total_catch': 0,
                'stability': 1000,
                'sustainability': 0,
                'cooperation': 0,
                'trust': 0
            }
        
        # Run for more timesteps to capture mature behavior
        fish_counts = []
        catch_counts = []
        cooperation_levels = []
        trust_levels = []
        
        for _ in range(n_timesteps):
            dc.update_one_unit_time()
            
            # Get current fish count
            current_fish = len([ag for ag in dc.agents if ag.type == 'fish'])
            fish_counts.append(current_fish)
            
            # Get current catch total
            fishers = [ag for ag in dc.agents if ag.type == 'fishers']
            current_catch = sum(ag.harvest for ag in fishers)
            catch_counts.append(current_catch)
            
            # Track cooperation and trust
            if fishers:
                avg_effort = np.mean([1 - ag.effort for ag in fishers])  # Convert effort to cooperation (1 - effort)
                cooperation_levels.append(avg_effort)
                
                # Calculate average trust across all fisher pairs
                all_trust_scores = []
                for fisher in fishers:
                    all_trust_scores.extend(fisher.trust_scores.values())
                avg_trust = np.mean(all_trust_scores) if all_trust_scores else 0
                trust_levels.append(avg_trust)
        
        # Calculate metrics
        final_fish = fish_counts[-1] if fish_counts else 0
        total_catch = catch_counts[-1] if catch_counts else 0
        stability = np.std(fish_counts[-50:]) if len(fish_counts) >= 50 else 1000
        sustainability = final_fish / parameters.K
        mean_cooperation = np.mean(cooperation_levels[-50:]) if len(cooperation_levels) >= 50 else 0
        mean_trust = np.mean(trust_levels[-50:]) if len(trust_levels) >= 50 else 0
        
        return {
            'final_fish': final_fish,
            'total_catch': total_catch,
            'stability': stability,
            'sustainability': sustainability,
            'cooperation': mean_cooperation,
            'trust': mean_trust
        }
    
    except Exception as e:
        print(f"Error in run: {str(e)}")
        return {
            'final_fish': 0,
            'total_catch': 0,
            'stability': 1000,
            'sustainability': 0,
            'cooperation': 0,
            'trust': 0
        }

def run_model_sobol(params):
    """Wrapper for parallel Sobol analysis"""
    try:
        results = run_model(params)
        return [results['final_fish'], results['total_catch'], 
                results['stability'], results['sustainability'],
                results['cooperation'], results['trust']]
    except Exception as e:
        print(f"Error in run: {str(e)}")
        return [0, 0, 1000, 0, 0, 0]  # Default values for failed runs

def sobol_analysis(n_samples=5000):
    """
    Perform Sobol sensitivity analysis with very large sample size.
    """
    print(f"\n=== Running Sobol Analysis (N={n_samples}) ===")
    
    # Define problem for all five parameters
    problem = {
        'num_vars': 5,
        'names': ['scale', 'imitation_period', 'cooperation_increase', 'q', 'trust_decrease'],
        'bounds': [
            [parameters.scale * 0.5, parameters.scale * 1.5],
            [max(1, int(parameters.imitation_period * 0.5)), 
             int(parameters.imitation_period * 1.5)],
            [parameters.cooperation_increase * 0.5, parameters.cooperation_increase * 1.5],
            [parameters.q * 0.5, parameters.q * 1.5],
            [parameters.trust_decrease * 0.5, parameters.trust_decrease * 1.5]
        ]
    }
    
    # Generate samples
    param_values = saltelli.sample(problem, n_samples, calc_second_order=True)
    print(f"Total model evaluations: {param_values.shape[0]}")
    
    # Run parallel analysis
    n_cores = cpu_count() - 1  # Leave one core free
    print(f"Running parallel analysis on {n_cores} cores...")
    
    results = []
    chunk_size = 100  # Process in smaller chunks to avoid memory issues
    
    with Pool(n_cores) as pool:
        for i in range(0, len(param_values), chunk_size):
            chunk = param_values[i:i + chunk_size]
            chunk_results = list(tqdm(
                pool.imap(run_model_sobol, chunk),
                total=len(chunk),
                desc=f"Sobol Analysis Chunk {i//chunk_size + 1}/{len(param_values)//chunk_size + 1}"
            ))
            results.extend(chunk_results)
    
    results = np.array(results)
    
    # Save raw results
    np.save(f'sobol_raw_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.npy', results)
    
    # Analyze for each metric
    metrics = ['Fish Population', 'Total Catch', 'Population Stability', 
              'Sustainability', 'Cooperation Level', 'Trust Level']
    Si_all = []
    
    for i, metric in enumerate(metrics):
        Si = sobol.analyze(problem, results[:, i], calc_second_order=True, print_to_console=False)
        Si_all.append(Si)
        
        print(f"\n=== Sobol Indices for {metric} ===")
        print("First-order indices (S1):")
        for j, param in enumerate(problem['names']):
            print(f"{param}: {Si['S1'][j]:.3f} ± {Si['S1_conf'][j]:.3f}")
        
        print("\nTotal-order indices (ST):")
        for j, param in enumerate(problem['names']):
            print(f"{param}: {Si['ST'][j]:.3f} ± {Si['ST_conf'][j]:.3f}")
        
        # Only print strongest second-order interactions
        print("\nStrongest parameter interactions (S2):")
        S2 = Si['S2']
        S2_conf = Si['S2_conf']
        param_pairs = [(i, j) for i in range(len(problem['names'])) 
                      for j in range(i+1, len(problem['names']))]
        interactions = [(problem['names'][i], problem['names'][j], S2[i][j], S2_conf[i][j])
                       for i, j in param_pairs]
        interactions.sort(key=lambda x: abs(x[2]), reverse=True)
        for p1, p2, s2, conf in interactions[:3]:  # Top 3 interactions
            print(f"{p1} × {p2}: {s2:.3f} ± {conf:.3f}")
    
    return Si_all

def ofat_analysis(n_samples=5000):
    """
    Perform One-Factor-at-a-Time analysis with very large sample size.
    """
    print(f"\n=== Running OFAT Analysis (N={n_samples}) ===")
    
    # Store original values
    orig_values = {
        'scale': parameters.scale,
        'imitation_period': parameters.imitation_period,
        'cooperation_increase': parameters.cooperation_increase,
        'q': parameters.q,
        'trust_decrease': parameters.trust_decrease
    }
    
    # Parameter ranges (20 points each)
    ranges = {
        'scale': np.linspace(orig_values['scale'] * 0.5, orig_values['scale'] * 1.5, 20),
        'imitation_period': np.linspace(max(1, int(orig_values['imitation_period'] * 0.5)),
                                      int(orig_values['imitation_period'] * 1.5), 20),
        'cooperation_increase': np.linspace(orig_values['cooperation_increase'] * 0.5,
                                          orig_values['cooperation_increase'] * 1.5, 20),
        'q': np.linspace(orig_values['q'] * 0.5, orig_values['q'] * 1.5, 20),
        'trust_decrease': np.linspace(orig_values['trust_decrease'] * 0.5,
                                    orig_values['trust_decrease'] * 1.5, 20)
    }
    
    metrics = ['fish', 'catch', 'stability', 'sustainability', 'cooperation', 'trust']
    results = {param: {metric: [] for metric in metrics} for param in ranges.keys()}
    
    # Analyze each parameter
    for param_name, param_range in ranges.items():
        print(f"\nAnalyzing {param_name}...")
        
        # Reset all parameters to original values
        for name, value in orig_values.items():
            setattr(parameters, name, value)
        
        for value in tqdm(param_range, desc=f"{param_name} Analysis"):
            setattr(parameters, param_name, value)
            metric_values = {metric: [] for metric in metrics}
            
            for _ in range(n_samples // 20):  # Divide samples across parameter values
                res = run_model()
                for metric in metrics:
                    metric_values[metric].append(res[metric])
            
            for metric in metrics:
                results[param_name][metric].append(np.mean(metric_values[metric]))
    
    # Restore original values
    for name, value in orig_values.items():
        setattr(parameters, name, value)
    
    # Plot results
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        for param_name in ranges.keys():
            param_range = ranges[param_name]
            relative_range = param_range / orig_values[param_name]
            ax.plot(relative_range, results[param_name][metric], 
                   label=param_name, linewidth=2)
        
        ax.set_xlabel('Parameter Value (relative to baseline)')
        ax.set_ylabel(metric.capitalize())
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('focused_ofat_analysis.png', dpi=300, bbox_inches='tight')
    print("\nOFAT analysis plot saved as 'focused_ofat_analysis.png'")
    
    return results

def plot_sobol_results(Si_all, problem, timestamp):
    """
    Create visualization plots for Sobol analysis results.
    """
    metrics = ['Fish Population', 'Total Catch', 'Population Stability', 
              'Sustainability', 'Cooperation Level', 'Trust Level']
    
    # Set up the plotting style
    plt.style.use('seaborn')
    
    # First-order and Total-order indices plot
    fig, axes = plt.subplots(3, 2, figsize=(15, 20))
    axes = axes.flatten()
    
    for idx, (metric, ax) in enumerate(zip(metrics, axes)):
        Si = Si_all[idx]
        
        # Data preparation
        indices = pd.DataFrame({
            'Parameter': problem['names'] * 2,
            'Index Type': ['First-order'] * len(problem['names']) + ['Total-order'] * len(problem['names']),
            'Value': np.concatenate([Si['S1'], Si['ST']]),
            'Confidence': np.concatenate([Si['S1_conf'], Si['ST_conf']])
        })
        
        # Create grouped bar plot
        sns.barplot(
            data=indices,
            x='Parameter',
            y='Value',
            hue='Index Type',
            ax=ax,
            capsize=0.1
        )
        
        # Customize plot
        ax.set_title(f'{metric} Sensitivity Indices')
        ax.set_xlabel('Parameters')
        ax.set_ylabel('Sensitivity Index')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'focused_sensitivity_results/sobol_indices_{timestamp}.png', 
                dpi=300, bbox_inches='tight')
    
    # Second-order interactions heatmap
    fig, axes = plt.subplots(3, 2, figsize=(15, 20))
    axes = axes.flatten()
    
    for idx, (metric, ax) in enumerate(zip(metrics, axes)):
        Si = Si_all[idx]
        
        # Create heatmap of second-order indices
        sns.heatmap(
            Si['S2'],
            ax=ax,
            xticklabels=problem['names'],
            yticklabels=problem['names'],
            cmap='YlOrRd',
            center=0,
            annot=True,
            fmt='.2f'
        )
        
        ax.set_title(f'{metric} Parameter Interactions')
    
    plt.tight_layout()
    plt.savefig(f'focused_sensitivity_results/sobol_interactions_{timestamp}.png',
                dpi=300, bbox_inches='tight')

def main():
    # Create results directory
    os.makedirs('focused_sensitivity_results', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Run Sobol analysis
    sobol_results = sobol_analysis(n_samples=5000)
    
    # Save results
    np.save(f'focused_sensitivity_results/sobol_results_{timestamp}.npy', sobol_results)
    
    # Create visualization plots
    problem = {
        'num_vars': 5,
        'names': ['scale', 'imitation_period', 'cooperation_increase', 'q', 'trust_decrease']
    }
    plot_sobol_results(sobol_results, problem, timestamp)
    
    print("\nAnalysis complete! Results saved in 'focused_sensitivity_results' directory")
    print(f"Plots saved as:")
    print(f"- focused_sensitivity_results/sobol_indices_{timestamp}.png")
    print(f"- focused_sensitivity_results/sobol_interactions_{timestamp}.png")

if __name__ == '__main__':
    main() 