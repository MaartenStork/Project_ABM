"""
Focused Sensitivity Analysis for Five Key Parameters
Analyzing only fish population for:

1. scale (μ*≈2.1, σ≈13.5)
2. imitation_period (μ*≈1.8, σ≈11.0)
3. cooperation_increase (μ*≈1.6, σ≈11.8)
4. q (catchability coefficient) (μ*≈1.6, σ≈12.6)
5. trust_decrease (μ*≈1.1, σ≈12.9)

Methods:
1. Sobol Analysis (5000 samples)
2. OFAT Analysis (100 points × 50 repetitions per parameter)
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import DynamicCoop as dc
import parameters
import os
from datetime import datetime
from multiprocessing import Pool, cpu_count

# For Sobol analysis
from SALib.sample import saltelli
from SALib.analyze import sobol

# Disable visualization
dc.VISUALIZATION_ENABLED = False
dc.CREATE_GIF = False

def run_model(param_values=None):
    """Run model and return final fish population"""
    try:
        if param_values is not None:
            # Set parameters
            parameters.scale = param_values[0]
            parameters.imitation_period = int(round(param_values[1]))  # Must be integer
            parameters.cooperation_increase = param_values[2]
            parameters.q = param_values[3]
            parameters.trust_decrease = param_values[4]
        
        # Initialize with default experiment
        dc.initialize('default')
        
        # Run simulation
        for _ in range(300):  # 300 timesteps
            dc.update_one_unit_time()
        
        return dc.total_fish_count[-1]
    
    except Exception as e:
        print(f"Error in run: {str(e)}")
        return 0  # Return 0 for failed runs

def sobol_analysis(n_samples=5000):
    """Perform Sobol sensitivity analysis"""
    print(f"\n=== Running Sobol Analysis (N={n_samples}) ===")
    
    # Define problem
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
    chunk_size = 100  # Process in smaller chunks
    
    with Pool(n_cores) as pool:
        for i in range(0, len(param_values), chunk_size):
            chunk = param_values[i:i + chunk_size]
            chunk_results = list(tqdm(
                pool.imap(run_model, chunk),
                total=len(chunk),
                desc=f"Sobol Analysis Chunk {i//chunk_size + 1}/{len(param_values)//chunk_size + 1}"
            ))
            results.extend(chunk_results)
    
    results = np.array(results)
    
    # Save raw results
    os.makedirs('sensitivity_results', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    np.save(f'sensitivity_results/sobol_raw_{timestamp}.npy', results)
    
    # Analyze results
    Si = sobol.analyze(problem, results, calc_second_order=True, print_to_console=True)
    
    # Plot results
    plot_sobol_results(Si, problem, timestamp)
    
    return Si

def plot_sobol_results(Si, problem, timestamp):
    """Create visualization of Sobol results"""
    # First-order and total-order indices
    plt.figure(figsize=(12, 6))
    
    indices = np.arange(len(problem['names']))
    width = 0.35
    
    plt.bar(indices - width/2, Si['S1'], width, label='First-order', 
            yerr=Si['S1_conf'], capsize=5)
    plt.bar(indices + width/2, Si['ST'], width, label='Total-order',
            yerr=Si['ST_conf'], capsize=5)
    
    plt.xlabel('Parameters')
    plt.ylabel('Sensitivity Index')
    plt.title('Sobol Sensitivity Analysis - Fish Population')
    plt.xticks(indices, problem['names'], rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(f'sensitivity_results/sobol_indices_{timestamp}.png', dpi=300)
    
    # Second-order interactions
    plt.figure(figsize=(8, 6))
    plt.imshow(Si['S2'], cmap='YlOrRd', aspect='equal')
    plt.colorbar(label='Second-order Sensitivity Index')
    plt.xticks(range(len(problem['names'])), problem['names'], rotation=45)
    plt.yticks(range(len(problem['names'])), problem['names'])
    plt.title('Parameter Interactions')
    plt.tight_layout()
    
    plt.savefig(f'sensitivity_results/sobol_interactions_{timestamp}.png', dpi=300)

def ofat_analysis(n_points=100, n_reps=50):
    """Perform One-Factor-At-A-Time analysis"""
    print(f"\n=== Running OFAT Analysis (points={n_points}, reps={n_reps}) ===")
    
    # Store original values
    orig_values = {
        'scale': parameters.scale,
        'imitation_period': parameters.imitation_period,
        'cooperation_increase': parameters.cooperation_increase,
        'q': parameters.q,
        'trust_decrease': parameters.trust_decrease
    }
    
    # Define parameter ranges
    ranges = {
        'scale': np.linspace(orig_values['scale']*0.5, orig_values['scale']*1.5, n_points),
        'imitation_period': np.linspace(max(1, int(orig_values['imitation_period']*0.5)),
                                      int(orig_values['imitation_period']*1.5), n_points),
        'cooperation_increase': np.linspace(orig_values['cooperation_increase']*0.5,
                                          orig_values['cooperation_increase']*1.5, n_points),
        'q': np.linspace(orig_values['q']*0.5, orig_values['q']*1.5, n_points),
        'trust_decrease': np.linspace(orig_values['trust_decrease']*0.5,
                                    orig_values['trust_decrease']*1.5, n_points)
    }
    
    results = {param: [] for param in ranges.keys()}
    
    # Analyze each parameter
    for param_name, param_range in ranges.items():
        print(f"\nAnalyzing {param_name}...")
        
        # Reset all parameters to original values
        for name, value in orig_values.items():
            setattr(parameters, name, value)
        
        param_results = []
        for value in tqdm(param_range, desc=f"{param_name} Analysis"):
            setattr(parameters, param_name, value)
            runs = [run_model() for _ in range(n_reps)]
            param_results.append((np.mean(runs), np.std(runs)))
        
        results[param_name] = param_results
    
    # Restore original values
    for name, value in orig_values.items():
        setattr(parameters, name, value)
    
    # Plot results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.figure(figsize=(12, 8))
    
    for param_name in ranges.keys():
        param_range = ranges[param_name]
        means, stds = zip(*results[param_name])
        relative_range = param_range / orig_values[param_name]
        
        plt.plot(relative_range, means, label=param_name, linewidth=2)
        plt.fill_between(relative_range, 
                        np.array(means) - np.array(stds),
                        np.array(means) + np.array(stds),
                        alpha=0.2)
    
    plt.xlabel('Parameter Value (relative to baseline)')
    plt.ylabel('Fish Population')
    plt.title('OFAT Sensitivity Analysis')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(f'sensitivity_results/ofat_analysis_{timestamp}.png', dpi=300)
    
    return results

def main():
    # Create results directory
    os.makedirs('sensitivity_results', exist_ok=True)
    
    # Run both analyses
    print("Starting sensitivity analysis for five key parameters...")
    print("This will take several hours. Results will be saved progressively.")
    
    # Run Sobol analysis
    Si = sobol_analysis(n_samples=5000)
    
    # Run OFAT analysis
    ofat_results = ofat_analysis(n_points=100, n_reps=50)
    
    print("\nAnalysis complete! Results saved in 'sensitivity_results' directory")

if __name__ == '__main__':
    main() 