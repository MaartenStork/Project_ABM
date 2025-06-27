"""
Realistic Stability Analysis for DynamicCoop Model

This version uses much more realistic stability criteria that account for
the inherent stochasticity and variability of agent-based models.

The goal is to find parameter ranges where the system maintains reasonable
dynamics rather than perfect stability.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from scipy import stats
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

# Import model components
import DynamicCoop as dc
import parameters

# Ensure no visualization during batch runs
dc.VISUALIZATION_ENABLED = False
dc.CREATE_GIF = False

def backup_and_set_parameters(param_dict):
    """Backup original parameters and set new ones"""
    backup = {}
    
    # Backup key parameters
    key_params = ['scale', 'imitation_period', 'cooperation_increase', 'q', 'trust_decrease']
    for param in key_params:
        if hasattr(parameters, param):
            backup[param] = getattr(parameters, param)
    
    # Backup scale-dependent parameters
    backup['rad_repulsion'] = parameters.rad_repulsion
    backup['rad_orientation'] = parameters.rad_orientation
    backup['rad_attraction'] = parameters.rad_attraction
    
    # Set new parameters
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
    
    # Ensure integer parameters
    if 'imitation_period' in param_dict:
        parameters.imitation_period = int(round(param_dict['imitation_period']))
    
    return backup

def restore_parameters(backup):
    """Restore parameters from backup"""
    for param_name, value in backup.items():
        if hasattr(parameters, param_name):
            setattr(parameters, param_name, value)

def run_realistic_stability_test(param_dict, n_timesteps=600, burn_in=100):
    """
    Run stability test with very realistic criteria for ABM
    """
    backup = backup_and_set_parameters(param_dict)
    
    try:
        # Initialize and run model
        dc.initialize('default')
        
        fish_counts = []
        for t in range(n_timesteps):
            dc.update_one_unit_time()
            if t >= burn_in:
                fish_counts.append(dc.total_fish_count[-1])
        
        fish_counts = np.array(fish_counts)
        
        # Basic metrics
        mean_fish = np.mean(fish_counts)
        min_fish = np.min(fish_counts)
        max_fish = np.max(fish_counts)
        cv_fish = np.std(fish_counts) / mean_fish if mean_fish > 0 else np.inf
        
        # VERY REALISTIC stability criteria for ABM
        # 1. Population doesn't completely crash
        no_extinction = min_fish >= 1  # At least 1 fish survives
        
        # 2. Population maintains reasonable average
        reasonable_population = mean_fish >= 8  # Average of 8+ fish
        
        # 3. Population doesn't explode unrealistically  
        not_explosive = max_fish <= 100  # Max 100 fish (reasonable upper bound)
        
        # 4. Variability is within reasonable bounds for ABM
        reasonable_variation = cv_fish <= 2.0  # CV up to 200% (very permissive)
        
        # 5. Population persists most of the time
        viable_count = np.sum(fish_counts >= 5)  # Count timesteps with 5+ fish
        persistence_rate = viable_count / len(fish_counts)
        persistent = persistence_rate >= 0.6  # 60% of time with viable population
        
        # Overall stability (much more lenient)
        stable = (no_extinction and reasonable_population and not_explosive and 
                 reasonable_variation and persistent)
        
        # Quality score (0-1) for ranking parameter sets
        quality_score = 0.0
        if no_extinction:
            quality_score += 0.2
        if reasonable_population:
            quality_score += 0.2
        if not_explosive:
            quality_score += 0.2
        if reasonable_variation:
            quality_score += 0.2
        quality_score += min(persistence_rate, 1.0) * 0.2  # Persistence bonus
        
        return {
            'stable': stable,
            'quality_score': quality_score,
            'mean_fish': mean_fish,
            'min_fish': min_fish,
            'max_fish': max_fish,
            'cv_fish': cv_fish,
            'persistence_rate': persistence_rate,
            'no_extinction': no_extinction,
            'reasonable_population': reasonable_population,
            'not_explosive': not_explosive,
            'reasonable_variation': reasonable_variation,
            'persistent': persistent,
            'trajectory': fish_counts
        }
        
    except Exception as e:
        print(f"Error in stability test: {e}")
        return {
            'stable': False,
            'quality_score': 0.0,
            'mean_fish': 0,
            'min_fish': 0,
            'max_fish': 0,
            'cv_fish': np.inf,
            'persistence_rate': 0,
            'no_extinction': False,
            'reasonable_population': False,
            'not_explosive': True,
            'reasonable_variation': False,
            'persistent': False,
            'trajectory': np.zeros(500)
        }
    finally:
        restore_parameters(backup)

def quick_ofat_analysis(param_ranges, n_samples=12, n_reps=3):
    """Quick OFAT with realistic criteria"""
    print("=== REALISTIC OFAT ANALYSIS ===")
    
    # Default parameter values
    defaults = {
        'scale': 2.0,
        'imitation_period': 8,
        'cooperation_increase': 0.25,
        'q': 0.6,
        'trust_decrease': 0.25
    }
    
    results = []
    
    for param_name, (min_val, max_val) in param_ranges.items():
        print(f"\nTesting parameter: {param_name}")
        
        # Generate parameter values
        if param_name == 'imitation_period':
            values = np.linspace(min_val, max_val, n_samples, dtype=int)
        else:
            values = np.linspace(min_val, max_val, n_samples)
        
        for value in tqdm(values, desc=f"OFAT {param_name}"):
            # Create parameter dict
            param_dict = defaults.copy()
            param_dict[param_name] = value
            
            # Run multiple repetitions
            rep_results = []
            for rep in range(n_reps):
                result = run_realistic_stability_test(param_dict)
                rep_results.append(result)
            
            # Aggregate results
            stability_count = sum(1 for r in rep_results if r['stable'])
            stability_prob = stability_count / n_reps
            
            mean_quality = np.mean([r['quality_score'] for r in rep_results])
            mean_fish = np.mean([r['mean_fish'] for r in rep_results])
            mean_persistence = np.mean([r['persistence_rate'] for r in rep_results])
            
            results.append({
                'parameter': param_name,
                'value': value,
                'stability_probability': stability_prob,
                'quality_score': mean_quality,
                'mean_fish_population': mean_fish,
                'mean_persistence': mean_persistence
            })
    
    return pd.DataFrame(results)

def analyze_results(df):
    """Analyze and report results"""
    print("\n=== ANALYSIS RESULTS ===")
    
    # Find stable ranges for each parameter
    stable_ranges = {}
    
    for param in df['parameter'].unique():
        param_data = df[df['parameter'] == param].sort_values('value')
        
        # Find points with good stability (>50%) or high quality (>0.7)
        good_points = param_data[
            (param_data['stability_probability'] > 0.5) | 
            (param_data['quality_score'] > 0.7)
        ]
        
        if len(good_points) > 0:
            min_good = good_points['value'].min()
            max_good = good_points['value'].max()
            
            # Find best point
            best_idx = good_points['quality_score'].idxmax()
            best_value = good_points.loc[best_idx, 'value']
            best_stability = good_points.loc[best_idx, 'stability_probability']
            best_quality = good_points.loc[best_idx, 'quality_score']
            
            stable_ranges[param] = {
                'range': (min_good, max_good),
                'best_value': best_value,
                'best_stability': best_stability,
                'best_quality': best_quality,
                'n_good_points': len(good_points)
            }
            
            print(f"\n{param}:")
            print(f"  Promising range: {min_good:.3f} - {max_good:.3f}")
            print(f"  Best value: {best_value:.3f}")
            print(f"  Best stability prob: {best_stability:.2f}")
            print(f"  Best quality score: {best_quality:.2f}")
        else:
            print(f"\n{param}: No clearly stable range found")
            stable_ranges[param] = None
    
    return stable_ranges

def create_plots(df):
    """Create visualization plots"""
    print("\n=== CREATING PLOTS ===")
    
    # Create plots directory
    os.makedirs('realistic_stability_plots', exist_ok=True)
    
    # Plot 1: Stability probability by parameter
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    params = df['parameter'].unique()
    for i, param in enumerate(params):
        if i < len(axes):
            param_data = df[df['parameter'] == param]
            
            ax = axes[i]
            # Plot stability probability
            ax.plot(param_data['value'], param_data['stability_probability'], 
                   'o-', linewidth=2, markersize=6, color='blue', label='Stability Prob')
            
            # Plot quality score
            ax2 = ax.twinx()
            ax2.plot(param_data['value'], param_data['quality_score'], 
                    's-', linewidth=2, markersize=4, color='red', alpha=0.7, label='Quality Score')
            
            ax.set_xlabel(param)
            ax.set_ylabel('Stability Probability', color='blue')
            ax2.set_ylabel('Quality Score', color='red')
            ax.set_title(f'{param}')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-0.05, 1.05)
            ax2.set_ylim(-0.05, 1.05)
            
            # Highlight good regions
            good_data = param_data[
                (param_data['stability_probability'] > 0.5) | 
                (param_data['quality_score'] > 0.7)
            ]
            if len(good_data) > 0:
                ax.fill_between(good_data['value'], 0, 1, alpha=0.2, color='green', label='Good Region')
    
    # Hide unused subplots
    for i in range(len(params), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('realistic_stability_plots/parameter_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main execution function"""
    print("="*60)
    print("REALISTIC PARAMETER STABILITY ANALYSIS")
    print("="*60)
    
    # Define parameter ranges
    param_ranges = {
        'scale': (0.5, 4.0),
        'imitation_period': (1, 15),
        'cooperation_increase': (0.05, 0.5),
        'q': (0.2, 1.0),
        'trust_decrease': (0.05, 0.5)
    }
    
    print("Using much more realistic stability criteria:")
    print("  - Population doesn't go extinct (min ≥ 1)")
    print("  - Reasonable average population (mean ≥ 8)")
    print("  - Not explosive (max ≤ 100)")
    print("  - Reasonable variation (CV ≤ 2.0)")
    print("  - Persistent (viable 60% of time)")
    
    # Set seed
    np.random.seed(42)
    
    # Run analysis
    df = quick_ofat_analysis(param_ranges, n_samples=12, n_reps=3)
    
    # Analyze results
    stable_ranges = analyze_results(df)
    
    # Create plots
    create_plots(df)
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    df.to_csv(f'realistic_stability_plots/results_{timestamp}.csv', index=False)
    
    # Save summary
    with open(f'realistic_stability_plots/summary_{timestamp}.txt', 'w') as f:
        f.write("REALISTIC STABILITY ANALYSIS SUMMARY\n")
        f.write("="*40 + "\n\n")
        
        for param, info in stable_ranges.items():
            f.write(f"{param}:\n")
            if info is not None:
                f.write(f"  Promising range: {info['range'][0]:.3f} - {info['range'][1]:.3f}\n")
                f.write(f"  Best value: {info['best_value']:.3f}\n")
                f.write(f"  Best stability: {info['best_stability']:.2f}\n")
                f.write(f"  Best quality: {info['best_quality']:.2f}\n")
            else:
                f.write("  No clearly stable range found\n")
            f.write("\n")
    
    print(f"\n=== ANALYSIS COMPLETE ===")
    print(f"Results saved to realistic_stability_plots/")
    
    return df, stable_ranges

if __name__ == '__main__':
    results = main() 