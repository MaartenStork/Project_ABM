"""
Balanced Stability Analysis for DynamicCoop Model

This version uses balanced stability criteria that can actually discriminate
between parameter values - not too strict (everything fails) and not too 
lenient (everything passes).

Goal: Find parameter ranges with meaningful stability differences.
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

def run_balanced_stability_test(param_dict, n_timesteps=800, burn_in=200):
    """
    Run stability test with balanced criteria that can discriminate
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
        
        # More balanced stability criteria
        # 1. No severe crashes (more restrictive than before)
        no_severe_crash = min_fish >= 5  # At least 5 fish always
        
        # 2. Good sustainable population (higher threshold)
        good_population = mean_fish >= 15  # Average of 15+ fish
        
        # 3. Population doesn't explode
        not_explosive = max_fish <= 80  # Max 80 fish
        
        # 4. Reasonable variability (more restrictive)
        reasonable_variation = cv_fish <= 0.8  # CV up to 80%
        
        # 5. Strong persistence (higher threshold)
        viable_count = np.sum(fish_counts >= 10)  # Count timesteps with 10+ fish
        persistence_rate = viable_count / len(fish_counts)
        strong_persistence = persistence_rate >= 0.8  # 80% of time viable
        
        # 6. Population trend analysis (new criterion)
        # Check if population is stable (not trending up or down strongly)
        final_third = fish_counts[-len(fish_counts)//3:]  # Last third of simulation
        x = np.arange(len(final_third))
        if len(final_third) > 5:
            slope, _, r_value, _, _ = stats.linregress(x, final_third)
            stable_trend = abs(slope) < 0.1 and abs(r_value) < 0.5  # Weak trend
        else:
            stable_trend = True
        
        # Overall stability (more discriminating)
        stable = (no_severe_crash and good_population and not_explosive and 
                 reasonable_variation and strong_persistence and stable_trend)
        
        # Quality score with more nuanced scoring
        quality_score = 0.0
        
        # Population quality (0-0.4)
        if min_fish >= 8:
            quality_score += 0.1
        if mean_fish >= 20:
            quality_score += 0.15
        elif mean_fish >= 15:
            quality_score += 0.1
        elif mean_fish >= 10:
            quality_score += 0.05
        
        if not_explosive:
            quality_score += 0.1
        
        # Variability quality (0-0.3)
        if cv_fish <= 0.5:
            quality_score += 0.15
        elif cv_fish <= 0.8:
            quality_score += 0.1
        elif cv_fish <= 1.2:
            quality_score += 0.05
        
        # Add trend stability bonus
        if stable_trend:
            quality_score += 0.1
        
        # Persistence quality (0-0.3)
        quality_score += min(persistence_rate, 1.0) * 0.3
        
        return {
            'stable': stable,
            'quality_score': quality_score,
            'mean_fish': mean_fish,
            'min_fish': min_fish,
            'max_fish': max_fish,
            'cv_fish': cv_fish,
            'persistence_rate': persistence_rate,
            'trend_slope': slope if len(final_third) > 5 else 0,
            'no_severe_crash': no_severe_crash,
            'good_population': good_population,
            'not_explosive': not_explosive,
            'reasonable_variation': reasonable_variation,
            'strong_persistence': strong_persistence,
            'stable_trend': stable_trend,
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
            'trend_slope': 0,
            'no_severe_crash': False,
            'good_population': False,
            'not_explosive': True,
            'reasonable_variation': False,
            'strong_persistence': False,
            'stable_trend': False,
            'trajectory': np.zeros(600)
        }
    finally:
        restore_parameters(backup)

def balanced_ofat_analysis(param_ranges, n_samples=15, n_reps=5):
    """OFAT with balanced criteria"""
    print("=== BALANCED OFAT ANALYSIS ===")
    print("Using more discriminating stability criteria:")
    print("  - No severe crashes (min ≥ 5)")
    print("  - Good population (mean ≥ 15)")
    print("  - Not explosive (max ≤ 80)")
    print("  - Reasonable variation (CV ≤ 0.8)")
    print("  - Strong persistence (viable 80% of time)")
    print("  - Stable trend (no strong up/down trends)")
    
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
                result = run_balanced_stability_test(param_dict)
                rep_results.append(result)
            
            # Aggregate results
            stability_count = sum(1 for r in rep_results if r['stable'])
            stability_prob = stability_count / n_reps
            
            mean_quality = np.mean([r['quality_score'] for r in rep_results])
            mean_fish = np.mean([r['mean_fish'] for r in rep_results])
            mean_cv = np.mean([r['cv_fish'] for r in rep_results if not np.isinf(r['cv_fish'])])
            mean_persistence = np.mean([r['persistence_rate'] for r in rep_results])
            
            # Detailed breakdown
            crash_rate = 1 - np.mean([r['no_severe_crash'] for r in rep_results])
            
            results.append({
                'parameter': param_name,
                'value': value,
                'stability_probability': stability_prob,
                'quality_score': mean_quality,
                'mean_fish_population': mean_fish,
                'mean_cv': mean_cv,
                'mean_persistence': mean_persistence,
                'crash_rate': crash_rate
            })
    
    return pd.DataFrame(results)

def analyze_balanced_results(df):
    """Analyze results with more nuanced interpretation"""
    print("\n=== BALANCED ANALYSIS RESULTS ===")
    
    # Overall statistics
    overall_stability = df['stability_probability'].mean()
    high_quality_rate = np.mean(df['quality_score'] > 0.7)
    
    print(f"Overall stability rate: {overall_stability:.2f}")
    print(f"High quality rate (score > 0.7): {high_quality_rate:.2f}")
    
    stable_ranges = {}
    
    for param in df['parameter'].unique():
        param_data = df[df['parameter'] == param].sort_values('value')
        
        # Find different quality tiers
        excellent_points = param_data[param_data['quality_score'] > 0.8]
        good_points = param_data[param_data['quality_score'] > 0.6]
        stable_points = param_data[param_data['stability_probability'] > 0.6]
        
        print(f"\n{param}:")
        print(f"  Excellent points (quality > 0.8): {len(excellent_points)}")
        print(f"  Good points (quality > 0.6): {len(good_points)}")
        print(f"  Stable points (prob > 0.6): {len(stable_points)}")
        
        # Report ranges for different quality levels
        if len(excellent_points) > 0:
            min_exc = excellent_points['value'].min()
            max_exc = excellent_points['value'].max()
            best_exc = excellent_points.loc[excellent_points['quality_score'].idxmax()]
            print(f"  EXCELLENT range: {min_exc:.3f} - {max_exc:.3f}")
            print(f"    Best: {best_exc['value']:.3f} (quality: {best_exc['quality_score']:.2f})")
            
            stable_ranges[param] = {
                'tier': 'excellent',
                'range': (min_exc, max_exc),
                'best_value': best_exc['value'],
                'best_quality': best_exc['quality_score'],
                'best_stability': best_exc['stability_probability']
            }
            
        elif len(good_points) > 0:
            min_good = good_points['value'].min()
            max_good = good_points['value'].max()
            best_good = good_points.loc[good_points['quality_score'].idxmax()]
            print(f"  GOOD range: {min_good:.3f} - {max_good:.3f}")
            print(f"    Best: {best_good['value']:.3f} (quality: {best_good['quality_score']:.2f})")
            
            stable_ranges[param] = {
                'tier': 'good',
                'range': (min_good, max_good),
                'best_value': best_good['value'],
                'best_quality': best_good['quality_score'],
                'best_stability': best_good['stability_probability']
            }
            
        elif len(stable_points) > 0:
            min_stable = stable_points['value'].min()
            max_stable = stable_points['value'].max()
            best_stable = stable_points.loc[stable_points['stability_probability'].idxmax()]
            print(f"  STABLE range: {min_stable:.3f} - {max_stable:.3f}")
            print(f"    Best: {best_stable['value']:.3f} (prob: {best_stable['stability_probability']:.2f})")
            
            stable_ranges[param] = {
                'tier': 'stable',
                'range': (min_stable, max_stable),
                'best_value': best_stable['value'],
                'best_quality': best_stable['quality_score'],
                'best_stability': best_stable['stability_probability']
            }
        else:
            print(f"  No clearly stable range found")
            stable_ranges[param] = None
    
    return stable_ranges

def create_balanced_plots(df):
    """Create plots showing the discrimination between parameter values"""
    print("\n=== CREATING BALANCED PLOTS ===")
    
    # Create plots directory
    os.makedirs('balanced_stability_plots', exist_ok=True)
    
    # Plot with better discrimination
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    params = df['parameter'].unique()
    for i, param in enumerate(params):
        if i < len(axes):
            param_data = df[df['parameter'] == param]
            
            ax = axes[i]
            
            # Plot stability probability with more visible variation
            ax.plot(param_data['value'], param_data['stability_probability'], 
                   'o-', linewidth=2, markersize=8, color='blue', label='Stability Prob')
            
            # Plot quality score
            ax2 = ax.twinx()
            ax2.plot(param_data['value'], param_data['quality_score'], 
                    's-', linewidth=2, markersize=6, color='red', alpha=0.8, label='Quality Score')
            
            ax.set_xlabel(param)
            ax.set_ylabel('Stability Probability', color='blue')
            ax2.set_ylabel('Quality Score', color='red')
            ax.set_title(f'{param}')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-0.05, 1.05)
            ax2.set_ylim(-0.05, 1.05)
            
            # Color-code regions by quality
            excellent = param_data[param_data['quality_score'] > 0.8]
            good = param_data[(param_data['quality_score'] > 0.6) & (param_data['quality_score'] <= 0.8)]
            
            if len(excellent) > 0:
                ax.fill_between(excellent['value'], 0, 1, alpha=0.3, color='darkgreen', label='Excellent')
            if len(good) > 0:
                ax.fill_between(good['value'], 0, 1, alpha=0.2, color='lightgreen', label='Good')
    
    # Hide unused subplots
    for i in range(len(params), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('balanced_stability_plots/balanced_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main execution function"""
    print("="*60)
    print("BALANCED PARAMETER STABILITY ANALYSIS")
    print("="*60)
    
    # Define parameter ranges
    param_ranges = {
        'scale': (0.5, 4.0),
        'imitation_period': (1, 15),
        'cooperation_increase': (0.05, 0.5),
        'q': (0.2, 1.0),
        'trust_decrease': (0.05, 0.5)
    }
    
    # Set seed
    np.random.seed(42)
    
    # Run analysis
    df = balanced_ofat_analysis(param_ranges, n_samples=15, n_reps=5)
    
    # Analyze results
    stable_ranges = analyze_balanced_results(df)
    
    # Create plots
    create_balanced_plots(df)
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    df.to_csv(f'balanced_stability_plots/results_{timestamp}.csv', index=False)
    
    # Save summary
    with open(f'balanced_stability_plots/summary_{timestamp}.txt', 'w') as f:
        f.write("BALANCED STABILITY ANALYSIS SUMMARY\n")
        f.write("="*40 + "\n\n")
        
        for param, info in stable_ranges.items():
            f.write(f"{param}:\n")
            if info is not None:
                f.write(f"  Tier: {info['tier']}\n")
                f.write(f"  Range: {info['range'][0]:.3f} - {info['range'][1]:.3f}\n")
                f.write(f"  Best value: {info['best_value']:.3f}\n")
                f.write(f"  Best quality: {info['best_quality']:.2f}\n")
                f.write(f"  Best stability: {info['best_stability']:.2f}\n")
            else:
                f.write("  No stable range found\n")
            f.write("\n")
    
    print(f"\n=== ANALYSIS COMPLETE ===")
    print(f"Results saved to balanced_stability_plots/")
    
    return df, stable_ranges

if __name__ == '__main__':
    results = main() 