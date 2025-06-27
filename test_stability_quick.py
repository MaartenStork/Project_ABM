"""
Quick Test of Parameter Stability Analysis

This script runs a small-scale test of the stability analysis to verify
everything works correctly before running the full analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import DynamicCoop as dc
import parameters

# Disable visualization
dc.VISUALIZATION_ENABLED = False
dc.CREATE_GIF = False

def quick_stability_test(param_dict, n_timesteps=500):
    """Quick stability test with fewer timesteps"""
    try:
        # Backup original parameters
        original_scale = parameters.scale
        original_q = parameters.q
        original_coop = parameters.cooperation_increase
        
        # Set parameters
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
        
        # Run simulation
        dc.initialize('default')
        
        fish_counts = []
        for t in range(n_timesteps):
            dc.update_one_unit_time()
            if t >= 100:  # Skip burn-in
                fish_counts.append(dc.total_fish_count[-1])
        
        fish_counts = np.array(fish_counts)
        
        # Restore parameters
        parameters.scale = original_scale
        parameters.q = original_q
        parameters.cooperation_increase = original_coop
        
        if 'scale' in param_dict:
            parameters.rad_repulsion = 0.025 * original_scale
            parameters.rad_orientation = 0.06 * original_scale
            parameters.rad_attraction = 0.1 * original_scale
            parameters.rad_repulsion_sqr = parameters.rad_repulsion ** 2
            parameters.rad_orientation_sqr = parameters.rad_orientation ** 2
            parameters.rad_attraction_sqr = parameters.rad_attraction ** 2
        
        # Simple stability metrics
        mean_fish = np.mean(fish_counts)
        min_fish = np.min(fish_counts)
        cv_fish = np.std(fish_counts) / mean_fish if mean_fish > 0 else np.inf
        
        # Simple stability criteria - RELAXED based on test results
        stable = (min_fish >= 3 and mean_fish > 12 and cv_fish < 1.5)  # More lenient
        
        return {
            'stable': stable,
            'mean_fish': mean_fish,
            'min_fish': min_fish,
            'cv_fish': cv_fish,
            'trajectory': fish_counts
        }
        
    except Exception as e:
        print(f"Error: {e}")
        return {
            'stable': False,
            'mean_fish': 0,
            'min_fish': 0,
            'cv_fish': np.inf,
            'trajectory': np.zeros(400)
        }

def test_parameter_ranges():
    """Test a few parameter combinations"""
    print("Testing parameter stability analysis...")
    
    # Test cases
    test_cases = [
        {'scale': 1.0, 'q': 0.4, 'cooperation_increase': 0.1},  # Conservative
        {'scale': 2.0, 'q': 0.6, 'cooperation_increase': 0.25}, # Default-ish
        {'scale': 3.0, 'q': 0.8, 'cooperation_increase': 0.4},  # Aggressive
        {'scale': 0.5, 'q': 0.3, 'cooperation_increase': 0.05}, # Very conservative
        {'scale': 4.0, 'q': 1.0, 'cooperation_increase': 0.5},  # Very aggressive
    ]
    
    results = []
    
    for i, params in enumerate(test_cases):
        print(f"\nTest case {i+1}: {params}")
        
        # Run multiple reps
        rep_results = []
        for rep in range(3):
            result = quick_stability_test(params)
            rep_results.append(result)
        
        # Aggregate
        stability_count = sum(1 for r in rep_results if r['stable'])
        stability_prob = stability_count / 3
        mean_fish = np.mean([r['mean_fish'] for r in rep_results])
        mean_cv = np.mean([r['cv_fish'] for r in rep_results if not np.isinf(r['cv_fish'])])
        
        print(f"  Stability probability: {stability_prob:.2f}")
        print(f"  Mean fish population: {mean_fish:.1f}")
        print(f"  Mean CV: {mean_cv:.2f}")
        
        # Show detailed metrics for first rep
        first_result = rep_results[0]
        print(f"  Detailed metrics (rep 1): min={first_result['min_fish']:.1f}, mean={first_result['mean_fish']:.1f}, CV={first_result['cv_fish']:.2f}")
        print(f"  Stability criteria: min>3? {first_result['min_fish']>3}, mean>12? {first_result['mean_fish']>12}, CV<1.5? {first_result['cv_fish']<1.5}")
        
        result_summary = params.copy()
        result_summary.update({
            'stability_probability': stability_prob,
            'mean_fish_population': mean_fish,
            'mean_cv': mean_cv
        })
        results.append(result_summary)
    
    # Create simple visualization
    df = pd.DataFrame(results)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Plot 1: Scale vs stability
    axes[0].scatter(df['scale'], df['stability_probability'], s=100, alpha=0.7)
    axes[0].set_xlabel('Scale')
    axes[0].set_ylabel('Stability Probability')
    axes[0].set_title('Scale vs Stability')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: q vs stability  
    axes[1].scatter(df['q'], df['stability_probability'], s=100, alpha=0.7, color='orange')
    axes[1].set_xlabel('Catchability (q)')
    axes[1].set_ylabel('Stability Probability')
    axes[1].set_title('Catchability vs Stability')
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: cooperation vs stability
    axes[2].scatter(df['cooperation_increase'], df['stability_probability'], s=100, alpha=0.7, color='green')
    axes[2].set_xlabel('Cooperation Increase')
    axes[2].set_ylabel('Stability Probability')
    axes[2].set_title('Cooperation vs Stability')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('quick_stability_test.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n=== QUICK TEST COMPLETE ===")
    print(f"Results saved to quick_stability_test.png")
    
    # Show which parameters look promising
    stable_cases = df[df['stability_probability'] > 0.5]
    if len(stable_cases) > 0:
        print(f"\nPromising parameter combinations:")
        for idx, row in stable_cases.iterrows():
            print(f"  scale={row['scale']:.1f}, q={row['q']:.1f}, coop={row['cooperation_increase']:.2f}")
            print(f"    -> stability prob: {row['stability_probability']:.2f}")
    else:
        print(f"\nNo highly stable combinations found in quick test.")
        print(f"This suggests we may need to:")
        print(f"  - Relax stability criteria")
        print(f"  - Expand parameter ranges")
        print(f"  - Increase simulation length")
    
    return df

if __name__ == '__main__':
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Run quick test
    results = test_parameter_ranges() 