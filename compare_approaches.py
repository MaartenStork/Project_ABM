"""
Comparison of Final-Value vs Time-Series Sensitivity Analysis

This script demonstrates why analyzing only final timesteps can be misleading
and shows what we gain by analyzing the full temporal dynamics.
"""

import numpy as np
import matplotlib.pyplot as plt
import DynamicCoop as dc
import parameters

def run_single_simulation(param_name, param_value, n_timesteps=1000):
    """Run a single simulation and return full time series"""
    # Set parameter
    original_value = getattr(parameters, param_name)
    setattr(parameters, param_name, param_value)
    
    # Handle scale-dependent parameters
    if param_name == 'scale':
        parameters.rad_repulsion = 0.025 * param_value
        parameters.rad_orientation = 0.06 * param_value
        parameters.rad_attraction = 0.1 * param_value
        parameters.rad_repulsion_sqr = parameters.rad_repulsion ** 2
        parameters.rad_orientation_sqr = parameters.rad_orientation ** 2
        parameters.rad_attraction_sqr = parameters.rad_attraction ** 2
    
    # Initialize and run
    dc.initialize('default')
    
    fish_counts = []
    for t in range(n_timesteps):
        dc.update_one_unit_time()
        fish_counts.append(dc.total_fish_count[-1])
    
    # Restore parameter
    setattr(parameters, param_name, original_value)
    if param_name == 'scale':
        parameters.rad_repulsion = 0.025 * original_value
        parameters.rad_orientation = 0.06 * original_value
        parameters.rad_attraction = 0.1 * original_value
        parameters.rad_repulsion_sqr = parameters.rad_repulsion ** 2
        parameters.rad_orientation_sqr = parameters.rad_orientation ** 2
        parameters.rad_attraction_sqr = parameters.rad_attraction ** 2
    
    return np.array(fish_counts)

def demonstrate_temporal_dynamics():
    """Show how different parameters affect temporal dynamics"""
    
    # Test different values of key parameters
    test_cases = [
        ('scale', [1.0, 2.0, 4.0]),
        ('imitation_period', [5, 10, 20]),
        ('cooperation_increase', [0.1, 0.3, 0.6]),
        ('q', [0.5, 1.0, 1.5])
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (param_name, values) in enumerate(test_cases):
        ax = axes[i]
        
        for value in values:
            # Run simulation
            fish_trajectory = run_single_simulation(param_name, value)
            
            # Plot full trajectory
            ax.plot(fish_trajectory, label=f'{param_name}={value}', alpha=0.8)
            
            # Mark final value
            final_value = fish_trajectory[-1]
            ax.scatter(len(fish_trajectory)-1, final_value, s=100, zorder=5)
        
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Fish Count')
        ax.set_title(f'Effect of {param_name} on Fish Population Dynamics')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('temporal_dynamics_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_final_vs_timeseries():
    """Compare insights from final-value vs time-series analysis"""
    
    param_name = 'cooperation_increase'
    param_values = np.linspace(0.1, 0.8, 8)
    
    final_values = []
    mean_values = []
    cv_values = []
    crash_counts = []
    
    print("Analyzing cooperation_increase parameter...")
    print("Value\tFinal\tMean\tCV\tCrashes")
    print("-" * 40)
    
    for value in param_values:
        # Run multiple simulations for this parameter value
        trajectories = []
        for rep in range(5):  # 5 reps for demonstration
            trajectory = run_single_simulation(param_name, value)
            trajectories.append(trajectory)
        
        # Aggregate across repetitions
        all_trajectories = np.array(trajectories)
        
        # Final value analysis (current approach)
        final_vals = all_trajectories[:, -1]
        final_mean = np.mean(final_vals)
        
        # Time-series analysis (new approach)
        mean_over_time = np.mean(all_trajectories)
        cv_over_time = np.std(all_trajectories) / np.mean(all_trajectories)
        
        # Crash analysis
        crashes = 0
        for traj in trajectories:
            if np.min(traj) < 10:  # Population crashes below 10
                crashes += 1
        
        final_values.append(final_mean)
        mean_values.append(mean_over_time)
        cv_values.append(cv_over_time)
        crash_counts.append(crashes)
        
        print(f"{value:.2f}\t{final_mean:.1f}\t{mean_over_time:.1f}\t{cv_over_time:.2f}\t{crashes}")
    
    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Final values only (current approach)
    axes[0,0].plot(param_values, final_values, 'o-', color='red', linewidth=2)
    axes[0,0].set_title('Current Approach: Final Values Only')
    axes[0,0].set_xlabel('cooperation_increase')
    axes[0,0].set_ylabel('Final Fish Count')
    axes[0,0].grid(True, alpha=0.3)
    
    # Mean over time (time-series approach)
    axes[0,1].plot(param_values, mean_values, 'o-', color='blue', linewidth=2)
    axes[0,1].set_title('Time-Series Approach: Mean Over Time')
    axes[0,1].set_xlabel('cooperation_increase')
    axes[0,1].set_ylabel('Mean Fish Count')
    axes[0,1].grid(True, alpha=0.3)
    
    # Stability (coefficient of variation)
    axes[1,0].plot(param_values, cv_values, 'o-', color='green', linewidth=2)
    axes[1,0].set_title('Stability Analysis: Coefficient of Variation')
    axes[1,0].set_xlabel('cooperation_increase')
    axes[1,0].set_ylabel('CV (lower = more stable)')
    axes[1,0].grid(True, alpha=0.3)
    
    # Crash frequency
    axes[1,1].bar(param_values, crash_counts, alpha=0.7, color='orange')
    axes[1,1].set_title('Risk Analysis: Population Crashes')
    axes[1,1].set_xlabel('cooperation_increase')
    axes[1,1].set_ylabel('Number of Crashes (out of 5)')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('final_vs_timeseries_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return param_values, final_values, mean_values, cv_values, crash_counts

def main():
    """Main demonstration function"""
    print("="*60)
    print("COMPARING FINAL-VALUE vs TIME-SERIES SENSITIVITY ANALYSIS")
    print("="*60)
    
    print("\n1. Demonstrating temporal dynamics for key parameters...")
    demonstrate_temporal_dynamics()
    
    print("\n2. Comparing analytical insights...")
    results = analyze_final_vs_timeseries()
    
    print("\n" + "="*60)
    print("KEY INSIGHTS:")
    print("="*60)
    
    print("\n🔍 WHAT THE CURRENT APPROACH MISSES:")
    print("   • Transient dynamics and approach to equilibrium")
    print("   • Population stability and variability")
    print("   • Risk of population crashes")
    print("   • Recovery patterns after disturbances")
    print("   • Time-dependent parameter effects")
    
    print("\n📊 WHAT TIME-SERIES ANALYSIS REVEALS:")
    print("   • Mean population levels (sustainability)")
    print("   • Population stability (management risk)")
    print("   • Crash frequency (conservation concern)")
    print("   • Recovery capacity (resilience)")
    print("   • Temporal patterns (policy timing)")
    
    print("\n💡 SCIENTIFIC IMPLICATIONS:")
    print("   • Parameters may have different effects on:")
    print("     - Final outcomes vs. process dynamics")
    print("     - Mean levels vs. variability")
    print("     - Short-term vs. long-term behavior")
    print("   • Management decisions need both perspectives!")
    
    print("\n✅ RECOMMENDATION:")
    print("   Use time_series_sensitivity.py for comprehensive analysis")
    print("   that captures the full richness of system dynamics.")

if __name__ == '__main__':
    main() 