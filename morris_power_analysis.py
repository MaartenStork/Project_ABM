"""
Power Analysis for Morris Sensitivity Analysis (Fast Version)

This script determines the minimum number of trajectories needed for reliable Morris analysis results.
Uses a reduced set of trajectory sizes and fewer repeats for faster computation.
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import DynamicCoop as dc
import parameters
from SALib.sample import morris as morris_sample
from SALib.analyze import morris as morris_analyze
import pandas as pd
from scipy import stats
import seaborn as sns

def run_model(n_timesteps=300):
    """Same model runner as in onlymorris.py"""
    dc.initialize('reproduction_rate')
    if hasattr(parameters, 'reproduction_rate'):
        for agent in dc.agents:
            if agent.type == 'fish':
                setattr(agent, 'reproduction_rate', parameters.reproduction_rate)
    for _ in range(n_timesteps):
        dc.update_one_unit_time()
    return dc.total_fish_count[-1]

def get_param_setup():
    """Get parameter names and bounds as in onlymorris.py"""
    if not hasattr(parameters, 'reproduction_rate'):
        setattr(parameters, 'reproduction_rate', 0.3)
    
    names = [name for name, val in vars(parameters).items()
             if not name.startswith('_') and isinstance(val, (int, float))]
    exclude = {
        'n', 'Area', 'Length_Area', 'half_length_area',
        'rad_repulsion_sqr', 'rad_orientation_sqr', 'rad_attraction_sqr', 'r_sqr',
        'Time_MPA', 'Type_MPA', 'Dist_MPA', 'Frac_MPA', 'Half_Length',
        'Xa', 'Xb', 'Ya', 'Yb', 'Xm', 'Xn', 'Ym', 'Yn', 'Xp', 'Xq', 'Yp', 'Yq',
        'plot_update_freq',
        'reproduction_rate'
    }
    names = [p for p in names if p not in exclude]
    names.sort()
    bounds = [[getattr(parameters, p)*0.5, getattr(parameters, p)*1.5] for p in names]
    return names, bounds

def run_morris_analysis(param_names, bounds, num_trajectories):
    """Run Morris analysis with specified number of trajectories"""
    problem = {'num_vars': len(param_names), 'names': param_names, 'bounds': bounds}
    X = morris_sample.sample(problem, N=num_trajectories, num_levels=10)
    Y = np.zeros(X.shape[0])
    
    for i, xi in enumerate(X):
        for name, val in zip(param_names, xi):
            setattr(parameters, name, val)
        Y[i] = run_model()
    
    Si = morris_analyze.analyze(problem, X, Y, conf_level=0.95, print_to_console=False)
    return Si

def bootstrap_morris(X, Y, problem, n_bootstrap=1000):
    """Perform bootstrap analysis of Morris results"""
    n_samples = len(Y)
    mu_stars = np.zeros((n_bootstrap, len(problem['names'])))
    sigmas = np.zeros((n_bootstrap, len(problem['names'])))
    
    for i in range(n_bootstrap):
        # Bootstrap sampling
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        X_boot = X[indices]
        Y_boot = Y[indices]
        
        # Analyze bootstrapped sample
        Si = morris_analyze.analyze(problem, X_boot, Y_boot, conf_level=0.95, print_to_console=False)
        mu_stars[i] = np.abs(Si['mu'])
        sigmas[i] = Si['sigma']
    
    return mu_stars, sigmas

def analyze_convergence(trajectory_sizes=[20, 50, 100, 200], n_repeats=3):
    """Analyze convergence of Morris results with increasing trajectories"""
    param_names, bounds = get_param_setup()
    problem = {'num_vars': len(param_names), 'names': param_names, 'bounds': bounds}
    
    results = {
        'n_trajectories': [],
        'param_rankings_std': [],
        'mu_star_cv': [],
        'sigma_cv': [],
        'bootstrap_ci_width': []
    }
    
    for n_traj in tqdm(trajectory_sizes, desc="Analyzing trajectory sizes"):
        rankings_across_repeats = []
        mu_stars_across_repeats = []
        sigmas_across_repeats = []
        
        for _ in range(n_repeats):
            # Run Morris analysis
            X = morris_sample.sample(problem, N=n_traj, num_levels=4)  # Reduced from 10 to 4
            Y = np.zeros(X.shape[0])
            
            for i, xi in enumerate(X):
                for name, val in zip(param_names, xi):
                    setattr(parameters, name, val)
                Y[i] = run_model()
            
            Si = morris_analyze.analyze(problem, X, Y, conf_level=0.95, print_to_console=False)
            
            # Store results
            mu_stars = np.abs(Si['mu'])
            sigmas = Si['sigma']
            rankings = stats.rankdata(-mu_stars)
            
            rankings_across_repeats.append(rankings)
            mu_stars_across_repeats.append(mu_stars)
            sigmas_across_repeats.append(sigmas)
            
            # Bootstrap analysis with fewer samples
            mu_stars_boot, sigmas_boot = bootstrap_morris(X, Y, problem, n_bootstrap=50)  # Reduced from 100
        
        # Calculate metrics
        rankings_std = np.std(rankings_across_repeats, axis=0).mean()
        mu_star_cv = np.std(mu_stars_across_repeats, axis=0).mean() / np.mean(mu_stars_across_repeats, axis=0).mean()
        sigma_cv = np.std(sigmas_across_repeats, axis=0).mean() / np.mean(sigmas_across_repeats, axis=0).mean()
        
        # Calculate bootstrap CI width
        ci_width = np.percentile(mu_stars_boot, 97.5, axis=0) - np.percentile(mu_stars_boot, 2.5, axis=0)
        avg_ci_width = np.mean(ci_width)
        
        # Store results
        results['n_trajectories'].append(n_traj)
        results['param_rankings_std'].append(rankings_std)
        results['mu_star_cv'].append(mu_star_cv)
        results['sigma_cv'].append(sigma_cv)
        results['bootstrap_ci_width'].append(avg_ci_width)
    
    return results, param_names

def plot_convergence_analysis(results):
    """Plot convergence analysis results"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Morris Sensitivity Analysis Power Analysis', fontsize=14)
    
    # Plot 1: Parameter ranking stability
    axes[0,0].plot(results['n_trajectories'], results['param_rankings_std'], 'o-')
    axes[0,0].set_xlabel('Number of Trajectories')
    axes[0,0].set_ylabel('Mean Std. Dev. of Parameter Rankings')
    axes[0,0].set_title('Parameter Ranking Stability')
    axes[0,0].grid(True)
    
    # Plot 2: Coefficient of Variation for mu*
    axes[0,1].plot(results['n_trajectories'], results['mu_star_cv'], 'o-')
    axes[0,1].set_xlabel('Number of Trajectories')
    axes[0,1].set_ylabel('CV of mu*')
    axes[0,1].set_title('mu* Stability')
    axes[0,1].grid(True)
    
    # Plot 3: Coefficient of Variation for sigma
    axes[1,0].plot(results['n_trajectories'], results['sigma_cv'], 'o-')
    axes[1,0].set_xlabel('Number of Trajectories')
    axes[1,0].set_ylabel('CV of sigma')
    axes[1,0].set_title('sigma Stability')
    axes[1,0].grid(True)
    
    # Plot 4: Bootstrap CI width
    axes[1,1].plot(results['n_trajectories'], results['bootstrap_ci_width'], 'o-')
    axes[1,1].set_xlabel('Number of Trajectories')
    axes[1,1].set_ylabel('Mean Bootstrap CI Width')
    axes[1,1].set_title('Parameter Uncertainty')
    axes[1,1].grid(True)
    
    plt.tight_layout()
    plt.savefig('morris_power_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def determine_minimum_trajectories(results, threshold_cv=0.1, threshold_ranking=1.0):
    """Determine minimum number of trajectories needed based on convergence criteria"""
    n_traj = np.array(results['n_trajectories'])
    
    # Find where CV of mu* and sigma become stable
    stable_cv_mu = n_traj[results['mu_star_cv'] < threshold_cv]
    stable_cv_sigma = n_traj[results['sigma_cv'] < threshold_cv]
    
    # Find where parameter rankings become stable
    stable_rankings = n_traj[results['param_rankings_std'] < threshold_ranking]
    
    if len(stable_cv_mu) > 0 and len(stable_cv_sigma) > 0 and len(stable_rankings) > 0:
        min_n = max(stable_cv_mu[0], stable_cv_sigma[0], stable_rankings[0])
        print(f"\nRecommended minimum number of trajectories: {min_n}")
        print(f"Based on:")
        print(f"- mu* CV stability threshold: {threshold_cv}")
        print(f"- sigma CV stability threshold: {threshold_cv}")
        print(f"- Parameter ranking stability threshold: {threshold_ranking}")
    else:
        print("\nCould not determine minimum number of trajectories.")
        print("Consider increasing the maximum number of trajectories analyzed.")

def main():
    print("Starting Fast Morris Sensitivity Analysis Power Analysis...")
    print("Using reduced trajectory sizes and repeats for faster computation...")
    
    # Run convergence analysis
    results, param_names = analyze_convergence()
    
    # Plot results
    plot_convergence_analysis(results)
    
    # Determine minimum number of trajectories needed
    determine_minimum_trajectories(results)
    
    print("\nPower analysis complete. Results saved to 'morris_power_analysis.png'")

if __name__ == '__main__':
    main() 