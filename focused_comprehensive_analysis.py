"""
Focused Comprehensive Stability Analysis for DynamicCoop Model

This analysis focuses on fish population stability with statistical rigor:
- Wide parameter ranges to find failure boundaries  
- Large sample sizes for statistical significance
- Focus on measurable fish population metrics
- Proper statistical testing and confidence intervals
- Clear identification of parameter thresholds

Goal: Find statistically significant parameter boundaries for fish population stability.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from scipy import stats
from scipy.stats import ttest_ind, mannwhitneyu
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

class FocusedStabilityAnalyzer:
    """Focused fish population stability analysis with statistical rigor"""
    
    def __init__(self, output_dir='focused_stability'):
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        os.makedirs(output_dir, exist_ok=True)
        
        # Store original parameters
        self.original_params = self._backup_parameters()
        
        print(f"FocusedStabilityAnalyzer initialized")
        print(f"Output directory: {output_dir}")
    
    def _backup_parameters(self):
        """Backup all relevant parameters"""
        backup = {}
        key_params = ['scale', 'imitation_period', 'cooperation_increase', 'q', 'trust_decrease']
        
        for param in key_params:
            if hasattr(parameters, param):
                backup[param] = getattr(parameters, param)
        
        # Backup scale-dependent parameters
        backup['rad_repulsion'] = parameters.rad_repulsion
        backup['rad_orientation'] = parameters.rad_orientation
        backup['rad_attraction'] = parameters.rad_attraction
        backup['rad_repulsion_sqr'] = parameters.rad_repulsion_sqr
        backup['rad_orientation_sqr'] = parameters.rad_orientation_sqr
        backup['rad_attraction_sqr'] = parameters.rad_attraction_sqr
        
        return backup
    
    def _restore_parameters(self):
        """Restore original parameters"""
        for param_name, value in self.original_params.items():
            if hasattr(parameters, param_name):
                setattr(parameters, param_name, value)
    
    def _set_parameters(self, param_dict):
        """Set model parameters with proper handling of dependencies"""
        # Set main parameters
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
    
    def run_single_simulation(self, param_dict, n_timesteps=1200, burn_in=300):
        """
        Run single simulation focusing on fish population dynamics
        """
        try:
            self._set_parameters(param_dict)
            
            # Initialize model
            dc.initialize('default')
            
            # Run simulation and collect fish population data
            fish_counts = []
            
            for t in range(n_timesteps):
                dc.update_one_unit_time()
                
                if t >= burn_in:
                    fish_counts.append(dc.total_fish_count[-1])
            
            fish_counts = np.array(fish_counts)
            
            # Calculate comprehensive fish population metrics
            metrics = self._calculate_fish_stability_metrics(fish_counts)
            
            return metrics
            
        except Exception as e:
            print(f"Simulation error: {e}")
            return self._get_failed_metrics()
        
        finally:
            self._restore_parameters()
    
    def _calculate_fish_stability_metrics(self, fish_counts):
        """Calculate focused fish population stability metrics"""
        
        n_timesteps = len(fish_counts)
        
        # Basic population statistics
        mean_fish = np.mean(fish_counts)
        median_fish = np.median(fish_counts)
        std_fish = np.std(fish_counts)
        min_fish = np.min(fish_counts)
        max_fish = np.max(fish_counts)
        cv_fish = std_fish / mean_fish if mean_fish > 0 else np.inf
        
        # Population health indicators
        extinction = min_fish < 1
        severe_crash = min_fish < 5
        population_viable = mean_fish >= 10
        population_healthy = mean_fish >= 20
        population_abundant = mean_fish >= 30
        
        # Variability assessment
        low_variability = cv_fish <= 0.4
        moderate_variability = cv_fish <= 0.8
        high_variability = cv_fish > 1.0
        
        # Persistence analysis
        viable_threshold = 10
        healthy_threshold = 20
        
        viable_timesteps = np.sum(fish_counts >= viable_threshold)
        healthy_timesteps = np.sum(fish_counts >= healthy_threshold)
        
        persistence_viable = viable_timesteps / n_timesteps
        persistence_healthy = healthy_timesteps / n_timesteps
        
        # Trend analysis (multiple windows)
        trends = {}
        for window_name, window_fraction in [('final_quarter', 0.25), ('final_half', 0.5)]:
            window_size = int(n_timesteps * window_fraction)
            window_data = fish_counts[-window_size:]
            
            if len(window_data) > 5:
                x = np.arange(len(window_data))
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, window_data)
                
                trends[window_name] = {
                    'slope': slope,
                    'r_value': r_value,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                    'direction': 'increasing' if slope > 0.1 else 'decreasing' if slope < -0.1 else 'stable'
                }
            else:
                trends[window_name] = {
                    'slope': 0, 'r_value': 0, 'p_value': 1, 'significant': False, 'direction': 'stable'
                }
        
        # Recovery analysis
        crash_threshold = mean_fish * 0.6  # 60% of mean
        low_points = fish_counts < crash_threshold
        
        if np.any(low_points):
            low_indices = np.where(low_points)[0]
            recoveries = 0
            
            for idx in low_indices:
                if idx < len(fish_counts) - 30:  # Need 30 steps to assess recovery
                    recovery_window = fish_counts[idx+1:idx+31]
                    recovery_mean = np.mean(recovery_window)
                    if recovery_mean > fish_counts[idx] * 1.25:  # 25% recovery
                        recoveries += 1
            
            recovery_rate = recoveries / len(low_indices) if len(low_indices) > 0 else 0
        else:
            recovery_rate = 1.0  # No crashes to recover from
        
        # Stability scoring (0-10 scale)
        stability_score = 0
        
        # Population level (0-3 points)
        if population_abundant:
            stability_score += 3
        elif population_healthy:
            stability_score += 2
        elif population_viable:
            stability_score += 1
        
        # Crash resistance (0-2 points)
        if not extinction:
            stability_score += 1
        if not severe_crash:
            stability_score += 1
        
        # Variability (0-2 points)
        if low_variability:
            stability_score += 2
        elif moderate_variability:
            stability_score += 1
        
        # Persistence (0-2 points)
        if persistence_healthy >= 0.8:
            stability_score += 2
        elif persistence_viable >= 0.8:
            stability_score += 1
        
        # Recovery (0-1 point)
        if recovery_rate >= 0.5:
            stability_score += 1
        
        # Normalize to 0-1 scale
        stability_score_normalized = stability_score / 10.0
        
        return {
            # Basic statistics
            'mean_fish': mean_fish,
            'median_fish': median_fish,
            'std_fish': std_fish,
            'min_fish': min_fish,
            'max_fish': max_fish,
            'cv_fish': cv_fish,
            
            # Health indicators
            'extinction': extinction,
            'severe_crash': severe_crash,
            'population_viable': population_viable,
            'population_healthy': population_healthy,
            'population_abundant': population_abundant,
            
            # Variability
            'low_variability': low_variability,
            'moderate_variability': moderate_variability,
            'high_variability': high_variability,
            
            # Persistence
            'persistence_viable': persistence_viable,
            'persistence_healthy': persistence_healthy,
            
            # Trends
            'trend_final_quarter_slope': trends['final_quarter']['slope'],
            'trend_final_quarter_significant': trends['final_quarter']['significant'],
            'trend_final_half_slope': trends['final_half']['slope'],
            'trend_final_half_significant': trends['final_half']['significant'],
            
            # Recovery
            'recovery_rate': recovery_rate,
            
            # Overall scores
            'stability_score': stability_score,
            'stability_score_normalized': stability_score_normalized,
            
            # Raw data
            'fish_trajectory': fish_counts
        }
    
    def _get_failed_metrics(self):
        """Return metrics for failed simulations"""
        return {
            'mean_fish': 0, 'median_fish': 0, 'std_fish': 0, 'min_fish': 0, 'max_fish': 0, 'cv_fish': np.inf,
            'extinction': True, 'severe_crash': True, 'population_viable': False, 'population_healthy': False,
            'population_abundant': False, 'low_variability': False, 'moderate_variability': False,
            'high_variability': True, 'persistence_viable': 0, 'persistence_healthy': 0,
            'trend_final_quarter_slope': 0, 'trend_final_quarter_significant': False,
            'trend_final_half_slope': 0, 'trend_final_half_significant': False,
            'recovery_rate': 0, 'stability_score': 0, 'stability_score_normalized': 0,
            'fish_trajectory': np.zeros(900)
        }
    
    def comprehensive_parameter_sweep(self, param_ranges, n_samples=30, n_reps=15):
        """
        Comprehensive parameter sweep with high statistical power
        """
        print("=== FOCUSED COMPREHENSIVE PARAMETER SWEEP ===")
        print(f"Samples per parameter: {n_samples}")
        print(f"Repetitions per sample: {n_reps}")
        print(f"Total simulations: {len(param_ranges) * n_samples * n_reps}")
        print(f"Estimated time: {len(param_ranges) * n_samples * n_reps * 4 / 60:.1f} minutes")
        
        # Default parameter values (baseline)
        defaults = {
            'scale': 2.0,
            'imitation_period': 8,
            'cooperation_increase': 0.25,
            'q': 0.6,
            'trust_decrease': 0.25
        }
        
        all_results = []
        
        for param_name, (min_val, max_val) in param_ranges.items():
            print(f"\n=== Testing parameter: {param_name} ===")
            print(f"Range: {min_val} - {max_val}")
            
            # Generate parameter values
            if param_name == 'imitation_period':
                values = np.linspace(min_val, max_val, n_samples, dtype=int)
            else:
                values = np.linspace(min_val, max_val, n_samples)
            
            for value in tqdm(values, desc=f"Testing {param_name}"):
                # Create parameter dict
                param_dict = defaults.copy()
                param_dict[param_name] = value
                
                # Run multiple repetitions
                rep_results = []
                for rep in range(n_reps):
                    result = self.run_single_simulation(param_dict)
                    rep_results.append(result)
                
                # Aggregate statistics
                aggregated = self._aggregate_repetitions(rep_results, param_name, value)
                all_results.append(aggregated)
        
        return pd.DataFrame(all_results)
    
    def _aggregate_repetitions(self, rep_results, param_name, param_value):
        """Aggregate results with comprehensive statistics"""
        
        n_reps = len(rep_results)
        
        # Extract key metrics
        stability_scores = [r['stability_score_normalized'] for r in rep_results]
        mean_fish_values = [r['mean_fish'] for r in rep_results]
        cv_fish_values = [r['cv_fish'] for r in rep_results if not np.isinf(r['cv_fish'])]
        persistence_viable = [r['persistence_viable'] for r in rep_results]
        persistence_healthy = [r['persistence_healthy'] for r in rep_results]
        
        # Calculate statistics with confidence intervals
        def calc_stats(values, name):
            if len(values) == 0:
                return {f'{name}_mean': 0, f'{name}_std': 0, f'{name}_se': 0, 
                       f'{name}_ci_lower': 0, f'{name}_ci_upper': 0}
            
            mean_val = np.mean(values)
            std_val = np.std(values, ddof=1) if len(values) > 1 else 0
            se_val = std_val / np.sqrt(len(values)) if len(values) > 1 else 0
            ci_lower = mean_val - 1.96 * se_val
            ci_upper = mean_val + 1.96 * se_val
            
            return {
                f'{name}_mean': mean_val,
                f'{name}_std': std_val,
                f'{name}_se': se_val,
                f'{name}_ci_lower': ci_lower,
                f'{name}_ci_upper': ci_upper
            }
        
        # Calculate statistics for each metric
        stability_stats = calc_stats(stability_scores, 'stability')
        fish_stats = calc_stats(mean_fish_values, 'fish')
        cv_stats = calc_stats(cv_fish_values, 'cv') if cv_fish_values else calc_stats([np.inf], 'cv')
        persist_viable_stats = calc_stats(persistence_viable, 'persist_viable')
        persist_healthy_stats = calc_stats(persistence_healthy, 'persist_healthy')
        
        # Binary outcome statistics
        n_viable = sum(1 for r in rep_results if r['population_viable'])
        n_healthy = sum(1 for r in rep_results if r['population_healthy'])
        n_no_crash = sum(1 for r in rep_results if not r['severe_crash'])
        n_low_var = sum(1 for r in rep_results if r['low_variability'])
        
        # Proportion confidence intervals
        def proportion_ci(count, total):
            if total == 0:
                return 0, 0, 0
            p = count / total
            se = np.sqrt(p * (1 - p) / total) if total > 0 else 0
            ci_lower = max(0, p - 1.96 * se)
            ci_upper = min(1, p + 1.96 * se)
            return p, ci_lower, ci_upper
        
        viable_rate, viable_ci_lower, viable_ci_upper = proportion_ci(n_viable, n_reps)
        healthy_rate, healthy_ci_lower, healthy_ci_upper = proportion_ci(n_healthy, n_reps)
        no_crash_rate, no_crash_ci_lower, no_crash_ci_upper = proportion_ci(n_no_crash, n_reps)
        low_var_rate, low_var_ci_lower, low_var_ci_upper = proportion_ci(n_low_var, n_reps)
        
        # Combine all statistics
        result = {
            'parameter': param_name,
            'value': param_value,
            'n_reps': n_reps,
            
            # Rates with confidence intervals
            'viable_rate': viable_rate,
            'viable_ci_lower': viable_ci_lower,
            'viable_ci_upper': viable_ci_upper,
            'healthy_rate': healthy_rate,
            'healthy_ci_lower': healthy_ci_lower,
            'healthy_ci_upper': healthy_ci_upper,
            'no_crash_rate': no_crash_rate,
            'no_crash_ci_lower': no_crash_ci_lower,
            'no_crash_ci_upper': no_crash_ci_upper,
            'low_var_rate': low_var_rate,
            'low_var_ci_lower': low_var_ci_lower,
            'low_var_ci_upper': low_var_ci_upper
        }
        
        # Add all calculated statistics
        result.update(stability_stats)
        result.update(fish_stats)
        result.update(cv_stats)
        result.update(persist_viable_stats)
        result.update(persist_healthy_stats)
        
        return result
    
    def find_parameter_boundaries(self, df, metric='stability_mean', threshold=0.5):
        """Find statistically significant parameter boundaries"""
        
        print(f"\n=== FINDING PARAMETER BOUNDARIES ===")
        print(f"Boundary metric: {metric}")
        print(f"Boundary threshold: {threshold}")
        
        boundaries = {}
        
        for param in df['parameter'].unique():
            param_data = df[df['parameter'] == param].sort_values('value')
            
            # Find working vs failing regions
            working = param_data[param_data[metric] >= threshold]
            failing = param_data[param_data[metric] < threshold]
            
            if len(working) > 0 and len(failing) > 0:
                # Statistical test for significant difference
                working_values = working[metric].values
                failing_values = failing[metric].values
                
                # Use Mann-Whitney U test
                try:
                    statistic, p_value = mannwhitneyu(working_values, failing_values, alternative='greater')
                    significant = p_value < 0.01  # Stricter significance level
                except:
                    significant = False
                    p_value = 1.0
                
                # Effect size (Cohen's d approximation)
                pooled_std = np.sqrt((np.var(working_values) + np.var(failing_values)) / 2)
                effect_size = (np.mean(working_values) - np.mean(failing_values)) / pooled_std if pooled_std > 0 else 0
                
                boundaries[param] = {
                    'working_range': (working['value'].min(), working['value'].max()),
                    'failing_range': (failing['value'].min(), failing['value'].max()),
                    'boundary_lower': working['value'].min(),
                    'boundary_upper': working['value'].max(),
                    'optimal_value': working.loc[working[metric].idxmax(), 'value'],
                    'optimal_score': working[metric].max(),
                    'n_working': len(working),
                    'n_failing': len(failing),
                    'p_value': p_value,
                    'significant': significant,
                    'effect_size': effect_size,
                    'working_mean': np.mean(working_values),
                    'failing_mean': np.mean(failing_values)
                }
                
                print(f"\n{param.upper()}:")
                print(f"  Working range: {working['value'].min():.3f} - {working['value'].max():.3f}")
                print(f"  Optimal value: {boundaries[param]['optimal_value']:.3f} (score: {boundaries[param]['optimal_score']:.3f})")
                print(f"  Statistical test: p = {p_value:.6f} ({'SIGNIFICANT' if significant else 'not significant'})")
                print(f"  Effect size: {effect_size:.3f} ({'large' if abs(effect_size) > 0.8 else 'medium' if abs(effect_size) > 0.5 else 'small'})")
            
            else:
                boundaries[param] = None
                if len(working) == 0:
                    print(f"\n{param.upper()}: NO WORKING VALUES FOUND (all below threshold)")
                else:
                    print(f"\n{param.upper()}: ALL VALUES WORKING (all above threshold)")
        
        return boundaries
    
    def create_publication_quality_plots(self, df, boundaries):
        """Create publication-quality plots with statistical annotations"""
        
        print("\n=== CREATING PUBLICATION-QUALITY PLOTS ===")
        
        # Set publication style
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16
        })
        
        params = df['parameter'].unique()
        n_params = len(params)
        
        # Main stability analysis plot
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, param in enumerate(params):
            if i < len(axes):
                param_data = df[df['parameter'] == param].sort_values('value')
                ax = axes[i]
                
                # Plot stability with confidence intervals
                ax.errorbar(param_data['value'], param_data['stability_mean'],
                           yerr=[param_data['stability_mean'] - param_data['stability_ci_lower'],
                                param_data['stability_ci_upper'] - param_data['stability_mean']],
                           fmt='o-', linewidth=2.5, markersize=6, capsize=4, capthick=2,
                           color='navy', label='Stability Score', alpha=0.8)
                
                # Add threshold line
                ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Threshold')
                
                # Mark boundaries
                if boundaries.get(param) is not None:
                    boundary = boundaries[param]
                    if boundary['significant']:
                        # Shade working region
                        ax.axvspan(boundary['working_range'][0], boundary['working_range'][1], 
                                  alpha=0.2, color='green', label='Working Range')
                        
                        # Mark optimal point
                        ax.axvline(x=boundary['optimal_value'], color='darkgreen', 
                                  linestyle=':', linewidth=2, label=f"Optimal ({boundary['optimal_value']:.2f})")
                
                ax.set_xlabel(param.replace('_', ' ').title(), fontweight='bold')
                ax.set_ylabel('Stability Score', fontweight='bold')
                ax.set_title(f'{param.replace("_", " ").title()} Parameter Analysis', fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.set_ylim(-0.05, 1.05)
                ax.legend(loc='best', framealpha=0.9)
                
                # Add statistical annotation
                if boundaries.get(param) is not None and boundaries[param]['significant']:
                    ax.text(0.05, 0.95, f"p < 0.01\nEffect size: {boundaries[param]['effect_size']:.2f}", 
                           transform=ax.transAxes, verticalalignment='top', 
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Hide unused subplot
        if n_params < len(axes):
            axes[-1].set_visible(False)
        
        plt.suptitle('Comprehensive Parameter Stability Analysis', fontsize=18, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/comprehensive_stability_analysis_{self.timestamp}.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # Create summary table plot
        self._create_summary_table(df, boundaries)
    
    def _create_summary_table(self, df, boundaries):
        """Create a summary table of results"""
        
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('tight')
        ax.axis('off')
        
        # Prepare table data
        table_data = []
        headers = ['Parameter', 'Working Range', 'Optimal Value', 'Optimal Score', 'P-value', 'Effect Size', 'Significance']
        
        for param in df['parameter'].unique():
            if boundaries.get(param) is not None:
                b = boundaries[param]
                table_data.append([
                    param.replace('_', ' ').title(),
                    f"{b['working_range'][0]:.2f} - {b['working_range'][1]:.2f}",
                    f"{b['optimal_value']:.3f}",
                    f"{b['optimal_score']:.3f}",
                    f"{b['p_value']:.2e}",
                    f"{b['effect_size']:.2f}",
                    '✓' if b['significant'] else '✗'
                ])
            else:
                table_data.append([
                    param.replace('_', ' ').title(),
                    'No boundary found',
                    'N/A',
                    'N/A',
                    'N/A',
                    'N/A',
                    '✗'
                ])
        
        table = ax.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # Style the table
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        plt.title('Parameter Analysis Summary Table', fontsize=16, fontweight='bold', pad=20)
        plt.savefig(f'{self.output_dir}/summary_table_{self.timestamp}.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_final_report(self, df, boundaries):
        """Generate comprehensive final report"""
        
        report_path = f'{self.output_dir}/final_statistical_report_{self.timestamp}.txt'
        
        with open(report_path, 'w') as f:
            f.write("COMPREHENSIVE STATISTICAL STABILITY ANALYSIS - FINAL REPORT\n")
            f.write("="*70 + "\n\n")
            f.write(f"Analysis completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total simulations: {len(df) * df['n_reps'].iloc[0]}\n")
            f.write(f"Statistical significance level: p < 0.01\n\n")
            
            # Executive summary
            f.write("EXECUTIVE SUMMARY:\n")
            f.write("-" * 20 + "\n")
            significant_params = [p for p, b in boundaries.items() if b is not None and b['significant']]
            f.write(f"Parameters with statistically significant boundaries: {len(significant_params)}/{len(boundaries)}\n")
            f.write(f"Significant parameters: {', '.join(significant_params)}\n\n")
            
            # Detailed findings
            f.write("DETAILED FINDINGS:\n")
            f.write("-" * 20 + "\n\n")
            
            for param, boundary in boundaries.items():
                f.write(f"{param.upper().replace('_', ' ')}:\n")
                
                if boundary is not None and boundary['significant']:
                    f.write(f"  ✓ STATISTICALLY SIGNIFICANT BOUNDARY FOUND\n")
                    f.write(f"  Working range: {boundary['working_range'][0]:.3f} - {boundary['working_range'][1]:.3f}\n")
                    f.write(f"  Recommended value: {boundary['optimal_value']:.3f}\n")
                    f.write(f"  Expected stability: {boundary['optimal_score']:.3f}\n")
                    f.write(f"  Statistical confidence: p = {boundary['p_value']:.2e}\n")
                    f.write(f"  Effect size: {boundary['effect_size']:.3f}\n")
                    f.write(f"  Working mean: {boundary['working_mean']:.3f}\n")
                    f.write(f"  Failing mean: {boundary['failing_mean']:.3f}\n")
                elif boundary is not None:
                    f.write(f"  ⚠ BOUNDARY DETECTED BUT NOT STATISTICALLY SIGNIFICANT\n")
                    f.write(f"  Tentative working range: {boundary['working_range'][0]:.3f} - {boundary['working_range'][1]:.3f}\n")
                    f.write(f"  Statistical confidence: p = {boundary['p_value']:.3f} (not significant)\n")
                else:
                    f.write(f"  ✗ NO CLEAR BOUNDARY DETECTED\n")
                    f.write(f"  Parameter may not have strong effect on stability\n")
                
                f.write("\n")
            
            # Recommendations
            f.write("FINAL RECOMMENDATIONS:\n")
            f.write("-" * 25 + "\n\n")
            f.write("Based on statistically rigorous analysis:\n\n")
            
            for param, boundary in boundaries.items():
                if boundary is not None and boundary['significant']:
                    f.write(f"{param} = {boundary['optimal_value']:.3f}  # High confidence\n")
                elif boundary is not None:
                    f.write(f"{param} = {boundary['optimal_value']:.3f}  # Low confidence\n")
                else:
                    param_data = df[df['parameter'] == param]
                    best_value = param_data.loc[param_data['stability_mean'].idxmax(), 'value']
                    f.write(f"{param} = {best_value:.3f}  # Best observed (no clear boundary)\n")
            
            f.write(f"\nThese recommendations are based on {len(df) * df['n_reps'].iloc[0]} simulations ")
            f.write(f"with {df['n_reps'].iloc[0]} repetitions per parameter value.\n")
        
        print(f"Final report saved to: {report_path}")
        return report_path

def main():
    """Main comprehensive analysis execution"""
    print("="*80)
    print("FOCUSED COMPREHENSIVE STABILITY ANALYSIS")
    print("STATISTICALLY RIGOROUS PARAMETER BOUNDARY DETECTION")
    print("="*80)
    
    # Define VERY WIDE parameter ranges to find actual failure boundaries
    param_ranges = {
        'scale': (0.1, 10.0),           # Extremely wide range
        'imitation_period': (1, 50),    # From very frequent to very rare
        'cooperation_increase': (0.01, 0.9),   # From minimal to maximum
        'q': (0.01, 2.0),               # From very low to extremely high fishing
        'trust_decrease': (0.01, 0.9)   # From very slow to very fast decay
    }
    
    print("EXTREME PARAMETER RANGES FOR BOUNDARY DETECTION:")
    for param, (min_val, max_val) in param_ranges.items():
        print(f"  {param}: {min_val} - {max_val}")
    
    print(f"\nThis analysis will:")
    print(f"  - Test {sum(30 for _ in param_ranges)} parameter values")
    print(f"  - Run {15} repetitions each for statistical power")
    print(f"  - Execute {sum(30 * 15 for _ in param_ranges)} total simulations")
    print(f"  - Use rigorous statistical testing (p < 0.01)")
    print(f"  - Generate confidence intervals and effect sizes")
    
    # Set random seed
    np.random.seed(42)
    
    # Create analyzer
    analyzer = FocusedStabilityAnalyzer()
    
    try:
        print(f"\nStarting comprehensive analysis...")
        print(f"Estimated completion time: {sum(30 * 15 for _ in param_ranges) * 4 / 60:.0f} minutes")
        
        # Run comprehensive analysis
        df = analyzer.comprehensive_parameter_sweep(param_ranges, n_samples=30, n_reps=15)
        
        # Find statistical boundaries
        boundaries = analyzer.find_parameter_boundaries(df, metric='stability_mean', threshold=0.5)
        
        # Create publication-quality plots
        analyzer.create_publication_quality_plots(df, boundaries)
        
        # Save all data
        df.to_csv(f'{analyzer.output_dir}/comprehensive_results_{analyzer.timestamp}.csv', index=False)
        
        # Generate final report
        report_path = analyzer.generate_final_report(df, boundaries)
        
        print(f"\n" + "="*80)
        print("COMPREHENSIVE ANALYSIS COMPLETE!")
        print("="*80)
        print(f"Results directory: {analyzer.output_dir}")
        print(f"Final report: {report_path}")
        print(f"Total simulations executed: {len(df) * df['n_reps'].iloc[0]}")
        
        # Print key findings
        significant_params = [p for p, b in boundaries.items() if b is not None and b['significant']]
        print(f"\nKEY FINDINGS:")
        print(f"  Statistically significant boundaries found for: {len(significant_params)}/{len(boundaries)} parameters")
        if significant_params:
            print(f"  Significant parameters: {', '.join(significant_params)}")
        
        return df, boundaries, analyzer
        
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
        return None, None, None
    except Exception as e:
        print(f"\nError during analysis: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

if __name__ == '__main__':
    results = main() 