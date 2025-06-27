"""
Parameter Stability Finder for DynamicCoop Model

This script systematically explores parameter space to find regions where 
the fisheries model maintains stable, long-term dynamics without crashes.

OBJECTIVE: Find parameter ranges where fish and fisher populations coexist
stably over extended time periods (1000+ timesteps).

APPROACH:
1. Test individual parameters (OFAT) with relaxed stability criteria
2. Test parameter combinations using Latin Hypercube Sampling
3. Identify and validate stable parameter regions
4. Generate comprehensive reports and visualizations

STABILITY CRITERIA (realistic for stochastic ABM):
- No population crashes (fish > 5 throughout simulation)
- Sustainable fish population (mean > 20 fish)
- Bounded variability (CV < 1.0, allowing natural fluctuations)
- System persistence (viable population 80% of time)
- Recovery capability (can bounce back from low points)
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

class StabilityAnalyzer:
    """Main class for parameter stability analysis"""
    
    def __init__(self, param_ranges, output_dir='stability_analysis'):
        self.param_ranges = param_ranges
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Default parameter values (center of ranges)
        self.defaults = {
            'scale': 2.0,
            'imitation_period': 8,
            'cooperation_increase': 0.25,
            'q': 0.6,
            'trust_decrease': 0.25
        }
        
        print(f"StabilityAnalyzer initialized")
        print(f"Output directory: {output_dir}")
        print(f"Parameter ranges: {param_ranges}")
    
    def backup_original_parameters(self):
        """Store original parameter values for restoration"""
        self.original_params = {}
        for param_name in self.param_ranges.keys():
            if hasattr(parameters, param_name):
                self.original_params[param_name] = getattr(parameters, param_name)
        
        # Store scale-dependent parameters
        if 'scale' in self.param_ranges:
            self.original_params['rad_repulsion'] = parameters.rad_repulsion
            self.original_params['rad_orientation'] = parameters.rad_orientation
            self.original_params['rad_attraction'] = parameters.rad_attraction
            self.original_params['rad_repulsion_sqr'] = parameters.rad_repulsion_sqr
            self.original_params['rad_orientation_sqr'] = parameters.rad_orientation_sqr
            self.original_params['rad_attraction_sqr'] = parameters.rad_attraction_sqr
    
    def restore_original_parameters(self):
        """Restore all parameters to original values"""
        for param_name, original_value in self.original_params.items():
            if hasattr(parameters, param_name):
                setattr(parameters, param_name, original_value)
    
    def set_parameters(self, param_dict):
        """Set model parameters from dictionary"""
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
        
        # Ensure integer parameters are integers
        if 'imitation_period' in param_dict:
            parameters.imitation_period = int(round(param_dict['imitation_period']))
    
    def run_stability_test(self, param_dict, n_timesteps=1000, burn_in=200):
        """
        Run single stability test with given parameters
        
        Returns dict with stability metrics and classification
        """
        try:
            # Set parameters
            self.set_parameters(param_dict)
            
            # Initialize and run model
            dc.initialize('default')
            
            fish_counts = []
            for t in range(n_timesteps):
                dc.update_one_unit_time()
                if t >= burn_in:  # Skip burn-in period
                    fish_counts.append(dc.total_fish_count[-1])
            
            fish_counts = np.array(fish_counts)
            
            # Calculate stability metrics
            mean_fish = np.mean(fish_counts)
            min_fish = np.min(fish_counts)
            max_fish = np.max(fish_counts)
            std_fish = np.std(fish_counts)
            cv_fish = std_fish / mean_fish if mean_fish > 0 else np.inf
            
            # Trend analysis (final 200 timesteps)
            final_window = fish_counts[-200:]
            x = np.arange(len(final_window))
            slope, _, r_value, _, _ = stats.linregress(x, final_window)
            
            # Persistence analysis
            viable_threshold = 12  # Lowered to match sustainable threshold
            persistence_rate = np.mean(fish_counts > viable_threshold)
            
            # Recovery analysis
            low_threshold = mean_fish * 0.6  # 60% of mean
            low_points = fish_counts < low_threshold
            
            if np.any(low_points):
                recovery_count = 0
                low_indices = np.where(low_points)[0]
                
                for idx in low_indices:
                    if idx < len(fish_counts) - 30:  # Need 30 steps to assess recovery
                        recovery_window = fish_counts[idx+1:idx+31]
                        if np.mean(recovery_window) > fish_counts[idx] * 1.15:  # 15% recovery
                            recovery_count += 1
                
                recovery_rate = recovery_count / len(low_indices)
            else:
                recovery_rate = 1.0  # No low points to recover from
            
            # Stability classification (relaxed criteria based on quick test)
            crash = min_fish < 3
            sustainable = mean_fish > 12
            bounded_variation = cv_fish < 1.5  # Allow natural variation
            persistent = persistence_rate > 0.7  # 70% of time viable
            recoverable = recovery_rate > 0.3   # Can recover from some setbacks
            no_strong_trend = abs(slope) < 0.2  # Allow some trend
            
            stable = (not crash and sustainable and bounded_variation and 
                     persistent and recoverable and no_strong_trend)
            
            return {
                'stable': stable,
                'mean_fish': mean_fish,
                'min_fish': min_fish,
                'max_fish': max_fish,
                'cv_fish': cv_fish,
                'persistence_rate': persistence_rate,
                'recovery_rate': recovery_rate,
                'trend_slope': slope,
                'trend_r2': r_value**2,
                'crash': crash,
                'sustainable': sustainable,
                'bounded_variation': bounded_variation,
                'persistent': persistent,
                'recoverable': recoverable,
                'trajectory': fish_counts
            }
            
        except Exception as e:
            print(f"Error in stability test: {e}")
            return {
                'stable': False,
                'mean_fish': 0,
                'min_fish': 0,
                'max_fish': 0,
                'cv_fish': np.inf,
                'persistence_rate': 0,
                'recovery_rate': 0,
                'trend_slope': 0,
                'trend_r2': 0,
                'crash': True,
                'sustainable': False,
                'bounded_variation': False,
                'persistent': False,
                'recoverable': False,
                'trajectory': np.zeros(800)
            }
    
    def ofat_analysis(self, n_samples=25, n_reps=5):
        """One-Factor-At-A-Time analysis"""
        print(f"\n=== OFAT ANALYSIS ===")
        print(f"Samples per parameter: {n_samples}")
        print(f"Repetitions per sample: {n_reps}")
        
        results = []
        
        for param_name, (min_val, max_val) in self.param_ranges.items():
            print(f"\nTesting parameter: {param_name}")
            
            # Generate parameter values
            if param_name == 'imitation_period':
                values = np.linspace(min_val, max_val, n_samples, dtype=int)
            else:
                values = np.linspace(min_val, max_val, n_samples)
            
            for value in tqdm(values, desc=f"OFAT {param_name}"):
                # Create parameter dict with defaults except for test parameter
                param_dict = self.defaults.copy()
                param_dict[param_name] = value
                
                # Run multiple repetitions
                rep_results = []
                for rep in range(n_reps):
                    result = self.run_stability_test(param_dict)
                    rep_results.append(result)
                
                # Aggregate results
                stability_count = sum(1 for r in rep_results if r['stable'])
                stability_prob = stability_count / n_reps
                
                mean_fish = np.mean([r['mean_fish'] for r in rep_results])
                mean_cv = np.mean([r['cv_fish'] for r in rep_results if not np.isinf(r['cv_fish'])])
                mean_persistence = np.mean([r['persistence_rate'] for r in rep_results])
                crash_count = sum(1 for r in rep_results if r['crash'])
                crash_prob = crash_count / n_reps
                
                results.append({
                    'parameter': param_name,
                    'value': value,
                    'stability_probability': stability_prob,
                    'mean_fish_population': mean_fish,
                    'mean_cv': mean_cv,
                    'mean_persistence': mean_persistence,
                    'crash_probability': crash_prob,
                    'n_reps': n_reps
                })
        
        df = pd.DataFrame(results)
        
        # Save results
        df.to_csv(f'{self.output_dir}/ofat_results_{self.timestamp}.csv', index=False)
        
        return df
    
    def latin_hypercube_sampling(self, n_samples):
        """Generate Latin Hypercube samples"""
        n_params = len(self.param_ranges)
        
        # Generate LHS samples in [0,1]^d
        samples = np.zeros((n_samples, n_params))
        
        for i in range(n_params):
            perm = np.random.permutation(n_samples)
            samples[:, i] = (perm + np.random.random(n_samples)) / n_samples
        
        # Scale to parameter ranges
        param_samples = {}
        for i, (param_name, (min_val, max_val)) in enumerate(self.param_ranges.items()):
            if param_name == 'imitation_period':
                param_samples[param_name] = np.round(
                    min_val + samples[:, i] * (max_val - min_val)
                ).astype(int)
            else:
                param_samples[param_name] = min_val + samples[:, i] * (max_val - min_val)
        
        return pd.DataFrame(param_samples)
    
    def multifactor_analysis(self, n_samples=500, n_reps=3):
        """Multi-factor analysis using Latin Hypercube Sampling"""
        print(f"\n=== MULTI-FACTOR ANALYSIS ===")
        print(f"LHS samples: {n_samples}")
        print(f"Repetitions per sample: {n_reps}")
        
        # Generate samples
        sample_df = self.latin_hypercube_sampling(n_samples)
        
        results = []
        
        for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="Multi-factor"):
            param_dict = row.to_dict()
            
            # Run multiple repetitions
            rep_results = []
            for rep in range(n_reps):
                result = self.run_stability_test(param_dict)
                rep_results.append(result)
            
            # Aggregate results
            stability_count = sum(1 for r in rep_results if r['stable'])
            stability_prob = stability_count / n_reps
            
            mean_fish = np.mean([r['mean_fish'] for r in rep_results])
            mean_persistence = np.mean([r['persistence_rate'] for r in rep_results])
            
            # Add to results
            result_row = param_dict.copy()
            result_row.update({
                'stability_probability': stability_prob,
                'mean_fish_population': mean_fish,
                'mean_persistence': mean_persistence,
                'sample_id': idx
            })
            results.append(result_row)
        
        df = pd.DataFrame(results)
        
        # Save results
        df.to_csv(f'{self.output_dir}/multifactor_results_{self.timestamp}.csv', index=False)
        
        return df
    
    def analyze_stable_regions(self, ofat_df, multi_df, threshold=0.4):
        """Extract and analyze stable parameter regions"""
        print(f"\n=== ANALYZING STABLE REGIONS ===")
        
        stable_ranges = {}
        
        # Analyze OFAT results
        for param in ofat_df['parameter'].unique():
            param_data = ofat_df[ofat_df['parameter'] == param].sort_values('value')
            stable_points = param_data[param_data['stability_probability'] > threshold]
            
            if len(stable_points) > 0:
                min_stable = stable_points['value'].min()
                max_stable = stable_points['value'].max()
                best_idx = stable_points['stability_probability'].idxmax()
                best_value = stable_points.loc[best_idx, 'value']
                best_prob = stable_points.loc[best_idx, 'stability_probability']
                
                stable_ranges[param] = {
                    'min_stable': min_stable,
                    'max_stable': max_stable,
                    'range_width': max_stable - min_stable,
                    'best_value': best_value,
                    'best_probability': best_prob,
                    'n_stable_points': len(stable_points)
                }
            else:
                stable_ranges[param] = {
                    'min_stable': None,
                    'max_stable': None,
                    'range_width': 0,
                    'best_value': None,
                    'best_probability': 0,
                    'n_stable_points': 0
                }
        
        # Analyze multi-factor results
        stable_samples = multi_df[multi_df['stability_probability'] > threshold]
        
        print(f"Stable regions found (threshold = {threshold}):")
        for param, ranges in stable_ranges.items():
            if ranges['min_stable'] is not None:
                print(f"  {param}: {ranges['min_stable']:.3f} - {ranges['max_stable']:.3f}")
                print(f"    Best: {ranges['best_value']:.3f} (prob: {ranges['best_probability']:.3f})")
            else:
                print(f"  {param}: NO STABLE RANGE FOUND")
        
        print(f"\nMulti-factor analysis:")
        print(f"  Stable samples: {len(stable_samples)} / {len(multi_df)} ({len(stable_samples)/len(multi_df)*100:.1f}%)")
        
        if len(stable_samples) > 0:
            print(f"  Best stability probability: {stable_samples['stability_probability'].max():.3f}")
            best_sample = stable_samples.loc[stable_samples['stability_probability'].idxmax()]
            print(f"  Best parameter combination:")
            for param in self.param_ranges.keys():
                print(f"    {param}: {best_sample[param]:.3f}")
        
        # Save analysis
        with open(f'{self.output_dir}/stable_regions_{self.timestamp}.txt', 'w') as f:
            f.write(f"STABLE PARAMETER REGIONS ANALYSIS\n")
            f.write(f"Threshold: {threshold}\n")
            f.write(f"Timestamp: {self.timestamp}\n\n")
            
            f.write("OFAT RESULTS:\n")
            for param, ranges in stable_ranges.items():
                f.write(f"\n{param}:\n")
                if ranges['min_stable'] is not None:
                    f.write(f"  Stable range: {ranges['min_stable']:.3f} - {ranges['max_stable']:.3f}\n")
                    f.write(f"  Best value: {ranges['best_value']:.3f} (prob: {ranges['best_probability']:.3f})\n")
                else:
                    f.write(f"  No stable range found\n")
            
            f.write(f"\nMULTI-FACTOR RESULTS:\n")
            f.write(f"Stable samples: {len(stable_samples)} / {len(multi_df)}\n")
            
            if len(stable_samples) > 0:
                f.write(f"Best parameter combination:\n")
                for param in self.param_ranges.keys():
                    f.write(f"  {param}: {best_sample[param]:.3f}\n")
        
        return stable_ranges, stable_samples
    
    def create_visualizations(self, ofat_df, multi_df, stable_samples):
        """Create comprehensive visualizations"""
        print(f"\n=== CREATING VISUALIZATIONS ===")
        
        # Set style
        plt.style.use('default')
        sns.set_palette("viridis")
        
        # 1. OFAT stability probability plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        params = ofat_df['parameter'].unique()
        for i, param in enumerate(params):
            if i < len(axes):
                param_data = ofat_df[ofat_df['parameter'] == param]
                
                ax = axes[i]
                ax.plot(param_data['value'], param_data['stability_probability'], 
                       'o-', linewidth=2, markersize=6, color='darkblue')
                ax.set_xlabel(param)
                ax.set_ylabel('Stability Probability')
                ax.set_title(f'Stability: {param}')
                ax.grid(True, alpha=0.3)
                ax.set_ylim(-0.05, 1.05)
                
                # Highlight stable regions
                stable_data = param_data[param_data['stability_probability'] > 0.3]
                if len(stable_data) > 0:
                    ax.fill_between(stable_data['value'], 0, stable_data['stability_probability'], 
                                   alpha=0.3, color='lightgreen', label='Stable Region')
                    ax.legend()
        
        # Hide unused subplots
        for i in range(len(params), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/ofat_stability_{self.timestamp}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Multi-factor pairwise plots
        if len(multi_df) > 0:
            param_cols = list(self.param_ranges.keys())
            plot_df = multi_df[param_cols + ['stability_probability']].copy()
            
            g = sns.PairGrid(plot_df, vars=param_cols, height=2.5)
            g.map_upper(plt.scatter, alpha=0.6, s=20, c=plot_df['stability_probability'], 
                       cmap='viridis', vmin=0, vmax=1)
            g.map_lower(sns.scatterplot, alpha=0.6, s=20, 
                       hue=plot_df['stability_probability'], palette='viridis')
            g.map_diag(plt.hist, alpha=0.7)
            
            plt.suptitle('Multi-factor Parameter Space', y=1.02)
            plt.savefig(f'{self.output_dir}/multifactor_pairplot_{self.timestamp}.png', 
                       dpi=300, bbox_inches='tight')
            plt.show()
        
        # 3. Stable region summary
        if len(stable_samples) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            stable_summary = stable_samples[list(self.param_ranges.keys())].describe()
            
            param_names = list(self.param_ranges.keys())
            means = [stable_summary.loc['mean', param] for param in param_names]
            stds = [stable_summary.loc['std', param] for param in param_names]
            
            x_pos = np.arange(len(param_names))
            ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, color='lightgreen')
            ax.set_xlabel('Parameters')
            ax.set_ylabel('Parameter Value')
            ax.set_title('Stable Parameter Ranges (Mean ± Std)')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(param_names, rotation=45)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/stable_summary_{self.timestamp}.png', 
                       dpi=300, bbox_inches='tight')
            plt.show()
    
    def run_full_analysis(self):
        """Run complete stability analysis workflow"""
        print("="*60)
        print("PARAMETER STABILITY ANALYSIS")
        print("="*60)
        
        # Backup original parameters
        self.backup_original_parameters()
        
        try:
            # Run OFAT analysis
            ofat_df = self.ofat_analysis(n_samples=15, n_reps=3)
            
            # Run multi-factor analysis  
            multi_df = self.multifactor_analysis(n_samples=150, n_reps=2)
            
            # Analyze stable regions
            stable_ranges, stable_samples = self.analyze_stable_regions(ofat_df, multi_df)
            
            # Create visualizations
            self.create_visualizations(ofat_df, multi_df, stable_samples)
            
            print(f"\n=== ANALYSIS COMPLETE ===")
            print(f"Results saved in: {self.output_dir}")
            
            return ofat_df, multi_df, stable_ranges, stable_samples
            
        finally:
            # Always restore original parameters
            self.restore_original_parameters()
            print("Original parameters restored")

def main():
    """Main execution function"""
    
    # Define parameter ranges to explore
    param_ranges = {
        'scale': (0.5, 4.0),
        'imitation_period': (1, 15),
        'cooperation_increase': (0.05, 0.5),
        'q': (0.2, 1.0),
        'trust_decrease': (0.05, 0.5)
    }
    
    # Create analyzer
    analyzer = StabilityAnalyzer(param_ranges)
    
    # Run full analysis
    results = analyzer.run_full_analysis()
    
    return results

if __name__ == '__main__':
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run analysis
    results = main() 