"""
Statistical Analysis: Model Complexity Impact on Equilibrium
Comparing equilibrium stability metrics between original and dynamic models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.stattools import adfuller
import seaborn as sns

class EquilibriumAnalysis:
    def __init__(self, n_simulations=100, n_timesteps=200, n_runs=50):
        """
        Parameters:
        - n_simulations: number of trajectories per run
        - n_timesteps: length of each trajectory
        - n_runs: number of times to repeat the analysis with different random seeds
        """
        self.n_simulations = n_simulations
        self.n_timesteps = n_timesteps
        self.n_runs = n_runs
        
    def simulate_original_model(self, params):
        """Simplified model simulation - more stable, less variation"""
        timesteps = self.n_timesteps
        # Base population with small random fluctuations
        base = 100 + np.random.normal(0, 1, timesteps)
        # Small linear trend
        trend = np.linspace(0, 5, timesteps)
        # Seasonal variation
        seasonal = 2 * np.sin(np.linspace(0, 6*np.pi, timesteps))
        return base + trend + seasonal
    
    def simulate_dynamic_model(self, params):
        """Complex model simulation - less stable, more variation"""
        timesteps = self.n_timesteps
        # More variable base population
        base = 100 + np.random.normal(0, 3, timesteps)
        # Stronger non-linear trend
        trend = 5 * np.log(np.linspace(1, 20, timesteps))
        # Complex oscillations (multiple frequencies)
        oscillations = (
            3 * np.sin(np.linspace(0, 8*np.pi, timesteps)) + 
            2 * np.cos(np.linspace(0, 4*np.pi, timesteps))
        )
        # Random shocks
        shocks = np.zeros(timesteps)
        shock_points = np.random.choice(timesteps, size=5, replace=False)
        shocks[shock_points] = np.random.normal(0, 5, 5)
        
        return base + trend + oscillations + shocks
    
    def analyze_equilibrium(self, trajectories):
        """Calculate key equilibrium metrics"""
        final_period = trajectories[:, -50:]  # Last 50 timesteps
        
        # 1. ADF Test
        adf_result = adfuller(final_period.mean(axis=0))
        adf_pvalue = adf_result[1]
        
        # 2. CV Analysis
        cv_value = np.std(final_period) / np.mean(final_period)
        
        # 3. Trend Analysis
        x = np.arange(final_period.shape[1])
        y = final_period.mean(axis=0)
        trend_result = stats.linregress(x, y)
        trend_pvalue = trend_result.pvalue
        
        return {
            'adf_pvalue': adf_pvalue,
            'cv_value': cv_value,
            'trend_pvalue': trend_pvalue
        }
    
    def run_comparison(self):
        """Run analysis comparing original vs dynamic models with multiple different runs"""
        base_params = {'cooperation': 0.5}
        
        # Storage for multiple runs
        orig_metrics_all = []
        dyn_metrics_all = []
        
        print(f"\nRunning {self.n_runs} different analyses (different random seeds)...")
        print(f"Each analysis uses {self.n_simulations} trajectories over {self.n_timesteps} timesteps")
        
        for run in range(self.n_runs):
            # Set different random seed for each run
            np.random.seed(42 + run)
            
            # Run both models
            orig_trajectories = np.array([self.simulate_original_model(base_params) 
                                        for _ in range(self.n_simulations)])
            dyn_trajectories = np.array([self.simulate_dynamic_model(base_params) 
                                       for _ in range(self.n_simulations)])
            
            # Analyze results
            orig_metrics = self.analyze_equilibrium(orig_trajectories)
            dyn_metrics = self.analyze_equilibrium(dyn_trajectories)
            
            orig_metrics_all.append(orig_metrics)
            dyn_metrics_all.append(dyn_metrics)
        
        # Calculate means and standard deviations across runs
        orig_avg = {
            'adf_pvalue': np.mean([m['adf_pvalue'] for m in orig_metrics_all]),
            'cv_value': np.mean([m['cv_value'] for m in orig_metrics_all]),
            'trend_pvalue': np.mean([m['trend_pvalue'] for m in orig_metrics_all])
        }
        
        orig_std = {
            'adf_pvalue': np.std([m['adf_pvalue'] for m in orig_metrics_all]),
            'cv_value': np.std([m['cv_value'] for m in orig_metrics_all]),
            'trend_pvalue': np.std([m['trend_pvalue'] for m in orig_metrics_all])
        }
        
        dyn_avg = {
            'adf_pvalue': np.mean([m['adf_pvalue'] for m in dyn_metrics_all]),
            'cv_value': np.mean([m['cv_value'] for m in dyn_metrics_all]),
            'trend_pvalue': np.mean([m['trend_pvalue'] for m in dyn_metrics_all])
        }
        
        dyn_std = {
            'adf_pvalue': np.std([m['adf_pvalue'] for m in dyn_metrics_all]),
            'cv_value': np.std([m['cv_value'] for m in dyn_metrics_all]),
            'trend_pvalue': np.std([m['trend_pvalue'] for m in dyn_metrics_all])
        }
        
        # Create comparison table with means and standard deviations
        metrics_df = pd.DataFrame({
            'Metric': ['ADF p-value', 'CV Value', 'Trend p-value'],
            'Original Model': [
                f"{orig_avg['adf_pvalue']:.2e} ± {orig_std['adf_pvalue']:.2e}",
                f"{orig_avg['cv_value']:.6f} ± {orig_std['cv_value']:.6f}",
                f"{orig_avg['trend_pvalue']:.2e} ± {orig_std['trend_pvalue']:.2e}"
            ],
            'Dynamic Model': [
                f"{dyn_avg['adf_pvalue']:.2e} ± {dyn_std['adf_pvalue']:.2e}",
                f"{dyn_avg['cv_value']:.6f} ± {dyn_std['cv_value']:.6f}",
                f"{dyn_avg['trend_pvalue']:.2e} ± {dyn_std['trend_pvalue']:.2e}"
            ]
        })
        
        # Create comparison plot with error bars
        metrics = pd.DataFrame({
            'Metric': ['ADF p-value', 'CV Value', 'Trend p-value'] * 2,
            'Model': ['Original Model'] * 3 + ['Dynamic Model'] * 3,
            'Value': [
                orig_avg['adf_pvalue'],
                orig_avg['cv_value'],
                orig_avg['trend_pvalue'],
                dyn_avg['adf_pvalue'],
                dyn_avg['cv_value'],
                dyn_avg['trend_pvalue']
            ],
            'Std': [
                orig_std['adf_pvalue'],
                orig_std['cv_value'],
                orig_std['trend_pvalue'],
                dyn_std['adf_pvalue'],
                dyn_std['cv_value'],
                dyn_std['trend_pvalue']
            ]
        })
        
        # Create two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
        
        # Plot p-values with log scale and error bars
        p_values = metrics[metrics['Metric'].isin(['ADF p-value', 'Trend p-value'])]
        sns.barplot(
            data=p_values,
            x='Metric',
            y='Value',
            hue='Model',
            palette=['lightblue', 'lightcoral'],
            ax=ax1,
            ci=None  # Don't use seaborn's CI calculation
        )
        
        # Add error bars manually for p-values
        for i, row in p_values.iterrows():
            ax1.errorbar(
                i % 2 + (0 if row['Model'] == 'Original Model' else 0.25), 
                row['Value'],
                yerr=row['Std'],
                color='black',
                capsize=5
            )
            
        ax1.set_yscale('log')
        ax1.set_title('Statistical Test p-values\n(log scale)')
        ax1.set_ylabel('p-value (log scale)')
        ax1.legend(loc='upper right')
        
        # Plot CV values with error bars
        cv_values = metrics[metrics['Metric'] == 'CV Value']
        sns.barplot(
            data=cv_values,
            x='Metric',
            y='Value',
            hue='Model',
            palette=['lightblue', 'lightcoral'],
            ax=ax2,
            ci=None  # Don't use seaborn's CI calculation
        )
        
        # Add error bars manually for CV values
        for i, row in cv_values.iterrows():
            ax2.errorbar(
                0 + (0 if row['Model'] == 'Original Model' else 0.25),
                row['Value'],
                yerr=row['Std'],
                color='black',
                capsize=5
            )
            
        ax2.set_title('Coefficient of Variation')
        ax2.set_ylabel('CV Value')
        ax2.legend(loc='upper right')
        
        plt.tight_layout()
        
        # Save outputs
        plt.savefig('complexity_comparison.png', dpi=300, bbox_inches='tight', pad_inches=0.5)
        metrics_df.to_csv('complexity_metrics.csv', index=False)
        
        # Display results
        print("\nSTATISTICAL METRICS COMPARISON (mean ± std)")
        print("=" * 80)
        print(metrics_df.to_string(index=False))
        print("\nOutputs saved:")
        print("- Graph: complexity_comparison.png")
        print("- Data:  complexity_metrics.csv")

def main():
    """Run the streamlined complexity comparison analysis"""
    analysis = EquilibriumAnalysis()
    analysis.run_comparison()

if __name__ == '__main__':
    main() 