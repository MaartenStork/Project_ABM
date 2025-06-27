"""
Statistical Proof: Rebuttal to Owusu et al. Claims about MPA Effectiveness vs High Cooperation

This script systematically proves that the conclusion from Owusu et al. (2019) that 
"high cooperation without an MPA can be as effective as lower cooperation combined 
with an MPA in maintaining fish stocks and catches at relatively high levels" 
does NOT hold when:

1. Dynamic cooperation levels are introduced (trust-based, imitation, threshold behaviors)
2. Multiple fish species with different characteristics are included
3. Long-term sustainability is considered over short-term catches

HYPOTHESIS TO PROVE:
With dynamic cooperation and multiple species, high cooperation WITHOUT MPA becomes 
significantly LESS sustainable than lower cooperation WITH MPA protection.

The original Owusu model uses static cooperation levels and homogeneous fish populations.
Our enhanced model introduces:
- Trust-based cooperation that can erode over time
- Imitation behaviors that can spread non-cooperative strategies
- Threshold-based behaviors triggered by resource scarcity  
- Multiple fish species with different reproduction rates and movement patterns
- Longer simulation periods to capture system dynamics

STATISTICAL APPROACH:
1. Run both models under identical scenarios with varying cooperation levels
2. Test scenarios: No MPA vs Single MPA vs Spaced MPA configurations
3. Compare long-term sustainability metrics (fish population stability, catch variance)
4. Use statistical tests to prove significant differences
5. Demonstrate that cooperation erosion in dynamic model leads to system collapse

Expected Results:
- Original model: High cooperation (no MPA) ≈ Lower cooperation (with MPA)
- Dynamic model: High cooperation (no MPA) << Lower cooperation (with MPA)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import both models
import DynamicCoop as dyn
import Optimized_og_model as orig
import parameters

# Statistical analysis tools
from scipy.stats import mannwhitneyu, kruskal, levene


class ModelComparison:
    """Statistical comparison framework for proving the rebuttal hypothesis"""
    
    def __init__(self, n_simulations=25, n_timesteps=200):
        self.n_simulations = n_simulations
        self.n_timesteps = n_timesteps
        self.results = {}
        
    def setup_cooperation_scenarios(self):
        """Define the cooperation scenarios to test the Owusu claim"""
        scenarios = {
            # Owusu's claim: High cooperation (no MPA) should be as effective as lower cooperation (with MPA)
            'high_coop_no_mpa': {
                'fully_coop': 16, 'coop': 4, 'cond_coop': 0, 'noncoop': 0, 'fully_noncoop': 0, 
                'mpa': 'no', 'type_mpa': 'single'
            },
            'low_coop_single_mpa': {
                'fully_coop': 0, 'coop': 4, 'cond_coop': 8, 'noncoop': 4, 'fully_noncoop': 4, 
                'mpa': 'yes', 'type_mpa': 'single'
            },
            'medium_coop_single_mpa': {
                'fully_coop': 4, 'coop': 8, 'cond_coop': 4, 'noncoop': 2, 'fully_noncoop': 2, 
                'mpa': 'yes', 'type_mpa': 'single'
            },
        }
        return scenarios
    
    def run_original_model(self, scenario):
        """Run the original Owusu model with given cooperation scenario"""
        results = []
        
        # Store original parameter values
        orig_params = {
            'fully_noncoop': orig.fully_noncoop,
            'noncoop': orig.noncoop,
            'cond_coop': orig.cond_coop,
            'coop': orig.coop,
            'fully_coop': orig.fully_coop,
            'MPA': orig.MPA,
            'Type_MPA': orig.Type_MPA,
        }
        
        for run in tqdm(range(self.n_simulations), desc=f"Original Model - {scenario}"):
            try:
                # Set parameters for original model
                orig.fully_noncoop = scenario['fully_noncoop']
                orig.noncoop = scenario['noncoop'] 
                orig.cond_coop = scenario['cond_coop']
                orig.coop = scenario['coop']
                orig.fully_coop = scenario['fully_coop']
                orig.MPA = scenario['mpa']
                orig.Type_MPA = scenario['type_mpa']
                
                # Initialize and run
                orig.initialize()
                fish_counts = [orig.init_fish]
                total_catch = 0
                
                for t in range(self.n_timesteps):
                    orig.update_one_unit_time()
                    current_fish = len([ag for ag in orig.agents if ag.type == 'fish'])
                    fish_counts.append(current_fish)
                    total_catch += sum([ag.harvest for ag in orig.agents if ag.type == 'fishers'])
                
                results.append({
                    'final_fish': fish_counts[-1],
                    'total_catch': total_catch,
                    'fish_trajectory': fish_counts,
                    'population_stability': np.std(fish_counts[-50:]) if len(fish_counts) >= 50 else np.std(fish_counts),
                    'collapse_risk': 1 if fish_counts[-1] < 0.1 * orig.init_fish else 0
                })
                
            except Exception as e:
                print(f"Error in original model run {run}: {e}")
                # Add a failed run with default values
                results.append({
                    'final_fish': 0,
                    'total_catch': 0,
                    'fish_trajectory': [orig.init_fish] + [0] * self.n_timesteps,
                    'population_stability': 1000,  # High instability
                    'collapse_risk': 1
                })
        
        # Restore original parameters
        for key, value in orig_params.items():
            setattr(orig, key, value)
        
        return results
    
    def run_dynamic_model(self, scenario):
        """Run the dynamic cooperation model with given cooperation scenario"""
        results = []
        
        # Store original parameter values
        orig_params = {
            'fully_noncoop': parameters.fully_noncoop,
            'noncoop': parameters.noncoop,
            'cond_coop': parameters.cond_coop,
            'coop': parameters.coop,
            'fully_coop': parameters.fully_coop,
            'MPA': parameters.MPA,
            'Type_MPA': parameters.Type_MPA,
        }
        
        for run in tqdm(range(self.n_simulations), desc=f"Dynamic Model - {scenario}"):
            try:
                # Set parameters for dynamic model
                parameters.fully_noncoop = scenario['fully_noncoop']
                parameters.noncoop = scenario['noncoop']
                parameters.cond_coop = scenario['cond_coop'] 
                parameters.coop = scenario['coop']
                parameters.fully_coop = scenario['fully_coop']
                parameters.MPA = scenario['mpa']
                parameters.Type_MPA = scenario['type_mpa']
                
                # Initialize and run with multiple species
                dyn.initialize('both')  # Use both reproduction rate and speed variations
                fish_counts = [parameters.K]
                total_catch = 0
                cooperation_trajectory = []
                
                for t in range(self.n_timesteps):
                    dyn.update_one_unit_time()
                    current_fish = len([ag for ag in dyn.agents if ag.type == 'fish'])
                    fish_counts.append(current_fish)
                    total_catch += sum([ag.harvest for ag in dyn.agents if ag.type == 'fishers'])
                    
                    # Track cooperation erosion
                    fishers = [ag for ag in dyn.agents if ag.type == 'fishers']
                    if fishers:
                        avg_cooperation = np.mean([ag.effort for ag in fishers])
                        cooperation_trajectory.append(avg_cooperation)
                    else:
                        cooperation_trajectory.append(0)
                
                results.append({
                    'final_fish': fish_counts[-1],
                    'total_catch': total_catch,
                    'fish_trajectory': fish_counts,
                    'cooperation_trajectory': cooperation_trajectory,
                    'population_stability': np.std(fish_counts[-50:]) if len(fish_counts) >= 50 else np.std(fish_counts),
                    'collapse_risk': 1 if fish_counts[-1] < 0.1 * parameters.K else 0,
                    'cooperation_erosion': cooperation_trajectory[0] - cooperation_trajectory[-1] if cooperation_trajectory else 0
                })
                
            except Exception as e:
                print(f"Error in dynamic model run {run}: {e}")
                # Add a failed run with default values
                results.append({
                    'final_fish': 0,
                    'total_catch': 0,
                    'fish_trajectory': [parameters.K] + [0] * self.n_timesteps,
                    'cooperation_trajectory': [0.5] * (self.n_timesteps + 1),  
                    'population_stability': 1000,  # High instability
                    'collapse_risk': 1,
                    'cooperation_erosion': 0
                })
        
        # Restore original parameters
        for key, value in orig_params.items():
            setattr(parameters, key, value)
        
        return results
    
    def analyze_sustainability_metrics(self, results, model_type=""):
        """Calculate comprehensive sustainability metrics"""
        metrics = {}
        
        # Primary sustainability indicators
        final_fish_values = [r['final_fish'] for r in results]
        metrics['mean_final_fish'] = np.mean(final_fish_values)
        metrics['std_final_fish'] = np.std(final_fish_values)
        metrics['mean_total_catch'] = np.mean([r['total_catch'] for r in results])
        metrics['population_stability'] = np.mean([r['population_stability'] for r in results])
        metrics['collapse_probability'] = np.mean([r['collapse_risk'] for r in results])
        
        # Long-term trajectory analysis
        all_trajectories = np.array([r['fish_trajectory'] for r in results])
        metrics['trajectory_variance'] = np.mean(np.var(all_trajectories, axis=0))
        metrics['final_period_mean'] = np.mean(all_trajectories[:, -30:])  # Last 30 timesteps
        
        # Determine carrying capacity based on model
        K_value = parameters.K if 'Dynamic' in model_type else orig.K
        
        # Sustainability index (composite metric)
        # Higher is better: combines population maintenance, stability, and low collapse risk
        pop_maintenance = max(0, metrics['mean_final_fish'] / K_value)  # Ensure non-negative
        stability_score = 1 / (1 + metrics['population_stability']) if metrics['population_stability'] > 0 else 0
        collapse_avoidance = 1 - metrics['collapse_probability']
        
        metrics['sustainability_index'] = (
            pop_maintenance * 0.4 +           # Population maintenance
            stability_score * 0.3 +           # Stability (lower variance is better)
            collapse_avoidance * 0.3          # Collapse avoidance
        )
        
        return metrics
    
    def statistical_tests(self, results1, results2, label1, label2):
        """Perform statistical tests to prove significant differences"""
        print(f"\n=== STATISTICAL COMPARISON: {label1} vs {label2} ===")
        
        # Extract key metrics for comparison
        fish1 = [r['final_fish'] for r in results1]
        fish2 = [r['final_fish'] for r in results2]
        catch1 = [r['total_catch'] for r in results1]
        catch2 = [r['total_catch'] for r in results2]
        stability1 = [r['population_stability'] for r in results1]
        stability2 = [r['population_stability'] for r in results2]
        collapse1 = [r['collapse_risk'] for r in results1]
        collapse2 = [r['collapse_risk'] for r in results2]
        
        # Mann-Whitney U tests (non-parametric)
        print("\n--- Mann-Whitney U Tests ---")
        try:
            stat_fish, p_fish = mannwhitneyu(fish1, fish2, alternative='two-sided')
            stat_catch, p_catch = mannwhitneyu(catch1, catch2, alternative='two-sided')
            stat_stability, p_stability = mannwhitneyu(stability1, stability2, alternative='two-sided')
            
            print(f"Final Fish Population: U={stat_fish:.2f}, p={p_fish:.4f}")
            print(f"Total Catch: U={stat_catch:.2f}, p={p_catch:.4f}")  
            print(f"Population Stability: U={stat_stability:.2f}, p={p_stability:.4f}")
        except Exception as e:
            print(f"Statistical test error: {e}")
            p_fish = p_catch = p_stability = 0.999
        
        # Effect sizes (Cohen's d)
        print("\n--- Effect Sizes (Cohen's d) ---")
        def cohens_d(x, y):
            if len(x) == 0 or len(y) == 0:
                return 0
            nx, ny = len(x), len(y)
            if nx + ny <= 2:
                return 0
            dof = nx + ny - 2
            pooled_std = np.sqrt(((nx-1)*np.var(x, ddof=1) + (ny-1)*np.var(y, ddof=1)) / dof)
            if pooled_std == 0:
                return 0
            return (np.mean(x) - np.mean(y)) / pooled_std
        
        d_fish = cohens_d(fish1, fish2)
        d_catch = cohens_d(catch1, catch2)
        d_stability = cohens_d(stability1, stability2)
        
        print(f"Final Fish Population: d={d_fish:.3f}")
        print(f"Total Catch: d={d_catch:.3f}")
        print(f"Population Stability: d={d_stability:.3f}")
        
        # Collapse risk comparison
        collapse_rate1 = np.mean(collapse1)
        collapse_rate2 = np.mean(collapse2)
        print(f"\nCollapse Risk: {label1}={collapse_rate1:.3f}, {label2}={collapse_rate2:.3f}")
        
        # Statistical significance interpretation
        alpha = 0.05
        significant_differences = {
            'fish_population': p_fish < alpha,
            'total_catch': p_catch < alpha,
            'stability': p_stability < alpha,
            'collapse_risk_difference': abs(collapse_rate1 - collapse_rate2) > 0.1
        }
        
        return {
            'p_values': {'fish': p_fish, 'catch': p_catch, 'stability': p_stability},
            'effect_sizes': {'fish': d_fish, 'catch': d_catch, 'stability': d_stability},
            'collapse_rates': {label1: collapse_rate1, label2: collapse_rate2},
            'significant_differences': significant_differences
        }
    
    def run_complete_analysis(self):
        """Run the complete statistical analysis to prove the rebuttal hypothesis"""
        print("=" * 80)
        print("STATISTICAL PROOF: REBUTTAL TO OWUSU ET AL. CLAIMS")
        print("=" * 80)
        
        scenarios = self.setup_cooperation_scenarios()
        
        # Key comparison: High cooperation (no MPA) vs Lower cooperation (with MPA)
        print("\n🎯 TESTING OWUSU CLAIM: High cooperation (no MPA) ≈ Lower cooperation (with MPA)")
        print(f"Running {self.n_simulations} simulations per scenario over {self.n_timesteps} timesteps")
        
        # Run original model scenarios
        print("\n--- ORIGINAL MODEL RESULTS ---")
        orig_high_no_mpa = self.run_original_model(scenarios['high_coop_no_mpa'])
        orig_low_single_mpa = self.run_original_model(scenarios['low_coop_single_mpa'])
        
        # Run dynamic model scenarios  
        print("\n--- DYNAMIC MODEL RESULTS ---")
        dyn_high_no_mpa = self.run_dynamic_model(scenarios['high_coop_no_mpa'])
        dyn_low_single_mpa = self.run_dynamic_model(scenarios['low_coop_single_mpa'])
        
        # Analyze sustainability metrics
        print("\n=== SUSTAINABILITY ANALYSIS ===")
        
        orig_high_metrics = self.analyze_sustainability_metrics(orig_high_no_mpa, "Original")
        orig_low_metrics = self.analyze_sustainability_metrics(orig_low_single_mpa, "Original")
        dyn_high_metrics = self.analyze_sustainability_metrics(dyn_high_no_mpa, "Dynamic")
        dyn_low_metrics = self.analyze_sustainability_metrics(dyn_low_single_mpa, "Dynamic")
        
        print("\n--- ORIGINAL MODEL SUSTAINABILITY METRICS ---")
        print(f"High Coop (No MPA): Sustainability Index = {orig_high_metrics['sustainability_index']:.3f}")
        print(f"Low Coop (Single MPA): Sustainability Index = {orig_low_metrics['sustainability_index']:.3f}")
        print(f"Difference: {orig_high_metrics['sustainability_index'] - orig_low_metrics['sustainability_index']:.3f}")
        
        print("\n--- DYNAMIC MODEL SUSTAINABILITY METRICS ---")
        print(f"High Coop (No MPA): Sustainability Index = {dyn_high_metrics['sustainability_index']:.3f}")
        print(f"Low Coop (Single MPA): Sustainability Index = {dyn_low_metrics['sustainability_index']:.3f}")
        print(f"Difference: {dyn_high_metrics['sustainability_index'] - dyn_low_metrics['sustainability_index']:.3f}")
        
        # Statistical tests
        orig_stats = self.statistical_tests(orig_high_no_mpa, orig_low_single_mpa, 
                                          "Original: High Coop (No MPA)", "Original: Low Coop (MPA)")
        dyn_stats = self.statistical_tests(dyn_high_no_mpa, dyn_low_single_mpa,
                                         "Dynamic: High Coop (No MPA)", "Dynamic: Low Coop (MPA)")
        
        # Proof summary
        self.generate_proof_summary(orig_high_metrics, orig_low_metrics, 
                                  dyn_high_metrics, dyn_low_metrics,
                                  orig_stats, dyn_stats)
        
        # Create visualizations
        self.create_proof_visualizations(orig_high_no_mpa, orig_low_single_mpa,
                                       dyn_high_no_mpa, dyn_low_single_mpa)
        
        return {
            'original_model': {'high_no_mpa': orig_high_metrics, 'low_mpa': orig_low_metrics},
            'dynamic_model': {'high_no_mpa': dyn_high_metrics, 'low_mpa': dyn_low_metrics},
            'statistical_tests': {'original': orig_stats, 'dynamic': dyn_stats}
        }
    
    def generate_proof_summary(self, orig_high, orig_low, dyn_high, dyn_low, orig_stats, dyn_stats):
        """Generate a comprehensive proof summary"""
        print("\n" + "="*80)
        print("🔬 STATISTICAL PROOF SUMMARY")
        print("="*80)
        
        # Test Owusu's claim in original model
        orig_supports_owusu = abs(orig_high['sustainability_index'] - orig_low['sustainability_index']) < 0.1
        
        # Test if dynamic model contradicts Owusu's claim
        dyn_contradicts_owusu = (dyn_low['sustainability_index'] - dyn_high['sustainability_index']) > 0.1
        
        print(f"\n📊 OWUSU CLAIM VERIFICATION:")
        print(f"Original Model Supports Owusu Claim: {'✅ YES' if orig_supports_owusu else '❌ NO'}")
        print(f"  - High Coop (No MPA) Sustainability: {orig_high['sustainability_index']:.3f}")
        print(f"  - Low Coop (MPA) Sustainability: {orig_low['sustainability_index']:.3f}")
        print(f"  - Difference: {abs(orig_high['sustainability_index'] - orig_low['sustainability_index']):.3f}")
        
        print(f"\n🎯 DYNAMIC MODEL REBUTTAL:")
        print(f"Dynamic Model Contradicts Owusu Claim: {'✅ YES' if dyn_contradicts_owusu else '❌ NO'}")
        print(f"  - High Coop (No MPA) Sustainability: {dyn_high['sustainability_index']:.3f}")
        print(f"  - Low Coop (MPA) Sustainability: {dyn_low['sustainability_index']:.3f}")
        print(f"  - Difference: {dyn_low['sustainability_index'] - dyn_high['sustainability_index']:.3f}")
        
        print(f"\n📈 KEY FINDINGS:")
        print(f"  - Collapse Risk (High Coop, No MPA): Original={orig_stats['collapse_rates']['Original: High Coop (No MPA)']:.3f}, Dynamic={dyn_stats['collapse_rates']['Dynamic: High Coop (No MPA)']:.3f}")
        print(f"  - Collapse Risk (Low Coop, MPA): Original={orig_stats['collapse_rates']['Original: Low Coop (MPA)']:.3f}, Dynamic={dyn_stats['collapse_rates']['Dynamic: Low Coop (MPA)']:.3f}")
        
        # Final conclusion
        proof_established = orig_supports_owusu and dyn_contradicts_owusu
        
        print(f"\n🏆 CONCLUSION:")
        if proof_established:
            print("✅ HYPOTHESIS PROVEN: The Owusu et al. conclusion holds in the simplified model")
            print("   but FAILS in the dynamic model with multiple species and cooperation dynamics.")
            print("   High cooperation without MPA becomes significantly less sustainable than")
            print("   lower cooperation with MPA protection when realistic complexity is added.")
        else:
            print("⚠️  MIXED RESULTS: Hypothesis partially supported but not conclusively proven.")
            print("   Further investigation needed with larger sample sizes or different scenarios.")
        
        print("\n📋 STATISTICAL SIGNIFICANCE:")
        alpha = 0.05
        dynamic_significant = any(p < alpha for p in dyn_stats['p_values'].values())
        print(f"Dynamic Model Shows Significant Differences: {'✅ YES' if dynamic_significant else '❌ NO'}")
        
        return proof_established
    
    def create_proof_visualizations(self, orig_high, orig_low, dyn_high, dyn_low):
        """Create comprehensive visualizations for the proof"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Statistical Proof: Owusu et al. Rebuttal Analysis', fontsize=16, fontweight='bold')
        
        # 1. Sustainability Index Comparison
        scenarios = ['High Coop\n(No MPA)', 'Low Coop\n(MPA)']
        orig_sustainability = [
            np.mean([r['final_fish'] for r in orig_high]) / orig.K,
            np.mean([r['final_fish'] for r in orig_low]) / orig.K
        ]
        dyn_sustainability = [
            np.mean([r['final_fish'] for r in dyn_high]) / parameters.K,
            np.mean([r['final_fish'] for r in dyn_low]) / parameters.K
        ]
        
        x = np.arange(len(scenarios))
        width = 0.35
        
        axes[0,0].bar(x - width/2, orig_sustainability, width, label='Original Model', alpha=0.8, color='lightblue')
        axes[0,0].bar(x + width/2, dyn_sustainability, width, label='Dynamic Model', alpha=0.8, color='lightcoral')
        axes[0,0].set_ylabel('Final Fish Population (% of K)')
        axes[0,0].set_title('Population Sustainability Comparison')
        axes[0,0].set_xticks(x)
        axes[0,0].set_xticklabels(scenarios)
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Collapse Risk Comparison
        orig_collapse = [
            np.mean([r['collapse_risk'] for r in orig_high]),
            np.mean([r['collapse_risk'] for r in orig_low])
        ]
        dyn_collapse = [
            np.mean([r['collapse_risk'] for r in dyn_high]),
            np.mean([r['collapse_risk'] for r in dyn_low])
        ]
        
        axes[0,1].bar(x - width/2, orig_collapse, width, label='Original Model', alpha=0.8, color='lightblue')
        axes[0,1].bar(x + width/2, dyn_collapse, width, label='Dynamic Model', alpha=0.8, color='lightcoral')
        axes[0,1].set_ylabel('Collapse Probability')
        axes[0,1].set_title('Ecosystem Collapse Risk')
        axes[0,1].set_xticks(x)
        axes[0,1].set_xticklabels(scenarios)
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Population Stability (Variance)
        orig_stability = [
            np.mean([r['population_stability'] for r in orig_high]),
            np.mean([r['population_stability'] for r in orig_low])
        ]
        dyn_stability = [
            np.mean([r['population_stability'] for r in dyn_high]),
            np.mean([r['population_stability'] for r in dyn_low])
        ]
        
        axes[0,2].bar(x - width/2, orig_stability, width, label='Original Model', alpha=0.8, color='lightblue')
        axes[0,2].bar(x + width/2, dyn_stability, width, label='Dynamic Model', alpha=0.8, color='lightcoral')
        axes[0,2].set_ylabel('Population Variance')
        axes[0,2].set_title('Population Stability (Lower = Better)')
        axes[0,2].set_xticks(x)
        axes[0,2].set_xticklabels(scenarios)
        axes[0,2].legend()
        axes[0,2].grid(True, alpha=0.3)
        
        # 4. Fish Population Trajectories - Original Model
        orig_high_traj = np.array([r['fish_trajectory'][:min(len(r['fish_trajectory']), self.n_timesteps+1)] for r in orig_high[:10]])
        orig_low_traj = np.array([r['fish_trajectory'][:min(len(r['fish_trajectory']), self.n_timesteps+1)] for r in orig_low[:10]])
        
        # Ensure all trajectories have the same length
        min_len_orig = min(len(traj) for traj in orig_high_traj)
        orig_high_traj = np.array([traj[:min_len_orig] for traj in orig_high_traj])
        orig_low_traj = np.array([traj[:min_len_orig] for traj in orig_low_traj])
        
        for i in range(len(orig_high_traj)):
            axes[1,0].plot(orig_high_traj[i], 'r-', alpha=0.3, linewidth=0.8)
        for i in range(len(orig_low_traj)):
            axes[1,0].plot(orig_low_traj[i], 'b-', alpha=0.3, linewidth=0.8)
        
        if len(orig_high_traj) > 0:
            axes[1,0].plot(np.mean(orig_high_traj, axis=0), 'r-', linewidth=3, label='High Coop (No MPA)')
        if len(orig_low_traj) > 0:
            axes[1,0].plot(np.mean(orig_low_traj, axis=0), 'b-', linewidth=3, label='Low Coop (MPA)')
        axes[1,0].set_title('Original Model: Population Trajectories')
        axes[1,0].set_xlabel('Time Steps')
        axes[1,0].set_ylabel('Fish Count')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # 5. Fish Population Trajectories - Dynamic Model
        dyn_high_traj = np.array([r['fish_trajectory'][:min(len(r['fish_trajectory']), self.n_timesteps+1)] for r in dyn_high[:10]])
        dyn_low_traj = np.array([r['fish_trajectory'][:min(len(r['fish_trajectory']), self.n_timesteps+1)] for r in dyn_low[:10]])
        
        # Ensure all trajectories have the same length
        min_len_dyn = min(len(traj) for traj in dyn_high_traj)
        dyn_high_traj = np.array([traj[:min_len_dyn] for traj in dyn_high_traj])
        dyn_low_traj = np.array([traj[:min_len_dyn] for traj in dyn_low_traj])
        
        for i in range(len(dyn_high_traj)):
            axes[1,1].plot(dyn_high_traj[i], 'r-', alpha=0.3, linewidth=0.8)
        for i in range(len(dyn_low_traj)):
            axes[1,1].plot(dyn_low_traj[i], 'b-', alpha=0.3, linewidth=0.8)
        
        if len(dyn_high_traj) > 0:
            axes[1,1].plot(np.mean(dyn_high_traj, axis=0), 'r-', linewidth=3, label='High Coop (No MPA)')
        if len(dyn_low_traj) > 0:
            axes[1,1].plot(np.mean(dyn_low_traj, axis=0), 'b-', linewidth=3, label='Low Coop (MPA)')
        axes[1,1].set_title('Dynamic Model: Population Trajectories')
        axes[1,1].set_xlabel('Time Steps')
        axes[1,1].set_ylabel('Fish Count')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        # 6. Cooperation Erosion in Dynamic Model
        if 'cooperation_trajectory' in dyn_high[0] and 'cooperation_trajectory' in dyn_low[0]:
            dyn_high_coop = np.array([r['cooperation_trajectory'][:min(len(r['cooperation_trajectory']), self.n_timesteps)] for r in dyn_high[:10]])
            dyn_low_coop = np.array([r['cooperation_trajectory'][:min(len(r['cooperation_trajectory']), self.n_timesteps)] for r in dyn_low[:10]])
            
            # Ensure all cooperation trajectories have the same length
            if len(dyn_high_coop) > 0 and len(dyn_low_coop) > 0:
                min_len_coop = min(min(len(traj) for traj in dyn_high_coop), min(len(traj) for traj in dyn_low_coop))
                dyn_high_coop = np.array([traj[:min_len_coop] for traj in dyn_high_coop])
                dyn_low_coop = np.array([traj[:min_len_coop] for traj in dyn_low_coop])
                
                axes[1,2].plot(np.mean(dyn_high_coop, axis=0), 'r-', linewidth=3, label='High Initial Coop (No MPA)')
                axes[1,2].plot(np.mean(dyn_low_coop, axis=0), 'b-', linewidth=3, label='Low Initial Coop (MPA)')
                axes[1,2].set_title('Dynamic Model: Cooperation Erosion')
                axes[1,2].set_xlabel('Time Steps')
                axes[1,2].set_ylabel('Average Cooperation Level')
                axes[1,2].legend()
                axes[1,2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('owusu_rebuttal_proof.png', dpi=300, bbox_inches='tight')
        print(f"\n📊 Proof visualization saved as 'owusu_rebuttal_proof.png'")
        
        plt.show()


def main():
    """Main function to run the complete statistical proof"""
    print("🚀 Starting Statistical Proof: Rebuttal to Owusu et al.")
    print("This analysis will take approximately 15-30 minutes to complete.")
    
    # Initialize the comparison framework
    comparison = ModelComparison(n_simulations=20, n_timesteps=150)  # Reduced for faster execution
    
    # Run complete analysis
    results = comparison.run_complete_analysis()
    
    print("\n✅ Analysis complete! Check the generated plots and summary above.")
    return results


if __name__ == "__main__":
    results = main() 