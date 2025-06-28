#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Combined Parameter Equilibrium Analysis Visualizer

This script loads and visualizes the combined results from both the original
parameter sweep (reproduction_rate: 0.05-0.25) and friend's extended sweep
(reproduction_rate: 0.25-0.45).
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import os
import seaborn as sns

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

def load_combined_results(results_dir="simulation_output/parameter_scan_combined"):
    """Load the combined parameter sweep results and analysis."""
    
    # Load combined results
    with open(os.path.join(results_dir, "combined_analysis.json"), 'r') as f:
        analysis = json.load(f)
    
    # Load detailed results
    with open(os.path.join(results_dir, "combined_results.json"), 'r') as f:
        results = json.load(f)
    
    return results, analysis

def create_combined_parameter_space_visualization(results, analysis):
    """Create comprehensive parameter space visualizations for combined dataset."""
    
    # Convert results to DataFrame and add dataset labels
    results_data = []
    for i, r in enumerate(results):
        data_point = {
            **r['params'],
            'equilibrium_reached': r['equilibrium_reached'],
            'equilibrium_value': r['equilibrium_value'] if r['equilibrium_reached'] else 0,
            'status': r.get('status', 'unknown')
        }
        
        # Determine which dataset this point belongs to
        if r['params']['reproduction_rate'] <= 0.25:
            data_point['dataset'] = 'Original (0.05-0.25)'
        else:
            data_point['dataset'] = 'Friend (0.25-0.45)'
            
        results_data.append(data_point)
    
    results_df = pd.DataFrame(results_data)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(24, 16))
    
    # Define parameter names and create color mapping for status and datasets
    param_names = list(analysis['equilibrium_ranges'].keys())
    status_colors = {'equilibrium': 'green', 'growth': 'orange', 'extinction': 'red', 'unstable': 'lightgray'}
    dataset_markers = {'Original (0.05-0.25)': 'o', 'Friend (0.25-0.45)': '^'}
    
    # Plot 1: 3D scatter plot of all three parameters with dataset distinction
    ax1 = fig.add_subplot(3, 4, 1, projection='3d')
    
    for dataset in results_df['dataset'].unique():
        for status in results_df['status'].unique():
            subset = results_df[(results_df['dataset'] == dataset) & (results_df['status'] == status)]
            if len(subset) > 0:
                ax1.scatter(
                    subset['reproduction_rate'],
                    subset['trust_increase'], 
                    subset['imitation_radius'],
                    c=status_colors.get(status, 'gray'),
                    marker=dataset_markers[dataset],
                    label=f'{dataset} - {status}' if status == 'equilibrium' else '',
                    alpha=0.6,
                    s=40 if status == 'equilibrium' else 20
                )
    
    ax1.set_xlabel('Reproduction Rate')
    ax1.set_ylabel('Trust Increase')
    ax1.set_zlabel('Imitation Radius')
    ax1.set_title('3D Parameter Space - Combined Datasets\n(Shape = Dataset, Color = Status)')
    ax1.legend()
    
    # Plot 2-4: 2D projections with dataset distinction
    param_pairs = [
        ('reproduction_rate', 'trust_increase'),
        ('reproduction_rate', 'imitation_radius'),
        ('trust_increase', 'imitation_radius')
    ]
    
    for i, (param1, param2) in enumerate(param_pairs):
        ax = fig.add_subplot(3, 4, i+2)
        
        # Plot by dataset and status
        for dataset in results_df['dataset'].unique():
            for status in results_df['status'].unique():
                subset = results_df[(results_df['dataset'] == dataset) & (results_df['status'] == status)]
                if len(subset) > 0:
                    ax.scatter(
                        subset[param1],
                        subset[param2],
                        c=status_colors.get(status, 'gray'),
                        marker=dataset_markers[dataset],
                        label=f'{dataset} - {status}' if i == 0 and status == 'equilibrium' else "",
                        alpha=0.6,
                        s=60 if status == 'equilibrium' else 20
                    )
        
        ax.set_xlabel(param1.replace('_', ' ').title())
        ax.set_ylabel(param2.replace('_', ' ').title())
        ax.set_title(f'{param1.replace("_", " ").title()} vs {param2.replace("_", " ").title()}\n(Combined Datasets)')
        ax.grid(True, alpha=0.3)
        
        if i == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot 5: Dataset comparison - equilibrium rates
    ax5 = fig.add_subplot(3, 4, 5)
    
    dataset_stats = results_df.groupby(['dataset', 'status']).size().unstack(fill_value=0)
    dataset_eq_rates = (dataset_stats['equilibrium'] / dataset_stats.sum(axis=1)) * 100
    
    bars = ax5.bar(range(len(dataset_eq_rates)), dataset_eq_rates.values, 
                   color=['blue', 'red'], alpha=0.7)
    ax5.set_xticks(range(len(dataset_eq_rates)))
    ax5.set_xticklabels(dataset_eq_rates.index, rotation=45)
    ax5.set_ylabel('Equilibrium Success Rate (%)')
    ax5.set_title('Equilibrium Success Rate by Dataset')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, rate in zip(bars, dataset_eq_rates.values):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2, height + height*0.01,
                f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Plot 6: Reproduction rate effect comparison
    ax6 = fig.add_subplot(3, 4, 6)
    
    # Bin reproduction rates and calculate equilibrium rates
    results_df['repro_bin'] = pd.cut(results_df['reproduction_rate'], bins=20)
    repro_eq_rates = results_df.groupby('repro_bin')['equilibrium_reached'].mean() * 100
    
    bin_centers = [interval.mid for interval in repro_eq_rates.index]
    ax6.plot(bin_centers, repro_eq_rates.values, 'o-', linewidth=2, markersize=6)
    ax6.set_xlabel('Reproduction Rate')
    ax6.set_ylabel('Equilibrium Success Rate (%)')
    ax6.set_title('Equilibrium Success vs Reproduction Rate\n(Combined Data)')
    ax6.grid(True, alpha=0.3)
    
    # Add vertical line at the boundary between datasets
    ax6.axvline(x=0.25, color='red', linestyle='--', alpha=0.7, label='Dataset Boundary')
    ax6.legend()
    
    # Plot 7: Equilibrium values distribution by dataset
    ax7 = fig.add_subplot(3, 4, 7)
    
    eq_data = results_df[results_df['equilibrium_reached']]
    if len(eq_data) > 0:
        for dataset in eq_data['dataset'].unique():
            subset = eq_data[eq_data['dataset'] == dataset]
            ax7.hist(subset['equilibrium_value'], bins=8, alpha=0.7, 
                    label=f'{dataset} (n={len(subset)})', density=True)
        
        ax7.set_xlabel('Equilibrium Fish Count')
        ax7.set_ylabel('Density')
        ax7.set_title('Distribution of Equilibrium Values\nby Dataset')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
    
    # Plot 8: Combined status distribution pie chart
    ax8 = fig.add_subplot(3, 4, 8)
    status_counts = results_df['status'].value_counts()
    colors = [status_colors.get(status, 'gray') for status in status_counts.index]
    
    wedges, texts, autotexts = ax8.pie(
        status_counts.values, 
        labels=[f'{status}\n({count})' for status, count in status_counts.items()],
        autopct='%1.1f%%',
        colors=colors,
        startangle=90
    )
    ax8.set_title('Combined Dataset\nOutcome Distribution')
    
    # Plot 9-12: Individual dataset statistics
    for plot_idx, dataset in enumerate(['Original (0.05-0.25)', 'Friend (0.25-0.45)']):
        ax = fig.add_subplot(3, 4, 9 + plot_idx*2)
        
        dataset_data = results_df[results_df['dataset'] == dataset]
        dataset_status = dataset_data['status'].value_counts()
        
        bars = ax.bar(range(len(dataset_status)), dataset_status.values,
                     color=[status_colors.get(status, 'gray') for status in dataset_status.index])
        ax.set_xticks(range(len(dataset_status)))
        ax.set_xticklabels(dataset_status.index, rotation=45)
        ax.set_ylabel('Count')
        ax.set_title(f'{dataset}\nOutcome Distribution')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, count in zip(bars, dataset_status.values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + height*0.01,
                   f'{count}', ha='center', va='bottom', fontweight='bold')
        
        # Parameter ranges plot
        ax_range = fig.add_subplot(3, 4, 10 + plot_idx*2)
        
        eq_subset = dataset_data[dataset_data['equilibrium_reached']]
        if len(eq_subset) > 0:
            param_ranges_data = []
            for param in param_names:
                param_ranges_data.append({
                    'parameter': param.replace('_', ' ').title(),
                    'min': eq_subset[param].min(),
                    'max': eq_subset[param].max(),
                    'range': eq_subset[param].max() - eq_subset[param].min()
                })
            
            param_df = pd.DataFrame(param_ranges_data)
            bars = ax_range.barh(range(len(param_df)), param_df['range'], 
                                left=param_df['min'], alpha=0.7)
            ax_range.set_yticks(range(len(param_df)))
            ax_range.set_yticklabels(param_df['parameter'])
            ax_range.set_xlabel('Parameter Value')
            ax_range.set_title(f'{dataset}\nEquilibrium Parameter Ranges')
            ax_range.grid(True, alpha=0.3, axis='x')
        else:
            ax_range.text(0.5, 0.5, 'No equilibrium\ncases found', 
                         ha='center', va='center', transform=ax_range.transAxes)
            ax_range.set_title(f'{dataset}\nNo Equilibrium Cases')
    
    plt.tight_layout()
    plt.savefig('simulation_output/parameter_scan_combined/combined_parameter_space.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_combined_radar_plots(results, analysis):
    """Create radar plots for both datasets showing parameter profiles."""
    
    # Get equilibrium results by dataset
    eq_results_original = []
    eq_results_friend = []
    
    for r in results:
        if r['equilibrium_reached']:
            if r['params']['reproduction_rate'] <= 0.25:
                eq_results_original.append(r)
            else:
                eq_results_friend.append(r)
    
    param_names = list(analysis['equilibrium_ranges'].keys())
    
    # Get parameter ranges for normalization (across all data)
    all_results_df = pd.DataFrame([r['params'] for r in results])
    param_mins = {param: all_results_df[param].min() for param in param_names}
    param_maxs = {param: all_results_df[param].max() for param in param_names}
    
    # Create figure
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle('Parameter Profiles: Original vs Friend\'s Extended Dataset', fontsize=16, fontweight='bold')
    
    # Plot 1: Average profiles comparison
    ax1 = fig.add_subplot(2, 3, 1, projection='polar')
    
    if len(eq_results_original) > 0:
        # Calculate average original profile
        orig_df = pd.DataFrame([r['params'] for r in eq_results_original])
        avg_orig_profile = []
        for param in param_names:
            avg_value = orig_df[param].mean()
            norm_value = (avg_value - param_mins[param]) / (param_maxs[param] - param_mins[param])
            avg_orig_profile.append(norm_value)
        
        avg_orig_profile.append(avg_orig_profile[0])  # Close the plot
        
        angles = np.linspace(0, 2 * np.pi, len(param_names), endpoint=False).tolist()
        angles.append(angles[0])
        
        ax1.plot(angles, avg_orig_profile, 'o-', linewidth=3, color='blue', 
                label=f'Original Dataset (n={len(eq_results_original)})')
        ax1.fill(angles, avg_orig_profile, alpha=0.3, color='blue')
    
    if len(eq_results_friend) > 0:
        # Calculate average friend profile
        friend_df = pd.DataFrame([r['params'] for r in eq_results_friend])
        avg_friend_profile = []
        for param in param_names:
            avg_value = friend_df[param].mean()
            norm_value = (avg_value - param_mins[param]) / (param_maxs[param] - param_mins[param])
            avg_friend_profile.append(norm_value)
        
        avg_friend_profile.append(avg_friend_profile[0])  # Close the plot
        
        ax1.plot(angles, avg_friend_profile, 's-', linewidth=3, color='red',
                label=f'Friend Dataset (n={len(eq_results_friend)})')
        ax1.fill(angles, avg_friend_profile, alpha=0.3, color='red')
    
    # Format comparison plot
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels([param.replace('_', '\n').title() for param in param_names])
    ax1.set_ylim(0, 1)
    ax1.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax1.set_yticklabels(['20%', '40%', '60%', '80%', '100%'])
    ax1.grid(True)
    ax1.set_title('Average Parameter Profiles\nOriginal vs Friend Dataset', size=12, weight='bold', pad=20)
    ax1.legend(bbox_to_anchor=(1.3, 1.0), loc='upper left')
    
    # Plot 2: Original dataset individual cases
    if len(eq_results_original) > 0:
        ax2 = fig.add_subplot(2, 3, 2, projection='polar')
        
        colors_orig = plt.cm.Blues(np.linspace(0.4, 1, len(eq_results_original)))
        
        for i, eq_result in enumerate(eq_results_original):
            normalized_values = []
            for param in param_names:
                value = eq_result['params'][param]
                norm_value = (value - param_mins[param]) / (param_maxs[param] - param_mins[param])
                normalized_values.append(norm_value)
            
            normalized_values.append(normalized_values[0])
            
            ax2.plot(angles, normalized_values, 'o-', linewidth=2, 
                    color=colors_orig[i], label=f'Case {i+1} (Fish: {eq_result["equilibrium_value"]:.0f})')
            ax2.fill(angles, normalized_values, alpha=0.1, color=colors_orig[i])
        
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels([param.replace('_', '\n').title() for param in param_names])
        ax2.set_ylim(0, 1)
        ax2.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax2.set_yticklabels(['20%', '40%', '60%', '80%', '100%'])
        ax2.grid(True)
        ax2.set_title('Original Dataset\nIndividual Equilibrium Cases', size=12, weight='bold', pad=20)
        ax2.legend(bbox_to_anchor=(1.3, 1.0), loc='upper left')
    
    # Plot 3: Friend dataset individual cases
    if len(eq_results_friend) > 0:
        ax3 = fig.add_subplot(2, 3, 3, projection='polar')
        
        colors_friend = plt.cm.Reds(np.linspace(0.4, 1, len(eq_results_friend)))
        
        for i, eq_result in enumerate(eq_results_friend):
            normalized_values = []
            for param in param_names:
                value = eq_result['params'][param]
                norm_value = (value - param_mins[param]) / (param_maxs[param] - param_mins[param])
                normalized_values.append(norm_value)
            
            normalized_values.append(normalized_values[0])
            
            ax3.plot(angles, normalized_values, 's-', linewidth=2,
                    color=colors_friend[i], label=f'Case {i+1} (Fish: {eq_result["equilibrium_value"]:.0f})')
            ax3.fill(angles, normalized_values, alpha=0.1, color=colors_friend[i])
        
        ax3.set_xticks(angles[:-1])
        ax3.set_xticklabels([param.replace('_', '\n').title() for param in param_names])
        ax3.set_ylim(0, 1)
        ax3.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax3.set_yticklabels(['20%', '40%', '60%', '80%', '100%'])
        ax3.grid(True)
        ax3.set_title('Friend Dataset\nIndividual Equilibrium Cases', size=12, weight='bold', pad=20)
        ax3.legend(bbox_to_anchor=(1.3, 1.0), loc='upper left')
    
    # Plot 4: Combined best cases
    ax4 = fig.add_subplot(2, 3, 4, projection='polar')
    
    # Get best case from each dataset
    if len(eq_results_original) > 0:
        best_orig = max(eq_results_original, key=lambda x: x['equilibrium_value'])
        normalized_values_orig = []
        for param in param_names:
            value = best_orig['params'][param]
            norm_value = (value - param_mins[param]) / (param_maxs[param] - param_mins[param])
            normalized_values_orig.append(norm_value)
        normalized_values_orig.append(normalized_values_orig[0])
        
        ax4.plot(angles, normalized_values_orig, 'o-', linewidth=3, color='blue',
                label=f'Best Original (Fish: {best_orig["equilibrium_value"]:.0f})')
        ax4.fill(angles, normalized_values_orig, alpha=0.3, color='blue')
    
    if len(eq_results_friend) > 0:
        best_friend = max(eq_results_friend, key=lambda x: x['equilibrium_value'])
        normalized_values_friend = []
        for param in param_names:
            value = best_friend['params'][param]
            norm_value = (value - param_mins[param]) / (param_maxs[param] - param_mins[param])
            normalized_values_friend.append(norm_value)
        normalized_values_friend.append(normalized_values_friend[0])
        
        ax4.plot(angles, normalized_values_friend, 's-', linewidth=3, color='red',
                label=f'Best Friend (Fish: {best_friend["equilibrium_value"]:.0f})')
        ax4.fill(angles, normalized_values_friend, alpha=0.3, color='red')
    
    ax4.set_xticks(angles[:-1])
    ax4.set_xticklabels([param.replace('_', '\n').title() for param in param_names])
    ax4.set_ylim(0, 1)
    ax4.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax4.set_yticklabels(['20%', '40%', '60%', '80%', '100%'])
    ax4.grid(True)
    ax4.set_title('Best Cases Comparison', size=12, weight='bold', pad=20)
    ax4.legend(bbox_to_anchor=(1.3, 1.0), loc='upper left')
    
    # Plot 5-6: Summary statistics
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.axis('off')
    
    summary_text = f"""
    DATASET COMPARISON SUMMARY
    
    Original Dataset (0.05-0.25):
    • Total combinations: {analysis['datasets']['original']['total_combinations']}
    • Equilibrium cases: {analysis['datasets']['original']['equilibrium_count']}
    • Success rate: {(analysis['datasets']['original']['equilibrium_count']/analysis['datasets']['original']['total_combinations']*100):.1f}%
    • Best fish count: {max([r['equilibrium_value'] for r in eq_results_original]) if eq_results_original else 'N/A'}
    
    Friend Dataset (0.25-0.45):
    • Total combinations: {analysis['datasets']['friend']['total_combinations']}
    • Equilibrium cases: {analysis['datasets']['friend']['equilibrium_count']}
    • Success rate: {(analysis['datasets']['friend']['equilibrium_count']/analysis['datasets']['friend']['total_combinations']*100):.1f}%
    • Best fish count: {max([r['equilibrium_value'] for r in eq_results_friend]) if eq_results_friend else 'N/A'}
    
    Combined:
    • Total combinations: {len(results)}
    • Total equilibrium: {analysis['status_counts']['equilibrium']}
    • Overall success rate: {analysis['equilibrium_percentage']:.1f}%
    """
    
    ax5.text(0, 0.9, summary_text, transform=ax5.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace')
    
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    
    insights_text = f"""
    KEY INSIGHTS
    
    • Higher reproduction rates (0.25-0.45) show 
      {(analysis['datasets']['friend']['equilibrium_count']/analysis['datasets']['friend']['total_combinations']*100):.1f}% vs {(analysis['datasets']['original']['equilibrium_count']/analysis['datasets']['original']['total_combinations']*100):.1f}% success rate
    
    • Friend's dataset has {analysis['datasets']['friend']['equilibrium_count']/analysis['datasets']['original']['equilibrium_count']:.1f}x more 
      equilibrium cases
    
    • Both datasets show similar trust_increase 
      and imitation_radius patterns
    
    • Extended reproduction rate range reveals
      sweet spot around 0.25-0.35
    
    • Combined data suggests reproduction rate
      is the critical limiting factor
    
    • Optimal range appears to be 0.25-0.45
      for reproduction rate
    """
    
    ax6.text(0, 0.9, insights_text, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace', color='darkblue')
    
    plt.tight_layout()
    plt.savefig('simulation_output/parameter_scan_combined/combined_radar_plots.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    print("Loading combined parameter equilibrium analysis results...")
    
    # Load combined results
    results, analysis = load_combined_results()
    
    print(f"Loaded {len(results)} total simulation results")
    print(f"Found {analysis['status_counts']['equilibrium']} equilibrium cases ({analysis['equilibrium_percentage']:.1f}%)")
    print(f"Original dataset: {analysis['datasets']['original']['equilibrium_count']} equilibrium cases")
    print(f"Friend dataset: {analysis['datasets']['friend']['equilibrium_count']} equilibrium cases")
    
    # Create comprehensive visualizations
    print("\nCreating combined parameter space visualization...")
    create_combined_parameter_space_visualization(results, analysis)
    
    print("Creating combined radar plots...")
    create_combined_radar_plots(results, analysis)
    
    print("\nVisualization complete! Check the following files:")
    print("• simulation_output/parameter_scan_combined/combined_parameter_space.png")
    print("• simulation_output/parameter_scan_combined/combined_radar_plots.png") 