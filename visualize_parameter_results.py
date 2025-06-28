#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Parameter Equilibrium Analysis Visualizer

This script loads the results from parameter_equilibrium_analyzer.py and creates
comprehensive visualizations of the parameter space exploration.
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

def load_results(results_dir="simulation_output/parameter_scan"):
    """Load the parameter sweep results and analysis."""
    
    # Load main results
    with open(os.path.join(results_dir, "parameter_analysis.json"), 'r') as f:
        analysis = json.load(f)
    
    # Load detailed results
    with open(os.path.join(results_dir, "parameter_sweep_results.json"), 'r') as f:
        results = json.load(f)
    
    return results, analysis

def create_parameter_space_visualization(results, analysis):
    """Create comprehensive parameter space visualizations."""
    
    # Convert results to DataFrame
    results_df = pd.DataFrame([
        {
            **r['params'],
            'equilibrium_reached': r['equilibrium_reached'],
            'equilibrium_value': r['equilibrium_value'] if r['equilibrium_reached'] else 0,
            'status': r.get('status', 'unknown')
        }
        for r in results
    ])
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    
    # Define parameter names and create color mapping for status
    param_names = list(analysis['equilibrium_ranges'].keys())
    status_colors = {'equilibrium': 'green', 'growth': 'orange', 'extinction': 'red', 'unstable': 'lightgray'}
    
    # Plot 1: 3D scatter plot of all three parameters
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    
    for status in results_df['status'].unique():
        status_data = results_df[results_df['status'] == status]
        ax1.scatter(
            status_data['reproduction_rate'],
            status_data['trust_increase'], 
            status_data['imitation_radius'],
            c=status_colors.get(status, 'gray'),
            label=f'{status} ({len(status_data)})',
            alpha=0.6,
            s=30
        )
    
    ax1.set_xlabel('Reproduction Rate')
    ax1.set_ylabel('Trust Increase')
    ax1.set_zlabel('Imitation Radius')
    ax1.set_title('3D Parameter Space\n(Color = Outcome Status)')
    ax1.legend()
    
    # Plot 2-4: 2D projections of parameter pairs
    param_pairs = [
        ('reproduction_rate', 'trust_increase'),
        ('reproduction_rate', 'imitation_radius'),
        ('trust_increase', 'imitation_radius')
    ]
    
    for i, (param1, param2) in enumerate(param_pairs):
        ax = fig.add_subplot(2, 3, i+2)
        
        # Plot all points
        for status in results_df['status'].unique():
            status_data = results_df[results_df['status'] == status]
            ax.scatter(
                status_data[param1],
                status_data[param2],
                c=status_colors.get(status, 'gray'),
                label=f'{status}' if i == 0 else "",  # Only show legend on first plot
                alpha=0.6,
                s=30
            )
        
        # Highlight equilibrium points
        eq_data = results_df[results_df['status'] == 'equilibrium']
        if len(eq_data) > 0:
            ax.scatter(
                eq_data[param1],
                eq_data[param2],
                c='darkgreen',
                s=100,
                marker='*',
                edgecolors='black',
                linewidth=1,
                label='Equilibrium' if i == 0 else ""
            )
        
        ax.set_xlabel(param1.replace('_', ' ').title())
        ax.set_ylabel(param2.replace('_', ' ').title())
        ax.set_title(f'{param1.replace("_", " ").title()} vs {param2.replace("_", " ").title()}')
        ax.grid(True, alpha=0.3)
        
        if i == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot 5: Status distribution pie chart
    ax5 = fig.add_subplot(2, 3, 5)
    status_counts = results_df['status'].value_counts()
    colors = [status_colors.get(status, 'gray') for status in status_counts.index]
    
    wedges, texts, autotexts = ax5.pie(
        status_counts.values, 
        labels=[f'{status}\n({count})' for status, count in status_counts.items()],
        autopct='%1.1f%%',
        colors=colors,
        startangle=90
    )
    ax5.set_title('Distribution of Simulation Outcomes')
    
    # Plot 6: Equilibrium values histogram
    ax6 = fig.add_subplot(2, 3, 6)
    eq_data = results_df[results_df['equilibrium_reached']]
    if len(eq_data) > 0:
        ax6.hist(eq_data['equilibrium_value'], bins=10, color='green', alpha=0.7, edgecolor='black')
        ax6.set_xlabel('Equilibrium Fish Count')
        ax6.set_ylabel('Frequency')
        ax6.set_title(f'Distribution of Equilibrium Values\n(n={len(eq_data)})')
        ax6.grid(True, alpha=0.3)
    else:
        ax6.text(0.5, 0.5, 'No equilibrium\ncombinations found', 
                ha='center', va='center', transform=ax6.transAxes, fontsize=12)
        ax6.set_title('No Equilibrium Values')
    
    plt.tight_layout()
    plt.savefig('simulation_output/parameter_scan/parameter_space_comprehensive.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_parameter_effects_plot(results, analysis):
    """Create detailed parameter effects visualization."""
    
    # Convert results to DataFrame
    results_df = pd.DataFrame([
        {
            **r['params'],
            'status': r.get('status', 'unknown')
        }
        for r in results
    ])
    
    param_names = list(analysis['equilibrium_ranges'].keys())
    
    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    
    for i, param in enumerate(param_names):
        ax = axes[i]
        
        # Calculate status percentages for each parameter value
        param_status = results_df.groupby([param, 'status']).size().unstack(fill_value=0)
        param_status_pct = param_status.div(param_status.sum(axis=1), axis=0) * 100
        
        # Create stacked bar plot
        param_status_pct.plot(kind='bar', stacked=True, ax=ax, 
                             color=[{'equilibrium': 'green', 'growth': 'orange', 
                                   'extinction': 'red', 'unstable': 'lightgray'}.get(col, 'gray') 
                                   for col in param_status_pct.columns])
        
        ax.set_title(f'Simulation Outcomes by {param.replace("_", " ").title()}')
        ax.set_xlabel(param.replace('_', ' ').title())
        ax.set_ylabel('Percentage of Simulations')
        ax.legend(title='Outcome', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Rotate x-axis labels for better readability
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('simulation_output/parameter_scan/parameter_effects.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_summary_dashboard(results, analysis):
    """Create a summary dashboard with key statistics."""
    
    fig = plt.figure(figsize=(16, 10))
    
    # Main title
    fig.suptitle('Parameter Equilibrium Analysis - Summary Dashboard', fontsize=16, fontweight='bold')
    
    # Key statistics text
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.axis('off')
    
    stats_text = f"""
    KEY FINDINGS
    
    Total Combinations Tested: {len(results):,}
    
    Equilibrium Found: {analysis['status_counts']['equilibrium']} ({analysis['equilibrium_percentage']:.1f}%)
    Growth Patterns: {analysis['status_counts'].get('growth', 0)}
    Unstable Systems: {analysis['status_counts']['unstable']:,}
    
    Best Equilibrium:
    • Fish Count: {analysis['best_parameters']['equilibrium_value']:.1f}
    • Time to Equilibrium: {analysis['best_parameters']['time_to_equilibrium']:.0f} steps
    
    Optimal Parameters:
    • Reproduction Rate: {analysis['best_parameters']['reproduction_rate']:.3f}
    • Trust Increase: {analysis['best_parameters']['trust_increase']:.3f}
    • Imitation Radius: {analysis['best_parameters']['imitation_radius']:.3f}
    """
    
    ax1.text(0, 0.9, stats_text, transform=ax1.transAxes, fontsize=11, 
             verticalalignment='top', fontfamily='monospace')
    
    # Parameter ranges visualization
    ax2 = fig.add_subplot(2, 3, 2)
    param_names = list(analysis['equilibrium_ranges'].keys())
    equilibrium_ranges = analysis['equilibrium_ranges']
    
    y_pos = np.arange(len(param_names))
    mins = [equilibrium_ranges[param]['min'] for param in param_names]
    maxs = [equilibrium_ranges[param]['max'] for param in param_names]
    ranges = [maxs[i] - mins[i] for i in range(len(param_names))]
    
    bars = ax2.barh(y_pos, ranges, left=mins, color='green', alpha=0.7)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([param.replace('_', ' ').title() for param in param_names])
    ax2.set_xlabel('Parameter Value')
    ax2.set_title('Parameter Ranges Leading to Equilibrium')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax2.text(bar.get_x() + width/2, bar.get_y() + bar.get_height()/2,
                f'{mins[i]:.3f} - {maxs[i]:.3f}', 
                ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Status distribution
    ax3 = fig.add_subplot(2, 3, 3)
    status_counts = analysis['status_counts']
    colors = {'equilibrium': 'green', 'growth': 'orange', 'unstable': 'lightgray'}
    
    bars = ax3.bar(range(len(status_counts)), list(status_counts.values()),
                   color=[colors.get(status, 'gray') for status in status_counts.keys()])
    ax3.set_xticks(range(len(status_counts)))
    ax3.set_xticklabels(list(status_counts.keys()), rotation=45)
    ax3.set_ylabel('Number of Simulations')
    ax3.set_title('Distribution of Outcomes')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, count in zip(bars, status_counts.values()):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2, height + height*0.01,
                f'{count}', ha='center', va='bottom', fontweight='bold')
    
    # Parameter correlation heatmap (for equilibrium cases only)
    ax4 = fig.add_subplot(2, 3, 4)
    
    # Get equilibrium results
    eq_results = [r for r in results if r['equilibrium_reached']]
    if len(eq_results) > 1:
        eq_df = pd.DataFrame([r['params'] for r in eq_results])
        corr_matrix = eq_df.corr()
        
        im = ax4.imshow(corr_matrix.values, cmap='RdBu_r', vmin=-1, vmax=1)
        ax4.set_xticks(range(len(param_names)))
        ax4.set_yticks(range(len(param_names)))
        ax4.set_xticklabels([param.replace('_', '\n') for param in param_names], fontsize=9)
        ax4.set_yticklabels([param.replace('_', '\n') for param in param_names], fontsize=9)
        ax4.set_title('Parameter Correlations\n(Equilibrium Cases Only)')
        
        # Add correlation values
        for i in range(len(param_names)):
            for j in range(len(param_names)):
                text = ax4.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                               ha="center", va="center", color="black", fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax4, shrink=0.8)
        cbar.set_label('Correlation Coefficient')
    else:
        ax4.text(0.5, 0.5, 'Insufficient equilibrium\ncases for correlation', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Parameter Correlations')
    
    # Runtime and efficiency stats
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.axis('off')
    
    success_rate = analysis['equilibrium_percentage']
    efficiency_text = f"""
    ANALYSIS EFFICIENCY
    
    Success Rate: {success_rate:.1f}%
    
    Parameter Space Coverage:
    • Reproduction Rate: {len(equilibrium_ranges['reproduction_rate']['values'])} successful values
    • Trust Increase: {len(equilibrium_ranges['trust_increase']['values'])} successful values  
    • Imitation Radius: {len(equilibrium_ranges['imitation_radius']['values'])} successful values
    
    Recommendations:
    {"• Focus on higher reproduction rates (>0.1)" if success_rate < 5 else "• Current ranges are effective"}
    {"• Increase trust parameter sampling" if len(equilibrium_ranges['trust_increase']['values']) < 3 else "• Trust range well covered"}
    {"• Test wider imitation radius" if len(equilibrium_ranges['imitation_radius']['values']) < 3 else "• Imitation range adequate"}
    """
    
    ax5.text(0, 0.9, efficiency_text, transform=ax5.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace')
    
    # Empty space for additional info
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    
    notes_text = f"""
    METHODOLOGY NOTES
    
    • Total simulation time: 350 steps
    • Warm-up period: 100 steps
    • Equilibrium detection window: 40 steps
    • Stability threshold: CV < 5%
    • Trend threshold: < 5% change
    
    • Parameter grid: 12×12×12 = 1,728 combinations
    • Estimated runtime: ~12-14 hours
    • Parallel processing: 4 cores
    
    Output saved to:
    simulation_output/parameter_scan/
    """
    
    ax6.text(0, 0.9, notes_text, transform=ax6.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace', color='darkblue')
    
    plt.tight_layout()
    plt.savefig('simulation_output/parameter_scan/summary_dashboard.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_radar_plots(results, analysis):
    """Create radar plots showing parameter profiles for equilibrium cases."""
    
    # Get equilibrium results
    eq_results = [r for r in results if r['equilibrium_reached']]
    
    if len(eq_results) == 0:
        print("No equilibrium cases found for radar plot")
        return
    
    # Extract parameter names and normalize values to 0-1 scale for radar plot
    param_names = list(analysis['equilibrium_ranges'].keys())
    
    # Get parameter ranges for normalization
    all_results_df = pd.DataFrame([r['params'] for r in results])
    param_mins = {param: all_results_df[param].min() for param in param_names}
    param_maxs = {param: all_results_df[param].max() for param in param_names}
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    
    # Plot 1: Individual radar plots for each equilibrium case
    n_eq = len(eq_results)
    n_cols = min(3, n_eq)
    n_rows = (n_eq + n_cols - 1) // n_cols
    
    for i, eq_result in enumerate(eq_results):
        ax = fig.add_subplot(n_rows + 1, n_cols, i + 1, projection='polar')
        
        # Normalize parameter values to 0-1 scale
        normalized_values = []
        for param in param_names:
            value = eq_result['params'][param]
            norm_value = (value - param_mins[param]) / (param_maxs[param] - param_mins[param])
            normalized_values.append(norm_value)
        
        # Add first value at end to close the radar plot
        normalized_values.append(normalized_values[0])
        
        # Calculate angles for each parameter
        angles = np.linspace(0, 2 * np.pi, len(param_names), endpoint=False).tolist()
        angles.append(angles[0])  # Close the plot
        
        # Plot the radar chart
        ax.plot(angles, normalized_values, 'o-', linewidth=2, label=f'Case {i+1}')
        ax.fill(angles, normalized_values, alpha=0.25)
        
        # Add parameter labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([param.replace('_', '\n').title() for param in param_names])
        
        # Set radial limits
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'])
        ax.grid(True)
        
        # Add title with parameter values and fish count
        title_text = f"Equilibrium Case {i+1}\nFish Count: {eq_result['equilibrium_value']:.0f}"
        ax.set_title(title_text, size=10, weight='bold', pad=20)
        
        # Add parameter values as text
        param_text = "\n".join([f"{param.replace('_', ' ').title()}: {eq_result['params'][param]:.3f}" 
                               for param in param_names])
        ax.text(1.3, 0.5, param_text, transform=ax.transAxes, fontsize=8, 
                verticalalignment='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
    
    # Plot 2: Overlay of all equilibrium cases
    ax_overlay = fig.add_subplot(n_rows + 1, n_cols, n_cols * n_rows + 1, projection='polar')
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(eq_results)))
    
    for i, eq_result in enumerate(eq_results):
        # Normalize parameter values
        normalized_values = []
        for param in param_names:
            value = eq_result['params'][param]
            norm_value = (value - param_mins[param]) / (param_maxs[param] - param_mins[param])
            normalized_values.append(norm_value)
        
        normalized_values.append(normalized_values[0])  # Close the plot
        
        # Calculate angles
        angles = np.linspace(0, 2 * np.pi, len(param_names), endpoint=False).tolist()
        angles.append(angles[0])
        
        # Plot
        ax_overlay.plot(angles, normalized_values, 'o-', linewidth=2, 
                       color=colors[i], label=f'Case {i+1} (Fish: {eq_result["equilibrium_value"]:.0f})')
        ax_overlay.fill(angles, normalized_values, alpha=0.1, color=colors[i])
    
    # Format overlay plot
    ax_overlay.set_xticks(angles[:-1])
    ax_overlay.set_xticklabels([param.replace('_', '\n').title() for param in param_names])
    ax_overlay.set_ylim(0, 1)
    ax_overlay.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax_overlay.set_yticklabels(['20%', '40%', '60%', '80%', '100%'])
    ax_overlay.grid(True)
    ax_overlay.set_title('All Equilibrium Cases Overlaid', size=12, weight='bold', pad=20)
    ax_overlay.legend(bbox_to_anchor=(1.3, 1.0), loc='upper left')
    
    # Plot 3: Average equilibrium profile vs failed cases
    if n_cols >= 2:
        ax_comparison = fig.add_subplot(n_rows + 1, n_cols, n_cols * n_rows + 2, projection='polar')
        
        # Calculate average equilibrium profile
        eq_df = pd.DataFrame([r['params'] for r in eq_results])
        avg_eq_profile = []
        for param in param_names:
            avg_value = eq_df[param].mean()
            norm_value = (avg_value - param_mins[param]) / (param_maxs[param] - param_mins[param])
            avg_eq_profile.append(norm_value)
        
        # Calculate average failed profile (sample of unstable cases)
        failed_results = [r for r in results if not r['equilibrium_reached']]
        failed_sample = np.random.choice(failed_results, min(50, len(failed_results)), replace=False)
        failed_df = pd.DataFrame([r['params'] for r in failed_sample])
        avg_failed_profile = []
        for param in param_names:
            avg_value = failed_df[param].mean()
            norm_value = (avg_value - param_mins[param]) / (param_maxs[param] - param_mins[param])
            avg_failed_profile.append(norm_value)
        
        # Close the plots
        avg_eq_profile.append(avg_eq_profile[0])
        avg_failed_profile.append(avg_failed_profile[0])
        
        angles = np.linspace(0, 2 * np.pi, len(param_names), endpoint=False).tolist()
        angles.append(angles[0])
        
        # Plot both profiles
        ax_comparison.plot(angles, avg_eq_profile, 'o-', linewidth=3, color='green', label='Equilibrium Average')
        ax_comparison.fill(angles, avg_eq_profile, alpha=0.3, color='green')
        
        ax_comparison.plot(angles, avg_failed_profile, 'o-', linewidth=3, color='red', label='Failed Average')
        ax_comparison.fill(angles, avg_failed_profile, alpha=0.3, color='red')
        
        # Format comparison plot
        ax_comparison.set_xticks(angles[:-1])
        ax_comparison.set_xticklabels([param.replace('_', '\n').title() for param in param_names])
        ax_comparison.set_ylim(0, 1)
        ax_comparison.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax_comparison.set_yticklabels(['20%', '40%', '60%', '80%', '100%'])
        ax_comparison.grid(True)
        ax_comparison.set_title('Equilibrium vs Failed Cases\n(Average Profiles)', size=12, weight='bold', pad=20)
        ax_comparison.legend(bbox_to_anchor=(1.2, 1.0), loc='upper left')
    
    plt.tight_layout()
    plt.savefig('simulation_output/parameter_scan/radar_plots.png', dpi=300, bbox_inches='tight')
    plt.show()

def display_existing_plots():
    """Display the existing trajectory and parameter plots."""
    
    plot_files = [
        'simulation_output/parameter_scan/trajectory_plots/equilibrium_trajectories.png',
        'simulation_output/parameter_scan/trajectory_plots/non_equilibrium_trajectories.png',
        'simulation_output/parameter_scan/parameter_plots/reproduction_rate_vs_trust_increase.png',
        'simulation_output/parameter_scan/parameter_plots/reproduction_rate_vs_imitation_radius.png',
        'simulation_output/parameter_scan/parameter_plots/trust_increase_vs_imitation_radius.png'
    ]
    
    print("Existing visualizations:")
    for plot_file in plot_files:
        if os.path.exists(plot_file):
            print(f"✓ {plot_file}")
        else:
            print(f"✗ {plot_file} (not found)")

if __name__ == "__main__":
    print("Loading parameter equilibrium analysis results...")
    
    # Load results
    results, analysis = load_results()
    
    print(f"Loaded {len(results)} simulation results")
    print(f"Found {analysis['status_counts']['equilibrium']} equilibrium cases ({analysis['equilibrium_percentage']:.1f}%)")
    
    # Create comprehensive visualizations
    print("\nCreating parameter space visualization...")
    create_parameter_space_visualization(results, analysis)
    
    print("Creating parameter effects plot...")
    create_parameter_effects_plot(results, analysis)
    
    print("Creating summary dashboard...")
    create_summary_dashboard(results, analysis)
    
    print("Creating radar plots...")
    create_radar_plots(results, analysis)
    
    print("\nChecking existing visualizations...")
    display_existing_plots()
    
    print("\nVisualization complete! Check the following files:")
    print("• simulation_output/parameter_scan/parameter_space_comprehensive.png")
    print("• simulation_output/parameter_scan/parameter_effects.png") 
    print("• simulation_output/parameter_scan/summary_dashboard.png")
    print("• simulation_output/parameter_scan/radar_plots.png")
    print("• simulation_output/parameter_scan/trajectory_plots/")
    print("• simulation_output/parameter_scan/parameter_plots/") 