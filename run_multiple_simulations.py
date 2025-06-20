import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import DynamicCoop as dc

# Configuration parameters
SIMULATION_CONFIG = {
    # Simulation runs
    'num_runs': 100,  # Keep high number of runs for statistical confidence
    'experiment': 'default',
    
    # Trust Parameters
    'initial_trust': 1.0,        # Maximum possible initial trust
    'trust_increase': 0.5,       # Extremely fast trust building
    'trust_decrease': 0.01,      # Extremely slow trust decay
    'trust_radius': 0.8,         # Very large trust radius for better observation
    'trust_memory': 25,          # Much longer trust memory
    'trust_threshold': 0.3,      # Very low threshold to maintain cooperation
    
    # Fisher Numbers - Overwhelming majority of cooperators
    'fully_noncoop': 1,          # Minimal non-cooperators
    'noncoop': 1,
    'cond_coop': 8,             # More conditional cooperators as buffer
    'coop': 15,                 # Large cooperative population
    'fully_coop': 25,           # Dominant fully cooperative population
    
    # MPA Settings
    'mpa_present': 'no',       # Enable MPAs
    'part_time': 'no',          # Full-time MPAs
    'mpa_time': 50,             # Duration if part-time (not used when part_time='no')
    'mpa_type': 'spaced',       # Use spaced MPAs for better coverage
    'mpa_distance': 0.2,        # Optimal distance between MPAs
    'mpa_fraction': 0.5,        # Maximum protected area (50%)
    
    # Other Parameters
    'carrying_capacity': 500,    # Much higher carrying capacity
    'time_steps': 300,          # Keep shorter simulation to see early dynamics
    'base_growth': 0.8,         # Maximum growth rate
    'initial_fish': 500         # Start with full carrying capacity
}

def run_multiple_simulations(num_runs=10, experiment='default'):
    """
    Run multiple simulations and collect statistics
    
    Args:
        num_runs (int): Number of simulation runs
        experiment (str): Experiment configuration name
        
    Returns:
        dict: Collected statistics from all runs
    """
    # Initialize data collection
    all_runs_data = {
        'fish_population': [],
        'fish_in_mpa': [],
        'fish_outside_mpa': [],
        'cooperation_levels': [],
        'trust_levels': [],
        'strategy_counts': {
            'fully_noncoop': [],
            'noncoop': [],
            'cond_coop': [],
            'coop': [],
            'fully_coop': []
        }
    }
    
    # Run simulations
    for run in tqdm(range(num_runs), desc="Running simulations"):
        # Reset global variables for each run
        dc.time1 = 0
        dc.agents = []
        dc.total_fish_count = [dc.K]  # Initialize with carrying capacity
        dc.fish_data_MPA = [0]  # Initialize with 0
        dc.total_hav_data = {}
        dc.current_hav_data = {}
        dc.fishermen_data1 = [0]
        dc.fishermen_data2 = [0]
        dc.fishermen_data3 = [0]  # Initialize with 0
        dc.trust_matrix = {}
        dc.trust_history = []
        dc.cooperation_levels = []
        dc.strategy_counts = {
            'fully_noncoop': [],
            'noncoop': [],
            'cond_coop': [],
            'coop': [],
            'fully_coop': []
        }
        
        # Initialize new simulation
        dc.initialize(experiment)
        
        # Run simulation steps and collect data at each step
        run_data = {
            'fish_population': [dc.K],  # Start with carrying capacity
            'fish_in_mpa': [0],         # Start with 0
            'fish_outside_mpa': [dc.K],  # Start with carrying capacity
            'cooperation_levels': [0.5],  # Start with neutral cooperation
            'trust_levels': [dc.initial_trust],  # Start with initial trust
            'strategy_counts': {
                'fully_noncoop': [dc.fully_noncoop],
                'noncoop': [dc.noncoop],
                'cond_coop': [dc.cond_coop],
                'coop': [dc.coop],
                'fully_coop': [dc.fully_coop]
            }
        }
        
        for step in range(dc.n):
            dc.update_one_unit_time()
            
            # Record data at each step
            run_data['fish_population'].append(len([ag for ag in dc.agents if ag.type == 'fish']))
            
            if dc.MPA == 'yes':
                fish_in_mpa = sum([1 for ag in dc.agents if ag.type == 'fish' and 
                                 ((dc.Xa <= ag.x <= dc.Xb) and (dc.Ya <= ag.y <= dc.Yb))])
            else:
                fish_in_mpa = 0
                
            run_data['fish_in_mpa'].append(fish_in_mpa)
            run_data['fish_outside_mpa'].append(run_data['fish_population'][-1] - fish_in_mpa)
            
            # Record cooperation and trust levels
            fishers = [ag for ag in dc.agents if ag.type == 'fishers']
            avg_cooperation = sum([1 - ag.effort for ag in fishers]) / len(fishers) if fishers else 0.5
            run_data['cooperation_levels'].append(avg_cooperation)
            
            avg_trust = sum([sum(ag.trust_scores.values()) / len(ag.trust_scores) 
                           for ag in fishers if ag.trust_scores]) / len(fishers) if fishers else dc.initial_trust
            run_data['trust_levels'].append(avg_trust)
            
            # Record strategy counts
            strategy_counts = {
                'fully_noncoop': 0,
                'noncoop': 0,
                'cond_coop': 0,
                'coop': 0,
                'fully_coop': 0
            }
            for fisher in fishers:
                strategy_counts[fisher.trait] += 1
            
            for strategy in strategy_counts:
                run_data['strategy_counts'][strategy].append(strategy_counts[strategy])
        
        # Add this run's data to all_runs_data
        all_runs_data['fish_population'].append(run_data['fish_population'])
        all_runs_data['fish_in_mpa'].append(run_data['fish_in_mpa'])
        all_runs_data['fish_outside_mpa'].append(run_data['fish_outside_mpa'])
        all_runs_data['cooperation_levels'].append(run_data['cooperation_levels'])
        all_runs_data['trust_levels'].append(run_data['trust_levels'])
        
        for strategy in all_runs_data['strategy_counts']:
            all_runs_data['strategy_counts'][strategy].append(run_data['strategy_counts'][strategy])
    
    return all_runs_data

def plot_with_ci(x, y_data, label, color):
    """Helper function to plot mean and confidence intervals"""
    mean = np.mean(y_data, axis=0)
    ci = 1.96 * np.std(y_data, axis=0) / np.sqrt(len(y_data))  # 95% CI
    
    plt.plot(x, mean, label=label, color=color)
    plt.fill_between(x, mean - ci, mean + ci, color=color, alpha=0.2)

def plot_statistics(all_runs_data):
    """
    Plot statistics from multiple simulation runs with confidence intervals
    
    Args:
        all_runs_data (dict): Data collected from multiple runs
    """
    plt.figure(figsize=(15, 20))
    
    # Calculate statistics for fish population
    fish_pop_data = np.array(all_runs_data['fish_population'])
    fish_mpa_data = np.array(all_runs_data['fish_in_mpa'])
    fish_outside_data = np.array(all_runs_data['fish_outside_mpa'])
    
    # Create time steps array matching the data length
    time_steps = np.arange(fish_pop_data.shape[1])
    
    # Plot 1: Fish Population
    plt.subplot(4, 1, 1)
    plot_with_ci(time_steps, fish_pop_data, label='Total fish population', color='blue')
    plot_with_ci(time_steps, fish_mpa_data, label='Fish in MPA', color='green')
    plot_with_ci(time_steps, fish_outside_data, label='Fish outside MPA', color='red')
    plt.xlabel('Time Step')
    plt.ylabel('Number of Fish')
    plt.title('Fish Population Dynamics')
    plt.legend()
    
    # Plot 2: Cooperation Levels
    plt.subplot(4, 1, 2)
    coop_data = np.array(all_runs_data['cooperation_levels'])
    coop_time_steps = np.arange(coop_data.shape[1])  # Create specific time steps for cooperation data
    plot_with_ci(coop_time_steps, coop_data, label='Cooperation Level', color='blue')
    plt.xlabel('Time Step')
    plt.ylabel('Cooperation Level')
    plt.title('Evolution of Cooperation')
    plt.legend()
    
    # Plot 3: Strategy Distribution
    plt.subplot(4, 1, 3)
    strategies = ['fully_coop', 'coop', 'cond_coop', 'noncoop', 'fully_noncoop']
    colors = ['darkgreen', 'lightgreen', 'yellow', 'orange', 'red']
    
    for strategy, color in zip(strategies, colors):
        strategy_data = np.array(all_runs_data['strategy_counts'][strategy])
        strategy_time_steps = np.arange(strategy_data.shape[1])  # Create specific time steps for strategy data
        plot_with_ci(strategy_time_steps, strategy_data, label=strategy, color=color)
    
    plt.xlabel('Time Step')
    plt.ylabel('Number of Fishers')
    plt.title('Evolution of Strategies')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    # Plot 4: Trust Levels
    plt.subplot(4, 1, 4)
    trust_data = np.array(all_runs_data['trust_levels'])
    trust_time_steps = np.arange(trust_data.shape[1])  # Create specific time steps for trust data
    plot_with_ci(trust_time_steps, trust_data, label='Trust Level', color='blue')
    plt.xlabel('Time Step')
    plt.ylabel('Trust Level')
    plt.title('Evolution of Trust')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('simulation_output/multiple_runs_statistics.png', bbox_inches='tight', dpi=200)
    plt.close()

def save_statistics(all_runs_data, filename='simulation_output/statistics.csv'):
    """Save statistical data to CSV file"""
    import pandas as pd
    
    # Calculate means and confidence intervals for each metric
    stats = {}
    
    # First determine the length of each data series
    lengths = {
        'fish_pop': len(all_runs_data['fish_population'][0]),
        'fish_mpa': len(all_runs_data['fish_in_mpa'][0]),
        'fish_outside': len(all_runs_data['fish_outside_mpa'][0]),
        'cooperation': len(all_runs_data['cooperation_levels'][0]),
        'trust': len(all_runs_data['trust_levels'][0])
    }
    for strategy in all_runs_data['strategy_counts']:
        lengths[strategy] = len(all_runs_data['strategy_counts'][strategy][0])
    
    # Find the minimum length to truncate all series to the same length
    min_length = min(lengths.values())
    
    # Process fish population data
    fish_pop_data = np.array([data[:min_length] for data in all_runs_data['fish_population']])
    fish_mpa_data = np.array([data[:min_length] for data in all_runs_data['fish_in_mpa']])
    fish_outside_data = np.array([data[:min_length] for data in all_runs_data['fish_outside_mpa']])
    
    stats['time_step'] = np.arange(min_length)
    stats['total_fish_mean'] = np.mean(fish_pop_data, axis=0)
    stats['total_fish_ci'] = 1.96 * np.std(fish_pop_data, axis=0) / np.sqrt(len(fish_pop_data))
    stats['mpa_fish_mean'] = np.mean(fish_mpa_data, axis=0)
    stats['mpa_fish_ci'] = 1.96 * np.std(fish_mpa_data, axis=0) / np.sqrt(len(fish_mpa_data))
    stats['outside_fish_mean'] = np.mean(fish_outside_data, axis=0)
    stats['outside_fish_ci'] = 1.96 * np.std(fish_outside_data, axis=0) / np.sqrt(len(fish_outside_data))
    
    # Process cooperation and trust data
    coop_data = np.array([data[:min_length] for data in all_runs_data['cooperation_levels']])
    trust_data = np.array([data[:min_length] for data in all_runs_data['trust_levels']])
    
    stats['cooperation_mean'] = np.mean(coop_data, axis=0)
    stats['cooperation_ci'] = 1.96 * np.std(coop_data, axis=0) / np.sqrt(len(coop_data))
    stats['trust_mean'] = np.mean(trust_data, axis=0)
    stats['trust_ci'] = 1.96 * np.std(trust_data, axis=0) / np.sqrt(len(trust_data))
    
    # Process strategy counts
    for strategy in all_runs_data['strategy_counts']:
        strategy_data = np.array([data[:min_length] for data in all_runs_data['strategy_counts'][strategy]])
        stats[f'{strategy}_mean'] = np.mean(strategy_data, axis=0)
        stats[f'{strategy}_ci'] = 1.96 * np.std(strategy_data, axis=0) / np.sqrt(len(strategy_data))
    
    # Create DataFrame and save
    df = pd.DataFrame(stats)
    df.to_csv(filename, index=False)

if __name__ == "__main__":
    # Create output directory if it doesn't exist
    os.makedirs('simulation_output', exist_ok=True)
    
    # Apply configuration to DynamicCoop module
    dc.initial_trust = SIMULATION_CONFIG['initial_trust']
    dc.trust_increase = SIMULATION_CONFIG['trust_increase']
    dc.trust_decrease = SIMULATION_CONFIG['trust_decrease']
    dc.trust_radius = SIMULATION_CONFIG['trust_radius']
    dc.trust_memory = SIMULATION_CONFIG['trust_memory']
    dc.trust_threshold = SIMULATION_CONFIG['trust_threshold']
    
    dc.fully_noncoop = SIMULATION_CONFIG['fully_noncoop']
    dc.noncoop = SIMULATION_CONFIG['noncoop']
    dc.cond_coop = SIMULATION_CONFIG['cond_coop']
    dc.coop = SIMULATION_CONFIG['coop']
    dc.fully_coop = SIMULATION_CONFIG['fully_coop']
    dc.num_fishers = sum([dc.fully_noncoop, dc.noncoop, dc.cond_coop, dc.coop, dc.fully_coop])
    
    dc.MPA = SIMULATION_CONFIG['mpa_present']
    dc.Both = SIMULATION_CONFIG['part_time']
    dc.Time_MPA = SIMULATION_CONFIG['mpa_time']
    dc.Type_MPA = SIMULATION_CONFIG['mpa_type']
    dc.Dist_MPA = SIMULATION_CONFIG['mpa_distance']
    dc.Frac_MPA = SIMULATION_CONFIG['mpa_fraction']
    
    dc.K = SIMULATION_CONFIG['carrying_capacity']
    dc.n = SIMULATION_CONFIG['time_steps']
    
    # Run simulations
    print("Running multiple simulations...")
    results = run_multiple_simulations(
        num_runs=SIMULATION_CONFIG['num_runs'],
        experiment=SIMULATION_CONFIG['experiment']
    )
    
    # Generate plots
    print("Generating plots...")
    plot_statistics(results)
    
    # Save statistics
    print("Saving statistics...")
    save_statistics(results)
    
    print(f"Done! Results saved in simulation_output/")
    print(f"Ran {SIMULATION_CONFIG['num_runs']} simulations with {SIMULATION_CONFIG['time_steps']} time steps each.") 