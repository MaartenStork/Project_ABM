# -*- coding: utf-8 -*-
#--------------------------------------------------------------------------------------

# Agent-Based Model (ABM) representative of an idealised small-scale, artisanal fishery. 
# The model has been developed to investigate the combined effects of fishing behaviour 
# (expressed by a cooperative trait associated to fishing effort) and different designs 
# of Marine Protected Areas (age, size, and distance between two MPAs).

# Original Model by : OWUSU, Kwabena Afriyie
# Adapted Model by : Maarten, Mikko, Yoad and Lucas
# Date : 30 June, 2025

#---------------------------------------------------------------------------

# Import relevent libraries #
from pylab import *
import copy as cp
import random as rd
import math
import numpy as np
import matplotlib.pyplot as plt
import csv 
from statistics import mean
from itertools import product
from fish import fish_experiment
from create_gif import create_gif
import os
import glob
from tqdm import tqdm


from parameters import *
#######################################################################################################################################################  

class agent:  # create an empty class
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    
#----------------------------------------------------------------------------------------------------------    

def delete_prev_sim():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sim_output_dir = os.path.join(script_dir, "simulation_output")
    pattern = os.path.join(sim_output_dir, "year_*.png")

    # Iterate and remove matching files
    for file_path in glob.glob(pattern):
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")


def initialize(experiment):
    delete_prev_sim()

    global time1, agents, fish, total_fish_count, species_count, fish_data_MPA, total_hav_data, current_hav_data, fishermen, fishermen_data1, fishermen_data2, fishermen_data3
    time1 = 0.
    agents = []
    total_fish_count = [K]
    species_count = {}
    total_hav_data = {}
    current_hav_data = {}
    fishermen_data1 = [0]
    fishermen_data2 = [0]
    
    # Initialize trust matrix
    global trust_matrix, trust_history
    trust_matrix = {}
    trust_history = []
    
    # First create all fisher agents to get their IDs
    fisher_agents = []
    for j in range(num_fishers):
        ag = agent()
        ag.type = 'fishers'
        ag.harvest = 0
        ag.low_density_memory = 0
        ag.base_effort = 1.0
        ag.trust_scores = {}
        ag.cooperative_actions = 0
        ag.total_actions = 0
        
        if j < (fully_noncoop):
            ag.effort = 1.0
            ag.trait = 'fully_noncoop'
            ag.num = f'fully_noncoop{1 + j}'
        elif (fully_noncoop) <= j < (fully_noncoop + noncoop):
            ag.effort = 0.8
            ag.trait = 'noncoop'
            ag.num = f'noncoop{(1 + j) - fully_noncoop}'
        elif (fully_noncoop + noncoop) <= j < (fully_noncoop + noncoop + cond_coop):
            ag.effort = 0.6
            ag.trait = 'cond_coop'
            ag.num = f'cond_coop{(1 + j) - (fully_noncoop + noncoop)}'
        elif (fully_noncoop + noncoop + cond_coop) <= j < (fully_noncoop + noncoop + cond_coop + coop):
            ag.effort = 0.4
            ag.trait = 'coop'
            ag.num = f'coop{1 + j - (fully_noncoop + noncoop + cond_coop)}'
        else:
            ag.effort = 0.2
            ag.trait = 'fully_coop'
            ag.num = f'fully_coop{1 + j - (fully_noncoop + noncoop + cond_coop + coop)}'
        
        ag.base_effort = ag.effort
        total_hav_data[ag.num] = [ag.harvest]
        current_hav_data[ag.num] = [ag.harvest]
        
        # Initialize position based on MPA configuration
        if (MPA == 'no' and Both == 'no'):  # only no MPA
            ag.x = rd.uniform(-Half_Length_Area, Half_Length_Area)
            ag.y = rd.uniform(-Half_Length_Area, Half_Length_Area)
        
        elif any([(MPA == 'yes' and Type_MPA == 'single' and Both == 'no'),
                 (MPA == 'no' and Both == 'yes' and Type_MPA == 'single')]):  # single MPA
            while True:
                ag.x = rd.uniform(-Half_Length_Area, Half_Length_Area)
                ag.y = rd.uniform(-Half_Length_Area, Half_Length_Area)
                if not((Xa <= ag.x <= Xb) and (Ya <= ag.y <= Yb)):
                    break
        
        elif any([(MPA == 'yes' and Type_MPA == 'spaced' and Both == 'no'),
                 (MPA == 'no' and Both == 'yes' and Type_MPA == 'spaced')]):  # spaced MPA
            while True:
                ag.x = rd.uniform(-Half_Length_Area, Half_Length_Area)
                ag.y = rd.uniform(-Half_Length_Area, Half_Length_Area)
                if all([not((Xm <= ag.x <= Xn) and (Ym <= ag.y <= Yn)),
                       not((Xp <= ag.x <= Xq) and (Yp <= ag.y <= Yq))]):
                    break
        
        fisher_agents.append(ag)
    
    # Now initialize trust scores between all fishers
    for fisher in fisher_agents:
        for other in fisher_agents:
            if fisher != other:
                fisher.trust_scores[other.num] = initial_trust
    
    # Add fishers to agents list
    agents.extend(fisher_agents)

    init_fish_agents(experiment)
#----------------------------------------------------------------------------------------------------------    
                                
    # Initialise the number of fishes in an MPA 
    if (MPA == 'no' and Both == 'no') :
        fish_data_MPA = [0] #  a zero because no mpa is available
        
    elif any([(MPA == 'yes' and Type_MPA == 'single' and Both == 'no'),(MPA == 'no' and Both == 'yes' and Type_MPA == 'single')]): # only single MPA , partly single MPA & partly no MPA
        fish_data_MPA = [sum([1 for j in agents if j.type == 'fish' and  ((Xa <= j.x <= Xb) and (Ya <= j.y <= Yb))])]
        
    elif any([(MPA == 'yes' and Type_MPA == 'spaced' and Both == 'no'), (MPA == 'no' and Both == 'yes' and Type_MPA == 'spaced')]):
        fish_data_MPA = [sum([1 for j in agents if j.type == 'fish' and any([((Xm <= j.x <= Xn) and (Ym <= j.y <= Yn)), ((Xp <= j.x <= Xq) and (Yp <= j.y <= Yq))])])]
    
    fishermen_data3 = [total_fish_count[-1] - fish_data_MPA[-1]] # initialise number of fishes outside MPA


def init_fish_agents(experiment):
    fish_params = fish_experiment(experiment, K)
    param_keys = sorted([k for k in fish_params if k != 'carrying_capacity'])
    param_values = [fish_params[k] for k in param_keys]

    combinations = list(product(*param_values))
    num_combinations = len(combinations)
    num_fish_per_combo = fish_params['carrying_capacity'] // num_combinations

    for combo in combinations:
        school_id = 0
        params = dict(zip(param_keys, combo))
        subtype_str = "_".join(f"{k}={v}" for k, v in params.items())
        species_count[subtype_str] = [num_fish_per_combo]
        for _ in range(num_fish_per_combo):
            ag = agent(**params)
            ag.type = 'fish'
            ag.school = school_id
            ag.x = rd.uniform(-Half_Length_Area, Half_Length_Area)
            ag.y = rd.uniform(-Half_Length_Area, Half_Length_Area)
            ag.subtype = subtype_str
            agents.append(ag)
        school_id += 1
    
######################################################################################################################################################    
        
def observe():
    # plt.ioff()
    global time1, agents
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_facecolor('lightskyblue')
    fishermen = [ag for ag in agents if ag.type == 'fishers']

    if len(fishermen) > 0:
        X_fully_noncoop = [ag.x for ag in fishermen if ag.trait == 'fully_noncoop']
        Y_fully_noncoop = [ag.y for ag in fishermen if ag.trait == 'fully_noncoop']
        X_noncoop = [ag.x for ag in fishermen if ag.trait == 'noncoop']
        Y_noncoop = [ag.y for ag in fishermen if ag.trait == 'noncoop']
        X_cond_coop = [ag.x for ag in fishermen if ag.trait == 'cond_coop']
        Y_cond_coop = [ag.y for ag in fishermen if ag.trait == 'cond_coop']
        X_coop = [ag.x for ag in fishermen if ag.trait == 'coop']
        Y_coop = [ag.y for ag in fishermen if ag.trait == 'coop']
        X_fully_coop = [ag.x for ag in fishermen if ag.trait == 'fully_coop']
        Y_fully_coop  = [ag.y for ag in fishermen if ag.trait == 'fully_coop']
        colors = np.linspace(0, 1, 5)
        mymap = plt.get_cmap("Greys")
        my_colors = mymap(colors)
        ax.plot(X_fully_coop, Y_fully_coop, 'o', color=my_colors[4], markersize=7.5, label='fully_coop')
        ax.plot(X_coop, Y_coop, 'o', color=my_colors[3], markersize=7.5, label='coop')
        ax.plot(X_cond_coop, Y_cond_coop, 'o', color=my_colors[2], markersize=7.5, label='conditional_coop')
        ax.plot(X_noncoop, Y_noncoop, 'o', color=my_colors[1], markersize=7.5, label='noncoop')
        ax.plot(X_fully_noncoop, Y_fully_noncoop, 'o', color=my_colors[0], markersize=7.5, label='fully_noncoop')

    fish = [ag for ag in agents if ag.type == 'fish']
    if len(fish) > 0:
        subtypes = list(set([ag.subtype for ag in fish]))
        subtypes.sort()  # optional: for consistent ordering
        cmap = plt.colormaps.get_cmap('viridis')
        colors = [cmap(i / len(subtypes)) for i in range(len(subtypes))]
        for i, subtype in enumerate(subtypes):
            subtype_fish = [ag for ag in fish if ag.subtype == subtype]
            X = [ag.x for ag in subtype_fish]
            Y = [ag.y for ag in subtype_fish]
            ax.plot(X, Y, '^', color=colors[i], markersize=3, label=f'fish {subtype}')
    if any([(MPA == 'yes' and Type_MPA == 'single' and Both == 'no'), (MPA == 'no' and Both == 'yes' and Type_MPA =='single' and time1 <= Time_MPA)]):
        ax.vlines(Xa, Ya, Yb, lw=2, color='k')
        ax.vlines(Xb, Ya, Yb, lw=2, color='k')
        ax.hlines(Ya, Xa, Xb, lw=2, color='k')
        ax.hlines(Yb, Xa, Xb, lw=2, color='k')
    elif any([(MPA == 'yes' and Type_MPA == 'spaced' and Both == 'no'), (MPA == 'no' and Both == 'yes' and Type_MPA =='spaced' and time1 <= Time_MPA)]):
        ax.vlines(Xm, Ym, Yn, lw=2, color='k')
        ax.vlines(Xn, Ym, Yn, lw=2, color='k')
        ax.hlines(Ym, Xm, Xn, lw=2, color='k')
        ax.hlines(Yn, Xm, Xn, lw=2, color='k')
        ax.vlines(Xp, Yp, Yq, lw=2, color='k')
        ax.vlines(Xq, Yp, Yq, lw=2, color='k')
        ax.hlines(Yp, Xp, Xq, lw=2, color='k')
        ax.hlines(Yq, Xp, Xq, lw=2, color='k')
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim([-Half_Length_Area, Half_Length_Area])
    ax.set_ylim([-Half_Length_Area, Half_Length_Area])
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f'year = {int(time1)}')
    ax.legend(numpoints=1, loc='center', bbox_to_anchor=(0.5, -0.072), ncol=3, prop={'size': 11},
              facecolor='lightskyblue')

    fig.savefig(f'simulation_output/year_{int(time1):04d}.png', bbox_inches='tight', pad_inches=0, dpi=200)
    plt.close(fig)

def plot_summary():
    global time1, agents, total_fish_count, species_count, fish_data_MPA, fishermen_data1, fishermen_data2, fishermen_data3
    plt.figure(figsize=(15, 25))
    
    # Plot 1: Fish population dynamics
    plt.subplot(5, 1, 1)
    plt.plot(total_fish_count, 'b-', label='Total fish population')
    plt.plot(fish_data_MPA, 'g-', label='Fish in MPA')
    plt.plot(fishermen_data3, 'r-', label='Fish outside MPA')
    plt.xlabel('Time')
    plt.ylabel('Number of fish')
    plt.title('Fish Population Dynamics')
    plt.legend()

    # Plot 2: Fish population dynamics per species
    plt.subplot(5, 1, 2)
    for species, counts in species_count.items():
        plt.plot(counts, label=f'Species: {species}')

    plt.xlabel('Time')
    plt.ylabel('Number of fish')
    plt.title('Fish Population Dynamics')
    plt.legend()

    # Plot 3: Fishing activity
    plt.subplot(5, 1, 3)
    plt.plot(fishermen_data1, 'b-', label='Total catch')
    plt.plot(fishermen_data2, 'r-', label='Current catch')
    plt.xlabel('Time')
    plt.ylabel('Number of fish caught')
    plt.title('Fishing Activity')
    plt.legend()
    
    # Plot 4: Cooperation levels
    plt.subplot(5, 1, 4)
    time_steps = range(len(cooperation_levels))
    
    # Plot strategy counts
    plt.stackplot(time_steps, 
                 [strategy_counts['fully_coop'], 
                  strategy_counts['coop'],
                  strategy_counts['cond_coop'],
                  strategy_counts['noncoop'],
                  strategy_counts['fully_noncoop']],
                 labels=['Fully Cooperative', 'Cooperative', 'Conditionally Cooperative', 
                        'Non-cooperative', 'Fully Non-cooperative'],
                 colors=['darkgreen', 'lightgreen', 'yellow', 'orange', 'red'])
    
    # Plot average cooperation level
    plt.plot(time_steps, cooperation_levels, 'k--', label='Average Cooperation Level', linewidth=2)
    
    plt.xlabel('Time')
    plt.ylabel('Number of Fishers / Cooperation Level')
    plt.title('Evolution of Cooperation Strategies')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    # Plot 5: Trust dynamics
    plt.subplot(5, 1, 5)
    plt.plot(trust_history, 'b-', label='Average Trust')
    plt.xlabel('Time')
    plt.ylabel('Trust Level')
    plt.title('Evolution of Trust Between Fishers')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('simulation_output/dynamics.png', bbox_inches='tight', dpi=200)
    plt.close()
    
    # Save cooperation evolution plot separately
    plt.figure(figsize=(10, 6))
    plt.stackplot(time_steps, 
                 [strategy_counts['fully_coop'], 
                  strategy_counts['coop'],
                  strategy_counts['cond_coop'],
                  strategy_counts['noncoop'],
                  strategy_counts['fully_noncoop']],
                 labels=['Fully Cooperative', 'Cooperative', 'Conditionally Cooperative', 
                        'Non-cooperative', 'Fully Non-cooperative'],
                 colors=['darkgreen', 'lightgreen', 'yellow', 'orange', 'red'])
    plt.plot(time_steps, cooperation_levels, 'k--', label='Average Cooperation Level', linewidth=2)
    plt.xlabel('Time')
    plt.ylabel('Number of Fishers / Cooperation Level')
    plt.title('Evolution of Cooperation Strategies')
    plt.legend(loc='center left', bbox_to_anchor=(1.15, 0.5))
    plt.tight_layout()
    plt.savefig('simulation_output/cooperation_evolution.png', bbox_inches='tight', dpi=200)
    plt.close('all')

def save_cooperation_data():
    with open('simulation_output/cooperation_data.csv', 'w') as f:
        writer = csv.writer(f)
        # Write header
        header = ['time', 'avg_cooperation'] + list(strategy_counts.keys())
        writer.writerow(header)
        
        # Write data
        for t in range(len(cooperation_levels)):
            row = [t, cooperation_levels[t]] + [strategy_counts[s][t] for s in strategy_counts]
            writer.writerow(row)

###################################################################################################################################################### 

def update_fish():
    """Updates fish positions using NumPy vectorized operations for maximum performance."""
    global time1, agents, fish, total_fish_count, species_count, fish_data_MPA, total_hav_data, current_hav_data, fishermen, fishermen_data1, fishermen_data2, fishermen_data3

    # Only get fish agents once
    fish_list = [j for j in agents if j.type == 'fish']
    if not fish_list:
        return

    # Initialize new_fish counter
    new_fish = 0
    new_species_count = {}
    for fish in fish_list:
        if fish.subtype not in new_species_count:
            new_species_count[fish.subtype] = 1
        else:
            new_species_count[fish.subtype] += 1

        repulsion = [
            nb for nb in agents
            if nb.type == 'fish'
               and nb != fish
               and nb.school == fish.school
               and ((fish.x - nb.x) ** 2 + (fish.y - nb.y) ** 2) < rad_repulsion_sqr
        ]
        alignment = [
            nb for nb in agents
            if nb.type == 'fish'
               and nb != fish
               and nb.school == fish.school
               and rad_repulsion_sqr < ((fish.x - nb.x) ** 2 + (fish_ag.y - nb.y) ** 2) < rad_orientation_sqr
        ]
        attraction = [
            nb for nb in agents
            if nb.type == 'fish'
               and nb != fish
               and nb.school == fish.school
               and rad_orientation_sqr < ((fish.x - nb.x) ** 2 + (fish.y - nb.y) ** 2) < rad_attraction_sqr
        ]

        if len(repulsion) > 0: # if fishes within repulsion zone, move away from the spot that would be the center of mass (midpoint) of all  fish within repulsion zone
            repulsion_x = mean([j.x for j in repulsion])
            repulsion_y = mean([j.y for j in repulsion])
            repulsion_1 = (math.atan2((repulsion_y - fish.y), (repulsion_x - fish.x)) + math.pi ) % (2 * math.pi) # if greater than  (2 * math.pi) then compute with a minus
            theta = repulsion_1
            fish.x +=  fish.speed*math.cos(theta)     # moves 'move_fish' step
            fish.y +=  fish.speed*math.sin(theta)
            fish.x = (fish.x % -Half_Length_Area) if fish.x > Half_Length_Area else (fish.x % Half_Length_Area) if fish.x < -Half_Length_Area else fish.x  # ( When fish-agent approach a border of the landscape,
            fish.y = (fish.y % -Half_Length_Area) if fish.y > Half_Length_Area else (fish.y % Half_Length_Area) if fish.y < -Half_Length_Area else fish.y  # they re-enter the system at the opposite border )

        elif all([len(repulsion) == 0, len(alignment) > 0]):   # if fishes within parallel-orientation zone, change direction to match the average direction of all the other fish  within parallel-orientation zone
            alignment_1 = mean([math.atan2((j.y - fish_ag.y),(j.x - fish_ag.x)) for j in alignment])
            theta = alignment_1
            fish.x +=   math.cos(theta)     # moves 'move_fish' step,  move_fish*math.cos(theta)
            fish.y +=   math.sin(theta)
            fish.x = (fish.x % -Half_Length_Area) if fish.x > Half_Length_Area else (fish_ag.x % Half_Length_Area) if fish_ag.x < -Half_Length_Area else fish_ag.x  # ( When fish-agent approach a border of the landscape,
            fish.y = (fish.y % -Half_Length_Area) if fish.y > Half_Length_Area else (fish_ag.y % Half_Length_Area) if fish_ag.y < -Half_Length_Area else fish_ag.y  # they re-enter the system at the opposite border )

        elif all([len(repulsion) == 0, len(alignment) == 0, len(attraction) > 0]): # if fishes within only the attraction zone, head towards the middle (midpoint) of the fishes in zone of attraction.
            attraction_x = mean([j.x for j in attraction ])
            attraction_y = mean([j.y for j in attraction])
            attraction_1 = math.atan2((attraction_y - fish_ag.y), (attraction_x - fish_ag.x))
            theta = attraction_1
            fish.x +=  fish_ag.speed*math.cos(theta)     # moves 'move_fish' step
            fish.y +=  fish_ag.speed*math.sin(theta)
            fish.x = (fish.x % -Half_Length_Area) if fish.x > Half_Length_Area else (fish.x % Half_Length_Area) if fish.x < -Half_Length_Area else fish.x  # ( When fish-agent approach a border of the landscape,
            fish.y = (fish.y % -Half_Length_Area) if fish.y > Half_Length_Area else (fish.y % Half_Length_Area) if fish.y < -Half_Length_Area else fish.y  # they re-enter the system at the opposite border )

        elif all([len(repulsion) == 0, len(alignment) == 0, len(attraction) == 0]): # if no fishes in all the zone, move in a random direction
            theta = 2*math.pi*rd.random()
            fish.x +=  fish.speed*math.cos(theta)     # moves 'move_fish' step
            fish.y +=  fish.speed*math.sin(theta)
            fish.x = (fish.x % -Half_Length_Area) if fish.x > Half_Length_Area else (fish.x % Half_Length_Area) if fish.x < -Half_Length_Area else fish.x  # ( When fish-agent approach a border of the landscape,
            fish.y = (fish.y % -Half_Length_Area) if fish.y > Half_Length_Area else (fish.y % Half_Length_Area) if fish.y < -Half_Length_Area else fish.y  # they re-enter the system at the opposite border )

        if total_fish_count[-1] + new_fish < total_fish_count[0] and rd.random() < fish_ag.reproduction_rate * (1-sum([1 for j in agents if j.type == 'fish'])/float(K)):  # logistic growth of fishes
            agents.append(cp.copy(fish_ag)) # add-copy of fish agent
            new_fish += 1
            if fish_ag.subtype in new_species_count:
                new_species_count[fish_ag.subtype] += 1
            else:
                new_species_count[fish_ag.subtype] = 1

    total_fish_count.append(len(fish_list) + new_fish)
    for species in species_count:
        if species in new_species_count:
            species_count[species].append(new_species_count[species])
        else:
            species_count[species].append(0)


def new_func():
    # Use NumPy for processing - pre-collect positions for vectorized calculations
    for school, school_fish in fish_by_school.items():
        # Skip schools with only one fish
        if len(school_fish) <= 1:
            continue

        # Extract all positions for this school into NumPy arrays
        positions = np.array([(fish.x, fish.y) for fish in school_fish])
        speeds = np.array([fish.speed for fish in school_fish])

        # Process each fish in this school using vectorized operations
        for i, fish_ag in enumerate(school_fish):
            # Get positions of all other fish in the same school (excluding self)
            other_indices = np.ones(len(school_fish), dtype=bool)
            other_indices[i] = False
            other_positions = positions[other_indices]

            # Calculate distances all at once using broadcasting
            fish_pos = np.array([fish_ag.x, fish_ag.y])
            differences = other_positions - fish_pos
            squared_distances = np.sum(differences**2, axis=1)

            # Identify fish in each zone using boolean masks
            repulsion_mask = squared_distances < rad_repulsion_sqr
            alignment_mask = (squared_distances >= rad_repulsion_sqr) & (squared_distances < rad_orientation_sqr)
            attraction_mask = (squared_distances >= rad_orientation_sqr) & (squared_distances < rad_attraction_sqr)

            # Get indices of fish in each zone
            other_fish_indices = np.where(other_indices)[0]
            repulsion_indices = other_fish_indices[repulsion_mask]
            alignment_indices = other_fish_indices[alignment_mask]
            attraction_indices = other_fish_indices[attraction_mask]

            # Extract positions for calculations
            repulsion_positions = positions[repulsion_indices] if len(repulsion_indices) > 0 else np.array([])
            alignment_positions = positions[alignment_indices] if len(alignment_indices) > 0 else np.array([])
            attraction_positions = positions[attraction_indices] if len(attraction_indices) > 0 else np.array([])

            # Determine movement based on which zones have fish
            if len(repulsion_positions) > 0:  # Repulsion takes priority
                # Calculate center of mass vectorized
                repulsion_center = np.mean(repulsion_positions, axis=0)

                # Calculate angle to move away from center of mass
                theta = (np.arctan2((repulsion_center[1] - fish_ag.y),
                                    (repulsion_center[0] - fish_ag.x)) + np.pi) % (2 * np.pi)

            elif len(alignment_positions) > 0:  # Then alignment
                # Calculate average direction of neighbors - vectorized
                rel_positions = alignment_positions - fish_pos
                angles = np.arctan2(rel_positions[:, 1], rel_positions[:, 0])
                theta = np.mean(angles)

            elif len(attraction_positions) > 0:  # Then attraction
                # Calculate center of mass for attraction zone - vectorized
                attraction_center = np.mean(attraction_positions, axis=0)

                # Calculate angle to move toward center of mass
                theta = np.arctan2((attraction_center[1] - fish_ag.y),
                                   (attraction_center[0] - fish_ag.x))

            else:  # If no neighbors, move randomly
                theta = 2 * np.pi * rd.random()

            # Move fish according to calculated direction
            if len(alignment_positions) > 0 and len(repulsion_positions) == 0 and len(attraction_positions) == 0:
                # Special case for alignment only
                fish_ag.x += np.cos(theta)
                fish_ag.y += np.sin(theta)
            else:
                fish_ag.x += fish_ag.speed * np.cos(theta)
                fish_ag.y += np.sin(theta) * fish_ag.speed

            # Handle boundary conditions using vectorized approach when possible
            if fish_ag.x > Half_Length_Area:
                fish_ag.x %= -Half_Length_Area
            elif fish_ag.x < -Half_Length_Area:
                fish_ag.x %= Half_Length_Area

            if fish_ag.y > Half_Length_Area:
                fish_ag.y %= -Half_Length_Area
            elif fish_ag.y < -Half_Length_Area:
                fish_ag.y %= Half_Length_Area

            # Implement logistic growth for fish reproduction
            total_fish_with_new = current_fish_count + new_fish
            if total_fish_with_new < total_fish_count[0] and \
                    rd.random() < fish_ag.reproduction_rate * (1 - total_fish_with_new / float(K)):
                agents.append(cp.copy(fish_ag))
                new_fish += 1

    # Handle fish that are the only ones in their school
    for school, school_fish in fish_by_school.items():
        if len(school_fish) == 1:
            fish_ag = school_fish[0]
            # Random movement for isolated fish
            theta = 2 * np.pi * rd.random()
            fish_ag.x += fish_ag.speed * np.cos(theta)
            fish_ag.y += fish_ag.speed * np.sin(theta)

            # Handle boundaries
            if fish_ag.x > Half_Length_Area:
                fish_ag.x %= -Half_Length_Area
            elif fish_ag.x < -Half_Length_Area:
                fish_ag.x %= Half_Length_Area

            if fish_ag.y > Half_Length_Area:
                fish_ag.y %= -Half_Length_Area
            elif fish_ag.y < -Half_Length_Area:
                fish_ag.y %= Half_Length_Area

            # Check reproduction
            total_fish_with_new = current_fish_count + new_fish
            if total_fish_with_new < total_fish_count[0] and \
                    rd.random() < fish_ag.reproduction_rate * (1 - total_fish_with_new / float(K)):
                agents.append(cp.copy(fish_ag))
                new_fish += 1

    # Update total fish count - appending current count, not just new fish
    total_fish_count.append(current_fish_count + new_fish)

######################################################################################################################################################                         
                  
def no_mpa():
    
    global time1, agents, fish, total_fish_count, species_count, fish_data_MPA, total_hav_data, current_hav_data, fishermen, fishermen_data1,  fishermen_data2, fishermen_data3
    fisherman_ag = rd.sample([j for j in agents if j.type == 'fishers'],1)[-1] # randomly sample a fisherman 
    
    fish_neighbors = [nb for nb in agents if nb.type == 'fish' and ((fisherman_ag.x - nb.x)**2 + (fisherman_ag.y - nb.y)**2) < r_sqr ] # detecting fishes in neighbourhood
    num_fish_harvest = int(round(q * fisherman_ag.effort * len(fish_neighbors))) # number of fish to be harvested based on (q*E*x), where x is number of fishes in neighborhood 
    if fish_neighbors and num_fish_harvest > 0:
        sample_fish_harvest = rd.sample(fish_neighbors, min(num_fish_harvest, len(fish_neighbors)))
        for j in sample_fish_harvest:
            agents.remove(j)  # remove catch  
            fisherman_ag.harvest += 1  # add to catch of a fisherman

    fishers_neighbors = [[nb.harvest, nb] for nb in agents if nb.type == 'fishers' and nb != fisherman_ag and ((fisherman_ag.x - nb.x)**2 + (fisherman_ag.y - nb.y)**2) < r_sqr] # detecting fishermen in neighbourhood 
    fishers_neighbors_harvest = sorted(fishers_neighbors, key=lambda HAV: HAV[0]) # sort fishermen in neighborhood according to catch
    if len(fishers_neighbors_harvest) == 0: # if there exist no fisherman in neighbourhood
        theta_1 = 2*math.pi*rd.random()
        fisherman_ag.x +=  move_fishers*math.cos(theta_1) # move  'move_fishers' step in a random direction
        fisherman_ag.y +=  move_fishers*math.sin(theta_1) 
        fisherman_ag.x = -Half_Length_Area if fisherman_ag.x > Half_Length_Area else  Half_Length_Area if fisherman_ag.x < -Half_Length_Area else fisherman_ag.x
        fisherman_ag.y = -Half_Length_Area if fisherman_ag.y > Half_Length_Area else  Half_Length_Area if fisherman_ag.y < -Half_Length_Area else fisherman_ag.y
    elif all([len(fishers_neighbors_harvest) > 0, fishers_neighbors_harvest[-1][0] > fisherman_ag.harvest]) : # if there exist fisherman with greater catch than focal fisherman 
            deltax = fishers_neighbors_harvest[-1][-1].x - fisherman_ag.x   #move in the direction of one with greater catch than focal fisherman 
            deltay = fishers_neighbors_harvest[-1][-1].y - fisherman_ag.y 
            theta_2 = math.atan2(deltay,deltax) 
            fisherman_ag.x +=  move_fishers*math.cos(theta_2) # move 'move_fishers' in the direction of neighbour fishermen with greatest catch
            fisherman_ag.y +=  move_fishers*math.sin(theta_2) 
            fisherman_ag.x = -Half_Length_Area if fisherman_ag.x > Half_Length_Area else  Half_Length_Area if fisherman_ag.x < -Half_Length_Area else fisherman_ag.x
            fisherman_ag.y = -Half_Length_Area if fisherman_ag.y > Half_Length_Area else  Half_Length_Area if fisherman_ag.y < -Half_Length_Area else fisherman_ag.y
    else: # if all fisherman have less or equal catch relativelly  to focal fisherman
            theta_2 = 2*math.pi*rd.random()
            fisherman_ag.x +=  move_fishers*math.cos(theta_2) # move  'move_fishers' step in a random direction
            fisherman_ag.y +=  move_fishers*math.sin(theta_2) 
            fisherman_ag.x = -Half_Length_Area if fisherman_ag.x > Half_Length_Area else  Half_Length_Area if fisherman_ag.x < -Half_Length_Area else fisherman_ag.x
            fisherman_ag.y = -Half_Length_Area if fisherman_ag.y > Half_Length_Area else  Half_Length_Area if fisherman_ag.y < -Half_Length_Area else fisherman_ag.y

   
###################################################################################################################################################### 

def single_mpa():
    
    global time1, agents, fish, total_fish_count, species_count, fish_data_MPA, total_hav_data, current_hav_data, fishermen, fishermen_data1,  fishermen_data2, fishermen_data3
    fisherman_ag = rd.sample([j for j in agents if j.type == 'fishers'],1)[-1]   #randomly select a fisherman
    
    fish_neighbors = [nb for nb in agents if nb.type == 'fish' and ((fisherman_ag.x - nb.x)**2 + (fisherman_ag.y - nb.y)**2) < r_sqr 
        and not((Xa <= nb.x <= Xb) and (Ya <= nb.y <= Yb))] # detecting fishes in neighbourhood and outside MPA
    num_fish_harvest = int(round(q * fisherman_ag.effort * len(fish_neighbors))) # number of fish catch based on (q*E*x), where x is fishes in neighborhood  and outside MPA
    if fish_neighbors and num_fish_harvest > 0:
        sample_fish_harvest = rd.sample(fish_neighbors, min(num_fish_harvest, len(fish_neighbors)))
        for j in sample_fish_harvest:
            agents.remove(j)  # remove fish catch
            fisherman_ag.harvest += 1  # add to catch of fisherman
    
    fishers_neighbors = [[nb.harvest, nb] for nb in agents if nb.type =='fishers' and nb != fisherman_ag  and ((fisherman_ag.x - nb.x)**2 + (fisherman_ag.y - nb.y)**2) < r_sqr] # detecting fishermen in neighbourhood 
    fishers_neighbors_harvest = sorted(fishers_neighbors, key=lambda HAV: HAV[0]) # sort fishermen in neighborhood according to catch
    if len(fishers_neighbors_harvest) == 0 : # if there exist no fisherman in neighbourhood:
        theta_empt1 = 0 ; theta_empt2 = 0
        while True: 
            theta_1 = 2*math.pi*rd.random()
            fisherman_ag.x +=  move_fishers*math.cos(theta_1) - theta_empt1  # move  'move_fishers' step in a random direction
            fisherman_ag.y +=  move_fishers*math.sin(theta_1) - theta_empt2 
            theta_empt1 = move_fishers*math.cos(theta_1) ; theta_empt2 = move_fishers*math.sin(theta_1)
            if not((Xa <= fisherman_ag.x <= Xb) and (Ya <= fisherman_ag.y <= Yb)):
                fisherman_ag.x = -Half_Length_Area if fisherman_ag.x > Half_Length_Area else  Half_Length_Area if fisherman_ag.x < -Half_Length_Area else fisherman_ag.x
                fisherman_ag.y = -Half_Length_Area if fisherman_ag.y > Half_Length_Area else  Half_Length_Area if fisherman_ag.y < -Half_Length_Area else fisherman_ag.y
                break
    elif all([len(fishers_neighbors_harvest) > 0, fishers_neighbors_harvest[-1][0] > fisherman_ag.harvest])  : # if there exist a fisherman in neighbourhood with greatest catch than focal fisherman
        deltax = fishers_neighbors_harvest[-1][-1].x - fisherman_ag.x   #move in the direction of one with greatest catch
        deltay = fishers_neighbors_harvest[-1][-1].y - fisherman_ag.y 
        theta_2 = math.atan2(deltay,deltax) 
        if not((Xa <= (fisherman_ag.x + move_fishers*math.cos(theta_2)) <= Xb) and (Ya <= (fisherman_ag.y + move_fishers*math.sin(theta_2)) <= Yb)):  # if updating  movement does not fall in MPA
            fisherman_ag.x +=  move_fishers*math.cos(theta_2) # move 'move_fishers' in the direction of neighbour fishermen with greatest catch 
            fisherman_ag.y +=  move_fishers*math.sin(theta_2) 
            fisherman_ag.x = -Half_Length_Area if fisherman_ag.x > Half_Length_Area else  Half_Length_Area if fisherman_ag.x < -Half_Length_Area else fisherman_ag.x
            fisherman_ag.y = -Half_Length_Area if fisherman_ag.y > Half_Length_Area else  Half_Length_Area if fisherman_ag.y < -Half_Length_Area else fisherman_ag.y
        else:  # in case moving in this direction lands you on an MPA, move in a random direction
            theta_empt1 = 0 ; theta_empt2 = 0
            while True: 
                theta_2 = 2*math.pi*rd.random()
                fisherman_ag.x +=  move_fishers*math.cos(theta_2) - theta_empt1  # move  'move_fishers' step in a random direction
                fisherman_ag.y +=  move_fishers*math.sin(theta_2) - theta_empt2 
                theta_empt1 = move_fishers*math.cos(theta_2) ; theta_empt2 = move_fishers*math.sin(theta_2)
                if not((Xa <= fisherman_ag.x <= Xb) and (Ya <= fisherman_ag.y <= Yb)):
                    fisherman_ag.x = -Half_Length_Area if fisherman_ag.x > Half_Length_Area else  Half_Length_Area if fisherman_ag.x < -Half_Length_Area else fisherman_ag.x
                    fisherman_ag.y = -Half_Length_Area if fisherman_ag.y > Half_Length_Area else  Half_Length_Area if fisherman_ag.y < -Half_Length_Area else fisherman_ag.y
                    break
    else:  # if all fisherman in neighbourhood have less or equal catch compared to focal fisherman
        theta_empt1 = 0 ; theta_empt2 = 0
        while True: 
            theta_2 = 2*math.pi*rd.random()
            fisherman_ag.x +=  move_fishers*math.cos(theta_2) - theta_empt1  # move  'move_fishers' step in a random direction
            fisherman_ag.y +=  move_fishers*math.sin(theta_2) - theta_empt2 
            theta_empt1 = move_fishers*math.cos(theta_2) ; theta_empt2 = move_fishers*math.sin(theta_2)
            if not((Xa <= fisherman_ag.x <= Xb) and (Ya <= fisherman_ag.y <= Yb)):
                fisherman_ag.x = -Half_Length_Area if fisherman_ag.x > Half_Length_Area else  Half_Length_Area if fisherman_ag.x < -Half_Length_Area else fisherman_ag.x
                fisherman_ag.y = -Half_Length_Area if fisherman_ag.y > Half_Length_Area else  Half_Length_Area if fisherman_ag.y < -Half_Length_Area else fisherman_ag.y
                break
                            
######################################################################################################################################################                                

def spaced_mpa():
    
    global time1, agents, fish, total_fish_count, species_count, fish_data_MPA, total_hav_data, current_hav_data, fishermen , fishermen_data1,  fishermen_data2, fishermen_data3
    fisherman_ag = rd.sample([j for j in agents if j.type == 'fishers'],1)[-1]    #randomly select an fisherman agent
    
    fish_neighbors = [nb for nb in agents if nb.type == 'fish' and ((fisherman_ag.x - nb.x)**2 + (fisherman_ag.y - nb.y)**2) < r_sqr and  all([not((Xm <= nb.x <= Xn) and (Ym <= nb.y <= Yn)), not((Xp <= nb.x <= Xq) and (Yp <= nb.y <= Yq))])] # detecting fishes in neighbourhood
    num_fish_harvest = int(round(q * fisherman_ag.effort * len(fish_neighbors))) # number of fish catch based on (q*E*x), where x is number of fishes in neighborhood 
    if fish_neighbors and num_fish_harvest > 0:
        sample_fish_harvest = rd.sample(fish_neighbors, min(num_fish_harvest, len(fish_neighbors)))
        for j in sample_fish_harvest:
            agents.remove(j)  # remove fish catch
            fisherman_ag.harvest += 1  # add to fish catch
    
    fishers_neighbors = [[nb.harvest, nb] for nb in agents if nb.type == 'fishers' and nb != fisherman_ag and ((fisherman_ag.x - nb.x)**2 + (fisherman_ag.y - nb.y)**2) < r_sqr] # detecting fishermen in neighbourhood 
    fishers_neighbors_harvest = sorted(fishers_neighbors, key=lambda HAV: HAV[0]) # sort fishermen in neighborhood according to catch
    if len(fishers_neighbors_harvest) == 0 : # if there are no fisherman in neighbourhood 
        theta_empt1 = 0 ; theta_empt2 = 0
        while True: 
            theta_1 = 2*math.pi*rd.random()
            fisherman_ag.x +=  move_fishers*math.cos(theta_1) - theta_empt1  # move  'move_fishers' step in a random direction
            fisherman_ag.y +=  move_fishers*math.sin(theta_1) - theta_empt2 
            theta_empt1 = move_fishers*math.cos(theta_1) ; theta_empt2 = move_fishers*math.sin(theta_1)
            if all([not((Xm <= fisherman_ag.x <= Xn) and (Ym <= fisherman_ag.y <= Yn)), not((Xp <= fisherman_ag.x <= Xq) and (Yp <= fisherman_ag.y <= Yq))]):
                    fisherman_ag.x = -Half_Length_Area if fisherman_ag.x > Half_Length_Area else  Half_Length_Area if fisherman_ag.x < -Half_Length_Area else fisherman_ag.x
                    fisherman_ag.y = -Half_Length_Area if fisherman_ag.y > Half_Length_Area else  Half_Length_Area if fisherman_ag.y < -Half_Length_Area else fisherman_ag.y
                    break
    elif all([len(fishers_neighbors_harvest) > 0, fishers_neighbors_harvest[-1][0] > fisherman_ag.harvest]) : # if there exist fisherman in neighbourhood with greatest catch than focal fisherman 
        deltax = fishers_neighbors_harvest[-1][-1].x - fisherman_ag.x   # move in the direction of the fisherman with greatest catch 
        deltay = fishers_neighbors_harvest[-1][-1].y - fisherman_ag.y 
        theta_2 = math.atan2(deltay,deltax) 
        if all([not((Xm <= (fisherman_ag.x + move_fishers*math.cos(theta_2)) <= Xn) and (Ym <= (fisherman_ag.y + move_fishers*math.sin(theta_2)) <= Yn)), not((Xp <= (fisherman_ag.x + move_fishers*math.cos(theta_2) <= Xq)) and (Yp <= (fisherman_ag.y + move_fishers*math.sin(theta_2)) <= Yq))]):
            fisherman_ag.x +=  move_fishers*math.cos(theta_2) # move 'move_fishers' in the direction of neighbour fishermen with greater harvest 
            fisherman_ag.y +=  move_fishers*math.sin(theta_2) 
            fisherman_ag.x = -Half_Length_Area if fisherman_ag.x > Half_Length_Area else  Half_Length_Area if fisherman_ag.x < -Half_Length_Area else fisherman_ag.x
            fisherman_ag.y = -Half_Length_Area if fisherman_ag.y > Half_Length_Area else  Half_Length_Area if fisherman_ag.y < -Half_Length_Area else fisherman_ag.y
        else:  # in the case this paths lands you on an MPA, move in a random direction
            theta_empt1 = 0 ; theta_empt2 = 0
            while True: 
                theta_2 = 2*math.pi*rd.random()
                fisherman_ag.x +=  move_fishers*math.cos(theta_2) - theta_empt1  # move  'move_fishers' step in a random direction
                fisherman_ag.y +=  move_fishers*math.sin(theta_2) - theta_empt2 
                theta_empt1 = move_fishers*math.cos(theta_2) ; theta_empt2 = move_fishers*math.sin(theta_2)
                if all([not((Xm <= fisherman_ag.x <= Xn) and (Ym <= fisherman_ag.y <= Yn)), not((Xp <= fisherman_ag.x <= Xq) and (Yp <= fisherman_ag.y <= Yq))]):
                    fisherman_ag.x = -Half_Length_Area if fisherman_ag.x > Half_Length_Area else  Half_Length_Area if fisherman_ag.x < -Half_Length_Area else fisherman_ag.x
                    fisherman_ag.y = -Half_Length_Area if fisherman_ag.y > Half_Length_Area else  Half_Length_Area if fisherman_ag.y < -Half_Length_Area else fisherman_ag.y
                    break
    else:  # if there exist fisherman in neighbourhood with less or equal catch compared to focal fisherman 
        theta_empt1 = 0 ; theta_empt2 = 0
        while True: 
            theta_2 = 2*math.pi*rd.random()
            fisherman_ag.x +=  move_fishers*math.cos(theta_2) - theta_empt1  # move  'move_fishers' step in a random direction
            fisherman_ag.y +=  move_fishers*math.sin(theta_2) - theta_empt2 
            theta_empt1 = move_fishers*math.cos(theta_2) ; theta_empt2 = move_fishers*math.sin(theta_2)
            if all([not((Xm <= fisherman_ag.x <= Xn) and (Ym <= fisherman_ag.y <= Yn)), not((Xp <= fisherman_ag.x <= Xq) and (Yp <= fisherman_ag.y <= Yq))]):
                fisherman_ag.x = -Half_Length_Area if fisherman_ag.x > Half_Length_Area else  Half_Length_Area if fisherman_ag.x < -Half_Length_Area else fisherman_ag.x
                fisherman_ag.y = -Half_Length_Area if fisherman_ag.y > Half_Length_Area else  Half_Length_Area if fisherman_ag.y < -Half_Length_Area else fisherman_ag.y
                break
   
######################################################################################################################################################                                 

def imitate_successful_strategies():
    """Allow fishers to imitate more successful strategies from their neighbors.
    Vectorized using NumPy for maximum performance.
    """
    global agents
    
    # Get all fishers once
    fishers = [ag for ag in agents if ag.type == 'fishers']
    if len(fishers) <= 1:  # Nothing to imitate with 0 or 1 fishers
        return

    # Extract fisher positions and harvests into NumPy arrays for vectorized operations
    positions = np.array([(fisher.x, fisher.y) for fisher in fishers])
    harvests = np.array([fisher.harvest for fisher in fishers])

    # Pre-calculate the squared imitation radius
    imitation_radius_squared = imitation_radius**2

    # For each fisher - we still need to loop through individual fishers
    # because we need to modify their attributes
    for i, fisher in enumerate(fishers):
        # Create a mask for all fishers except the current one
        other_fishers_mask = np.ones(len(fishers), dtype=bool)
        other_fishers_mask[i] = False
        
        # Calculate distances to all other fishers at once
        fisher_pos = positions[i]
        other_positions = positions[other_fishers_mask]
        differences = other_positions - fisher_pos
        distances_squared = np.sum(differences**2, axis=1)

        # Create mask for fishers within imitation radius
        within_radius = distances_squared < imitation_radius_squared

        # Get harvests of fishers within radius
        neighbor_harvests = harvests[other_fishers_mask][within_radius]

        # If no neighbors within radius or no neighbors with better harvest, continue
        if len(neighbor_harvests) == 0 or np.max(neighbor_harvests) <= fisher.harvest:
            continue

        # Find the most successful neighbor within imitation radius
        max_harvest_idx = np.argmax(neighbor_harvests)

        # Get the actual index in the original fishers list
        # First get indices of fishers that are both other fishers and within radius
        valid_indices = np.where(other_fishers_mask)[0][within_radius]
        most_successful_idx = valid_indices[max_harvest_idx]
        most_successful = fishers[most_successful_idx]

        # Imitate with probability imitation_prob
        if rd.random() < imitation_prob:
            # Copy effort level and trait
            fisher.effort = most_successful.effort
            fisher.trait = most_successful.trait

def track_cooperation_levels():
    """Track the current levels of cooperation strategies."""
    global agents
    
    # Count strategies
    counts = {
        'fully_noncoop': 0,
        'noncoop': 0,
        'cond_coop': 0,
        'coop': 0,
        'fully_coop': 0
    }
    
    fishers = [ag for ag in agents if ag.type == 'fishers']
    for fisher in fishers:
        counts[fisher.trait] += 1
    
    # Store counts
    for strategy in counts:
        strategy_counts[strategy].append(counts[strategy])
    
    # Calculate and store average cooperation level
    # Weight: fully_noncoop=0, noncoop=0.25, cond_coop=0.5, coop=0.75, fully_coop=1
    weights = {'fully_noncoop': 0, 'noncoop': 0.25, 'cond_coop': 0.5, 'coop': 0.75, 'fully_coop': 1}
    weighted_values = [counts[s] * weights[s] for s in weights.keys()]  # Create list first
    total_weight = sum(weighted_values)  # Then sum the list
    avg_cooperation = total_weight / len(fishers) if fishers else 0
    cooperation_levels.append(avg_cooperation)

def check_threshold_behavior():
    """Check local fish density and adjust fishing effort based on thresholds."""
    global agents
    
    # Process each fisher
    for fisher in [ag for ag in agents if ag.type == 'fishers']:
        # Count nearby fish
        local_fish = len([f for f in agents if f.type == 'fish' and 
                         ((fisher.x - f.x)**2 + (fisher.y - f.y)**2) < threshold_radius**2])
        
        # Calculate local density (fish per unit area)
        local_density = local_fish / (math.pi * threshold_radius**2)
        
        # If density is below threshold
        if local_density < fish_density_threshold:
            fisher.low_density_memory += 1
            if fisher.low_density_memory >= 1:  # Immediate response to low density
                # Increase cooperation (decrease effort) while maintaining trait category
                fisher.effort = max(0.2, fisher.base_effort - cooperation_increase)
        else:
            # If density is good for threshold_memory steps, revert to base effort
            if fisher.low_density_memory > 0:
                fisher.low_density_memory -= 1
                if fisher.low_density_memory == 0:
                    fisher.effort = fisher.base_effort

def update_trust():
    """Update trust scores between fishers based on their behavior."""
    fishers = [ag for ag in agents if ag.type == 'fishers']
    
    # For each fisher
    for fisher in fishers:
        # Find neighbors within trust radius
        neighbors = [nb for nb in fishers if nb != fisher and 
                    ((fisher.x - nb.x)**2 + (fisher.y - nb.y)**2) < trust_radius**2]
        
        for neighbor in neighbors:
            # Calculate cooperation score based on effort level
            # Lower effort = more cooperative
            fisher_coop_score = 1 - fisher.effort
            neighbor_coop_score = 1 - neighbor.effort
            
            # Update trust based on neighbor's behavior
            if neighbor_coop_score >= 0.5:  # If neighbor is being cooperative
                fisher.trust_scores[neighbor.num] = min(1.0, 
                    fisher.trust_scores[neighbor.num] + trust_increase)
            else:  # If neighbor is being non-cooperative
                fisher.trust_scores[neighbor.num] = max(0.0, 
                    fisher.trust_scores[neighbor.num] - trust_decrease)
            
            # Update cooperation counts
            fisher.total_actions += 1
            if neighbor_coop_score >= 0.5:
                fisher.cooperative_actions += 1

def adjust_effort_based_on_trust():
    """Adjust fishing effort based on trust levels."""
    fishers = [ag for ag in agents if ag.type == 'fishers']
    
    for fisher in fishers:
        if fisher.trait == 'cond_coop':  # Only conditional cooperators adjust based on trust
            # Calculate average trust in neighbors
            neighbors = [nb for nb in fishers if nb != fisher and 
                        ((fisher.x - nb.x)**2 + (fisher.y - nb.y)**2) < trust_radius**2]
            
            if neighbors:
                avg_trust = mean([fisher.trust_scores[nb.num] for nb in neighbors])
                
                # Adjust effort based on trust
                if avg_trust > trust_threshold:
                    # More cooperative (lower effort) when trust is high
                    fisher.effort = max(0.2, fisher.base_effort - 0.2)
                else:
                    # Less cooperative (higher effort) when trust is low
                    fisher.effort = min(1.0, fisher.base_effort + 0.2)

def track_trust_metrics():
    """Track and store trust-related metrics."""
    fishers = [ag for ag in agents if ag.type == 'fishers']
    
    # Calculate average trust across all fisher pairs
    all_trust_scores = []
    for fisher in fishers:
        all_trust_scores.extend(fisher.trust_scores.values())
    
    avg_trust = mean(all_trust_scores) if all_trust_scores else 0
    trust_history.append(avg_trust)
    
    # Calculate cooperation rates using list comprehension instead of generators
    coop_actions = sum([f.cooperative_actions for f in fishers])
    total_actions = sum([f.total_actions for f in fishers])
    total_coop_rate = coop_actions / max(1, total_actions)
    
    return avg_trust, total_coop_rate

def save_trust_data():
    """Save trust and cooperation data to CSV."""
    with open('simulation_output/trust_data.csv', 'w') as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(['time', 'average_trust', 'cooperation_rate'])
        
        # Calculate cooperation rates for each time step
        fishers = [ag for ag in agents if ag.type == 'fishers']
        coop_actions = sum([f.cooperative_actions for f in fishers])
        total_actions = sum([f.total_actions for f in fishers])
        coop_rate = coop_actions / max(1, total_actions)
        
        # Write data
        for t in range(len(trust_history)):
            writer.writerow([t, trust_history[t], coop_rate])

def plot_trust_dynamics():
    """Plot trust and cooperation dynamics."""
    plt.figure(figsize=(10, 6))
    plt.plot(trust_history, 'b-', label='Average Trust')
    plt.xlabel('Time')
    plt.ylabel('Trust Level')
    plt.title('Evolution of Trust Between Fishers')
    plt.grid(True)
    plt.legend()
    plt.savefig('simulation_output/trust_dynamics.png')
    plt.close()

def update_one_unit_time():
    """Main update function for one simulation time step with NumPy vectorized optimization."""
    global time1, agents, fish, total_fish_count, species_count, fish_data_MPA, total_hav_data, current_hav_data, fishermen, fishermen_data1, fishermen_data2, fishermen_data3

    time1 += 1
    
    # Cache agents by type for faster lookups
    fish_agents = [j for j in agents if j.type == 'fish']
    fisher_agents = [j for j in agents if j.type == 'fishers']

    # Update fish positions and growth
    update_fish()
    
    # Update trust and adjust behavior
    update_trust()
    adjust_effort_based_on_trust()
    
    # Track trust metrics
    track_trust_metrics()
    
    # Check and update threshold-based behavior
    check_threshold_behavior()
    
    # Cache MPA type condition checks
    is_no_mpa = (MPA == 'no' and Both == 'no')
    is_single_mpa = any([(MPA == 'yes' and Type_MPA == 'single' and Both == 'no'),
                        (MPA == 'no' and Both == 'yes' and Type_MPA == 'single')])
    is_spaced_mpa = any([(MPA == 'yes' and Type_MPA == 'spaced' and Both == 'no'),
                        (MPA == 'no' and Both == 'yes' and Type_MPA == 'spaced')])

    # Use NumPy for fish and fisher positions
    fish_to_remove = []
    catch_made_array = np.zeros(len(fisher_agents), dtype=bool)

    if fish_agents and fisher_agents:  # Only proceed if we have both fish and fishers
        # Convert positions to NumPy arrays for vectorized operations
        fish_positions = np.array([(fish.x, fish.y) for fish in fish_agents])
        fisher_positions = np.array([(fisher.x, fisher.y) for fisher in fisher_agents])
        fisher_efforts = np.array([fisher.effort for fisher in fisher_agents])

        # Process each fisher's catches efficiently with NumPy
        for i, fisher in enumerate(fisher_agents):
            # Calculate distances to all fish at once
            differences = fish_positions - fisher_positions[i]
            squared_distances = np.sum(differences**2, axis=1)

            # Find fish within catch radius
            within_radius = squared_distances <= r_sqr
            catchable_fish_indices = np.where(within_radius)[0]

            # If any fish are within radius, try to catch one
            if len(catchable_fish_indices) > 0:
                # Pick the first fish in range
                fish_idx = catchable_fish_indices[0]
                target_fish = fish_agents[fish_idx]

                # Try to catch the fish with probability based on effort
                if rd.random() < q * fisher.effort:
                    fisher.harvest += 1
                    total_hav_data[fisher.num].append(fisher.harvest)
                    current_hav_data[fisher.num].append(1)
                    fish_to_remove.append(target_fish)  # Mark fish for later removal
                    catch_made_array[i] = True
                    continue  # Continue to next fisher

            # If no catch was made, record 0 catch
            current_hav_data[fisher.num].append(0)

        # Update fisherman position (using single calls instead of repeated any() checks)
        if is_no_mpa:
            no_mpa()
        elif is_single_mpa:
            single_mpa()
        elif is_spaced_mpa:
            spaced_mpa()
    else:
        # Handle case with no fish or no fishers
        for fisher in fisher_agents:
            current_hav_data[fisher.num].append(0)

    # Remove caught fish all at once (faster than removing one by one in the loop)
    for fish in fish_to_remove:
        if fish in agents:  # Make sure fish still exists (could have been removed by another fisher)
            agents.remove(fish)
    
    # Calculate and update statistics - vectorized where possible
    # Update MPA fish count
    if is_no_mpa:
        fish_data_MPA.append(0)
    elif fish_agents:  # Only do these calculations if we have fish
        if is_single_mpa:
            # Count fish in single MPA using NumPy vectorized operations
            fish_positions = np.array([(fish.x, fish.y) for fish in fish_agents])
            in_mpa = ((fish_positions[:, 0] >= Xa) & (fish_positions[:, 0] <= Xb) &
                      (fish_positions[:, 1] >= Ya) & (fish_positions[:, 1] <= Yb))
            fish_data_MPA.append(np.sum(in_mpa))

        elif is_spaced_mpa:
            # Count fish in spaced MPA areas using NumPy
            fish_positions = np.array([(fish.x, fish.y) for fish in fish_agents])
            in_mpa1 = ((fish_positions[:, 0] >= Xm) & (fish_positions[:, 0] <= Xn) &
                       (fish_positions[:, 1] >= Ym) & (fish_positions[:, 1] <= Yn))
            in_mpa2 = ((fish_positions[:, 0] >= Xp) & (fish_positions[:, 0] <= Xq) &
                       (fish_positions[:, 1] >= Yp) & (fish_positions[:, 1] <= Yq))
            fish_data_MPA.append(np.sum(in_mpa1 | in_mpa2))
    else:
        # No fish case
        fish_data_MPA.append(0)
    
    # Update other statistics with NumPy
    if fisher_agents:  # Only if we have fishers
        harvest_values = np.array([j.harvest for j in fisher_agents])
        total_harvest = np.sum(harvest_values)

        catch_values = np.array([current_hav_data[j.num][-1] for j in fisher_agents])
        current_catches = np.sum(catch_values)

        fishermen_data1.append(total_harvest)
        fishermen_data2.append(current_catches)
    else:
        # No fishers case
        fishermen_data1.append(0)
        fishermen_data2.append(0)

    # Calculate outside MPA fish count
    fishermen_data3.append(total_fish_count[-1] - fish_data_MPA[-1])

    # Imitation and tracking
    if time1 % imitation_period == 0:  # Every X time steps
        imitate_successful_strategies()
    track_cooperation_levels()
    
    # Save data to CSV - only do this every few time steps to reduce I/O overhead
    if time1 % 10 == 0 or time1 >= n - 1:  # Every 10 steps or at the end
        csvfile = "simulation_output/simulation_data.csv"   # a csv-file output
        header = [key for key in sorted(current_hav_data)]
        header.append('total_catch') ; header.append('total_biomass') ; header.append('biomass_inside_MPA') ; header.append('biomass_outside_MPA')
        main_data = [current_hav_data[key] for key in sorted(current_hav_data)]
        main_data.append(fishermen_data2) ; main_data.append(total_fish_count) ; main_data.append(fish_data_MPA) ; main_data.append(fishermen_data3)
        with open(csvfile, "w") as output:
            writer = csv.writer(output)
            writer.writerow(header)
            writer.writerows(zip(*main_data))

def setup_live_plot():
    """Setup the live plotting figure."""
    fig = plt.figure(figsize=(10, 8))
    
    # Fish population subplot
    ax1 = fig.add_subplot(211)
    ax1.set_title('Fish Population')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Number of Fish')
    pop_line, = ax1.plot([], [], 'b-', label='Total Population')
    mpa_line, = ax1.plot([], [], 'g-', label='In MPA')
    outside_line, = ax1.plot([], [], 'r-', label='Outside MPA')
    ax1.legend()
    ax1.grid(True)
    
    # Catch subplot
    ax2 = fig.add_subplot(212)
    ax2.set_title('Fishing Activity')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Number of Fish Caught')
    total_catch_line, = ax2.plot([], [], 'b-', label='Total Catch')
    current_catch_line, = ax2.plot([], [], 'r-', label='Current Catch')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    return (ax1, ax2), (pop_line, mpa_line, outside_line, total_catch_line, current_catch_line)

def update_live_plot(axes, lines, step):
    """Update the live plot with current data."""
    plt.ion()
    ax1, ax2 = axes
    pop_line, mpa_line, outside_line, total_catch_line, current_catch_line = lines

    # Get time range
    times = list(range(step + 1))

    # Update fish population lines
    pop_line.set_data(times, total_fish_count)
    mpa_line.set_data(times, fish_data_MPA)
    outside_line.set_data(times, fishermen_data3)

    if len(species_count) > 1:
        # Clear previous species lines first to avoid overlap
        # You can either clear ax1 lines or store lines for species separately
        # Here, let's remove any previously added species lines by clearing lines except the original three:
        while len(ax1.lines) > 3:  # keep pop_line, mpa_line, outside_line only
            ax1.lines.pop()

        # Plot each species population with different color and label
        for species_name, pop_list in species_count.items():
            species_data = pop_list[:step + 1]  # slice up to current step
            ax1.plot(times, species_data, label=species_name, linestyle='--')

        ax1.legend()

    # Update catch lines
    total_catch_line.set_data(times, fishermen_data1)
    current_catch_line.set_data(times, fishermen_data2)

    # Adjust axes limits
    ax1.set_xlim(0, max(n, step + 1))
    max_y1 = max(
        max(total_fish_count),
        max(fish_data_MPA),
        max(fishermen_data3),
        *(max(pop_list[:step + 1]) for pop_list in species_count.values()) if species_count else [0]
    )
    ax1.set_ylim(0, max_y1 * 1.1)

    ax2.set_xlim(0, max(n, step + 1))
    ax2.set_ylim(0, max(max(fishermen_data1), max(fishermen_data2)) * 1.1)

    plt.draw()
    plt.pause(0.01)
    plt.ioff()

######################################################################################################################################################       

experiment_label = 'reproduction_rate'
initialize(experiment_label)
observe()

# Setup live plot
axes, plot_lines = setup_live_plot()

print("Starting simulation...")
for j in tqdm(range(1, n), desc="Simulating", unit="step"):
    update_one_unit_time()
    observe()
    
    # # Update live plot periodically
    # if j % plot_update_freq == 0:
    #     update_live_plot(axes, plot_lines, j)

# Final plot update
if len(total_fish_count) == n:
    update_live_plot(axes, plot_lines, n-1)

plot_summary()
save_cooperation_data()  # Save cooperation data
save_trust_data()  # Save trust data

plt.ioff()  # Disable interactive mode
plt.show()  # Keep the final plot window open

print('Creating gif...')
create_gif()
delete_prev_sim()

# Remove or comment out the ffmpeg video creation line at the end
# os.system("ffmpeg -v quiet -r 5 -i year_%04d.png -vcodec mpeg4  -y -s:v 1920x1080 simulation_movie.mp4")


#------------------------------------------------------------------------------------------------------------------ 

# os.chdir(os.pardir) # optional: move up to parent folder

#----------------------------------------------------------------------------------------------------------------
