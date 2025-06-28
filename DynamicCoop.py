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
from snapshot_visualization import save_snapshot


from parameters import *
#######################################################################################################################################################  

class agent:  # create an empty class
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    
#----------------------------------------------------------------------------------------------------------    

def delete_prev_sim():
    """
    Clean up previous simulation visualization files.
    Handles parallel execution gracefully by silently ignoring file deletion errors.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sim_output_dir = os.path.join(script_dir, "simulation_output")
    pattern = os.path.join(sim_output_dir, "year_*.png")

    # Iterate and remove matching files (silently handle parallel deletion conflicts)
    for file_path in glob.glob(pattern):
        try:
            os.remove(file_path)
        except (FileNotFoundError, PermissionError, OSError):
            # Silently ignore deletion errors (file already deleted by another process)
            pass


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
            ag.x = rd.uniform(-half_length_area, half_length_area)
            ag.y = rd.uniform(-half_length_area, half_length_area)
        
        elif any([(MPA == 'yes' and Type_MPA == 'single' and Both == 'no'),
                 (MPA == 'no' and Both == 'yes' and Type_MPA == 'single')]):  # single MPA
            while True:
                ag.x = rd.uniform(-half_length_area, half_length_area)
                ag.y = rd.uniform(-half_length_area, half_length_area)
                if not((Xa <= ag.x <= Xb) and (Ya <= ag.y <= Yb)):
                    break
        
        elif any([(MPA == 'yes' and Type_MPA == 'spaced' and Both == 'no'),
                 (MPA == 'no' and Both == 'yes' and Type_MPA == 'spaced')]):  # spaced MPA
            while True:
                ag.x = rd.uniform(-half_length_area, half_length_area)
                ag.y = rd.uniform(-half_length_area, half_length_area)
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
        for _ in range(num_fish_per_combo):
            ag = agent(**params)
            ag.type = 'fish'
            ag.school = school_id
            ag.x = rd.uniform(-half_length_area, half_length_area)
            ag.y = rd.uniform(-half_length_area, half_length_area)
            ag.subtype = subtype_str
            species_count[subtype_str] = [num_fish_per_combo]
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
    ax.set_xlim([-half_length_area, half_length_area])
    ax.set_ylim([-half_length_area, half_length_area])
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f'year = {int(time1)}')
    ax.legend(numpoints=1, loc='center', bbox_to_anchor=(0.5, -0.15), ncol=3, prop={'size': 11},
              facecolor='lightskyblue')

    fig.savefig(f'simulation_output/year_{int(time1):04d}.png', bbox_inches='tight', pad_inches=0.1, dpi=200)
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

    # Plot 2: Subspecies population dynamics
    plt.subplot(5, 1, 2)
    for species in species_count:
        plt.plot(species_count[species], label=species)

    plt.xlabel('Time')
    plt.ylabel('Number of fish')
    plt.title('Species Population Dynamics')
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
    
    # Plot 5: Average cooperation level
    plt.plot(time_steps, cooperation_levels, 'k--', label='Average Cooperation Level', linewidth=2)
    
    plt.xlabel('Time')
    plt.ylabel('Number of Fishers / Cooperation Level')
    plt.title('Evolution of Cooperation Strategies')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    # Plot 4: Trust dynamics
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

def init_boids_zones(fish_agent, all_agents, square_repulsion_rad, square_orientation_rad, square_attraction_rad):
    repulsion = [
        nb for nb in all_agents
        if nb.type == 'fish'
           and nb != fish_agent
           and nb.school == fish_agent.school
           and ((fish_agent.x - nb.x) ** 2 + (fish_agent.y - nb.y) ** 2) < square_repulsion_rad
    ]
    alignment = [
        nb for nb in all_agents
        if nb.type == 'fish'
           and nb != fish_agent
           and nb.school == fish_agent.school
           and square_repulsion_rad < ((fish_agent.x - nb.x) ** 2 + (fish_agent.y - nb.y) ** 2) < square_orientation_rad
    ]
    attraction = [
        nb for nb in all_agents
        if nb.type == 'fish'
           and nb != fish_agent
           and nb.school == fish_agent.school
           and square_orientation_rad < ((fish_agent.x - nb.x) ** 2 + (fish_agent.y - nb.y) ** 2) < square_attraction_rad
    ]
    return repulsion, alignment, attraction


def repulsion_update(fish_agent, repulsion_agents, border):
    repulsion_x = mean([j.x for j in repulsion_agents])
    repulsion_y = mean([j.y for j in repulsion_agents])
    theta = (math.atan2((repulsion_y - fish_agent.y), (repulsion_x - fish_agent.x)) + math.pi) % (
                2 * math.pi)  # if greater than  (2 * math.pi) then compute with a minus
    fish_agent.x += fish_agent.speed * math.cos(theta)  # moves 'move_fish' step
    fish_agent.y += fish_agent.speed * math.sin(theta)
    fish_agent.x = (fish_agent.x % -border) if fish_agent.x > border else (
            fish_agent.x % border) if fish_agent.x < -border else fish_agent.x  # ( When fish-agent approach a border of the landscape,
    fish_agent.y = (fish_agent.y % -border) if fish_agent.y > border else (
            fish_agent.y % border) if fish_agent.y < -border else fish_agent.y  # they re-enter the system at the opposite border)


def alignment_update(fish_agent, alignment_agents, border):
    theta = mean([math.atan2((j.y - fish_agent.y),(j.x - fish_agent.x)) for j in alignment_agents])
    fish_agent.x +=   math.cos(theta)     # moves 'move_fish' step,  move_fish*math.cos(theta)
    fish_agent.y +=   math.sin(theta)
    fish_agent.x = (fish_agent.x % -border) if fish_agent.x > border else (
            fish_agent.x % border) if fish_agent.x < -border else fish_agent.x  # ( When fish-agent approach a border of the landscape,
    fish_agent.y = (fish_agent.y % -border) if fish_agent.y > border else (
            fish_agent.y % border) if fish_agent.y < -border else fish_agent.y  # they re-enter the system at the opposite border )


def attraction_update(fish_agent, attraction_agents, border):
    attraction_x = mean([j.x for j in attraction_agents])
    attraction_y = mean([j.y for j in attraction_agents])
    theta = math.atan2((attraction_y - fish_agent.y), (attraction_x - fish_agent.x))
    fish_agent.x +=  fish_agent.speed*math.cos(theta)
    fish_agent.y +=  fish_agent.speed*math.sin(theta)
    fish_agent.x = (fish_agent.x % -border) if fish_agent.x > border else (
            fish_agent.x % border) if fish_agent.x < -border else fish_agent.x  # ( When fish-agent approach a border of the landscape,
    fish_agent.y = (fish_agent.y % -border) if fish_agent.y > border else (
            fish_agent.y % border) if fish_agent.y < -border else fish_agent.y  # they re-enter the system at the opposite border )


def random_update(fish_agent, border):
    theta = 2 * math.pi * rd.random()
    fish_agent.x += fish_agent.speed * math.cos(theta)  # moves 'move_fish' step
    fish_agent.y += fish_agent.speed * math.sin(theta)
    fish_agent.x = (fish_agent.x % -border) if fish_agent.x > border else (
                fish_agent.x % border) if fish_agent.x < -border else fish_agent.x  # ( When fish-agent approach a border of the landscape,
    fish_agent.y = (fish_agent.y % -border) if fish_agent.y > border else (
                fish_agent.y % border) if fish_agent.y < -border else fish_agent.y  # they re-enter the system at the opposite border )


def reproduce(fish_agent, new_fish, new_species_count):
    agents.append(cp.copy(fish_agent))
    new_fish += 1
    new_species_count[fish_agent.subtype] += 1
    return new_fish


def update_counts(old_fish, new_fish, new_species_count):
    total_fish_count.append(old_fish + new_fish)
    for species in species_count:
        if species in new_species_count:
            species_count[species].append(new_species_count[species])
        else:
            species_count[species].append(0)


def update_fish():
    
    global time1, agents, fish, total_fish_count, species_count, fish_data_MPA, total_hav_data, current_hav_data, fishermen , fishermen_data1,  fishermen_data2, fishermen_data3
    fish_list = [j for j in agents if j.type == 'fish']
    # shuffle list to make sure all fish have a fair chance at reproduction
    rd.shuffle(fish_list)
    if not fish_list:
        total_fish_count.append(0)
        for species in species_count:
            species_count[species].append(0)
        return

    # fish_ag = rd.sample(fish_list, 1)[-1]
    new_fish = 0
    new_species_count = {}
    for fish_ag in fish_list:
        if fish_ag not in new_species_count:
            new_species_count[fish_ag.subtype] = 1
        else:
            new_species_count[fish_ag.subtype] += 1

        repulsion, alignment, attraction = init_boids_zones(fish_ag, agents, rad_repulsion_sqr, rad_orientation_sqr,
                                                            rad_attraction_sqr)

        # if fishes within repulsion zone, move away from the spot that would be the center of mass (midpoint)
        # of all  fish within repulsion zone
        if len(repulsion) > 0:
            repulsion_update(fish_ag, repulsion, half_length_area)

        # if fishes within parallel-orientation zone, change direction to match the average direction of all the other
        # fish within parallel-orientation zone
        elif all([len(repulsion) == 0, len(alignment) > 0]):
            alignment_update(fish_ag, alignment, half_length_area)

        elif all([len(repulsion) == 0, len(alignment) == 0, len(attraction) > 0]): # if fishes within only the attraction zone, head towards the middle (midpoint) of the fishes in zone of attraction.
            attraction_update(fish_ag, attraction, half_length_area)

        elif all([len(repulsion) == 0, len(alignment) == 0, len(attraction) == 0]): # if no fishes in all the zone, move in a random direction
            random_update(fish_ag, half_length_area)

        if len(fish_list) + new_fish < total_fish_count[0] and rd.random() < fish_ag.reproduction_rate * (1-sum([1 for j in agents if j.type == 'fish'])/float(K)):  # logistic growth of fishes
            new_fish = reproduce(fish_ag, new_fish, new_species_count)

    update_counts(len(fish_list), new_fish, new_species_count)
       
######################################################################################################################################################                         
                  
def move_fisher(fisher):
    global time1, agents, fish, total_fish_count, fish_data_MPA, total_hav_data, current_hav_data, fishermen, fishermen_data1,  fishermen_data2, fishermen_data3

    fishers_neighbors = [[nb.harvest, nb] for nb in agents if nb.type == 'fishers' and nb != fisher and ((fisher.x - nb.x)**2 + (fisher.y - nb.y)**2) < r_sqr] # detecting fishermen in neighbourhood
    fishers_neighbors_harvest = sorted(fishers_neighbors, key=lambda HAV: HAV[0]) # sort fishermen in neighborhood according to catch
    if len(fishers_neighbors_harvest) == 0: # if there exist no fisherman in neighbourhood
        theta_1 = 2*math.pi*rd.random()
        fisher.x +=  move_fishers*math.cos(theta_1) # move  'move_fishers' step in a random direction
        fisher.y +=  move_fishers*math.sin(theta_1)
        fisher.x = -half_length_area if fisher.x > half_length_area else  half_length_area if fisher.x < -half_length_area else fisher.x
        fisher.y = -half_length_area if fisher.y > half_length_area else  half_length_area if fisher.y < -half_length_area else fisher.y
    elif all([len(fishers_neighbors_harvest) > 0, fishers_neighbors_harvest[-1][0] > fisher.harvest]) : # if there exist fisherman with greater catch than focal fisherman
            deltax = fishers_neighbors_harvest[-1][-1].x - fisher.x   #move in the direction of one with greater catch than focal fisherman
            deltay = fishers_neighbors_harvest[-1][-1].y - fisher.y
            theta_2 = math.atan2(deltay,deltax)
            fisher.x +=  move_fishers*math.cos(theta_2) # move 'move_fishers' in the direction of neighbour fishermen with greatest catch
            fisher.y +=  move_fishers*math.sin(theta_2)
            fisher.x = -half_length_area if fisher.x > half_length_area else  half_length_area if fisher.x < -half_length_area else fisher.x
            fisher.y = -half_length_area if fisher.y > half_length_area else  half_length_area if fisher.y < -half_length_area else fisher.y
    else: # if all fisherman have less or equal catch relativelly  to focal fisherman
            theta_2 = 2*math.pi*rd.random()
            fisher.x +=  move_fishers*math.cos(theta_2) # move  'move_fishers' step in a random direction
            fisher.y +=  move_fishers*math.sin(theta_2)
            fisher.x = -half_length_area if fisher.x > half_length_area else  half_length_area if fisher.x < -half_length_area else fisher.x
            fisher.y = -half_length_area if fisher.y > half_length_area else  half_length_area if fisher.y < -half_length_area else fisher.y

   
###################################################################################################################################################### 

def classify_trait_from_effort(effort):
    """Classifies a fisher's trait based on their continuous effort value."""
    if effort >= 0.9:
        return 'fully_noncoop'
    elif 0.7 <= effort < 0.9:
        return 'noncoop'
    elif 0.5 <= effort < 0.7:
        return 'cond_coop'
    elif 0.3 <= effort < 0.5:
        return 'coop'
    else:
        return 'fully_coop'

def imitate_successful_strategies():
    """Allow fishers to imitate more successful strategies from their neighbors."""
    global agents

    # Get all fishers
    fishers = [ag for ag in agents if ag.type == 'fishers']

    # For each fisher
    for fisher in fishers:
        # Find neighbors within imitation radius
        neighbors = [nb for nb in fishers if nb != fisher and
                    ((fisher.x - nb.x)**2 + (fisher.y - nb.y)**2) < imitation_radius**2]

        if neighbors:
            # Find the most successful neighbor
            most_successful = max(neighbors, key=lambda x: x.harvest)

            # If the most successful neighbor has higher harvest
            if most_successful.harvest > fisher.harvest:
                # Probabilistically imitate their strategy
                if rd.random() < imitation_prob:
                    # Nudge effort towards the more successful agent
                    nudge = (most_successful.effort - fisher.effort) * imitation_nudge_factor
                    fisher.effort += nudge
                    
                    # Re-classify the trait based on the new effort
                    fisher.trait = classify_trait_from_effort(fisher.effort)

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
    global time1, agents, fish, total_fish_count, fish_data_MPA, total_hav_data, current_hav_data, fishermen, fishermen_data1, fishermen_data2, fishermen_data3

    # First timestep: update fish, let fishers catch fish
    time1 += 1
    
    # Update fish positions and growth
    update_fish()
    
    # Update trust and adjust behavior
    update_trust()
    adjust_effort_based_on_trust()
    
    # Track trust metrics
    track_trust_metrics()
    
    # Check and update threshold-based behavior
    check_threshold_behavior()
    
    # Update fishermen positions and catches
    fishers = [j for j in agents if j.type == 'fishers']
    for fisher in fishers:
        fish_neighbors = [nb for nb in agents if nb.type == 'fish' and ((fisher.x - nb.x) ** 2 + (
                    fisher.y - nb.y) ** 2) < r_sqr]  # detecting fishes in neighbourhood
        num_fish_harvest = int(round(q * fisher.effort * len(
            fish_neighbors)))  # number of fish to be harvested based on (q*E*x), where x is number of fishes in neighborhood
        if fish_neighbors and num_fish_harvest > 0:
            sample_fish_harvest = rd.sample(fish_neighbors, min(num_fish_harvest, len(fish_neighbors)))
            for j in sample_fish_harvest:
                agents.remove(j)  # remove catch
                fisher.harvest += 1  # add to catch of a fisherman
        else:
            current_hav_data[fisher.num].append(0)
    for fisher in fishers:
        move_fisher(fisher)

    # Next timestep: update stats, fishers imitate strategies
    time1 += 1
    
    # Update MPA fish count
    fish_data_MPA.append(0)
    
    fishermen_data1.append(sum([j.harvest for j in agents if j.type == 'fishers']))
    fishermen_data2.append(sum([current_hav_data[j.num][-1] for j in agents if j.type == 'fishers']))
    fishermen_data3.append(total_fish_count[-1] - fish_data_MPA[-1])
   
    # Imitation and tracking
    if time1 % imitation_period == 0:  # Every X time steps
        imitate_successful_strategies()
    track_cooperation_levels()
    
    # Save data to CSV
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

    # Update catch lines
    total_catch_line.set_data(times, fishermen_data1)
    current_catch_line.set_data(times, fishermen_data2)

    # Adjust axes limits
    ax1.set_xlim(0, max(n, step + 1))
    ax1.set_ylim(0, max(max(total_fish_count), max(fish_data_MPA), max(fishermen_data3)) * 1.1)

    ax2.set_xlim(0, max(n, step + 1))
    ax2.set_ylim(0, max(max(fishermen_data1), max(fishermen_data2)) * 1.1)

    plt.draw()
    plt.pause(0.01)
    plt.ioff()

######################################################################################################################################################       

# Only run simulation if this script is executed directly (not imported)
if __name__ == "__main__":
experiment_label = 'both'
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

# Create a snapshot with the new visualization
print("\nCreating snapshot visualization...")
save_snapshot(agents, time1, 'final_state.png', half_length_area=half_length_area)

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
