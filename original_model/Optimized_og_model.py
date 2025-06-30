# -*- coding: utf-8 -*-
#--------------------------------------------------------------------------------------

# Agent-Based Model (ABM) representative of an idealised small-scale, artisanal fishery. 
# The model has been developed to investigate the combined effects of fishing behaviour 
# (expressed by a cooperative trait associated to fishing effort) and different designs 
# of Marine Protected Areas (age, size, and distance between two MPAs).

# By : OWUSU, Kwabena Afriyie
# Date : 16th April, 2019

#---------------------------------------------------------------------------

## Create a subfolder to save data ##
import shutil, os
subdir = 'simulation_output' # subfolder name
if os.path.isdir(subdir):    # if subfolder already exist
    shutil.rmtree(subdir)    # delete the whole folder
os.mkdir(subdir)             # make new subfolder
os.chdir(subdir)             # move to subfolder

#---------------------------------------------------------------------------

# Import relevent libraries #
import os
import shutil
import random as rd
import math
import numpy as np
import matplotlib.pyplot as plt
import csv 
from statistics import mean
from functools import lru_cache

#---------------------------------------------------------------------------

# Parameters #

# Fishing ground and time #
K = 200 # carrying capacity of fishing ground
n = 150 # number of simulation time steps

# Attributes of fish agents #
growth_prob =  0.26    # maximum intrinsic growth rate
init_fish = 200        # initial number of fish agents
move_fish = 0.2        # speed of fish 
rad_repulsion = 0.025  # radius of repulsion zone
rad_orientation = 0.06 # radius of orientation zone 
rad_attraction =  0.1  # radius of attraction zone 
rad_repulsion_sqr = rad_repulsion ** 2     
rad_orientation_sqr = rad_orientation ** 2 
rad_attraction_sqr = rad_attraction ** 2   

# Attributes of fishing agents (pirogues) #
num_fishers = 20     # number of pirogues
move_fishers = 0.3   # speed of a pirogue 
q = 0.6              # catchability coefficient
r = 0.2              # neighbourhood radius 
r_sqr = r ** 2       # neighbourhood radius squared

# Cooperation scenarios (summ of all cooperation types = num_fishers) #
fully_noncoop = 4     # number of fully non-cooperative pirogues
noncoop = 4           # number of non-cooperative pirogues
cond_coop = 4         # number of conditional cooperative pirogues
coop = 4              # number of cooperative pirogues
fully_coop = 4        # number of fully cooperative pirogues

# Design of the MPA (presence/absence, size, age, and distance of between two) #
MPA = 'yes'         # Presence or absence of MPA ('yes' for presence, 'no' for absence)
Both = 'no'         # Presence of MPA ('no' for full-time presence, 'yes' for part-time presence)
Time_MPA = 50       # Period of time over which MPA is active (when Both = 'yes') 
Type_MPA = 'single' # Spacial configuration of MPA ('spaced' for two MPAs, 'single' for one MPA)
Dist_MPA = 0.2      # Distance between two MPAs (when Type_MPA = 'spaced')
Frac_MPA = 0.25     # Fraction of fishing grounde covered by MPA(s)

# Coordinates of the fishing ground #
Area = 2.0000 
Length_Area = math.sqrt(Area)
Half_Length_Area = Length_Area / 2

# Coordinates of the MPA #' 
Half_Length = (math.sqrt(Frac_MPA* Area)) / 2 # compute half the length  of MPA 

# Coordinates for a single MPA #
Xa = - Half_Length 
Xb =   Half_Length 
Ya = - Half_Length 
Yb =   Half_Length

# Coordinates of first spaced MPA #
Xm = - Half_Length - (Dist_MPA / 2)
Xn = -(Dist_MPA / 2) 
Ym = - Half_Length 
Yn =   Half_Length 

# Coordinates of second spaced MPA #
Xp = (Dist_MPA / 2) 
Xq =  Half_Length + (Dist_MPA / 2)
Yp = -Half_Length 
Yq =  Half_Length 

#######################################################################################################################################################  

class agent:  # create an empty class
    pass     
    
#----------------------------------------------------------------------------------------------------------    

def initialize():
    
    global time1, agents, fish, fish_data, fish_data_MPA, total_hav_data, current_hav_data, fishermen, fishermen_data1, fishermen_data2, fishermen_data3 
    time1 = 0. # time
    agents = []  # list containing fishes and fishermen
    fish_data = [init_fish]  # list containing number of fishes
    total_hav_data = {} # dictionary containing total catch of fishermen according to cooperative-trait 
    current_hav_data = {} # dictionary containing current catch of fishermen according to cooperative-trait 
    fishermen_data1 = [0] # list containing total catch of fishermen 
    fishermen_data2 = [0] # list containing current catch of fishermen 

    # Pre-compute common values to avoid repetition
    mpa_scenario = 0  # 0=no MPA, 1=single MPA, 2=spaced MPA
    
    # Determine the MPA scenario once
    if MPA == 'no' and Both == 'no':
        mpa_scenario = 0
    elif (MPA == 'yes' and Type_MPA == 'single' and Both == 'no') or \
         (MPA == 'no' and Both == 'yes' and Type_MPA == 'single'):
        mpa_scenario = 1
    elif (MPA == 'yes' and Type_MPA == 'spaced' and Both == 'no') or \
         (MPA == 'no' and Both == 'yes' and Type_MPA == 'spaced'):
        mpa_scenario = 2

    # Helper function to check if a point is in an MPA
    def is_in_single_mpa(x, y):
        return (Xa <= x <= Xb) and (Ya <= y <= Yb)
    
    def is_in_spaced_mpa(x, y):
        in_mpa1 = (Xm <= x <= Xn) and (Ym <= y <= Yn)
        in_mpa2 = (Xp <= x <= Xq) and (Yp <= y <= Yq)
        return in_mpa1 or in_mpa2
    
    # Helper function to get position outside MPA
    def get_position_outside_mpa(mpa_check_func):
        max_attempts = 100  # Prevent infinite loop
        for _ in range(max_attempts):
            x = rd.uniform(-Half_Length_Area, Half_Length_Area)
            y = rd.uniform(-Half_Length_Area, Half_Length_Area)
            if not mpa_check_func(x, y):
                return x, y
        # Fallback to random position if max attempts reached
        return rd.uniform(-Half_Length_Area, Half_Length_Area), rd.uniform(-Half_Length_Area, Half_Length_Area)

    # Set up trait information
    trait_data = [
        # effort, trait name, count, start_idx
        (1.0, 'fully_noncoop', fully_noncoop, 0),
        (0.8, 'noncoop', noncoop, fully_noncoop),
        (0.6, 'cond_coop', cond_coop, fully_noncoop + noncoop),
        (0.4, 'coop', coop, fully_noncoop + noncoop + cond_coop),
        (0.2, 'fully_coop', fully_coop, fully_noncoop + noncoop + cond_coop + coop)
    ]
    
    # Create fishermen agents with proper traits
    for j in range(num_fishers):
        ag = agent()
        ag.type = 'fishers'
        ag.harvest = 0
        
        # Assign effort and trait
        for effort, trait, count, start_idx in trait_data:
            if start_idx <= j < start_idx + count:
                ag.effort = effort
                ag.trait = trait
                ag.num = f"{trait}{j - start_idx + 1}"
                break
        
        # Initialize harvest data
        total_hav_data[ag.num] = [ag.harvest]
        current_hav_data[ag.num] = [ag.harvest]
        
        # Assign position based on MPA scenario
        if mpa_scenario == 0:
            # No MPA - random position
            ag.x = rd.uniform(-Half_Length_Area, Half_Length_Area)
            ag.y = rd.uniform(-Half_Length_Area, Half_Length_Area)
        elif mpa_scenario == 1:
            # Single MPA - position outside MPA
            ag.x, ag.y = get_position_outside_mpa(is_in_single_mpa)
        elif mpa_scenario == 2:
            # Spaced MPA - position outside both MPAs
            ag.x, ag.y = get_position_outside_mpa(is_in_spaced_mpa)
        
        agents.append(ag)
    
    # Create fish agents efficiently
    for _ in range(init_fish):
        fish_ag = agent()
        fish_ag.type = 'fish'
        fish_ag.x = rd.uniform(-Half_Length_Area, Half_Length_Area)
        fish_ag.y = rd.uniform(-Half_Length_Area, Half_Length_Area)
        agents.append(fish_ag)
    
    # Initialize MPA fish counts
    fish_agents = [j for j in agents if j.type == 'fish']
    
    if mpa_scenario == 0:
        fish_data_MPA = [0]  # No MPA
    elif mpa_scenario == 1:
        # Single MPA
        fish_in_mpa = sum(1 for j in fish_agents if is_in_single_mpa(j.x, j.y))
        fish_data_MPA = [fish_in_mpa]
    elif mpa_scenario == 2:
        # Spaced MPAs
        fish_in_mpa = sum(1 for j in fish_agents if is_in_spaced_mpa(j.x, j.y))
        fish_data_MPA = [fish_in_mpa]
    
    fishermen_data3 = [fish_data[-1] - fish_data_MPA[-1]]  # Initialize number of fishes outside MPA
    
######################################################################################################################################################    
        
def observe():
    
    global time1, agents, fish, fish_data, fish_data_MPA, total_hav_data, current_hav_data, fishermen, fishermen_data1, fishermen_data2, fishermen_data3    
    
    # Clear only the active figure to avoid overhead of searching for all figures
    plt.figure(figsize=(8, 8))
    plt.clf()
    plt.subplot(111, facecolor='lightskyblue')  # Set the background color
    
    # Separate agents by type once instead of multiple list comprehensions
    fish_agents = []
    fishermen_by_trait = {
        'fully_noncoop': [],
        'noncoop': [],
        'cond_coop': [],
        'coop': [],
        'fully_coop': []
    }
    
    for agent in agents:
        if agent.type == 'fish':
            fish_agents.append(agent)
        elif agent.type == 'fishers':
            fishermen_by_trait[agent.trait].append(agent)
    
    # Plot fish agents if there are any
    if fish_agents:
        fish_x = [ag.x for ag in fish_agents]
        fish_y = [ag.y for ag in fish_agents]
        plt.plot(fish_x, fish_y, '^', color='darkgreen', markersize=3, label='fish')
    
    # Plot fishermen by trait using a colormap
    colors = np.linspace(0, 1, 5)
    mymap = plt.get_cmap("Greys")
    my_colors = mymap(colors)
    
    # Plot each fisherman type with its corresponding color
    traits = ['fully_coop', 'coop', 'cond_coop', 'noncoop', 'fully_noncoop']
    color_indices = [4, 3, 2, 1, 0]  # Color indices from darkest to lightest
    
    for trait, color_idx in zip(traits, color_indices):
        if fishermen_by_trait[trait]:
            x_positions = [ag.x for ag in fishermen_by_trait[trait]]
            y_positions = [ag.y for ag in fishermen_by_trait[trait]]
            plt.plot(x_positions, y_positions, 'o', color=my_colors[color_idx], markersize=7.5, label=trait)
    
    # Store the current fishermen for later reference
    fishermen = [ag for trait in fishermen_by_trait.values() for ag in trait]
    
    # Pre-compute MPA visibility conditions to avoid redundant checks
    show_single_mpa = (MPA == 'yes' and Type_MPA == 'single' and Both == 'no') or \
                       (MPA == 'no' and Both == 'yes' and Type_MPA == 'single' and time1 <= Time_MPA)
    
    show_spaced_mpa = (MPA == 'yes' and Type_MPA == 'spaced' and Both == 'no') or \
                       (MPA == 'no' and Both == 'yes' and Type_MPA == 'spaced' and time1 <= Time_MPA)
    
    # Draw MPAs if needed
    if show_single_mpa:
        # More efficient to do one call with multiple line segments
        plt.vlines([Xa, Xb], [Ya, Ya], [Yb, Yb], lw=2, color='k')
        plt.hlines([Ya, Yb], [Xa, Xa], [Xb, Xb], lw=2, color='k')
        
    elif show_spaced_mpa:
        # First MPA
        plt.vlines([Xm, Xn], [Ym, Ym], [Yn, Yn], lw=2, color='k')
        plt.hlines([Ym, Yn], [Xm, Xm], [Xn, Xn], lw=2, color='k')
        
        # Second MPA
        plt.vlines([Xp, Xq], [Yp, Yp], [Yq, Yq], lw=2, color='k')
        plt.hlines([Yp, Yq], [Xp, Xp], [Xq, Xq], lw=2, color='k')
    
    # Set up the plot appearance
    plt.axis('image')
    plt.axis([-Half_Length_Area, Half_Length_Area, -Half_Length_Area, Half_Length_Area])
    plt.grid(False)
    plt.xticks([], [])
    plt.yticks([], [])
    plt.title('year =' + str(int(time1)))
    
    # Add legend with better positioning
    plt.legend(numpoints=1, loc='center', bbox_to_anchor=(0.5, -0.072), 
              ncol=3, prop={'size': 11}, facecolor='lightskyblue')
    
    # Save figure with optimized settings
    plt.savefig(f'year_{int(time1):04d}.png', bbox_inches='tight', pad_inches=0, dpi=200)
    
    # Close figure to avoid memory issues and warnings
    plt.close()

    
###################################################################################################################################################### 

def update_fish():
    
    global time1, agents, fish, fish_data, fish_data_MPA, total_hav_data, current_hav_data, fishermen, fishermen_data1, fishermen_data2, fishermen_data3    
    
    # Find all fish and select one randomly
    fish_list = [j for j in agents if j.type == 'fish']
    fish_ag = rd.choice(fish_list)
    
    # Filter fish neighbors by zone - compute distance only once for each neighbor
    neighbors = []
    for nb in fish_list:
        if nb == fish_ag:
            continue
            
        # Calculate distance squared only once
        dist_squared = (fish_ag.x - nb.x)**2 + (fish_ag.y - nb.y)**2
        
        if dist_squared < rad_repulsion_sqr:
            neighbors.append((nb, 'repulsion', dist_squared))
        elif dist_squared < rad_orientation_sqr:
            neighbors.append((nb, 'alignment', dist_squared))
        elif dist_squared < rad_attraction_sqr:
            neighbors.append((nb, 'attraction', dist_squared))
    
    # Filter by zone
    repulsion = [n[0] for n in neighbors if n[1] == 'repulsion']
    alignment = [n[0] for n in neighbors if n[1] == 'alignment']
    attraction = [n[0] for n in neighbors if n[1] == 'attraction']
    
    # Determine movement direction
    if repulsion:
        # Move away from center of mass of fish in repulsion zone
        repulsion_x = sum(j.x for j in repulsion) / len(repulsion)
        repulsion_y = sum(j.y for j in repulsion) / len(repulsion)
        theta = (math.atan2((repulsion_y - fish_ag.y), (repulsion_x - fish_ag.x)) + math.pi) % (2 * math.pi)
        
    elif alignment:
        # Match average direction of fish in alignment zone
        theta = sum(math.atan2((j.y - fish_ag.y), (j.x - fish_ag.x)) for j in alignment) / len(alignment)
        
    elif attraction:
        # Move toward center of mass of fish in attraction zone
        attraction_x = sum(j.x for j in attraction) / len(attraction)
        attraction_y = sum(j.y for j in attraction) / len(attraction)
        theta = math.atan2((attraction_y - fish_ag.y), (attraction_x - fish_ag.x))
        
    else:
        # Random movement if no neighbors
        theta = 2 * math.pi * rd.random()
    
    # Update position
    dx = move_fish * math.cos(theta)
    dy = move_fish * math.sin(theta)
    fish_ag.x += dx
    fish_ag.y += dy
    
    # Handle boundary conditions - wrap around edges
    if fish_ag.x > Half_Length_Area:
        fish_ag.x %= -Half_Length_Area
    elif fish_ag.x < -Half_Length_Area:
        fish_ag.x %= Half_Length_Area
        
    if fish_ag.y > Half_Length_Area:
        fish_ag.y %= -Half_Length_Area
    elif fish_ag.y < -Half_Length_Area:
        fish_ag.y %= Half_Length_Area
    
    # Handle fish reproduction
    current_fish_count = sum(1 for j in agents if j.type == 'fish')
    if rd.random() < growth_prob * (1 - current_fish_count/float(K)):
        # Create a new fish by copying attributes from the current one
        new_fish = agent()
        new_fish.type = 'fish'
        new_fish.x = fish_ag.x
        new_fish.y = fish_ag.y
        agents.append(new_fish)
       
######################################################################################################################################################
# Helper functions for fishermen movement and harvesting

def calculate_distance_squared(x1, y1, x2, y2):
    """Calculate squared distance between two points"""
    return (x1 - x2)**2 + (y1 - y2)**2

def find_fish_neighbors(fisherman, agents, r_squared, in_mpa_func=None):
    """Find fish neighbors of a fisherman within radius r"""
    fish_neighbors = []
    for nb in agents:
        if nb.type == 'fish':
            # Skip fish in MPA if an MPA check function is provided
            if in_mpa_func and in_mpa_func(nb.x, nb.y):
                continue
            
            # Check if within radius
            dist_squared = calculate_distance_squared(fisherman.x, fisherman.y, nb.x, nb.y)
            if dist_squared < r_squared:
                fish_neighbors.append(nb)
    return fish_neighbors

def find_fishermen_neighbors(fisherman, fishers_list, r_squared):
    """Find neighboring fishermen within radius r"""
    fishers_neighbors = []
    for nb in fishers_list:
        if nb is not fisherman:  # Use identity comparison for better performance
            dist_squared = calculate_distance_squared(fisherman.x, fisherman.y, nb.x, nb.y)
            if dist_squared < r_squared:
                fishers_neighbors.append((nb.harvest, nb))
    return fishers_neighbors

def harvest_fish(fisherman, fish_neighbors, agents):
    """Calculate and perform fish harvest"""
    if not fish_neighbors:
        return 0
        
    # Calculate harvest amount safely
    num_fish_harvest = min(len(fish_neighbors), int(round(q * fisherman.effort * len(fish_neighbors))))
    
    # Safe sampling and processing
    if num_fish_harvest > 0:
        sample_fish_harvest = rd.sample(fish_neighbors, num_fish_harvest)
        for fish in sample_fish_harvest:
            agents.remove(fish)
            fisherman.harvest += 1
        return num_fish_harvest
    return 0

def enforce_boundaries(x, y):
    """Enforce boundary conditions with wrap-around"""
    if x > Half_Length_Area:
        x = -Half_Length_Area
    elif x < -Half_Length_Area:
        x = Half_Length_Area
        
    if y > Half_Length_Area:
        y = -Half_Length_Area
    elif y < -Half_Length_Area:
        y = Half_Length_Area
    return x, y

def calculate_movement(fisherman, fishers_neighbors):
    """Calculate movement direction based on neighbors"""
    # Default is random direction
    theta = 2 * math.pi * rd.random()
    
    # If we have neighbors with better harvests, move toward the best one
    if fishers_neighbors and any(harvest > fisherman.harvest for harvest, _ in fishers_neighbors):
        # Find fisherman with greatest catch
        max_harvest_fisher = max(fishers_neighbors, key=lambda x: x[0])
        if max_harvest_fisher[0] > fisherman.harvest:
            target_fisher = max_harvest_fisher[1]
            dx = target_fisher.x - fisherman.x
            dy = target_fisher.y - fisherman.y
            theta = math.atan2(dy, dx)
    
    return theta
                         
def no_mpa():
    """Update fisherman behavior when no MPA is present"""
    global time1, agents, fish, fish_data, fish_data_MPA, total_hav_data, current_hav_data, fishermen, fishermen_data1, fishermen_data2, fishermen_data3 
    
    # Get all fishermen and select one randomly
    fishers_list = [j for j in agents if j.type == 'fishers']
    fisherman_ag = rd.choice(fishers_list)
    
    # Find fish neighbors and harvest them
    fish_neighbors = find_fish_neighbors(fisherman_ag, agents, r_sqr)
    harvest_fish(fisherman_ag, fish_neighbors, agents)
    
    # Find fisher neighbors and decide movement
    fishers_neighbors = find_fishermen_neighbors(fisherman_ag, fishers_list, r_sqr)
    theta = calculate_movement(fisherman_ag, fishers_neighbors)
    
    # Update position
    fisherman_ag.x += move_fishers * math.cos(theta)
    fisherman_ag.y += move_fishers * math.sin(theta)
    
    # Apply boundary conditions
    fisherman_ag.x, fisherman_ag.y = enforce_boundaries(fisherman_ag.x, fisherman_ag.y)

   
###################################################################################################################################################### 

def single_mpa():
    """Update fisherman behavior with single MPA"""
    global time1, agents, fish, fish_data, fish_data_MPA, total_hav_data, current_hav_data, fishermen, fishermen_data1, fishermen_data2, fishermen_data3   
    
    # Get all fishermen and select one randomly
    fishers_list = [j for j in agents if j.type == 'fishers']
    fisherman_ag = rd.choice(fishers_list)
    
    # Define MPA check function for this scenario
    def is_in_mpa(x, y):
        return (Xa <= x <= Xb) and (Ya <= y <= Yb)
    
    # Find fish neighbors and harvest them
    fish_neighbors = find_fish_neighbors(fisherman_ag, agents, r_sqr, in_mpa_func=is_in_mpa)
    harvest_fish(fisherman_ag, fish_neighbors, agents)
    
    # Find fisher neighbors and decide movement
    fishers_neighbors = find_fishermen_neighbors(fisherman_ag, fishers_list, r_sqr)
    
    # Determine movement direction
    if not fishers_neighbors:  # No neighbors
        move_outside_mpa_random(fisherman_ag, is_in_mpa)
    else:
        # Try to move toward fisher with highest catch
        move_toward_best_catch_avoiding_mpa(fisherman_ag, fishers_neighbors, is_in_mpa)

# Helper function for MPA-aware movement
def move_outside_mpa_random(fisherman, is_in_mpa_func, max_attempts=5):
    """Move fisherman in a random direction that avoids MPA areas"""
    attempt = 0
    while attempt < max_attempts:
        # Try random direction
        theta = 2 * math.pi * rd.random()
        dx = move_fishers * math.cos(theta)
        dy = move_fishers * math.sin(theta)
        
        # Check if new position would be in MPA
        new_x = fisherman.x + dx
        new_y = fisherman.y + dy
        
        if not is_in_mpa_func(new_x, new_y):
            fisherman.x = new_x
            fisherman.y = new_y
            fisherman.x, fisherman.y = enforce_boundaries(fisherman.x, fisherman.y)
            return True
        
        attempt += 1
    return False

def move_toward_best_catch_avoiding_mpa(fisherman, fishers_neighbors, is_in_mpa_func, max_attempts=5):
    """Move fisherman toward the fisher with best catch while avoiding MPA areas"""
    # Check if any neighbor has better harvest
    if any(harvest > fisherman.harvest for harvest, _ in fishers_neighbors):
        # Find fisherman with greatest catch
        max_harvest_fisher = max(fishers_neighbors, key=lambda x: x[0])
        target_fisher = max_harvest_fisher[1]
        
        # Calculate direction toward target
        dx = target_fisher.x - fisherman.x
        dy = target_fisher.y - fisherman.y
        theta = math.atan2(dy, dx)
        
        # Try to move toward target
        dx = move_fishers * math.cos(theta)
        dy = move_fishers * math.sin(theta)
        new_x = fisherman.x + dx
        new_y = fisherman.y + dy
        
        # If new position is outside MPA, move there
        if not is_in_mpa_func(new_x, new_y):
            fisherman.x = new_x
            fisherman.y = new_y
            fisherman.x, fisherman.y = enforce_boundaries(fisherman.x, fisherman.y)
            return True
        else:
            # Otherwise try random directions
            return move_outside_mpa_random(fisherman, is_in_mpa_func, max_attempts)
    else:
        # Move in random direction outside MPA
        return move_outside_mpa_random(fisherman, is_in_mpa_func, max_attempts)
                            
######################################################################################################################################################                                

def spaced_mpa():
    """Update fisherman behavior with multiple spaced MPAs"""
    global time1, agents, fish, fish_data, fish_data_MPA, total_hav_data, current_hav_data, fishermen, fishermen_data1, fishermen_data2, fishermen_data3   
    
    # Get all fishermen and select one randomly
    fishers_list = [j for j in agents if j.type == 'fishers']
    fisherman_ag = rd.choice(fishers_list)
    
    # Define MPA check function for spaced MPAs
    def is_in_mpa(x, y):
        in_mpa1 = (Xm <= x <= Xn) and (Ym <= y <= Yn)
        in_mpa2 = (Xp <= x <= Xq) and (Yp <= y <= Yq)
        return in_mpa1 or in_mpa2
    
    # Find fish neighbors and harvest them
    fish_neighbors = find_fish_neighbors(fisherman_ag, agents, r_sqr, in_mpa_func=is_in_mpa)
    harvest_fish(fisherman_ag, fish_neighbors, agents)
    
    # Find fisher neighbors and decide movement
    fishers_neighbors = find_fishermen_neighbors(fisherman_ag, fishers_list, r_sqr)
    
    # Determine movement direction 
    if not fishers_neighbors:  # No neighbors
        move_outside_mpa_random(fisherman_ag, is_in_mpa)
    else:
        # Try to move toward fisher with highest catch
        move_toward_best_catch_avoiding_mpa(fisherman_ag, fishers_neighbors, is_in_mpa)
   
######################################################################################################################################################                                 

def update_one_unit_time():
    
    global time1, agents, fish, fish_data, fish_data_MPA, total_hav_data, current_hav_data, fishermen, fishermen_data1, fishermen_data2, fishermen_data3  
    time1 += 1  # update time
    
    # Optimize the fish update loop
    fish_agents = [j for j in agents if j.type == 'fish']
    fish = fish_agents  # Update the global fish variable
    
    # Only run fish update if there are fish
    if fish_agents:
        fish_count = len(fish_agents)
        update_steps = min(fish_count, 100)  # Cap the number of updates for performance
        step_size = 1.0 / update_steps
        
        for _ in range(update_steps):
            update_fish()
    
    # Get fishermen for later use
    fishermen = [j for j in agents if j.type == 'fishers']
    fishermen_count = len(fishermen)
    
    # Helper functions to avoid code duplication
    def run_fishermen_updates(update_func, steps=100):
        step_size = 1.0 / min(steps, fishermen_count)
        t = 0.0
        while t < 1.0:
            t += step_size
            update_func()
    
    def count_fish_in_single_mpa():
        return sum(1 for j in agents if j.type == 'fish' and 
                  (Xa <= j.x <= Xb) and (Ya <= j.y <= Yb))
    
    def count_fish_in_spaced_mpa():
        return sum(1 for j in agents if j.type == 'fish' and 
                  (((Xm <= j.x <= Xn) and (Ym <= j.y <= Yn)) or 
                   ((Xp <= j.x <= Xq) and (Yp <= j.y <= Yq))))
    
    # Pre-calculate the MPA scenario once
    use_no_mpa = False
    use_single_mpa = False
    use_spaced_mpa = False
    
    # Determine which MPA scenario to use
    if MPA == 'no' and Both == 'no':
        use_no_mpa = True
    elif MPA == 'yes' and Both == 'no':
        if Type_MPA == 'single':
            use_single_mpa = True
        else:  # 'spaced'
            use_spaced_mpa = True
    elif MPA == 'no' and Both == 'yes' and Type_MPA == 'single':
        if time1 < Time_MPA:
            use_single_mpa = True
        else:
            use_no_mpa = True
    elif MPA == 'no' and Both == 'yes' and Type_MPA == 'spaced':
        if time1 < Time_MPA:
            use_spaced_mpa = True
        else:
            use_no_mpa = True
    
    # Run the appropriate fisherman update routine
    if use_no_mpa:
        run_fishermen_updates(no_mpa)
        fish_data_MPA.append(0)  # No fish in MPA
    elif use_single_mpa:
        run_fishermen_updates(single_mpa)
        fish_data_MPA.append(count_fish_in_single_mpa())
    elif use_spaced_mpa:
        run_fishermen_updates(spaced_mpa)
        fish_data_MPA.append(count_fish_in_spaced_mpa())
    
    # Optimize data collection
    # Count total fish once instead of multiple times
    total_fish_count = len(fish_agents)
    fish_data.append(total_fish_count)
    
    # Process fishermen data more efficiently
    fishers_by_num = {j.num: j.harvest for j in fishermen}
    
    # Update harvest data in one pass
    for fisher_num, harvest in fishers_by_num.items():
        total_hav_data[fisher_num].append(harvest)
        # Calculate current catch (difference from previous total)
        prev_total = total_hav_data[fisher_num][-2]
        current_hav_data[fisher_num].append(harvest - prev_total)
    
    # Update summary statistics
    total_harvest = sum(j.harvest for j in fishermen)
    fishermen_data1.append(total_harvest)
    fishermen_data2.append(total_harvest - fishermen_data1[-2])
    fishermen_data3.append(total_fish_count - fish_data_MPA[-1])
    
    # Write data to CSV more efficiently
    csvfile = "simulation_data.csv"
    sorted_keys = sorted(current_hav_data.keys())
    
    # Create header and data in a more direct way
    header = sorted_keys + ['total_catch', 'total_biomass', 'biomass_inside_MPA', 'biomass_outside_MPA']
    
    # Prepare data columns
    data_columns = []
    for key in sorted_keys:
        data_columns.append(current_hav_data[key])
    
    data_columns.extend([fishermen_data2, fish_data, fish_data_MPA, fishermen_data3])
    
    # Write to CSV efficiently
    with open(csvfile, "w", newline='') as output:
        writer = csv.writer(output)
        writer.writerow(header)
        writer.writerows(zip(*data_columns))
       
######################################################################################################################################################       

# Print simulation start message
print("\n=== Starting CoopFishABM Simulation ===")
print(f"MPA: {MPA}, Type: {Type_MPA}, Both: {Both}")
print(f"Initial fish: {init_fish}, Fishermen: {num_fishers}")
print(f"Simulation steps: {n}\n")

# Initialize and observe initial state
initialize()  
print("Initialization complete")
observe()
print(f"Year 0: {len([j for j in agents if j.type == 'fish'])} fish")
    
# Main simulation loop with progress indication
for j in range(1, n):  
    update_one_unit_time()
    observe()
    
    # Print progress every 10 steps or at the final step
    if j % 10 == 0 or j == n-1:
        fish_count = len([ag for ag in agents if ag.type == 'fish'])
        mpa_fish = fish_data_MPA[-1]
        total_harvest = fishermen_data1[-1]
        print(f"Year {j}: {fish_count} fish ({mpa_fish} in MPA), Total harvest: {total_harvest}")

# Create the movie from the generated images
print("\nGenerating simulation video...")
os.system("ffmpeg -v quiet -r 5 -i year_%04d.png -vcodec mpeg4 -y -s:v 1920x1080 simulation_movie.mp4") 
print("Video created: simulation_movie.mp4")

# Final output summary
print("\n=== Simulation Complete ===")
print(f"Final fish population: {fish_data[-1]} fish")
print(f"Fish in MPA: {fish_data_MPA[-1]} fish")
print(f"Fish outside MPA: {fishermen_data3[-1]} fish")
print(f"Total harvest: {fishermen_data1[-1]} fish")
print(f"Data saved to: simulation_data.csv")


#------------------------------------------------------------------------------------------------------------------ 

os.chdir(os.pardir) # optional: move up to parent folder

#----------------------------------------------------------------------------------------------------------------
