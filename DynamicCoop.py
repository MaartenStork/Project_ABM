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
from tqdm import tqdm 
plt.ion()

#---------------------------------------------------------------------------

# Parameters #

# Trust and loyalty parameters
initial_trust = 0.5  # Initial trust score between fishers
trust_increase = 0.1  # How much trust increases when seeing cooperation
trust_decrease = 0.2  # How much trust decreases when seeing non-cooperation
trust_radius = 0.3   # Radius within which fishers observe each other's behavior
trust_memory = 5     # How many time steps to remember trust changes
trust_threshold = 0.6  # Trust threshold for cooperative behavior

# Fishing ground and time #
K = 200 # carrying capacity of fishing ground
n = 150 # number of simulation time steps

# Threshold-based behavior parameters
fish_density_threshold = 3.0  # threshold for local fish density that triggers cooperation
threshold_radius = 0.4  # radius to check local fish density
threshold_memory = 5  # how many time steps to remember low density before switching back
cooperation_increase = 0.2  # how much to increase cooperation when threshold is crossed

# Imitation parameters
imitation_period = 5  # how often agents compare and potentially imitate (every X time steps)
imitation_radius = 0.3  # radius within which agents can observe others' success
imitation_prob = 0.3  # probability of imitating when a more successful strategy is found

# Cooperation level tracking
cooperation_levels = []  # track average cooperation level over time
strategy_counts = {'fully_noncoop': [], 'noncoop': [], 'cond_coop': [], 'coop': [], 'fully_coop': []}

# Attributes of fish agents #
base_growth_prob = 0.3    # base growth rate
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

# Live plotting parameters
plot_update_freq = 5  # Update plot every X steps

#######################################################################################################################################################  

class agent:  # create an empty class
    pass     
    
#----------------------------------------------------------------------------------------------------------    

def initialize():
    global time1, agents, fish, fish_data, fish_data_MPA, total_hav_data, current_hav_data, fishermen, fishermen_data1, fishermen_data2, fishermen_data3
    time1 = 0.
    agents = []
    fish_data = [init_fish]
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
    
    # Add fish agents
    for j in range(init_fish):
        ag = agent()
        ag.type = 'fish'
        ag.x = rd.uniform(-Half_Length_Area, Half_Length_Area)
        ag.y = rd.uniform(-Half_Length_Area, Half_Length_Area)
        agents.append(ag)

#----------------------------------------------------------------------------------------------------------    
                                
    # Initialise the number of fishes in an MPA 
    if (MPA == 'no' and Both == 'no') :
        fish_data_MPA = [0] #  a zero because no mpa is available
        
    elif any([(MPA == 'yes' and Type_MPA == 'single' and Both == 'no'),(MPA == 'no' and Both == 'yes' and Type_MPA == 'single')]): # only single MPA , partly single MPA & partly no MPA
        fish_data_MPA = [sum([1 for j in agents if j.type == 'fish' and  ((Xa <= j.x <= Xb) and (Ya <= j.y <= Yb))])]
        
    elif any([(MPA == 'yes' and Type_MPA == 'spaced' and Both == 'no'), (MPA == 'no' and Both == 'yes' and Type_MPA == 'spaced')]):
        fish_data_MPA = [sum([1 for j in agents if j.type == 'fish' and any([((Xm <= j.x <= Xn) and (Ym <= j.y <= Yn)), ((Xp <= j.x <= Xq) and (Yp <= j.y <= Yq))])])]
    
    fishermen_data3 = [fish_data[-1] - fish_data_MPA[-1]] # initialise number of fishes outside MPA
    
######################################################################################################################################################    
        
def observe():
    global time1, agents
    plt.clf()
    plt.figure(figsize=(10, 10))  # Set a fixed figure size
    plt.subplot(111, facecolor='lightskyblue')
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
        plt.plot(X_fully_coop, Y_fully_coop, 'o', color = my_colors[4], markersize=7.5, label='fully_coop')
        plt.plot(X_coop, Y_coop, 'o', color = my_colors[3], markersize=7.5, label='coop')
        plt.plot(X_cond_coop, Y_cond_coop, 'o', color = my_colors[2], markersize=7.5, label='conditional_coop')
        plt.plot(X_noncoop, Y_noncoop,  'o', color = my_colors[1], markersize=7.5, label='noncoop')
        plt.plot(X_fully_noncoop, Y_fully_noncoop, 'o', color = my_colors[0], markersize=7.5, label='fully_noncoop')
    fish = [ag for ag in agents if ag.type == 'fish']
    if len(fish) > 0:
        X_fish = [ag.x for ag in fish]
        Y_fish = [ag.y for ag in fish]
        plt.plot(X_fish, Y_fish, '^', color='darkgreen', markersize=3, label='fish')
    if any([(MPA == 'yes' and Type_MPA == 'single' and Both == 'no'), (MPA == 'no' and Both == 'yes' and Type_MPA =='single' and time1 <= Time_MPA)]):
        plt.vlines(Xa, Ya, Yb, lw=2, color='k')
        plt.vlines(Xb, Ya, Yb, lw=2, color='k')
        plt.hlines(Ya, Xa, Xb, lw=2, color='k')
        plt.hlines(Yb, Xa, Xb, lw=2, color='k')
    elif any([(MPA == 'yes' and Type_MPA == 'spaced' and Both == 'no'), (MPA == 'no' and Both == 'yes' and Type_MPA =='spaced' and time1 <= Time_MPA)]):
        plt.vlines(Xm, Ym, Yn, lw=2, color='k')
        plt.vlines(Xn, Ym, Yn, lw=2, color='k')
        plt.hlines(Ym, Xm, Xn, lw=2, color='k')
        plt.hlines(Yn, Xm, Xn, lw=2, color='k')
        plt.vlines(Xp, Yp, Yq, lw=2, color='k')
        plt.vlines(Xq, Yp, Yq, lw=2, color='k')
        plt.hlines(Yp, Xp, Xq, lw=2, color='k')
        plt.hlines(Yq, Xp, Xq, lw=2, color='k')
    axis('image')
    axis([-Half_Length_Area, Half_Length_Area,-Half_Length_Area, Half_Length_Area])
    plt.grid(False)
    plt.xticks([], [])
    plt.yticks([], [])
    plt.title('year =' + str(int(time1)))
    plt.legend(numpoints=1, loc='center', bbox_to_anchor=(0.5, -0.072), ncol=3, prop={'size':11}, facecolor='lightskyblue')
    plt.savefig(f'simulation_output/year_{int(time1):04d}.png', bbox_inches='tight', pad_inches=0, dpi=200)
    plt.close()

def plot_summary():
    global time1, agents, fish_data, fish_data_MPA, fishermen_data1, fishermen_data2, fishermen_data3
    plt.figure(figsize=(15, 20))
    
    # Plot 1: Fish population dynamics
    plt.subplot(4, 1, 1)
    plt.plot(fish_data, 'b-', label='Total fish population')
    plt.plot(fish_data_MPA, 'g-', label='Fish in MPA')
    plt.plot(fishermen_data3, 'r-', label='Fish outside MPA')
    plt.xlabel('Time')
    plt.ylabel('Number of fish')
    plt.title('Fish Population Dynamics')
    plt.legend()
    
    # Plot 2: Fishing activity
    plt.subplot(4, 1, 2)
    plt.plot(fishermen_data1, 'b-', label='Total catch')
    plt.plot(fishermen_data2, 'r-', label='Current catch')
    plt.xlabel('Time')
    plt.ylabel('Number of fish caught')
    plt.title('Fishing Activity')
    plt.legend()
    
    # Plot 3: Cooperation levels
    plt.subplot(4, 1, 3)
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
    
    # Plot 4: Trust dynamics
    plt.subplot(4, 1, 4)
    plt.plot(trust_history, 'b-', label='Average Trust')
    plt.xlabel('Time')
    plt.ylabel('Trust Level')
    plt.title('Evolution of Trust Between Fishers')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('simulation_output/dynamics.png', bbox_inches='tight', dpi=200)
    
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
    
    global time1, agents, fish, fish_data, fish_data_MPA, total_hav_data, current_hav_data, fishermen , fishermen_data1,  fishermen_data2, fishermen_data3    
    fish_list = [j for j in agents if j.type == 'fish']
    if not fish_list:
        return
    fish_ag = rd.sample(fish_list, 1)[-1]
    
    repulsion = [nb for nb in agents if nb.type == 'fish' and nb != fish_ag and ((fish_ag.x - nb.x)**2 + (fish_ag.y - nb.y)**2) < rad_repulsion_sqr] # fishes within the repulsion zone
    alignment = [nb for nb in agents if nb.type == 'fish' and nb != fish_ag and rad_repulsion_sqr < ((fish_ag.x - nb.x)**2 + (fish_ag.y - nb.y)**2) < rad_orientation_sqr ] # fishes within the parallel-orientation zone
    attraction =[nb for nb in agents if nb.type == 'fish' and nb != fish_ag and rad_orientation_sqr < ((fish_ag.x - nb.x)**2 + (fish_ag.y - nb.y)**2) < rad_attraction_sqr ] # fishes within the attraction zone
    
    if len(repulsion) > 0: # if fishes within repulsion zone, move away from the spot that would be the center of mass (midpoint) of all  fish within repulsion zone
        repulsion_x = mean([j.x for j in repulsion])
        repulsion_y = mean([j.y for j in repulsion])
        repulsion_1 = (math.atan2((repulsion_y - fish_ag.y), (repulsion_x - fish_ag.x)) + math.pi ) % (2 * math.pi) # if greater than  (2 * math.pi) then compute with a minus
        theta = repulsion_1
        fish_ag.x +=  move_fish*math.cos(theta)     # moves 'move_fish' step    
        fish_ag.y +=  move_fish*math.sin(theta) 
        fish_ag.x = (fish_ag.x % -Half_Length_Area) if fish_ag.x > Half_Length_Area else (fish_ag.x % Half_Length_Area) if fish_ag.x < -Half_Length_Area else fish_ag.x  # ( When fish-agent approach a border of the landscape, 
        fish_ag.y = (fish_ag.y % -Half_Length_Area) if fish_ag.y > Half_Length_Area else (fish_ag.y % Half_Length_Area) if fish_ag.y < -Half_Length_Area else fish_ag.y  # they re-enter the system at the opposite border )
    
    elif all([len(repulsion) == 0, len(alignment) > 0]):   # if fishes within parallel-orientation zone, change direction to match the average direction of all the other fish  within parallel-orientation zone     
        alignment_1 = mean([math.atan2((j.y - fish_ag.y),(j.x - fish_ag.x)) for j in alignment])
        theta = alignment_1
        fish_ag.x +=   math.cos(theta)     # moves 'move_fish' step,  move_fish*math.cos(theta)
        fish_ag.y +=   math.sin(theta)  
        fish_ag.x = (fish_ag.x % -Half_Length_Area) if fish_ag.x > Half_Length_Area else (fish_ag.x % Half_Length_Area) if fish_ag.x < -Half_Length_Area else fish_ag.x  # ( When fish-agent approach a border of the landscape, 
        fish_ag.y = (fish_ag.y % -Half_Length_Area) if fish_ag.y > Half_Length_Area else (fish_ag.y % Half_Length_Area) if fish_ag.y < -Half_Length_Area else fish_ag.y  # they re-enter the system at the opposite border )

    elif all([len(repulsion) == 0, len(alignment) == 0, len(attraction) > 0]): # if fishes within only the attraction zone, head towards the middle (midpoint) of the fishes in zone of attraction.   
        attraction_x = mean([j.x for j in attraction ])
        attraction_y = mean([j.y for j in attraction])
        attraction_1 = math.atan2((attraction_y - fish_ag.y), (attraction_x - fish_ag.x))
        theta = attraction_1
        fish_ag.x +=  move_fish*math.cos(theta)     # moves 'move_fish' step      
        fish_ag.y +=  move_fish*math.sin(theta) 
        fish_ag.x = (fish_ag.x % -Half_Length_Area) if fish_ag.x > Half_Length_Area else (fish_ag.x % Half_Length_Area) if fish_ag.x < -Half_Length_Area else fish_ag.x  # ( When fish-agent approach a border of the landscape, 
        fish_ag.y = (fish_ag.y % -Half_Length_Area) if fish_ag.y > Half_Length_Area else (fish_ag.y % Half_Length_Area) if fish_ag.y < -Half_Length_Area else fish_ag.y  # they re-enter the system at the opposite border )

    elif all([len(repulsion) == 0, len(alignment) == 0, len(attraction) == 0]): # if no fishes in all the zone, move in a random direction  
        theta = 2*math.pi*rd.random()  
        fish_ag.x +=  move_fish*math.cos(theta)     # moves 'move_fish' step     
        fish_ag.y +=  move_fish*math.sin(theta) 
        fish_ag.x = (fish_ag.x % -Half_Length_Area) if fish_ag.x > Half_Length_Area else (fish_ag.x % Half_Length_Area) if fish_ag.x < -Half_Length_Area else fish_ag.x  # ( When fish-agent approach a border of the landscape, 
        fish_ag.y = (fish_ag.y % -Half_Length_Area) if fish_ag.y > Half_Length_Area else (fish_ag.y % Half_Length_Area) if fish_ag.y < -Half_Length_Area else fish_ag.y  # they re-enter the system at the opposite border )
                                       
    if rd.random() < base_growth_prob * (1-sum([1 for j in agents if j.type == 'fish'])/float(K)):  # logistic growth of fishes
        agents.append(cp.copy(fish_ag)) # add-copy of fish agent  
       
######################################################################################################################################################                         
                  
def no_mpa():
    
    global time1, agents, fish, fish_data, fish_data_MPA, total_hav_data, current_hav_data, fishermen, fishermen_data1,  fishermen_data2, fishermen_data3 
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
    
    global time1, agents, fish, fish_data, fish_data_MPA, total_hav_data, current_hav_data, fishermen, fishermen_data1,  fishermen_data2, fishermen_data3   
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
    
    global time1, agents, fish, fish_data, fish_data_MPA, total_hav_data, current_hav_data, fishermen , fishermen_data1,  fishermen_data2, fishermen_data3   
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
    global time1, agents, fish, fish_data, fish_data_MPA, total_hav_data, current_hav_data, fishermen, fishermen_data1, fishermen_data2, fishermen_data3
    
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
    for fisher in [j for j in agents if j.type == 'fishers']:
        # Calculate catch
        if any([(j.type == 'fish') and ((j.x - fisher.x) ** 2 + (j.y - fisher.y) ** 2) <= r_sqr for j in agents]):
            if rd.random() < q * fisher.effort:
                fisher.harvest += 1
                total_hav_data[fisher.num].append(fisher.harvest)
                current_hav_data[fisher.num].append(1)
                
                # Remove caught fish
                for fish in [j for j in agents if j.type == 'fish']:
                    if ((fish.x - fisher.x) ** 2 + (fish.y - fisher.y) ** 2) <= r_sqr:
                        agents.remove(fish)
                        break
            else:
                current_hav_data[fisher.num].append(0)
        else:
            current_hav_data[fisher.num].append(0)
            
        # Update fisherman position
        if MPA == 'no' and Both == 'no':
            no_mpa()
        elif any([(MPA == 'yes' and Type_MPA == 'single' and Both == 'no'), (MPA == 'no' and Both == 'yes' and Type_MPA == 'single')]):
            single_mpa()
        elif any([(MPA == 'yes' and Type_MPA == 'spaced' and Both == 'no'), (MPA == 'no' and Both == 'yes' and Type_MPA == 'spaced')]):
            spaced_mpa()
    
    # Update time and data
    time1 += 1
    fish_data.append(sum([1 for j in agents if j.type == 'fish']))
    
    # Update MPA fish count
    if MPA == 'no' and Both == 'no':
        fish_data_MPA.append(0)
    elif any([(MPA == 'yes' and Type_MPA == 'single' and Both == 'no'), (MPA == 'no' and Both == 'yes' and Type_MPA == 'single')]):
        fish_data_MPA.append(sum([1 for j in agents if j.type == 'fish' and ((Xa <= j.x <= Xb) and (Ya <= j.y <= Yb))]))
    elif any([(MPA == 'yes' and Type_MPA == 'spaced' and Both == 'no'), (MPA == 'no' and Both == 'yes' and Type_MPA == 'spaced')]):
        fish_data_MPA.append(sum([1 for j in agents if j.type == 'fish' and any([((Xm <= j.x <= Xn) and (Ym <= j.y <= Yn)), ((Xp <= j.x <= Xq) and (Yp <= j.y <= Yq))])]))
    
    fishermen_data1.append(sum([j.harvest for j in agents if j.type == 'fishers']))
    fishermen_data2.append(sum([current_hav_data[j.num][-1] for j in agents if j.type == 'fishers']))
    fishermen_data3.append(fish_data[-1] - fish_data_MPA[-1])
   
    # Imitation and tracking
    if time1 % imitation_period == 0:  # Every X time steps
        imitate_successful_strategies()
    track_cooperation_levels()
    
    # Save data to CSV
    csvfile = "simulation_output/simulation_data.csv"   # a csv-file output 
    header = [key for key in sorted(current_hav_data)]
    header.append('total_catch') ; header.append('total_biomass') ; header.append('biomass_inside_MPA') ; header.append('biomass_outside_MPA')
    main_data = [current_hav_data[key] for key in sorted(current_hav_data)]
    main_data.append(fishermen_data2) ; main_data.append(fish_data) ; main_data.append(fish_data_MPA) ; main_data.append(fishermen_data3)
    with open(csvfile, "w") as output:
        writer = csv.writer(output) 
        writer.writerow(header)
        writer.writerows(zip(*main_data))

def setup_live_plot():
    """Setup the live plotting figure."""
    plt.ion()  # Enable interactive plotting
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
    ax1, ax2 = axes
    pop_line, mpa_line, outside_line, total_catch_line, current_catch_line = lines
    
    # Get time range
    times = list(range(step + 1))
    
    # Update fish population lines
    pop_line.set_data(times, fish_data)
    mpa_line.set_data(times, fish_data_MPA)
    outside_line.set_data(times, fishermen_data3)
    
    # Update catch lines
    total_catch_line.set_data(times, fishermen_data1)
    current_catch_line.set_data(times, fishermen_data2)
    
    # Adjust axes limits
    ax1.set_xlim(0, max(n, step + 1))
    ax1.set_ylim(0, max(max(fish_data), max(fish_data_MPA), max(fishermen_data3)) * 1.1)
    
    ax2.set_xlim(0, max(n, step + 1))
    ax2.set_ylim(0, max(max(fishermen_data1), max(fishermen_data2)) * 1.1)
    
    plt.draw()
    plt.pause(0.01)

######################################################################################################################################################       

initialize()
observe()

# Setup live plot
axes, plot_lines = setup_live_plot()

print("Starting simulation...")
for j in tqdm(range(1, n), desc="Simulating", unit="step"):
    update_one_unit_time()
    observe()
    
    # Update live plot periodically
    if j % plot_update_freq == 0:
        update_live_plot(axes, plot_lines, j)

# Final plot update
update_live_plot(axes, plot_lines, n-1)

plot_summary()
save_cooperation_data()  # Save cooperation data
save_trust_data()  # Save trust data

plt.ioff()  # Disable interactive mode
plt.show()  # Keep the final plot window open

# Remove or comment out the ffmpeg video creation line at the end
# os.system("ffmpeg -v quiet -r 5 -i year_%04d.png -vcodec mpeg4  -y -s:v 1920x1080 simulation_movie.mp4")


#------------------------------------------------------------------------------------------------------------------ 

# os.chdir(os.pardir) # optional: move up to parent folder

#----------------------------------------------------------------------------------------------------------------
