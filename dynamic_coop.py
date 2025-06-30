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
# from pylab import *
import random as rd
import matplotlib.pyplot as plt
import csv 
from statistics import mean
from agents.fish import Fish
from agents.fisher import Fisher
from tqdm import tqdm
from parameters.parameters import BaseParameters, ModelParameters
from initialize import initialize
from plots.plotting import plot_summary, observe
from plots.create_gif import create_gif

######################################################################################################################################################  

def save_cooperation_data(model_parameters):
    """
    Saves cooperation data to csv to prevent loss due to technical issues.
    """
    with open('simulation_output/cooperation_data.csv', 'w') as f:
        writer = csv.writer(f)
        # Write header
        header = ['time', 'avg_cooperation'] + list(model_parameters.strategy_counts.keys())
        writer.writerow(header)
        
        # Write data
        for t in range(len(model_parameters.cooperation_levels)):
            row = [t, model_parameters.cooperation_levels[t]] + [model_parameters.strategy_counts[s][t] for s in model_parameters.strategy_counts]
            writer.writerow(row)

###################################################################################################################################################### 

def track_cooperation_levels(model_parameters):
    """Track the current levels of cooperation strategies."""
    
    # Count strategies
    counts = {
        'fully_noncoop': 0,
        'noncoop': 0,
        'cond_coop': 0,
        'coop': 0,
        'fully_coop': 0
    }
    
    fishers = [ag for ag in model_parameters.agents if ag.type == 'fishers']
    for fisher in fishers:
        counts[fisher.trait] += 1
    
    # Store counts
    for strategy in counts:
        model_parameters.strategy_counts[strategy].append(counts[strategy])
    
    # Calculate and store average cooperation level
    # Weight: fully_noncoop=0, noncoop=0.25, cond_coop=0.5, coop=0.75, fully_coop=1
    weights = {'fully_noncoop': 0, 'noncoop': 0.25, 'cond_coop': 0.5, 'coop': 0.75, 'fully_coop': 1}
    weighted_values = [counts[s] * weights[s] for s in weights.keys()]  # Create list first
    total_weight = sum(weighted_values)  # Then sum the list
    avg_cooperation = total_weight / len(fishers) if fishers else 0
    model_parameters.cooperation_levels.append(avg_cooperation)

def track_trust_metrics(model_parameters):
    """Track and store trust-related metrics."""
    fishers = [ag for ag in model_parameters.agents if ag.type == 'fishers']
    
    # Calculate average trust across all fisher pairs
    all_trust_scores = []
    for fisher in fishers:
        all_trust_scores.extend(fisher.trust_scores.values())
    
    avg_trust = mean(all_trust_scores) if all_trust_scores else 0
    model_parameters.trust_history.append(avg_trust)
    
    # Calculate cooperation rates using list comprehension instead of generators
    coop_actions = sum([f.cooperative_actions for f in fishers])
    total_actions = sum([f.total_actions for f in fishers])
    total_coop_rate = coop_actions / max(1, total_actions)
    
    return avg_trust, total_coop_rate


def save_trust_data(model_parameters):
    """Save trust and cooperation data to CSV."""
    with open('simulation_output/trust_data.csv', 'w') as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(['time', 'average_trust', 'cooperation_rate'])
        
        # Calculate cooperation rates for each time step
        fishers = [ag for ag in model_parameters.agents if ag.type == 'fishers']
        coop_actions = sum([f.cooperative_actions for f in fishers])
        total_actions = sum([f.total_actions for f in fishers])
        coop_rate = coop_actions / max(1, total_actions)
        
        # Write data
        for t in range(len(model_parameters.trust_history)):
            writer.writerow([t, model_parameters.trust_history[t], coop_rate])


def plot_trust_dynamics(model_parameters):
    """Plot trust and cooperation dynamics."""
    plt.figure(figsize=(10, 6))
    plt.plot(model_parameters.trust_history, 'b-', label='Average Trust')
    plt.xlabel('Time')
    plt.ylabel('Trust Level')
    plt.title('Evolution of Trust Between Fishers')
    plt.grid(True)
    plt.legend()
    plt.savefig('simulation_output/trust_dynamics.png')
    plt.close()

def update_one_unit_time(model_parameters, base_parameters):

    # First timestep: update fish, let fishers catch fish
    model_parameters.time += 1
    
    # Update fish positions and growth
    Fish.update_fish(model_parameters, base_parameters)
    Fisher.update_fisher(model_parameters, base_parameters)
    
    # Track trust metrics
    track_trust_metrics(model_parameters)
    
    # Update fishermen positions and catches
    fishers = [j for j in model_parameters.agents if j.type == 'fishers']
    for fisher in fishers:
        fish_neighbors = [nb for nb in model_parameters.agents if nb.type == 'fish' and ((fisher.x - nb.x) ** 2 + (
                    fisher.y - nb.y) ** 2) < base_parameters.r_sqr]  # detecting fishes in neighbourhood
        num_fish_harvest = int(round(base_parameters.q * fisher.effort * len(
            fish_neighbors)))  # number of fish to be harvested based on (q*E*x), where x is number of fishes in neighborhood
        if fish_neighbors and num_fish_harvest > 0:
            sample_fish_harvest = rd.sample(fish_neighbors, min(num_fish_harvest, len(fish_neighbors)))
            for j in sample_fish_harvest:
                model_parameters.agents.remove(j)  # remove catch
                fisher.harvest += 1  # add to catch of a fisherman
        else:
            model_parameters.current_hav_data[fisher.num].append(0)
    for fisher in fishers:
        fisher.move(model_parameters, base_parameters)

    # Next timestep: update stats, fishers imitate strategies
    model_parameters.time += 1
    
    model_parameters.fishermen_data1.append(sum([j.harvest for j in model_parameters.agents if j.type == 'fishers']))
    model_parameters.fishermen_data2.append(sum([model_parameters.current_hav_data[j.num][-1] for j in model_parameters.agents if j.type == 'fishers']))
   
    # Imitation and tracking
    if model_parameters.time % base_parameters.imitation_period == 0:  # Every X time steps
        Fisher.imitate_successful_strategies(model_parameters, base_parameters)
    track_cooperation_levels(model_parameters)
    
    # Save data to CSV
    csvfile = "simulation_output/simulation_data.csv"   # a csv-file output 
    header = [key for key in sorted(model_parameters.current_hav_data)]
    header.append('total_catch')
    header.append('total_biomass')
    main_data = [model_parameters.current_hav_data[key] for key in sorted(model_parameters.current_hav_data)]
    main_data.append(model_parameters.fishermen_data2)
    main_data.append(model_parameters.total_fish_count)
    with open(csvfile, "w") as output:
        writer = csv.writer(output) 
        writer.writerow(header)
        writer.writerows(zip(*main_data))


def run_model(model_params, base_params, t_max=150, experiment_label='default', create_plots=False):
    initialize(experiment_label, base_params, model_params)
    observe(base_params, model_params)

    print("Starting simulation...")
    for j in tqdm(range(1, t_max), desc="Simulating", unit="step"):
        update_one_unit_time(model_params, base_params)
        observe(model_parameters=model_params, base_parameters=base_params)

    if create_plots:
        plot_summary(model_params)
        save_cooperation_data(model_params)  # Save cooperation data
        save_trust_data(model_params)  # Save trust data

        print('Creating gif...')
        create_gif()
    return model_params

######################################################################################################################################################       

if __name__ == "__main__":
    base_parameters = BaseParameters()
    model_parameters = ModelParameters()
    run_model(model_params=model_parameters, base_params=base_parameters, experiment_label='both', create_plots=True)

# Remove or comment out the ffmpeg video creation line at the end
# os.system("ffmpeg -v quiet -r 5 -i year_%04d.png -vcodec mpeg4  -y -s:v 1920x1080 simulation_movie.mp4")


#------------------------------------------------------------------------------------------------------------------ 

# os.chdir(os.pardir) # optional: move up to parent folder

#----------------------------------------------------------------------------------------------------------------
