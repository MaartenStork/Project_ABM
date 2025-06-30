import os
import glob
import random as rd
from itertools import product
from agents.fisher import Fisher
from agents.fish import Fish


def delete_prev_sim():
    """
    Removes model state images from the previous simulation if they exist.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sim_output_dir = os.path.join(script_dir, "simulation_output")
    pattern = os.path.join(sim_output_dir, "year_*.png")

    # Iterate and remove matching files
    for file_path in glob.glob(pattern):
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")


def assign_cooperation(ag, n, base_parameters):
    """
    Assigns a cooperation strategy to an agentbased on the number of agents defined for
    each strategy.

    Args:
    n (int): The index of the agent to assign a cooperation strategy to (0-based).
    base_parameters (BaseParameters): Contains the number of agents for each
        cooperation strategy type (e.g., fully_noncoop, noncoop, cond_coop, etc.).
    """    
    fully_noncoop = base_parameters.fully_noncoop
    noncoop = base_parameters.noncoop
    cond_coop = base_parameters.cond_coop
    coop = base_parameters.coop

    if n < (fully_noncoop):
        ag.effort = 1.0
        ag.trait = 'fully_noncoop'
        ag.num = f'fully_noncoop{1 + n}'
    elif (base_parameters.fully_noncoop) <= n < (fully_noncoop + noncoop):
        ag.effort = 0.8
        ag.trait = 'noncoop'
        ag.num = f'noncoop{(1 + n) - fully_noncoop}'
    elif (fully_noncoop + noncoop) <= n < (fully_noncoop + noncoop + cond_coop):
        ag.effort = 0.6
        ag.trait = 'cond_coop'
        ag.num = f'cond_coop{(1 + n) - (fully_noncoop + noncoop)}'
    elif (fully_noncoop + noncoop + cond_coop) <= n < (fully_noncoop + noncoop + cond_coop + coop):
        ag.effort = 0.4
        ag.trait = 'coop'
        ag.num = f'coop{1 + n - (fully_noncoop + noncoop + cond_coop)}'
    else:
        ag.effort = 0.2
        ag.trait = 'fully_coop'
        ag.num = f'fully_coop{1 + n - (fully_noncoop + noncoop + cond_coop + coop)}'


def init_fisher_agents(base_parameters, model_parameters):
    """
    Initializes the fisher agents with base parameter values and assigns a cooperation strategy.

    Args:
    base_parameters (BaseParameters): Contains base parameters.
    model_parameters (ModelParameters): Records the total and current harvest in the model.
    """
    # First create all fisher agents to get their IDs
    fisher_agents = []
    for j in range(round(base_parameters.num_fishers)):
        ag = Fisher()
        assign_cooperation(ag, j, base_parameters)
        model_parameters.total_hav_data[ag.num] = [ag.harvest]
        model_parameters.current_hav_data[ag.num] = [ag.harvest]
        ag.x = rd.uniform(-base_parameters.half_length_area, base_parameters.half_length_area)
        ag.y = rd.uniform(-base_parameters.half_length_area, base_parameters.half_length_area)

        ag.base_effort = ag.effort
        
        fisher_agents.append(ag)
    
    # Now initialize trust scores between all fishers
    for fisher in fisher_agents:
        for other in fisher_agents:
            if fisher != other:
                fisher.trust_scores[other.num] = base_parameters.initial_trust

    # Add fishers to agents list
    model_parameters.agents.extend(fisher_agents)


def init_fish_agents(experiment, base_parameters, model_parameters):
    """
    Initializes the fish agents with varying base parameter values and assigns a school ID
    used for movement.

    Args:
    experiment (string): String that determines what fish parameters to vary. Options:
        'default', 'reproduction_rate', 'speed', 'both'
    base_parameters (BaseParameters): Contains base parameters.
    model_parameters (ModelParameters): Records the counts of the species and the agent
        objects.
    """
    model_parameters.total_fish_count.append(base_parameters.K)
    fish_params = Fish.fish_experiment(experiment, base_parameters.K)
    param_keys = sorted([k for k in fish_params if k != 'carrying_capacity'])
    param_values = [fish_params[k] for k in param_keys]

    combinations = list(product(*param_values))
    num_combinations = len(combinations)
    num_fish_per_combo = fish_params['carrying_capacity'] // num_combinations

    for combo in combinations:
        school_id = 0
        params = dict(zip(param_keys, combo))
        subtype_str = "_".join(f"{k}={v}" for k, v in params.items())
        for _ in range(round(num_fish_per_combo)):
            ag = Fish(**params)
            ag.school = school_id
            ag.x = rd.uniform(-base_parameters.half_length_area, base_parameters.half_length_area)
            ag.y = rd.uniform(-base_parameters.half_length_area, base_parameters.half_length_area)
            ag.subtype = subtype_str
            model_parameters.species_count[subtype_str] = [num_fish_per_combo]
            model_parameters.agents.append(ag)
        school_id += 1


def initialize(experiment, base_parameters, model_parameters):
    """
    Initializes the agents and data tracking objects of the model before first use.

    Args:
    experiment (string): String that determines what fish parameters to vary. Options:
        'default', 'reproduction_rate', 'speed', 'both'
    """
    delete_prev_sim()
    init_fisher_agents(base_parameters, model_parameters)
    init_fish_agents(experiment, base_parameters, model_parameters)