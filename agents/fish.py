import math
import copy as cp
import random as rd
from statistics import mean
from agents.agent import Agent


class Fish(Agent):
    def __init__(self, **params):
        super().__init__(**params)
        self.type = 'fish'

    @staticmethod
    def init_boids_zones(fish_agent, all_agents, square_repulsion_rad, square_orientation_rad, square_attraction_rad):
        """
        Categorizes neighboring fish into the boids zones.

        Args:
        fish_agent (Fish): Agent to categorize neighbors for.
        all_agents (array): List containing all model agents.
        square_repulsion_rad (float): Radius to check repulsion area for.
        square_orientation_rad (float): Radius to check alignment area for.
        square_attraction_rad (float): Radius to check attraction area for.
        """
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

    @staticmethod
    def repulsion_update(fish_agent, repulsion_agents, border):
        """
        Moves fish agent according to repulsion rules, leading to movement away from neighbors.

        Args:
        fish_agent (Fish): Agent to move.
        repulsion_agents (array): Fish agents to move away from.
        border (int): X/Y location of the model's (positive) borders.
        """
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

    @staticmethod
    def alignment_update(fish_agent, alignment_agents, border):
        """
        Moves fish agent according to alignment rules, leading to movement aligned with neighbors.

        Args:
        fish_agent (Fish): Agent to move.
        alignment_agents (array): Fish agents to move along with.
        border (int): X/Y location of the model's (positive) borders.
        """
        theta = mean([math.atan2((j.y - fish_agent.y),(j.x - fish_agent.x)) for j in alignment_agents])
        fish_agent.x +=   math.cos(theta)     # moves 'move_fish' step,  move_fish*math.cos(theta)
        fish_agent.y +=   math.sin(theta)
        fish_agent.x = (fish_agent.x % -border) if fish_agent.x > border else (
                fish_agent.x % border) if fish_agent.x < -border else fish_agent.x  # ( When fish-agent approach a border of the landscape,
        fish_agent.y = (fish_agent.y % -border) if fish_agent.y > border else (
                fish_agent.y % border) if fish_agent.y < -border else fish_agent.y  # they re-enter the system at the opposite border )

    @staticmethod
    def attraction_update(fish_agent, attraction_agents, border):
        """
        Moves fish agent according to attraction rules, leading to movement towards neighbors.

        Args:
        fish_agent (Fish): Agent to move.
        attraction_agents (array): Fish agents to move towards.
        border (int): X/Y location of the model's (positive) borders.
        """
        attraction_x = mean([j.x for j in attraction_agents])
        attraction_y = mean([j.y for j in attraction_agents])
        theta = math.atan2((attraction_y - fish_agent.y), (attraction_x - fish_agent.x))
        fish_agent.x +=  fish_agent.speed*math.cos(theta)
        fish_agent.y +=  fish_agent.speed*math.sin(theta)
        fish_agent.x = (fish_agent.x % -border) if fish_agent.x > border else (
                fish_agent.x % border) if fish_agent.x < -border else fish_agent.x  # ( When fish-agent approach a border of the landscape,
        fish_agent.y = (fish_agent.y % -border) if fish_agent.y > border else (
                fish_agent.y % border) if fish_agent.y < -border else fish_agent.y  # they re-enter the system at the opposite border )

    @staticmethod
    def random_update(fish_agent, border):
        """
        Moves fish agent in a random direction.

        Args:
        fish_agent (Fish): Agent to move.
        border (int): X/Y location of the model's (positive) borders.
        """
        theta = 2 * math.pi * rd.random()
        fish_agent.x += fish_agent.speed * math.cos(theta)  # moves 'move_fish' step
        fish_agent.y += fish_agent.speed * math.sin(theta)
        fish_agent.x = (fish_agent.x % -border) if fish_agent.x > border else (
                    fish_agent.x % border) if fish_agent.x < -border else fish_agent.x  # ( When fish-agent approach a border of the landscape,
        fish_agent.y = (fish_agent.y % -border) if fish_agent.y > border else (
                    fish_agent.y % border) if fish_agent.y < -border else fish_agent.y  # they re-enter the system at the opposite border )

    @staticmethod
    def reproduce(model_parameters, fish_agent, new_fish, new_species_count):
        """
        Copy a fish agent.

        Args:
        model_parameters (ModelParameters): Contains list of all agents.
        fish_agent (Fish): Fish agent to copy.
        new_fish (int): Amount of new fish created in one timestep.
        new_species_count (dict): Amount of new fish per species created in one timestep.
        """
        model_parameters.agents.append(cp.copy(fish_agent))
        new_fish += 1
        new_species_count[fish_agent.subtype] += 1
        return new_fish

    @staticmethod
    def update_counts(model_parameters, old_fish, new_fish, new_species_count):
        """
        Add the number of new fish over all species as well as per species to the simulation data.

        Args:
        model_parameters (ModelParameters): Contains fish counts of simulation.
        old_fish (int): Amount of fish agents in the previous timestep.
        new_fish (int): Amount of new fish created in one timestep.
        new_species_count (dict): Amount of new fish per species created in one timestep.
        """
        model_parameters.total_fish_count.append(old_fish + new_fish)
        for species in model_parameters.species_count:
            if species in new_species_count:
                model_parameters.species_count[species].append(new_species_count[species])
            else:
                model_parameters.species_count[species].append(0)

    @staticmethod
    def update_fish(model_parameters, base_parameters):
        """
        Make all fish move and potentially reproduce.

        model_parameters (ModelParameters): Contains runtime data of simulation.
        base_parameters (BaseParameters): Contains base parameters of the simulation.
        """
        fish_list = [j for j in model_parameters.agents if j.type == 'fish']
        # shuffle list to make sure all fish have a fair chance at reproduction
        rd.shuffle(fish_list)
        if not fish_list:
            model_parameters.total_fish_count.append(0)
            for species in model_parameters.species_count:
                model_parameters.species_count[species].append(0)
            return

        # fish_ag = rd.sample(fish_list, 1)[-1]
        new_fish = 0
        new_species_count = {}
        for fish_ag in fish_list:
            if fish_ag not in new_species_count:
                new_species_count[fish_ag.subtype] = 1
            else:
                new_species_count[fish_ag.subtype] += 1

            repulsion, alignment, attraction = Fish.init_boids_zones(fish_ag,
                                                                model_parameters.agents,
                                                                base_parameters.rad_repulsion_sqr,
                                                                base_parameters.rad_orientation_sqr,
                                                                base_parameters.rad_attraction_sqr)

            # if fishes within repulsion zone, move away from the spot that would be the center of mass (midpoint)
            # of all  fish within repulsion zone
            if len(repulsion) > 0:
                Fish.repulsion_update(fish_ag, repulsion, base_parameters.half_length_area)

            # if fishes within parallel-orientation zone, change direction to match the average direction of all the other
            # fish within parallel-orientation zone
            elif all([len(repulsion) == 0, len(alignment) > 0]):
                Fish.alignment_update(fish_ag, alignment, base_parameters.half_length_area)

            elif all([len(repulsion) == 0, len(alignment) == 0, len(attraction) > 0]): # if fishes within only the attraction zone, head towards the middle (midpoint) of the fishes in zone of attraction.
                Fish.attraction_update(fish_ag, attraction, base_parameters.half_length_area)

            elif all([len(repulsion) == 0, len(alignment) == 0, len(attraction) == 0]): # if no fishes in all the zone, move in a random direction
                Fish.random_update(fish_ag, base_parameters.half_length_area)

            if len(fish_list) + new_fish < model_parameters.total_fish_count[0] and rd.random() < fish_ag.reproduction_rate * (1-sum([1 for j in model_parameters.agents if j.type == 'fish'])/float(base_parameters.K)):  # logistic growth of fishes
                new_fish = Fish.reproduce(model_parameters, fish_ag, new_fish, new_species_count)

        Fish.update_counts(model_parameters, len(fish_list), new_fish, new_species_count)
        
    @staticmethod
    def init_fish(carrying_capacity, reproduction_rate=False, speed=False):
        """
        Function that initializes the parameters to use for each fish population in a single simulation.
        """
        fish_params = {}

        if reproduction_rate:
            fish_params['reproduction_rate'] = [0.3, 0.5, 0.7, 0.9]
        else:
            fish_params['reproduction_rate'] = [0.3]

        if speed:
            fish_params['speed'] = [0.05, 0.1, 0.15, 0.2]
        else:
            fish_params['speed'] = [0.05]

        fish_params['carrying_capacity'] = carrying_capacity

        return fish_params


    @staticmethod
    def fish_experiment(exp_label, carrying_capacity=200):
        """
        Returns default parameter settings for multiple fish species depending on the passed experiment label.
        """
        if exp_label == 'default':
            return Fish.init_fish(carrying_capacity)
        elif exp_label == 'reproduction_rate':
            return Fish.init_fish(carrying_capacity, reproduction_rate=True)
        elif exp_label == 'speed':
            return Fish.init_fish(carrying_capacity, speed=True)
        elif exp_label == 'both':
            return Fish.init_fish(carrying_capacity, reproduction_rate=True, speed=True)
