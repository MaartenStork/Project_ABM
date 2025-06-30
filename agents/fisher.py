import math
import random as rd
from statistics import mean
from agents.agent import Agent


class Fisher(Agent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.type = 'fishers'
        self.harvest = 0
        self.low_density_memory = 0
        self.trust_scores = {}
        self.cooperative_actions = 0
        self.total_actions = 0

    @staticmethod
    def update_trust(model_parameters, base_parameters):
        """Update trust scores between fishers based on their behavior."""
        fishers = [ag for ag in model_parameters.agents if ag.type == 'fishers']
        
        # For each fisher
        for fisher in fishers:
            # Find neighbors within trust radius
            neighbors = [nb for nb in fishers if nb != fisher and 
                        ((fisher.x - nb.x)**2 + (fisher.y - nb.y)**2) < base_parameters.trust_radius**2]
            
            for neighbor in neighbors:
                # Calculate cooperation score based on effort level
                # Lower effort = more cooperative
                fisher_coop_score = 1 - fisher.effort
                neighbor_coop_score = 1 - neighbor.effort
                
                # Update trust based on neighbor's behavior
                if neighbor_coop_score >= 0.5:  # If neighbor is being cooperative
                    fisher.trust_scores[neighbor.num] = min(1.0, 
                        fisher.trust_scores[neighbor.num] + base_parameters.trust_increase)
                else:  # If neighbor is being non-cooperative
                    fisher.trust_scores[neighbor.num] = max(0.0, 
                        fisher.trust_scores[neighbor.num] - base_parameters.trust_decrease)
                
                # Update cooperation counts
                fisher.total_actions += 1
                if neighbor_coop_score >= 0.5:
                    fisher.cooperative_actions += 1

    @staticmethod
    def adjust_effort_based_on_trust(model_parameters, base_parameters):
        """Adjust fishing effort based on trust levels."""
        fishers = [ag for ag in model_parameters.agents if ag.type == 'fishers']
        
        for fisher in fishers:
            if fisher.trait == 'cond_coop':  # Only conditional cooperators adjust based on trust
                # Calculate average trust in neighbors
                neighbors = [nb for nb in fishers if nb != fisher and 
                            ((fisher.x - nb.x)**2 + (fisher.y - nb.y)**2) < base_parameters.trust_radius**2]
                
                if neighbors:
                    avg_trust = mean([fisher.trust_scores[nb.num] for nb in neighbors])
                    
                    # Adjust effort based on trust
                    if avg_trust > base_parameters.trust_threshold:
                        # More cooperative (lower effort) when trust is high
                        fisher.effort = max(0.2, fisher.base_effort - 0.2)
                    else:
                        # Less cooperative (higher effort) when trust is low
                        fisher.effort = min(1.0, fisher.base_effort + 0.2)

    @staticmethod
    def check_threshold_behavior(model_parameters, base_parameters):
        """Check local fish density and adjust fishing effort based on thresholds."""
        
        # Process each fisher
        for fisher in [ag for ag in model_parameters.agents if ag.type == 'fishers']:
            # Count nearby fish
            local_fish = len([f for f in model_parameters.agents if f.type == 'fish' and 
                            ((fisher.x - f.x)**2 + (fisher.y - f.y)**2) < base_parameters.threshold_radius**2])
            
            # Calculate local density (fish per unit area)
            local_density = local_fish / (math.pi * base_parameters.threshold_radius**2)
            
            # If density is below threshold
            if local_density < base_parameters.fish_density_threshold:
                fisher.low_density_memory += 1
                if fisher.low_density_memory >= 1:  # Immediate response to low density
                    # Increase cooperation (decrease effort) while maintaining trait category
                    fisher.effort = max(0.2, fisher.base_effort - base_parameters.cooperation_increase)
            else:
                # If density is good for threshold_memory steps, revert to base effort
                if fisher.low_density_memory > 0:
                    fisher.low_density_memory -= 1
                    if fisher.low_density_memory == 0:
                        fisher.effort = fisher.base_effort

    @staticmethod
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
    
    def move(self, model_parameters, base_parameters):
        move_fishers = base_parameters.move_fishers
        half_length_area = base_parameters.half_length_area
        fishers_neighbors = [[nb.harvest, nb] for nb in model_parameters.agents if nb.type == 'fishers' and nb != self and ((self.x - nb.x)**2 + (self.y - nb.y)**2) < base_parameters.r_sqr] # detecting fishermen in neighbourhood
        fishers_neighbors_harvest = sorted(fishers_neighbors, key=lambda HAV: HAV[0]) # sort fishermen in neighborhood according to catch
        if len(fishers_neighbors_harvest) == 0: # if there exist no fisherman in neighbourhood
            theta_1 = 2*math.pi*rd.random()
            self.x +=  move_fishers*math.cos(theta_1) # move  'move_fishers' step in a random direction
            self.y +=  move_fishers*math.sin(theta_1)
            self.x = -half_length_area if self.x > half_length_area else  half_length_area if self.x < -half_length_area else self.x
            self.y = -half_length_area if self.y > half_length_area else  half_length_area if self.y < -half_length_area else self.y
        elif all([len(fishers_neighbors_harvest) > 0, fishers_neighbors_harvest[-1][0] > self.harvest]) : # if there exist fisherman with greater catch than focal fisherman
                deltax = fishers_neighbors_harvest[-1][-1].x - self.x   #move in the direction of one with greater catch than focal fisherman
                deltay = fishers_neighbors_harvest[-1][-1].y - self.y
                theta_2 = math.atan2(deltay,deltax)
                self.x +=  move_fishers*math.cos(theta_2) # move 'move_fishers' in the direction of neighbour fishermen with greatest catch
                self.y +=  move_fishers*math.sin(theta_2)
                self.x = -half_length_area if self.x > half_length_area else  half_length_area if self.x < -half_length_area else self.x
                self.y = -half_length_area if self.y > half_length_area else  half_length_area if self.y < -half_length_area else self.y
        else: # if all fisherman have less or equal catch relativelly  to focal fisherman
                theta_2 = 2*math.pi*rd.random()
                self.x +=  move_fishers*math.cos(theta_2) # move  'move_fishers' step in a random direction
                self.y +=  move_fishers*math.sin(theta_2)
                self.x = -half_length_area if self.x > half_length_area else  half_length_area if self.x < -half_length_area else self.x
                self.y = -half_length_area if self.y > half_length_area else  half_length_area if self.y < -half_length_area else self.y

    @staticmethod
    def imitate_successful_strategies(model_parameters, base_parameters):
        """Allow fishers to imitate more successful strategies from their neighbors."""

        # Get all fishers
        fishers = [ag for ag in model_parameters.agents if ag.type == 'fishers']

        # For each fisher
        for fisher in fishers:
            # Find neighbors within imitation radius
            neighbors = [nb for nb in fishers if nb != fisher and
                        ((fisher.x - nb.x)**2 + (fisher.y - nb.y)**2) < base_parameters.imitation_radius**2]

            if neighbors:
                # Find the most successful neighbor
                most_successful = max(neighbors, key=lambda x: x.harvest)

                # If the most successful neighbor has higher harvest
                if most_successful.harvest > fisher.harvest:
                    # Probabilistically imitate their strategy
                    if rd.random() < base_parameters.imitation_prob:
                        # Nudge effort towards the more successful agent
                        nudge = (most_successful.effort - fisher.effort) * base_parameters.imitation_nudge_factor
                        fisher.effort += nudge
                        
                        # Re-classify the trait based on the new effort
                        fisher.trait = Fisher.classify_trait_from_effort(fisher.effort)
    
    @staticmethod
    def update_fisher(model_parameters, base_parameters):
        """
        Makes all fishers update their trust in neighbors, adjust their effort based on
        mean trust in the area, and adjust their effort based on fish density in the area.
        """
        # Update trust and adjust behavior
        Fisher.update_trust(model_parameters, base_parameters)
        Fisher.adjust_effort_based_on_trust(model_parameters, base_parameters)
        # Check and update threshold-based behavior
        Fisher.check_threshold_behavior(model_parameters, base_parameters)

