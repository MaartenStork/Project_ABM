#---------------------------------------------------------------------------
import math


# Parameters #

class BaseParameters:  # Class containing all base model parameters
    def __init__(self):
        # Trust and loyalty parameters
        self.initial_trust = 0.5  # Initial trust score between fishers
        self.trust_increase = 0.1  # How much trust increases when seeing cooperation
        self.trust_decrease = 0.2  # How much trust decreases when seeing non-cooperation
        self.trust_radius = 0.3   # Radius within which fishers observe each other's behavior
        self.trust_memory = 5     # How many time steps to remember trust changes
        self.trust_threshold = 0.6  # Trust threshold for cooperative behavior
        
        # Fishing ground and time #
        self.K = 500 # carrying capacity of fishing ground
        self.n = 150 # number of simulation time steps

        # Threshold-based behavior parameters
        self.fish_density_threshold = 3.0  # threshold for local fish density that triggers cooperation
        self.threshold_radius = 0.4  # radius to check local fish density
        self.threshold_memory = 5  # how many time steps to remember low density before switching back
        self.cooperation_increase = 0.2  # how much to increase cooperation when threshold is crossed

        # Imitation parameters
        self.imitation_period = 5  # how often agents compare and potentially imitate (every X time steps)
        self.imitation_radius = 0.3  # radius within which agents can observe others' success
        self.imitation_prob = 0.3  # probability of imitating when a more successful strategy is found
        self.imitation_nudge_factor = 0.25 # how much to nudge effort towards a more successful strategy

        # Attributes of fish agents #
        self.scale = 2
        self.rad_repulsion = 0.025 * self.scale  # radius of repulsion zone
        self.rad_orientation = 0.06 * self.scale # radius of orientation zone
        self.rad_attraction =  0.1 * self.scale  # radius of attraction zone
        self.rad_repulsion_sqr = self.rad_repulsion ** 2     
        self.rad_orientation_sqr = self.rad_orientation ** 2
        self.rad_attraction_sqr = self.rad_attraction ** 2

        # Attributes of fishing agents (pirogues) #
        self.num_fishers = 20     # number of pirogues
        self.move_fishers = 0.3   # speed of a pirogue 
        self.q = 0.6              # catchability coefficient
        self.r = 0.2              # neighbourhood radius 
        self.r_sqr = self.r ** 2       # neighbourhood radius squared

        # Cooperation scenarios (summ of all cooperation types = num_fishers) #
        self.fully_noncoop = 4     # number of fully non-cooperative pirogues
        self.noncoop = 4           # number of non-cooperative pirogues
        self.cond_coop = 4         # number of conditional cooperative pirogues
        self.coop = 4              # number of cooperative pirogues
        self.fully_coop = 4        # number of fully cooperative pirogues

        # Design of the MPA (presence/absence, size, age, and distance of between two) #
        self.MPA = 'no'         # Presence or absence of MPA ('yes' for presence, 'no' for absence)
        self.Both = 'no'         # Presence of MPA ('no' for full-time presence, 'yes' for part-time presence)

        # Coordinates of the fishing ground #
        self.Area = 2.0000 
        self.Length_Area = math.sqrt(self.Area)
        self.half_length_area = self.Length_Area / 2

        # Live plotting parameters
        self.plot_update_freq = 25  # Update plot every X steps


class ModelParameters:
    def __init__(self):
        self.time = 0.
        self.agents = []
        self.total_fish_count = []
        self.species_count = {}
        self.total_hav_data = {}
        self.current_hav_data = {}
        self.fishermen_data1 = [0]
        self.fishermen_data2 = [0]

        # Cooperation level tracking
        self.cooperation_levels = []  # track average cooperation level over time
        self.strategy_counts = {'fully_noncoop': [], 'noncoop': [], 'cond_coop': [], 'coop': [], 'fully_coop': []}
        self.trust_history = []
