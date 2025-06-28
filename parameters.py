#---------------------------------------------------------------------------
import math


# Parameters #

# Trust and loyalty parameters
initial_trust = 0.5  # Initial trust score between fishers
trust_increase = 0.1  # How much trust increases when seeing cooperation
trust_decrease = 0.2  # How much trust decreases when seeing non-cooperation
trust_radius = 0.3   # Radius within which fishers observe each other's behavior
trust_memory = 5     # How many time steps to remember trust changes
trust_threshold = 0.6  # Trust threshold for cooperative behavior

# Fishing ground and time #
K = 500 # carrying capacity of fishing ground
n = 350 # number of simulation time steps - EXTENDED for better equilibrium detection

# Threshold-based behavior parameters
fish_density_threshold = 3.0  # threshold for local fish density that triggers cooperation
threshold_radius = 0.4  # radius to check local fish density
threshold_memory = 5  # how many time steps to remember low density before switching back
cooperation_increase = 0.2  # how much to increase cooperation when threshold is crossed

# Imitation parameters
imitation_period = 5  # how often agents compare and potentially imitate (every X time steps)
imitation_radius = 0.3  # radius within which agents can observe others' success
imitation_prob = 0.3  # probability of imitating when a more successful strategy is found
imitation_nudge_factor = 0.25 # how much to nudge effort towards a more successful strategy

# Cooperation level tracking
cooperation_levels = []  # track average cooperation level over time
strategy_counts = {'fully_noncoop': [], 'noncoop': [], 'cond_coop': [], 'coop': [], 'fully_coop': []}

# Attributes of fish agents #
scale = 2
rad_repulsion = 0.025 * scale  # radius of repulsion zone
rad_orientation = 0.06 * scale # radius of orientation zone
rad_attraction =  0.1 * scale  # radius of attraction zone
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
MPA = 'no'         # Presence or absence of MPA ('yes' for presence, 'no' for absence)
Both = 'no'         # Presence of MPA ('no' for full-time presence, 'yes' for part-time presence)
Time_MPA = 50       # Period of time over which MPA is active (when Both = 'yes') 
Type_MPA = 'single' # Spacial configuration of MPA ('spaced' for two MPAs, 'single' for one MPA)
Dist_MPA = 0.2      # Distance between two MPAs (when Type_MPA = 'spaced')
Frac_MPA = 0.25     # Fraction of fishing grounde covered by MPA(s)

# Coordinates of the fishing ground #
Area = 2.0000 
Length_Area = math.sqrt(Area)
half_length_area = Length_Area / 2

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
plot_update_freq = 25  # Update plot every X steps
