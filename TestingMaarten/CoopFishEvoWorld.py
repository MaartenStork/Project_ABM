import numpy as np
import matplotlib.pyplot as plt
import random as rd
import math
import copy as cp
# from PIL import Image  # Uncomment if you want to use a PNG as a world mask

# Map settings
MAP_SIZE = 100
WATER = 1
LAND = 0

# --- Optionally, load a PNG as a world mask ---
def load_world_from_png(png_path):
    from PIL import Image
    img = Image.open(png_path).convert('L').resize((MAP_SIZE, MAP_SIZE))
    arr = np.array(img)
    # Threshold: white (>=128) is water, black (<128) is land
    return np.where(arr >= 128, WATER, LAND)

# world = load_world_from_png('your_map.png')  # Uncomment and provide your PNG path to use a custom map

# --- Procedurally generate a more interesting world map ---
world = np.zeros((MAP_SIZE, MAP_SIZE), dtype=int)
world[:, :25] = LAND  # left 25 columns = land
world[:, 25:] = WATER  # rest = water
# Peninsula
for x in range(25, 45):
    for y in range(30, 70):
        if abs(y-50) < 15 - (x-25)//2:
            world[y, x] = LAND
# Large island
cx, cy, r = 70, 30, 10
for x in range(MAP_SIZE):
    for y in range(MAP_SIZE):
        if (x-cx)**2 + (y-cy)**2 < r**2:
            world[y, x] = LAND
# Small island
cx2, cy2, r2 = 80, 80, 6
for x in range(MAP_SIZE):
    for y in range(MAP_SIZE):
        if (x-cx2)**2 + (y-cy2)**2 < r2**2:
            world[y, x] = LAND
# Cove (indent land)
for y in range(40, 60):
    for x in range(15, 25):
        if abs(y-50) < 8 and x > 20 - (8-abs(y-50)):
            world[y, x] = WATER
# Lake (in land)
lake_cx, lake_cy, lake_r = 10, 80, 7
for x in range(MAP_SIZE):
    for y in range(MAP_SIZE):
        if (x-lake_cx)**2 + (y-lake_cy)**2 < lake_r**2:
            world[y, x] = WATER

# Harbors: on the coast (land/water boundary)
harbors = [(24, y) for y in [15, 35, 55, 75, 90]]

# Parameters
K = 200
n = 1000  # Increased for more generations
base_growth_prob = 0.3
init_fish = 200
num_fishers = 20
move_fish = 1.5
move_fishers = 2.5
q = 0.6
r = 3  # catch radius in grid units

# Evolution
GEN_LENGTH = 50  # steps per generation
MUTATION_STD = 0.1

# Agent class
def agent():
    return type('agent', (), {})()

def is_water(x, y):
    xi, yi = int(round(x)), int(round(y))
    return 0 <= xi < MAP_SIZE and 0 <= yi < MAP_SIZE and world[yi, xi] == WATER

def is_land(x, y):
    xi, yi = int(round(x)), int(round(y))
    return 0 <= xi < MAP_SIZE and 0 <= yi < MAP_SIZE and world[yi, xi] == LAND

# Safety score: high in coves, far from boats, near land but not too close
def safety_score(x, y, boats):
    # Prefer water near land (coves), but not too close to boats
    if not is_water(x, y):
        return -np.inf
    # Distance to nearest boat
    min_boat_dist = min([math.hypot(x-b.x, y-b.y) for b in boats], default=MAP_SIZE)
    # Distance to nearest land
    min_land_dist = min([math.hypot(x-lx, y-ly) for lx in range(MAP_SIZE) for ly in range(MAP_SIZE) if world[ly, lx]==LAND], default=MAP_SIZE)
    # Coves: water near land but not exposed
    return min_boat_dist + max(0, 10-min_land_dist)

# Initialize agents
def initialize():
    global time1, agents, fish_data, gen
    time1 = 0
    gen = 0
    agents = []
    fish_data = [init_fish]
    # Boats start at harbors
    for i in range(num_fishers):
        ag = agent()
        ag.type = 'fishers'
        ag.x, ag.y = harbors[i % len(harbors)]
        ag.harvest = 0
        agents.append(ag)
    # Fish only in water, with random cautiousness
    for i in range(init_fish):
        ag = agent()
        ag.type = 'fish'
        while True:
            ag.x = rd.randint(25, MAP_SIZE-1)
            ag.y = rd.randint(0, MAP_SIZE-1)
            if is_water(ag.x, ag.y):
                break
        ag.cautiousness = max(0, min(1, rd.gauss(0.5, 0.2)))
        agents.append(ag)

def update_fish():
    fish_list = [j for j in agents if j.type == 'fish']
    boats = [j for j in agents if j.type == 'fishers']
    if not fish_list:
        return
    fish_ag = rd.choice(fish_list)
    # Cautious fish bias toward safe places
    if rd.random() < fish_ag.cautiousness:
        # Try 8 directions, pick the safest
        best_score = -np.inf
        best_x, best_y = fish_ag.x, fish_ag.y
        for d in np.linspace(0, 2*np.pi, 8, endpoint=False):
            nx = fish_ag.x + move_fish*math.cos(d)
            ny = fish_ag.y + move_fish*math.sin(d)
            score = safety_score(nx, ny, boats)
            if score > best_score:
                best_score = score
                best_x, best_y = nx, ny
        if is_water(best_x, best_y):
            fish_ag.x, fish_ag.y = best_x, best_y
    else:
        # Random move in water
        for _ in range(8):
            theta = 2*math.pi*rd.random()
            nx = fish_ag.x + move_fish*math.cos(theta)
            ny = fish_ag.y + move_fish*math.sin(theta)
            if is_water(nx, ny):
                fish_ag.x, fish_ag.y = nx, ny
                break
    # Growth
    if rd.random() < base_growth_prob * (1-sum([1 for j in agents if j.type == 'fish'])/float(K)):
        new_fish = cp.copy(fish_ag)
        # Inherit cautiousness with mutation
        new_fish.cautiousness = max(0, min(1, rd.gauss(fish_ag.cautiousness, MUTATION_STD)))
        agents.append(new_fish)

def update_one_unit_time():
    update_fish()
    # Fisher movement: move from harbor into water, stay in water
    for fisher in [j for j in agents if j.type == 'fishers']:
        # If on land, move right into water
        if is_land(fisher.x, fisher.y):
            fisher.x = 26
        # Random walk in water
        for _ in range(8):
            theta = 2*math.pi*rd.random()
            nx = fisher.x + move_fishers*math.cos(theta)
            ny = fisher.y + move_fishers*math.sin(theta)
            if is_water(nx, ny):
                fisher.x, fisher.y = nx, ny
                break
        # Catch fish in neighborhood
        fish_neighbors = [nb for nb in agents if nb.type == 'fish' and (fisher.x - nb.x)**2 + (fisher.y - nb.y)**2 < r**2]
        num_fish_harvest = int(round(q * len(fish_neighbors)))
        if fish_neighbors and num_fish_harvest > 0:
            sample_fish_harvest = rd.sample(fish_neighbors, min(num_fish_harvest, len(fish_neighbors)))
            for j in sample_fish_harvest:
                agents.remove(j)
                fisher.harvest += 1
    # Record data
    fish_data.append(sum([1 for j in agents if j.type == 'fish']))

def evolve():
    # At the end of each generation, surviving fish reproduce with mutation
    global agents, gen
    fish_list = [j for j in agents if j.type == 'fish']
    new_agents = [j for j in agents if j.type == 'fishers']
    for fish in fish_list:
        for _ in range(2):  # each survivor produces 2 offspring
            ag = agent()
            ag.type = 'fish'
            ag.x, ag.y = fish.x, fish.y
            ag.cautiousness = max(0, min(1, rd.gauss(fish.cautiousness, MUTATION_STD)))
            new_agents.append(ag)
    agents = new_agents
    gen += 1

def observe(step):
    plt.figure(figsize=(8, 8))
    # Show map
    plt.imshow(world, cmap='terrain', origin='lower', extent=[0, MAP_SIZE, 0, MAP_SIZE], alpha=0.5)
    # Show safety heatmap (optional, comment out if not wanted)
    # boats = [j for j in agents if j.type == 'fishers']
    # safety = np.array([[safety_score(x, y, boats) for x in range(MAP_SIZE)] for y in range(MAP_SIZE)])
    # plt.imshow(safety, cmap='Blues', origin='lower', extent=[0, MAP_SIZE, 0, MAP_SIZE], alpha=0.2)
    # Harbors
    hx, hy = zip(*harbors)
    plt.scatter(hx, hy, c='red', s=100, marker='s', label='Harbor', zorder=2)
    # Fish
    fish = [ag for ag in agents if ag.type == 'fish']
    if fish:
        plt.scatter([ag.x for ag in fish], [ag.y for ag in fish], c='green', marker='^', s=30, label='Fish', zorder=3)
    # Boats
    boats = [ag for ag in agents if ag.type == 'fishers']
    if boats:
        plt.scatter([ag.x for ag in boats], [ag.y for ag in boats], c='blue', marker='o', s=60, label='Boat', zorder=4)
    plt.legend(loc='upper right')
    plt.xlim(0, MAP_SIZE)
    plt.ylim(0, MAP_SIZE)
    plt.title(f'Worldmap Evolution, Gen {gen}, Step {step}')
    plt.tight_layout()
    plt.savefig(f'year_{step:04d}.png')
    plt.close()

if __name__ == '__main__':
    initialize()
    for step in range(1, n+1):
        update_one_unit_time()
        observe(step)
        if step % GEN_LENGTH == 0:
            evolve()
    print('Done! Frames saved as year_XXXX.png') 