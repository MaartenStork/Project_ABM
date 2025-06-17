# -*- coding: utf-8 -*-
#--------------------------------------------------------------------------------------
# CoopFishWorldMap.py: ABM with a worldmap (land, water, harbors)
#--------------------------------------------------------------------------------------
import random as rd
import math
import numpy as np
import matplotlib.pyplot as plt
import copy as cp
from statistics import mean

# Parameters
K = 200
n = 150
base_growth_prob = 0.3
init_fish = 200
move_fish = 0.2
num_fishers = 20
move_fishers = 0.3
q = 0.6
r = 0.2
r_sqr = r ** 2

# Worldmap settings
MAP_WIDTH = 2.0
MAP_HEIGHT = 2.0
LAND_FRAC = 0.3  # left 30% is land, right 70% is water
LAND_X = -MAP_WIDTH/2
WATER_X = LAND_X + MAP_WIDTH * LAND_FRAC

# Harbors (on land, evenly spaced vertically)
NUM_HARBORS = 4
harbor_x = LAND_X + MAP_WIDTH * LAND_FRAC * 0.5
harbor_ys = np.linspace(-MAP_HEIGHT/2 + 0.2, MAP_HEIGHT/2 - 0.2, NUM_HARBORS)
harbors = [(harbor_x, y) for y in harbor_ys]

# Agent class
def agent():
    return type('agent', (), {})()

# Initialize agents

def initialize():
    global time1, agents, fish_data
    time1 = 0
    agents = []
    fish_data = [init_fish]
    # Boats start at harbors
    for i in range(num_fishers):
        ag = agent()
        ag.type = 'fishers'
        ag.x, ag.y = harbors[i % NUM_HARBORS]
        ag.harvest = 0
        agents.append(ag)
    # Fish only in water
    for i in range(init_fish):
        ag = agent()
        ag.type = 'fish'
        while True:
            ag.x = rd.uniform(WATER_X, MAP_WIDTH/2)
            ag.y = rd.uniform(-MAP_HEIGHT/2, MAP_HEIGHT/2)
            if ag.x > WATER_X:
                break
        agents.append(ag)

def is_water(x, y):
    return x > WATER_X

def is_land(x, y):
    return x <= WATER_X

# Update fish movement (only in water)
def update_fish():
    fish_list = [j for j in agents if j.type == 'fish']
    if not fish_list:
        return
    fish_ag = rd.sample(fish_list, 1)[-1]
    theta = 2*math.pi*rd.random()
    new_x = fish_ag.x + move_fish*math.cos(theta)
    new_y = fish_ag.y + move_fish*math.sin(theta)
    # Keep in water
    if is_water(new_x, new_y) and -MAP_HEIGHT/2 <= new_y <= MAP_HEIGHT/2 and WATER_X < new_x <= MAP_WIDTH/2:
        fish_ag.x = new_x
        fish_ag.y = new_y
    # Growth
    if rd.random() < base_growth_prob * (1-sum([1 for j in agents if j.type == 'fish'])/float(K)):
        agents.append(cp.copy(fish_ag))

def update_one_unit_time():
    update_fish()
    # Fisher movement: move from harbor into water, stay in water
    for fisher in [j for j in agents if j.type == 'fishers']:
        # If on land, move right into water
        if is_land(fisher.x, fisher.y):
            fisher.x = WATER_X + 0.01
        # Random walk in water
        theta = 2*math.pi*rd.random()
        new_x = fisher.x + move_fishers*math.cos(theta)
        new_y = fisher.y + move_fishers*math.sin(theta)
        if is_water(new_x, new_y) and -MAP_HEIGHT/2 <= new_y <= MAP_HEIGHT/2 and WATER_X < new_x <= MAP_WIDTH/2:
            fisher.x = new_x
            fisher.y = new_y
        # Catch fish in neighborhood
        fish_neighbors = [nb for nb in agents if nb.type == 'fish' and ((fisher.x - nb.x)**2 + (fisher.y - nb.y)**2) < r_sqr]
        num_fish_harvest = int(round(q * len(fish_neighbors)))
        if fish_neighbors and num_fish_harvest > 0:
            sample_fish_harvest = rd.sample(fish_neighbors, min(num_fish_harvest, len(fish_neighbors)))
            for j in sample_fish_harvest:
                agents.remove(j)
                fisher.harvest += 1
    # Record data
    fish_data.append(sum([1 for j in agents if j.type == 'fish']))

def observe():
    plt.figure(figsize=(10, 10))
    # Draw land
    plt.fill_betweenx([-MAP_HEIGHT/2, MAP_HEIGHT/2], LAND_X, WATER_X, color='#bfa76f', zorder=0)
    # Draw water
    plt.fill_betweenx([-MAP_HEIGHT/2, MAP_HEIGHT/2], WATER_X, MAP_WIDTH/2, color='#7ec0ee', zorder=0)
    # Draw harbors
    hx, hy = zip(*harbors)
    plt.scatter(hx, hy, c='red', s=120, marker='s', label='Harbor', zorder=2)
    # Draw fish
    fish = [ag for ag in agents if ag.type == 'fish']
    if fish:
        plt.scatter([ag.x for ag in fish], [ag.y for ag in fish], c='green', marker='^', s=40, label='Fish', zorder=3)
    # Draw boats
    boats = [ag for ag in agents if ag.type == 'fishers']
    if boats:
        plt.scatter([ag.x for ag in boats], [ag.y for ag in boats], c='blue', marker='o', s=80, label='Boat', zorder=4)
    plt.legend(loc='upper right')
    plt.xlim(LAND_X, MAP_WIDTH/2)
    plt.ylim(-MAP_HEIGHT/2, MAP_HEIGHT/2)
    plt.title('Worldmap: Land, Water, Harbors, Fish, Boats')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.tight_layout()
    plt.savefig('year_%04d.png' % len(fish_data))
    plt.close()

# Main loop
if __name__ == '__main__':
    initialize()
    observe()
    for j in range(1, n):
        update_one_unit_time()
        observe()
    print('Done! Frames saved as year_XXXX.png') 