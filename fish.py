def init_fish(carrying_capacity, reproduction_rate=False, speed=False):
    """
    Function that initializes the parameters to use for each fish population in a single simulation.
    """
    fish_params = {}

    if reproduction_rate:
        fish_params['reproduction_rate'] = [0.05, 0.1, 0.2, 0.3]
    else:
        fish_params['reproduction_rate'] = [0.2]

    if speed:
        fish_params['speed'] = [0.03, 0.04, 0.05, 0.06]
    else:
        fish_params['speed'] = [0.05]

    fish_params['carrying_capacity'] = carrying_capacity

    return fish_params


def fish_experiment(exp_label, carrying_capacity=200):
    if exp_label == 'default':
        return init_fish(carrying_capacity)
    elif exp_label == 'reproduction_rate':
        return init_fish(carrying_capacity, reproduction_rate=True)
    elif exp_label == 'speed':
        return init_fish(carrying_capacity, speed=True)
