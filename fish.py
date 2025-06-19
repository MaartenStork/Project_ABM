def init_fish(carrying_capacity, reproduction_rate=False, desirability=False, catchability=False):
    """
    Function that initializes the parameters to use for each fish population in a single simulation.
    """
    fish_params = {}

    if reproduction_rate:
        fish_params['reproduction_rate'] = [0.1, 0.3, 0.5, 0.7]
    else:
        fish_params['reproduction_rate'] = [0.3]

    if desirability:
        fish_params['desirability'] = [0.25, 0.5, 0.75, 1.0]
    else:
        fish_params['desirability'] = [1.0]

    if catchability:
        fish_params['catchability'] = [0.25, 0.5, 0.75, 1.0]
    else:
        fish_params['catchability'] = [1.0]

    fish_params['carrying_capacity'] = carrying_capacity

    return fish_params


def fish_experiment(exp_label, carrying_capacity=200):
    if exp_label == 'default':
        return init_fish(carrying_capacity)
    elif exp_label == 'reproduction_rate':
        return init_fish(carrying_capacity, reproduction_rate=True)
    elif exp_label == 'desirability':
        return init_fish(carrying_capacity, desirability=True)
    elif exp_label == 'catchability':
        return init_fish(carrying_capacity, catchability=True)
