def init_fish(reproduction_rate=True, desirability=False, catchability=False, carrying_capacity=200):
    """
    Function that initializes the parameters to use for each fish population in a single simulation.
    """
    fish_params = {}

    if reproduction_rate:
        fish_params['reproduction_rate'] = [0.1, 0.3, 0.5]
    else:
        fish_params['reproduction_rate'] = [0.3]

    if desirability:
        fish_params['desirability'] = [0.3, 0.6, 1.0]
    else:
        fish_params['desirability'] = [1.0]

    if catchability:
        fish_params['catchability'] = [0.3, 0.6, 1.0]
    else:
        fish_params['catchability'] = [1.0]

    fish_params['carrying_capacity'] = carrying_capacity

    return fish_params