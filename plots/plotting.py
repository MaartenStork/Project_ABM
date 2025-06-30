import matplotlib.pyplot as plt
import numpy as np


def plot_fishers(agents, ax):
    """
    Plots the location of each fisher agent in the model in its current state.

    Args:
    agents (array): All agents in the model.
    ax (Ax): Matplotlib ax object to plot in.
    """
    fishermen = [ag for ag in agents if ag.type == 'fishers']

    if len(fishermen) > 0:
        X_fully_noncoop = [ag.x for ag in fishermen if ag.trait == 'fully_noncoop']
        Y_fully_noncoop = [ag.y for ag in fishermen if ag.trait == 'fully_noncoop']
        X_noncoop = [ag.x for ag in fishermen if ag.trait == 'noncoop']
        Y_noncoop = [ag.y for ag in fishermen if ag.trait == 'noncoop']
        X_cond_coop = [ag.x for ag in fishermen if ag.trait == 'cond_coop']
        Y_cond_coop = [ag.y for ag in fishermen if ag.trait == 'cond_coop']
        X_coop = [ag.x for ag in fishermen if ag.trait == 'coop']
        Y_coop = [ag.y for ag in fishermen if ag.trait == 'coop']
        X_fully_coop = [ag.x for ag in fishermen if ag.trait == 'fully_coop']
        Y_fully_coop  = [ag.y for ag in fishermen if ag.trait == 'fully_coop']
        colors = np.linspace(0, 1, 5)
        mymap = plt.get_cmap("Greys")
        my_colors = mymap(colors)
        ax.plot(X_fully_coop, Y_fully_coop, 'o', color=my_colors[4], markersize=7.5, label='fully_coop')
        ax.plot(X_coop, Y_coop, 'o', color=my_colors[3], markersize=7.5, label='coop')
        ax.plot(X_cond_coop, Y_cond_coop, 'o', color=my_colors[2], markersize=7.5, label='conditional_coop')
        ax.plot(X_noncoop, Y_noncoop, 'o', color=my_colors[1], markersize=7.5, label='noncoop')
        ax.plot(X_fully_noncoop, Y_fully_noncoop, 'o', color=my_colors[0], markersize=7.5, label='fully_noncoop')



def plot_fish(agents, ax):
    """
    Plots the location of each fish agent in the model in its current state.
    """
    fish = [ag for ag in agents if ag.type == 'fish']
    if len(fish) > 0:
        subtypes = list(set([ag.subtype for ag in fish]))
        subtypes.sort()
        cmap = plt.colormaps.get_cmap('viridis')
        colors = [cmap(i / len(subtypes)) for i in range(len(subtypes))]
        for i, subtype in enumerate(subtypes):
            subtype_fish = [ag for ag in fish if ag.subtype == subtype]
            X = [ag.x for ag in subtype_fish]
            Y = [ag.y for ag in subtype_fish]
            ax.plot(X, Y, '^', color=colors[i], markersize=3, label=f'fish {subtype}')
    
        
def observe(base_parameters, model_parameters):
    """
    Records the state of the model's agents as they currently exist, plots them, and outputs the plot
    as a PNG file.

    Args:
        base_parameters (BaseParameters): Contains information about the half length of the model area.
        model_parameters (ModelParameters): Contains information about all agents in the model.
    """

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_facecolor('lightskyblue')

    plot_fishers(model_parameters.agents, ax)
    plot_fish(model_parameters.agents, ax)
    
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim([-base_parameters.half_length_area, base_parameters.half_length_area])
    ax.set_ylim([-base_parameters.half_length_area, base_parameters.half_length_area])
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f'year = {int(model_parameters.time)}')
    ax.legend(numpoints=1, loc='center', bbox_to_anchor=(0.5, -0.072), ncol=3, prop={'size': 11},
              facecolor='lightskyblue')

    fig.savefig(f'simulation_output/year_{int(model_parameters.time):04d}.png', bbox_inches='tight', pad_inches=0, dpi=200)
    plt.close(fig)


def plot_fish_population(ax, model_parameters):
    ax.plot(model_parameters.total_fish_count, 'b-', label='Total fish population')
    ax.set_xlabel('Time')
    ax.set_ylabel('Number of fish')
    ax.set_title('Fish Population Dynamics')
    ax.legend()


def plot_species_population(ax, model_parameters):
    for species, counts in model_parameters.species_count.items():
        ax.plot(counts, label=species)
    ax.set_xlabel('Time')
    ax.set_ylabel('Number of fish')
    ax.set_title('Species Population Dynamics')
    ax.legend()


def plot_fishing_activity(ax, model_parameters):
    ax.plot(model_parameters.fishermen_data1, 'b-', label='Total catch')
    ax.plot(model_parameters.fishermen_data2, 'r-', label='Current catch')
    ax.set_xlabel('Time')
    ax.set_ylabel('Number of fish caught')
    ax.set_title('Fishing Activity')
    ax.legend()


def plot_cooperation(ax, model_parameters):
    time_steps = range(len(model_parameters.cooperation_levels))
    ax.stackplot(
        time_steps,
        [
            model_parameters.strategy_counts['fully_coop'],
            model_parameters.strategy_counts['coop'],
            model_parameters.strategy_counts['cond_coop'],
            model_parameters.strategy_counts['noncoop'],
            model_parameters.strategy_counts['fully_noncoop'],
        ],
        labels=[
            'Fully Cooperative', 'Cooperative', 'Conditionally Cooperative',
            'Non-cooperative', 'Fully Non-cooperative'
        ],
        colors=['darkgreen', 'lightgreen', 'yellow', 'orange', 'red']
    )
    ax.plot(
        time_steps,
        model_parameters.cooperation_levels,
        'k--',
        label='Average Cooperation Level',
        linewidth=2
    )
    ax.set_xlabel('Time')
    ax.set_ylabel('Number of Fishers / Cooperation Level')
    ax.set_title('Evolution of Cooperation Strategies')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))


def plot_trust_dynamics(ax, model_parameters):
    ax.plot(model_parameters.trust_history, 'b-', label='Average Trust')
    ax.set_xlabel('Time')
    ax.set_ylabel('Trust Level')
    ax.set_title('Evolution of Trust Between Fishers')
    ax.legend()


def plot_summary(model_parameters):
    """
    Plots a comprehensive summary of the model's data over the timespan of a simulation
    and outputs it to PNG files.
    """
    # Plot combined figure
    fig, axs = plt.subplots(5, 1, figsize=(15, 25))

    plot_fish_population(axs[0], model_parameters)
    plot_species_population(axs[1], model_parameters)
    plot_fishing_activity(axs[2], model_parameters)
    plot_cooperation(axs[3], model_parameters)
    plot_trust_dynamics(axs[4], model_parameters)

    plt.tight_layout()
    plt.savefig('simulation_output/dynamics.png', bbox_inches='tight', dpi=200)
    plt.close()

    # Plot cooperation separately
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_cooperation(ax, model_parameters)
    plt.tight_layout()
    plt.savefig('simulation_output/cooperation_evolution.png', bbox_inches='tight', dpi=200)
    plt.close('all')