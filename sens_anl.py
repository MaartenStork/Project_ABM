# import numpy as np
# import matplotlib.pyplot as plt
# from tqdm import tqdm
# import DynamicCoop as dc
# import parameters

# # For Sobol sampling
# from SALib.sample import sobol as sobol_sample
# from SALib.analyze import sobol as sobol_analyze


# def run_model():
#     """
#     Re-initialize and run the DynamicCoop model for n steps,
#     return final total fish count.
#     """
#     dc.initialize('reproduction_rate')
#     for _ in range(parameters.n):
#         dc.update_one_unit_time()
#     return dc.total_fish_count[-1]


# def get_param_names_and_bounds():
#     # gather numeric parameters
#     names = [name for name, val in vars(parameters).items()
#              if not name.startswith('_') and isinstance(val, (int, float))]
#     # exclude constants and fixed scenario params
#     exclude = {
#         'n', 'Area', 'Length_Area', 'Half_Length_Area', 'Half_Length',
#         'rad_repulsion_sqr','rad_orientation_sqr','rad_attraction_sqr','r_sqr',
#         'Xa','Xb','Ya','Yb','Xm','Xn','Ym','Yn','Xp','Xq','Yp','Yq',
#         'fully_noncoop','noncoop','cond_coop','coop','fully_coop', 'Dist_MPA', 'Frac_MPA', 'K', 
#         'Time_MPA', 'cooperation_increase', 'fish_density_threshold', 'imitation_period', 
#         'imitation_prob', 'imitation_radius', 'initial_trust', 'move_fishers', 
#         'num_fishers', 'plot_update_freq', 'q', 'r', 'rad_attraction', 'rad_orientation', 
#         'rad_repulsion', 'scale', 'threshold_memory', 'threshold_radius', 'trust_decrease', 
#         'trust_increase', 'trust_memory'
#     }
#     names = [p for p in names if p not in exclude]
#     names.sort()
#     # bounds ±50%
#     bounds = [[getattr(parameters, p)*0.5, getattr(parameters, p)*1.5] for p in names]
#     return names, bounds


# def ofat_analysis(param_names, n_points=4, n_reps=2):  # reduced for quick test
#     # store defaults
#     defaults = {p: getattr(parameters, p) for p in param_names}
#     ofat_results = {}
#     for p in param_names:
#         x = np.linspace(defaults[p]*0.5, defaults[p]*1.5, n_points)
#         means = np.zeros(n_points)
#         stds  = np.zeros(n_points)
#         for i, xi in enumerate(tqdm(x, desc=f"OFAT {p}")):
#             setattr(parameters, p, xi)
#             Y = [run_model() for _ in range(n_reps)]
#             means[i] = np.mean(Y)
#             stds[i]  = np.std(Y, ddof=1)
#         ofat_results[p] = (x, means, stds)
#         setattr(parameters, p, defaults[p])
#     return ofat_results


# def plot_ofat(ofat_results, filename='ofat_full.png'):
#     n = len(ofat_results)
#     cols = 3
#     rows = int(np.ceil(n/cols))
#     fig, axes = plt.subplots(rows, cols, figsize=(5*cols,4*rows))
#     axes = axes.flatten()
#     for ax, (p,(x,m,s)) in zip(axes, ofat_results.items()):
#         ax.plot(x, m, '-k', lw=2)
#         ax.fill_between(x, m-s, m+s, color='lightgray', alpha=0.5)
#         ax.set_title(p)
#         ax.set_xlabel(p)
#         ax.set_ylabel('Final fish')
#         ax.grid(True)
#     for ax in axes[n:]: ax.set_visible(False)
#     plt.tight_layout()
#     plt.savefig(filename, dpi=200)
#     print(f"OFAT results saved to {filename}")
#     plt.show()


# def sobol_analysis(param_names, bounds, sample_size=64):  # reduced for quick test
#     # define problem
#     problem = {'num_vars': len(param_names), 'names': param_names, 'bounds': bounds}
#     # sample
#     print(f"Sampling for Sobol: N={sample_size}")
#     X = sobol_sample.sample(problem, sample_size, calc_second_order=False)
#     Y = np.zeros(X.shape[0])
#     for i, xi in enumerate(tqdm(X, desc='Sobol runs')):
#         # set parameters
#         for name, val in zip(param_names, xi): setattr(parameters, name, val)
#         Y[i] = run_model()
#     # restore defaults
#     for name in param_names: setattr(parameters, name, getattr(parameters, name))
#     # analyze
#     Si = sobol_analyze.analyze(problem, Y, calc_second_order=False, print_to_console=True)
#     return Si


# def plot_sobol(Si, param_names, filename='sobol_indices.png'):
#     S1 = Si['S1']
#     S1_conf = Si['S1_conf']
#     x = np.arange(len(param_names))
#     fig, ax = plt.subplots(figsize=(8,5))
#     ax.errorbar(x, S1, yerr=S1_conf, fmt='o', capsize=5)
#     ax.set_xticks(x); ax.set_xticklabels(param_names, rotation=45, ha='right')
#     ax.set_ylabel('First-order Sobol index')
#     ax.set_title('Sobol Sensitivity')
#     ax.grid(True)
#     plt.tight_layout()
#     plt.savefig(filename, dpi=200)
#     print(f"Sobol plot saved to {filename}")
#     plt.show()


# def main():
#     # get parameters
#     param_names, bounds = get_param_names_and_bounds()
#     print(f"Parameters included: {param_names}")

#     # OFAT
#     ofat_res = ofat_analysis(param_names, n_points=4, n_reps=2)  # quick test settings
#     plot_ofat(ofat_res, filename='ofat_full.png')

#     # Sobol
#     Si = sobol_analysis(param_names, bounds, sample_size=64)  # quick test settings
#     plot_sobol(Si, param_names, filename='sobol_indices.png')

# if __name__=='__main__':
#     main()

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import DynamicCoop as dc
import parameters

# For Sobol sampling
from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol as sobol_analyze
# For Morris screening
from SALib.sample import morris as morris_sample
from SALib.analyze import morris as morris_analyze


def run_model():
    """
    Re-initialize and run the DynamicCoop model for n steps,
    return final total fish count.
    """
    dc.initialize('reproduction_rate')
    for _ in range(parameters.n):
        dc.update_one_unit_time()
    return dc.total_fish_count[-1]


def get_param_names_and_bounds():
    # gather numeric parameters
    names = [name for name, val in vars(parameters).items()
             if not name.startswith('_') and isinstance(val, (int, float))]
    exclude = {
        'n', 'Area', 'Length_Area', 'Half_Length_Area', 'Half_Length',
        'rad_repulsion_sqr','rad_orientation_sqr','rad_attraction_sqr','r_sqr',
        'Xa','Xb','Ya','Yb','Xm','Xn','Ym','Yn','Xp','Xq','Yp','Yq',
        'fully_noncoop','noncoop','cond_coop','coop','fully_coop','Dist_MPA', 
        'Frac_MPA', 'K', 'Time_MPA', 'cooperation_increase', 'fish_density_threshold', 
        'imitation_period', 'imitation_prob', 'imitation_radius', 'initial_trust', 
        'move_fishers', 'num_fishers', 'plot_update_freq', 'q', 'r', 'rad_attraction', 
        'rad_orientation', 'rad_repulsion', 'scale', 'threshold_memory', 'threshold_radius', 
        'trust_decrease', 'trust_increase', 'trust_memory'
    }
    names = [p for p in names if p not in exclude]
    names.sort()
    # bounds ±50%
    bounds = [[getattr(parameters, p)*0.5, getattr(parameters, p)*1.5] for p in names]
    return names, bounds


def ofat_analysis(param_names, n_points=4, n_reps=2):
    defaults = {p: getattr(parameters, p) for p in param_names}
    ofat_results = {}
    for p in param_names:
        x = np.linspace(defaults[p]*0.5, defaults[p]*1.5, n_points)
        means = np.zeros(n_points)
        stds  = np.zeros(n_points)
        for i, xi in enumerate(tqdm(x, desc=f"OFAT {p}")):
            setattr(parameters, p, xi)
            Y = [run_model() for _ in range(n_reps)]
            means[i] = np.mean(Y)
            stds[i]  = np.std(Y, ddof=1)
        ofat_results[p] = (x, means, stds)
        setattr(parameters, p, defaults[p])
    return ofat_results


def plot_ofat(ofat_results, filename='ofat_full.png'):
    n = len(ofat_results)
    cols = 3
    rows = int(np.ceil(n/cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols,4*rows))
    axes = axes.flatten()
    for ax, (p,(x,m,s)) in zip(axes, ofat_results.items()):
        ax.plot(x, m, '-k', lw=2)
        ax.fill_between(x, m-s, m+s, color='lightgray', alpha=0.5)
        ax.set_title(p)
        ax.set_xlabel(p)
        ax.set_ylabel('Final fish')
        ax.grid(True)
    for ax in axes[n:]: ax.set_visible(False)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    print(f"OFAT results saved to {filename}")
    plt.show()


def sobol_analysis(param_names, bounds, sample_size=16):
    problem = {'num_vars': len(param_names), 'names': param_names, 'bounds': bounds}
    print(f"Sampling for Sobol: N={sample_size}")
    X = sobol_sample.sample(problem, sample_size, calc_second_order=False)
    Y = np.zeros(X.shape[0])
    for i, xi in enumerate(tqdm(X, desc='Sobol runs')):
        for name, val in zip(param_names, xi): setattr(parameters, name, val)
        Y[i] = run_model()
    Si = sobol_analyze.analyze(problem, Y, calc_second_order=False, print_to_console=False)
    return Si


def plot_sobol(Si, param_names, filename='sobol_indices.png'):
    S1 = np.array(Si['S1'])
    S1_conf = np.array(Si['S1_conf'])
    idx = np.argsort(-S1)
    sorted_names = [param_names[i] for i in idx]
    sorted_S1 = S1[idx]
    sorted_conf = S1_conf[idx]
    fig, ax = plt.subplots(figsize=(8, max(4, len(param_names)*0.3)))
    y_pos = np.arange(len(sorted_names))
    ax.barh(y_pos, sorted_S1, xerr=sorted_conf, align='center', color='skyblue', ecolor='gray', capsize=4)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_names)
    ax.invert_yaxis()
    ax.set_xlabel('First-order Sobol index')
    ax.set_title('Factor Prioritization via Sobol Indices')
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    print(f"Sobol prioritization plot saved to {filename}")
    plt.show()


def morris_analysis(param_names, bounds, num_trajectories=10, grid_levels=4):
    problem = {'num_vars': len(param_names), 'names': param_names, 'bounds': bounds}
    print(f"Sampling for Morris: trajectories={num_trajectories}, levels={grid_levels}")
    X = morris_sample.sample(problem, N=num_trajectories, num_levels=grid_levels, optimal_trajectories=None)
    Y = np.zeros(X.shape[0])
    for i, xi in enumerate(tqdm(X, desc='Morris runs')):
        for name, val in zip(param_names, xi): setattr(parameters, name, val)
        Y[i] = run_model()
    Si = morris_analyze.analyze(problem, X, Y, conf_level=0.95, print_to_console=False)
    return Si


def plot_morris(Si, param_names, filename='morris_ee.png'):
    mu = np.abs(Si['mu'])        # absolute mean elementary effects
    sigma = Si['sigma']
    fig, ax = plt.subplots(figsize=(6,6))
    ax.scatter(mu, sigma)
    for i, name in enumerate(param_names):
        ax.text(mu[i], sigma[i], name, fontsize=8)
    ax.set_xlabel(r'$\mu^*$ (mean absolute EE)')
    ax.set_ylabel(r'$\sigma$ (EE standard deviation)')
    ax.set_title('Morris Elementary Effects Screening')
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    print(f"Morris EE plot saved to {filename}")
    plt.show()


def main():
    param_names, bounds = get_param_names_and_bounds()
    print(f"Parameters included ({len(param_names)}): {param_names}")

    # # OFAT
    # ofat_res = ofat_analysis(param_names, n_points=4, n_reps=2)
    # plot_ofat(ofat_res)

    # # Sobol
    # Si_sob = sobol_analysis(param_names, bounds, sample_size=16)
    # plot_sobol(Si_sob, param_names)

    # Morris
    Si_m = morris_analysis(param_names, bounds, num_trajectories=10, grid_levels=4)
    plot_morris(Si_m, param_names)

if __name__=='__main__':
    main()
