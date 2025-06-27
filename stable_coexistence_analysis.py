import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from scipy import stats
from datetime import datetime
import os

# Assume DynamicCoop model and parameters module available
import DynamicCoop as dc
import parameters

def run_stability_test(param_dict, n_timesteps=1000, analysis_window=200):
    """
    Run model and compute stability metrics for a given parameter set.
    Returns a dict including a boolean `stable_coexistence`.
    """
    # Backup and set parameters
    orig = {}
    for k, v in param_dict.items():
        if hasattr(parameters, k):
            orig[k] = getattr(parameters, k)
            setattr(parameters, k, v)
    # Handle scale-driven radii
    if 'scale' in param_dict:
        s = param_dict['scale']
        parameters.rad_repulsion   = 0.025 * s
        parameters.rad_orientation = 0.06  * s
        parameters.rad_attraction  = 0.10  * s
        parameters.rad_repulsion_sqr   = parameters.rad_repulsion**2
        parameters.rad_orientation_sqr = parameters.rad_orientation**2
        parameters.rad_attraction_sqr  = parameters.rad_attraction**2

    # Run simulation
    dc.VISUALIZATION_ENABLED = False
    dc.CREATE_GIF = False
    dc.initialize('default')
    fish = []
    for t in range(n_timesteps):
        dc.update_one_unit_time()
        fish.append(dc.total_fish_count[-1])
    fish = np.array(fish)

    # Restore parameters
    for k, v in orig.items(): setattr(parameters, k, v)
    if 'scale' in orig:
        s = orig['scale']
        parameters.rad_repulsion   = 0.025 * s
        parameters.rad_orientation = 0.06  * s
        parameters.rad_attraction  = 0.10  * s
        parameters.rad_repulsion_sqr   = parameters.rad_repulsion**2
        parameters.rad_orientation_sqr = parameters.rad_orientation**2
        parameters.rad_attraction_sqr  = parameters.rad_attraction**2

    # Metrics
    mean_all = fish.mean()
    cv_all   = fish.std() / mean_all
    window   = fish[-analysis_window:]
    mean_win = window.mean()
    cv_win   = window.std() / mean_win
    slope, *_ = stats.linregress(np.arange(len(window)), window)
    persist  = np.mean(fish > 15)
    crash    = fish.min() < 5
    collapse = mean_win < 10
    # recovery
    lows = np.where(fish < mean_all*0.5)[0]
    recov = 1.0
    if len(lows)>0:
        count=0
        for idx in lows:
            if idx < len(fish)-50 and fish[idx+1:idx+51].mean()>1.2*fish[idx]: count += 1
        recov = count/len(lows)
    stable = (not crash and not collapse and mean_win>15 and cv_all<0.8 and
              abs(slope)<0.2 and persist>0.7 and recov>0.3)
    return {'stable_coexistence': stable,
            'mean_all':mean_all,'cv_all':cv_all,'mean_win':mean_win,
            'cv_win':cv_win,'slope':slope,'persist':persist,'recovery':recov}


def explore_ofat(param_ranges, samples=20, reps=5):
    """
    OFAT exploration: varies one param at a time, records stability probability.
    """
    rows=[]
    defaults = {k:np.mean(v) for k,v in param_ranges.items()}
    for pname,(lo,hi) in param_ranges.items():
        values = np.linspace(lo,hi,samples)
        for val in tqdm(values, desc=f"OFAT {pname}"):
            d=defaults.copy(); d[pname]=val
            results=[run_stability_test(d) for _ in range(reps)]
            prob=np.mean([r['stable_coexistence'] for r in results])
            rows.append({'parameter':pname,'value':val,'probability':prob})
    return pd.DataFrame(rows)


def latin_hypercube(n_samples, param_ranges):
    """Generate Latin Hypercube samples within param_ranges."""
    dims = len(param_ranges)
    # unit LHS
    u = np.zeros((n_samples, dims))
    rng = np.random.default_rng()
    for i in range(dims):
        perm = rng.permutation(n_samples)
        u[:,i] = (perm + rng.random(n_samples)) / n_samples
    # scale to ranges
    samples = np.zeros_like(u)
    for i,(k,(lo,hi)) in enumerate(param_ranges.items()):
        samples[:,i] = lo + u[:,i]*(hi-lo)
    return pd.DataFrame(samples, columns=list(param_ranges.keys()))


def explore_multifactor(param_ranges, n_samples=300, reps=3):
    """Multi-factor LHS exploration."""
    samp_df = latin_hypercube(n_samples, param_ranges)
    out=[]
    for _,row in tqdm(samp_df.iterrows(), total=len(samp_df), desc="Multi-factor"):
        d=row.to_dict()
        results=[run_stability_test(d) for _ in range(reps)]
        prob=np.mean([r['stable_coexistence'] for r in results])
        row['probability']=prob
        out.append(row)
    return pd.DataFrame(out)


def plot_results(ofat_df, multi_df, param_ranges):
    """Create OFAT plots and multi-factor visualizations."""
    os.makedirs('figures', exist_ok=True)
    # OFAT: stability probability by param
    g = sns.FacetGrid(ofat_df, col='parameter', col_wrap=3, height=3)
    g.map(plt.plot, 'value', 'probability', marker='o')
    g.set_titles("{col_name}")
    g.set(ylim=(-0.05,1.05))
    plt.tight_layout(); plt.savefig('figures/ofat_prob.png',dpi=300)

    # Multi-factor pairwise
    sns.pairplot(multi_df, vars=list(param_ranges.keys()),
                 hue='probability', palette='viridis', diag_kind='hist')
    plt.suptitle('Multi-factor Stability Landscape', y=1.02)
    plt.savefig('figures/multifactor_pairplot.png',dpi=300)

    # 2D contour for two key params
    p1,p2 = list(param_ranges.keys())[:2]
    pivot = multi_df.pivot_table(index=p2, columns=p1, values='probability')
    plt.figure(figsize=(6,5))
    cs = plt.contourf(pivot.columns, pivot.index, pivot.values, levels=20, cmap='viridis')
    plt.xlabel(p1); plt.ylabel(p2); plt.title(f'Stability Prob. over {p1} vs {p2}')
    plt.colorbar(cs, label='Probability')
    plt.savefig('figures/contour_2D.png', dpi=300)


def main():
    # define ranges for five parameters
    param_ranges = {
        'scale': (0.5, 4.0),
        'imitation_period': (1, 15),
        'cooperation_increase': (0.05, 0.5),
        'q': (0.2, 1.0),
        'trust_decrease': (0.05, 0.5)
    }
    ofat_df = explore_ofat(param_ranges, samples=20, reps=5)
    multi_df = explore_multifactor(param_ranges, n_samples=300, reps=3)
    plot_results(ofat_df, multi_df, param_ranges)
    # save data
    ofat_df.to_csv('figures/ofat_results.csv',index=False)
    multi_df.to_csv('figures/multifactor_results.csv',index=False)

if __name__ == '__main__':
    main()
