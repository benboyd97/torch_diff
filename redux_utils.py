
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import json
import os

def load_and_plot(filepath,label=None,ls = None,color=None):
    with open(filepath, 'r') as f:
        # Standard JSON load
        res = json.load(f)
    
    # Convert lists to numpy for plotting
    cx = np.array(res['cal_x'])
    cy = np.array(res['cal_y'])
    
    # Handle NaNs for the plot (mask them out so the line doesn't break)
    mask = ~np.isnan(cy)
    
    plt.plot(cx[mask], cy[mask], marker='s', label=label,linestyle=ls,color=color)



def get_redux_stats(stat_name, prior_list, ht='lowrank', st='samp', folder='laplace_redux_dicts',subnet = 'all'):
    """
    Returns a numpy array of a specific metric across your prior sweep.
    """
    vals = []
    redux_prec = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]


    
    for p in prior_list:
        # Matches your filename: f'laplace_redux_dicts/redux_{ht}_{st}_prior{str(p)}.json'
        filename = f'redux_{subnet}_{ht}_{st}_prior{str(p)}.json'
        path = os.path.join(folder, filename)
        
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
                # .get() prevents crashing if a specific metric failed to save
                vals.append(data.get(stat_name, np.nan))
        else:
            print(f"Missing: {filename}")
            vals.append(np.nan)
            
    return np.array(vals)


#load_and_plot('laplace_redux_dicts/redux_kron_linear_prior10.0.json')

#priors = [0.1, 1.0, 10.0]
#lowrank_samp_nll = get_redux_stats('ece', priors, ht='kron', st='samp')