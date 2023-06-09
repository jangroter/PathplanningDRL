import numpy as np
import matplotlib.pyplot as plt
import networkx as nwx

import matplotlib.cm as cm
import matplotlib.colors as colors

import pandas as pd

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

ams = np.genfromtxt('gridworld40.csv', delimiter=',')
m,n = ams.shape
ams = ams

noise_cost_map = ams.flatten()
rewardgrid = np.reshape(noise_cost_map,(m,n))

fig, ax = plt.subplots(figsize=(10, 10), dpi=100)

c_map = cm.get_cmap("Blues").copy()
c_map = truncate_colormap(c_map,0,0.8)
c_map.set_under(color='w')  
c_map.set_over(color='k')  

im = ax.imshow(rewardgrid,cmap=c_map, norm=colors.LogNorm(vmin=1000,vmax=1000000))
cbar = ax.figure.colorbar(im, ax=ax,fraction=0.046, pad=0.04, label='Population size (log scale)')
cbar.ax.tick_params(labelsize=15) 

fig.savefig('population_map.png')