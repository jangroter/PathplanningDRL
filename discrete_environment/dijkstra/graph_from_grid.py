"""
This script takes as an input the gridworld for a specified resolution and generates the Dijkstra solutions for the corresponding starting nodes
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nwx

import matplotlib.cm as cm
import matplotlib.colors as colors

import pandas as pd

def offGridMove(newState, oldState,n):
    if oldState % n == 0 and newState  % n == n - 1:
        return True
    elif oldState % n == n - 1 and newState % n == 0:
        return True
    else:
        return False

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

runway = '27'

resolution = 65
km_per_cell = 520/resolution

block_cell_length = int(39/km_per_cell)+2
block_cell_height = int(13/km_per_cell)+1

ams = np.genfromtxt(f'gridworlds/gridworld{str(resolution)}.csv', delimiter=',')
m,n = ams.shape

noise_cost_map = ams.flatten()
print(noise_cost_map)
rewardgrid = np.reshape(np.copy(noise_cost_map),(m,n))
valid_states = np.arange(len(noise_cost_map))

print(np.mean(noise_cost_map))
fuel_cost = np.mean(noise_cost_map)*1

if resolution%2==0:
    terminal_state = int((resolution/2)+(resolution*resolution/2))
else:
    terminal_state = int((resolution*resolution/2))

blockedsquares = np.array([])
for i in [-1,1]:
    for j in range(block_cell_height):
        indices = np.arange(block_cell_length) + terminal_state - 1 + i*(j+1)*resolution
        blockedsquares = np.append(blockedsquares,indices)

blockedsquares = np.append(blockedsquares,terminal_state-1).astype(int)
noise_cost_map[blockedsquares] = 100000000
rewardgrid = np.reshape(np.copy(noise_cost_map),(m,n))

circle_rad = 255

for i in range(len(noise_cost_map)):
    y = ((i // n) - n/2)*km_per_cell
    x = ((i % n) - n/2)*km_per_cell
    dis = np.sqrt(y**2 + x**2)
    if dis > circle_rad:
        noise_cost_map[i] = 10000000

action_space = [[-n,1],[n,1],[-1,1],[1,1],[-n-1,1.41],[-n+1,1.41],[n-1,1.41],[n+1,1.41]] #[U,D,L,R,UL,UR,DL,DR]

ams_graph = nwx.DiGraph()

for node in valid_states:
    ams_graph.add_node(node)
    for action,cost in action_space:
        connected_node = node + action
        if not offGridMove(connected_node,node,n) and connected_node in valid_states:
            edge_weight = fuel_cost*cost + noise_cost_map[connected_node]*cost
            ams_graph.add_edge(node,connected_node,weight=edge_weight)

fig, ax = plt.subplots(figsize=(10, 10), dpi=100)

c_map = cm.get_cmap("Blues").copy()
c_map = truncate_colormap(c_map,0,0.7)
c_map.set_under(color='w')  
c_map.set_over(color='k')  

im = ax.imshow(rewardgrid,cmap=c_map,norm=colors.LogNorm(vmin=10000,vmax=1000000))

startstates = np.genfromtxt(f'start_nodes/start_nodes_{str(resolution)}.csv').astype(int)

found_paths = pd.DataFrame({'paths': pd.Series(dtype=int)})
for state in startstates:
    found_path = np.array(nwx.dijkstra_path(ams_graph, state, terminal_state))
    delpath = []
    for i in range(len(found_path)-2):
        if noise_cost_map[found_path[i+1]] == 0 and noise_cost_map[found_path[i+2]] == 0 and noise_cost_map[found_path[i+2]+n] == 0 and noise_cost_map[found_path[i+2]-n] == 0 and noise_cost_map[found_path[i+2]+1] == 0 and noise_cost_map[found_path[i+2]-1] == 0:
            delpath = delpath + [(i+1)]
    found_path = np.delete(found_path, delpath)
    ys = found_path // n
    xs = found_path % n
    ax.plot(xs,ys,linestyle='solid',color='black',alpha=0.2)
    found_paths.loc[len(found_paths)] = [found_path]

found_paths.to_pickle(f'node_paths/found_paths_{str(resolution)}.pkl')


