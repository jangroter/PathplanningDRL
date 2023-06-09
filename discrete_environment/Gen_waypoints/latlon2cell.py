"""
Script that generates a list of nodes on a nxn grid of m km per cell
closest resembling a grid of 36 points equally spaced on a circle of 
radius 250 km centered around Schiphol.
"""

import numpy as np
import functions as fn
import areafilter as af

m = 40
n = 40

nm = 1852.
nm2km = nm/1000.

cell_size = 13

bearing = np.arange(0,360,10)
distance = np.zeros(36)+250

circle_lat = 52.3322
circle_lon = 4.75
circlerad = 150. * nm2km

start_nodes = []
for i,j in zip(bearing,distance):
    lat, lon = fn.get_spawn(i, radius = j)    
    x, y = fn.get_state(lat, lon)
    x = x * circlerad
    y = y * circlerad

    x_index = round(x / cell_size)+int(n/2)
    y_index = -1*round(y / cell_size) + int(m/2)

    print(x_index, y_index)
    start_node = int(x_index + y_index*m)
    start_nodes = start_nodes + [start_node]

start_nodes = np.array(start_nodes)
np.savetxt('start_nodes_40.csv', start_nodes)