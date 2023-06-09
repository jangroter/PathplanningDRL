"""
Script that generates the corresponding lat lon coordinates of the waypoints set by the Dijkstra algorithm
"""

import numpy as np
import pandas as pd
import functions as fn
import areafilter as af

resolution = 65
weight = 1

found_paths = pd.read_pickle(f'node_paths/found_paths_{str(resolution)}.pkl')

latlon_paths = pd.DataFrame({'lat': pd.Series(dtype=float), 'lon': pd.Series(dtype=float)})

cell_size = int(520/resolution)
m = resolution
n = resolution

bearing = np.arange(0,360,10)
distance = np.zeros(36)+250

nm = 1852.
nm2km = nm/1000.
circle_lat = 52.3322
circle_lon = 4.75
circlerad = 150. * nm2km

for i,bear,dist in zip(range(len(found_paths)),bearing,distance):
    found_path = found_paths['paths'].iloc[i]

    ys = found_path // n
    xs = found_path % n

    x = (xs - int(n/2))*cell_size
    y = ((ys - int(m/2))*cell_size*-1)

    x = x/circlerad
    y = y/circlerad

    lat = np.array([])
    lon = np.array([])

    for xi, yi in zip(x,y):
        state = np.array([xi,yi])
        lati,loni = fn.get_latlon_state(state)
        lat = np.append(lat,lati)
        lon = np.append(lon,loni)
    
    lat[-1] = circle_lat
    lon[-1] = circle_lon

    latspawn, lonspawn = fn.get_spawn(bear,radius=dist)

    lat[0] = latspawn
    lon[0] = lonspawn

    latlon_paths.loc[i+1] = [lat,lon]

latlon_paths.to_pickle(f'latlon_paths/latlon_paths_x{weight}_{resolution}.pkl')