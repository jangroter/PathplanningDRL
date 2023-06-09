import functions as fn
import areafilter as af
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import matplotlib.cm as cm
import pandas as pd

pop_array = np.genfromtxt('population_1km.csv',delimiter=',')

x_index_min = int(((500000)/1000)-300)
x_index_max = int(((500000)/1000)+300)
y_index_min = int(((500000)/1000)-300)
y_index_max = int(((500000)/1000)+300)

x_array = np.genfromtxt('x_1km.csv',delimiter=',')
y_array = np.genfromtxt('y_1km.csv',delimiter=',')

pop_array = pop_array[y_index_min:y_index_max,x_index_min:x_index_max]

circle_lat = 52.3322
circle_lon = 4.75

bearing = np.arange(0,360,0.5)
distance = np.zeros(720)+275

c_map = cm.get_cmap("Blues").copy()
c_map.set_bad('w')

line_color = 'k' #(155/255,66/255,165/255,0.5)
marker_color = 'k' #(155/255,66/255,165/255,1)

models =['x025_65','x05_65','x1_65','x2_65','x4_65']
num_index = 36

for model in models:
    waypoints = pd.read_pickle(f'dijkstra_paths/latlon_paths_{model}.pkl')

    line = af.poly_arc(circle_lat,circle_lon,20,30,-30)
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.tick_params(left = False, right = False , labelleft = False ,
                    labelbottom = False, bottom = False)
    ax.imshow(pop_array,cmap=c_map,extent=[-300000,300000,-300000,300000],norm=LogNorm(vmin=100,vmax=100000))

    turns = []
    for index in np.arange(num_index):
        # for j in distance:
        make_line = True
        count = 0

        lats = waypoints['lat'].iloc[index]
        lons = waypoints['lon'].iloc[index]

        x = np.array([])
        y = np.array([])

        for lat,lon in zip(lats,lons):
            x_, y_ = fn.get_xy(lat,lon)
            x = np.append(x,x_)
            y = np.append(y,y_)
        
        turn = 0
        for i in range(len(x)-2):
            x1 = x[i]
            y1 = y[i]
            
            x2 = x[i+1]
            y2 = y[i+1]

            x3 = x[i+2]
            y3 = y[i+2]

            a = np.sqrt((x1-x2)**2 + (y1-y2)**2)
            b = np.sqrt((x2-x3)**2 + (y2-y3)**2)
            c = np.sqrt((x1-x3)**2 + (y1-y3)**2)

            angle = np.arccos((a**2 + b**2 - c**2)/(2*a*b))
            if angle < 0.95*np.pi:
                turn+=1
        
        turns.append(turn)

        ax.plot(x,y,'k',alpha=1,linewidth=3,color=line_color,marker='',markersize=5,markeredgecolor=marker_color,markerfacecolor=marker_color)

    fig.savefig(f'output/Dijkstra_{model}.png',bbox_inches='tight')
    print(model, np.mean(turns))
    plt.close(fig)