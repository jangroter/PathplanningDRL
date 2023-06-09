"""
Script used for plotting the Artificial Potential Field representation of the value function
"""

import agent as agent
import functions as fn
import areafilter as af
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import matplotlib.cm as cm

# !! Make sure that the network architecture used in "agent" matches the loaded models
value = agent.Value(modelname="E_2_256-x1/model/vf.pt")
actor = agent.Agent(modelname="E_2_256-x1/model/actor.pt")

x0 = np.arange(-1, 1, 0.005)
y0 = np.arange(-1, 1, 0.005)
x,y = np.meshgrid(x0,y0)

index = np.arange(len(x0))

z = np.zeros(np.shape(x))

for i in index:
    for j in index:
        state = [x0[i],y0[j]]
        z[i,j] = value.step(state)[0].detach().cpu().numpy() * -1

fig, ax = plt.subplots(figsize=(15, 15))

print(np.max(z))
print(np.min(z))
z[z>0] = 0
ax.imshow(np.rot90(z),extent=[-277800,277800,-277800,277800], cmap=cm.coolwarm)

circle_lat = 52.3322
circle_lon = 4.75

lat_0 = 0
lon_0 = 0

lat_1 = 0
lon_1 = 0

bearing = np.arange(0,360,1)
distance = np.random.uniform(low=50.0, high=275.0, size=(360,))

line = af.poly_arc(circle_lat,circle_lon,20,30,-30)

for i,j in zip(bearing,distance):
    make_line = True
    count = 0

    lat = np.array([])
    lon = np.array([])

    x = np.array([])
    y = np.array([])

    while make_line:
        if count == 0:
            lat_0, lon_0 = fn.get_spawn(i, radius = j)
            lat = np.append(lat,lat_0)
            lon = np.append(lon,lon_0)
            x_, y_ = fn.get_xy(lat_0,lon_0)
            x = np.append(x,x_)
            y = np.append(y,y_)

        state = fn.get_state(lat_0, lon_0)
        action = actor.step(state)
        lat_1, lon_1 = np.rad2deg(fn.do_action(action,lat_0,lon_0))
        lat = np.append(lat,lat_1)
        lon = np.append(lon,lon_1)
        x_, y_ = fn.get_xy(lat_1,lon_1)
        x = np.append(x,x_)
        y = np.append(y,y_)
        if count > 35:
            make_line = False
        if af.checkIntersect(line,lat_0,lon_0,lat_1,lon_1):
            make_line = False
        
        count += 1
        lat_0 = lat_1
        lon_0 = lon_1
    ax.plot(x,y,'w',alpha=0.4,linewidth=3)

fig.savefig('value_E_2_256-x1.png')
plt.close(fig)