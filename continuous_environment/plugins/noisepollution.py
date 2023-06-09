"""
Plugin to keep track of all the Noise emissions corrected for the number of people affected,
Noise computed in accordance with doc 29. and NPD tables, although for now just for 1 thrust setting.
"""

import numpy as np
import bluesky as bs
from bluesky import core, stack, traf
from bluesky.tools.aero import ft, nm, fpm, Rearth, kts
from bluesky.tools import geo, aero, areafilter, plotter
from openap import FuelFlow

#NPD data
base_noise = 85 #dBA
base_distance = 1000*0.3048

noise_cutoff = 55 #dBA

W_0 = 10**-12

base_sound = 10**(base_noise/10) * W_0
base_sound_1m = base_sound/((1/base_distance)**2)

sound_cutoff = 10**(noise_cutoff/10) * W_0

noisepollution = None

def init_plugin():

    global noisepollution
    noisepollution = Noisepollution()

    config = {
        'plugin_name': 'NOISEPOLLUTION',
        'plugin_type': 'sim',
        }
    
    return config

class Noisepollution(core.Entity):
    def __init__(self):
        super().__init__()
        
        self.pop_array = np.genfromtxt('population_1km.csv', delimiter = ',')
        self.x_array = np.genfromtxt('x_1km.csv', delimiter = ',')
        self.y_array = np.genfromtxt('y_1km.csv', delimiter = ',')

        self.schiphol = [52.3068953,4.760783] # lat,lon coords of schiphol for reference to x_array and y_array

        with self.settrafarrays():
            self.noise = np.array([])
        
    def create(self, n=1):
        super().create(n)
        self.noise[-n:] = 0

    def update(self, dt):
        for i in range(traf.ntraf):
            brg, dist = geo.kwikqdrdist(self.schiphol[0], self.schiphol[1], bs.traf.lat[i], bs.traf.lon[i])

            x = np.sin(np.radians(brg))*dist*nm
            y = np.cos(np.radians(brg))*dist*nm
            z = bs.traf.alt[i]

            x_index_min = int(((x+500000)/1000)-10)
            x_index_max = int(((x+500000)/1000)+10)
            y_index_min = int(((500000 - y)/1000)-10)
            y_index_max = int(((500000 - y)/1000)+10)

            distance2 = (self.x_array[y_index_min:y_index_max,x_index_min:x_index_max]-x)**2 + (self.y_array[y_index_min:y_index_max,x_index_min:x_index_max]-y)**2 + z**2
            sound = base_sound_1m/(distance2)
            sound[sound<sound_cutoff] = 0

            self.noise[i] += np.sum(self.pop_array[y_index_min:y_index_max,x_index_min:x_index_max] * sound)*dt
