"""
Plugin to keep track of all the fuel used by the different aircraft according to the OpenAP performance models
"""

import numpy as np
import bluesky as bs
from bluesky import core, stack, traf
from openap import FuelFlow

fuelconsumption = None

def init_plugin():

    global fuelconsumption
    fuelconsumption = Fuelconsumption()

    config = {
        'plugin_name': 'FUELCONSUMPTION',
        'plugin_type': 'sim',
        }
    
    return config

class Fuelconsumption(core.Entity):
    def __init__(self):
        super().__init__()
        
        self.ff = FuelFlow('a320')
        self.mass = 66000

        with self.settrafarrays():
            self.fuelconsumed = np.array([])
            self.fuelconsumedtotal = np.array([])
        
    def create(self, n=1):
        super().create(n)
        self.fuelconsumed[-n:] = 0
        self.fuelconsumedtotal[-n:] = 0

    def update(self, dt):
        for i in range(traf.ntraf):
            path_angle = np.degrees(np.arctan2(traf.vs[i],traf.tas[i]))
            fuelflow = self.ff.enroute(self.mass,traf.tas[i],traf.alt[i],path_angle)
            self.fuelconsumed[i] += fuelflow*dt
            self.fuelconsumedtotal[i] += fuelflow*dt
