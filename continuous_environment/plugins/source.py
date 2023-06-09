from bluesky import core, stack, traf  #, settings, navdb, sim, scr, tools
from bluesky.tools.aero import ft, nm, fpm, Rearth, kts
from bluesky.tools import geo, aero, areafilter, plotter

import bluesky as bs
import numpy as np
import area
import random
import math

from sink import poly_arc


def init_plugin():
    
    source = Source()
    
    # Configuration parameters
    config = {
        # The name of your plugin
        'plugin_name':     'SOURCE',

        # The type of this plugin. For now, only simulation plugins are possible.
        'plugin_type':     'sim',
        }

    # init_plugin() should always return a configuration dict.
    return config


class Source(core.Entity):
    
    # Define border of airspace with center at AMS
    circlelat           = 52.3322
    circlelon           = 4.75
    circlerad           = 150
    
    speed               = 250

    ac_nmbr = 1

    def __init__(self):
        super().__init__()
        stack.stack(f'Circle source {self.circlelat} {self.circlelon} {self.circlerad}')
        stack.stack('COLOR source red')
        poly_arc(self.circlelat,self.circlelon,20,30,-30)
    
    def create_ac(self): 
        acid                    = 'KL' + str(self.ac_nmbr)
        heading                 = random.randint(0,359)
        altitude                = 15000 * ft
        lat,lon                 = self.get_spawn(heading)
        speed                   = self.speed * kts
        
        traf.cre(acid,'a320',lat,lon,heading,altitude,speed)
        stack.stack(f'ADDWPTMODE {acid} FLYOVER')
        self.ac_nmbr += 1
                
    
    def get_spawn(self, heading):

        enterpoint = random.randint(-9999,9999)/10000
       
        bearing     = np.deg2rad(heading + 180) + math.asin(enterpoint)

        lat         = np.deg2rad(self.circlelat)
        lon         = np.deg2rad(self.circlelon)

        radius = self.circlerad * 1.852 * random.random()

        latspawn    = np.rad2deg(self.get_new_latitude(bearing,lat, radius))
        lonspawn    = np.rad2deg(self.get_new_longitude(bearing, lon, lat, np.deg2rad(latspawn), radius))
        
        return latspawn, lonspawn
        
    def get_new_latitude(self,bearing,lat,radius):
        R = Rearth/1000.
        return math.asin( math.sin(lat)*math.cos(radius/R) +\
               math.cos(lat)*math.sin(radius/R)*math.cos(bearing))
        
    def get_new_longitude(self,bearing,lon,lat1,lat2,radius):
        R   = Rearth/1000.
        return lon + math.atan2(math.sin(bearing)*math.sin(radius/R)*\
                     math.cos(lat1),math.cos(radius/R)-math.sin(lat1)*math.sin(lat2))
        