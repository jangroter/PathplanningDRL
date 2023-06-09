from ctypes.wintypes import WPARAM
from bluesky import core, stack, traf, tools, settings 
from bluesky.tools.aero import ft, nm, fpm, Rearth, kts
from bluesky.tools import geo, aero, areafilter, plotter
from bluesky.traffic import Route

import bluesky as bs
import numpy as np
import math
import matplotlib.pyplot as plt
from pathlib import Path

from plugins.source import Source
import plugins.functions as fn
import plugins.fuelconsumption as fc 
import plugins.noisepollution as noisepol

import pandas as pd

timestep = 5
state_size = 2
action_size = 2

nm2km = nm/1000

circlerad = 150. * nm2km
max_action = 25. * nm2km

max_episode_length = 15

circle_lat = 52.3322
circle_lon = 4.75

model_name = 'dijkstra_x1_65' 

fuel_array = np.array([])
noise_array = np.array([])

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def init_plugin():
    
    dijkstra_evaluator = dijkstra_evaluator()
    
    # Configuration parameters
    config = {
        # The name of your plugin
        'plugin_name':     'DIJKSTRA_EVALUATOR',

        # The type of this plugin. For now, only simulation plugins are possible.
        'plugin_type':     'sim',
        }

    # init_plugin() should always return a configuration dict.
    return config

class dijkstra_evaluator(core.Entity):  
    def __init__(self):
        super().__init__()

        self.rewards = np.array([])

        self.index = 0
        # load Dijkstra output waypoints that are to be evaluated
        self.waypoints = pd.read_pickle('latlon_paths_x1_65.pkl')
        self.indexmax = len(self.waypoints)

        self.first = True

        self.fuel_array = np.array([])
        self.noise_array = np.array([])

        self.action_required = False
        self.AC_present = False
        self.new_wpt_set = True
        self.print = False
        
        self.wptdist = 0
        self.wptdist_old = 0

        self.nac = 0

        self.source = Source()

        self.fuel = np.array([])
        self.noise = np.array([])

        with self.settrafarrays():
            self.totreward = np.array([])  # Total reward this AC has accumulated
            self.nactions = np.array([])  # Total reward this AC has accumulated

            self.state = []    # Latest state information
            self.action = []    # Latest selected actions

    def create(self, n=1):
        super().create(n)
        self.totreward[-n:] = 0

    @core.timed_function(name='dijkstra_evaluator', dt=timestep)
    def update(self):
        fc.fuelconsumption.update(timestep)
        noisepol.noisepollution.update(timestep)
        
        idx = 0
        
        self.check_ac()
        if bs.traf.ntraf != 0:
            self.check_done(idx)

    def check_ac(self):
        if bs.traf.ntraf == 0 and self.index < self.indexmax:
            lat0 = self.waypoints['lat'].iloc[self.index][0]
            lon0 = self.waypoints['lon'].iloc[self.index][0]
            heading,_ = geo.kwikqdrdist(lat0,lon0,circle_lat,circle_lon)

            stack.stack(f'CRE kl001 a320 {lat0} {lon0} {heading} {15000} {250}')
            for lat,lon,name in zip(self.waypoints['lat'].iloc[self.index][1:],self.waypoints['lon'].iloc[self.index][1:],np.arange(len(self.waypoints['lon'].iloc[self.index][1:]))):
                wpt = 'wpt' + str(name)
                stack.stack(f'ADDWPT kl001 {lat} {lon}')
            stack.stack('LNAV kl001 ON')

            self.index += 1

    def check_past_wpt(self, idx):
        if self.new_wpt_set:
            dis = fn.haversine(bs.traf.lat[idx], bs.traf.lon[idx], bs.traf.actwp.lat[idx], bs.traf.actwp.lon[idx])
            self.wptdist = dis
            self.wptdist_old = dis
            self.new_wpt_set = False
        else:
            dis = fn.haversine(bs.traf.lat[idx], bs.traf.lon[idx], bs.traf.actwp.lat[idx], bs.traf.actwp.lon[idx])
            if self.wptdist - dis < 0 and self.wptdist_old - self.wptdist > 0:
                return True
            if self.wptdist - dis < 0 and self.wptdist_old - self.wptdist < 0:
                wptlat = bs.traf.actwp.lat[idx]
                wptlon = bs.traf.actwp.lon[idx]
                stack.stack(f'DELRTE {bs.traf.id[idx]}')
                stack.stack(f'ADDWPT {bs.traf.id[idx]} {wptlat},{wptlon}')

            self.wptdist_old = self.wptdist
            self.wptdist = dis
            
        return False

    def check_done(self,idx):

        fuel = self.get_rew_fuel(idx,coeff = -0.03/16.1458)
        noise = self.get_rew_noise(idx, coeff = -0.03)
        state = self.get_state(0)
        finish, d_f = self.get_rew_finish(idx,state)
        oob, d_oob = self.get_rew_outofbounds(idx)

        done = min(d_f + d_oob, 1)
    
        if done:
            self.fuel_array = np.append(self.fuel_array,fuel)
            self.noise_array = np.append(self.noise_array,noise)
            np.savetxt('output/resolution_experiments/fuel_'+model_name+'.csv',self.fuel_array)
            np.savetxt('output/resolution_experiments/noise_'+model_name+'.csv',self.noise_array)
            bs.traf.delete(idx)

    def get_state(self,idx):
        brg, dist = geo.kwikqdrdist(circle_lat, circle_lon, bs.traf.lat[idx], bs.traf.lon[idx])

        x = np.sin(np.radians(brg))*dist*nm2km / circlerad
        y = np.cos(np.radians(brg))*dist*nm2km / circlerad

        return [x,y]

    def get_latlon_state(self,state):
        distance = np.sqrt(state[0]**2 + state[1]**2)*circlerad
        bearing = math.atan2(state[0],state[1])

        lat = self.get_new_latitude(bearing,np.deg2rad(circle_lat),distance)
        lon = self.get_new_longitude(bearing,np.deg2rad(circle_lon),np.deg2rad(circle_lat),lat,distance)   

        return np.rad2deg(lat), np.rad2deg(lon)

    def do_action(self,action,idx):
        acid = bs.traf.id[idx]

        action = action[0]
        distance = max(math.sqrt(action[0]**2 + action[1]**2)*max_action,2)
        bearing = math.atan2(action[0],action[1])

        ac_lat = np.deg2rad(bs.traf.lat[idx])
        ac_lon = np.deg2rad(bs.traf.lon[idx])

        new_lat = self.get_new_latitude(bearing,ac_lat,distance)
        new_lon = self.get_new_longitude(bearing,ac_lon,ac_lat,new_lat,distance)

        self.action_required = False
        self.new_wpt_set = True

        if not self.first:
            stack.stack(f'DELRTE {acid}')

        stack.stack(f'ADDWPT {acid} {np.rad2deg(new_lat)},{np.rad2deg(new_lon)}')

    def get_new_latitude(self,bearing,lat,radius):
        R = Rearth/1000.
        return math.asin( math.sin(lat)*math.cos(radius/R) +\
               math.cos(lat)*math.sin(radius/R)*math.cos(bearing))
        
    def get_new_longitude(self,bearing,lon,lat1,lat2,radius):
        R   = Rearth/1000.
        return lon + math.atan2(math.sin(bearing)*math.sin(radius/R)*\
                     math.cos(lat1),math.cos(radius/R)-math.sin(lat1)*math.sin(lat2))

    def get_reward(self,idx,state,state_):
        fuel = self.get_rew_fuel(idx, coeff = -0.03/16.1458) 
        noise = self.get_rew_noise(idx, coeff = -0.03) 
        finish, d_f = self.get_rew_finish(idx,state)
        oob, d_oob = self.get_rew_outofbounds(idx)

        self.fuel = np.append(self.fuel,fuel)
        self.noise = np.append(self.noise,noise)

        fc.fuelconsumption.fuelconsumed[idx] = 0
        noisepol.noisepollution.noise[idx] = 0
        
        done = min(d_f, 1)
        reward = fuel+noise+finish+oob

        return reward, done
    
    def get_rew_fuel(self, idx, coeff = -0.005):
        fuel = fc.fuelconsumption.fuelconsumed[idx]
        reward = coeff * fuel
                
        return reward
    
    def get_rew_noise(self,idx, coeff = -1):
        noise = noisepol.noisepollution.noise[idx]
        reward = coeff * noise

        return reward

    def get_rew_finish(self, idx, state, coeff = 0):
        lat, lon = self.get_latlon_state(state)

        lat_ = bs.traf.lat[idx]
        lon_ = bs.traf.lon[idx]

        if areafilter.checkIntersect('SINK', lat, lon, lat_, lon_):
            return 5, 1

        if areafilter.checkIntersect('RESTRICT', lat, lon, lat_, lon_):
            return -1, 1

        else:
            return 0, 0

    def get_rew_outofbounds(self, idx, coeff = 0):
        dis_origin = fn.haversine(circle_lat, circle_lon, bs.traf.lat[idx], bs.traf.lon[idx])
        if dis_origin > circlerad*1.10:
            return coeff, 1
        else:
            return 0, 0

    def log(self):
        np.savetxt(model_name+'_reward.csv', self.rewards)
