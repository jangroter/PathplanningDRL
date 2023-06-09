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

import plugins.SAC_n_layers.sac_agent as sac
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

# '\\' for windows, '/' for linux or mac
model_name = 'E_3_1024-x1_v2' 
#poly_arc(self.circlelat,self.circlelon,20,30,-30)
    

# dir_symbol = '\\'
# model_path = 'output' + dir_symbol + model_name + dir_symbol + 'model'

model_path = 'output' + '/' + model_name + '/' + 'model'

# Make folder for logfiles
path = Path(model_path)
path.mkdir(parents=True, exist_ok=True)

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def init_plugin():
    
    experiment_drl_test = Experiment_drl_test()
    
    # Configuration parameters
    config = {
        # The name of your plugin
        'plugin_name':     'EXPERIMENT_DRL_TEST',

        # The type of this plugin. For now, only simulation plugins are possible.
        'plugin_type':     'sim',
        }

    # init_plugin() should always return a configuration dict.
    return config

class Experiment_drl_test(core.Entity):  
    def __init__(self):
        super().__init__()

        self.agent = sac.SAC(action_size,state_size,path,test=True)
        self.agent.load_models()

        self.rewards = np.array([])
        self.state_size = state_size
        self.action_size = action_size

        self.index = 0
        self.waypoints = pd.read_pickle('latlon_paths.pkl')
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
        self.nactions[-n:] = 0

        self.state[-n:] = list(np.zeros(self.state_size))
        self.action[-n:] = list(np.zeros(self.action_size))

    @core.timed_function(name='experiment_drl_test', dt=timestep)
    def update(self):
        fc.fuelconsumption.update(timestep)
        noisepol.noisepollution.update(timestep)
        
        idx = 0
        
        self.check_ac()
        
        if not self.first:
            self.check_done(idx)

        if self.action_required:
            state = self.get_state(idx)
            action = self.agent.step(state)
            self.do_action(action,idx)
            
            self.state[idx] = state
            self.action[idx] = action

            self.nactions[idx] += 1

            self.first = False

    def check_ac(self):
        if bs.traf.ntraf == 0 and self.index < self.indexmax:
            lat0 = self.waypoints['lat'].iloc[self.index][0]
            lon0 = self.waypoints['lon'].iloc[self.index][0]
            heading,_ = geo.kwikqdrdist(lat0,lon0,circle_lat,circle_lon)

            # stack.stack(f'CRE kl001 a320 {lat0} {lon0} {heading} {15000} {250}')
            traf.cre('KL001','a320',lat0,lon0,heading,15000*ft,250*kts)
            stack.stack('ADDWPTMODE KL001 FLYOVER')

            self.action_required = True
            self.index += 1
            self.first = True
        else: 
            if self.check_past_wpt(0):
                self.action_required = True
            else:
                self.action_required = False
        
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
        reward = finish+oob+fuel

        if done:
            # terminate episode, but dont let model know about it if out-of-bounds,
            # this limits cheesing of episodes by quickly flying out of bounds/
            if d_oob == 1:
                done = 0
            state = self.get_state(idx)
            self.totreward[idx] += reward
            self.rewards = np.append(self.rewards, self.totreward[idx])
 
            self.action_required = False

            self.fuel_array = np.append(self.fuel_array,fuel)
            self.noise_array = np.append(self.noise_array,noise)
            np.savetxt('fuel'+model_name+'.csv',self.fuel_array)
            np.savetxt('noise'+model_name+'.csv',self.noise_array)

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
        dis = self.get_rew_distance(state,state_)
        step = self.get_rew_step(coeff = 0)
        # Basevalue fuel = -0.03/17, exp ~ -0.0075, -0.015, -0.03, -0.06, -0.12
        fuel = self.get_rew_fuel(idx, coeff = -0.03/17.) #Fuel ~ Noise * 17 
        noise = self.get_rew_noise(idx, coeff = -0.03) #Coeff ~ -0.002
        finish, d_f = self.get_rew_finish(idx,state)
        oob, d_oob = self.get_rew_outofbounds(idx)

        self.fuel = np.append(self.fuel,fuel)
        self.noise = np.append(self.noise,noise)
        
        done = min(d_f, 1)
        reward = dis+step+fuel+noise+finish+oob

        return reward, done

    def get_rew_distance(self,state,state_, coeff = 0.005):
        old_distance = np.sqrt(state[0]**2 + state[1]**2)*circlerad
        new_distance = np.sqrt(state_[0]**2 + state_[1]**2)*circlerad

        d_dis = old_distance - new_distance

        """ c1 """
        # d_dis = min(d_dis,0)
        # return d_dis * coeff

        """ c2 """
        # return d_dis * coeff

        """ c3 """
        return 0

    def get_rew_step(self, coeff = -0.01):
        return coeff

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

    def get_rew_outofbounds(self, idx, coeff = -1):
        dis_origin = fn.haversine(circle_lat, circle_lon, bs.traf.lat[idx], bs.traf.lon[idx])
        if dis_origin > circlerad*1.10:
            return coeff, 1
        else:
            return 0, 0

    def log(self):
        np.savetxt(model_name+'_reward.csv', self.rewards)
        self.agent.save_models()