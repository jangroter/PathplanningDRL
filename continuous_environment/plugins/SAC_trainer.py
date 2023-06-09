""" 
Main plugin used for training the SAC models
"""

from ctypes.wintypes import WPARAM
from bluesky import core, stack, traf, tools, settings 
from bluesky.tools.aero import ft, nm, fpm, Rearth, kts
from bluesky.tools import geo, aero, areafilter, plotter
from bluesky.traffic import Route

import bluesky as bs
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')

from pathlib import Path

import plugins.SAC.sac_agent as sac
from plugins.source import Source
import plugins.functions as fn
import plugins.fuelconsumption as fc 
import plugins.noisepollution as noisepol

# Setting initial variables
timestep = 5
state_size = 2
action_size = 2

nm2km = nm/1000

circlerad = 150. * nm2km
max_action = 25. * nm2km

max_episode_length = 15

circle_lat = 52.3322
circle_lon = 4.75

model_name = 'E_3_1024-x1_v2' 

# Set the noise to fuel ratio
# Basevalue fuels = 0.25, 0.5, 1.0, 2.0, 4.0
fuel_noise_ratio = 1.0
# Get the fuel cost, -0.03 is the default scalar used for noise, 16.1458 is for normalization 
fuel_coeff = fuel_noise_ratio*-0.03/16.1458

# '\\' for windows, '/' for linux or mac
model_path = 'output' + '/' + model_name + '/' + 'model'

# Make folder for logfiles
path = Path(model_path)
path.mkdir(parents=True, exist_ok=True)

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def init_plugin():
    
    experiment_drl = Experiment_drl()
    
    # Configuration parameters
    config = {
        # The name of your plugin
        'plugin_name':     'EXPERIMENT_DRL',

        # The type of this plugin. For now, only simulation plugins are possible.
        'plugin_type':     'sim',
        }

    # init_plugin() should always return a configuration dict.
    return config

class Experiment_drl(core.Entity):  
    def __init__(self):
        super().__init__()

        self.agent = sac.SAC(action_size,state_size,path)

        self.rewards = np.array([])
        self.state_size = state_size
        self.action_size = action_size

        self.first = True

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
            self.nactions = np.array([])  # Number of actions this AC has done

            self.state = []    # Latest state information
            self.action = []    # Latest selected actions

    def create(self, n=1):
        super().create(n)
        self.totreward[-n:] = 0
        self.nactions[-n:] = 0

        self.state[-n:] = list(np.zeros(self.state_size))
        self.action[-n:] = list(np.zeros(self.action_size))

    @core.timed_function(name='experiment_main', dt=timestep)
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

            if not self.first:
                reward, done = self.get_reward(idx,self.state[idx],state)
                self.totreward[idx] += reward
                self.agent.store_transition(self.state[idx],self.action[idx][0],reward,state,done)
                self.agent.train()
            
            self.state[idx] = state
            self.action[idx] = action

            self.nactions[idx] += 1

            self.first = False

        if len(self.rewards) % 50 == 0 and self.print == True:
            self.print = False
            print(np.mean(self.rewards[-2000:]))
            print(f'Fuel: {np.mean(self.fuel[-2000:])}')
            print(f'Noise: {np.mean(self.noise[-2000:])}')

            fig, ax = plt.subplots()
            ax.plot(self.agent.qf1_lossarr, label='qf1')
            ax.plot(self.agent.qf2_lossarr, label='qf2')
            fig.savefig('qloss.png')
            plt.close(fig)

            fig, ax = plt.subplots()
            ax.plot(self.rewards, label='total reward', alpha = 0.5)
            ax.plot(moving_average(self.rewards,500))
            fig.savefig('reward.png')
            plt.close(fig)

            self.log()

    def check_ac(self):
        """
        Check at the beginning of each timestep to see if an action is required or a new aircraft should be created
        """
        if bs.traf.ntraf == 0:
            self.source.create_ac()

            self.first = True
            self.action_required = True

            self.nac += 1
        else: 
            if self.check_past_wpt(0):
                self.action_required = True
            else:
                self.action_required = False
        
    def check_past_wpt(self, idx):
        """
        Function that checks if the aircraft has passed the previous set waypoint
        """
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
        """
        Checks if the aircraft is done with the episode
        """

        fuel = self.get_rew_fuel(idx,coeff=fuel_coeff)
        finish, d_f = self.get_rew_finish(idx,self.state[idx])
        oob, d_oob = self.get_rew_outofbounds(idx)

        done = min(d_f + d_oob, 1)
        reward = finish+oob+fuel

        if done:
            # terminate episode, but dont let model know about it is out-of-bounds,
            # this limits cheesing of episodes by quickly flying out of bounds.
            if d_oob == 1:
                done = 0
            state = self.get_state(idx)
            self.totreward[idx] += reward
            self.rewards = np.append(self.rewards, self.totreward[idx])

            self.agent.store_transition(self.state[idx],self.action[idx][0],reward,state,done)
            self.agent.train()
 
            self.action_required = False
            self.print = True

            bs.traf.delete(idx)
        
        elif self.nactions[idx] > max_episode_length:
            self.rewards = np.append(self.rewards, self.totreward[idx])
            self.print = True
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
        fuel = self.get_rew_fuel(idx, coeff = fuel_coeff)
        noise = self.get_rew_noise(idx, coeff = -0.03)
        finish, d_f = self.get_rew_finish(idx,state)

        self.fuel = np.append(self.fuel,fuel)
        self.noise = np.append(self.noise,noise)

        fc.fuelconsumption.fuelconsumed[idx] = 0
        noisepol.noisepollution.noise[idx] = 0
        
        done = min(d_f, 1)
        reward = fuel+noise+finish

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
        self.agent.save_models()