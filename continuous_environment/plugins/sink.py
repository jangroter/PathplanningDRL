from bluesky import core, stack, traf  #, settings, navdb, sim, scr, tools
from bluesky.tools import geo, aero, areafilter, plotter
from bluesky.tools.aero import ft, nm, fpm, Rearth, kts
import bluesky as bs
import numpy as np
import area
import random
import math



def poly_arc(lat_c,lon_c,radius,lowerbound,upperbound):
    # Input data is latctr,lonctr,radius[nm]
    # Convert circle into polyline list

    # Circle parameters
    Rearth = 6371000.0             # radius of the Earth [m]
    numPoints = 36                 # number of straight line segments that make up the circrle

    # Inputs
    lat0 = lat_c             # latitude of the center of the circle [deg]
    lon0 = lon_c            # longitude of the center of the circle [deg]
    Rcircle = radius * 1852.0  # radius of circle [m]

    # Compute flat Earth correction at the center of the experiment circle
    coslatinv = 1.0 / np.cos(np.deg2rad(lat0))

    lower = np.deg2rad(lowerbound)
    upper = np.deg2rad(upperbound)
    # compute the x and y coordinates of the circle
    angles = np.linspace(lower,upper,numPoints)   # ,endpoint=True) # [rad]

    # Calculate the circle coordinates in lat/lon degrees.
    # Use flat-earth approximation to convert from cartesian to lat/lon.
    latCircle = lat0 + np.rad2deg(Rcircle * np.sin(angles) / Rearth)  # [deg]
    lonCircle = lon0 + np.rad2deg(Rcircle * np.cos(angles) * coslatinv / Rearth)  # [deg]

    # make the data array in the format needed to plot circle
    Coordinates = np.empty(2 * numPoints, dtype=np.float32)  # Create empty array
    Coordinates[0::2] = latCircle  # Fill array lat0,lon0,lat1,lon1....
    Coordinates[1::2] = lonCircle

    command = 'POLYLINE SINK'
    for i in range(0,len(latCircle)):
        command += ' '+str(latCircle[i])+' '
        command += str(lonCircle[i])
    stack.stack(command)
 
    stack.stack(f'POLYLINE RESTRICT {latCircle[0]} {lonCircle[0]} {lat_c} {lon_c} {latCircle[-1]} {lonCircle[-1]}')
    stack.stack('COLOR RESTRICT red')

    