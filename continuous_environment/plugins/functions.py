import math as m
from bluesky.tools.aero import ft, nm, fpm, Rearth, kts

def haversine(lat1,lon1,lat2,lon2):
    R = Rearth

    dLat = m.radians(lat2 - lat1)
    dLon = m.radians(lon2 - lon1)
    lat1 = m.radians(lat1)
    lat2 = m.radians(lat2)
 
    a = m.sin(dLat/2)**2 + m.cos(lat1)*m.cos(lat2)*m.sin(dLon/2)**2
    c = 2*m.asin(m.sqrt(a))

    return (R * c)/1000.   