"""
Converting latitude, declination and hour angle into radians
"""

import pandas as pd
import re
import numpy as np


"""
Loading the antenna layout file and the observation parameters
"""
enu_data = pd.read_excel ('C:/Users/insig_000/Desktop/ENU coordinates of KAT-7.xlsx', index_col = 0) ## Put file path here.
### Cleaning the data a bit ###
enu_data.columns = ['x','y','z'] # Renaming columns for convenience
enu_data['x'] = enu_data['x'].apply(lambda x: x.replace('m', '').replace('(x1)', '')).astype('float')
enu_data['y'] = enu_data['y'].apply(lambda x: x.replace('m', '').replace('(y1)', '')).astype('float')
enu_data['z'] = enu_data['z'].apply(lambda x: x.replace('m', '').replace('(z1)', '')).astype('float')
enu_data.head()
print(enu_data)
  
enu_data1 = pd.read_excel ('C:/Users/insig_000/Desktop/Additional Info.xlsx', index_col = 0) ## Put file path here.
print(enu_data1)


def latitude():
    L = (enu_data1.iloc[0,0])
    L = re.findall("[-0-9.eE]+", L)
    L = float(L[0]), float (L[1]), float (L[2])
    L_deg = (L[0] + (L[1]/60) + (L[2]/3600))  
    L_rad = (np.pi/180)*L_deg
    return L_rad


def declination():
    Dec = (enu_data1.iloc[3,0])
    Dec = re.findall("[-0-9.eE]+", Dec)
    Dec = float(Dec[0]), float (Dec[1]), float (Dec[2])
    Dec_deg = (Dec[0] + (Dec[1]/60) + (Dec[2]/3600)) 
    Dec_rad = (np.pi/180)*Dec_deg
    return Dec_rad


def hour_angle_range(nsteps=100):                             
    H_start = (enu_data1.iloc[1,0])
    H_start = re.findall("[-0-9.eE]+", H_start)
    H_start = float(H_start[0])*np.pi/12 

    H_stop = (enu_data1.iloc[2,0])
    H_stop = re.findall("[-0-9.eE]+", H_stop)
    H_stop = float(H_stop[0])*np.pi/12 
                                         
    H_rad = np.linspace(H_start, H_stop, nsteps)           
    return H_rad

L_rad = latitude()
#print(L_rad)
Dec_rad = declination()
H_rad = hour_angle_range()
