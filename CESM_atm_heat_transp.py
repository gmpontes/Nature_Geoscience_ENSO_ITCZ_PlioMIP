import matplotlib.pyplot as plt
import numpy as np 
import glob
import xarray as xr
import math
from mpl_toolkits.basemap import Basemap
import matplotlib
import matplotlib.gridspec as gridspec
import scipy.io as sio
import scipy.interpolate as inter
import scipy
from matplotlib.font_manager import FontProperties
from scipy import signal
import cmaps
from climlab import constants as const
plt.ion()

def inferred_heat_transport( energy_in, lat_deg ):
    '''Returns the inferred heat transport (in PW) by integrating the net energy imbalance from pole to pole.'''
    from scipy import integrate
    from climlab import constants as const
    lat_rad = np.deg2rad( lat_deg )
    return ( 1E-15 * 2 * np.math.pi * const.a**2 * 
            integrate.cumtrapz( np.cos(lat_rad)*energy_in,
            x=lat_rad, initial=0. ) )

ens=5
sim=4

cHT1=np.zeros((12,96,ens))
pHT1=np.zeros((12,96,ens))
cHT2=np.zeros((12,96,ens))
pHT2=np.zeros((12,96,ens))

cFatmIN1=np.zeros((12,96,144,ens))
pFatmIN1=np.zeros((12,96,144,ens))
cFatmIN2=np.zeros((12,96,144,ens))
pFatmIN2=np.zeros((12,96,144,ens))

for s in range(0,sim):
	if s==0:
		sn='F1850pliomip'
	elif s==1:
		sn='F_sst_sic_plio'
	elif s==3:
		sn='FE280'
	elif s==2:
		sn='FEoi400'

	z1=sorted(glob.glob('/aurora/wk3/CESM_gabriel/'+sn+'*/atm/hist/*VT*'))

	for e in range(0,ens):
		data1 = xr.open_dataset(z1[e],decode_times=False)

		VT,x = clim_anom(data1['FLNT'][12:,:,:].values)
		lat=data1['lat'].values
		lon=data1['lon'].values

		if s==0:
			cVT1[:,:,:,e]=Fatmin
		elif s==1:
			pVT1[:,:,:,e]=Fatmin					
		elif s==2:
			cVT2[:,:,:,e]=Fatmin
		elif s==3:
			pVT2[:,:,:,e]=Fatmin

long,latg=np.meshgrid(lon,lat)

cFatmIN2=cFatmIN2.mean(axis=0).mean(axis=2)
pFatmIN2=pFatmIN2.mean(axis=0).mean(axis=2)

cHT1=xr.DataArray(cHT1,[('time',np.arange(1,13)),('lat', lat),('lon',lon),('ens',np.arange(1,6))])
pHT1=xr.DataArray(pHT1,[('time',np.arange(1,13)),('lat', lat),('lon',lon),('ens',np.arange(1,6))])
cHT2=xr.DataArray(cHT2,[('time',np.arange(1,13)),('lat', lat),('lon',lon),('ens',np.arange(1,6))])
pHT2=xr.DataArray(pHT2,[('time',np.arange(1,13)),('lat', lat),('lon',lon),('ens',np.arange(1,6))])

cHT1djf=cHT1.roll(time=1)[:3,:,:].mean('time')
pHT1djf=pHT1.roll(time=1)[:3,:,:].mean('time')
cHT2djf=cHT2.roll(time=1)[:3,:,:].mean('time')
pHT2djf=pHT2.roll(time=1)[:3,:,:].mean('time')

d1=pHT1-cHT1
d2=pHT2-cHT2

f=12
fig = plt.figure(2,dpi=200)
ax = fig.add_subplot(1, 1, 1)
# Move left y-axis and bottim x-axis to centre, passing through (0,0)
ax.spines['left'].set_position('center')
ax.spines['bottom'].set_position('center')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
plt.plot(lat,movingaverage(d1.mean(axis=1),5),color='magenta',linewidth=2)
plt.plot(lat,movingaverage(d2.mean(axis=1),5),color='red',linewidth=2)
plt.ylim([-0.2,0.2])
plt.xlim([-90,90])
plt.xticks(np.arange(-90,91,30),['90S','60S','30S','EQ','30N','60N','90N'],fontsize=f)
plt.yticks(np.arange(-0.2,0.21,0.05),fontsize=f)
