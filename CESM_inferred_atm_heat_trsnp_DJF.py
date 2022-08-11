import matplotlib.pyplot as plt
import numpy as np 
import glob
import xarray as xr
import math
import matplotlib
import matplotlib.gridspec as gridspec
import scipy.io as sio
import scipy.interpolate as inter
import scipy
from matplotlib.font_manager import FontProperties
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

cHT1=np.zeros((96,ens))
pHT1=np.zeros((96,ens))
cHT2=np.zeros((96,ens))
pHT2=np.zeros((96,ens))

time=np.arange(1,13,1)
for s in range(0,sim):
	if s==0:
		sn='F1850pliomip'
	elif s==1:
		sn='F_sst_sic_plio'
	elif s==3:
		sn='FE280'
	elif s==2:
		sn='FEoi400'

	z1=sorted(glob.glob('/aurora/wk3/CESM_gabriel/'+sn+'*/atm/hist/*FLNT*'))
	z2=sorted(glob.glob('/aurora/wk3/CESM_gabriel/'+sn+'*/atm/hist/*FSNT*'))
	z3=sorted(glob.glob('/aurora/wk3/CESM_gabriel/'+sn+'*/atm/hist/*FLNS*'))
	z4=sorted(glob.glob('/aurora/wk3/CESM_gabriel/'+sn+'*/atm/hist/*FSNS*'))
	z5=sorted(glob.glob('/aurora/wk3/CESM_gabriel/'+sn+'*/atm/hist/*PRECSC*'))
	z6=sorted(glob.glob('/aurora/wk3/CESM_gabriel/'+sn+'*/atm/hist/*PRECSL*'))
	z7=sorted(glob.glob('/aurora/wk3/CESM_gabriel/'+sn+'*/atm/hist/*QFLX*'))
	z8=sorted(glob.glob('/aurora/wk3/CESM_gabriel/'+sn+'*/atm/hist/*SHFLX*'))
	z9=sorted(glob.glob('/aurora/wk3/CESM_gabriel/'+sn+'*/atm/hist/*LHFLX*'))
	z10=sorted(glob.glob('/aurora/wk3/CESM_gabriel/'+sn+'*/atm/hist/*PRECC*'))
	z11=sorted(glob.glob('/aurora/wk3/CESM_gabriel/'+sn+'*/atm/hist/*PRECL*'))

	for e in range(0,ens):
		data1 = xr.open_dataset(z1[e],decode_times=False)
		data2 = xr.open_dataset(z2[e],decode_times=False)
		data3 = xr.open_dataset(z3[e],decode_times=False)
		data4 = xr.open_dataset(z4[e],decode_times=False)
		data5 = xr.open_dataset(z5[e],decode_times=False)
		data6 = xr.open_dataset(z6[e],decode_times=False)
		data8 = xr.open_dataset(z8[e],decode_times=False)
		data9 = xr.open_dataset(z9[e],decode_times=False)

		lat=data1['lat'].values
		lon=data1['lon'].values
		
		OLR,x = clim_anom(data1['FLNT'][12:,:,:])
		OLR=xr.DataArray(OLR,[('time',time),('lat', lat),('lon',lon)])
		OLR=OLR.roll(time=1)[:3,:,:].mean('lon').mean('time').values
		
		ASR,x = clim_anom(data2['FSNT'][12:,:,:])
		ASR=xr.DataArray(ASR,[('time',time),('lat', lat),('lon',lon)])
		ASR=ASR.roll(time=1)[:3,:,:].mean('lon').mean('time').values
		
		LWsfc,x = clim_anom(data3['FLNS'][12:,:,:])  # net longwave radiation at surface
		LWsfc=xr.DataArray(LWsfc,[('time',time),('lat', lat),('lon',lon)])
		LWsfc=LWsfc.roll(time=1)[:3,:,:].mean('lon').mean('time').values

		SWsfc,x = clim_anom(-data4['FSNS'][12:,:,:])  # net shortwave radiation at surface
		SWsfc=xr.DataArray(SWsfc,[('time',time),('lat', lat),('lon',lon)])
		SWsfc=SWsfc.roll(time=1)[:3,:,:].mean('lon').mean('time').values
		
		LHF,x = clim_anom(data9['LHFLX'][12:,:,:])  # latent heat flux (evaporation)
		LHF=xr.DataArray(LHF,[('time',time),('lat', lat),('lon',lon)])
		LHF=LHF.roll(time=1)[:3,:,:].mean('lon').mean('time').values

		SHF,x = clim_anom(data8['SHFLX'][12:,:,:])  # latent heat flux (evaporation)
		SHF=xr.DataArray(SHF,[('time',time),('lat', lat),('lon',lon)])
		SHF=SHF.roll(time=1)[:3,:,:].mean('lon').mean('time').values

		PRECSC,x = clim_anom(data5['PRECSC'][12:,:,:])  # latent heat flux (evaporation)
		PRECSC=xr.DataArray(PRECSC,[('time',time),('lat', lat),('lon',lon)])
		PRECSC=PRECSC.roll(time=1)[:3,:,:].mean('lon').mean('time').values

		PRECSL,x = clim_anom(data6['PRECSL'][12:,:,:])  # latent heat flux (evaporation)
		PRECSL=xr.DataArray(PRECSL,[('time',time),('lat', lat),('lon',lon)])
		PRECSL=PRECSL.roll(time=1)[:3,:,:].mean('lon').mean('time').values

		SnowFlux =  (PRECSC+PRECSL)*const.rho_w*const.Lhfus
		Rtoa = ASR - OLR  # net downwelling radiation
		SurfaceRadiation = LWsfc + SWsfc  # net upward radiation from surface
		SurfaceHeatFlux = SurfaceRadiation + LHF + SHF + SnowFlux  # net upward surface heat flux
		Fatmin = Rtoa + SurfaceHeatFlux  # net heat flux in to atmosphere

		if s==0:
			cHT1[:,e] = inferred_heat_transport(Fatmin, lat)
		elif s==1:
			pHT1[:,e] = inferred_heat_transport(Fatmin, lat)
		elif s==3:
			cHT2[:,e] = inferred_heat_transport(Fatmin, lat)
		elif s==2:
			pHT2[:,e] = inferred_heat_transport(Fatmin, lat)

dHT1=pHT1-cHT1
dHT2=pHT2-cHT2

f=14
fig = plt.figure(1,figsize=(10,6))
ax = fig.add_subplot(1, 1, 1)
# Move left y-axis and bottim x-axis to centre, passing through (0,0)
ax.spines['left'].set_position('center')
ax.spines['bottom'].set_position('center')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
plt.plot(lat,dHT1.mean(axis=1),color='m',linewidth=1)
plt.plot(lat,dHT2.mean(axis=1),color='r',linewidth=1)
plt.xlim([-90,90]);plt.ylim([-.3,.3])
plt.yticks(np.arange(-0.3,0.31,0.1),['','','','','','',''],fontsize=f)
plt.xticks(np.arange(-90,91,30),['','','','','','',''],fontsize=f)
plt.fill_between(lat,dHT1.mean(axis=1)-dHT1.std(axis=1),dHT1.mean(axis=1)+dHT1.std(axis=1),
	color='m',alpha=.1,edgecolor='none')
plt.fill_between(lat,dHT2.mean(axis=1)-dHT2.std(axis=1),dHT2.mean(axis=1)+dHT2.std(axis=1),
	color='r',alpha=.1,edgecolor='none')

