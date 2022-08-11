import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import glob
import xarray as xr
import math
from mpl_toolkits.basemap import Basemap
import matplotlib
import matplotlib.gridspec as gridspec
import scipy.interpolate as inter
from matplotlib.font_manager import FontProperties
import cmaps
import scipy.stats as ss
plt.ion()

############## Thermocline
z1=sorted(glob.glob('/clima/db2/PLIOMIP2/thetao/*E280*'))
z2=sorted(glob.glob('/clima/db2/PLIOMIP2/thetao/*Eoi400*'))

hEpi=np.zeros((len(z1)+1));hWpi=np.zeros((len(z1)+1))
hEplio=np.zeros((len(z1)+1));hWplio=np.zeros((len(z1)+1))

for m in range(0,len(z1)+1):
	
	if m==0:
		data1 = xr.open_dataset(z1[m],decode_times=False)
		data2 = xr.open_dataset(z2[m],decode_times=False)
		thetaEpi=data1['thetao'].sel(plev=slice(0,1e3),lon=slice(210,270),lat=slice(-5,5)).mean('lat').mean('lon').mean('time')
		thetaWpi=data1['thetao'].sel(plev=slice(0,1e3),lon=slice(160,210),lat=slice(-5,5)).mean('lat').mean('lon').mean('time')
		thetaEplio=data2['thetao'].sel(plev=slice(0,1e3),lon=slice(210,270),lat=slice(-5,5)).mean('lat').mean('lon').mean('time')
		thetaWplio=data2['thetao'].sel(plev=slice(0,1e3),lon=slice(160,210),lat=slice(-5,5)).mean('lat').mean('lon').mean('time')
	elif m==1:
		data1 = xr.open_dataset(z1[m],decode_times=False)
		data2 = xr.open_dataset(z2[m],decode_times=False)
		lat=data1['nav_lat'][:,0].values
		lon=data1['nav_lon'][160,:].values
		indi0=findnearest(lon,160)
		indi1=findnearest(lon,210-360);indi2=findnearest(lon,270-360)
		indj1=findnearest(lat,-5);indj2=findnearest(lat,5)
		thetaEpi=data1['thetao'][:,:,indj1:indj2,indi1:indi2].sel(olevel=slice(0,1000)).mean('y').mean('x').mean('time_counter').squeeze()
		thetaWpi=data1['thetao'][:,:,indj1:indj2,indi0:indi1].sel(olevel=slice(0,1000)).mean('y').mean('x').mean('time_counter').squeeze()
		thetaEplio=data2['thetao'][:,:,indj1:indj2,indi1:indi2].sel(olevel=slice(0,1000)).mean('y').mean('x').mean('time_counter').squeeze()
		thetaWplio=data2['thetao'][:,:,indj1:indj2,indi0:indi1].sel(olevel=slice(0,1000)).mean('y').mean('x').mean('time_counter').squeeze()
	elif m==2:
		data1 = xr.open_dataset(z1[m],decode_times=False)
		data2 = xr.open_dataset(z2[m],decode_times=False)
		lat=data1['TLAT'][:,0]
		lon=data1['TLONG'][0,:]
		indi0=findnearest(lon.values,160)
		indi1=findnearest(lon.values,210);indi2=findnearest(lon.values,270)
		indj1=findnearest(lat.values,-5);indj2=findnearest(lat.values,5)
		thetaEpi=data1['thetao'][:,:,indj1:indj2,indi1:indi2].sel(z_t=slice(0,1e5)).mean('y').mean('x').mean('time').squeeze()
		thetaWpi=data1['thetao'][:,:,indj1:indj2,indi0:indi1].sel(z_t=slice(0,1e5)).mean('y').mean('x').mean('time').squeeze()
		thetaEplio=data2['thetao'][:,:,indj1:indj2,indi1:indi2].sel(z_t=slice(0,1e5)).mean('y').mean('x').mean('time').squeeze()
		thetaWplio=data2['thetao'][:,:,indj1:indj2,indi0:indi1].sel(z_t=slice(0,1e5)).mean('y').mean('x').mean('time').squeeze()
	elif m==3:
		z1=sorted(glob.glob('/clima/db2/PLIOMIP2/GISS-E2-1-G/thetao*pi*'))
		z2=sorted(glob.glob('/clima/db2/PLIOMIP2/GISS-E2-1-G/thetao*eoi400*'))
		for f in range(0,len(z1)):
			datapi = xr.open_dataset(z1[f],decode_times=False)
			dataplio = xr.open_dataset(z2[f],decode_times=False)	
			if f==0:
				thetapi=datapi['thetao']
				thetaplio=dataplio['thetao']
			else:
				thetapi=xr.concat((thetapi,datapi['thetao']),dim='time')	
				thetaplio=xr.concat((thetaplio,dataplio['thetao']),dim='time')	
		thetaEpi=thetapi.sel(lev=slice(0,1e3),lon=slice(210,270),lat=slice(-5,5)).mean('lat').mean('lon').mean('time')
		thetaWpi=thetapi.sel(lev=slice(0,1e3),lon=slice(160,210),lat=slice(-5,5)).mean('lat').mean('lon').mean('time')
		thetaEplio=thetaplio.sel(lev=slice(0,1e3),lon=slice(210,270),lat=slice(-5,5)).mean('lat').mean('lon').mean('time')
		thetaWplio=thetaplio.sel(lev=slice(0,1e3),lon=slice(160,210),lat=slice(-5,5)).mean('lat').mean('lon').mean('time')

	lev=['plev','olevel','z_t','lev']
	zi=np.arange(thetaEpi[lev[m]].min(),thetaEpi[lev[m]].max(),1)
	
	f=inter.CubicSpline(thetaEpi[lev[m]],thetaEpi)
	spline=f(zi)
	spline=xr.DataArray(spline,[('lev', zi)])
	x=spline.differentiate(coord='lev')
	b=np.where(x<=x.min()*.5)
	hEpi[m]=np.sum((zi[b]*x[b]))/np.sum(x[b])

	f=inter.CubicSpline(thetaWpi[lev[m]],thetaWpi)
	spline=f(zi)
	spline=xr.DataArray(spline,[('lev', zi)])
	x=spline.differentiate(coord='lev')
	b=np.where(x<=x.min()*.5)
	hWpi[m]=np.sum((zi[b]*x[b]))/np.sum(x[b])

	####### plio
	f=inter.CubicSpline(thetaEplio[lev[m]],thetaEplio)
	spline=f(zi)
	spline=xr.DataArray(spline,[('lev', zi)])
	x=spline.differentiate(coord='lev')
	b=np.where(x<=x.min()*.5)
	hEplio[m]=np.sum((zi[b]*x[b]))/np.sum(x[b])

	f=inter.CubicSpline(thetaWplio[lev[m]],thetaWplio)
	spline=f(zi)
	spline=xr.DataArray(spline,[('lev', zi)])
	x=spline.differentiate(coord='lev')
	b=np.where(x<=x.min()*.5)
	hWplio[m]=np.sum((zi[b]*x[b]))/np.sum(x[b])

Spi=hEpi-hWpi
Splio=hEplio-hWplio
dSp2_2=((Splio-Spi)*100)/Spi
modelsp2_2=['COSMOS','EC-EARTH3.3','CCSM4-2deg','GISS-E2-1-G']

modelsp2_1=['CCSM4-1deg','CESM1.2','CESM2','IPSL6','Miroc4m']
dSp2_1=np.array([6,.7,-18,-12,-7.2])

modelsp2=['CCSM4-2deg','CCSM4-1deg','CESM1.2',
	'CESM2','COSMOS','EC-EARTH3.3','GISS-E2-1-G',
	'IPSL6','Miroc4m']

dSp2=np.concatenate((np.array([dSp2_2[2]]),dSp2_1[:3],dSp2_2[:2],np.array([dSp2_2[-1]]),dSp2_1[3:]))

# ####P1
z1=sorted(glob.glob('/polar/wk1/pliomip1/oceantemp/timmean/*pi*'))
z2=sorted(glob.glob('/polar/wk1/pliomip1/oceantemp/timmean/*plio*'))
z1=z1[1:];z2=z2[1:]

hEpi=np.zeros((len(z1)));hWpi=np.zeros((len(z1)))
hEplio=np.zeros((len(z1)));hWplio=np.zeros((len(z1)))

for m in range(0,len(z1)):
	
	if m==0:
		data1 = xr.open_dataset(z1[m],decode_times=False)
		data2 = xr.open_dataset(z2[m],decode_times=False)
		thetaEpi=data1['THO'].sel(lev=slice(0,1e3),lon=slice(210,270),lat=slice(-5,5)).mean('lat').mean('lon').squeeze()
		thetaWpi=data1['THO'].sel(lev=slice(0,1e3),lon=slice(160,210),lat=slice(-5,5)).mean('lat').mean('lon').squeeze()
		thetaEplio=data2['THO'].sel(lev=slice(0,1e3),lon=slice(210,270),lat=slice(-5,5)).mean('lat').mean('lon').squeeze()
		thetaWplio=data2['THO'].sel(lev=slice(0,1e3),lon=slice(160,210),lat=slice(-5,5)).mean('lat').mean('lon').squeeze()
	elif m==1:
		data1 = xr.open_dataset(z1[m],decode_times=False)
		data2 = xr.open_dataset(z2[m],decode_times=False)
		thetaEpi=data1['temp_ym_dpth'].sel(depth_1=slice(0,1e3),longitude=slice(210,270),latitude=slice(-5,5)).mean('latitude').mean('longitude').squeeze()
		thetaWpi=data1['temp_ym_dpth'].sel(depth_1=slice(0,1e3),longitude=slice(160,210),latitude=slice(-5,5)).mean('latitude').mean('longitude').squeeze()
		thetaEplio=data2['temp_ym_dpth'].sel(depth_1=slice(0,1e3),longitude=slice(210,270),latitude=slice(-5,5)).mean('latitude').mean('longitude').squeeze()
		thetaWplio=data2['temp_ym_dpth'].sel(depth_1=slice(0,1e3),longitude=slice(160,210),latitude=slice(-5,5)).mean('latitude').mean('longitude').squeeze()
	elif m==2:
		data1 = xr.open_dataset(z1[m],decode_times=False)
		data2 = xr.open_dataset(z2[m],decode_times=False)
		thetaEpi=data1['to'].sel(depth=slice(0,1e3),lon=slice(210,270),lat=slice(-5,5)).mean('lat').mean('lon').squeeze()
		thetaWpi=data1['to'].sel(depth=slice(0,1e3),lon=slice(160,210),lat=slice(-5,5)).mean('lat').mean('lon').squeeze()
		thetaEplio=data2['to'].sel(depth=slice(0,1e3),lon=slice(210,270),lat=slice(-5,5)).mean('lat').mean('lon').squeeze()
		thetaWplio=data2['to'].sel(depth=slice(0,1e3),lon=slice(160,210),lat=slice(-5,5)).mean('lat').mean('lon').squeeze()
	elif m==3:
		data1 = xr.open_dataset(z1[m],decode_times=False)
		data2 = xr.open_dataset(z2[m],decode_times=False)
		thetaEpi=data1['to'].sel(lev=slice(1e3,0),lon=slice(210,270),lat=slice(-5,5)).mean('lat').mean('lon').squeeze()
		lev=np.flipud(thetaEpi['lev'].values)
		thetaEpi=np.flipud(thetaEpi.values)
		thetaEpi=xr.DataArray(thetaEpi,[('lev',lev)])
		thetaWpi=data1['to'].sel(lev=slice(1e3,0),lon=slice(160,210),lat=slice(-5,5)).mean('lat').mean('lon').squeeze()
		thetaWpi=np.flipud(thetaWpi.values)
		thetaWpi=xr.DataArray(thetaWpi,[('lev',lev)])
		thetaEplio=data2['to'].sel(lev=slice(1e3,0),lon=slice(210,270),lat=slice(-5,5)).mean('lat').mean('lon').squeeze()
		thetaEplio=np.flipud(thetaEplio.values)
		thetaEplio=xr.DataArray(thetaEplio,[('lev',lev)])
		thetaWplio=data2['to'].sel(lev=slice(1e3,0),lon=slice(160,210),lat=slice(-5,5)).mean('lat').mean('lon').squeeze()
		thetaWplio=np.flipud(thetaWplio.values)
		thetaWplio=xr.DataArray(thetaWplio,[('lev',lev)])
	elif m==4:
		data1 = xr.open_dataset(z1[m],decode_times=False)
		data2 = xr.open_dataset(z2[m],decode_times=False)
		lat=data1['lat'][:,0].values
		lon=data1['lon'][0,:].values
		indi0=findnearest(lon,160)
		indi1=findnearest(lon,210);indi2=findnearest(lon,270)
		indj1=findnearest(lat,-5);indj2=findnearest(lat,5)
		thetaEpi=data1['thetao'][:,:,indj1:indj2,indi1:indi2].sel(lev=slice(0,1000)).mean('y').mean('x').squeeze()
		thetaWpi=data1['thetao'][:,:,indj1:indj2,indi0:indi1].sel(lev=slice(0,1000)).mean('y').mean('x').squeeze()
		thetaEplio=data2['thetao'][:,:,indj1:indj2,indi1:indi2].sel(lev=slice(0,1000)).mean('y').mean('x').squeeze()
		thetaWplio=data2['thetao'][:,:,indj1:indj2,indi0:indi1].sel(lev=slice(0,1000)).mean('y').mean('x').squeeze()

	lev=['lev','depth_1','depth','lev','lev']
	zi=np.arange(thetaEpi[lev[m]].min(),thetaEpi[lev[m]].max(),1)
	
	f=inter.CubicSpline(thetaEpi[lev[m]],thetaEpi)
	spline=f(zi)
	spline=xr.DataArray(spline,[('lev', zi)])
	x=spline.differentiate(coord='lev')
	b=np.where(x<=x.min()*.5)
	hEpi[m]=np.sum((zi[b]*x[b]))/np.sum(x[b])

	f=inter.CubicSpline(thetaWpi[lev[m]],thetaWpi)
	spline=f(zi)
	spline=xr.DataArray(spline,[('lev', zi)])
	x=spline.differentiate(coord='lev')
	b=np.where(x<=x.min()*.5)
	hWpi[m]=np.sum((zi[b]*x[b]))/np.sum(x[b])

	####### plio
	f=inter.CubicSpline(thetaEplio[lev[m]],thetaEplio)
	spline=f(zi)
	spline=xr.DataArray(spline,[('lev', zi)])
	x=spline.differentiate(coord='lev')
	b=np.where(x<=x.min()*.5)
	hEplio[m]=np.sum((zi[b]*x[b]))/np.sum(x[b])

	f=inter.CubicSpline(thetaWplio[lev[m]],thetaWplio)
	spline=f(zi)
	spline=xr.DataArray(spline,[('lev', zi)])
	x=spline.differentiate(coord='lev')
	b=np.where(x<=x.min()*.5)
	hWplio[m]=np.sum((zi[b]*x[b]))/np.sum(x[b])

Spi=hEpi-hWpi
Splio=hEplio-hWplio
dSp1=((Splio-Spi)*100)/Spi

modelsp1=['CCSM4','COSMOS','HadCM3','MIROC4m',
	'MRI','NorESM-L']

dSccsm4p1=np.array([10])
dSp1=np.concatenate((dSccsm4p1,dSp1))

denso=np.concatenate((P1dstdENSO,P2dstdENSO))
dS=np.concatenate((dSp1,dSp2))

print(ss.pearsonr(denso,dS))
print(ss.spearmanr(denso,dS))

f=np.polyfit(dS,denso,1)
font=25
plt.figure(21,figsize=(9.5,7.5))
plt.scatter(dSp2,P2dstdENSO,s=90,c='red',alpha=.7,)
plt.scatter(dSp1,P1dstdENSO,marker='s',s=90,c='magenta',alpha=.7)
plt.plot([0,0],[-100,100],linewidth=0.3,color='dimgrey')
plt.plot([-100,100],[0,0],linewidth=0.3,color='dimgrey')
plt.ylim(-80,20)
plt.xlim(-40,40)
plt.plot(np.arange(-25,26,5),np.polyval(f,np.arange(-25,26,5)),'--',
	linewidth=4,color='k',alpha=.7)

########################################################
########### cross-section

z1=sorted(glob.glob('/clima/db2/PLIOMIP2/thetao/*E280*'))
z2=sorted(glob.glob('/clima/db2/PLIOMIP2/thetao/*Eoi400*'))

levi=np.arange(0,500,1)
loni=np.arange(140,301,1)

Tpi=np.zeros((len(z1)+5,len(levi),len(loni)))
Tplio=np.zeros((len(z1)+5,len(levi),len(loni)))

for m in range(0,len(z1)+5):
	
	if m==0:
		data1 = xr.open_dataset(z1[m],decode_times=False)
		data2 = xr.open_dataset(z2[m],decode_times=False)
		thetapi=data1['thetao'].sel(plev=slice(0,5e2),lon=slice(140,300),lat=slice(-5,5)).mean('lat').mean('time').values
		thetaplio=data2['thetao'].sel(plev=slice(0,5e2),lon=slice(140,300),lat=slice(-5,5)).mean('lat').mean('time').values
		lev=data1['plev'].sel(plev=slice(0,5e2)).values
		lon=data1['lon'].sel(lon=slice(140,300)).values
	elif m==1:
		data1 = xr.open_dataset(z1[m],decode_times=False)
		data2 = xr.open_dataset(z2[m],decode_times=False)
		lat=data1['nav_lat'][:,0].values
		lon=data1['nav_lon'][160,:].values
		thetapi=data1['thetao'].mean('time_counter').sel(olevel=slice(0,5e2))
		thetaplio=data2['thetao'].mean('time_counter').sel(olevel=slice(0,5e2))
		lev=data1['olevel'].sel(olevel=slice(0,5e2)).values
		x1=np.zeros((thetapi.shape))
		x2=np.zeros((thetapi.shape))
		for l in range(0,len(lev)):
			lonp,x1[l,:,:]=shift_pcf(lon,thetapi[l,:,:].values)
			lonp,x2[l,:,:]=shift_pcf(lon,thetaplio[l,:,:].values)
		indi0=findnearest(lonp,140)
		indi1=findnearest(lonp,290)
		indj1=findnearest(lat,-5);indj2=findnearest(lat,5)
		thetapi=x1[:,indj1:indj2,indi0:indi1].mean(axis=1).squeeze()
		thetaplio=x2[:,indj1:indj2,indi0:indi1].mean(axis=1).squeeze()		
		lon=lonp[indi0:indi1]
	elif m==2:
		data1 = xr.open_dataset(z1[m],decode_times=False)
		data2 = xr.open_dataset(z2[m],decode_times=False)
		lat=data1['TLAT'][:,0]
		lon=data1['TLONG'][0,:]
		indi0=findnearest(lon.values,140)
		indi1=findnearest(lon.values,300)
		indj1=findnearest(lat.values,-5);indj2=findnearest(lat.values,5)
		thetapi=data1['thetao'][:,:,indj1:indj2,indi0:indi1].sel(z_t=slice(0,5e4)).mean('y').mean('time').squeeze().values
		thetaplio=data2['thetao'][:,:,indj1:indj2,indi0:indi1].sel(z_t=slice(0,5e4)).mean('y').mean('time').squeeze().values
		lon=lon[indi0:indi1]
		lev=data1['z_t'].sel(z_t=slice(0,5e4)).values/1e2
	elif m==3:
		z1=sorted(glob.glob('/clima/db2/PLIOMIP2/GISS-E2-1-G/thetao*pi*'))
		z2=sorted(glob.glob('/clima/db2/PLIOMIP2/GISS-E2-1-G/thetao*eoi400*'))
		for f in range(0,len(z1)):
			datapi = xr.open_dataset(z1[f],decode_times=False)
			dataplio = xr.open_dataset(z2[f],decode_times=False)	
			if f==0:
				thetapi=datapi['thetao']
				thetaplio=dataplio['thetao']
			else:
				thetapi=xr.concat((thetapi,datapi['thetao']),dim='time')	
				thetaplio=xr.concat((thetaplio,dataplio['thetao']),dim='time')	
		thetapi=thetapi.sel(lev=slice(0,5e2),lon=slice(140,300),lat=slice(-5,5)).mean('lat').mean('time').values
		thetaplio=thetaplio.sel(lev=slice(0,5e2),lon=slice(140,300),lat=slice(-5,5)).mean('lat').mean('time').values
		lev=datapi['lev'].sel(lev=slice(0,5e2)).values
		lon=datapi['lon'].sel(lon=slice(140,300)).values
	elif m==4 or m==5:
		if m==4:
			z3=sorted(glob.glob('/clima/db2/PLIOMIP2/NCAR/b40*TEMP*eq*'))
		else:
			z3=sorted(glob.glob('/clima/db2/PLIOMIP2/NCAR/b.e12*TEMP*eq*'))
		
		for f in range(0,len(z3)):
			data = xr.open_dataset(z3[f],decode_times=False)

			lat=data['TLAT'][:,0]
			lon=data['TLONG'][0,:]
			indi0=findnearest(lon.values,140)
			indi1=findnearest(lon.values,300)
			indj1=findnearest(lat.values,-5);indj2=findnearest(lat.values,5)

			if f==0:
				thetaplio=data['TEMP'][:,:,indj1:indj2,indi0:indi1].sel(z_t=slice(0,5e4)).mean('nlat').mean('time').values
			else:
				thetapi=data['TEMP'][:,:,indj1:indj2,indi0:indi1].sel(z_t=slice(0,5e4)).mean('nlat').mean('time').values
		lon=lon[indi0:indi1]
		lev=data['z_t'].sel(z_t=slice(0,5e4)).values/1e2
	elif m==6:
		z1=sorted(glob.glob('/clima/db2/PLIOMIP2/CESM2/thetao*1deg*'))
		data1 = xr.open_dataset(z1[1],decode_times=False)
		data2 = xr.open_dataset(z1[0],decode_times=False)

		thetapi=data1['thetao'].sel(lev=slice(0,5e4),lon=slice(140,300),lat=slice(-5,5)).mean('lat').mean('time').values
		thetaplio=data2['thetao'].sel(lev=slice(0,5e4),lon=slice(140,300),lat=slice(-5,5)).mean('lat').mean('time').values
		lev=data1['lev'].sel(lev=slice(0,5e4)).values*1e-2
		lon=data1['lon'].sel(lon=slice(140,300)).values
	elif m==7:
		z1='/aurora/wk3/pliomip2/IPSL-CM6A-LR/thetao_Omon_IPSL-CM6A-LR_piControl_r1i1p1f1_gn_295001-304912.nc'
		z2='/aurora/wk3/pliomip2/IPSL-CM6A-LR/thetao_Omon_IPSL-CM6A-LR_midPliocene-eoi400_r1i1p1f1_gn_195001-204912.nc'
		data1 = xr.open_dataset(z1,decode_times=False)
		data2 = xr.open_dataset(z2,decode_times=False)
		lat=data1['nav_lat'][:,0].values
		lon=data1['nav_lon'][160,:].values
		lev=data1['olevel'].sel(olevel=slice(0,5e2)).values
		thetapi=data1['thetao'].mean('time').sel(olevel=slice(0,5e2))
		thetaplio=data2['thetao'].mean('time').sel(olevel=slice(0,5e2))
		x1=np.zeros((thetapi.shape))
		x2=np.zeros((thetapi.shape))
		for l in range(0,len(lev)):
			lonp,x1[l,:,:]=shift_pcf(lon,thetapi[l,:,:].values)
			lonp,x2[l,:,:]=shift_pcf(lon,thetaplio[l,:,:].values)
		indi0=findnearest(lonp,140)
		indi1=findnearest(lonp,290)
		indj1=findnearest(lat,-5);indj2=findnearest(lat,5)
		thetapi=x1[:,indj1:indj2,indi0:indi1].mean(axis=1).squeeze()
		thetaplio=x2[:,indj1:indj2,indi0:indi1].mean(axis=1).squeeze()		
		lon=lonp[indi0:indi1]

	Tpi[m,:,:]=interpola(lev,lon,thetapi,levi,loni)
	Tplio[m,:,:]=interpola(lev,lon,thetaplio,levi,loni)

Tpip2=Tpi
Tpliop2=Tplio
dTp2=Tplio-Tpi

lonig,levig=np.meshgrid(loni,levi)
sig,levsig,lonsig=same_sign(dTp2[:,::20,::5],loni[::5],levi[::20],7)

plt.figure(22,figsize=(6,3))
plt.contourf(lonig,-levig,dTp2.mean(axis=0),np.arange(-2.4,2.5,0.2),
	cmap=cmaps.BlueWhiteOrangeRed,extend='both')
plt.plot(lonsig,-levsig,'x',color='black',markersize=1)
plt.xlim(155,260)
plt.ylim(-300,0)

### ocean stratification
modelp2=['COSMOS','EC-EARTH3.3','CCSM4-2deg','GISS',
	'CCSM4-1deg','CESM1.2','CESM2','IPSL6']
Tpip2=xr.DataArray(Tpip2,[('model',modelp2),('lev', levi),('lon',loni)])
Tpliop2=xr.DataArray(Tpliop2,[('model',modelp2),('lev', levi),('lon',loni)])

dTpidzp2=Tpip2.sel(lev=slice(0,75),lon=slice(210,270)).mean('lev').mean('lon')-Tpip2.sel(lev=100,lon=slice(210,270)).mean('lon')
dTpliodzp2=Tpliop2.sel(lev=slice(0,75),lon=slice(210,270)).mean('lev').mean('lon')-Tpliop2.sel(lev=100,lon=slice(210,270)).mean('lon')
dTdzp2=dTpliodzp2-dTpidzp2

## Miroc4m-p2
z1=sorted(glob.glob('/clima/db2/PLIOMIP2/MIROC4m/thetao/*'))
data1 = xr.open_dataset(z1[0],decode_times=False)
dTdzpi=data1['thetao'].sel(lon=slice(150,220),lat=slice(-5,5),depth=slice(0,75)).mean()-data1['thetao'].sel(lon=slice(150,220),lat=slice(-5,5),depth=110).mean()
data2 = xr.open_dataset(z1[1],decode_times=False)
dTdzplio=data2['thetao'].sel(lon=slice(150,220),lat=slice(-5,5),depth=slice(0,75)).mean()-data2['thetao'].sel(lon=slice(150,220),lat=slice(-5,5),depth=110).mean()
mir=dTdzplio-dTdzpi

modelp2=['COSMOS','EC-EARTH3.3','CCSM4-2deg','GISS',
	'CCSM4-1deg','CESM1.2','CESM2','IPSL6','Miroc4m']

dTdzchangep2=np.concatenate((dTdzp2,[mir.values]))

####
####P1
z1=sorted(glob.glob('/polar/wk1/pliomip1/oceantemp/timmean/*pi*'))
z2=sorted(glob.glob('/polar/wk1/pliomip1/oceantemp/timmean/*plio*'))

levi=np.arange(0,500,1)
loni=np.arange(140,301,1)

Tpi=np.zeros((len(z1),len(levi),len(loni)))
Tplio=np.zeros((len(z1),len(levi),len(loni)))

for m in range(0,len(z1)):
	data1 = xr.open_dataset(z1[m],decode_times=False)
	data2 = xr.open_dataset(z2[m],decode_times=False)
	
	if m==1:
		thetapi=data1['THO'].sel(lev=slice(0,5e2),lon=slice(140,300),lat=slice(-5,5)).mean('lat').squeeze().values
		thetaplio=data2['THO'].sel(lev=slice(0,5e2),lon=slice(140,300),lat=slice(-5,5)).mean('lat').squeeze().values
		lon=data1['lon'].sel(lon=slice(140,300))
		lev=data1['lev'].sel(lev=slice(0,5e2))		
	elif m==2:
		thetapi=data1['temp_ym_dpth'].sel(depth_1=slice(0,5e2),longitude=slice(140,300),latitude=slice(-5,5)).mean('latitude').squeeze().values
		thetaplio=data2['temp_ym_dpth'].sel(depth_1=slice(0,5e2),longitude=slice(140,300),latitude=slice(-5,5)).mean('latitude').squeeze().values
		lon=data1['longitude'].sel(longitude=slice(140,300))
		lev=data1['depth_1'].sel(depth_1=slice(0,5e2))		
	elif m==3:
		thetapi=data1['to'].sel(depth=slice(0,5e2),lon=slice(140,300),lat=slice(-5,5)).mean('lat').squeeze().values
		thetaplio=data2['to'].sel(depth=slice(0,5e2),lon=slice(140,300),lat=slice(-5,5)).mean('lat').squeeze().values
		lon=data1['lon'].sel(lon=slice(140,300))
		lev=data1['depth'].sel(depth=slice(0,5e2))		
	elif m==4:
		thetapi=data1['to'].sel(lev=slice(5e2,0),lon=slice(140,300),lat=slice(-5,5)).mean('lat').squeeze().values
		lev=np.flipud(data1['lev'].sel(lev=slice(5e2,0)).values)
		thetapi=np.flipud(thetapi)
		thetaplio=data2['to'].sel(lev=slice(5e2,0),lon=slice(140,300),lat=slice(-5,5)).mean('lat').squeeze().values
		thetaplio=np.flipud(thetaplio)
		lon=data1['lon'].sel(lon=slice(140,300))
	elif m==5:
		lat=data1['lat'][:,0].values
		lon=data1['lon'][0,:].values
		lev=data1['lev'].sel(lev=slice(0,5e2)).values
		thetapi=data1['thetao'].mean('time').sel(lev=slice(0,5e2))
		thetaplio=data2['thetao'].mean('time').sel(lev=slice(0,5e2))
		x1=np.zeros((thetapi.shape))
		x2=np.zeros((thetapi.shape))
		for l in range(0,len(lev)):
			lonp,x1[l,:,:]=shift_pcf(lon,thetapi[l,:,:].values)
			lonp,x2[l,:,:]=shift_pcf(lon,thetaplio[l,:,:].values)
		indi0=findnearest(lonp,140)
		indi1=findnearest(lonp,300)
		indj1=findnearest(lat,-5);indj2=findnearest(lat,5)
		thetapi=x1[:,indj1:indj2,indi0:indi1].mean(axis=1).squeeze()-273
		thetaplio=x2[:,indj1:indj2,indi0:indi1].mean(axis=1).squeeze()-273	
		lon=lonp[indi0:indi1]
	elif m==0:
		lat=data1['TLAT'][:,0]
		lon=data1['TLONG'][0,:]
		indi0=findnearest(lon.values,140)
		indi1=findnearest(lon.values,300)
		indj1=findnearest(lat.values,-5);indj2=findnearest(lat.values,5)
		thetaplio=data2['TEMP'][:,:,indj1:indj2,indi0:indi1].sel(z_t=slice(0,5e4)).mean('y_2').mean('time').values
		thetapi=data1['TEMP'][:,:,indj1:indj2,indi0:indi1].sel(z_t=slice(0,5e4)).mean('y_2').mean('time').values
		lon=lon[indi0:indi1]
		lev=data1['z_t'].sel(z_t=slice(0,5e4)).values/1e2

	Tpi[m,:,:]=interpola(lev,lon,thetapi,levi,loni)
	Tplio[m,:,:]=interpola(lev,lon,thetaplio,levi,loni)

Tpip1=Tpi
Tpliop1=Tplio
dTp1=Tpliop1-Tpip1
modelp1=['CCSM4','COSMOS','HadCM3','MIROC4m','MRI-CGCM2.3','NorESM-L']


lonig,levig=np.meshgrid(loni,levi)

sig,levsig,lonsig=same_sign(dT[:,::20,::5],loni[::5],levi[::20],6)

plt.figure(22,figsize=(6,3))
plt.contourf(lonig,-levig,dT.mean(axis=0),np.arange(-1.8,1.9,0.1),
	cmap=cmaps.BlueWhiteOrangeRed,extend='both')
plt.plot(lonsig,-levsig,'x',color='black',markersize=1)
plt.xlim(155,260)
plt.ylim(-300,0)

### ocean strafication
Tpip1=xr.DataArray(Tpip1,[('model',modelp1),('lev', levi),('lon',loni)])
Tpliop1=xr.DataArray(Tpliop1,[('model',modelp1),('lev', levi),('lon',loni)])

dTpidzp1=Tpip1.sel(lev=slice(0,75),lon=slice(210,270)).mean('lev').mean('lon')-Tpip1.sel(lev=100,lon=slice(210,270)).mean('lon')
dTpliodzp1=Tpliop1.sel(lev=slice(0,75),lon=slice(210,270)).mean('lev').mean('lon')-Tpliop1.sel(lev=100,lon=slice(210,270)).mean('lon')
dTdzchangep1=dTpliodzp1-dTpidzp1

###
dTdz=np.concatenate((dTdzchangep1,dTdzchangep2))
denso=np.concatenate((P1dstdnino3,P2dstdnino3))

print(ss.spearmanr(dTdz,denso))

f=np.polyfit(dTdz,denso,1)

plt.figure(8,figsize=(9.5,7.5))
plt.scatter(dTdzchangep1,P1dstdnino3,s=110,marker='s',c='magenta',alpha=.7)
plt.scatter(dTdzchangep2,P2dstdnino3,s=110,c='red',alpha=.7)
plt.plot([0,0],[-100,100],linewidth=0.3,color='dimgrey')
plt.plot([-100,100],[0,0],linewidth=0.3,color='dimgrey')
plt.ylim(-80,20)
plt.xlim(-1.5,2.5)
plt.xticks(np.arange(-1,3,1),fontsize=font)
plt.yticks(np.arange(-80,21,20),fontsize=font)
plt.xlabel('Change in ocean stratification [K]',fontsize=font)
plt.ylabel('Change in ENSO amplitude [%]',fontsize=font)
