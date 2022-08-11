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
import scipy.stats as stats
import cmaps
from eofs.standard import Eof
plt.ion()

##############################################
################## PLIOMIP2
z2=sorted(glob.glob('/aurora/wk3/pliomip2/tos/*E280*'))
z2=[z2[1]]+z2[3:]
z3=sorted(glob.glob('/aurora/wk3/pliomip2/tos/*Eoi400*'))
z3=[z3[1]]+[z3[3]]+z3[4:]

datax=xr.open_dataset(z2[3],decode_times=False)
lat=datax['lat'].sel(lat=slice(-40,30))
lon=datax['lon']
long,latg=np.meshgrid(lon,lat)
time=np.arange(0,1200)

SSTplio=np.zeros((len(z2)+3,1200,len(lat),360))
SSTpi=np.zeros((len(z2)+3,1200,len(lat),360))

for m in range(0,len(z2)+3):
	if m==6:
		fnplio='/aurora/wk3/pliomip2/MIROC4m/tos/MIROC4m_Eoi400_Omon_tos.nc'
		fnpi='/aurora/wk3/pliomip2/MIROC4m/tos/MIROC4m_E280_Omon_tos.nc'
		dataplio = xr.open_dataset(fnplio,decode_times=False)
		datapi = xr.open_dataset(fnpi,decode_times=False)
		x,sstpliox=clim_anom(dataplio['tos'].sel(lat=slice(-40,30)).values)
		x,sstpix=clim_anom(datapi['tos'].sel(lat=slice(-40,30)).values)
		lonx=datapi['lon']
		latx=datapi['lat'].sel(lat=slice(-40,30))
		time=datapi['time']
		for t in range(0,len(time)):
			sstplio[t,:,:] = interpola(latx,lonx, sstpliox[t,:,:],lat,lon)
			sstpi[t,:,:] = interpola(latx,lonx, sstpix[t,:,:],lat,lon)
	elif m==0:
		datapi = xr.open_dataset(z2[m],decode_times=False)
		dataplio = xr.open_dataset(z3[m],decode_times=False)
		latx=datapi['lat'][:,0]
		lonx=datapi['lon'][0,:]
		sstpliox=xr.DataArray(dataplio['tos'],coords=[time,latx,lonx],dims=['time','lat','lon'])
		sstpix=xr.DataArray(datapi['tos'],coords=[time,latx,lonx],dims=['time','lat','lon'])
		x,sstpliox2=clim_anom(sstpliox.sel(lat=slice(-40,30)).values)
		x,sstpix2=clim_anom(sstpix.sel(lat=slice(-40,30)).values)
		latx=sstpix['lat'].sel(lat=slice(-40,30))
		sstplio=np.zeros((len(time),len(lat),len(lon)))
		sstpi=np.zeros((len(time),len(lat),len(lon)))
		for t in range(0,len(time)):
			sstplio[t,:,:] = interpola(latx,lonx, sstpliox2[t,:,:],lat,lon)
			sstpi[t,:,:] = interpola(latx,lonx, sstpix2[t,:,:],lat,lon)
	elif m==3:
		zx1=sorted(glob.glob('/aurora/wk3/pliomip2/HadCM3/e280/SST/*'))
		zx2=sorted(glob.glob('/aurora/wk3/pliomip2/HadCM3/eoi400/SST/*'))
		for f in range(0,len(zx1)):
			data1 = xr.open_dataset(zx1[f],decode_times=True)
			data2 = xr.open_dataset(zx2[f],decode_times=False)
			if f==0:
				sstpix=data1['temp'].sel(latitude=slice(-40,30)).squeeze()
				sstpliox=data2['temp'].sel(latitude=slice(-40,30)).squeeze()
				time1x=data1['t']
				time2x=data2['t']
			else:
				sstpix=np.concatenate((sstpix,data1['temp'].sel(latitude=slice(-40,30)).squeeze()),axis=0)			
				sstpliox=np.concatenate((sstpliox,data2['temp'].sel(latitude=slice(-40,30)).squeeze()),axis=0)
				time1x=xr.concat((time1x,data1['t']),dim='t')	
				time2x=xr.concat((time2x,data2['t']),dim='t')	
		x,sstpliox=clim_anom(sstpliox)
		x,sstpix=clim_anom(sstpix)
		latx=data1['latitude'].sel(latitude=slice(-40,30)).values
		lonx=data1['longitude'].values
		sstplio=np.zeros((1200,len(lat),len(lon)))
		sstpi=np.zeros((1200,len(lat),len(lon)))
		for t in range(0,1200):
			sstplio[t,:,:] = interpola(latx,lonx, sstpliox[t,:,:],lat,lon)
			sstpi[t,:,:] = interpola(latx,lonx, sstpix[t,:,:],lat,lon)
	elif m==11:
		zx1=sorted(glob.glob('/clima/db2/PLIOMIP2/CESM2/thetao*1deg*'))
		data1 = xr.open_dataset(zx1[1],decode_times=False)
		data2 = xr.open_dataset(zx1[0],decode_times=False)
		x,sstpi=clim_anom(data1['thetao'][12:,:,:].sel(lev=500,lat=slice(-40,30)).squeeze())
		x,sstplio=clim_anom(data2['thetao'].sel(lev=500,lat=slice(-40,30)).squeeze())
	elif m==12:
		zx1='/aurora/wk3/pliomip2/IPSL-CM6A-LR/thetao_Omon_IPSL-CM6A-LR_piControl_r1i1p1f1_gn_295001-304912.nc'
		zx2='/aurora/wk3/pliomip2/IPSL-CM6A-LR/thetao_Omon_IPSL-CM6A-LR_midPliocene-eoi400_r1i1p1f1_gn_195001-204912.nc'
		data1 = xr.open_dataset(zx1,decode_times=False)
		data2 = xr.open_dataset(zx2,decode_times=False)
		latx=data1['nav_lat'][:,0].values
		lonx1=data1['nav_lon'][160,:].values
		indj1=findnearest(latx,-40);indj2=findnearest(latx,30)
		x,sstpliox=clim_anom(data2['thetao'][:,0,indj1:indj2,:])
		x,sstpix=clim_anom(data1['thetao'][:,0,indj1:indj2,:])
		latx=latx[indj1:indj2]
		for t in range(0,1200):
			lonx,sstpix[t,:,:]=shift_pcf(lonx1,sstpix[t,:,:])
			lonx,sstpliox[t,:,:]=shift_pcf(lonx1,sstpliox[t,:,:])
		sstplio=np.zeros((1200,len(lat),len(lon)))
		sstpi=np.zeros((1200,len(lat),len(lon)))
		for t in range(0,1200):
			sstplio[t,:,:] = interpola(latx,lonx, sstpliox[t,:,:],lat,lon)
			sstpi[t,:,:] = interpola(latx,lonx, sstpix[t,:,:],lat,lon)
	elif m==13:
		z1=sorted(glob.glob('/clima/db2/PLIOMIP2/GISS-E2-1-G/thetao*pi*'))
		z2=sorted(glob.glob('/clima/db2/PLIOMIP2/GISS-E2-1-G/thetao*eoi400*'))
		for f in range(0,len(z1)):
			datapi = xr.open_dataset(z1[f],decode_times=False)
			dataplio = xr.open_dataset(z2[f],decode_times=False)	
			if f==0:
				sstpix=datapi['thetao'][:,0,:,:].sel(lat=slice(-40,30)).squeeze()
				sstpliox=dataplio['thetao'][:,0,:,:].sel(lat=slice(-40,30)).squeeze()
			else:
				sstpix=np.concatenate((sstpix,datapi['thetao'][:,0,:,:].sel(lat=slice(-40,30)).squeeze()),axis=0)	
				sstpliox=np.concatenate((sstpliox,dataplio['thetao'][:,0,:,:].sel(lat=slice(-40,30)).squeeze()),axis=0)	
		x,sstpliox=clim_anom(sstpliox)
		x,sstpix=clim_anom(sstpix)
		latx=datapi['lat'].sel(lat=slice(-40,30)).values
		lonx=datapi['lon'].values
		sstplio=np.zeros((1200,len(lat),len(lon)))
		sstpi=np.zeros((1200,len(lat),len(lon)))
		for t in range(0,1200):
			sstplio[t,:,:] = interpola(latx,lonx, sstpliox[t,:,:],lat,lon)
			sstpi[t,:,:] = interpola(latx,lonx, sstpix[t,:,:],lat,lon)
	else:
		datapi = xr.open_dataset(z2[m],decode_times=False)
		dataplio = xr.open_dataset(z3[m],decode_times=False)
		x,sstplio=clim_anom(dataplio['tos'].sel(lat=slice(-40,30)).squeeze())
		x,sstpi=clim_anom(datapi['tos'].sel(lat=slice(-40,30)).squeeze())

	SSTplio[m,:,:,:]=sstplio
	SSTpi[m,:,:,:]=sstpi


############################################################
cvarstd=np.nanstd(SSTpi,axis=1)
pvarstd=np.nanstd(SSTplio,axis=1)

# cvarstdm=np.nanmean(cvarstdi,axis=0)
dvarstd=pvarstd-cvarstd
dvarstdm=np.mean(dvarstd,axis=0)
dvarstdi=np.zeros((len(z2),len(np.arange(-30,31,2)),len(np.arange(0,360,2))))
for m in range(0,len(z2)):
	dvarstdi[m,:,:]=interpola(lat,lon,dvarstd[m,:,:],
		np.arange(-30,31,2),np.arange(0,360,2))
sig,latsig,lonsig=same_sign(dvarstdi,np.arange(0,360,2),
	np.arange(-30,31,2),9)

fig = plt.figure(1,figsize=(12,10),dpi=150)
mp = Basemap(projection='robin',lon_0=-180,resolution='c')
mp.fillcontinents(color='darkgrey',alpha=1)
mp.drawparallels(np.arange(-90.,120.,30.))
clim2=np.arange(-0.5,0.51,0.02)
d=mp.contourf(long,latg,dvarstdm,clim2,
	latlon=True,cmap=cmaps.BlWhRe,extend='both')
for x in range(len(lonsig)):
	mp.plot(lonsig[x],latsig[x],'x',latlon=True,
		color='black',markersize=.75)

cax2 = fig.add_axes([0.31, 0.1, 0.4, 0.02])
cbar2=fig.colorbar(d,cax2,ticks=np.arange(-0.5,0.5,0.25),
	label=r'$\Delta$ SST [$^{o}$C]',
	orientation='horizontal',ticklocation='bottom',extend='both')

################################################################
models=['CCSM4-UTRECHT','COSMOS','EC-EARTH3.3',
	'HadCM3','IPSLCM5A','IPSLCM5A2','MIROC4m','MRI-CGCM2.3',
	'NorESM-L','NorESM1-F','CCSM4-UofT','CESM2',
	'IPSLCM6ALR','GISS-E2-1-G']

SSTpia=xr.DataArray(SSTpi,coords=[models,time,lat,lon],
	dims=['model','time','lat','lon'])
SSTplioa=xr.DataArray(SSTplio,coords=[models,time,lat,lon],
	dims=['model','time','lat','lon'])

SSTpia2=xr.concat((SSTpia[0],SSTpia[10],SSTpia[-3],SSTpia[1:3],
	SSTpia[-1],SSTpia[3],SSTpia[-2],SSTpia[4:10]),dim='model')
SSTplioa2=xr.concat((SSTplioa[0],SSTplioa[10],SSTplioa[-3],SSTplioa[1:3],
	SSTplioa[-1],SSTplioa[3],SSTplioa[-2],SSTplioa[4:10]),dim='model')

models=['CCSM4-2deg','CCSM4-UofT','CESM2','COSMOS',
	'EC-EARTH3.3','GISS-E2-1-G','HadCM3','IPSLCM6ALR',
	'IPSLCM5A','IPSLCM5A2','MIROC4m','MRI-CGCM2.3',
	'NorESM-L','NorESM1-F']

P2nino3pi=SSTpia2.sel(lat=slice(-5,5),lon=slice(210,270)).mean('lat').mean('lon')
P2nino3plio=SSTplioa2.sel(lat=slice(-5,5),lon=slice(210,270)).mean('lat').mean('lon')
P2dstdnino3=((P2nino3plio.std('time')-P2nino3pi.std('time'))*100)/P2nino3pi.std('time')

models2=['CCSM4-2deg','CCSM4-UofT','CCSM4-1deg','CESM1.2',
	'CESM2','COSMOS',
	'EC-EARTH3.3','GISS-E2-1-G','HadCM3','IPSLCM6ALR',
	'IPSLCM5A','IPSLCM5A2','MIROC4m','MRI-CGCM2.3',
	'NorESM-L','NorESM1-F']

fig1 = plt.figure(1,figsize=(7,12),dpi=150)
x1=np.arange(5,17,1)
ax1 = fig1.add_axes([0.3,.65,.47,.27])
ax1.bar([1,2],P2dstdnino3[0:2],width=0.3,color='blue',alpha=0.7)
ax1.bar([3],-43.01,width=0.3,color='blue',alpha=0.7)
ax1.bar([4],-23.32,width=0.3,color='blue',alpha=0.7)
ax1.bar(x1,P2dstdnino3[2:],width=0.3,color='blue',alpha=0.7)
plt.xticks(np.arange(1,18,1),models2,rotation='vertical')
plt.plot([0,18],[0,0],color='k',linewidth=0.2)
plt.xlim([0,18])
plt.ylim([-70,10])

#################### CCSM4 1deg #########################
z1=sorted(glob.glob('/clima/db2/PLIOMIP2/NCAR/b40*TEMP*eq.*'))
datapi = xr.open_dataset(z1[1],decode_times=False)
dataplio = xr.open_dataset(z1[0],decode_times=False)

time=datapi['time']
lat=datapi['TLAT'][:,0].values
lon=datapi['TLONG'][0,:].values
xpi=xr.DataArray(datapi['TEMP'][:,0,:,:].values,[('time', time),
	('lat', lat),('lon', lon)])
xplio=xr.DataArray(dataplio['TEMP'][:,0,:,:].values,[('time', time),
	('lat', lat),('lon', lon)])

nino3pi=xpi.sel(lon=slice(210,270))
nino3plio=xplio.sel(lon=slice(210,270))

x,nino3pi=clim_anom(nino3pi)
x,nino3plio=clim_anom(nino3plio)

nino3piCCSM41deg=np.nanmean(np.nanmean(nino3pi,axis=1),axis=1) 
nino3plioCCSM41deg=np.nanmean(np.nanmean(nino3plio,axis=1),axis=1) 

#################### CESM1.2 #########################
z1=sorted(glob.glob('/clima/db2/PLIOMIP2/NCAR/b.e12*TEMP*eq.*'))
datapi = xr.open_dataset(z1[1],decode_times=False)
dataplio = xr.open_dataset(z1[0],decode_times=False)

time=datapi['time']
lat=datapi['TLAT'][:,0].values
lon=datapi['TLONG'][0,:].values
xpi=xr.DataArray(datapi['TEMP'][:,0,:,:].values,[('time', time),
	('lat', lat),('lon', lon)])
xplio=xr.DataArray(dataplio['TEMP'][:,0,:,:].values,[('time', time),
	('lat', lat),('lon', lon)])

nino3pi=xpi.sel(lon=slice(210,270))
nino3plio=xplio.sel(lon=slice(210,270))

x,nino3pi=clim_anom(nino3pi)
x,nino3plio=clim_anom(nino3plio)

nino3piCESM12=np.nanmean(np.nanmean(nino3pi,axis=1),axis=1) 
nino3plioCESM12=np.nanmean(np.nanmean(nino3plio,axis=1),axis=1) 

time=np.arange(0,1200,1)
Cpi=np.concatenate([nino3piCCSM41deg,nino3piCESM12]).reshape([2,1200])
Cplio=np.concatenate([nino3plioCCSM41deg,nino3plioCESM12]).reshape([2,1200])

Cpi=xr.DataArray(Cpi,[('model',['CCSM4-1deg','CESM1.2']),('time',time)])
Cplio=xr.DataArray(Cplio,[('model',['CCSM4-1deg','CESM1.2']),('time',time)])

P2nino3pi=xr.DataArray(P2nino3pi, 
	[('time', time), ('model', models)])
P2nino3plio=xr.DataArray(P2nino3plio, 
	[('time', time), ('model', models)])

ENSOpi=xr.concat((P2nino3pi[:,0:2],Cpi,P2nino3pi[:,2:]),dim='model')
ENSOplio=xr.concat((P2nino3plio[:,0:2],Cplio,P2nino3plio[:,2:]),dim='model')

ENSOpi=ENSOpi.to_dataset(name='ENSO_E280')
ENSOplio=ENSOplio.to_dataset(name='ENSO_Eoi400')

ENSO_P2=xr.merge([ENSOpi,ENSOplio])
ENSO_P2.to_netcdf('/home/gabriel/enso/ENSO_PlioMIP2.nc')

#### PMM change
P2PMMpi=SSTpia2.sel(lat=slice(-25,-15),lon=slice(250,260)).mean('lat').mean('lon')
P2PMMplio=SSTplioa2.sel(lat=slice(-25,-15),lon=slice(250,260)).mean('lat').mean('lon')

P2dstdPMM=((P2PMMplio.std('time')-P2PMMpi.std('time'))*100)/P2PMMpi.std('time')

fig1 = plt.figure(1,figsize=(7,12),dpi=150)
ax1 = fig1.add_axes([0.17,.65,.43,.27])
ax1.bar(x1,P2dstdPMM,width=0.3,color='royalblue',alpha=1)
plt.xticks(x1,models,rotation='vertical')
plt.plot([0,16],[0,0],color='k',linewidth=0.2)
plt.xlim([0,16])
plt.ylim([-30,10])

#### NPMM change
P2PMMpi=SSTpia2.sel(lat=slice(20,25),lon=slice(218,222)).mean('lat').mean('lon')
P2PMMplio=SSTplioa2.sel(lat=slice(20,25),lon=slice(218,222)).mean('lat').mean('lon')

PMMpi=P2PMMpi.to_dataset(name='PMM_E280')
PMMplio=P2PMMplio.to_dataset(name='PMM_Eoi400')

P2dstdPMM=((np.nanstd(P2PMMplio,axis=0)-np.nanstd(P2PMMpi,axis=0))*100)/np.nanstd(P2PMMpi,axis=0)

x1=np.arange(1,15,1)

fig1 = plt.figure(2,figsize=(7,12),dpi=150)
ax1 = fig1.add_axes([0.17,.65,.43,.27])
ax1.bar(x1,P2dstdPMM,width=0.3,color='royalblue',alpha=1)
plt.xticks(x1,models,rotation='vertical')
plt.plot([0,16],[0,0],color='k',linewidth=0.2)
plt.xlim([0,15])
######

SSTpip=SSTpia.sel(lat=slice(-40,20),lon=slice(140,300))
SSTpliop=SSTplioa.sel(lat=slice(-40,20),lon=slice(140,300))
long,latg=np.meshgrid(SSTpip['lon'],SSTpip['lat'])

P2PMMpi2=(P2PMMpi-P2PMMpi.mean('time'))/P2PMMpi.std('time')
P2PMMplio2=(P2PMMplio-P2PMMplio.mean('time'))/P2PMMplio.std('time')

regpi=np.zeros((len(SSTpip['model']),len(SSTpip['lat']),len(SSTpip['lon'])))
regplio=np.zeros((len(SSTpip['model']),len(SSTpip['lat']),len(SSTpip['lon'])))
for m in range(0,len(SSTpip['model'])):
	for i in range(0,len(SSTpip['lon'])):
		for j in range(0,len(SSTpip['lat'])):
			regpi[m,j,i]=stats.linregress(P2PMMpi2[:,m].values,SSTpip[:,j,i,m].values)[0]
			regplio[m,j,i]=stats.linregress(P2PMMplio2[:,m].values,SSTpliop[:,j,i,m].values)[0]


fig = plt.figure(10,figsize=(12,8),dpi=150)
gs = gridspec.GridSpec(4,4)
clim=np.arange(-.6,.7,0.1)
clim2=np.arange(-.3,.4,.1)
for m in range(0,len(SSTpip['model'])):
	ax1 = fig.add_subplot(gs[m])
	ax1 = Basemap(projection='mill',llcrnrlat=-40,urcrnrlat=20,\
	            llcrnrlon=140,urcrnrlon=280,resolution='c')
	ax1.drawparallels(np.arange(-20,20,10),linewidth=.3)
	ax1.drawmeridians(np.arange(160,280,40),linewidth=.3)
	d=ax1.contourf(long,latg,regpi[m,:,:],clim,
			latlon=True,cmap=cmaps.BlueWhiteOrangeRed,extend='both')
	plt.title(models[m])
gs.update(hspace=.5,wspace=.5)
cax2 = fig.add_axes([0.31, 0.3, 0.4, 0.02])
cbar2=fig.colorbar(d,cax2,ticks=np.arange(-.3,.4,.1),
	label=r'$\Delta$ SST [$^{o}$C]',
	orientation='horizontal',ticklocation='bottom',extend='both')

######## removing decadal period (11yr)
z1=sorted(glob.glob('/clima/db2/PLIOMIP2/NCAR/b.e12*TEMP*eq.*'))
datapi = xr.open_dataset(z1[1],decode_times=False)
dataplio = xr.open_dataset(z1[0],decode_times=False)
time=datapi['time']
lat=datapi['TLAT'][:,0].values
lon=datapi['TLONG'][0,:].values
sstpi=xr.DataArray(datapi['TEMP'][:,0,:,:].values,[('time', time),
	('lat', lat),('lon', lon)])
sstplio=xr.DataArray(dataplio['TEMP'][:,0,:,:].values,[('time', time),
	('lat', lat),('lon', lon)])
cesm12pi=sstpi.sel(lon=slice(210,270))
cesm12plio=sstplio.sel(lon=slice(210,270))
x,cesm12pi=clim_anom(cesm12pi)
x,cesm12plio=clim_anom(cesm12plio)
cesm12pi=np.nanmean(np.nanmean(cesm12pi,axis=1),axis=1) 
cesm12plio=np.nanmean(np.nanmean(cesm12plio,axis=1),axis=1) 

z1=sorted(glob.glob('/clima/db2/PLIOMIP2/NCAR/b40*TEMP*eq.*'))
datapi = xr.open_dataset(z1[1],decode_times=False)
dataplio = xr.open_dataset(z1[0],decode_times=False)
time=datapi['time']
lat=datapi['TLAT'][:,0].values
lon=datapi['TLONG'][0,:].values
sstpi=xr.DataArray(datapi['TEMP'][:,0,:,:].values,[('time', time),
	('lat', lat),('lon', lon)])
sstplio=xr.DataArray(dataplio['TEMP'][:,0,:,:].values,[('time', time),
	('lat', lat),('lon', lon)])
ccsm4pi=sstpi.sel(lon=slice(210,270))
ccsm4plio=sstplio.sel(lon=slice(210,270))
x,ccsm4pi=clim_anom(ccsm4pi)
x,ccsm4plio=clim_anom(ccsm4plio)
ccsm4pi=np.nanmean(np.nanmean(ccsm4pi,axis=1),axis=1) 
ccsm4plio=np.nanmean(np.nanmean(ccsm4plio,axis=1),axis=1) 

x=xr.DataArray([ccsm4pi,cesm12pi],[('model', ['CCSM4-1deg','CESM1.2']),('time', P2nino3pi['time'])])
P2nino3pi_2=xr.concat((P2nino3pi[:,0:2],x,P2nino3pi[:,2:4],P2nino3pi[:,5:]),dim='model')

x2=xr.DataArray([ccsm4plio,cesm12plio],[('model', ['CCSM4-1deg','CESM1.2']),('time', P2nino3pi['time'])])
P2nino3plio_2=np.concatenate((P2nino3plio[:,0:2],x2.T,P2nino3plio[:,2:4],P2nino3plio[:,5:]),axis=1)

models=['CCSM4-2deg','CCSM4-UofT','CCSM4-1deg','CESM1.2','CESM2',
	'COSMOS','EC-EARTH3.3','GISS-E2-1-G','HadCM3','IPSLCM6ALR',
	'IPSLCM5A','IPSLCM5A2','MIROC4m','MRI-CGCM2.3',
	'NorESM-L','NorESM1-F']

P2nino3pi_iann=np.zeros((1068,P2nino3pi_2.shape[1]))
P2nino3plio_iann=np.zeros((1068,P2nino3pi_2.shape[1]))
P2nino3pi_dec=np.zeros((1068,P2nino3pi_2.shape[1]))
P2nino3plio_dec=np.zeros((1068,P2nino3pi_2.shape[1]))
for m in range(0,len(models)):
	P2nino3pi_dec[:,m]=moving_average(P2nino3pi_2[:,m].values,11*12)[:-1]
	P2nino3pi_iann[:,m]=P2nino3pi_2[11*12:,m].values-P2nino3pi_dec[:,m]
	P2nino3plio_dec[:,m]=moving_average(P2nino3plio_2[:,m],11*12)[:-1]
	P2nino3plio_iann[:,m]=P2nino3plio_2[11*12:,m]-P2nino3plio_dec[:,m]

P2dstdnino3_iann=np.round(((P2nino3plio_iann.std(axis=0)-P2nino3pi_iann.std(axis=0))*100)/P2nino3pi_iann.std(axis=0),0)
P2dstdnino3_dec=np.round(((P2nino3plio_dec.std(axis=0)-P2nino3pi_dec.std(axis=0))*100)/P2nino3pi_dec.std(axis=0),0)

fig1 = plt.figure(3,figsize=(7,12),dpi=150)
x1=np.arange(1,17)
ax1 = fig1.add_axes([0.3,.65,.47,.27])
ax1.bar(x1-0.15,P2dstdnino3_iann,width=0.3,color='royalblue',alpha=0.7)
ax1.bar(x1+0.15,P2dstdnino3_dec,width=0.3,color='deepskyblue',alpha=0.7)
plt.ylim([-70,60])
plt.yticks(np.arange(-70,70,10))
plt.xticks(x1,models,rotation='vertical')
plt.plot([0,18],[0,0],color='k',linewidth=0.2)
plt.xlim([0,17])


##############################################
################## PLIOMIP1 ##############################
z1=sorted(glob.glob('/polar/wk1/pliomip1/sst/anom/*pi*atl*'))
z2=sorted(glob.glob('/polar/wk1/pliomip1/sst/anom/*lio*atl*'))

mymodels=['CCSM4','COSMOS-2','FGOALS-g2',
	'GISS-E2-R','HadCM3','IPSLCM5A-2','MIROC4m',
	'MRI-CGCM2.3','NorESM-L']

nino3pistd=np.zeros((len(z1)))
nino3pliostd=np.zeros((len(z1)))
nino34pistd=np.zeros((len(z1)))
nino34pliostd=np.zeros((len(z1)))

for m in range(0,len(z1)):
	cdata = xr.open_dataset(z1[m],decode_times=False)
	pdata = xr.open_dataset(z2[m],decode_times=False)
	
	nino3pi=cdata['anomtos'].sel(lat=slice(-5,5),lon=slice(-150,-90)).mean('lat').mean('lon')
	nino3plio=pdata['anomtos'].sel(lat=slice(-5,5),lon=slice(-150,-90)).mean('lat').mean('lon')
	
	nino3pi=xr.DataArray(nino3pi.values,coords=[nino3pi['time']],dims=['timepi'])
	nino3plio=xr.DataArray(nino3plio.values,coords=[nino3plio['time']],dims=['timeplio'])

	long,latg=np.meshgrid(cdata['lon'],cdata['lat'])

	nino3pistd[m]=np.std(nino3pi)
	nino3pliostd[m]=np.std(nino3plio)

P1dstdnino3=((nino3pliostd-nino3pistd)*100)/nino3pistd

fig1 = plt.figure(2,figsize=(7,12),dpi=150)
x=np.arange(1,10,1)
ax1 = fig1.add_axes([0.17,.65,.27,.27])
ax1.bar(x,P1dstdnino3,width=0.3,color='blue',alpha=0.7)
plt.xticks(x,mymodels,rotation='vertical')
plt.plot([0,10],[0,0],color='k',linewidth=0.2)
plt.xlim([0,10])
plt.ylim([-70,10])

z1=sorted(glob.glob('/polar/wk1/pliomip1/sst/anom/*pi*atl*'))
z2=sorted(glob.glob('/polar/wk1/pliomip1/sst/anom/*lio*atl*'))

mymodels=['CCSM4','COSMOS-2','FGOALS-g2',
	'GISS-E2-R','HadCM3','IPSLCM5A-2','MIROC4m',
	'MRI-CGCM2.3','NorESM-L']

nino3pi=np.zeros((len(z1),1164))
nino3plio=np.zeros((len(z1),1164))

for m in range(0,len(z1)):
	cdata = xr.open_dataset(z1[m],decode_times=False)
	pdata = xr.open_dataset(z2[m],decode_times=False)
	
	nino3pi[m,:]=cdata['anomtos'][-1164:,:,:].sel(lat=slice(-5,5),lon=slice(-150,-90)).mean('lat').mean('lon')
	nino3plio[m,:]=pdata['anomtos'][-1164:,:,:].sel(lat=slice(-5,5),lon=slice(-150,-90)).mean('lat').mean('lon')

######## removing decadal period (11yr)
P1nino3pi_iann=np.zeros((1032,len(z1)))
P1nino3plio_iann=np.zeros((1032,len(z1)))
P1nino3pi_dec=np.zeros((1032,len(z1)))
P1nino3plio_dec=np.zeros((1032,len(z1)))
for m in range(0,len(z1)):
	P1nino3pi_dec[:,m]=moving_average(nino3pi[m,:],11*12)[:-1]
	P1nino3pi_iann[:,m]=nino3pi[m,11*12:]-P1nino3pi_dec[:,m]
	P1nino3plio_dec[:,m]=moving_average(nino3plio[m,:],11*12)[:-1]
	P1nino3plio_iann[:,m]=nino3plio[m,11*12:]-P1nino3plio_dec[:,m]

P1dstdnino3_iann=np.round(((P1nino3plio_iann.std(axis=0)-P1nino3pi_iann.std(axis=0))*100)/P1nino3pi_iann.std(axis=0),0)
P1dstdnino3_dec=np.round(((P1nino3plio_dec.std(axis=0)-P1nino3pi_dec.std(axis=0))*100)/P1nino3pi_dec.std(axis=0),0)

fig1 = plt.figure(2,figsize=(7,12),dpi=150)
x=np.arange(1,10,1)
ax1 = fig1.add_axes([0.17,.65,.27,.27])
ax1.bar(x-0.15,P1dstdnino3_iann,width=0.3,color='royalblue',alpha=0.7)
ax1.bar(x+0.15,P1dstdnino3_dec,width=0.3,color='deepskyblue',alpha=0.7)
plt.ylim([-70,60])
plt.yticks(np.arange(-70,70,10))
plt.xticks(x,mymodels,rotation='vertical')
plt.plot([0,10],[0,0],color='k',linewidth=0.2)
plt.xlim([0,10])