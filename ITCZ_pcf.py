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
import scipy.stats as ss
plt.ion()

z1=sorted(glob.glob('/polar/wk1/pliomip1/ppt/clim/*cont*.nc'))
z2=sorted(glob.glob('/polar/wk1/pliomip1/ppt/clim/*plio*.nc'))

modelp1=['CCSM4','COSMOS','GISS-E2-R','HadCM3',
	'IPSLCM5A','MIROC4m','MRI-CGCM2.3','NorESM-L']

nlati=91;nloni=181
lati=np.arange(-90,90.5,2)
loni=np.arange(0,361,2)
lonig,latig=np.meshgrid(loni,lati)
xi=(lonig,latig)
cvarm=np.zeros((len(z1),nlati,nloni))
dvarm=np.zeros((len(z1),nlati,nloni))
pvarm=np.zeros((len(z1),nlati,nloni))
dausPliomip1=np.zeros((len(z1)))
citcz=np.zeros((len(z1)))
pitcz=np.zeros((len(z1)))
AVcpri=np.array([])
AVppri=np.array([])

zmask=sorted(glob.glob('/polar/wk1/pliomip1/land_mask/*mask.nc'))[0]
mdata = xr.open_dataset(zmask,decode_times=False)
mask=mdata['mask']
long,latg=np.meshgrid(mdata['lon'],mdata['lat'])
values = (long.flatten(),latg.flatten());
mask = inter.griddata(values, mask.values.flatten(),xi);

# fig = plt.figure(2,figsize=(12,8),dpi=150)
# gs = gridspec.GridSpec(2,5)
for m in range(0,len(z1)):
	cdata = xr.open_dataset(z1[m],decode_times=False)
	pdata = xr.open_dataset(z2[m],decode_times=False)
	if m==4:
		cdata.rename({'time':'time1'},inplace=True)
		cdata.rename({'time_counter':'time'},inplace=True)
		cdata.rename({'time1':'time'},inplace=True)
		pdata.rename({'time':'time1'},inplace=True)
		pdata.rename({'time_counter':'time'},inplace=True)
		pdata.rename({'time1':'time'},inplace=True)
	elif m==3:
		cdata.rename({'lon':'lon1','lat':'lat1','time':'time1'},inplace=True)
		cdata.rename({'longitude':'lon','latitude':'lat','t':'time'},inplace=True)
		cdata.rename({'lon1':'lon','lat1':'lat','time1':'time'},inplace=True)
		pdata.rename({'lon':'lon1','lat':'lat1','time':'time1'},inplace=True)
		pdata.rename({'longitude':'lon','latitude':'lat','t':'time'},inplace=True)
		pdata.rename({'lon1':'lon','lat1':'lat','time1':'time'},inplace=True)		
	
	if m==0:
		cvar=cdata['pr']
		pvar=pdata['pr']
	elif m==2:
		cvar=cdata['pr']
		pvar=pdata['pr']
	elif m==3:
		cvar=np.squeeze(cdata['pr'])
		pvar=np.squeeze(pdata['pr'])
	elif m==6:
		cvar=np.squeeze(cdata['pr'])
		pvar=np.squeeze(pdata['pr'])
	elif m==7:
		cvar=cdata['pr']
		pvar=pdata['pr']
	else:
		cvar=cdata['pr']
		pvar=pdata['pr']

	cvar = cvar.roll(time=3)[:5,:,:].mean('time').values
	pvar = pvar.roll(time=3)[:5,:,:].mean('time').values
	
	lon1=np.array(cdata['lon'].values)
	lat=cdata['lat']
	if m==4:
		lon,cvar=shift_pcf(lon1,cvar)
		lon,pvar=shift_pcf(lon1,pvar)
	else:
		lon=lon1

	long,latg=np.meshgrid(lon,lat)
	lon=xr.DataArray(lon,dims='lon',name='lon')
	cvar=xr.DataArray(cvar,coords=[lat,lon],dims=['lat','lon'])
	pvar=xr.DataArray(pvar,coords=[lat,lon],dims=['lat','lon'])

	values = (long.flatten(),latg.flatten());
	cvari = inter.griddata(values, cvar.values.flatten(),xi);
	pvari = inter.griddata(values, pvar.values.flatten(),xi);

	dvarm[m,:,:]=pvari-cvari
	cvarm[m,:,:]=cvari
	pvarm[m,:,:]=pvari

	cpr=cvari*mask; ppr=pvari*mask

	del lon; del lat
	lon=loni;lat=lati
	avcpr=np.zeros((len(lat)))
	avppr=np.zeros((len(lat)))
	# fig = plt.figure(m,figsize=(12,12),dpi=150)
	# ax1=fig.add_axes([.1,.1,.8,.8])
	# ax1.contourf(lon,lat,cpr)
	for j in range(0,len(lat)):
		indi=findnearest(lon,200) #central pacific
		tmp=np.squeeze(cpr[j,:]);tmp=np.argwhere(np.isnan(tmp))
		a=np.where(tmp<indi)[0]
		b=np.where(tmp>indi)[0]
		indi1=tmp[a.max()]-1
		indi2=tmp[b.min()]+1
		if indi2>=len(lon):
			indi1=0;indi2=-1;		
		avcpr[j]=np.nanmean(cpr[j,int(indi1):int(indi2)])
		avppr[j]=np.nanmean(ppr[j,int(indi1):int(indi2)])
		# ax1.plot([lon[indi1],lon[indi2]],[lat[j],lat[j]],
		# 	color='magenta',linewidth=0.5)
		# #global
		# avcpr[j]=np.nanmean(cpr[j,:])
		# avppr[j]=np.nanmean(ppr[j,:])

	f=inter.interp1d(lat,avcpr)
	lati2=np.arange(-70,70,0.1)
	avcpri=f(lati2)
	indj1=findnearest(lati2,-20)
	indj2=findnearest(lati2,20)
	avcpri2=avcpri[indj1:indj2]
	a=avcpri2.max()/2
	b=np.where(avcpri2>=a)
	lat2=lati2[indj1:indj2]
	citcz[m]=np.sum((lat2[b]*avcpri2[b]))/np.sum(avcpri2[b])

	f=inter.interp1d(lat,avppr)
	avppri=f(lati2)
	avppri2=avppri[indj1:indj2]
	a=avppri2.max()/2
	b=np.where(avppri2>=a)
	pitcz[m]=np.sum((lat2[b]*avppri2[b]))/np.sum(avppri2[b])
ditczPliomip1=pitcz-citcz

p1itczbias=citcz-(7.4)

# 	ax1 = fig.add_subplot(gs[m])
# 	ax1.plot(avcpri2,lat2,color='blue')
# 	ax1.plot(avppri2,lat2,color='red')
# 	ax1.plot([0,60],[citcz[m],citcz[m]],color='blue')
# 	ax1.plot([0,60],[pitcz[m],pitcz[m]],color='red')
# 	plt.xlim(0,60)
# 	plt.ylim(-20,20)
# 	plt.title(modelp1[m])
# gs.update(hspace=.5,wspace=.5)
# itczpip1=np.concatenate(([citcz[0]],citcz[2:]))

dPRp1=xr.DataArray(dvarm,[('model', modelp1),('lat', lati),('lon',loni)])

ditczPliomip1=pitcz-citcz
itczP1pi=citcz
itczP1plio=pitcz

############## PLIOMIP2 ##################
#------------------------------------------------------------------
z1=sorted(glob.glob('/aurora/wk3/pliomip2/pr/*E280*1deg.nc'))
z2=sorted(glob.glob('/aurora/wk3/pliomip2/pr/*oi400*1deg.nc'))

modelspr=['CCSM4-UTRECHT','UofT-CCSM4','CCSM4-1deg',
	'CESM1.2','CESM2.1','COSMOS',
	'EC-EARTH3.3','GISS-E2-1-G','HadCM3','IPSL6',
	'IPSLCM5A','IPSLCM5A2','MIROC4m','MRI-CGCM2.3',
	'NorESM-L','NorESM1-F']

fnmask='/aurora/wk3/pliomip2/mask.1deg.nc'
maskd = xr.open_dataset(fnmask,decode_times=False)
mask=maskd['temp'].mean('t').squeeze()/maskd['temp'].mean('t').squeeze()

itczE280=np.zeros((len(z1)))
itczEoi400=np.zeros((len(z1)))

PRE280=np.zeros((len(z1),180,360))
PREoi400=np.zeros((len(z1),180,360))
# fig = plt.figure(2,figsize=(12,8),dpi=150)
# gs = gridspec.GridSpec(3,5)
for m in range(0,len(z1)):
	cdata = xr.open_dataset(z1[m],decode_times=False)
	pdata = xr.open_dataset(z2[m],decode_times=False)
	if m==2 or m==3:
		prE280,xx=clim_anom(cdata['PRECC'].values+cdata['PRECL'].values)
		prEoi400,xx=clim_anom(pdata['PRECC'].values+pdata['PRECL'].values)
		lat=cdata['lat']
		lon=cdata['lon']
	elif m==12:
		fn1='/polar/wk1/pliomip2/MIROC4m_E280_Amon_pr.nc'
		fn2='/polar/wk1/pliomip2/MIROC4m_Eoi400_Amon_pr.nc'
		cdata = xr.open_dataset(z1[m-1],decode_times=False)
		cdatax = xr.open_dataset(fn1,decode_times=False)
		pdatax = xr.open_dataset(fn2,decode_times=False)
		pr1x,xx=clim_anom(cdatax['pr'].values)
		pr2x,xx=clim_anom(pdatax['pr'].values)
		lon=cdata['lon']
		lat=cdata['lat']
		long,latg=np.meshgrid(lon,lat)
		lonx=cdatax['lon']
		latx=cdatax['lat']
		lonx,latx=np.meshgrid(lonx,latx)
		prE280=np.zeros((12,180,360))
		prEoi400=np.zeros((12,180,360))
		for t in range(0,12):
			prE280[t,:,:] = inter.griddata((lonx.flatten(),latx.flatten()), 
				pr1x[t,:,:].flatten(),
				(long.flatten(),latg.flatten())).reshape(180,360)
			prEoi400[t,:,:] = inter.griddata((lonx.flatten(),latx.flatten()), 
				pr2x[t,:,:].flatten(),
				(long.flatten(),latg.flatten())).reshape(180,360)
	else:
		prE280,xx=clim_anom(cdata['pr'].squeeze().values)
		prEoi400,xx=clim_anom(pdata['pr'].squeeze().values)
		lat=cdata['lat']
		lon=cdata['lon']
		long,latg=np.meshgrid(lon,lat)

	time_clim=np.arange(1,13,1)
	prE280=xr.DataArray(prE280,[('time', time_clim),('lat', lat),('lon',lon)])
	prEoi400=xr.DataArray(prEoi400,[('time', time_clim),('lat', lat),('lon',lon)])

	prE280=prE280.roll(time=3)[:5,:,:].mean('time')
	prEoi400=prEoi400.roll(time=3)[:5,:,:].mean('time')

	PRE280[m,:,:]=prE280.values
	PREoi400[m,:,:]=prEoi400.values

	lat=cdata['lat'].values
	long,latg=np.meshgrid(lon,lat)

	prE280=xr.DataArray(prE280,[('lat', lat),('lon',lon)])
	prEoi400=xr.DataArray(prEoi400,[('lat', lat),('lon',lon)])

	avprE280=np.zeros((len(lat)))
	avprEoi400=np.zeros((len(lat)))
	cpr=prE280.values*mask.values
	ppr=prEoi400.values*mask.values
	avcpr=np.zeros((len(lat)))
	avppr=np.zeros((len(lat)))
	# fig = plt.figure(m,figsize=(12,12),dpi=150)
	# ax1=fig.add_axes([.1,.1,.8,.8])
	# ax1.contourf(lon,lat,cpr)
	for j in range(0,len(lat)):
		indi=findnearest(lon.values,200) #central pacific
		tmp=np.squeeze(cpr[j,:]);tmp=np.argwhere(np.isnan(tmp))
		a=np.where(tmp<indi)[0]
		b=np.where(tmp>indi)[0]
		indi1=tmp[a.max()]-1
		indi2=tmp[b.min()]+1
		if indi2>=len(lon):
			indi1=0;indi2=-1;		
		avcpr[j]=np.nanmean(cpr[j,int(indi1):int(indi2)])
		avppr[j]=np.nanmean(ppr[j,int(indi1):int(indi2)])
		# # ax1.plot([lon[indi1],lon[indi2]],[lat[j],lat[j]],
		# # 	color='magenta',linewidth=0.5)
		
	f=inter.interp1d(lat,avcpr)
	lati=np.arange(-70,70,0.1)
	avcpri=f(lati)
	indj1=findnearest(lati,-20)
	indj2=findnearest(lati,20)
	avcpri2=avcpri[indj1:indj2]
	a=avcpri2.max()*.5
	b=np.where(avcpri2>=a)
	lat2=lati[indj1:indj2]
	itczE280[m]=np.sum((lat2[b]*avcpri2[b]))/np.sum(avcpri2[b])

	f=inter.interp1d(lat,avppr)
	avppri=f(lati)
	avppri2=avppri[indj1:indj2]
	a=avppri2.max()*.5
	b=np.where(avppri2>=a)
	itczEoi400[m]=np.sum((lat2[b]*avppri2[b]))/np.sum(avppri2[b])
ditczPliomip2=itczEoi400-itczE280


dPRp2=PREoi400-PRE280
dPRp2=xr.DataArray(PREoi400-PRE280,[('model', modelspr),('lat', lat),('lon',lon)])

fig = plt.figure(1,figsize=(12,10),dpi=150)
m1 = Basemap(projection='robin',lon_0=-60,resolution='c')
m1.drawcoastlines(color='darkgrey',linewidth=0.7)
m1.drawparallels(np.arange(-60,30,20),color='dimgrey')
d=m1.contourf(long[:,2:-2],latg[:,2:-2],PRE280.mean(axis=0)[:,2:-2],np.arange(0,11,1),
	latlon=True,
	cmap=cmaps.MPL_Blues,extend='both')
kw1 = {'levels':[-12,-8,8,12], 'colors':['k']}
p=(dPRp2.mean(axis=0)*mask)
a=m1.contour(long2,latg2,dPRp2.mean(axis=0)*mask,
	latlon=True,**kw1,linewidths=0.7,alpha=.8)
plt.clabel(a, inline=1, fontsize=8, fmt='%1.1f')

kw1 = {'levels':[0.8,1.2,1.4,1.6,1.8], 'colors':['k']}
a=m1.contour(long1,latg1,sstdm,
	latlon=True,**kw1,linewidths=1,alpha=1)
plt.clabel(a, inline=1, fontsize=8, fmt='%1.1f')

cax2 = fig.add_axes([0.31, 0.1, 0.4, 0.02])
cbar2=fig.colorbar(d,cax2,ticks=np.arange(0,10,2),
	label=r'$\Delta$ SST [$^{o}$C]',
	orientation='horizontal',ticklocation='bottom',extend='both')
plt.savefig('/home/gabriel/paper_sa_monsoon/MMM_PPT_p2_.pdf',
		dpi=300,facecolor='w', edgecolor='w',
        orientation='landscape', papertype=None, format='pdf',
        transparent=True, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)

#####################################
ditcz=np.concatenate((ditczPliomip1,ditczPliomip2))
denso=np.concatenate((P1dstdnino3,P2dstdnino3))

print(np.round(ss.spearmanr(ditcz,denso),2))

f=np.polyfit(denso[a[0]],ditcz[a[0]],1)

font=25
plt.figure(10,figsize=(9.5,7.5))
plt.scatter(P1dstdnino3,ditczPliomip1,s=110,marker='s',c='magenta',alpha=.7)
plt.scatter(P2dstdnino3,ditczPliomip2,s=110,c='red',alpha=.7)
for m in range(0,len(ditczPliomip2)):
	plt.text(P2dstdnino3[m],ditczPliomip2[m],modelspr[m])
for m in range(0,len(ditczPliomip1)):
	plt.text(P1dstdnino3[m],ditczPliomip1[m],modelp1[m])
# plt.scatter(dnino3obs,ditczobs,s=600,marker=(5, 1),c='limegreen')
plt.plot([-55,10],np.polyval(f,[-55,10]),'--',
	linewidth=4,color='k',alpha=.7)
plt.plot([0,0],[-100,100],linewidth=0.3,color='dimgrey')
plt.plot([-100,100],[0,0],linewidth=0.3,color='dimgrey')
plt.xlim(-80,20)
plt.ylim(-2,7)
plt.xticks(np.arange(-80,30,10),fontsize=font)
plt.yticks(np.arange(-2,8,1),fontsize=font)
plt.xlabel('ENSO amplitude change [%]',fontsize=font)
plt.ylabel('ITCZ shift [deg]',fontsize=font)

############################################################
ditczPliomip1_2=np.concatenate(([ditczPliomip1[0]],ditczPliomip1[2:]))
duasip1_W_2=np.concatenate(([duasip1_W[0]],duasip1_W[2:]))

ditczPliomip2_2=np.concatenate(([ditczPliomip2[2]],[ditczPliomip2[1]],ditczPliomip2[4:]))

ditcz=np.concatenate((ditczPliomip1_2,ditczPliomip2_2))
duas=np.concatenate((duasip1_W_2,duasip2_W))

print(ss.spearmanr(duas,ditcz))

f=np.polyfit(duas,ditcz,1)

models=['CESM1deg','CCSM4-UofT','CESM2','COSMOS',
	'EC-EARTH3.3','GISS-E2-1-G','HadCM3','IPSLCM6ALR',
	'IPSLCM5A','IPSLCM5A2','MIROC4m','MRI-CGCM2.3',
	'NorESM-L','NorESM1-F']
modelp1=['CCSM4','GISS','HadCM3','IPSL',
	'MIROC','MRI','NorESM']

font=25
plt.figure(20,figsize=(9.5,7.5))
plt.scatter(duasip1_W_2,ditczPliomip1_2,marker='s',s=90,c='magenta',alpha=.7)
plt.scatter(duasip2_W[:5],ditczPliomip2_2[:5],s=90,c='red',alpha=.7)
plt.scatter(duasip2_W[6:],ditczPliomip2_2[6:],s=90,c='red',alpha=.7)
plt.plot([-25,35],np.polyval(f,[-25,35]),'--',
	linewidth=4,color='k',alpha=.7)
plt.plot([0,0],[-100,100],linewidth=0.3,color='dimgrey')
plt.plot([-100,100],[0,0],linewidth=0.3,color='dimgrey')
plt.ylim(-2,8)
plt.xlim(-30,40)
plt.xticks(np.arange(-30,41,10),fontsize=font)
plt.yticks(np.arange(-2,9,1),fontsize=font)
