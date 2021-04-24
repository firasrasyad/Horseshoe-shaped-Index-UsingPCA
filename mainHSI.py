from netCDF4 import Dataset
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import pandas as pd
import xarray as xr
import numpy as np
from sklearn.decomposition import PCA

dataset = Dataset('ccam_gfdlcm3_mon.194901_209911.nc')
time = dataset.variables['time'][:]
Temp200 = dataset.variables['ta200'][:]

Temp200_Eq=np.nanmean(Temp200[:,50:166,:],axis=1)
Temp200_7510N=np.nanmean(Temp200[:,166:,:],axis=1)
Temp200_7510S=np.nanmean(Temp200[:,0:50,:],axis=1)

CCAM_HSIK=np.empty(shape=(1812,386-77))
CCAM_HSIR=np.empty(shape=(1812,386-77))

for i in range (77,386):
    CCAM_HSIK[:,i-77] = Temp200_Eq[:,i+77] - Temp200_Eq[:,i-77]
    CCAM_HSIR[:,i-77] = (Temp200_7510N[:,i] + Temp200_7510S[:,i])/2 - Temp200_Eq[:,i]

#Climatology HSI
waktu=[]
for a in range (0,time.size): #1812 grid
    ws=pd.to_datetime(int(time[a]),unit='m',origin='1949-1-1')
    waktu.append(ws)

Lon = dataset.variables['lon'][77:386]
XHSIK=xr.DataArray(CCAM_HSIK,coords={'time':waktu,'lon':Lon},dims=['time','lon'])
XHSIR=xr.DataArray(CCAM_HSIR,coords={'time':waktu,'lon':Lon},dims=['time','lon'])

Clim_HSIK=XHSIK.groupby('time.month').mean('time')
Clim_HSIR=XHSIR.groupby('time.month').mean('time')

#Compute PCA (EOF)

HSI_C = [CCAM_HSIK,CCAM_HSIR]
HSI_C = np.reshape(HSI_C,[2,1812*(386-77)])
HSI_C = np.transpose(HSI_C)

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(HSI_C)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])

print(pca.explained_variance_ratio_)

PC1 = np.reshape(principalComponents[:,0],[1812,386-77])
PC2 = np.reshape(principalComponents[:,1],[1812,386-77])

XPC1=xr.DataArray(PC1,coords={'time':waktu,'lon':Lon},dims=['time','lon'])
XPC2=xr.DataArray(PC2,coords={'time':waktu,'lon':Lon},dims=['time','lon'])

Clim_PC1=XPC1.groupby('time.month').mean('time')
Clim_PC2=XPC2.groupby('time.month').mean('time')

#Plotting Result Climatology HSIK
mon_list=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

fig, (ax1, ax2) = plt.subplots(figsize=(20,15),nrows=2,sharex=True)
plt.subplots_adjust(left=0.1, bottom=0.05, right=0.9, top=0.95, wspace=0, hspace=0.095)

# fig = plt.figure(figsize=(10,4))
# ax2 = plt.axes()

lev=np.arange(-0.4,0.41,0.05)
K=ax1.contourf(Lon[:],mon_list,Clim_HSIK[:],cmap='RdBu_r',levels=lev,extend='both')
# ax1.set_xlabel('Longitude')
fig.colorbar(K,label='HSI-K',ax=ax1)
ax1.set_title('Climatolgical HSI-K')

R=ax2.contourf(Lon[:],mon_list,Clim_HSIR[:],cmap='RdBu_r',extend='both')
ax2.set_xlabel('Longitude')
fig.colorbar(R,label='HSI-R',ax=ax2)
ax2.set_title('Climatolgical HSI-R')
plt.show()

# K=ax1.contourf(Lon[:],mon_list,Clim_PC1[:],cmap='RdBu_r',levels=lev,extend='both')
# # ax1.set_xlabel('Longitude')
# fig.colorbar(K,label='PC 1',ax=ax1)
# ax1.set_title('Climatolgical PC 1 %1.3f percent variance' %(pca.explained_variance_ratio_[0]*100))

# R=ax2.contourf(Lon[:],mon_list,Clim_PC2[:],cmap='RdBu_r',extend='both')
# ax2.set_xlabel('Longitude')
# fig.colorbar(R,label='PC 2',ax=ax2)
# ax2.set_title('Climatolgical PC 2 %1.3f percent variance' %(pca.explained_variance_ratio_[1]*100))
# plt.show()

# fig.savefig('Clim_HSI',bbox='tight')