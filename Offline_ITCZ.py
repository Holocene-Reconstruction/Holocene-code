#==============================================================================
# Script for caclulating ITCZ
# Work in progress
# Meant to be run offline before performing DA.
#    author: Chris Hancock
#    date  : 2/15/2023
#==============================================================================

#setup file locations
model = 'DAMP12kTraCE' #Only stable up for Trace_21ka. #TODO: HadCM / othermodels
data_dir="/Users/chrishancock/Library/CloudStorage/OneDrive-NorthernArizonaUniversity/Research/Manuscript/DAMP12k/Data/models/"+model+"/"

#Load Packages
import xarray as xr
import numpy as np
import math
import scipy 
import pyet                               # Package for calciulating potential ET
import xesmf as xe                        # Regridding package needed for ITCZ
import matplotlib.pyplot   as plt         # Packages for making figures
import cartopy.crs         as ccrs        # Packages for mapping in python
import cartopy.util        as cutil

#Save Created Files? T/F
save = False
#%% Load TraCE data and covert to desired units
minAge  = 0
maxAge  = 12000
binsize = 10
binyrs  = [*range(minAge,maxAge+1,binsize)]
binvec  = [*range(minAge-int(binsize/2),maxAge+int(binsize/2)+1,binsize)] 
binvec = [np.NaN] #set as [np.NaN] to retain native resolution

def savedata(dataarray,var_text,data_dir=data_dir,save=False):
    agerange = [str(round(np.max(dataarray.age.data/1000))),str(round(np.min(dataarray.age.data/1000)))]
    if save: dataarray.rename(var_text).to_netcdf(data_dir+'processed/'+'trace.DAMP12k.'+var_text+'.'+agerange[0]+'ka_decavg_'+agerange[1]+'ka.nc')
    ax = plt.axes(projection=ccrs.Robinson()) 
    p=ax.pcolormesh(dataarray.lon,dataarray.lat,dataarray[:,0,:,:].mean(axis=0),transform=ccrs.PlateCarree())
    ax.spines['geo'].set_edgecolor('black'); ax.set_global(); ax.coastlines()
    plt.title(var_text+' ('+dataarray.units+')')
    plt.colorbar(p,orientation='horizontal')
    plt.show()
    print("###################")
    print(var_text+"/"+dataarray.attrs['long_name']+" ("+dataarray.units+")")
    print(np.shape(dataarray))
    print("min:"+str(int(np.min(dataarray)))+"; max: "+str(int(np.max(dataarray)))+"; mean:"+str(float(np.mean(dataarray))))
    #

#%%Load Trace Data
#########################################
varkey = ['V','U','PRECT']
heights = [500]
#
model_data={}

for var_name in varkey:             
    print(var_name)
    handle_model=xr.open_dataset(data_dir+'original/trace.01-36.22000BP.cam2.'+var_name+'.22000BP_decavg_400BCE.nc' ,decode_times=False)
    model_data[var_name] =  [handle_model[var_name].expand_dims(dim='season', axis=1,).assign_coords({"season": ("season", ['ANN'])})]
    for szn in ['JJA','DJF','MAM','SON']:       
        try:
            handle_model=xr.open_dataset(data_dir+'original/trace.01-36.22000BP.cam2.'+var_name+'.22000BP_decavg'+szn+'_400BCE.nc' ,decode_times=False)
            model_data[var_name].append(handle_model[var_name].expand_dims(dim='season', axis=1,).assign_coords({"season": ("season", [szn])}))
        except: continue
    handle_model.close()
    model_data[var_name] = xr.concat(model_data[var_name],dim='season')
    #Reshape time as needed
    model_data[var_name]=model_data[var_name].assign_coords(time=model_data[var_name]['time']*-1000).rename({'time': 'age'})
    if binvec != [np.NaN]: #If choosing to bin time data to a courser resolution
        model_data[var_name] = model_data[var_name].groupby_bins("age",binvec).mean(dim='age').rename({'age_bins': 'time'})
        model_data[var_name]['age'] = binyrs  
    #Convert valeus
    if var_name  in ['U','V']: #data already in m/s 
        model_data[var_name] = model_data[var_name].sel(lev=slice(800,2000)).mean(dim='lev')
        model_data[var_name].attrs['long_name'] = 'Surface '+var_name#model_data[var_name].attrs['long_name']
    elif  var_name == 'PRECT': #Convert m/s to mm/day
        model_data[var_name].data = model_data[var_name].data*1000*60*60*24
        model_data[var_name].rename('precip')
        model_data[var_name].attrs['units'] ='mm/day'
        
        
   #%%Streamfunction     
from windspharm.standard import VectorWind
from windspharm.tools import prep_data, recover_data, order_latdim
from windspharm.examples import example_data_path
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.util import add_cyclic_point
lat=90
itcz = {}
for season in ['DJF','JJA']:
    # Read zonal and meridional wind components from file using the netCDF4
    # module. The components are defined on pressure levels and are in separate
    # files.
    uwnd = model_data['U'].sel(season=season).sel(lat=slice(-lat,lat))
    vwnd = model_data['V'].sel(season=season).sel(lat=slice(-lat,lat))
    sf_array=(model_data['U'].sel(season=season).sel(lat=slice(-lat,lat))*np.NaN).rename('streamfunction')
    lons = vwnd.lon
    lats = vwnd.lat
    
    # The standard interface requires that latitude and longitude be the leading
    # dimensions of the input wind components, and that wind components must be
    # either 2D or 3D arrays. The data read in is 3D and has latitude and
    # longitude as the last dimensions. The bundled tools can make the process of
    # re-shaping the data a lot easier to manage.
    uwnd, uwnd_info = prep_data(uwnd.data, 'tyx')
    vwnd, vwnd_info = prep_data(vwnd.data, 'tyx')
    # It is also required that the latitude dimension is north-to-south. Again the
    # bundled tools make this easy.
    lats, uwnd, vwnd = order_latdim(lats, uwnd, vwnd)
    
    # Create a VectorWind instance to handle the computation of streamfunction and
    # velocity potential.
    w = VectorWind(uwnd, vwnd)
    
    # Compute the streamfunction and velocity potential. Also use the bundled
    # tools to re-shape the outputs to the 4D shape of the wind components as they
    # were read off files.
    sf = w.streamfunction()
    sf = recover_data(sf, uwnd_info)
    vals=  sf[:,:-1,:] - sf[:,1:,:]
    sf_array=sf_array[:,:-1,:]
    sf_array.data = vals
    itcz[season] = (lats[sf_array.argmax(dim='lat')])#+lats[sf_array.argmin(dim='lat')])/2
    sf_dec, lons_c = add_cyclic_point(np.mean(sf,axis=0), lons)
    ax1 = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
    clevs = [-120, -100, -80, -60, -40, -20, 0, 20, 40, 60, 80, 100, 120]
    sf_fill = ax1.contourf(lons_c, lats, sf_dec * 1e-06, clevs,
                           transform=ccrs.PlateCarree(), cmap=plt.cm.RdBu_r,
                           extend='both')
    ax1.coastlines()
    ax1.gridlines()
    ax1.set_xticks([0, 60, 120, 180, 240, 300, 359.99], crs=ccrs.PlateCarree())
    ax1.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=True,
                                       number_format='.0f')
    lat_formatter = LatitudeFormatter()
    ax1.xaxis.set_major_formatter(lon_formatter)
    ax1.yaxis.set_major_formatter(lat_formatter)
    plt.colorbar(sf_fill, orientation='horizontal')
    plt.title(season+' Streamfunction ($10^6$m$^2$s$^{-1}$)', fontsize=16)
    plt.show()



for season in ['DJF','JJA']: plotITCZ(itcz,precip.lon,[model_data['U'],model_data['V']],model_data['PRECT'],season,lat)




#%%
########################################################################
def zonal_mpsi(mlon,nlat,klev,v,lat,p,ps):
########################################################################
    # calculate atmospheric meridional stream function
    # input v needs to be in pressure coordinates
    zmpsi=np.zeros((nlat,klev))
    ptmp=np.zeros(2*klev+1)
    dp=np.zeros(2*klev+1)
    vvprof=np.zeros(2*klev+1)
    vbar=np.zeros((nlat,klev))
    vtmp=np.zeros((mlon,nlat,klev))
    #Constants
    g   = 9.80616e0
    a   = 6.37122e6
    pi  = 4.e0*np.arctan(1.e0)
    rad = pi/180.e0
    con = 2.e0*pi*a/g
    # calculate presssure at all levels [even half levels]
    knt = -1
    #do kl = 1,2*klev - 1,2
    for kl in range(0,2*klev - 2,2):
        knt = knt + 1
        ptmp[kl] = p[knt]
    #do kl = 2,2*klev - 2,2
    for kl in range(1,2*klev - 2,2):
        ptmp[kl] = (ptmp[kl+1]+ptmp[kl-1])*0.5e0
    ptmp[0] = p[0] # ptop
    ptmp[2*klev] = p[-1] # pbot
    # dp at all levels
    dp[0] = 0.e0
    #do kl = 1,2*klev - 1
    for kl in range(1,2*klev - 1):
        dp[kl] = ptmp[kl+1] - ptmp[kl-1]
    dp[2*klev] = 0.e0
    # make copy; set any p > ps to NaN:
    vtmp=1.0*v
    p3d=0.0*v+p
    ps3d=0.0*v
    for kkk in range(len(p)):
        ps3d[:,:,kkk]=ps
    vtmp[p3d>ps3d]=np.nan
    # compute zonal mean v using the vtmp variable
    vbar=np.nanmean(vtmp,axis=0)
    # compute mpsi at each latitude [reuse ptmp]
    #do nl = 1,nlat
    for nl in range(1,nlat):
        c = con*np.cos(lat[nl]*rad)
        ptmp[0] = 0.0e0
        #do kl = 1,2*klev
        for kl in range(1,2*klev+1):
            ptmp[kl] = np.nan
        #do kl = 0,2*klev,2
        for kl in range(0,2*klev+2,2):
            vvprof[kl] = 0.0e0
        knt = 0
        #do kl = 1,2*klev - 1,2
        for kl in range(1,2*klev-1,2):
            knt = knt + 1
            vvprof[kl] = vbar[nl,knt]
        # integrate from top of atmosphere down for each level where vbar
        # is not missing
        stop=False
        #do kl = 1,2*klev - 1,2
        for kl in range(1,2*klev - 1,2):
            if ((not stop) and (np.isnan(vvprof[kl]))):
                stop=True
            if not stop:
                kflag = kl
                ptmp[kl+1] = ptmp[kl-1] - c*vvprof[kl]*dp[kl]
        # impose lower boundary condition to ensure the zmpsi is 0
        # at the bottom boundary
        ptmp[kflag+1] = -ptmp[kflag-1]
        # streamfunction is obtained as average from its values
        # at the intermediate half levels. the minus sign before
        # ptmp in the last loop is to conform to a csm convention.
        #do kl = 1,kflag,2
        for kl in range(1,kflag,2):
            ptmp[kl] = (ptmp[kl+1]+ptmp[kl-1])*0.5e0
        knt = 0
        #do kl = 1,2*klev - 1,2
        for kl in range(1,2*klev - 1,2):
            knt = knt + 1
            if (not np.isnan(ptmp[kl])): # then
                zmpsi[nl,knt] = -ptmp[kl]
            else:
                zmpsi[nl,knt] = ptmp[kl]
    return zmpsi,vbar


varkey = ['V','PS']
#
msf = {}
for season in ['ANN','JJA','DJF']:   
    print(season)
    model_data2={}
    for var_name in varkey:             
        print(var_name)
        if season == 'ANN': handle_model=xr.open_dataset(data_dir+'original/trace.01-36.22000BP.cam2.'+var_name+'.22000BP_decavg_400BCE.nc' ,decode_times=False)
        else: handle_model=xr.open_dataset(data_dir+'original/trace.01-36.22000BP.cam2.'+var_name+'.22000BP_decavg'+season+'_400BCE.nc' ,decode_times=False)
        model_data2[var_name] = handle_model[var_name]
        handle_model.close()
        model_data2[var_name]=model_data2[var_name].assign_coords(time=model_data2[var_name]['time']*-1000).rename({'time': 'age'})
    #
    msf[season] = model_data2['V']*np.NaN
    #
    V_all  = model_data2['V'].data
    PS_all = model_data2['PS'].data
    lats   = model_data2['V'].lat.data
    lons   = model_data2['V'].lon.data
    pres   = model_data2['V'].lev.data
    #    
    phi = np.radians(lats)    # latitude in radians
    nlat=len(lats)
    vmsg=1.e36
    print("calculating psi...")
    for lon in range(len(msf[season].lon)):
        print(np.round(100*lon/len(msf[season].lon)))
        for age in range(len(msf[season].age)):
            V=1.0*V_all[age,:,:,lon:(lon+1)]
            V=1.0*V.transpose(2,1,0)
            PS=1.0*PS_all[age,:,lon:(lon+1)]
            PS=1.0*PS.transpose(1,0)
            #
            mlon,nlat,klev=V.shape
            psi,vbar=zonal_mpsi(mlon,nlat,klev,V,lats,pres,PS) # multiply by 100 because pres is in mb #don't do this because in PA
            psi[np.abs(psi)>1.e30]=np.nan
            msf[season][age,:,:,lon] = np.transpose(psi)
    
#%%
print("plotting...")
for season in ['JJA','DJF']:
    fig=plt.figure(figsize=(5,5),dpi=300)
    plt.set_cmap('bwr')
    levels=np.arange(-140,150,10)
    plt.contourf(lats,pres,msf[season].mean(dim=['lon','age'])*100/1.e9,levels=levels,extend='both')
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.xticks(ticks=range(-90,120,30))
    plt.xlabel("Latitude")
    plt.ylabel("Pressure (hPa)")
    plt.title(season+" Meridional streamfunction (Sv)")
    plt.show()
    #fig.savefig("Output/streamfunction_python.pdf")

for season in ['JJA','DJF']:
    msf[season].attrs['units'] ='500 hPa'
    msf[season].attrs['long_name'] = ' Meridional Streamfunction'
    savedata(msf[season][:,18:19,:,:],season+' Meridional Streamfunction')





#%%
lat=20
# Calculate meridional circulation
itcz={}
for season in ['DJF','JJA']:
    vals = msf[season].sel(lev=slice(300,700)).mean(dim='lev')
    newlats = (vals.lat[:-1]+vals.lat[1:])/2
    idx_trp = np.where((newlats>=-lat) & (newlats<= lat))[0]
    vals_array = (vals[:,1:,:].data - vals[:,:-1,:].data)[:,idx_trp,:]
    vals2=vals[:,idx_trp,:]
    vals2.data=vals_array
    itcz[season] = newlats[idx_trp][vals2.argmax(dim='lat')]
#%%
for age in [0]:#[18,12,6,0]:
    itczage = {}
    for szn in ['JJA','DJF']: 
        itczage[szn] = itcz[szn]#.sel(age=slice(age*1000+100,age*1000-100))
        plotITCZ(itczage,msf[season].lon,[model_data['U'],model_data['V']],model_data['PRECT'],szn)


#%%

def plotITCZ(itczArray,lons,UV=False,precip=False,season='ANN',bounds=40):
    plt.figure(figsize=(8,3),dpi=600)
    ax = plt.axes(projection=ccrs.Robinson()) 
    ax.spines['geo'].set_edgecolor('black'); ax.coastlines()
    ax.set_extent((-180,180,-bounds,bounds),ccrs.PlateCarree())
    #
    if len(np.shape(precip)) >0:
        p=ax.pcolormesh(precip.lon,precip.lat,precip.sel(season=season).mean(dim='age'),
                        vmin=0,vmax=10,cmap='Blues',transform=ccrs.PlateCarree())
        plt.colorbar(p,orientation='horizontal',label=season+' precip ('+precip.units+')',shrink=0.3,aspect=50,extend='max')
    #
    if len(np.shape(UV)) >0:
        U, V = np.meshgrid(UV[0].lon.data,UV[0].lat.data)
        ax.quiver(U,V,
                  UV[0].sel(season=season).mean(dim='age').data,
                  UV[1].sel(season=season).mean(dim='age').data,
                  transform=ccrs.PlateCarree(),scale=400,color='k',width=.002)
    #
    itczArray_cyclic,lon_cyclic = cutil.add_cyclic_point(itczArray[season],coord=lons,axis=1)
    ax.plot(lon_cyclic,np.mean(itczArray_cyclic,axis=0),color='firebrick',transform=ccrs.PlateCarree())
    ax.fill_between(lon_cyclic, np.quantile(itczArray_cyclic,0,axis=0), np.quantile(itczArray_cyclic,1,axis=0),color='firebrick', alpha=0.5,transform=ccrs.PlateCarree())

    #ax.set_global()
    #plt.title(season)
    plt.tight_layout()

def plotITCZ2(t,itczArray,lons,UV=False,precip=False,season='ANN',bounds=40):
    plt.figure(figsize=(8,3),dpi=600)
    ax = plt.axes(projection=ccrs.Robinson()) 
    ax.spines['geo'].set_edgecolor('black'); ax.coastlines()
    ax.set_extent((-180,180,-bounds,bounds),ccrs.PlateCarree())
    #
    if len(np.shape(precip)) >0:
        p=ax.pcolormesh(precip.lon,precip.lat,precip.sel(season=season).mean(dim='age'),
                        vmin=0,vmax=10,cmap='Blues',transform=ccrs.PlateCarree())
        plt.colorbar(p,orientation='horizontal',label=str(t)+'-0 ka'+season+' precip ('+precip.units+')',shrink=0.3,aspect=50,extend='max')
    #
    if len(np.shape(UV)) >0:
        U, V = np.meshgrid(UV[0].lon.data,UV[0].lat.data)
        ax.quiver(U,V,
                  UV[0].sel(season=season).mean(dim='age').data,
                  UV[1].sel(season=season).mean(dim='age').data,
                  transform=ccrs.PlateCarree(),scale=400,color='k',width=.002)
    #
    itczArray_cyclic,lon_cyclic = cutil.add_cyclic_point(itczArray[0],coord=lons,axis=1)
    ax.plot(lon_cyclic,np.mean(itczArray_cyclic,axis=0),color='grey',transform=ccrs.PlateCarree())
    ax.fill_between(lon_cyclic, np.quantile(itczArray_cyclic,0,axis=0), np.quantile(itczArray_cyclic,1,axis=0),color='grey', alpha=0.5,transform=ccrs.PlateCarree())
    #
    itczArray_cyclic,lon_cyclic = cutil.add_cyclic_point(itczArray[1],coord=lons,axis=1)
    ax.plot(lon_cyclic,np.mean(itczArray_cyclic,axis=0),color='firebrick',transform=ccrs.PlateCarree())
    ax.fill_between(lon_cyclic, np.quantile(itczArray_cyclic,0,axis=0), np.quantile(itczArray_cyclic,1,axis=0),color='firebrick', alpha=0.5,transform=ccrs.PlateCarree())
    #ax.set_global()
    #plt.title(season)
    plt.tight_layout()


#%%Standard precip max
lat,agemin,agemax=30,0,22000
precip=model_data['PRECT'].sel(age=slice(agemax,agemin)).sel(lat=slice(-lat,lat))
#
itcz_lat_precip={}
for season in ['ANN','JJA','DJF']:
    itcz_lat_precip[season]      = np.zeros((len(precip.age),len(precip.lon))); itcz_lat_precip[season][:]      = np.nan
    for i in range(len(precip.age)):
        precip_selected = precip.sel(season=season)[i,:,:].data
        itcz_lat_precip[season][i,:] = (precip.lat.data)[np.argmax(precip_selected,axis=0)]
for t in [18,6]:
    for szn in ['JJA','DJF']:
        itczage = []
        itczage.append(itcz[szn].sel(age=slice(0*1000+1000,0*1000-1000)))
        itczage.append(itcz[szn].sel(age=slice(t*1000+1000,t*1000-1000)))
        plotITCZ2(t,itczage,precip.lon,[model_data['U'],model_data['V']],model_data['PRECT'],szn)

#%% PHYDA
# ITCZ_max = latitude of expected latitudes using a weighting function of 
# an integer power N=10 of the area-weighted precipitation P integrated over
# the tropics. See Adam et al 2016: https://doi.org/10.1175/JCLI-D-15-0512.1
#
#Calculate for each season and lon
lat,agemin,agemax=30,0,22000
precip=model_data['PRECT'].sel(age=slice(agemax,agemin)).sel(lat=slice(-lat,lat))
itcz = {}
for szn in ['JJA','DJF']: 
    ITCZ_max_byLon = [] 
    for lon in precip.lon:
        pr       = precip.sel(season=szn).sel(lon=lon)
        coslat   = np.array([np.cos(np.radians(x)) for x in pr.lat])
        coslatPr = (coslat*pr)**10
        ITCZ_max_byLon.append((pr.lat*coslatPr).sum(dim='lat')/coslatPr.sum(dim='lat'))
    itcz[szn] = xr.concat(ITCZ_max_byLon,dim='lon').transpose()
for t in [18,6]:
    for szn in ['JJA','DJF']:
        itczage = []
        itczage.append(itcz[szn].sel(age=slice(0*1000+1000,0*1000-1000)))
        itczage.append(itcz[szn].sel(age=slice(t*1000+1000,t*1000-1000)))
        plotITCZ2(t,itczage,precip.lon,[model_data['U'],model_data['V']],model_data['PRECT'],szn)

#%%Weighted mean lat
lat,agemin,agemax=30,0,22000
precip=model_data['PRECT'].sel(age=slice(agemax,agemin)).sel(lat=slice(-lat,lat))
itcz = precip.lat.weighted(precip*np.cos(np.radians(precip.lat))).mean(dim='lat')
#
for t in [18,6]:
    for szn in ['JJA','DJF']: 
        itczage = []
        itczage.append(itcz.sel(season=szn).sel(age=slice(0*1000+1000,0*1000-1000)))
        itczage.append(itcz.sel(season=szn).sel(age=slice(t*1000+1000,t*1000-1000)))
        plotITCZ2(t,itczage,precip.lon,[model_data['U'],model_data['V']],model_data['PRECT'],szn)

#%%Convergence
#%%Standard convergence

lat,agemin,agemax=30,0,22000
v_surface=model_data['V'].sel(age=slice(agemax,agemin)).sel(lat=slice(-lat,lat))
u_surface=model_data['U'].sel(age=slice(agemax,agemin)).sel(lat=slice(-lat,lat))
precip=model_data['PRECT'].sel(age=slice(agemax,agemin)).sel(lat=slice(-lat,lat))

lat_central = (v_surface.lat.data[1:] + v_surface.lat.data[:-1])/2
itcz = {}
for season in ['ANN','JJA','DJF']:
    itcz[season] = np.zeros((len(v_surface.age),len(v_surface.lon))); itcz[season][:] = np.nan
    v_surface_diff = v_surface.sel(season=season)[:,1:,:].data -  v_surface.sel(season=season)[:,:-1,:].data
    itcz[season] = lat_central[np.argmin(v_surface_diff,axis=1)]
#
for t in [18,6]:
    for szn in ['JJA','DJF']:
        itczage = []
        itczage.append(itcz[szn][np.where((precip.age.data>=0*1000-1000) & (precip.age<=0*1000+1000))[0],:])
        itczage.append(itcz[szn][np.where((precip.age.data>=t*1000-1000) & (precip.age<=t*1000+1000))[0],:])
        plotITCZ2(t,itczage,precip.lon,[model_data['U'],model_data['V']],model_data['PRECT'],szn)


#%%UV convergence

agemin=0
agemax=22000
lat=30
v_surface=model_data['V'].sel(age=slice(agemax,agemin)).sel(lat=slice(-lat,lat))
u_surface=model_data['U'].sel(age=slice(agemax,agemin)).sel(lat=slice(-lat,lat))
precip=model_data['PRECT'].sel(age=slice(agemax,agemin)).sel(lat=slice(-lat,lat))

vals = v_surface

lat_central = (v_surface.lat.data[1:] + v_surface.lat.data[:-1])/2
lon_central = (v_surface.lon.data[1:] + v_surface.lon.data[:-1])/2

itcz_lat_convergence = {}
for season in ['ANN','JJA','DJF']:
    itcz_lat_convergence[season] = np.zeros((len(v_surface.age),len(lon_central))); itcz_lat_convergence[season][:] = np.nan
    v_surface_diff = v_surface.sel(season=season)[:,1:,:].data - v_surface.sel(season=season)[:,:-1,:].data
    u_surface_diff = u_surface.sel(season=season)[:,:,1:].data - u_surface.sel(season=season)[:,:,:-1].data
    v_surface_diff = (v_surface_diff[:,:,1:]+v_surface_diff[:,:,:-1])/2
    u_surface_diff = (u_surface_diff[:,1:,:]+u_surface_diff[:,:-1,:])/2
    surface_diff = v_surface_diff + np.abs(u_surface_diff)
    itcz_lat_convergence[season] = lat_central[np.argmin(surface_diff,axis=1)]
    
plotITCZ(itcz_lat_convergence,lon_central,[model_data['U'],model_data['V']],model_data['PRECT'],'JJA')
plotITCZ(itcz_lat_convergence,lon_central,[model_data['U'],model_data['V']],model_data['PRECT'],'DJF')

#%%






z=model_data['V500'].sel(lat=slice(-35,35))
dydx = z.diff(dim='lat')/np.median(np.diff(z.lat))

zzz = np.argmin(np.abs(dydx).data,axis=2)

np.argmin(np.abs(dydx[:,0,:,:].data))

#%%


#%%
n_time=len(var_regrid.age)
n_lon=len(var_regrid.lon)
lat=var_regrid.lat.data
# For each longitude, find the latitude in the topics with the maximum convergence.
lat_central = (lat[1:] + lat[:-1])/2
ind_tropics_v2 = np.where((lat_central >= -30) & (lat_central <= 30))[0]
lat_central_selected = lat_central[ind_tropics_v2]
itcz_lat_convergence = {}

#%%
for season in ['ANN']:#,'JJA','DJF']:
    itcz_lat_convergence[season] = np.zeros((n_time,n_lon)); itcz_lat_convergence[season][:] = np.nan
    v_surface_diff = var_regrid.sel(season=season)[:,:-1,:].data - var_regrid.sel(season=season)[:,1:,:].data
    for i in range(n_time):
        v_surface_diff_selected = v_surface_diff[i,ind_tropics_v2,:]
        for j in range(np.shape(v_surface_diff_selected)[1]):
            try: itcz_lat_convergence[season][i,j] = lat_central_selected[np.nanargmax(v_surface_diff_selected2[:,j],axis=0)]
            except: itcz_lat_convergence[season][i,j] = np.NaN#lat_central_selected[np.nanargmax(v_surface_diff_selected[:,j],axis=0)]
     
#%%
#        itcz_lat_convergence[season][i,:] = lat_central_selected[np.argmax(v_surface_diff_selected,axis=0)]
#%%
lat = 35
u_surface=model_data['U'].sel(lat=slice(-lat,lat))
v_surface=model_data['V'].sel(lat=slice(-lat,lat))
uv_surface=model_data['UV'].sel(lat=slice(-lat,lat))

precip=model_data['PRECT']

#For each longitude, find the latitude in the topics with the maximum convergence.
lat_central = (v_surface.lat[1:] + v_surface.lat[:-1])/2

itcz_lat_convergence = {}
for season in ['ANN']:
    itcz_lat_convergence[season] = np.zeros((len(v_surface.age),len(v_surface.lon))); itcz_lat_convergence[season][:] = np.nan
    v_surface_diff = v_surface.sel(season=season)[:,:-1,:].data - v_surface.sel(season=season)[:,1:,:].data
    u_surface_diff = u_surface.sel(season=season)[:,:-1,:].data - u_surface.sel(season=season)[:,1:,:].data
    uv_surface_mean= (uv_surface.sel(season=season)[:,:-1,:].data + uv_surface.sel(season=season)[:,1:,:].data)/2
    u_surface_diffsign = np.sign(u_surface.sel(season=season)[:,:-1,:].data) - np.sign(u_surface.sel(season=season)[:,1:,:].data)
    v_surface_diffsign = np.sign(v_surface.sel(season=season)[:,:-1,:].data) - np.sign(v_surface.sel(season=season)[:,1:,:].data)
    surface_diff = np.where((v_surface_diffsign<0),v_surface_diff, np.NaN)
    for i in range(len(v_surface.age)):
        for j in range(len(v_surface.lon)):
            try: itcz_lat_convergence[season][i,j] = lat_central[np.argmax(surface_diff[i,:,j])].data
            except:itcz_lat_convergence[season][i,j] = np.NaN



#%%
for season in ['ANN','JJA','DJF']:
    v_surface_diffsign = np.sign(v_surface.sel(season=season)[:,:-1,:].data) - np.sign(v_surface.sel(season=season)[:,1:,:].data)
    v_surface_diff2 = np.where(v_surface_diffsign>0,v_surface_diff, np.NaN)
    for i in range(n_time):
        v_surface_diff_selected2 = v_surface_diff2[i,ind_tropics_v2,:]
        for j in range(np.shape(v_surface_diff_selected)[1]):
            try: itcz_lat_convergence[season][i,j] = lat_central_selected[np.nanargmax(v_surface_diff_selected2[:,j],axis=0)]
            except: itcz_lat_convergence[season][i,j] = np.NaN#lat_central_selected[np.nanargmax(v_surface_diff_selected[:,j],axis=0)]
     


#%%



 #%%
plt.figure(figsize=(12,7))
ax1 = plt.subplot2grid((1,1),(0,0),projection=ccrs.Robinson()); ax1.set_global()

itcz_precip_djf_cyclic,lon_cyclic = cutil.add_cyclic_point(itcz_lat_convergence['DJF'],coord=precip.lon)
for szn in ['DJF','JJA']:
    if   szn == 'DJF': c='tab:red'
    elif szn == 'JJA': c='tab:blue'
    itcz_precip_cyclic,lon_cyclic = cutil.add_cyclic_point(itcz_lat_convergence[szn],coord=precip.lon)
    plt.fill_between(lon_cyclic,np.percentile(itcz_precip_cyclic,40,axis=0),np.percentile(itcz_precip_cyclic,60,axis=0),color=c,alpha=0.5,transform=ccrs.PlateCarree())
    plt.fill_between(lon_cyclic,np.percentile(itcz_precip_cyclic,20,axis=0),np.percentile(itcz_precip_cyclic,80,axis=0),color=c,alpha=0.4,transform=ccrs.PlateCarree())
    plt.fill_between(lon_cyclic,np.percentile(itcz_precip_cyclic,0,axis=0),np.percentile(itcz_precip_cyclic,100,axis=0),color=c,alpha=0.3,transform=ccrs.PlateCarree())
ax1.coastlines()
ax1.add_feature(cfeature.LAKES,facecolor='none',edgecolor='k')
ax1.gridlines(color='k',linewidth=1,linestyle=(0,(1,5)))
plt.title('Maximum ranges of JJA and DJF decadal-mean ITCZ position\nduring 0-12 ka, based on latitude of max convergence',fontsize=16)
plt.show()
