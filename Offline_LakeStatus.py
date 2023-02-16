#==============================================================================
# Script for caclulating lake status prior and climate indicies for paleoDA
# Meant to be run offline before performing DA.
#    author: Chris Hancock
#    date  : 2/15/2023
#==============================================================================

#setup file locations
model = 'DAMP12kTraCE' 
data_dir="/Users/chrishancock/Library/CloudStorage/OneDrive-NorthernArizonaUniversity/Research/Manuscript/DAMP12k/Data/models/"+model+"/"

#Load Packages
import xarray as xr
import numpy as np
import scipy 
import pyet                               # Package for calciulating potential ET
import matplotlib.pyplot   as plt         # Packages for making figures
import cartopy.crs         as ccrs        # Packages for mapping in python

#Save Created Files? T/F
save = False
#%% Load TraCE data and covert to desired units
minAge  = 0
maxAge  = 12000
binsize = 10
binyrs  = [*range(minAge,maxAge+1,binsize)]
binvec  = [*range(minAge-int(binsize/2),maxAge+int(binsize/2)+1,binsize)] 
binvec = [np.NaN] #set as [np.NaN] to retain native resolution

def savedata(dataarray,var_text,data_dir=data_dir,save=save):
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
#%%
#########################################
varkey = ['TREFHT','PS','FSDS','FSNS','FLNS','RELHUM','V','U','Z3','PRECT','QFLX','QOVER','SNOWICE'] 
heights = [250,500,800]
#
model_data={}

for var_name in varkey:
    #Load Data and create season dimension
    if var_name in ['QOVER','SNOWICE']: 
        handle_model=xr.open_dataset(data_dir+'original/trace.01-36.22000BP.clm2.'+var_name+'.22000BP_decavg_400BCE.nc' ,decode_times=False)
        model_data[var_name] =  [handle_model[var_name].expand_dims(dim='season', axis=1,).assign_coords({"season": ("season", ['ANN'])})]
    else:                   
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
    if var_name == 'TREFHT': #Convert K to degC
        model_data[var_name].data = model_data[var_name].data-273.15
        model_data[var_name].rename('tas')
        model_data[var_name].attrs['units'] ='degC'
    elif var_name == 'PS': #Convert Pa to kPa
        model_data[var_name].data = model_data[var_name].data/1000
        model_data[var_name].attrs['units'] ='kPa'
    elif var_name in ['FLNS','FSNS','FSNSOI','FSDS']: #Convert W/m2 to MJ/m2/day.
        model_data[var_name].data = model_data[var_name].data*0.0864
        model_data[var_name].attrs['units'] = 'MJ/(m2*day)'
    elif var_name == 'Z3': #data already in m above sea level
        for height in heights:
            model_data[var_name+str(height)] = model_data[var_name][:,:,np.argmin(abs(model_data[var_name].lev.data-height)),:,:].rename(var_name+str(height))
            model_data[var_name+str(height)].attrs['long_name'] = str(height)+' '+model_data[var_name].attrs['long_name']
        model_data[var_name] = model_data[var_name][:,:,-1,:,:]
        model_data[var_name].attrs['long_name'] = 'Surface '+model_data[var_name].attrs['long_name']
    elif var_name  in ['U','V']: #data already in m/s 
        for height in heights:
            var_new = var_name+str(height)
            model_data[var_name+str(height)] = model_data[var_name][:,:,np.argmin(abs(model_data[var_name].lev.data-height)),:,:].rename(var_name+str(height))
            model_data[var_name+str(height)].attrs['long_name'] = str(height)+' '+model_data[var_name].attrs['long_name']
        model_data[var_name] = model_data[var_name][:,:,-1,:,:]
        model_data[var_name].attrs['long_name'] = 'Surface '+model_data[var_name].attrs['long_name']
    elif var_name == 'RELHUM':  #get surface values
        model_data[var_name] = model_data[var_name][:,:,-1,:,:]
        model_data[var_name].attrs['long_name'] = 'Surface '+model_data[var_name].attrs['long_name']
    elif  var_name == 'PRECT': #Convert m/s to mm/day
        model_data[var_name].data = model_data[var_name].data*1000*60*60*24
        model_data[var_name].rename('precip')
        model_data[var_name].attrs['units'] ='mm/day'
    elif  var_name == 'QFLX': #Convert m/s to mm/day
        model_data[var_name].data = model_data[var_name].data*60*60*24
        model_data[var_name].attrs['units'] ='mm/day'
    elif var_name == 'QOVER': #Convert mm/s to mm/day
        model_data[var_name].data = model_data[var_name].data*60*60*24
        model_data[var_name].attrs['units'] ='mm/day'
    #
    if var_name in ['Z3','V','U']:
        for height in heights:
            var_text = var_name+str(height)
            savedata(model_data[var_text],var_text,save=save)
        var_text = var_name
    elif var_name == 'TREFHT': var_text = 'tas'
    elif var_name == 'PRECT': var_text = 'precip'
    else: var_text = var_name
    savedata(model_data[var_name],var_text,save=save)

#Calculate % of precipitation within each season
var_name = 'precip_%ofANN'
model_data[var_name] = (model_data['PRECT']*1).rename(var_name)
for szn in model_data[var_name].season:
    i = int(np.where(model_data['PRECT'].season.data==szn)[0])
    model_data[var_name][:,i,:,:] /= model_data['PRECT'][:,1:,:,:].sum(dim='season')
    if i == 0: model_data[var_name][:,i,:,:]*=400
    else:      model_data[var_name][:,i,:,:]*=100
model_data[var_name].attrs['units']     = '% of annual'
model_data[var_name].attrs['long_name'] = 'precip (seasonal/annual)' 
savedata(model_data[var_name],var_name,save=save)
#########################################

#Calculate wind speed from U and V vectors
var_name = 'UV'
model_data[var_name] = np.sqrt(model_data['V']**2+model_data['U']**2).rename(var_name)
model_data[var_name].attrs['units']     = model_data['V'].attrs['units']
model_data[var_name].attrs['long_name'] =  'surface wind spped (sqrt(U^2+V^2))' 
savedata(model_data[var_name],var_name,save=save)
#########################################

#Calculate P-E
var_name = 'P-E'
model_data[var_name] = (model_data['PRECT']-model_data['QFLX']).rename(var_name)
model_data[var_name].attrs['units']     = model_data['PRECT'].attrs['units']
model_data[var_name].attrs['long_name'] =  'Precip - Evap' 
savedata(model_data[var_name],var_name,save=save)
#########################################

#Calculate P-E
var_name = 'P-E'
model_data[var_name] = (model_data['PRECT']-model_data['QFLX']).rename(var_name)
model_data[var_name].attrs['units']     = model_data['PRECT'].attrs['units']
model_data[var_name].attrs['long_name'] =  'Precip - Evap' 
savedata(model_data[var_name],var_name,save=save)
#########################################

#Calculate net radiation from FSNS and FLNS 
#Can't use available #SRFRAD file becase calculated incorrectly https://bb.cgd.ucar.edu/cesm/threads/srfrad.3450/
var_name = 'SRFRAD'
model_data[var_name] = (model_data['FSNS']-model_data['FLNS']).rename(var_name)
#model_data[var_name] = ((model_data['FSDS']*0.95)-(model_data['FLNS']*1)).rename(var_name)
model_data[var_name].attrs['units']     = model_data['FSNS'].attrs['units']
model_data[var_name].attrs['long_name'] =  'Net radiative flux at surface (FSNS-FLNS)' 
savedata(model_data[var_name],var_name,save=save)
#########################################

# Calculate Lake Status and associated variables

#Calculate potential evaporation for lake psm
model_data['pet_penman'] = pyet.penman(tmean=    model_data['TREFHT'][:,model_data['TREFHT'].season=='ANN',:,:], #degC
                                       wind =    model_data['UV'][:,model_data['UV'].season=='ANN',:,:],         #m/s
                                       rn=       model_data['SRFRAD'][:,model_data['SRFRAD'].season=='ANN',:,:], #MJ m-2 d-1 
                                       rh=       model_data['RELHUM'][:,model_data['RELHUM'].season=='ANN',:,:], #%
                                       pressure= model_data['PS'][:,model_data['PS'].season=='ANN',:,:])     #kPa
 
model_data['pet_priestley_taylor'] = pyet.priestley_taylor(tmean=    model_data['TREFHT'][:,model_data['TREFHT'].season=='ANN',:,:],
                                                           rn=       model_data['SRFRAD'][:,model_data['SRFRAD'].season=='ANN',:,:],
                                                           rh=       model_data['RELHUM'][:,model_data['RELHUM'].season=='ANN',:,:],
                                                           pressure= model_data['PS'][:,model_data['PS'].season=='ANN',:,:])
#
for var_name in ['pet_penman','pet_priestley_taylor']:
    model_data[var_name].attrs['units'] = "mm/day"
    if var_name == 'pet_penman':             model_data[var_name].attrs['long_name'] =  'Potential evapotranspiration (penman_natural_1948)' 
    elif var_name == 'pet_priestley_taylor': model_data[var_name].attrs['long_name'] =  'Potential evapotranspiration (priestley_assessment_1972)' 
    savedata(model_data[var_name],var_name,save=save)
#########################################

#Calculate Lake Status and El-Pl
pet_method,var_name  = 'pet_priestley_taylor','ElminusPl'
model_data[var_name] = model_data[pet_method] - model_data['PRECT'] 
#Add metadata
model_data[var_name].attrs['units']     = model_data['PRECT'].attrs['units']
model_data[var_name].attrs['long_name'] = 'Net lake Evap ('+model_data[pet_method].attrs['long_name']+') minus Precip' 
savedata(model_data[var_name],var_name,save=save)
#########################################

#QOVER lats are rounded versions. replace so dimmensions match
for var_name in ['QOVER','SNOWICE']:
    model_data[var_name] = model_data[var_name].assign_coords(lat=model_data['ElminusPl'].lat.data)
    model_data[var_name] = model_data[var_name].assign_coords(lon=model_data['ElminusPl'].lon.data)
    model_data[var_name] = model_data[var_name].assign_coords(age=model_data['ElminusPl'].age.data)
#########################################
#%%Caculate Lake Status based on ranking of Q/(PET-P) with negative values converted to > max
#
def lake2percentile(inarray):
    for szni in range(len(inarray.season)):
        for lati in range(len(inarray.lat)):
            for loni in range(len(inarray.lon)): 
                if np.sum(np.isfinite(inarray[:,szni,lati,loni]))>0:
                    ranks = scipy.stats.mstats.rankdata(np.ma.masked_invalid(inarray[:,szni,lati,loni]))                                        #Additional step to mask nan values
                    ranks[ranks == 0] = np.nan
                    if np.sum(np.isfinite(ranks)) > 1:
                        ranks-=1
                        ranks/= np.sum(np.isfinite(ranks))-1
                    else: ranks*=np.NaN
                    inarray[:,szni,lati,loni]=ranks
    return(inarray)

var_name = 'LakeStatus'
model_data[var_name] = np.divide(model_data['QOVER'],model_data['ElminusPl']).rename(var_name)
#(model_data[var_name].max(dim='age') - model_data[var_name].min(dim='age'))
#Order PET>P
positive = lake2percentile(model_data[var_name].where(model_data[var_name]>0))
#Order P>PET
negative = lake2percentile(model_data[var_name].where(model_data[var_name]<0))
negative+= 1.1
runoff0  = model_data[var_name].where(model_data[var_name]==0)
runoff0  = runoff0.where(np.isnan(model_data['SNOWICE'].where(model_data['SNOWICE']>1))) 
#Combine
vals = np.where(np.isfinite(positive),positive,negative)
vals = np.where(np.isfinite(vals),vals,runoff0)
#Reorder
model_data[var_name].data=vals
model_data[var_name] = model_data[var_name].where(model_data[var_name].sum(dim='age')!=0)
model_data[var_name] = lake2percentile(model_data[var_name])
#Metadata
model_data[var_name].attrs['units']     = 'percentile'
model_data[var_name].attrs['long_name'] = 'Lake Status (Runoff/(LakeEvap-LakePrecip))'

savedata(model_data[var_name],var_name,save=False)

#%%Visialize Lake Status
dataarray=model_data[var_name].sel(season='ANN')
age1=6
age2=0.5
plt.figure(dpi=400)
ax = plt.axes(projection=ccrs.Robinson()) 
p=ax.pcolormesh(dataarray.lon,dataarray.lat,
                dataarray.sel(age=slice(age1*1000+500,age1*1000-500)).mean(dim='age')-dataarray.sel(age=slice(age2*1000+500,age2*1000-500)).mean(dim='age'),
                vmin=-1,vmax=1,cmap='BrBG',transform=ccrs.PlateCarree())
ax.spines['geo'].set_edgecolor('black'); ax.set_global(); ax.coastlines()
plt.title(str(age1)+'-'+str(age2)+' ka LakeStatus (percentile)')
plt.colorbar(p,orientation='horizontal')
plt.show()
#%%
t=22000
thrsh = 1
vals = model_data['ElminusPl'].where(model_data['QOVER']==0)#.where(model_data['SNOWICE']>0)
vals=vals.where(model_data['ElminusPl']<0)
vals=vals.where(model_data['SNOWICE']<thrsh)
vals=vals.sel(season='ANN').sel(age=slice(t,0)).count(dim='age')*10
plt.figure(dpi=400)
ax = plt.axes(projection=ccrs.Robinson()) 
p=ax.pcolormesh(runoff0.lon,runoff0.lat,vals.where(vals>0),vmin=0,
                cmap='Blues',transform=ccrs.PlateCarree())
ax.spines['geo'].set_edgecolor('black'); ax.set_global(); ax.coastlines()
plt.title('snowice<'+str(thrsh)+'kg/m2 and QOVER=0 but P>PET \n('+str(np.sum(vals).data)+' since '+str(int(t/1000))+'ka)')
plt.colorbar(p,orientation='horizontal')
plt.show()


#%% Original version for lake status
var_name='LakeStatus'
#Calculate
model_data[var_name] = np.divide(model_data['QOVER'],model_data['ElminusPl']).rename(var_name).where(model_data['ElminusPl']>0).rename(var_name)
#Replace nan values caused by Pl > El with max value
model_data[var_name].data = np.where(model_data['ElminusPl']<=0, model_data[var_name].max(dim='age'), model_data[var_name])
#If runoff is 0, lake status is 0
model_data[var_name].data = np.where(model_data['QOVER']==0, 0, model_data['LakeStatus'])
#If sum of timeseries (runoff = 0 and Pl > El cause nan)
model_data[var_name].data = np.where(model_data[var_name].sum(dim='age')==0, np.NaN, model_data[var_name])
#Add metadata
model_data[var_name].attrs['units']     = 'ratio'
model_data[var_name].attrs['long_name'] = 'Lake Status (Runoff/(LakeEvap-LakePrecip))'
savedata(model_data[var_name],var_name,save=False)
#########################################
vals = model_data[var_name] 
counts, bins = np.histogram(vals.data[np.isfinite(vals.data)])
plt.stairs(counts, bins,color='indigo',fill=True)




































