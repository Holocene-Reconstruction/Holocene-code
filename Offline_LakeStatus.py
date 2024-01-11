#==============================================================================
# Script for caclulating lake status prior and climate indicies for paleoDA
# Meant to be run offline before performing DA.
#    author: Chris Hancock
#    date  : 2/15/2023
#==============================================================================

#Load Packages
import cartopy.crs         as ccrs        # Packages for mapping in python
import matplotlib.pyplot   as plt         # Packages for making figures
import metpy.calc as mpcalc
import numpy as np
import os
import pyet                               # Package for calciulating potential ET
import xarray as xr
import xesmf as xe

wd = '/Users/chrishancock/Library/CloudStorage/OneDrive-NorthernArizonaUniversity/Research/Manuscript/DAMP21k/' #changed
os.chdir(wd+'Holocene-code') #changed
import da_utils
import da_utils_lakestatus as da_utils_ls

#setup file locations
model = 'DAMP_TraCE' 
#model = 'DAMP_HadCM' 
wd='/Users/chrishancock/Library/CloudStorage/OneDrive-NorthernArizonaUniversity/Research/Manuscript/DAMP21k/Data/models/'

#Save Created Files? T/F
save    = False
overide = False #True will replace files in 'processed' model data folder (takes longer)
regrid  = False
print('save = '+str(save))
heights = [200]

#Set up standardized names
#Format = name, conversion multiply, conversion shift, new units
varkey = {}
varkey['DAMP_TraCE'] = {
    'tas':   ['TREFHT',-273.15,'degC'],            #Convert K to degC
    'precip':['PRECT',(1000*60*60*24),'mm/day'],   #Convert m/s to mm/day  
    'evap':  ['QFLX',(60*60*24),'mm/day'],         #Convert kg/m2/s to mm/day  
    'press': ['PS',0.001,'kPa'],                   #Convert Pa to kPa 
    'slp':   ['PSL',0.001,'kPa'],                  #Convert Pa to kPa cfrA
    'U':     ['U',None,'m/s'],                     #Units already m/s
    'V':     ['V',None,'m/s'],                     #Units already m/s
    'U200':  None,                                 #Calculate from U file
    'V200':  None,                                 #Calculate from V file
    'netrad':None,                                 #Not Available
    'downSW':['FSDS',0.0864,'MJ/(m2*day)'],        #Convert W/m2 to MJ/(m2*day)
    'toaSW': None,                                 #Don't need if have netSW
    'netSW': ['FSNS',0.0864,'MJ/(m2*day)'],        #Convert W/m2 to MJ/(m2*day)
    'netLW': ['FLNS',0.0864,'MJ/(m2*day)'],        #Convert W/m2 to MJ/(m2*day)
    'relhum':['RELHUM',None,'percent'],            #Units already %
    'spehum':None,                                 #Don't need if have relhum
    'runoff':['QOVER',(60*60*24),'mm/day'],        #Convert mm/s to mm/day
    'snow':  ['SNOWICE',None,'kg/m2'],             #Units already in kg/m2
    'land': ['LANDFRAC',None,'percent'],           #nits already % (0-1)
    'elev':  None,                                 #Don't need with land
    }
varkey['DAMP_HadCM'] = {
    'tas':   ['temp_mm_1_5m',-273.15,'degC'],      #Convert K to degC
    'precip':['precip_mm_srf',(60*60*24),'mm/day'],#Convert kg/m2/s to mm/day 
    'evap':  ['totalEvap_mm_srf',None,'mm/day'],   #Units already mm/day
    'press': ['p_mm_srf',0.001,'kPa'],             #Convert Pa to kPa 
    'slp':   ['p_mm_msl',0.001,'kPa'],             #Convert Pa to kPa 
    'U':     ['u_mm_10m',None,'m/s'],              #Units already m/s
    'V':     ['v_mm_10m',None,'m/s'],              #Units already m/s
    'U200':  ['u_mm_p_200',None,'m/s'],            #Units already m/s 
    'V200':  ['v_mm_p_200',None,'m/s'],            #Units already m/sdownSol_mm_TOA
    'netrad':None,                                 #TODO: Not Available 
    'downSW':['downSol_Seaice_mm_s3_srf',0.0864,'MJ/(m2*day)'],#W/m2 to MJ/(m2*day)
    'toaSW': ['downSol_mm_TOA',0.0864,'MJ/(m2*day)'],          #W/m2 to MJ/(m2*day)
    'netSW': None,                                 #Not Available
    'netLW': None,                                 #Not Available
    'relhum':None,                                 #Not Available
    'spehum':['q_mm_1_5m',None,'kg/kg'],           #Units alread kg/kg
    'runoff':None,                                 #TODO: Not Available
    'snow':  ['snowdepth_mm_srf',None,'kg/m2'],    #Units already in kg/m2
    'land':  None,                                 #Don't need with elev
    'elev':  ['ht_mm_srf',None,'m'],               #Units already in m
    }

if (varkey['DAMP_HadCM'].keys()==varkey['DAMP_TraCE'].keys())==False: print('Warning: HadCM and TraCE variable lists are not the same')

# A function for saving and plotting the data
def savedata(dataarray,var,wd=wd,save=save,
             regrid=False,
             plot=True,
             mask=False,
             maskthresh=0):
    #
    agerange = [str(round(np.max(dataarray.age.data/1000))),str(round(np.min(dataarray.age.data/1000)))]
    if regrid:
        #set new lat/lon grid
        dataarray=model_data[var]
        lat_regrid = np.arange(-88.59375,90,2.8125)
        lon_regrid = np.arange(0,360,3.75) 
        data_format = xr.Dataset(
            {'lat': (['lat'],lat_regrid,{'units':'degrees_north'}),
             'lon': (['lon'],lon_regrid,{'units':'degrees_east'})})
        #regrid the data
        regridder = xe.Regridder(dataarray.to_dataset(),data_format,'conservative_normed',periodic=True)
        dataarray_regrid = regridder(dataarray.to_dataset(),keep_attrs=True)
        dataarray_regrid = dataarray_regrid.to_array()[0]
        dataarray_regrid.attrs['units'] = dataarray.attrs['units'] 
        dataarray_regrid.attrs['long_name'] = dataarray.attrs['long_name'] 
        dataarray = dataarray_regrid.rename(var)
        if type(mask)==xr.core.dataarray.DataArray: 
            #mask_regrid = regridder(mask.to_dataset(),keep_attrs=True).to_array()[0]
            dataarray.data= np.where(mask>maskthresh,dataarray.data,np.NaN)
            print(dataarray[0,0,30:40,30:40])
        if save: dataarray.to_netcdf(wd+'processed_model_data/'+model+'_regrid.'+var+'.'+agerange[0]+'ka_decavg_'+agerange[1]+'ka.nc')
    else:            
        if save: dataarray.to_netcdf(wd+'processed_model_data/'+model+'.'+var+'.'+agerange[0]+'ka_decavg_'+agerange[1]+'ka.nc')
    if plot: 
        ax = plt.axes(projection=ccrs.Robinson()) 
        p=ax.pcolormesh(dataarray.lon,dataarray.lat,dataarray.isel(season=np.where(dataarray.season=='ANN')[0]).mean(dim=['age','season']),transform=ccrs.PlateCarree())
        ax.spines['geo'].set_edgecolor('black'); ax.set_global(); ax.coastlines()
        plt.title(var+' ('+dataarray.units+')')
        plt.colorbar(p,orientation='horizontal')
        plt.show()
    print("###################")
    print(var+"/"+dataarray.attrs['long_name']+" ("+dataarray.units+")")
    print(np.shape(dataarray))
    print("min:"+str(int(np.min(dataarray)))+"; max: "+str(int(np.max(dataarray)))+"; mean:"+str(float(np.mean(dataarray))))
    #

#Load model data and covert to desired units
minAge,maxAge,binsize = 0,22000,10
binyrs  = [*range(minAge,maxAge+1,binsize)]
binvec  = [*range(minAge-int(binsize/2),maxAge+int(binsize/2)+1,binsize)] 
binvec  = False #set as False to retain native resolution

#%%Load original model data
model_data={}
print(model)
for var in list(varkey[model].keys()):
    #Load already processed data if available
    if overide == False:
        filenames = [fn for fn in os.listdir(wd+'processed_model_data/') if '.'+var+'.' in fn]
        filenames = [fn for fn in filenames if model in fn]
        filenames = [fn for fn in filenames if 'regrid' not in fn]
        if len(filenames) == 1: 
            filename = filenames[0]
            handle_model=xr.open_dataarray(wd+'processed_model_data/'+filename, decode_times=False)
            model_data[var] = handle_model.rename(var)
            handle_model.close
            continue
    #Get info about variable
    if  varkey[model][var] == None: 
        print('======================================================\nNo '+var+' in available files')
        continue 
    var_name,conversion,units = varkey[model][var]
    print('======================================================\n'+var+'('+var_name+')')
    #########################################
    #Load Data and create season dimension to combine the same variables
    #########################################
    wd_orig = wd+'original_model_data/'+model+'/'
    filenames = [fn for fn in os.listdir(wd_orig) if '.'+var_name+'.' in fn]
    if model == 'DAMP_TraCE': filename = [fn for fn in filenames if 'decavg_400BCE' in fn][0]
    if model == 'DAMP_HadCM': filename = [fn for fn in filenames if 'ANN' in fn][0]
    handle_model=xr.open_dataset(wd_orig+filename, decode_times=False)
    if var_name[1:7]=='_mm_p_': var_name=var_name[:6]
    model_data[var] =  [handle_model[var_name].expand_dims(dim='season',axis=1).assign_coords({"season": ("season", ['ANN'])})]
    for season in ['JJA','DJF','MAM','SON']:       
        try:
            filename = [fn for fn in filenames if season in fn][0]
            handle_model=xr.open_dataset(wd_orig+filename, decode_times=False)
            model_data[var].append(handle_model[var_name].expand_dims(dim='season',axis=1).assign_coords({"season": ("season", [season])}))
        except: continue
    model_data[var] = xr.concat(model_data[var],dim='season')
    handle_model.close
    #########################################
    #Standardize variables
    #########################################
    model_data[var] = model_data[var].rename(var)
    #Ages #Reshape time as needed #TraCE centered on 10s, HadCM on 5.5s
    if model == 'DAMP_TraCE': model_data[var]=model_data[var].assign_coords(time=model_data[var]['time']*-1000).rename({'time': 'age'})
    if model == 'DAMP_HadCM': model_data[var]=model_data[var].assign_coords(t=model_data[var]['t']*-1+5.5).rename({'t': 'age'})
    #Bin time as needed
    if binvec: #If choosing to bin time data to a courser resolution
        model_data[var] = model_data[var].groupby_bins("age",binvec).mean(dim='age').rename({'age_bins': 'time'})
        model_data[var]['age'] = binyrs  
    else: model_data[var]=model_data[var].sel(age=slice(maxAge,minAge))
    #Standarduze lat/lon values
    if 'latitude'   in model_data[var].dims: model_data[var] = model_data[var].rename({'latitude': 'lat', 'longitude': 'lon'}) #change HadCM
    if 'latitude_1' in model_data[var].dims: model_data[var] = model_data[var].rename({'latitude_1': 'lat', 'longitude_1': 'lon'}) #HadCM wind slightly different   
    #TraCE and clm and cam have slighlty different values for the same grid
    if 'clm2' in filename:
        model_data[var] = model_data[var].assign_coords(lat=model_data['tas'].lat.data)
        model_data[var] = model_data[var].assign_coords(lon=model_data['tas'].lon.data)
        model_data[var] = model_data[var].assign_coords(age=model_data['tas'].age.data)
    #Convert U/V vector of HadCM to same grid as other climate variables
    #if model == 'DAMP_HadCM': 
       #TODO 
    #Standardize units among variables (and for pyet calculations)
    print('-\nOriginal units = '+model_data[var].units+' (min/max = '+str(np.min(model_data[var]).data)+'/'+str(np.max(model_data[var]).data)+')')
    #TODO: negative total evap values for HadCM #True for both _s and non smoothed files in evap (but SW Asia vs Antarctic. just _s for precip. I think it has to do with smoothing 
    if var in ['precip','evap']:  model_data[var]=np.clip(model_data[var],0,None)
    if conversion != None:
        if var == 'tas': model_data[var].data += conversion
        else:            model_data[var].data *= conversion
    model_data[var].attrs['units'] = units
    print('New units = '+model_data[var].units+' (min/max = '+str(np.min(model_data[var]).data)+'/'+str(np.max(model_data[var]).data)+')')
    #########################################
    #Reshape variables as needed 
    #########################################
    print('-\nOriginal data shape = '+str(np.shape(model_data[var].data)))
    #Standardize as surface values
    v = [x for x in model_data[var].dims if x in ['ht','surface','msl','p','toa']]
    if len(v)>0:#HadCM
        if (len(model_data[var][v[0]])) == 1 :model_data[var] = model_data[var].squeeze(v)
    if 'lev' in model_data[var].dims:
        if (heights != None) & (var in ['U','V']):
            for h in heights: 
                hi = np.argmin(np.abs(model_data[var].lev.data-h))
                model_data[var+str(h)] = model_data[var][:,:,hi,:,:]
                print(var+str(h)+ ' calculated from '+var+ ' file')
        model_data[var] = model_data[var][:,:,-1,:,:]
    print('New data shape = '+str(np.shape(model_data[var].data)))
    #
    if 'long_name' not in model_data[var].attrs.keys():
        model_data[var].attrs['long_name'] =  var_name
    #Plot the data
    savedata(model_data[var],var,save=False) 

#%% Save winter values as annual for some variables (easier for DA)
var = 'U200djf'
model_data[var] = model_data['U200'][:,0:1]
model_data[var].data = model_data['U200'][:,2:3].data
savedata(model_data[var],var,save=False)

var = 'SLPdjf'
model_data[var] = model_data['slp'][:,0:1]
model_data[var].data = model_data['slp'][:,2:3].data
savedata(model_data[var],var,save=False)

#%% #Save these
for var in  ['tas','precip','evap','U','U200','V','V200','U200djf','slp','SLPdjf']:
    if overide:
        savedata(model_data[var],var,save=save) 
        if regrid: savedata(model_data[var],var,save=save,regrid=True) 

for var in ['land','elev','snow']:
    if overide:
        try: 
            savedata(model_data[var],var,save=save) 
            if regrid: savedata(model_data[var],var,save=save,regrid=True) 
        except: continue


#%%Calculate some secondary model data

#Calculate P-E
var = 'P-E'
model_data[var] = (model_data['precip']-model_data['evap']).rename(var)
model_data[var].attrs['units']     = model_data['precip'].attrs['units']
model_data[var].attrs['long_name'] =  'Precip - Evap' 
savedata(model_data[var],var,save=False)
#########################################

#Calculate % of precipitation within each season
var = 'precip_%ofANN'
model_data[var] = (model_data['precip']*1).rename(var)
for szn in model_data[var].season.data:
    i = int(np.where(model_data['precip'].season.data==szn)[0])
    model_data[var][:,i,:,:] /= model_data['precip'][:,1:,:,:].sum(dim='season')
    if i == 0: model_data[var][:,i,:,:]*=400
    else:      model_data[var][:,i,:,:]*=100
model_data[var].attrs['units']     = '% of annual'
model_data[var].attrs['long_name'] = 'precip (seasonal/annual)' 
savedata(model_data[var],var,save=False)
#########################################
#
#Calculate wind speed from U and V vectors
var = 'UV'
model_data[var] = np.sqrt(model_data['V']**2+model_data['U']**2).rename(var)
if model=='DAMP_HadCM':
    if model_data[var].lat.data != model_data['tas'].lat.data:#HadCM on a slighly different grid
        data_format = xr.Dataset(
            {'lat': (['lat'],model_data['tas'].lat.data,{'units':'degrees_north'}),
             'lon': (['lon'],model_data['tas'].lon.data,{'units':'degrees_east'})})
        #regrid the data
        regridder = xe.Regridder(model_data[var].to_dataset(),data_format,'conservative_normed',periodic=True)
        model_data[var]=regridder(model_data[var].to_dataset(),keep_attrs=True).to_array()[0]
model_data[var].attrs['units']     = model_data['V'].attrs['units']
model_data[var].attrs['long_name'] = 'Surface wind speed '
savedata(model_data[var],var,save=False)
#########################################

#%%Estimate RH from Specific Humidity (if needed):
var = 'relhum'
if varkey[model][var] == None:
    model_data[var] = mpcalc.relative_humidity_from_specific_humidity(
                pressure          = model_data['press'],
                temperature       = model_data['tas'],
                specific_humidity = model_data['spehum'])#.to('percent')
    #TODO: Some issues over ice sheets where it creates super saturated RH values
    model_data[var].data=np.where(model_data[var]>1,1,model_data[var])
    #convert to %
    model_data[var]*=100 
    #
    model_data[var].rename('relhum')
    model_data[var].attrs['units']     = 'percent'
    model_data[var].attrs['long_name'] =  'Relative Humidity (converted from specific)' 
    savedata(model_data[var],var,save=False)
#########################################


#%% Calcualte net radiation (if needed)

#########################################
#Function modified from mpcalc package (to not need datetime age format)
def calc_rad_long(rs, tmean, rh, ra, elevation, rso=None, 
                  a=1.35, b=-0.35, ea=None, kab=None):
    """Net longwave radiation [MJ m-2 d-1].

    Parameters
    ----------
    rs: float/pandas.Series/xarray.DataArray, required
        incoming solar radiation [MJ m-2 d-1]
    tmean: float/pandas.Series/xarray.DataArray, required
        average day temperature [°C]
    rh: float/pandas.Series/xarray.DataArray, optional (need one of ea or rh)
        mean daily relative humidity [%]
    ra: float/pandas.Series/xarray.DataArray, required
        Extraterrestrial daily radiation [MJ m-2 d-1]
    elevation: float/xarray.DataArray, optional (need one of elevation or rso)
        the site elevation [m]
    rso: float/pandas.Series/xarray.DataArray, optional (need one of elevation or rso)
        clear-sky solar radiation [MJ m-2 day-1]
    a: float, optional
        empirical coefficient for Net Long-Wave radiation [-]
    b: float, optional
        empirical coefficient for Net Long-Wave radiation [-]
    ea: float/pandas.Series/xarray.DataArray, optional  (need one of ea or rh)
        actual vapor pressure [kPa]
    kab: float, optional
        coefficient that can be derived from the as and bs coefficients of the
        Angstrom formula, where Kab = as + bs, and where Kab represents the
        fraction of extraterrestrial radiation reaching the earth on clear-sky
        days [-]

    Returns
    -------
    float/pandas.Series/xarray.DataArray, optional containing the calculated
        net longwave radiation

    Notes
    -----
    Modified from pyet to not need datatime age configuration
    Based on equation 39 in :cite:t:`allen_crop_1998`.

    """
    STEFAN_BOLTZMANN_DAY = 4.903 * 10 ** -9 
    if ea is None:   ea = pyet.calc_ea(tmean=tmean,rh=rh)
    if rso is None: rso =  (0.75 + (2 * 10 ** -5) * elevation) * ra #modified of calc_rso
    solar_rat = np.clip(rs / rso, 0.3, 1)
    tmp1 = STEFAN_BOLTZMANN_DAY * (tmean + 273.16) ** 4
    tmp2 = 0.34 - 0.14 * np.sqrt(ea)  # OK
    tmp3 = a * solar_rat + b  # OK
    tmp3 = np.clip(tmp3, 0.05, 1) 
    return tmp1 * tmp2 * tmp3

#########################################

var = 'netLW'
if varkey[model][var] == None:
    model_data['netLW'] = calc_rad_long(rs=model_data['downSW'],
                  tmean=model_data['tas'],
                  rh=model_data['relhum'],
                  ra=model_data['toaSW'],
                  elevation=model_data['elev']
                  )
    model_data[var].attrs['units']     = model_data['downSW'].units
    model_data[var].attrs['long_name'] = 'net longwave estimate'
    savedata(model_data[var],var,save=False)
#########################################

var = 'netSW'
if varkey[model][var] == None:
    model_data[var] = model_data['downSW']*0.9 #estimate  based on albedo of water
    #model_data[var].data = np.where(model_data['snow']>0,model_data['downSW']*0.15,model_data['downSW']*0.95)
    model_data[var].data[:,0] = np.mean(model_data[var].data[:,1:],axis=1)
    model_data[var].attrs['units']     = model_data['downSW'].units
    model_data[var].attrs['long_name'] = 'net shortwave estimate (downSW*0.95)'
    savedata(model_data[var],var,save=False)
#########################################
    
#Calculate net radiation from FSNS and FLNS 
#Can't use available #SRFRAD file becase calculated incorrectly https://bb.cgd.ucar.edu/cesm/threads/srfrad.3450/
var = 'netrad'
if varkey[model][var] == None:
    model_data[var] = (model_data['netSW'] - model_data['netLW']).rename(var)
    model_data[var].attrs['units']     = model_data['downSW'].attrs['units']
    model_data[var].attrs['long_name'] =  'Net radiative flux at surface (FSNS-FLNS)'  
    savedata(model_data[var],var,save=False)
#########################################

#%% Calculate PET

#TODO: wind vectors shifted off variable grid
model_data['pet_penman'] = pyet.penman(tmean=    model_data['tas'][:,model_data['tas'].season=='ANN',:,:],       #degC
                                       wind=     model_data['UV'][:,model_data['UV'].season=='ANN',:,:],         #m/s
                                       rn=       model_data['netrad'][:,model_data['netrad'].season=='ANN',:,:], #MJ m-2 d-1 
                                       rh=       model_data['relhum'][:,model_data['relhum'].season=='ANN',:,:], #%
                                       pressure= model_data['press'][:,model_data['press'].season=='ANN',:,:])   #kPa

model_data['pet_priestley_taylor'] = pyet.priestley_taylor(tmean=    model_data['tas'][:,model_data['tas'].season=='ANN',:,:],
                                                           rn=       model_data['netrad'][:,model_data['netrad'].season=='ANN',:,:],
                                                           rh=       model_data['relhum'][:,model_data['relhum'].season=='ANN',:,:],
                                                           pressure= model_data['press'][:,model_data['press'].season=='ANN',:,:])
    
for var in ['pet_priestley_taylor','pet_penman']:
    model_data[var].attrs['units'] = "mm/day"
    if var == 'pet_penman':           model_data[var].attrs['long_name'] =  'Potential evapotranspiration (penman_natural_1948)' 
    if var == 'pet_priestley_taylor': model_data[var].attrs['long_name'] =  'Potential evapotranspiration (priestley_assessment_1972)' 
    model_data[var].attrs['long_name']='Evap ('+var+')'
    try: savedata(model_data[var],var,save=False)
    except: print(var+' failure')
    

#%%#%% #Calculate numerator and denominator of Lake Status calcualtion
if model == 'DAMP_TraCE':   
    Q_method = 'runoff'
    var = 'runoff'
    savedata(model_data[var],var,save=save)
    if regrid: savedata(model_data[var],var,save=save,regrid=True)
elif model == 'DAMP_HadCM': Q_method = 'P/E' #'P-PET' #use runoff or P-E or P/E
pet_method = 'pet_priestley_taylor' #use pet_penman or pet_priestley_taylor
snowthresh = 500 #must be an integer to work (i.e. 300) #kg/m2
#########################################

#Calculate Lake Status and El-Pl
var  = 'ElminusPl'
model_data[var] = model_data[pet_method] - model_data['precip'][:,0:1,:,:]
#Add metadata
model_data[var].attrs['units']     = model_data['precip'].attrs['units']
model_data[var].attrs['long_name'] = 'Net lake Evap ('+model_data[pet_method].attrs['long_name']+') minus Precip' 
savedata(model_data[var],var,save=False)
#########################################

# Calculate Qestimate
var = 'Qest'
model_data[var] = (model_data[pet_method]*np.NaN).rename(var)
if Q_method == 'runoff': model_data[var].data = model_data['runoff'].data 
elif Q_method == 'P-E':    model_data[var].data = np.where(model_data['P-E']<0,0,model_data['P-E'])[:,0:1,:,:] #And set negative values as 0
elif Q_method == 'P/E':    model_data[var] = np.divide(model_data['precip'],model_data['evap'].where(model_data['evap']!=0))[:,0:1,:,:]
elif Q_method == 'P-PET':  model_data[var] = model_data['precip'][:,0:1,:,:] - model_data[pet_method] 
else: print("not a valid Q_method")
#Mask by ice sheets
#if isinstance(snowthresh, int): 
#    model_data[var].data = np.where(model_data['snow'][:,model_data['snow'].season=='ANN',:,:]>snowthresh,np.NaN,model_data[var])
#Mask by ice sheets (useful for if using P-E instead of runoff)
#if Q_method != 'runoff':
    #if   varkey[model]['land'] != None: model_data[var].data = np.where(model_data['land'][:,model_data['land'].season=='ANN',:,:]<=0,np.NaN,model_data[var])
    #elif varkey[model]['elev'] != None: model_data[var].data = np.where(model_data['elev'][:,model_data['elev'].season=='ANN',:,:]<=0,np.NaN,model_data[var])
model_data[var].attrs['units']     = model_data['precip'].attrs['units']
model_data[var].attrs['long_name'] = 'Estimated Q using '+Q_method
savedata(model_data[var],var,save=False)
#########################################    

#%%
# Calculate Lake Status
var='LakeStatus'
model_data[var] = np.divide(model_data['Qest'],model_data['ElminusPl']).rename(var)
model_data[var][:,0] = da_utils_ls.calcLakeStatus(runoff=model_data['Qest'][:,0],
                                           precip=model_data['precip'][:,0],
                                           levap=model_data[pet_method][:,0],
                                           Qmethod='runoff')

model_data[var].attrs['units']     = 'percentile'
model_data[var].attrs['long_name'] = 'Lake Status ('+Q_method+'/(LakeEvap-LakePrecip))'

#%%
# Calcualte land/snow mask for regridded data
snowthresh=snowthresh

m,var='TraCE','snow'
filenames = [fn for fn in os.listdir(wd+'processed_model_data/') if '.'+var+'.' in fn]
filenames = [fn for fn in filenames if m in fn]
filenames = [fn for fn in filenames if 'regrid' in fn]
print(len(filenames))
filename = filenames[0]
handle_model=xr.open_dataarray(wd+'processed_model_data/'+filename, decode_times=False)
print(np.shape(handle_model))
trace_snow = handle_model[:,0:1,:,:].rename(var)
handle_model.close
#
landmask = trace_snow*np.NaN
landmask[:] = 1
landmask.data=np.where((trace_snow<snowthresh) | (np.isnan(trace_snow)),1,np.NaN)
#

m,var='HadCM','snow'
filenames = [fn for fn in os.listdir(wd+'processed_model_data/') if '.'+var+'.' in fn]
filenames = [fn for fn in filenames if m in fn]
filenames = [fn for fn in filenames if 'regrid' in fn]
print(len(filenames))
filename = filenames[0]
handle_model=xr.open_dataarray(wd+'processed_model_data/'+filename, decode_times=False)
print(np.shape(handle_model))
hadcm_snow = handle_model[:,0:1,:,:].rename(var)
handle_model.close
#
landmask.data=np.where((hadcm_snow<snowthresh) | (np.isnan(hadcm_snow)),landmask.data,np.NaN)
#

m,var='TraCE','land'
filenames = [fn for fn in os.listdir(wd+'processed_model_data/') if '.'+var+'.' in fn]
filenames = [fn for fn in filenames if m in fn]
filenames = [fn for fn in filenames if 'regrid' in fn]
print(len(filenames))
filename = filenames[0]
handle_model=xr.open_dataarray(wd+'processed_model_data/'+filename, decode_times=False)
print(np.shape(handle_model))
trace_land = handle_model[:,0:1,:,:].rename(var)
handle_model.close
#
landmask.data = np.where(trace_land>0,landmask.data,np.NaN)
#

m,var='TraCE','runoff'
filenames = [fn for fn in os.listdir(wd+'processed_model_data/') if '.'+var+'.' in fn]
filenames = [fn for fn in filenames if m in fn]
filenames = [fn for fn in filenames if 'regrid' in fn]
print(len(filenames))
filename = filenames[0]
handle_model=xr.open_dataarray(wd+'processed_model_data/'+filename, decode_times=False)
print(np.shape(handle_model))
trace_land = handle_model[:,0:1,:,:].rename(var)
handle_model.close
#
landmask.data = np.where(trace_land>0,landmask.data,np.NaN)
#

m,var='HadCM','elev'
filenames = [fn for fn in os.listdir(wd+'processed_model_data/') if '.'+var+'.' in fn]
filenames = [fn for fn in filenames if m in fn]
filenames = [fn for fn in filenames if 'regrid' in fn]
print(len(filenames))
filename = filenames[0]
handle_model=xr.open_dataarray(wd+'processed_model_data/'+filename, decode_times=False)
print(np.shape(handle_model))
hadcm_elev = handle_model[:,0:1,:,:].rename(var)
handle_model.close
#
landmask.data = np.where(hadcm_elev>0,landmask.data,np.NaN)
landmask.data = np.where(np.sum(landmask,axis=0)>(12000/50),landmask.data,np.NaN)

#Save with land/snow mask
var='LakeStatus'
savedata(model_data[var],var,save=save)
if regrid: savedata(model_data[var],var,save=save,regrid=True,mask=landmask)


#%% Plot sensitivity tests for publication (figure 8)
import os
wd2 = '/Users/chrishancock/Library/CloudStorage/OneDrive-NorthernArizonaUniversity/Research/Manuscript/DAMP21k/' #changed
os.chdir(wd2+'Holocene-code') #changed
import csv
import sys
import numpy as np
import yaml
import time
import datetime
import netCDF4
import da_utils
import da_utils_lmr
import da_load_models
import da_load_proxies
import da_psms
import regionmask

starttime_total = time.time() # Start timer
# Use a given config file.  If not given, use config_default.yml.
if len(sys.argv) > 1: config_file = sys.argv[1]
#else:                 config_file = 'config.yml'
else:                 config_file = 'config_default.yml'
# Load the configuration options and print them to the screen.
print('Using configuration file: '+config_file)
with open(config_file,'r') as file: options = yaml.load(file,Loader=yaml.FullLoader)
print('=== SETTINGS ===')
for key in options.keys():
    print('%30s: %-15s' % (key,str(options[key])))
print('=== END SETTINGS ===')
options['exp_name_long'] = options['exp_name']+'.'+str(options['localization_radius'])+'loc.'+str(options['prior_window'])+'window.'+str(options['time_resolution'])+'.'
# Load the chosen proxy data
proxy_ts,collection_all = da_load_proxies.load_proxies(options)
proxy_data = da_load_proxies.process_proxies(proxy_ts,collection_all,options)

#%% Mask
if  model == 'DAMP_HadCM':
    mask = model_data['snow'][:,0]
    m = np.where(model_data['snow'][:,0]>500,np.NaN,mask.data)
    m = np.where(model_data['elev'][:,0].data > 0,m,np.NaN)
    mask.data=m
    #mask.data = np.where(model_data['snow'][:,0]>500,mask.data,np.NaN)
elif model == 'DAMP_TraCE':
    mask = model_data['land'][:,0]
    m = np.where(model_data['land'][0,0].data > 0,mask,np.NaN)
    m = np.where(model_data['snow'][:,0]>500,np.NaN,m)
    m = np.where(model_data['runoff'][:,0]>0,m,np.NaN)
    # = np.where(model_data['tas'][:,1].data<0,np.NaN,m)
    mask.data=m
proxyGrid = model_data['precip'][0,0]*np.NaN
for i in range(len(proxy_data['lats'])):
    lat = proxy_data['lats'][i]
    lon = proxy_data['lons'][i]
    lati = np.argmin(np.abs(proxyGrid.lat.data-proxy_data['lats'][i]))
    loni = np.argmin(np.abs(proxyGrid.lon.data-proxy_data['lons'][i]))
    proxyGrid[lati,loni] = 1

#%% #Sensitivity tests
var='LakeStatus'
tests,rmsedata = {},{}
for scale in range(4):
    rmsedata[scale],tests[scale] = {},{}
    if scale < 3: agemin,agemax=0+7000*scale,7000+7000*scale
    else: agemin,agemax=0,21000
    #
    precip = model_data['precip'][:,0].sel(age=slice(agemax,agemin))*1
    levap  = model_data[pet_method][:,0].sel(age=slice(agemax,agemin))
    runoff = model_data['Qest'][:,0].sel(age=slice(agemax,agemin))*1
    evap   = model_data['evap'][:,0].sel(age=slice(agemax,agemin))*1

    runoff = runoff.where(np.isfinite(mask.sel(age=slice(agemax,agemin))))*1
    #
    tests[scale]['LakeStatus']  = (model_data['Qest'][:,0]*np.NaN).rename(var).sel(age=slice(agemax,agemin))
    tests[scale]['Pconstant']   = (model_data['Qest'][:,0]*np.NaN).rename(var).sel(age=slice(agemax,agemin))
    tests[scale]['PETconstant'] = (model_data['Qest'][:,0]*np.NaN).rename(var).sel(age=slice(agemax,agemin))
    tests[scale]['Qconstant']   = (model_data['Qest'][:,0]*np.NaN).rename(var).sel(age=slice(agemax,agemin))
    #
    tests[scale]['LakeStatus']  = da_utils_ls.calcLakeStatus(runoff=runoff,levap=levap,precip=precip,evap=evap,Qmethod='runoff')*100
    tests[scale]['Qconstant']   = da_utils_ls.calcLakeStatus(runoff=runoff,levap=(levap*0+(levap.mean(dim='age'))),precip=(precip*0+(precip.mean(dim='age'))),evap=evap,Qmethod='runoff')*100
    runoff = runoff * runoff*0+runoff.max(dim='age')
    tests[scale]['Pconstant']   = da_utils_ls.calcLakeStatus(runoff=runoff,levap=levap,precip=(precip*0+(precip.mean(dim='age'))),evap=evap,Qmethod='runoff')*100
    tests[scale]['PETconstant'] = da_utils_ls.calcLakeStatus(runoff=runoff,levap=(levap*0+(levap.mean(dim='age'))),precip=precip,evap=evap,Qmethod='runoff')*100



np.nanmedian(da_utils_ls.calcSkill(tests[scale]['LakeStatus'] ,tests[scale]['PETconstant'],method='r2',dim='age',calcMean=False))


#%% Plot figure 8 RMSE of sensativity tests

# DA Functions
import da_utils_plotting as da_plot
import da_load_proxies
import da_load_models
import da_psms
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

print('packages loaded')
font = {'family': 'sans-serif', 'sans-serif': 'Lucida Grande'}
import cartopy.util        as cutil
scale=3
label = [['c','f'],['b','e'],['a','d'],['a','b']]
binsize=21000
levs=np.linspace(0,50,26)
m = 'RMSE'
cm='YlOrRd'
for scale in [3]:# range(0,4):
    if scale < 3: agemin,agemax=0+7000*scale,7000+7000*scale
    else: agemin,agemax=0,21000
    for i,t in enumerate(['Pconstant','PETconstant']):
        if t == 'PETconstant': name = ['(a)','Contribution of precipitation ('+str(int(agemax/1000))+'-'+str(int(agemin/1000))+' ka)\n'+"$P_{varying}$"+" & "+"$E_{constant}$" ]
        elif t == 'Pconstant': name = ['(b)','Contribution of evaporation ('+str(int(agemax/1000))+'-'+str(int(agemin/1000))+' ka)\n'+"$E_{varying}$"  +" & "+"$P_{constant}$" ]
        else: name = ''
        #Data to plot
        skill= da_utils_ls.calcSkill(tests[scale]['LakeStatus'],tests[scale][t],method=m,calcMean=False,dim='age',w=True)
        mean = str(np.round(da_utils_ls.calcSkill(tests[scale]['LakeStatus'],tests[scale][t],method=m,calcMean=True,dim='age'),1))
        #Set up plot
        plt.figure(figsize=(4,3),dpi=400)
        plt.rc('font', **font)
        ax = plt.axes(projection=ccrs.Robinson()) 
        da_plot.plotBaseMap(ax,ccrs.Robinson(),lims=False)
        ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,lw=0.1, color='k', alpha=0.7, linestyle=(0,(5,10)))
        #
        data_cyclic,lon_cyclic = cutil.add_cyclic_point(skill,coord=skill.lon)
        p=ax.pcolormesh(lon_cyclic,skill.lat,data_cyclic,transform=ccrs.PlateCarree(),vmin=0,vmax=50,cmap=cm)#levels=np.linspace(-0.3,0.3,11),extend='both',cmap='coolwarm')
        ax.set_title(name[0]+'\n',loc='left',fontsize=8)
        ax.set_title(name[1]+' lake status '+'(mean RMSE = '+mean+')',fontsize=8)#
        cbar = plt.colorbar(p,orientation='horizontal',shrink=0.9,aspect=25)
        cbar.set_ticks(np.linspace(0,50,6))
        cbar.set_ticklabels(['0\nNo Skill Loss','10','20','30','40','50\nHigh Skill Loss'],fontsize=8)
        cbar.ax.set_title('\n Darker colors indicate poor skill if only one variable is considered',fontsize=8)
        #cbar.ax.set_title('RMSE with lake status with varying Q, P, & E',fontsize=8)#,y=-2.8)
        #
        plt.tight_layout()
        plt.savefig(wd2+'Figures/Fig8/'+t+'_'+str(agemin)+'-'+str(agemax)+'.png', dpi=400)
        plt.show()   
        #%%
#Plot panel c (zonal means)
plt.figure(figsize=(2.5,5),dpi=400)
plt.rc('font', **font)
ax = plt.subplot()
for scale in [2,1,0]:
    mean = str(np.round(da_utils_ls.calcSkill(tests[scale]['LakeStatus'],tests[scale][t],method=m,calcMean=True,dim='age'),1))
    #Set up plot
    color = ['#695D8C','#C26989','#F8B988'][scale]
    if scale < 3: agemin,agemax=0+7000*scale,7000+7000*scale
    else: agemin,agemax=0,22000
    #plot
    skill= da_utils_ls.calcSkill(tests[scale]['LakeStatus'],tests[scale]['Pconstant'],method=m,calcMean=False,dim='age',w=True)
    skill = skill.mean(dim='lon').rolling(lat=3, center=True).mean()
    ax.plot(skill,skill.lat,linestyle='-',c=color,label="$E_{varying}$"+'\n'+str(int(agemax/1000))+'-'+str(int(agemin/1000))+' ka')
    skill= da_utils_ls.calcSkill(tests[scale]['LakeStatus'],tests[scale]['PETconstant'],method=m,calcMean=False,dim='age',w=True)
    skill = skill.mean(dim='lon').rolling(lat=3, center=True).mean()
    ax.plot(skill,skill.lat,linestyle='--',c=color,label="$P_{varying}$"+'\n'+str(int(agemax/1000))+'-'+str(int(agemin/1000))+' ka')

ax.legend(loc='lower center',ncol=3,fontsize=7,bbox_to_anchor=(0.6, -0.4),columnspacing=1)
ax.set_yticks(range(-90,91,30)); ax.set_ylim([-60,90]);  ax.set_ylabel('Latitude (°N)',fontsize=8);
ax.set_xlim([0,50]); ax.set_xlabel('RMSE\n(percentile)',fontsize=8)
ax.yaxis.grid(alpha=0.5,linestyle='-',color='k',lw=0.4)
ax.tick_params(labelsize=8) 
ax.yaxis.tick_right()
ax.yaxis.set_label_position('right')
ax.set_title('(c)',loc='left',fontsize=8)
plt.title('\nZonal Mean RMSE',fontsize=8)
plt.tight_layout()
plt.savefig(wd2+'Figures/Fig8/'+'zonalmean.png', dpi=400)
plt.show()
    
#%% Plot correlations between TraCE variables

#
m,cm,levs='r','coolwarm_r',np.linspace(-1,1,11)
#m,cm,levs='r2','YlOrRd_r',np.linspace(0,1,11)

count = 0
labs = ['a','b','c']
for var1 ,var2 in [['P/E','runoff']]:
    for agemin,agemax in [[0,21000],[11700,21000],[0,11700]]:
        if var1 == var2:continue
        if var1 == 'P/E':
            d1=model_data['precip'][:,0]/model_data['evap'][:,0].where(np.isfinite(mask)).sel(age=slice(agemax,agemin))
        else: 
            d1=model_data[var1][:,0].where(np.isfinite(mask)).sel(age=slice(agemax,agemin))
        name = labs[count]+') Correlation between '+var1+' & '+var2
        #Data to plot
        skill= da_utils_ls.calcSkill(d1,model_data[var2][:,0],method=m,calcMean=False,dim='age',w=True)
        Pmean = 'proxy site mean = '+str(np.round(da_utils_ls.calcSkill(d1,(model_data[var2][:,0]).where(np.isfinite(proxyGrid)),method=m,w=False,calcMean=True,dim='age'),2))
        Gmean = 'global mean = '+str(np.round(da_utils_ls.calcSkill(d1,model_data[var2][:,0],method=m,calcMean=True,dim='age',w=True),2))
        Pmed = 'proxy site median = '+str(np.round(np.nanmedian(da_utils_ls.calcSkill(d1,(model_data[var2][:,0]).where(np.isfinite(proxyGrid)),method=m,calcMean=False,dim='age')),2))
        #mean=str(round(np.nanmedian(skill),2))
        #Set up plot
        #
        plt.figure(figsize=(3.75,3),dpi=400)
        plt.rc('font', **font)
        ax = plt.axes(projection=ccrs.Robinson()) 
        da_plot.plotBaseMap(ax,ccrs.Robinson(),lims=False)
        ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,lw=0.1, color='k', alpha=0.7, linestyle=(0,(5,10)))
        #
        data_cyclic,lon_cyclic = cutil.add_cyclic_point(skill,coord=skill.lon)
        p=ax.contourf(lon_cyclic,skill.lat,data_cyclic,transform=ccrs.PlateCarree(),levels=levs,cmap=cm)#levels=np.linspace(-0.3,0.3,11),extend='both',cmap='coolwarm')
        ax.scatter(proxy_data['lons'],proxy_data['lats'],transform=ccrs.PlateCarree(),c='w',ec='k',s=2,linewidth=0.3,label='Proxy sites')
        plt.title(name+'\n('+Gmean+' / '+Pmed+')',fontsize=7)#+' Lake Status\n'+m+' values for '+str(int(agemax/1000))+'-'+str(int(agemin/1000))+' ka ('+mean+')',fontsize=8)
        cbar = plt.colorbar(p,orientation='vertical',shrink=0.6,aspect=15)
        cbar.set_ticks(np.linspace(-1,1,6))
        cbar.ax.tick_params(labelsize=6)
        cbar.set_label('Correlation Coefficient\n('+str(agemax/1000)+'-'+str(agemin/1000)+' ka)',fontsize=7)#,y=-2.8)
        plt.savefig(wd2+'Figures/FigS1/'+labs[count]+'_correlation_Q&PE_'+str(agemin)+'-'+str(agemax)+'_variable.png', dpi=400)
        plt.show()
        count+=1
        

#%%    
var='LakeStatus'
tests,rmsedata = {},{}
for scale in range(3):
    rmsedata[scale],tests[scale] = {},{}
    if scale == 2: agemin,agemax=0,11700
    elif scale == 1: agemin,agemax=11700,21000
    else:          agemin,agemax=0,21000
    #
    precip = model_data['precip'][:,0].sel(age=slice(agemax,agemin))
    evap   = model_data['evap'][:,0].sel(age=slice(agemax,agemin))
    levap  = model_data[pet_method][:,0].sel(age=slice(agemax,agemin))
    runoff = model_data['Qest'][:,0].sel(age=slice(agemax,agemin))
    runoff = runoff.where(np.isfinite(mask.sel(age=slice(agemax,agemin))))
    #
    tests[scale]['LakeStatusQ']  = (model_data['Qest'][:,0]*np.NaN).rename(var).sel(age=slice(agemax,agemin))
    tests[scale]['LakeStatusPE'] = (model_data['Qest'][:,0]*np.NaN).rename(var).sel(age=slice(agemax,agemin))
    #
    tests[scale]['LakeStatusQ']  = da_utils_ls.calcLakeStatus(runoff=runoff,levap=levap,evap=evap,precip=precip,Qmethod='runoff')*100
    tests[scale]['LakeStatusPE'] = da_utils_ls.calcLakeStatus(runoff=runoff,levap=levap,evap=evap,precip=precip,Qmethod='P/E')*100


#%% Plot correlation between lake status P/E and runoff in lake status percentile
agemin,agemax=11700,21000
#
m,cm,levs='r','coolwarm_r',np.linspace(-1,1,11)
#m,cm,levs='r2','YlOrRd_r',np.linspace(0,1,11)

count = 0
labs = ['d','e','f']
for var1 ,var2 in [['P/E','runoff']]:
    for scale,agemin,agemax in [[0,0,21000],[1,11700,21000],[2,0,11700]]:
        d1=tests[scale]['LakeStatusQ']
        print(np.shape(d1))
        d2=tests[scale]['LakeStatusPE']
        name = labs[count]+') Correlation between P/E & Runoff lake status'
        #Data to plot
        skill= da_utils_ls.calcSkill(d1,d2,method=m,calcMean=False,dim='age',w=True)
        Pmean = 'proxy site mean = ' +str(np.round(da_utils_ls.calcSkill(d1,d2,w=False,      method=m,calcMean=True,dim='age'),2))
        Gmean = 'global mean = '     +str(np.round(da_utils_ls.calcSkill(d1,d2,w=True,       method=m,calcMean=True,dim='age'),2))
        Pmed = 'proxy site median = '+str(np.round(np.nanmedian(da_utils_ls.calcSkill(d1,d2,method=m,calcMean=False,dim='age')),2))
        #mean=str(round(np.nanmedian(skill),2))
        #Set up plot
        #
        plt.figure(figsize=(3.75,3),dpi=400)
        plt.rc('font', **font)
        ax = plt.axes(projection=ccrs.Robinson()) 
        da_plot.plotBaseMap(ax,ccrs.Robinson(),lims=False)
        ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,lw=0.1, color='k', alpha=0.7, linestyle=(0,(5,10)))
        #
        data_cyclic,lon_cyclic = cutil.add_cyclic_point(skill,coord=skill.lon)
        p=ax.contourf(lon_cyclic,skill.lat,data_cyclic,transform=ccrs.PlateCarree(),levels=levs,cmap=cm)#levels=np.linspace(-0.3,0.3,11),extend='both',cmap='coolwarm')
        ax.scatter(proxy_data['lons'],proxy_data['lats'],transform=ccrs.PlateCarree(),c='w',ec='k',s=2,linewidth=0.3,label='Proxy sites')
        plt.title(name+'\n('+Gmean+' / '+Pmed+')',fontsize=7)#+' Lake Status\n'+m+' values for '+str(int(agemax/1000))+'-'+str(int(agemin/1000))+' ka ('+mean+')',fontsize=8)
        cbar = plt.colorbar(p,orientation='vertical',shrink=0.6,aspect=15)
        cbar.set_ticks(np.linspace(-1,1,6))
        cbar.ax.tick_params(labelsize=6)
        cbar.set_label('Correlation Coefficient\n('+str(agemax/1000)+'-'+str(agemin/1000)+' ka)',fontsize=7)#,y=-2.8)
        plt.savefig(wd2+'Figures/FigS1/'+labs[count]+'_correlation_Q&PE_'+str(agemin)+'-'+str(agemax)+'_lakestatus.png', dpi=400)
        plt.show()
        count+=1


#%% Plot LS midHolocene anomalies
var='LakeStatus'
import matplotlib as mpl
dataarray = model_data['runoff'][:,0]#*10000
dataarray  -=dataarray.sel(age=slice(1000,0)).mean(dim='age')
dataarray = dataarray.sel(age=slice(6500,5500)).mean(dim='age')

lat_regrid = np.arange(-88.59375,90,2.8125)
lon_regrid = np.arange(0,360,3.75) 
data_format = xr.Dataset(
            {'lat': (['lat'],lat_regrid,{'units':'degrees_north'}),
             'lon': (['lon'],lon_regrid,{'units':'degrees_east'})})
#regrid the data
regridder = xe.Regridder(dataarray.to_dataset(),data_format,'conservative_normed',periodic=True)
dataarray_regrid = regridder(dataarray.to_dataset(),keep_attrs=True)
dataarray_regrid = dataarray_regrid.to_array()[0]
#dataarray_regrid.attrs['units'] = dataarray.attrs['units'] 
#dataarray_regrid.attrs['long_name'] = dataarray.attrs['long_name'] 
dataarray = dataarray_regrid.rename(var)
ax = plt.axes(projection=ccrs.Robinson()) 
p=ax.pcolormesh(dataarray.lon,dataarray.lat,dataarray.where(np.isfinite(landmask[-1,0])),
                vmin=-0.1,vmax=0.1,transform=ccrs.PlateCarree(),cmap='BrBG')#,norm=mpl.colors.LogNorm())
ax.spines['geo'].set_edgecolor('black'); ax.set_global(); ax.coastlines()
#plt.title(var+' ('+dataarray.units+')')
plt.colorbar(p,orientation='horizontal')
plt.show()
#%%
