#==============================================================================
# Script for caclulating lake status prior and climate indicies for paleoDA
# Meant to be run offline before performing DA.
#    author: Chris Hancock
#    date  : 2/15/2023
#==============================================================================

#Load Packages
#import cartopy.crs         as ccrs        # Packages for mapping in python
#import matplotlib.pyplot   as plt         # Packages for making figures
import numpy as np
import os
import xarray as xr
#import xesmf as xe
import pyet
import metpy.calc as mpcalc
#from scipy.stats import rankdata

#setup file locations
wd='/Users/christopherhancock/Library/CloudStorage/OneDrive-NorthernArizonaUniversity/ECS_DA/data/models/'

#Set up standardized names
#Format = name, conversion multiply, conversion shift, new units
varkey = {}
varkey['TraCE_21ka'] = {
    'tas':   ['TREFHT',-273.15,'degC'],            #Convert K to degC
    'precip':None,   #Convert m/s to mm/day  
    'precipC':['PRECC',(1000*60*60*24),'mm/day'],   #Convert m/s to mm/day  
    'precipL':['PRECL',(1000*60*60*24),'mm/day'],   #Convert m/s to mm/day  
    'evap':  ['QFLX',(60*60*24),'mm/day'],         #Convert kg/m2/s to mm/day  
    'press': ['PS',0.001,'kPa'],                   #Convert Pa to kPa 
    #'slp':   ['PSL',0.001,'kPa'],                  #Convert Pa to kPa cfrA
    #'U':     ['Usurface',None,'m/s'],              #Created from multilevel U file; Units already m/s
    #'V':     ['Vsurface',None,'m/s'],              #Created from multilevel U file; Units already m/s
    #'U200':  ['U200',None,'m/s'],                  #Created from multilevel U file; Units already m/s
    #'V200':  ['V200',None,'m/s'],                  #Created from multilevel U file; Units already m/s
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
varkey['HadCM3B_transient21k'] = {
    'tas':   ['temp_mm_1_5m',-273.15,'degC'],      #Convert K to degC
    'precip':['precip_mm_srf',(60*60*24),'mm/day'],#Convert kg/m2/s to mm/day 
    'precipC':None,
    'precipL':None,
    'evap':  ['totalEvap_mm_srf',None,'mm/day'],   #Units already mm/day
    'press': ['p_mm_srf',0.001,'kPa'],             #Convert Pa to kPa 
    #'slp':   ['p_mm_msl',0.001,'kPa'],             #Convert Pa to kPa 
    #'U':     ['u_mm_10m',None,'m/s'],              #Units already m/s
    #'V':     ['v_mm_10m',None,'m/s'],              #Units already m/s
    #'U200':  ['u_mm_p_200',None,'m/s'],            #Units already m/s 
    #'V200':  ['v_mm_p_200',None,'m/s'],            #Units already m/sdownSol_mm_TOA
    'netrad':None,                                 #TODO: Not Available 
    'downSW':['downSol_Seaice_mm_s3_srf',0.0864,'MJ/(m2*day)'],#W/m2 to MJ/(m2*day)
    'toaSW': ['downSol_mm_TOA',0.0864,'MJ/(m2*day)'],          #W/m2 to MJ/(m2*day)
    'netSW': None,                                 #Not Available
    'netLW': None,                                 #Not Available
    'relhum':None,                                 #Not Available
    'spehum':['q_mm_1_5m',None,'kg/kg'],           #Units alread kg/kg
    'runoff':['totalRunoff_mm_srf',None,'mm/day'], #Units not provided but appear to be mm/day
    'snow':  ['snowdepth_mm_srf',None,'kg/m2'],    #Units already in kg/m2
    'land':  None,                                 #Don't need with elev
    'elev':  ['ht_mm_srf',None,'m'],               #Units already in m
    }

if (varkey['HadCM3B_transient21k'].keys()==varkey['TraCE_21ka'].keys())==False: print('Warning: HadCM and TraCE variable lists are not the same')

#%% Functions used to calculate PET and lake status

#Create a function which uses pyet and mpcalc to calculate PET using priestley_taylor or penman methods and xarray dataarrays
def calculatePET(method='priestley_taylor', tas = None, press = None, relhum=None, spehum=None,
                 netrad=None, netSW=None, netLW=None, downSW=None, toaSW=None, elev=None,
                 wind=None ):
    # Potential evaporation mm/day.
    # Parameters
    # ----------
    # method string, required (one of 'priestley_taylor' or 'penman')
    #     equation to use
    # tas: xarray.DataArray, required
    #     average day temperature [°C]
    # press: xarray.DataArray, required
    #     surface pressure [kPa]
    # relhum: xarray.DataArray, optional (need one of relhum or spehum (preferably relhum))
    #     relative humidity [%]
    # spehum: xarray.DataArray, optional (need one of relhum or spehum (preferably relhum))
    #     surface humidity [kg/kg]
    # netrad: xarray.DataArray, optional [MJ/(m2*day)]
    #     net radiation
    # spehum: xarray.DataArray, optional (need one of relhum or spehum (preferably relhum))
    #     wind [,/s]
    # Notes
    # -----
    # Modified from pyet to not need datatime age configuration
    # Based on equation 39 in :cite:t:`allen_crop_1998`.
    #
    #Calculate humidity
    if relhum is None:
        if spehum is None:
            print('!!!ERROR: One of relhum or spehum must be provided!!!!')
            #return None
        else:
            print('estimating relative humidity from specific humidity using mpcalc.relative_humidity_from_specific_humidity')
            rh = mpcalc.relative_humidity_from_specific_humidity(
                        pressure          = press,
                        temperature       = tas,
                        specific_humidity = spehum)#.to('percent')
            rh = rh.clip(0,1) *100
    else: rh = relhum
    #
    #Calcualte Net Radiation
    if netrad is None:
        #Need shortwave
        print('calculating net radiation from available radiation paramaters provided')
        if netSW is None: 
            if downSW is None: 
                print('!!!ERROR: One of netrad, netSW, or downSW!!!')
                return None
            else: SWn = downSW*0.9 #estimate albedo of water
        else: SWn = netSW
        #and longwave
        if netLW is None:  #Could cause error if don't provide correct data
            #modified from pyet.calc_rad_long() :cite:t:`allen_crop_1998`.
            a=1.35; b=-0.35
            STEFAN_BOLTZMANN_DAY = 4.903 * 10 ** -9 
            ea = pyet.calc_ea(tmean=tas,rh=rh)  
            ea.data = np.array(ea.data)
            rso =  (0.75 + (2 * 10 ** -5) * elev) * toaSW #modified of calc_rso
            solar_rat = np.clip(downSW / rso, 0.3, 1)
            tmp1 = STEFAN_BOLTZMANN_DAY * (tas + 273.16) ** 4
            tmp2 = 0.34 - 0.14 * np.sqrt(ea)  # OK
            tmp3 = a * solar_rat + b  # OK
            tmp3 = np.clip(tmp3, 0.05, 1) 
            LWn = (tmp1 * tmp2 * tmp3)
        else: LWn = netLW
        rn = SWn - LWn        
    else: rn = netrad
    #
    #Calculate PET
    if method == 'priestley_taylor':
        pet = pyet.priestley_taylor(tmean=tas,rn=rn,rh=rh,pressure=press)
        pet.attrs={}
        pet.attrs['long_name'] =  'Potential evapotranspiration (priestley_assessment_1972)' 
    elif method == 'penman':
        pet = pyet.penman(tmean=tas,rn=rn,rh=rh,pressure=press,wind=wind)
        pet.attrs={}
        pet.attrs['long_name'] =  'Potential evapotranspiration (penman_natural_1948)' 
    else: 
        print('!!!ERROR: Chosen method is not available. Choose one of priestley_taylor or penman!!!!')
        return None
    #Return PET dataarray
    pet.attrs['units'] = "mm/day"
    try: 
        pet=pet.reset_coords(['toa','ht'], drop=True).rename('PET')
    except:
        pet = pet.rename('PET')
    return pet
  #    

# Calculate lake values
def LakePercentile(LakeStatus=False,runoff=False,precip=False,levap=False,mask=False, time='time'):
    #Calculate lake status if not given
    if len(np.shape(LakeStatus))==0:
        LakeStatus = runoff/(levap-precip)
    #Mask data
    if len(np.shape(mask))>0:
        LakeStatus.data = np.where(mask,LakeStatus,np.nan)
    LakeStatus.rename('LakeStatus')
    #Calculate ranks
    ranks =  LakeStatus.rank(dim=time)
    ranks -= 1
    ranks /= (np.isfinite(ranks).sum(dim=time)-1)
    #Reformat for negative data
    positive_ranks = ranks.where(LakeStatus>=0).rank(dim=time)
    negative_ranks = ranks.where(LakeStatus<0).rank(dim=time)
    negative_ranks +=1.1
    #
    ranks.data = np.where(np.isfinite(positive_ranks),positive_ranks,negative_ranks)
    out =  ranks.rank(dim=time)
    out -= 1
    out /= (np.isfinite(out).sum(dim=time)-1)
    out.attrs['units'] = 'percentile'
    return(out)
                                
#%%
model = 'TraCE_21ka'
#model = 'HadCM3B_transient21k'
wdir=f'{wd}original_model_data/{model}/decadal/'

model_data={}
for var in [x for x in varkey[model].keys() if varkey[model][x]]:
    ##################################################################################
    #Get info about variable
    if  varkey[model][var] == None: 
        print('======================================================\nNo '+var+' in available files')
        continue 
    else: orignal_varname,conversion,units = varkey[model][var]
    if orignal_varname[1:7]=='_mm_p_': orignal_varname=orignal_varname[:6] #for hadcm
    print('======================================================\n'+var+'('+orignal_varname+')')
    #File to load 
    filenames = [fn for fn in os.listdir(wdir) if '.'+orignal_varname+'.' in fn]
    if model == 'TraCE_21ka': filename = [fn for fn in filenames if 'decavg_400BCE' in fn][0]
    elif model == 'HadCM3B_transient21k': filename = [fn for fn in filenames if '.monthly.ANN.' in fn][0]   
    else: 
        print("model must be TraCE_21ka or HadCM3B_transient21k")
        continue
    #Load Data
    handle_model=xr.open_dataset(wdir+filename, decode_times=False)
    da_var = handle_model[orignal_varname].rename(var)
    handle_model.close()
    if  model == 'HadCM3B_transient21k':
        #Drop extra dim for HadCM3B_transient21k
        dim_remove = [x for x in da_var.dims if x in ['ht','surface','msl','p','toa']]
        if (len(da_var[dim_remove[0]])) == 1 : da_var = da_var.squeeze(dim_remove)
        #HadCM wind slightly different 
        if 'latitude_1' in da_var.dims: da_var = da_var.rename({'latitude_1': 'lat', 'longitude_1': 'lon'})   
    #Standardize units among variables (and for pyet calculations)
    if 'units' in da_var.attrs: print(f'Original units = {da_var.units} (min/max = {str(np.min(da_var).data)}/{str(np.max(da_var).data)}')
    if conversion != None: 
        da_var.data =  da_var.data+conversion if var == 'tas' else da_var.data*conversion
        da_var.attrs['units'] = units
        print(f'New units = {da_var.units} (min/max = {str(np.min(da_var).data)}/{str(np.max(da_var).data)}')
    #Save to dictionary
    if 'long_name' not in da_var.attrs.keys(): da_var.attrs['long_name'] =  orignal_varname
    model_data[var] = da_var

#Calculate total precip from convective and large scale
if (model == 'TraCE_21ka'):
    model_data['precip'] =  model_data['precipC'] +  model_data['precipL'] 
    #Standardize slight offsets in land moodel and atmospheric model (TraCE)
    for var in ['snow','runoff']:
        model_data[var] = model_data[var].assign_coords(lat=model_data['tas'].lat.data)
        model_data[var] = model_data[var].assign_coords(lon=model_data['tas'].lon.data)
        model_data[var] = model_data[var].assign_coords(time=model_data['tas'].time.data)

#Negative total evap values for HadCM #True for both _s and non smoothed files in evap (but SW Asia vs Antarctic. just _s for precip. I think it has to do with smoothing 
for var in ['precip','evap','runoff']:  
    model_data[var]=np.clip(model_data[var],0,None)
    
#Land/Ice mask
model_data['snow'].data = np.where(~np.isfinite(model_data['snow']),0,model_data['snow'])
snowmask = (model_data['snow']<int(300)) #must be an integer to work (i.e. 600 kg/m2)
if model == 'TraCE_21ka':             landmask = (model_data['land']>0)
elif model == 'HadCM3B_transient21k': landmask = (model_data['elev']>0)


# Calculate Potential Evapotranspiration
method = 'priestley_taylor'
if model == 'TraCE_21ka':            
    model_data['PET'] = calculatePET(method=method,
        tas       = model_data['tas'],
        press     = model_data['press'],
        relhum    = model_data['relhum'][:,0],
        netSW     = model_data['netSW'],
        netLW     = model_data['netLW'],
        )  
elif model == 'HadCM3B_transient21k':
    model_data['PET'] = calculatePET(method=method,
        tas       = model_data['tas'],
        press     = model_data['press'],
        spehum    = model_data['spehum'],
        downSW    = model_data['downSW'],
        toaSW     = model_data['toaSW'],
        elev      = model_data['elev'],
        )

#%% Calculate Lake Status Percentile
if model == 'TraCE_21ka':             time = 'time'
elif model == 'HadCM3B_transient21k': time = 't'
model_data['LakeStatus'] = LakePercentile(runoff=model_data['runoff'],
                                          precip=model_data['precip'],
                                          levap=model_data['PET'],
                                          mask=(landmask & snowmask),
                                          time=time)
model_data['LakeStatus'].attrs['long_name'] = 'Lake Status (Runoff / (Lake Evaporation - Precipitation))'

#%%Save
print('Saving processed .nc files')
if model == 'TraCE_21ka':             filename = 'trace.01-36.22000BP.cam2.LakeStatus.22000BP_decavg_400BCE'
elif model == 'HadCM3B_transient21k': filename = 'deglh.vn1_0.LakeStatus.monthly.ANN.001.nc'
model_data['LakeStatus'].to_dataset(name='LakeStatus').to_netcdf(f'{wd}/original_model_data/{model}/{filename}.nc')

#%%
lat,lon=44.648,252.554
lati = np.argmin(abs(model_data['LakeStatus'].lat.data-lat))
loni = np.argmin(abs(model_data['LakeStatus'].lon.data-lon))
(model_data['precip'][:,lati,loni]).plot()
model_data['LakeStatus'][:,lati,loni].plot()
#(model_data['LakeStatus2'][:,lati,loni]*-1).plot()
#(model_data['precip'][:,lati,loni]/model_data['evap'][:,lati,loni]).plot(label='p-e')
plt.legend()

#%%
lat,lon=44.648,252.554
lati = np.argmin(abs(model_data['LakeStatus'].lat.data-lat))
loni = np.argmin(abs(model_data['LakeStatus'].lon.data-lon))
#(model_data['precip'][:,lati,loni]).plot()
(model_data['runoff'][:,lati,loni]).plot()
((model_data['PET'][:,lati,loni]-model_data['precip'][:,lati,loni])).plot()
(model_data['LakeStatus'][:,lati,loni]*-1).plot()
#(model_data['precip'][:,lati,loni]/model_data['evap'][:,lati,loni]).plot(label='p-e')
#plt.legend()

#%%
LakeStatus = model_data['runoff']/(model_data['PET']-model_data['precip'])

#%%
model_data['LakeStatus2'] = LakePercentile2(runoff=model_data['runoff'],
                                          precip=model_data['precip'],
                                          levap=model_data['PET'],
                                          mask=(landmask & snowmask),
                                          time=time)
model_data['LakeStatus2'].attrs['long_name'] = 'Lake Status (Runoff / (Lake Evaporation - Precipitation))'

#%%
# Calculate lake values
def LakePercentile2(LakeStatus=False,runoff=False,precip=False,levap=False,mask=False, time='time'):
    #Calculate lake status if not given
    if len(np.shape(LakeStatus))==0:
        LakeStatus = runoff/(levap-precip)
    #Mask data
    if len(np.shape(mask))>0:
        LakeStatus.data = np.where(mask,LakeStatus,np.nan)
    LakeStatus.rename('LakeStatus')
    #Calculate ranks
    ranks =  LakeStatus.rank(dim=time)
    ranks -= 1
    ranks /= (np.isfinite(ranks).sum(dim=time)-1)
    #Reformat for negative data
    positive_ranks = ranks.where(LakeStatus>=0).rank(dim=time)
    negative_ranks = ranks.where(LakeStatus<0).rank(dim=time)
    positive_ranks +=1.1
    #
    ranks.data = np.where(np.isfinite(negative_ranks),negative_ranks,negative_ranks)
    out =  ranks.rank(dim=time)
    out -= 1
    out /= (np.isfinite(out).sum(dim=time)-1)
    out.attrs['units'] = 'percentile'
    return(out)

LakeStatus[:300,lati,loni].plot()
    
        
#%% Create TraCE wind vectors at desired atmospheric levels (only need to do this once)
# wdir = wd+'original_model_data/DAMP_TraCE/'
# for var in ['U','V' ]:
#     filenames = [fn for fn in os.listdir(wdir) if '.'+var+'.' in fn]
#     for fn in filenames:
#         print(fn)
#         wind =  handle_model=xr.open_dataset(wdir+fn, decode_times=False)
#         #Create new datasets
#         windSurface = xr.Dataset({'time': wind['time'],
#                               'lat':wind['lat'],
#                               'lon':wind['lon'],
#                               var+'surface': wind[var][:,np.argmax(wind.lev.data),:,:]
#                               })
#         wind200 = xr.Dataset({'time': wind['time'],
#                               'lat':wind['lat'],
#                               'lon':wind['lon'],
#                               var+'200': wind[var][:,np.argmin(np.abs(wind.lev.data-200)),:,:]
#                               })
#         #Save
#         windSurface.to_netcdf(wdir+fn.replace("cam2."+var+".", "cam2."+var+"surface."))
#         wind200.to_netcdf(wdir+fn.replace("cam2."+var+".", "cam2."+var+"200."))
#         wind.close()

#%%
# #Calculate wind speed from U and V vectors (not needed for Penman method)
# var = 'UV'
# model_data[var] = np.sqrt(model_data['V']**2+model_data['U']**2).rename(var)
# #HadCM has a different spatial pattern. Fix this 
# if model=='DAMP_HadCM':
#     if model_data[var].lat.data != model_data['tas'].lat.data:#HadCM on a slighly different grid
#         data_format = xr.Dataset(
#             {'lat': (['lat'],model_data['tas'].lat.data,{'units':'degrees_north'}),
#               'lon': (['lon'],model_data['tas'].lon.data,{'units':'degrees_east'})})
#         #regrid the data
#         regridder = xe.Regridder(model_data[var].to_dataset(),data_format,'conservative_normed',periodic=True)
#         model_data[var]=regridder(model_data[var].to_dataset(),keep_attrs=True).to_array()[0]
# #
# model_data[var].attrs['units']     = model_data['V'].attrs['units']
# model_data[var].attrs['long_name'] = 'Surface wind speed'

























