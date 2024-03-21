#==============================================================================
# Script for caclulating lake status prior and climate indicies for paleoDA
# Meant to be run offline before performing DA.
#    author: Chris Hancock
#    date  : 2/15/2023
#==============================================================================

#Load Packages
import cartopy.crs         as ccrs        # Packages for mapping in python
import matplotlib.pyplot   as plt         # Packages for making figures
import numpy as np
import os
import xarray as xr
import xesmf as xe

wd = '/Users/chrishancock/Library/CloudStorage/OneDrive-NorthernArizonaUniversity/Research/Manuscript/DAMP21k/' #changed
os.chdir(wd+'Holocene-code') #changed
import da_utils_lakestatus as da_utils_ls
#setup file locations
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
    'U':     ['Usurface',None,'m/s'],              #Created from multilevel U file; Units already m/s
    'V':     ['Vsurface',None,'m/s'],              #Created from multilevel U file; Units already m/s
    'U200':  ['U200',None,'m/s'],                  #Created from multilevel U file; Units already m/s
    'V200':  ['V200',None,'m/s'],                  #Created from multilevel U file; Units already m/s
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
    'runoff':['totalRunoff_mm_srf',None,'mm/day'], #Units not provided but appear to be mm/day
    'snow':  ['snowdepth_mm_srf',None,'kg/m2'],    #Units already in kg/m2
    'land':  None,                                 #Don't need with elev
    'elev':  ['ht_mm_srf',None,'m'],               #Units already in m
    }

if (varkey['DAMP_HadCM'].keys()==varkey['DAMP_TraCE'].keys())==False: print('Warning: HadCM and TraCE variable lists are not the same')

#%% Define function to use to load and regrid data for DAMP project

# A function to load the original HadCM and TraCE models and reformat them to a standard dataarray
def dampLoadOriginal(model,orignal_varname,new_varname,binseq=[0,21000],wdir=wd+'original_model_data/',seasons=True):
    filenames = [fn for fn in os.listdir(wdir) if '.'+orignal_varname+'.' in fn]
    #Load annual as default
    if   model == 'DAMP_TraCE': filename = [fn for fn in filenames if 'decavg_400BCE' in fn][0]
    elif model == 'DAMP_HadCM': filename = [fn for fn in filenames if 'ANN' in fn][0]
    else: 
        print("model must be HadCM or TraCE")
    handle_model=xr.open_dataset(wdir+filename, decode_times=False)
    if orignal_varname[1:7]=='_mm_p_': orignal_varname=orignal_varname[:6]
    #Create dataarray using annual data
    out = [handle_model[orignal_varname].expand_dims(dim='season',axis=1).assign_coords({"season": ("season", ['ANN'])})]
    #Add season if available
    if seasons: 
        for season in ['JJA','DJF','MAM','SON']:       
            try:
                filename = [fn for fn in filenames if season in fn][0]
                handle_model=xr.open_dataset(wdir+filename, decode_times=False)
                out.append(handle_model[orignal_varname].expand_dims(dim='season',axis=1).assign_coords({"season": ("season", [season])}))
            except: continue
    out = xr.concat(out,dim='season')
    #Drop extra dim for hadcm
    if  model == 'DAMP_HadCM':
        v = [x for x in out.dims if x in ['ht','surface','msl','p','toa']]
        if (len(out[v[0]])) == 1 : out = out.squeeze(v)
    #Standardize lat/lon names
    if 'latitude'   in out.dims: out = out.rename({'latitude': 'lat', 'longitude': 'lon'})     #change HadCM to match TraCE
    if 'latitude_1' in out.dims: out = out.rename({'latitude_1': 'lat', 'longitude_1': 'lon'}) #HadCM wind slightly different   
    #Standardize variable names
    out = out.rename(new_varname)
    #Standardize ages and Bin time as needed
    if len(binseq)==3: #If choosing to bin time data to a courser resolution
        if model == 'DAMP_TraCE': out = out.assign_coords(time=out['time']*-1000).rename({'time': 'age'})
        if model == 'DAMP_HadCM': out = out.assign_coords(t=out['t']*-1+5.5).rename({'t': 'age'})
        binyrs  = [*range(binseq[0],binseq[1]+1,binsize)]
        binvec  = [*range(binseq[0]-int(binseq[2]/2),binseq[1]+int(binseq[2]/2)+1,binseq[2])] 
        out = out.groupby_bins("age",binvec).mean(dim='age').rename({'age_bins': 'time'})
        out['age'] = binyrs  
    elif len(binseq)==2: 
        #Ages #Reshape time as needed #TraCE centered on 10s, HadCM on 5.5s
        if model == 'DAMP_TraCE': out = out.assign_coords(time=out['time']*-1000).rename({'time': 'age'})
        if model == 'DAMP_HadCM': out = out.assign_coords(t=out['t']*-1+5.5).rename({'t': 'age'})
        out = out.sel(age=slice(binseq[1],binseq[0]))
    #Finish
    handle_model.close
    return(out)

#A function to regrid the data into the desired spatial resolution for multimodel comparisons
def dampRegrid(dataarray, method = 'conservative_normed', lat_regrid = np.arange(-88.59375,90,2.8125), lon_regrid = np.arange(0,360,3.75)):
    data_format = xr.Dataset(
            {'lat': (['lat'],lat_regrid,{'units':'degrees_north'}),
             'lon': (['lon'],lon_regrid,{'units':'degrees_east'})})
    regridder = xe.Regridder(dataarray.to_dataset(),data_format,method,periodic=True)
    dataarray_regrid = regridder(dataarray.to_dataset(),keep_attrs=True)
    dataarray_regrid = dataarray_regrid.to_array()[0]
    return(dataarray_regrid)

        
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

#%%Load original model data ##################################################################################
#choose which model to use 
model = 'DAMP_TraCE'
#model = 'DAMP_HadCM'

model_data={}
print(model)
for var in list(varkey[model].keys()):
    ##################################################################################
    #Get info about variable
    if  varkey[model][var] == None: 
        print('======================================================\nNo '+var+' in available files')
        continue 
    else: var_name,conversion,units = varkey[model][var]
    print('======================================================\n'+var+'('+var_name+')')
    #Load Data and create season dimension to combine the same variables
    model_data[var] = dampLoadOriginal(model=model,orignal_varname=varkey[model][var][0],new_varname=var,wdir=wd+'original_model_data/'+model+'/',seasons=True)
    ##################################################################################
    ##################################################################################
    #Standardize units among variables (and for pyet calculations)
    if 'units' in model_data[var].attrs: 
        print('-\nOriginal units = '+model_data[var].units+' (min/max = '+str(np.min(model_data[var]).data)+'/'+str(np.max(model_data[var]).data)+')')
    #Standardize units 
    if conversion != None:
        if var == 'tas': model_data[var].data += conversion
        else:            model_data[var].data *= conversion
    #TODO: negative total evap values for HadCM #True for both _s and non smoothed files in evap (but SW Asia vs Antarctic. just _s for precip. I think it has to do with smoothing 
    if var in ['precip','evap']:  model_data[var]=np.clip(model_data[var],0,None)
    model_data[var].attrs['units'] = units
    if 'long_name' not in model_data[var].attrs.keys(): model_data[var].attrs['long_name'] =  var_name
    print('New units = '+model_data[var].units+' (min/max = '+str(np.min(model_data[var]).data)+'/'+str(np.max(model_data[var]).data)+')')
    ##################################################################################
    ##################################################################################
    #Standardize slight offsets in land moodel and atmospheric model (TraCE)
    #########################################
    if (model == 'DAMP_TraCE') & (var in ['snow','runoff']):
         model_data[var] = model_data[var].assign_coords(lat=model_data['tas'].lat.data)
         model_data[var] = model_data[var].assign_coords(lon=model_data['tas'].lon.data)
         model_data[var] = model_data[var].assign_coords(age=model_data['tas'].age.data)
    ##################################################################################
    ##################################################################################

#Calculate wind speed from U and V vectors (not needed for Priestley_taylor method)
var = 'UV'
model_data[var] = np.sqrt(model_data['V']**2+model_data['U']**2).rename(var)
#HadCM has a different spatial pattern. Fix this 
if model=='DAMP_HadCM':
    if model_data[var].lat.data != model_data['tas'].lat.data:#HadCM on a slighly different grid
        data_format = xr.Dataset(
            {'lat': (['lat'],model_data['tas'].lat.data,{'units':'degrees_north'}),
              'lon': (['lon'],model_data['tas'].lon.data,{'units':'degrees_east'})})
        #regrid the data
        regridder = xe.Regridder(model_data[var].to_dataset(),data_format,'conservative_normed',periodic=True)
        model_data[var]=regridder(model_data[var].to_dataset(),keep_attrs=True).to_array()[0]
#
model_data[var].attrs['units']     = model_data['V'].attrs['units']
model_data[var].attrs['long_name'] = 'Surface wind speed'
##################################################################################
##################################################################################


#%% Calcualte PET 
       
method = 'priestley_taylor'
if model == 'DAMP_TraCE':            
    model_data['PET'] = da_utils_ls.calculatePET(method=method,
        tas       = model_data['tas'][:,0:1],
        press     = model_data['press'][:,0:1],
        relhum    = model_data['relhum'][:,0:1],
        netSW     = model_data['netSW'][:,0:1],
        netLW     = model_data['netLW'][:,0:1],
        wind      = model_data['UV'][:,0:1] #not needed for priestley_taylor
        )
    
elif model == 'DAMP_HadCM':
    model_data['PET'] = da_utils_ls.calculatePET(method=method,
        tas       = model_data['tas'][:,0:1],
        press     = model_data['press'][:,0:1],
        spehum    = model_data['spehum'][:,0:1],
        downSW    = model_data['downSW'][:,0:1],
        toaSW     = model_data['toaSW'][:,0:1],
        elev      = model_data['elev'][:,0:1],
        wind      = model_data['UV'][:,0:1] #not needed for priestley_taylor
        )
#
#%%Create a uniform snow/land mask for the lake status 
#Load the data for each model
m='DAMP_TraCE'
Tland = dampLoadOriginal(model=m,orignal_varname=varkey[m]['land'][0],new_varname='land',wdir=wd+'original_model_data/'+m+'/',seasons=False)
Tsnow = dampLoadOriginal(model=m,orignal_varname=varkey[m]['snow'][0],new_varname='snow',wdir=wd+'original_model_data/'+m+'/',seasons=False)
Tsnow.data = np.where(Tsnow==np.NaN,0,Tsnow)
m='DAMP_HadCM'
Helev = dampLoadOriginal(model=m,orignal_varname=varkey[m]['elev'][0],new_varname='elev',wdir=wd+'original_model_data/'+m+'/',seasons=False)
Hsnow = dampLoadOriginal(model=m,orignal_varname=varkey[m]['snow'][0],new_varname='snow',wdir=wd+'original_model_data/'+m+'/',seasons=False)
#Create a mask
#%%
snowthresh=600 #must be an integer to work (i.e. 300) #kg/m2
Hmask = Helev>0;   
Hmask.data = np.where(Hsnow > snowthresh,False,Hmask) #mask for hadcm
Tmask = Tland>0; 
Tmask.data = np.where(Tsnow > snowthresh,False,Tmask) #mask for hadcm
Rmask = dampRegrid(Helev) > 0; 
Rmask.data =  np.where(dampRegrid(Tland,) < 0.25,False,Rmask,) #Mask regridded land
Rmask.data = np.where(dampRegrid(Hsnow) > snowthresh,False,Rmask) #mask for regridded hadcm snow
Rmask.data = np.where(dampRegrid(Tsnow) > snowthresh,False,Rmask) #mask for regridded trace snow
#Also mask so covers the Holocene at least 
Hmask = Hmask.where(Hmask.sum(dim='age') > 12000/10).rename('mask')
Tmask = Tmask.where(Tmask.sum(dim='age') > 12000/10).rename('mask')
Rmask = Rmask.where(Rmask.sum(dim='age') > 12000/10).rename('mask')
#%%
# #Plot to make sure looks ok
# dataarray = xr.corr(model_data['runoff'][:,0],model_data['precip'][:,0]/model_data['evap'][:,0],dim='age')
# ax = plt.axes(projection=ccrs.Robinson()) 
# p=ax.pcolormesh(dataarray.lon,dataarray.lat,dataarray),transform=ccrs.PlateCarree())
# #p=ax.pcolormesh(dataarray.lon,dataarray.lat,dataarray.isel(season=np.where(dataarray.season=='ANN')[0]).mean(dim=['age','season']),transform=ccrs.PlateCarree())
# ax.spines['geo'].set_edgecolor('black'); ax.set_global(); ax.coastlines(); ax.set_title(Rmask.name+'\n(mean value 21-0 ka)')
# plt.colorbar(p,orientation='horizontal')
# plt.show()
#%%Calculate Lake Status
if model   == 'DAMP_TraCE': mask = Tmask
elif model == 'DAMP_HadCM': mask = Hmask
else: mask = None

model_data['LakeStatus'] = da_utils_ls.calcLakeStatus(Qmethod='runoff',
    runoff = model_data['runoff'][:,0:1].where(mask.data==True), 
    precip = model_data['precip'][:,0:1].where(mask.data==True),
    levap  = model_data['PET'][:,0:1].where(mask.data==True), 
    ).rename('LakeStatus')

#%%

runoff_regrid = dampRegrid(model_data['runoff'][:,0:1],method='nearest_s2d')
#Fix  boundery issue from regridding
if model == 'DAMP_TraCE':
    runoff_regrid.values[:,0,46,77] =runoff_regrid.values[:,0,46,76] 
elif model == 'DAMP_HadCM':
    runoff_regrid.values[:,0,46,77] = runoff_regrid.values[:,0,46,76] 
    runoff_regrid.values[:,0,51,5] = runoff_regrid.values[:,0,50,5] 
    runoff_regrid.values[:,0,43,9] = runoff_regrid.values[:,0,42,9] 
    runoff_regrid.values[:,0,25,39] = runoff_regrid.values[:,0,25,38] 

model_data['LakeStatus_regrid'] = da_utils_ls.calcLakeStatus(Qmethod='runoff',
    runoff = runoff_regrid.where(Rmask.data==True), 
    precip = dampRegrid(model_data['precip'][:,0:1]).where(Rmask.data==True),
    levap  = dampRegrid(model_data['PET'][:,0:1]).where(Rmask.data==True)
    ).rename('LakeStatus')

model_data['LakeStatus'].attrs['units'] = 'percentile'
model_data['LakeStatus_regrid'].attrs['units'] = 'percentile'


#%%

# count=0
# vals = model_data['LakeStatus_regrid'][:,0]  #model_data['runoff'][:,0:1].where(Tmask.data==True)
# #vals.data = np.where(Rmask==True,vals.data,np.NaN)
# for i in range(len(proxy_data['lats'])):
#     lati = np.argmin(abs(vals.lat.data-proxy_data['lats'][i]))
#     loni = np.argmin(abs(vals.lon.data-proxy_data['lons'][i]))
#     v = np.isfinite(vals[:,lati,loni]).sum(axis=0).data
#     #print(v)
#     if np.isnan(v):
#         count=0
#     elif (v <1000): #< len(vals.age)):
#         #print(loni)
#         #print(proxy_data['lons'][i])
#         print(str(lati)+', '+str(loni))
#         print(str(proxy_data['lats'][i])+', '+str(proxy_data['lons'][i]))
#         count+=1
#     elif (np.nansum(vals[:,lati,loni])==0):
#         count+=1
#     elif v > 2101:
#         count+-1
# count
        
        

#%%Save
save = True
if save: 
    print('Saving processed .nc files')
    # for var in ['tas','precip','evap','PET','U','U200','V','V200','slp','runoff']:
    #     array_orig = model_data[var]
    #     array_rg = dampRegrid(array_orig)
    #     array_rg = array_rg.rename(var)
    #     array_rg.attrs['units'] = model_data[var].units
    #     array_rg.attrs['long_name'] = model_data[var].long_name
    #     array_orig.to_netcdf(wd+'processed_model_data/'+model+'.'+var+'.21ka_decavg_0ka.nc')
    #     array_rg.to_netcdf(wd+'processed_model_data/'+model+'_regrid.'+var+'.21ka_decavg_0ka.nc')
    model_data['LakeStatus'].to_netcdf(wd+'processed_model_data/'+model+'.'+'LakeStatus'+'.21ka_decavg_0ka.nc')
    model_data['LakeStatus_regrid'].to_netcdf(wd+'processed_model_data/'+model+'_regrid.'+'LakeStatus'+'.21ka_decavg_0ka.nc')
    print("Files saved")






























#%%
#%%
#%%
#%%
#%%
#%% Plot sensitivity tests for publication (figure 8)
import os
import numpy as np
import da_utils_plotting as da_plot

wd2 = '/Users/chrishancock/Library/CloudStorage/OneDrive-NorthernArizonaUniversity/Research/Manuscript/DAMP21k/' #changed
os.chdir(wd2+'Holocene-code') #changed

#m = 'TraCE'
#model='DAMP_'+m
#load data
data = {}
if model   == 'DAMP_TraCE': mask = Tmask
elif model == 'DAMP_HadCM': mask = Hmask
mask = Rmask
for var in ['precip','runoff','PET','LakeStatus']:
    handle_model=xr.open_dataarray(wd+'/processed_model_data/'+model+'_regrid.'+var+'.21ka_decavg_0ka.nc', decode_times=False)
    data[var] = handle_model
    data[var].data = np.where(mask==True,data[var],np.NaN)
    data[var]=data[var][:,0]
    handle_model.close()

#%% #Sensitivity tests
tests = {}

for scale in range(4):
    print(scale)
    tests[scale] = {}
    if scale < 3: agemin,agemax=0+7000*scale,7000+7000*scale
    else: agemin,agemax=0,21000
    #Sel ages 
    precip = data['precip'].sel(age=slice(agemax,agemin))*1
    levap  = data['PET'].sel(age=slice(agemax,agemin))*1
    runoff = data['runoff'].sel(age=slice(agemax,agemin))*1
    #Create empty arrays to fill data
    tests[scale]['LakeStatus']  = (data['runoff']*np.NaN).rename(var).sel(age=slice(agemax,agemin))
    tests[scale]['Pconstant']   = (data['runoff']*np.NaN).rename(var).sel(age=slice(agemax,agemin))
    tests[scale]['PETconstant'] = (data['runoff']*np.NaN).rename(var).sel(age=slice(agemax,agemin))
    #Calculate Lake status for each scanrio
    tests[scale]['LakeStatus']  = da_utils_ls.calcLakeStatus(runoff=runoff,levap=levap,precip=precip,Qmethod='runoff')*100
    runoff = runoff * runoff*0 + runoff.max(dim='age') #keep constant
    tests[scale]['Pconstant']   = da_utils_ls.calcLakeStatus(runoff=runoff,levap=levap,precip=(precip*0+(precip.mean(dim='age'))),Qmethod='runoff')*100
    tests[scale]['PETconstant'] = da_utils_ls.calcLakeStatus(runoff=runoff,levap=(levap*0+(levap.mean(dim='age'))),precip=precip,Qmethod='runoff')*100




#%% Plot figure 8 RMSE of sensativity tests
# DA Functions

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
        cbar.set_ticklabels(['0\nNo skill loss','10','20','30','40','50\nHigh skill loss'],fontsize=8)
        cbar.ax.set_title('\n Darker colors indicate poor skill if only one variable is considered',fontsize=8)
        #cbar.ax.set_title('RMSE with lake status with varying Q, P, & E',fontsize=8)#,y=-2.8)
        #
        plt.tight_layout()
        plt.savefig(wd2+'Figures/Fig8/'+t+'_'+str(agemin)+'-'+str(agemax)+'_'+model+'.png', dpi=400)
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
ax.set_yticks(range(-90,91,30)); ax.set_ylim([-60,90]);  ax.set_ylabel('latitude (°N)',fontsize=8);
ax.set_xlim([0,50]); ax.set_xlabel('RMSE\n(percentile)',fontsize=8)
ax.yaxis.grid(alpha=0.5,linestyle='-',color='k',lw=0.4)
ax.tick_params(labelsize=8) 
ax.yaxis.tick_right()
ax.yaxis.set_label_position('right')
ax.set_title('(c)',loc='left',fontsize=8)
plt.title('\nZonal mean RMSE',fontsize=8)
plt.tight_layout()
plt.savefig(wd2+'Figures/Fig8/'+'zonalmean'+'_'+model+'.png', dpi=400)
plt.show()
#%%
