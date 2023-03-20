#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 07:56:31 2023

@author: chrishancock
"""
import os
wd = '/Users/chrishancock/Library/CloudStorage/OneDrive-NorthernArizonaUniversity/Research/Manuscript/DAMP12k/' #changed
os.chdir(wd+'Holocene-code') #changed
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


import numpy as np
import xarray as xr
import glob
import da_utils
import netCDF4
from scipy import stats

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

# A function to load model data
def load_model_data(options):
    #
    model_dir          = options['data_dir']+'models/processed_model_data/'
    original_model_dir = options['data_dir']+'models/original_model_data/'
    if options['models_for_prior'] == ['DAMP12kTraCE']: model_dir = options['data_dir']+'models/DAMP12kTraCE/processed/'
    age_range_model_txt = str(options['age_range_model'][1]-1)+'-'+str(options['age_range_model'][0])
    #
    # Load the model data
    n_models = len(options['models_for_prior'])
    model_data = {}
    model_data['units'] = {}
    for j,var_name in enumerate(options['vars_to_reconstruct']):
        for i,model in enumerate(options['models_for_prior']):
            #
            print('Loading variable '+var_name+' for model '+str(i+1)+'/'+str(n_models)+': '+model)
            #
            # Get the model filename
            if options['models_for_prior'] == ['DAMP12kTraCE']: model_filename = 'trace.DAMP12k.'+var_name+'.22ka_decavg_0ka.nc'
            else: model_filename = model+'.'+age_range_model_txt+'BP.'+var_name+'.timeres_'+str(options['time_resolution'])+'.nc'
            #
            # Check to see if the file exists.  If not, create it.
            filenames_all = glob.glob(model_dir+'*.nc')
            filenames_all = [filename.split('/')[-1] for filename in filenames_all]
            if model_filename not in filenames_all:
                print('File '+model_dir+model_filename+' does not exist.  Creating it now.')
                #process_models(model,var_name,options['time_resolution'],options['age_range_model'],model_dir,original_model_dir)
                print('File '+model_dir+model_filename+' created!')
            #
            # Load selected variables
            model_individual = {}
            handle_model = xr.open_dataset(model_dir+model_filename,decode_times=False)
            if j == 0: # Vectors with no lat infomration such as itcz position will mess up these dims #TODO
                model_data['lat']              = handle_model['lat'].values
                model_data['lon']              = handle_model['lon'].values
            if 'month'  in handle_model.dims: model_individual['time_ndays'] = handle_model['days_per_month_all'].values
            if 'season' in handle_model.dims: model_individual['season']     = handle_model['season'].values
            if len(handle_model['lat'].values)  == 1: # Vectors with no lat infomration such as itcz position will mess up these dims #TODO
                  model_individual[var_name] = np.repeat(handle_model[var_name],len(model_data['lat']),axis=2)
                  model_individual[var_name][:,:,1:,:] *= np.NaN
                  model_individual[var_name] = model_individual[var_name].assign_coords(lat= model_data['lat'])
            else: model_individual[var_name] = handle_model[var_name]
            model_data['units'][var_name] = handle_model[var_name].units #Save units so know reconstructing based on what we think we are
            handle_model.close()
            #
            #Allow for more custom time range and resolution
            age_bounds = np.arange(options['age_range_to_reconstruct'][0],options['age_range_to_reconstruct'][1]+1,options['time_resolution']) - 0.5
            #model_individual['age']     = (age_bounds[:-1]+age_bounds[1:])/2     
            #model_individual[var_name]  = model_individual[var_name].groupby_bins("age",age_bounds).mean(dim='age').values
            model_individual['age']     = handle_model.age.values  
            model_individual[var_name]  = model_individual[var_name].values
            #
            # Compute annual, jja, and djf means of the model data
            n_lat = len(model_data['lat'])
            n_lon = len(model_data['lon'])
            ind_jja = [5,6,7]
            ind_djf = [11,0,1]
            if 'month' in handle_model.dims:
                time_ndays_model_latlon = np.repeat(np.repeat(model_individual['time_ndays'][:,:,None,None],n_lat,axis=2),n_lon,axis=3)
                model_individual[var_name+'_annual'] = np.average(model_individual[var_name],axis=1,weights=time_ndays_model_latlon)
                model_individual[var_name+'_jja']    = np.average(model_individual[var_name][:,ind_jja,:,:],axis=1,weights=time_ndays_model_latlon[:,ind_jja,:,:]) #TODO: Check this.
                model_individual[var_name+'_djf']    = np.average(model_individual[var_name][:,ind_djf,:,:],axis=1,weights=time_ndays_model_latlon[:,ind_djf,:,:]) #TODO: Check this.
            elif 'season' in handle_model.dims:
                model_individual[var_name+'_annual'] = model_individual[var_name][:,model_individual['season']=='ANN',:,:][:,0,:,:]
                for szn in ['jja','djf']:
                    if szn.upper() in model_individual['season']:
                        model_individual[var_name+'_'+szn] = model_individual[var_name][:,model_individual['season']==szn.upper(),:,:][:,0,:,:]
                    else: model_individual[var_name+'_'+szn] = model_individual[var_name][:,model_individual['season']=='ANN',:,:][:,0,:,:]
            #
            # In each model, central values will not be selected within max_resolution/2 of the edges
            n_time = len(model_individual['age'])
            model_individual['valid_inds'] = np.full((n_time),True,dtype=bool)
            buffer = int(np.floor((options['maximum_resolution']/options['time_resolution'])/2))
            if buffer > 0:
                model_individual['valid_inds'][:buffer]  = False
                model_individual['valid_inds'][-buffer:] = False
            #
            # Set the model number for each data point
            model_individual['number'] = np.full((n_time),(i+1),dtype=int)
            #
            # Join the values together
            for key in list(model_individual.keys()):
                if i == 0: model_data[key] = model_individual[key]
                else:      model_data[key] = np.concatenate((model_data[key],model_individual[key]),axis=0)
    #
    print('\n=== MODEL DATA LOADED ===')
    print('Models loaded    (n='+str(n_models)+'):'+str(options['models_for_prior']))
    print('Variables loaded (n='+str(len(options['vars_to_reconstruct']))+'):'+str(options['vars_to_reconstruct']))
    print('---')
    print('Data stored in dictionary "model_data", with keys and dimensions:')
    for key in list(model_data.keys()): print('%20s %-20s' % (key,str(np.shape(model_data[key]))))
    print(model_data['units'])
    print('=========================\n')
    #
    return model_data

#%%
# Load the chosen proxy data
proxy_ts,collection_all = da_load_proxies.load_proxies(options)
proxy_data = da_load_proxies.process_proxies(proxy_ts,collection_all,options) #proxy_ts[1:2]

# Get some dimensions
n_models_in_prior = len(options['models_for_prior'])
n_proxies         = proxy_data['values_binned'].shape[0]

model_data = load_model_data(options)

# If the prior is allowed to change through time, remove the mean of the reference period from each model.
if options['reconstruction_type'] == 'relative':
    for i in range(n_models_in_prior):
        ind_for_model = (model_data['number'] == (i+1))
        ind_ref = (model_data['age'] >= options['reference_period'][0]) & (model_data['age'] < options['reference_period'][1]) & ind_for_model
        for var in options['vars_to_reconstruct']:
            model_data[var][ind_for_model,:,:,:]         = model_data[var][ind_for_model,:,:,:]         - np.mean(model_data[var][ind_ref,:,:,:],axis=0)
            model_data[var+'_annual'][ind_for_model,:,:] = model_data[var+'_annual'][ind_for_model,:,:] - np.mean(model_data[var+'_annual'][ind_ref,:,:],axis=0)
            model_data[var+'_jja'][ind_for_model,:,:]    = model_data[var+'_jja'][ind_for_model,:,:]    - np.mean(model_data[var+'_jja'][ind_ref,:,:],axis=0)
            model_data[var+'_djf'][ind_for_model,:,:]    = model_data[var+'_djf'][ind_for_model,:,:]    - np.mean(model_data[var+'_djf'][ind_ref,:,:],axis=0)
#%%
# Use PSMs to get model-based proxy estimates
#proxy_estimates_all = da_psms.psm_main(model_data,proxy_data,options)


n_proxies = proxy_data['values_binned'].shape[0]
proxy_estimates_all = np.zeros((n_proxies,len(model_data['age']))) * np.NaN

for i in range(n_proxies):
    #
    psm_selected = proxy_data['metadata'][i][-1]
    psm_selected='get_LakeStatus'
    #
    # Calculate the model-based proxy estimate depending on the PSM (or variable to compare, it the proxy is already calibrated)
    # Model values are in units of degree C (for tas) and mm/day (for precip)
    if   psm_selected == 'get_tas':        proxy_estimate = da_psms.get_model_values(model_data,proxy_data,'tas',i)
    elif psm_selected == 'get_precip':     proxy_estimate = da_psms.get_model_values(model_data,proxy_data,'precip',i)
    elif psm_selected == 'get_LakeStatus': proxy_estimate = da_psms.get_model_values(model_data,proxy_data,'LakeStatus',i)  ##DAMP12k- add lake option
    elif psm_selected == 'use_nans':       proxy_estimate = da_psms.use_nans(model_data,'LakeStatus')
    else:                                  proxy_estimate = da_psms.use_nans(model_data,'LakeStatus')
    if sum(np.isfinite(proxy_estimate))==0:proxy_estimate = da_psms.use_nans(model_data,'LakeStatus')
    # If the proxy units are mm/a, convert the model-based estimates from mm/day to mm/year
    if proxy_data['units'][i] == 'mm/a': proxy_estimate = proxy_estimate*365.25  #TODO: Is there a better way to account for leap years in these decadal means?
    proxy_estimates_all[i] = proxy_estimate

#%%

import matplotlib.pyplot   as plt         # Packages for making figures

var_name= 'LakeStatus'
lat = 23
lon = 31
i = np.where(proxy_data['lons']==lon)[0]

modelvals = model_data[var_name+'_annual'][:,np.argmin(abs(model_data['lat']-lat)),np.argmin(abs(model_data['lon']-lon))]
proxy_estimates_all_age=proxy_estimates_all[i]

plt.figure(dpi=400)
plt.axhline(y = 0, color = 'grey', linestyle = '-',lw=0.2)
plt.plot(model_data['age'],modelvals,label='model',alpha=0.5)
plt.plot(model_data['age'],proxy_estimates_all_age[0],label='psm',alpha=0.5)
plt.scatter(proxy_data['age_centers'],proxy_data['values_binned'][i],label='proxy',color='goldenrod')
plt.gca().invert_xaxis()
plt.legend()
plt.show()

#%%
n_iterations=100
percent = 0.5
unc_scale = 0.5
lats,lons =model_data['lat'],model_data['lon']
recon_ens  = np.zeros((len(proxy_data['age_centers']),
                       int(np.ceil(len(model_data['age'])*percent/100)*n_iterations),
                       len(lats),len(lons))); 
recon_ens[:] = np.nan

options['prior_window']=options['prior_window']
options['prior_window'] = 22000
for age_counter,age in enumerate(proxy_data['age_centers']): 
    proxy_vals_age = proxy_data['values_binned'][:,age_counter:(age_counter+1)]
    #
    print(age)
    for lati,lat in enumerate(lats):
        for loni,lon in  enumerate(lons):
            proxyi = np.where((np.abs(proxy_data['lats']-lat)<45/2) & 
                              (np.abs(proxy_data['lons']-lon)<75/2))[0]
            if len(proxyi) > 0:
                xb_idx = []
                proxy_estimates_all_age = proxy_estimates_all[proxyi]
                for u in range(n_iterations):
                    proxy_vals_age_unc = np.random.normal(0,proxy_data['uncertainty']*unc_scale)
                    proxy_vals_age_sel = proxy_vals_age[proxyi]
                    proxy_vals_age_sel[:,0] += proxy_vals_age_unc[proxyi]
                    proxy_vals_idx = np.abs(np.nansum(np.subtract(proxy_estimates_all_age,proxy_vals_age_sel),axis=0)**2)
                    [xb_idx.append(x) for x in np.where(proxy_vals_idx<=np.nanquantile(proxy_vals_idx,(percent/100)))[0]]    
                recon_ens[age_counter,:,lati,loni] = xb_idx[:np.shape(recon_ens)[1]]
            else: print('No data within geographic range at age '+str(age)+': lat='+str(lat)+'; lon='+str(lon))
            
            

#%%


    #%%    #%%
          
    #%%

# for age_counter,age in enumerate(proxy_data['age_centers']): 
#     xb_idx = []
#     for u in range(n_iterations):
#         proxy_vals_age_unc = np.random.normal(0,proxy_data['uncertainty'])*0.5
#         proxy_vals_age_sel = proxy_vals_age[]
#         proxy_estimates_all_age = proxy_estimates_all*1
#         #i=np.where((model_data['age'] <= age-options['prior_window']/2) | (model_data['age'] >= age+options['prior_window']/2))[0]
#         proxy_vals_idx = np.abs(np.nansum(np.subtract(proxy_estimates_all_age,np.expand_dims(proxy_vals_age+proxy_vals_age_unc,1)),axis=0))**2
#         #proxy_vals_idx[i]*=np.nan
#         [xb_idx.append(x) for x in np.where(proxy_vals_idx<=np.nanquantile(proxy_vals_idx,(percent/100)))[0]]    
#     x.append(np.array(xb_idx))

# #x=np.array(x)
# #    proxy_estimate_age = 

#%%
import xarray as xr
import numpy as np
import matplotlib
import matplotlib.pyplot   as plt         # Packages for making figures
import matplotlib.cm       as cm
import matplotlib.gridspec as gridspec
import cartopy.crs         as ccrs        # Packages for mapping in python
import cartopy.feature     as cfeature
import scipy.stats
import cartopy.util        as cutil
import regionmask as rm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import da_psms
fig, ax = plt.subplots(figsize =(10, 7))
ax.hist(proxy_data['values_binned'].flatten(),alpha=0.5,density=True,label='Proxy')
ax.hist(proxy_estimates_all.flatten(),alpha=0.5,density=True,label='PSM')
ax.legend()
# Show plot
plt.show()

#
c='teal'
plt.figure(figsize=(6,6),dpi=400)
ax=plt.subplot()
for age_counter,age in enumerate(proxy_data['age_centers']):
    ax.scatter([age]*int(len(x[age_counter])),model_data['age'][x[age_counter]],c='k',alpha=0.05,s=1)
ax.fill_between(proxy_data['age_centers'], [np.quantile(model_data['age'][i],0.005) for i in x], [np.quantile(model_data['age'][i],0.995) for i in x], facecolor=c, alpha=0.1)
ax.fill_between(proxy_data['age_centers'], [np.quantile(model_data['age'][i],0.250) for i in x], [np.quantile(model_data['age'][i],0.750) for i in x], facecolor=c, alpha=0.3)
ax.plot(proxy_data['age_centers'],[np.mean(model_data['age'][i]) for i in x],c=c,lw=2)
ax.invert_xaxis()
ax.set_ylabel("Ages of matching model states");ax.set_xlabel("Age of proxy bins");
ax.set_xlim(22000,0);ax.set_ylim(0,22000)
ax.plot([22000,0],[22000,0], c='grey',linestyle='-.',linewidth=2)
plt.show()
#

#
z = []
ages=[]
for age_counter,age in enumerate(model_data['age']):
    z.append(np.sum(np.concatenate(x)==age_counter))
    ages.append(age)
    

plt.figure(figsize=(9,9),dpi=400)
ax=plt.subplot()
ax.bar(ages,100*np.array(z)/(np.shape(x)[0]*n_iterations),facecolor=c,width=10)
ax.invert_xaxis()
ax.set_ylabel("Percent of times used used");ax.set_xlabel("Age of Model States");
ax.set_xlim(np.max(ages),0);#ax.set_ylim(0,1)
plt.show()

#%%
var_name='LakeStatus'
age = 8000
ageref = 0
agei = np.argmin(abs(proxy_data['age_centers']-age))
xi = x[agei]
latlonbin=[5,5]



#%%
precipRecon = np.mean(model_data[var_name+'_annual'][xi],axis=0)
precipRecon -= np.mean(model_data[var_name+'_annual'][x[np.argmin(abs(proxy_data['age_centers']-ageref))]],axis=0)
v = np.nanquantile(np.abs(precipRecon),0.99)

plt.figure(figsize=(10,5),dpi=400)
ax = plt.subplot(projection=ccrs.Robinson()) 
ax.spines['geo'].set_edgecolor('black'); ax.set_global(); ax.coastlines(lw=0.5)
ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,lw=0.9, color='k', alpha=0.4, linestyle=(0,(5,10)))
p = ax.pcolormesh(model_data['lon'],model_data['lat'],(precipRecon),cmap='BrBG',vmin=-v,vmax=v,transform=ccrs.PlateCarree())
#

lats,lons,proxyarray = [],[],[]
for lat in range(-90,91,latlonbin[0]):
    for lon in range(0,360,latlonbin[1]):
        proxyi = np.where((proxy_data['lats']>=lat-latlonbin[0]/2) & (proxy_data['lats']< lat+latlonbin[0]/2) &(proxy_data['lons']>=lon-latlonbin[1]/2) & (proxy_data['lons']< lon+latlonbin[1]/2))[0]
        if len(proxyi) > 0:   
            proxyarray= np.mean(proxy_data['values_binned'][proxyi],axis=0)
            proxyarray= proxyarray[np.argmin(np.abs(proxy_data['age_centers']-age))] - proxyarray[np.argmin(np.abs(proxy_data['age_centers']-ageref))] 
            ax.scatter(lon,lat,c=proxyarray,
                       vmin=-v,vmax=v,cmap='BrBG',s=30,ec='k',lw=1,transform=ccrs.PlateCarree())
    #Colorbar
cbar = plt.colorbar(p,orientation='horizontal',ticks=[-v,-v/2,0,v/2,v],extend='both',fraction=0.05,aspect=30,pad=0.05)#,cax=inset_axes(ax,width="7%",height="80%",bbox_to_anchor=(0,0,1,1),bbox_transform=ax.transAxes, loc="center left"))
plt.suptitle(str(int(age/1000))+' - 0 ka [Reconstructed & Proxies] '+var_name+'')
plt.tight_layout()



#%%
precipRecon = np.mean(model_data[var_name+'_annual'][xi],axis=0)
precipRecon -= np.mean(model_data[var_name+'_annual'][x[np.argmin(abs(proxy_data['age_centers']-ageref))]],axis=0)
precipInput = np.mean(model_data[var_name+'_annual'][(model_data['age']>age-500) & (model_data['age']<age+500)],axis=0)
precipInput -= np.mean(model_data[var_name+'_annual'][(model_data['age']>ageref-500) & (model_data['age']<ageref+500)],axis=0)
v = np.nanquantile(np.abs(precipRecon-precipInput),0.99)

plt.figure(figsize=(10,5),dpi=400)
ax = plt.subplot(projection=ccrs.Robinson()) 
ax.spines['geo'].set_edgecolor('black'); ax.set_global(); ax.coastlines(lw=0.5)
ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,lw=0.9, color='k', alpha=0.4, linestyle=(0,(5,10)))
p = ax.pcolormesh(model_data['lon'],model_data['lat'],(precipRecon-precipInput),cmap='BrBG',vmin=-v,vmax=v,transform=ccrs.PlateCarree())
#


cbar = plt.colorbar(p,orientation='horizontal',ticks=[-v,-v/2,0,v/2,v],extend='both',fraction=0.05,aspect=30,pad=0.05)#,cax=inset_axes(ax,width="7%",height="80%",bbox_to_anchor=(0,0,1,1),bbox_transform=ax.transAxes, loc="center left"))
plt.suptitle(str(int(age/1000))+'- 0 ka [Reconstructed - Input '+var_name+']')
plt.tight_layout()
#%%


var_name='LakeStatus'
age = 6000
ageref = 0
agei = np.argmin(abs(proxy_data['age_centers']-age))
latlonbin=[5,5]

valsi = recon_ens[agei]

var_recon_age = recon_ens[agei,0]*np.nan
for lati,lat in enumerate(lats):
    for loni,lon in  enumerate(lons):
        idx = valsi[:,lati,loni]
        if sum(np.isfinite(idx)) > 0:
            vals = np.nanmean(model_data[var_name+'_annual'][[int(x) for x in idx],lati,loni])
        else: vals=np.nan
        var_recon_age[lati,loni] = vals
        

#
precipInput = np.mean(model_data[var_name+'_annual'][(model_data['age']>age-500) & (model_data['age']<age+500)],axis=0)
#precipInput -= np.mean(model_data[var_name+'_annual'][(model_data['age']>ageref-500) & (model_data['age']<ageref+500)],axis=0)
#precipRecon = np.mean(model_data[var_name+'_annual'][xi],axis=0)
#precipRecon -= np.mean(model_data[var_name+'_annual'][x[np.argmin(abs(proxy_data['age_centers']-ageref))]],axis=0)
v = np.nanquantile(np.abs(var_recon_age-precipInput),0.99)
#%%


plt.figure(figsize=(10,5),dpi=400)
ax = plt.subplot(projection=ccrs.Robinson()) 
ax.spines['geo'].set_edgecolor('black'); ax.set_global(); ax.coastlines(lw=0.5)
ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,lw=0.9, color='k', alpha=0.4, linestyle=(0,(5,10)))


p = ax.pcolormesh(model_data['lon'],model_data['lat'],(var_recon_age-precipInput),cmap='BrBG',vmin=-v,vmax=v,transform=ccrs.PlateCarree())


for lat in range(-90,91,latlonbin[0]):
    for lon in range(0,360,latlonbin[1]):
        proxyi = np.where((proxy_data['lats']>=lat-latlonbin[0]/2) & (proxy_data['lats']< lat+latlonbin[0]/2) &(proxy_data['lons']>=lon-latlonbin[1]/2) & (proxy_data['lons']< lon+latlonbin[1]/2))[0]
        if len(proxyi) > 0:   
            proxyarray= np.mean(proxy_data['values_binned'][proxyi],axis=0)
            proxyarray= proxyarray[np.argmin(np.abs(proxy_data['age_centers']-age))] - proxyarray[np.argmin(np.abs(proxy_data['age_centers']-ageref))] 
            if np.isfinite(proxyarray):
                ax.scatter(lon,lat,c=proxyarray,
                           vmin=-v,vmax=v,cmap='BrBG',s=30,ec='k',lw=1,transform=ccrs.PlateCarree())
    #Colorbar
cbar = plt.colorbar(p,orientation='horizontal',ticks=[-v,-v/2,0,v/2,v],extend='both',fraction=0.05,aspect=30,pad=0.05)#,cax=inset_axes(ax,width="7%",height="80%",bbox_to_anchor=(0,0,1,1),bbox_transform=ax.transAxes, loc="center left"))
#plt.suptitle(str(int(age/1000))+' - 0 ka [Reconstructed & Proxies] '+var_name+'')
plt.suptitle(str(int(age/1000))+' - 0 ka [Reconstructed - Input] '+var_name+'')
plt.tight_layout()



 #%%

proxy_data.keys()
var_name= 'LakeStatus'
lat = 23
lon = 31
i = np.where(proxy_data['lons']==lon)[0]

modelvals = model_data[var_name+'_annual'][:,np.argmin(abs(model_data['lat']-lat)),np.argmin(abs(model_data['lon']-lon))]
proxy_estimates_all_age=proxy_estimates_all[i]

plt.figure()
plt.plot(model_data['age'],modelvals,label='model',alpha=0.5)
plt.plot(model_data['age'],proxy_estimates_all_age[0],label='psm',alpha=0.5)
plt.scatter(proxy_data['age_centers'],proxy_data['values_binned'][i],label='proxy',color='goldenrod')
plt.gca().invert_xaxis()
plt.legend()
plt.show()

ii = np.argmin(np.abs(np.nansum(np.subtract(proxy_estimates_all_age,np.array([[1]])),axis=0)**2))
#%%