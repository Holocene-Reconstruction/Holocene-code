#==============================================================================
# This script contains the main code of the Holocene data assimilation.
# Options are set in the "config.yml" file.  See README.txt for a more complete
# explanation of the code and setup.
#    author: Michael P. Erb
#    date  : 10/7/2020
#==============================================================================

import sys
import da_utils
import da_utils_lmr
import da_load_data
import numpy as np
import netCDF4
from datetime import datetime
import time
import yaml
import matplotlib.pyplot as plt


### OPTIONS

starttime_total = time.time() # Start timer

# The config file can be speficied when running the code.
if len(sys.argv) > 1: config_file = sys.argv[1]
else: config_file = 'config_default.yml'
print('Using configuration file: '+config_file)
with open(config_file,'r') as file:
    options = yaml.load(file,Loader=yaml.FullLoader)

# Print the options
print('=== OPTIONS ===')
for key in options.keys(): print(key+': '+str(options[key]))
print('=== END OPTIONS ===')


### LOAD DATA

# Load the chosen proxy model data
tas_model,age_model,time_ndays_model,valid_model_inds,model_num,lat_model,lon_model = da_load_data.load_model_data(options)

# Load the chosen proxy data
filtered_ts,collection_all = da_load_data.load_proxies(options)


### SET THINGS UP

# Set age range to reconstruct, as wel as the reference period
age_bounds = np.arange(options['age_range_to_reconstruct'][0],options['age_range_to_reconstruct'][1]+1,options['time_resolution'])
age_centers = (age_bounds[:-1]+age_bounds[1:])/2
age_bounds_ref = np.arange(options['reference_period'][0],options['reference_period'][1]+1,options['time_resolution'])
age_centers_ref = (age_bounds_ref[:-1]+age_bounds_ref[1:])/2

# Get dimensions
n_models  = len(options['models_for_prior'])
n_ages    = len(age_centers)
n_proxies = len(filtered_ts)
n_lat     = tas_model.shape[2]
n_lon     = tas_model.shape[3]
n_latlon  = n_lat*n_lon
n_state = n_latlon + n_proxies

# Set up arrays (y, ya, HXb, and R)
proxy_values_all       = np.zeros((n_proxies,n_ages));         proxy_values_all[:]      = np.nan  # y
proxy_resolution_all   = np.zeros((n_proxies,n_ages));         proxy_resolution_all[:]  = np.nan  # ya
proxy_uncertainty_all  = np.zeros((n_proxies));                proxy_uncertainty_all[:] = np.nan  # R
proxy_metadata_all     = np.zeros((n_proxies,8),dtype=object); proxy_metadata_all[:]    = np.nan
proxy_estimates_all    = np.array([dict() for k in range(n_proxies)])  # HXb

# Compute annual means of the model data
time_ndays_model_latlon = np.repeat(np.repeat(time_ndays_model[:,:,None,None],n_lat,axis=2),n_lon,axis=3)
tas_model_annual = np.average(tas_model,axis=1,weights=time_ndays_model_latlon)

# Set the maximum proxy resolution
max_res_value = int(options['maximum_resolution']/options['time_resolution'])

# If requested, use the uncertainty values from a file.
if options['filename_mse'] != 'None':
    print(' === USING UNCERTAINTY VALUES FROM THE FOLLOWING FILE ===')
    print(options['filename_mse'])
    proxy_uncertainties_from_file = np.genfromtxt(options['filename_mse'],delimiter=',',dtype='str')

# Loop through proxies, saving the necessary values to common variables.
no_ref_data = 0; missing_uncertainty = 0
i=0
for i in range(n_proxies):
    #
    if options['verbose_level'] > 0: print(' - Calculating estimates for proxy '+str(i)+'/'+str(n_proxies))
    #
    # Get proxy data
    proxy_ages = np.array(filtered_ts[i]['age']).astype(np.float)
    proxy_data = np.array(filtered_ts[i]['paleoData_values']).astype(np.float)
    #
    # If any NaNs exist in the ages, remove those values
    proxy_data = proxy_data[np.isfinite(proxy_ages)]
    proxy_ages = proxy_ages[np.isfinite(proxy_ages)]
    #
    # Make sure the ages go from newest to oldest
    proxy_median_res = np.median(proxy_ages[1:] - proxy_ages[:-1])
    if proxy_median_res < 0:
        proxy_data = np.flip(proxy_data)
        proxy_ages = np.flip(proxy_ages)
    #
    # compute age bounds of the proxy observations as the midpoints between data
    proxy_age_bounds = (proxy_ages[1:]+proxy_ages[:-1])/2
    end_newest = proxy_ages[0]  - (proxy_ages[1]-proxy_ages[0])/2
    end_oldest = proxy_ages[-1] + (proxy_ages[-1]-proxy_ages[-2])/2
    proxy_age_bounds = np.insert(proxy_age_bounds,0,end_newest)
    proxy_age_bounds = np.append(proxy_age_bounds,end_oldest)
    #
    # Interpolate proxy data to the base resolution, using nearest-neighbor interpolation
    #TODO: Consider removing the reference period before shortening the proxies, so that the code will still work if these two periods don't overlap.
    proxy_data_12ka = np.zeros((n_ages)); proxy_data_12ka[:] = np.nan
    proxy_res_12ka  = np.zeros((n_ages)); proxy_res_12ka[:]  = np.nan
    proxy_data_12ka_ref = np.zeros((len(age_centers_ref))); proxy_data_12ka_ref[:] = np.nan
    for j in range(len(proxy_age_bounds)-1):
        indices_selected = np.where((age_centers >= proxy_age_bounds[j]) & (age_centers < proxy_age_bounds[j+1]))[0]
        proxy_data_12ka[indices_selected] = proxy_data[j]
        proxy_res_12ka[indices_selected]  = int(round((proxy_age_bounds[j+1] - proxy_age_bounds[j]) / options['time_resolution']))
        #
        indices_selected_ref = np.where((age_centers_ref >= proxy_age_bounds[j]) & (age_centers_ref < proxy_age_bounds[j+1]))[0]
        proxy_data_12ka_ref[indices_selected_ref] = proxy_data[j]
    #
    #plt.plot(age_centers,proxy_data_12ka)
    #plt.show()
    #
    # Set resolutions to a minimum of 1 and a maximum of max_res_value
    proxy_res_12ka[proxy_res_12ka == 0] = 1
    proxy_res_12ka[proxy_res_12ka > max_res_value] = max_res_value
    #
    # Remove the mean of the reference period
    proxy_data_12ka_anom = proxy_data_12ka - np.nanmean(proxy_data_12ka_ref)
    if np.isnan(proxy_data_12ka_ref).all(): print('No data in reference period, index: '+str(i)); no_ref_data += 1
    #
    # Save to common variables (y and ya)
    proxy_values_all[i,:]     = proxy_data_12ka_anom
    proxy_resolution_all[i,:] = proxy_res_12ka
    #
    # Get proxy metdata
    proxy_lat                 = filtered_ts[i]['geo_meanLat']
    proxy_lon                 = filtered_ts[i]['geo_meanLon']
    proxy_seasonality_txt     = filtered_ts[i]['paleoData_interpretation'][0]['seasonality']
    proxy_seasonality_general = filtered_ts[i]['paleoData_interpretation'][0]['seasonalityGeneral']
    try:    proxy_uncertainty = filtered_ts[i]['paleoData_temperature12kUncertainty']
    except: proxy_uncertainty = np.nan; missing_uncertainty += 1
    if proxy_uncertainty  == 'NA': proxy_uncertainty = np.nan
    #
    # Convert seasonality to a list of months, with negative values corresponding to the previous year.
    try:
        proxy_seasonality_array = filtered_ts[i]['seasonality_array']
    except:
        proxy_seasonality = da_utils.interpret_seasonality(proxy_seasonality_txt,proxy_lat,'annual')
        proxy_seasonality_array = np.array(proxy_seasonality.split()).astype(np.int)
    #
    # If requested, prescribe seasonalities
    if options['assign_seasonality'] == 'annual':
        proxy_seasonality_array = np.array([1,2,3,4,5,6,7,8,9,10,11,12])
    elif options['assign_seasonality'] == 'summer':
        if proxy_lat >= 0: proxy_seasonality_array = np.array([6,7,8])
        else:              proxy_seasonality_array = np.array([-12,1,2])
    elif options['assign_seasonality'] == 'winter':
        if proxy_lat >= 0: proxy_seasonality_array = np.array([-12,1,2])
        else:              proxy_seasonality_array = np.array([6,7,8])
    #
    # Save some metadata to a common variable
    proxy_uncertainty_all[i] = np.square(proxy_uncertainty)  # Proxy uncertainty was give as RMSE, but the code uses MSE
    proxy_metadata_all[i,0] = filtered_ts[i]['dataSetName']
    proxy_metadata_all[i,1] = filtered_ts[i]['paleoData_TSid']
    proxy_metadata_all[i,2] = str(filtered_ts[i]['geo_meanLat'])
    proxy_metadata_all[i,3] = str(filtered_ts[i]['geo_meanLon'])
    proxy_metadata_all[i,4] = str(proxy_seasonality_array)
    proxy_metadata_all[i,5] = proxy_seasonality_general
    proxy_metadata_all[i,6] = str(np.median(proxy_ages[1:]-proxy_ages[:-1]))
    proxy_metadata_all[i,7] = collection_all[i]
    #
    # If requested, adjust the uncertainty with a multiplier
    if options['uncertainty_multiplier'] != 1:
        proxy_uncertainty_all[i] = proxy_uncertainty_all[i]*options['uncertainty_multiplier']
    #
    # If requested, replace the uncertainty will a value from a file which contains TSids and MSE uncertainty values
    if options['filename_mse'] != 'None':
        index_uncertainty = np.where(filtered_ts[i]['paleoData_TSid'] == proxy_uncertainties_from_file[:,0])[0]
        proxy_uncertainty_all[i] = proxy_uncertainties_from_file[index_uncertainty,1].astype(np.float)
    #
    # Find the model gridpoint closest to the proxy location
    if proxy_lon < 0: proxy_lon = proxy_lon+360
    lon_model_wrapped = np.append(lon_model,lon_model[0]+360)
    j_selected = np.argmin(np.abs(lat_model-proxy_lat))
    i_selected = np.argmin(np.abs(lon_model_wrapped-proxy_lon))
    if np.abs(proxy_lat-lat_model[j_selected])         > 2: print('WARNING: Too large of a lat difference. Proxy lat: '+str(proxy_lat)+', model lat: '+str(lat_model[j_selected]))
    if np.abs(proxy_lon-lon_model_wrapped[i_selected]) > 2: print('WARNING: Too large of a lon difference. Proxy lon: '+str(proxy_lon)+', model lon: '+str(lon_model_wrapped[i_selected]))
    if i_selected == len(lon_model_wrapped)-1: i_selected = 0
    if options['verbose_level'] > 1: print('Proxy location vs. nearest model gridpoint.  Lat: '+str(proxy_lat)+', '+str(lat_model[j_selected])+'.  Lon: '+str(proxy_lon)+', '+str(lon_model[i_selected]))
    tas_model_location = tas_model[:,:,j_selected,i_selected]
    #
    # Compute an average over months according to the proxy seasonality #TODO: Can this be done better?
    proxy_seasonality_indices = np.abs(proxy_seasonality_array)-1
    proxy_seasonality_indices[proxy_seasonality_indices > 11] = proxy_seasonality_indices[proxy_seasonality_indices > 11] - 12
    tas_model_location_season = np.average(tas_model_location[:,proxy_seasonality_indices],weights=time_ndays_model[:,proxy_seasonality_indices],axis=1)
    #
    # If a PSM is available, use it to transform the temperature data into the proxy units
    try:    use_psm = filtered_ts[i]['psm']['use_psm']
    except: use_psm = False
    if use_psm:
        #print('Using PSM:'+str(i))
        psm_slope         = filtered_ts[i]['psm']['slope']
        psm_intercept     = filtered_ts[i]['psm']['intercept']
        proxy_uncertainty = filtered_ts[i]['psm']['R']
        proxy_uncertainty_all[i] = proxy_uncertainty
        tas_model_location_season = (tas_model_location_season*psm_slope) + psm_intercept
    #
    # Find all time resolutions in the record
    proxy_res_12ka_unique = np.unique(proxy_res_12ka)
    proxy_res_12ka_unique_sorted = np.sort(proxy_res_12ka_unique[np.isfinite(proxy_res_12ka_unique)]).astype(np.int)
    #
    # Loop through each time resolution, computing a running mean of the selected duration and save the values to a common variable
    for res in proxy_res_12ka_unique_sorted:
        tas_model_location_season_nyear_mean = np.convolve(tas_model_location_season,np.ones((res,))/res,mode='same')
        proxy_estimates_all[i][int(res)] = tas_model_location_season_nyear_mean

print('Finished preprocessing proxies and making model-based proxy estimates.')

# Calculate the localization matrix, if needed
proxy_localization_all = da_load_data.loc_matrix(options,lon_model,lat_model,proxy_metadata_all)

# Determine the number of ensemble members
n_ens = len(da_load_data.get_age_indices_for_prior(options,age_model,0,valid_model_inds))

# Randomly select a the ensemble members to save (to reduce output filesizes)
np.random.seed(seed=0)  #TODO: Allow this seed to be changed in the future?
n_ens_tosave = min([n_ens,100])
rens = np.random.choice(n_ens,n_ens_tosave,replace=False)

# Set up arrays for mean, median, and ensembles
recon_mean    = np.zeros((n_state,n_ages));              recon_mean[:]    = np.nan
recon_median  = np.zeros((n_state,n_ages));              recon_median[:]  = np.nan
recon_ens     = np.zeros((n_state,n_ens_tosave,n_ages)); recon_ens[:]     = np.nan
recon_gmt_all = np.zeros((n_ages,n_ens));                recon_gmt_all[:] = np.nan
prior_gmt_all = np.zeros((n_ages,n_ens));                prior_gmt_all[:] = np.nan
ind_with_data_all = np.zeros((n_ages,n_proxies));        ind_with_data_all[:] = np.nan

# If requested, select the portion of the proxies which are to be assimilated
if options['percent_to_assimilate'] < 100:
    np.random.seed(seed=options['seed_for_proxy_choice'])
    n_proxies_to_assimilate = int(n_proxies*(options['percent_to_assimilate']/100))
    proxy_ind_to_assimilate = np.random.choice(n_proxies,n_proxies_to_assimilate,replace=False)
    proxy_ind_to_assimilate = np.sort(proxy_ind_to_assimilate)
else:
    proxy_ind_to_assimilate = np.arange(n_proxies)

proxy_ind_to_assimilate_boolean = np.full((n_proxies),False,dtype=bool)
proxy_ind_to_assimilate_boolean[proxy_ind_to_assimilate] = True

# Loop through every age, doing the data assimilation with a time-varying prior
print('Starting data assimilation')
age_counter = 0; age = age_centers[age_counter]
for age_counter,age in enumerate(age_centers):
    #
    starttime_loop = time.time()
    #
    # Get all proxy values and resolutions for the current age
    proxy_values_for_age     = proxy_values_all[:,age_counter]
    proxy_resolution_for_age = proxy_resolution_all[:,age_counter]
    #
    # Get the indices of the prior which will be used for this data assimilation step
    age_indices_for_prior = da_load_data.get_age_indices_for_prior(options,age_model,age,valid_model_inds)
    if len(age_indices_for_prior) != n_ens: print(' !!! Warning: number of prior ages selected does not match n_ens.  Age='+str(age))
    #
    # Get the prior values and remove the mean for each model used in the prior seperately
    age_model_for_prior = age_model[age_indices_for_prior]
    age_indices_for_prior_model = {}
    for i in range(n_models):
        indices_for_model = np.where(model_num == (i+1))[0]
        age_indices_for_prior_model[i] = np.intersect1d(age_indices_for_prior,indices_for_model)
        tas_model_annual_for_prior_model = tas_model_annual[age_indices_for_prior_model[i],:,:] - np.mean(tas_model_annual[age_indices_for_prior_model[i],:,:],axis=0)  #TODO: Should the median be removed instead of the mean?
        if i == 0: tas_model_annual_for_prior = tas_model_annual_for_prior_model
        else:      tas_model_annual_for_prior = np.concatenate((tas_model_annual_for_prior,tas_model_annual_for_prior_model),axis=0)
    #
    # For each proxy, get the proxy estimates for the correct resolution
    model_estimates_for_age = np.zeros((n_ens,n_proxies)); model_estimates_for_age[:] = np.nan
    for j in range(n_proxies):
        res = proxy_resolution_for_age[j]
        if np.isnan(res): continue
        model_estimates_chosen_all = proxy_estimates_all[j][int(res)]
        for i in range(n_models):
            model_estimates_chosen_model = model_estimates_chosen_all[age_indices_for_prior_model[i]] - np.mean(model_estimates_chosen_all[age_indices_for_prior_model[i]])
            if i == 0: model_estimates_chosen = model_estimates_chosen_model
            else:      model_estimates_chosen = np.concatenate((model_estimates_chosen,model_estimates_chosen_model),axis=0)
        #
        # Save to a common variable
        model_estimates_for_age[:,j] = model_estimates_chosen
    #
    # Make the prior (Xb)
    prior = np.reshape(tas_model_annual_for_prior,(n_ens,n_latlon))
    #
    # Append the proxy estimate to the prior, so that proxy estimates are reconstructed too
    #TODO: Should I reconstruct proxies on their own age resolution (which means that the reconstructed proxies are the same length
    # as the real proxies) or should I reconstruct proxies at the reconstruction's base resolution?
    prior = np.append(prior,model_estimates_for_age,axis=1)
    Xb = np.transpose(prior)
    #
    # Select only the proxies which meet the criteria
    ind_with_data = np.isfinite(proxy_uncertainty_all) & np.isfinite(proxy_values_for_age) & proxy_ind_to_assimilate_boolean
    proxy_med_res = proxy_metadata_all[:,6].astype(np.float)
    if options['resolution_band'][0] != 'None': ind_with_data = ind_with_data & (proxy_med_res >= options['resolution_band'][0])
    if options['resolution_band'][1] != 'None': ind_with_data = ind_with_data & (proxy_med_res < options['resolution_band'][1])
    ind_with_data_all[age_counter,:] = ind_with_data  # Keep a record of which proxies are assimilated
    #
    # If valid proxies are present for this time step, do the data assimilation
    kidx = np.where(ind_with_data)[0]
    if kidx.shape[0] > 0:
        #
        proxy_values_selected       = proxy_values_for_age[kidx]
        proxy_uncertainty_selected  = proxy_uncertainty_all[kidx]
        model_estimates_selected    = model_estimates_for_age[:,kidx]
        proxy_localization_selected = proxy_localization_all[kidx,:]
        R_diagonal = np.diag(proxy_uncertainty_selected)
        #
        # Do the DA update, either together or one at a time.
        if options['assimate_together']:
            Xa,_,_ = da_utils.damup(Xb,np.transpose(model_estimates_selected),R_diagonal,proxy_values_selected)
        else:
            #
            # Loop through the proxies, updating the prior.
            n_proxies_at_age = kidx.shape[0]
            for proxy in range(n_proxies_at_age):
                #
                if options['verbose_level'] > 2: print(' - Assimilating proxy '+str(proxy)+'/'+str(n_proxies_at_age))
                #
                # Get values for proxy
                proxy_value              = proxy_values_selected[proxy]
                proxy_uncertainty        = proxy_uncertainty_selected[proxy]
                tas_modelbased_estimates = Xb[n_latlon+kidx[proxy],:]
                loc                      = proxy_localization_selected[proxy,:]
                #
                # Do data assimilation
                if options['localization_radius'] == 'None': loc = None
                updated_prior = da_utils_lmr.enkf_update_array(Xb,proxy_value,tas_modelbased_estimates,proxy_uncertainty,loc=loc,inflate=None)
                if np.isnan(updated_prior).all(): print(' !!! WARNING.  ALL RECONSTRUCTION VALUES SET TO NAN.  Age='+str(age)+', proxy number='+str(proxy)+' !!!')
                #
                # Set updated prior as prior
                Xb = updated_prior
            #
            # Set the final values
            Xa = Xb
        #
    else:
        # No proxies are assimilated
        Xa = Xb
    #
    # Compute the global-mean of the prior
    prior_gmt = da_utils.global_mean(tas_model_annual_for_prior,lat_model,1,2)
    prior_gmt_all[age_counter,:] = prior_gmt
    #
    # Compute the global-mean of the reconstruction
    Xa_latlon = np.reshape(Xa[:(n_lat*n_lon),:],(n_lat,n_lon,n_ens))
    recon_gmt = da_utils.global_mean(Xa_latlon,lat_model,0,1)
    recon_gmt_all[age_counter,:] = recon_gmt
    #
    # Get the mean, median, and selected ensemble values
    recon_mean[:,age_counter]   = np.mean(Xa,axis=1)
    recon_median[:,age_counter] = np.median(Xa,axis=1)
    recon_ens[:,:,age_counter]  = Xa[:,rens]
    #
    # Note progression of the reconstruction
    print('Time step '+str(age_counter)+'/'+str(len(age_centers))+' complete.  Time: '+str('%1.2f' % (time.time()-starttime_loop))+' sec')

# Reshape the data arrays
recon_mean   = np.transpose(recon_mean)
recon_median = np.transpose(recon_median)
recon_ens    = np.swapaxes(recon_ens,0,2)

# Reshape the gridded reconstruction to a lat-lon grid
recon_mean_grid   = np.reshape(recon_mean[:,:(n_lat*n_lon)],  (n_ages,             n_lat,n_lon))
recon_median_grid = np.reshape(recon_median[:,:(n_lat*n_lon)],(n_ages,             n_lat,n_lon))
recon_ens_grid    = np.reshape(recon_ens[:,:,:(n_lat*n_lon)], (n_ages,n_ens_tosave,n_lat,n_lon))

# Put the proxy reconstructions into separate variables
recon_mean_proxies   = recon_mean[:,(n_lat*n_lon):]
recon_median_proxies = recon_median[:,(n_lat*n_lon):]
recon_ens_proxies    = recon_ens[:,:,(n_lat*n_lon):]

# Store the options into as list, for saving purposes
n_options = len(options.keys())
options_list = []
for key,value in options.items():
    options_list.append(key+':'+str(value))


### SAVE THE OUTPUT

time_str = str(datetime.now()).replace(' ','_')
output_filename = 'holocene_recon_'+time_str+'_timevarying_'+str(options['prior_window'])+'yr_prior'
print('Saving the reconstruction as '+output_filename)

# Save all data into a netCDF file
output_dir = options['data_dir']+'results/'
outputfile = netCDF4.Dataset(output_dir+output_filename+'.nc','w')
outputfile.createDimension('state',       n_state)
outputfile.createDimension('age',         n_ages)
outputfile.createDimension('ens',         n_ens)
outputfile.createDimension('ens_selected',n_ens_tosave)
outputfile.createDimension('lat',         n_lat)
outputfile.createDimension('lon',         n_lon)
outputfile.createDimension('proxy',       n_proxies)
outputfile.createDimension('metadata',    proxy_metadata_all.shape[1])
outputfile.createDimension('options',     n_options)

output_recon_mean          = outputfile.createVariable('recon_mean',         'f4',('age','lat','lon',))  
output_recon_median        = outputfile.createVariable('recon_median',       'f4',('age','lat','lon',))  
output_recon_ens           = outputfile.createVariable('recon_ens',          'f4',('age','ens_selected','lat','lon',))
output_proxyrecon_mean     = outputfile.createVariable('proxyrecon_mean',    'f4',('age','proxy',))  
output_proxyrecon_median   = outputfile.createVariable('proxyrecon_median',  'f4',('age','proxy',))  
output_proxyrecon_ens      = outputfile.createVariable('proxyrecon_ens',     'f4',('age','ens_selected','proxy',))
output_recon_gmt           = outputfile.createVariable('recon_gmt',          'f4',('age','ens',))  
output_prior_gmt           = outputfile.createVariable('prior_gmt',          'f4',('age','ens',))  
output_ages                = outputfile.createVariable('ages',               'f4',('age',))
output_lat                 = outputfile.createVariable('lat',                'f4',('lat',))
output_lon                 = outputfile.createVariable('lon',                'f4',('lon',))
output_proxy_vals          = outputfile.createVariable('proxy_values',       'f4',('proxy','age',))
output_proxy_res           = outputfile.createVariable('proxy_resolutions',  'f4',('proxy','age',))
output_proxy_uncer         = outputfile.createVariable('proxy_uncertainty',  'f4',('proxy',))
output_metadata            = outputfile.createVariable('proxy_metadata',     'str',('proxy','metadata',))
output_options             = outputfile.createVariable('options',            'str',('options',))
output_proxies_selected    = outputfile.createVariable('proxies_selected',   'i1',('proxy',))
output_proxies_assimilated = outputfile.createVariable('proxies_assimilated','i1',('age','proxy',))

output_recon_mean[:]          = recon_mean_grid
output_recon_median[:]        = recon_median_grid
output_recon_ens[:]           = recon_ens_grid
output_proxyrecon_mean[:]     = recon_mean_proxies
output_proxyrecon_median[:]   = recon_median_proxies
output_proxyrecon_ens[:]      = recon_ens_proxies
output_recon_gmt[:]           = recon_gmt_all
output_prior_gmt[:]           = prior_gmt_all
output_ages[:]                = age_centers
output_lat[:]                 = lat_model
output_lon[:]                 = lon_model
output_proxy_vals[:]          = proxy_values_all
output_proxy_res[:]           = proxy_resolution_all
output_proxy_uncer[:]         = proxy_uncertainty_all
output_metadata[:]            = proxy_metadata_all
output_options[:]             = np.array(options_list)
output_proxies_selected[:]    = proxy_ind_to_assimilate_boolean.astype(int)
output_proxies_assimilated[:] = ind_with_data_all.astype(int)

outputfile.title = 'Holocene climate reconstruction'
outputfile.close()

endtime_total = time.time()  # End timer
print('Total time: '+str('%1.2f' % ((endtime_total-starttime_total)/60))+' minutes')
print(' === Reconstruction complete ===')
