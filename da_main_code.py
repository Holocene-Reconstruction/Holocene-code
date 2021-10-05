#==============================================================================
# This script contains the main code of the Holocene data assimilation.
# Options are set in the "config.yml" file.  See README.txt for a more complete
# explanation of the code and setup.
#    author: Michael P. Erb
#    date  : 10/4/2021
#==============================================================================

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


#%% OPTIONS

starttime_total = time.time() # Start timer

# Use a given config file.  If not given, use config.yml.
if len(sys.argv) > 1: config_file = sys.argv[1]
else:                 config_file = 'config_default.yml'
#else:                 config_file = 'config.yml'

# Load the configuration options and print them to the screen.
print('Using configuration file: '+config_file)
with open(config_file,'r') as file: options = yaml.load(file,Loader=yaml.FullLoader)

print('=== SETTINGS ===')
for key in options.keys():
    print('%30s: %-15s' % (key,str(options[key])))
print('=== END SETTINGS ===')


#%% LOAD AND PROCESS DATA

# Load the chosen proxy model data
if 
options['time_resolution_adjusted'] = int(options['time_resolution']*options['prior_time_factor'])
model_data = da_load_models.load_model_data(options)

# Detrend the model data if selected
model_data = da_load_models.detrend_model_data(model_data,options)

# Load the chosen proxy data
filtered_ts,collection_all = da_load_proxies.load_proxies(options)
proxy_data = da_load_proxies.process_proxies(filtered_ts,collection_all,options)

# Use PSMs to get model-based proxy estimates
proxy_estimates_all,_ = da_psms.psm_main(model_data,proxy_data,options)


#%% SET THINGS UP

# Get dimensions
n_vars            = len(options['vars_to_reconstruct'])
n_models_in_prior = len(options['models_for_prior'])
n_proxies         = proxy_data['values_binned'].shape[0]
n_ages            = proxy_data['values_binned'].shape[1]
n_lat             = len(model_data['lat'])
n_lon             = len(model_data['lon'])
n_latlonvars      = n_lat*n_lon*n_vars
n_state           = (n_latlonvars) + n_proxies

# Determine the number of ensemble members
n_ens = len(da_load_models.get_indices_for_prior(options,model_data,0))

# Randomly select a the ensemble members to save (to reduce output filesizes)
np.random.seed(seed=0)  #TODO: Allow this seed to be changed in the future?
n_ens_tosave = min([n_ens,100])
ind_tosave = np.random.choice(n_ens,n_ens_tosave,replace=False)

# Set up arrays for reconstruction values and more outputs
recon_ens         = np.zeros((n_state,n_ens_tosave,n_ages)); recon_ens[:]         = np.nan
recon_mean        = np.zeros((n_state,n_ages));              recon_mean[:]        = np.nan
recon_global_all  = np.zeros((n_ages,n_ens,n_vars));         recon_global_all[:]  = np.nan
recon_nh_all      = np.zeros((n_ages,n_ens,n_vars));         recon_nh_all[:]      = np.nan
recon_sh_all      = np.zeros((n_ages,n_ens,n_vars));         recon_sh_all[:]      = np.nan
prior_global_all  = np.zeros((n_ages,n_ens,n_vars));         prior_global_all[:]  = np.nan
prior_proxy_means = np.zeros((n_ages,n_proxies));            prior_proxy_means[:] = np.nan
proxies_to_assimilate_all = np.zeros((n_ages,n_proxies));    proxies_to_assimilate_all[:] = np.nan


#%% This section adjusts variables based on settings

# If requested, change uncertainty values
#TODO: This should probably be moved earlier, so that PSMs are cleared with the updated uncertainty values.  Where is the uncertainty scaling done too?
if options['change_uncertainty']:
    if options['change_uncertainty'][0:5] == 'mult_':
        uncertainty_multiplier = np.float(options['change_uncertainty'][5:])
        proxy_data['uncertainty'] = proxy_data['uncertainty']*options['uncertainty_multiplier']
        print(' --- Processing: All uncertainty values multiplied by '+str(uncertainty_multiplier)+' ---')
    if options['change_uncertainty'][0:4] == 'all_':
        prescribed_uncertainty = np.float(options['change_uncertainty'][4:])
        for i in range(n_proxies): proxy_data['uncertainty'][i] = prescribed_uncertainty
        print(' --- Processing: All uncertainty values set to '+str(prescribed_uncertainty)+' ---')
    else:
        # If using this option, the text file below should contain TSids and MSE for every proxy record
        print(' --- Processing: All uncertainty values set to values from the following file ---')
        print(options['change_uncertainty'])
        proxy_uncertainties_from_file = np.genfromtxt(options['change_uncertainty'],delimiter=',',dtype='str')
        #
        for i in range(n_proxies):
            index_uncertainty = np.where(proxy_data['metadata'][i,1] == proxy_uncertainties_from_file[:,0])[0]
            if len(index_uncertainty) == 0:
                print('No prescribed error value in file for proxy '+str(i)+', TSid: '+str(proxy_data['metadata'][i,1])+'.  Setting to NaN.')
                proxy_data['uncertainty'][i] = np.nan
            else:
                proxy_data['uncertainty'][i] = proxy_uncertainties_from_file[index_uncertainty,1].astype(np.float)

# If requested, select the portion of the proxies which are to be assimilated
#TODO: Should this be done after making other selections below?
proxy_ind_random_selected = np.full((n_proxies),False,dtype=bool)
if options['percent_to_assimilate'] < 100:
    print(' --- Processing: Choosing only '+str(options['percent_to_assimilate'])+'% of possible proxies ---')
    np.random.seed(seed=options['seed_for_proxy_choice'])
    n_proxies_to_choose = int(n_proxies*(options['percent_to_assimilate']/100))
    proxy_ind_random = np.random.choice(n_proxies,n_proxies_to_choose,replace=False)
    proxy_ind_random = np.sort(proxy_ind_random)
    proxy_ind_random_selected[proxy_ind_random] = True
else:
    proxy_ind_random_selected[:] = True

# If requested, select only certain archive types
if options['assimilate_selected_archives']:
    print(' --- Processing: Choosing only the following archive types ---')
    print(options['assimilate_selected_archives'])
    proxy_ind_of_archive_type = np.full((n_proxies),False,dtype=bool)
    ind_archives = [i for i, atype in enumerate(proxy_data['archivetype']) if atype in options['assimilate_selected_archives']]
    proxy_ind_of_archive_type[ind_archives] = True
    print(' - Number of records: '+str(sum(proxy_ind_of_archive_type)))
else:
    proxy_ind_of_archive_type = np.full((n_proxies),True,dtype=bool)

# If requested, select the proxies within the specified region
if options['assimilate_selected_region']:
    print(' --- Processing: Choosing proxies only the following region ---')
    print(options['assimilate_selected_region'])
    region_lat_min,region_lat_max,region_lon_min,region_lon_max = options['assimilate_selected_region']
    proxy_ind_in_region = (proxy_data['lats'] >= region_lat_min) & (proxy_data['lats'] <= region_lat_max) & (proxy_data['lons'] >= region_lon_min) & (proxy_data['lons'] <= region_lon_max)
else:
    proxy_ind_in_region = np.full((n_proxies),True,dtype=bool)

# If requested, select the proxies with median resolution within a certain window
if options['assimilate_selected_resolution']: 
    print(' --- Processing: Choosing proxies with with median resolution in the range '+str(options['assimilate_selected_resolution'])+' ---')
    proxy_med_res = proxy_data['metadata'][:,6].astype(np.float)
    proxy_ind_in_resolution_band = (proxy_med_res >= options['assimilate_selected_resolution'][0]) & (proxy_med_res < options['assimilate_selected_resolution'][1])
else:
    proxy_ind_in_resolution_band = np.full((n_proxies),True,dtype=bool)

# Calculate the localization matrix (it may not be used)
if options['assimate_together'] == False:
    proxy_localization_all = da_utils.loc_matrix(options,model_data,proxy_data)


#%%
# Loop through every age, doing the data assimilation with a time-varying prior
print(' === Starting data assimilation === ')
age_counter = 0; age = proxy_data['age_centers'][age_counter]
for age_counter,age in enumerate(proxy_data['age_centers']):
    #
    starttime_loop = time.time()
    #
    # Get all proxy values and resolutions for the current age
    proxy_values_for_age     = proxy_data['values_binned'][:,age_counter]
    proxy_resolution_for_age = proxy_data['resolution_binned'][:,age_counter]
    #
    # Get the indices of the prior which will be used for this data assimilation step
    indices_for_prior = da_load_models.get_indices_for_prior(options,model_data,age)
    model_number_for_prior = model_data['number'][indices_for_prior]
    if len(indices_for_prior) != n_ens: print(' !!! Warning: number of prior ages selected does not match n_ens.  Age='+str(age))
    #
    # Get the prior values for the variables to reconstruct
    for j,var_name in enumerate(options['vars_to_reconstruct']):
        var_annual_for_prior = model_data[var_name+'_annual'][indices_for_prior,:,:][:,:,:,None]
        if j == 0: vars_annual_for_prior_all = var_annual_for_prior
        else:      vars_annual_for_prior_all = np.concatenate((vars_annual_for_prior_all,var_annual_for_prior),axis=3)
    #
    # For each proxy, get the proxy estimates for the correct resolution
    model_estimates_for_age = np.zeros((n_ens,n_proxies)); model_estimates_for_age[:] = np.nan
    for j in range(n_proxies):
        res = proxy_resolution_for_age[j]
        #if np.isnan(res): continue  #TODO: This line reconstructs prior values even if they are nan.  Do I want to use this instead of the next line?
        if np.isnan(proxy_values_for_age[j]): continue
        model_estimates_for_age[:,j] = proxy_estimates_all[j][int(res)][indices_for_prior]  #TODO: Think more aboute this.  Should the averaging happen just for the covariances, or keep things as-is?
    #
    # For a relative reconstruction, remove the means of each model seperately
    if options['reconstruction_type'] == 'relative': 
        for i in range(n_models_in_prior):
            ind_for_model = np.where(model_number_for_prior == (i+1))[0]
            vars_annual_for_prior_all[ind_for_model,:,:,:] = vars_annual_for_prior_all[ind_for_model,:,:,:] - np.mean(vars_annual_for_prior_all[ind_for_model,:,:,:],axis=0)
            model_estimates_for_age[ind_for_model,:]       = model_estimates_for_age[ind_for_model,:]       - np.mean(model_estimates_for_age[ind_for_model,:],axis=0)
    #
    # Make the prior (Xb)
    prior = np.reshape(vars_annual_for_prior_all,(n_ens,n_latlonvars))
    #
    # Append the proxy estimate to the prior, so that proxy estimates are reconstructed too
    #TODO: Should I reconstruct proxies on their own age resolution (which means that the reconstructed proxies are the same length
    # as the real proxies) or should I reconstruct proxies at the reconstruction's base resolution?
    prior = np.append(prior,model_estimates_for_age,axis=1)
    Xb = np.transpose(prior)
    #
    # Save the prior estimates of proxies, for analysis later
    prior_proxy_means[age_counter,:] = np.mean(model_estimates_for_age,axis=0)
    #
    # Select only the proxies which meet the criteria
    proxies_to_assimilate = np.isfinite(proxy_data['uncertainty']) &\
                            np.isfinite(proxy_values_for_age) &\
                            proxy_ind_random_selected &\
                            proxy_ind_in_region &\
                            proxy_ind_of_archive_type &\
                            proxy_ind_in_resolution_band
    #
    # Keep a record of which proxies are assimilated
    proxies_to_assimilate_all[age_counter,:] = proxies_to_assimilate
    #
    # If valid proxies are present for this time step, do the data assimilation
    proxy_ind_to_assimilate = np.where(proxies_to_assimilate)[0]
    n_proxies_at_age = proxy_ind_to_assimilate.shape[0]
    if n_proxies_at_age > 0:
        #
        proxy_values_selected      = proxy_values_for_age[proxy_ind_to_assimilate]
        proxy_uncertainty_selected = proxy_data['uncertainty'][proxy_ind_to_assimilate]
        model_estimates_selected   = model_estimates_for_age[:,proxy_ind_to_assimilate]
        R_diagonal = np.diag(proxy_uncertainty_selected)
        #
        # Do the DA update, either together or one at a time.
        if options['assimate_together']:
            Xa,_,_ = da_utils.damup(Xb,np.transpose(model_estimates_selected),R_diagonal,proxy_values_selected)
        else:
            for proxy in range(n_proxies_at_age):
                if options['verbose_level'] > 2: print(' - Assimilating proxy '+str(proxy)+'/'+str(n_proxies_at_age))
                #
                # Get values for proxy
                proxy_value              = proxy_values_selected[proxy]
                proxy_uncertainty        = proxy_uncertainty_selected[proxy]
                tas_modelbased_estimates = Xb[n_latlonvars+proxy_ind_to_assimilate[proxy],:]
                if options['localization_radius']: loc = proxy_localization_all[proxy_ind_to_assimilate[proxy],:]
                else: loc = None
                #
                # Do data assimilation
                Xb = da_utils_lmr.enkf_update_array(Xb,proxy_value,tas_modelbased_estimates,proxy_uncertainty,loc=loc,inflate=None)
                if np.isnan(Xb).all(): print(' !!! ERROR.  ALL RECONSTRUCTION VALUES SET TO NAN.  Age='+str(age)+', proxy number='+str(proxy)+' !!!')
            #
            # Set the final values
            Xa = Xb
        #
    else:
        # No proxies are assimilated
        Xa = Xb
    #
    # Compute the global-mean of the prior
    prior_global = da_utils.global_mean(vars_annual_for_prior_all,model_data['lat'],1,2)
    prior_global_all[age_counter,:,:] = prior_global
    #
    # Compute the global and hemispheric means of the reconstruction
    Xa_latlon = np.reshape(Xa[:n_latlonvars,:],(n_lat,n_lon,n_vars,n_ens))
    recon_global = da_utils.global_mean(Xa_latlon,model_data['lat'],0,1)
    recon_nh     = da_utils.spatial_mean(Xa_latlon,model_data['lat'],model_data['lon'],  0,90,0,360,0,1)
    recon_sh     = da_utils.spatial_mean(Xa_latlon,model_data['lat'],model_data['lon'],-90, 0,0,360,0,1)
    recon_global_all[age_counter,:,:] = np.transpose(recon_global)
    recon_nh_all[age_counter,:,:]     = np.transpose(recon_nh)
    recon_sh_all[age_counter,:,:]     = np.transpose(recon_sh)
    #
    # Get the mean and selected ensemble values
    recon_mean[:,age_counter]  = np.mean(Xa,axis=1)
    recon_ens[:,:,age_counter] = Xa[:,ind_tosave]
    #
    # Note progression of the reconstruction
    print('Time step '+str(age_counter)+'/'+str(len(proxy_data['age_centers'] ))+' complete.  Time: '+str('%1.2f' % (time.time()-starttime_loop))+' sec')


# Reshape the data arrays
recon_mean = np.transpose(recon_mean)
recon_ens  = np.swapaxes(recon_ens,0,2)

# Reshape the gridded reconstruction to a lat-lon grid
recon_mean_grid = np.reshape(recon_mean[:,:(n_lat*n_lon*n_vars)], (n_ages,             n_lat,n_lon,n_vars))
recon_ens_grid  = np.reshape(recon_ens[:,:,:(n_lat*n_lon*n_vars)],(n_ages,n_ens_tosave,n_lat,n_lon,n_vars))

# Put the proxy reconstructions into separate variables
recon_mean_proxies = recon_mean[:,(n_lat*n_lon*n_vars):]
recon_ens_proxies  = recon_ens[:,:,(n_lat*n_lon*n_vars):]

# Store the options into as list to save
n_options = len(options.keys())
options_list = []
for key,value in options.items():
    options_list.append(key+':'+str(value))


#%% SAVE THE OUTPUT

#TODO: Rename some of the output variables to be more descriptive to their variable
time_str = str(datetime.datetime.now()).replace(' ','_')
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
outputfile.createDimension('metadata',    proxy_data['metadata'].shape[1])
outputfile.createDimension('options',     n_options)

output_recon_mean,output_recon_ens,output_recon_global,output_recon_nh,output_recon_sh,output_prior_global = {},{},{},{},{},{}
for i,var_name in enumerate(options['vars_to_reconstruct']):
    output_recon_mean[var_name]   = outputfile.createVariable('recon_'+var_name+'_mean',       'f4',('age','lat','lon',))
    output_recon_ens[var_name]    = outputfile.createVariable('recon_'+var_name+'_ens',        'f4',('age','ens_selected','lat','lon',))
    output_recon_global[var_name] = outputfile.createVariable('recon_'+var_name+'_global_mean','f4',('age','ens',))
    output_recon_nh[var_name]     = outputfile.createVariable('recon_'+var_name+'_nh_mean',    'f4',('age','ens',))
    output_recon_sh[var_name]     = outputfile.createVariable('recon_'+var_name+'_sh_mean',    'f4',('age','ens',))
    output_prior_global[var_name] = outputfile.createVariable('prior_'+var_name+'_global_mean','f4',('age','ens',))
    output_recon_mean[var_name][:]   = recon_mean_grid[:,:,:,i]
    output_recon_ens[var_name][:]    = recon_ens_grid[:,:,:,:,i]
    output_recon_global[var_name][:] = recon_global_all[:,:,i]
    output_recon_nh[var_name][:]     = recon_nh_all[:,:,i]
    output_recon_sh[var_name][:]     = recon_sh_all[:,:,i]
    output_prior_global[var_name][:] = prior_global_all[:,:,i]

output_proxyprior_mean     = outputfile.createVariable('proxyprior_mean',    'f4',('age','proxy',))
output_proxyrecon_mean     = outputfile.createVariable('proxyrecon_mean',    'f4',('age','proxy',))
output_proxyrecon_ens      = outputfile.createVariable('proxyrecon_ens',     'f4',('age','ens_selected','proxy',))
output_ages                = outputfile.createVariable('ages',               'f4',('age',))
output_lat                 = outputfile.createVariable('lat',                'f4',('lat',))
output_lon                 = outputfile.createVariable('lon',                'f4',('lon',))
output_proxy_vals          = outputfile.createVariable('proxy_values',       'f4',('age','proxy',))
output_proxy_res           = outputfile.createVariable('proxy_resolutions',  'f4',('age','proxy',))
output_proxy_uncer         = outputfile.createVariable('proxy_uncertainty',  'f4',('proxy',))
output_metadata            = outputfile.createVariable('proxy_metadata',     'str',('proxy','metadata',))
output_options             = outputfile.createVariable('options',            'str',('options',))
output_proxies_selected    = outputfile.createVariable('proxies_selected',   'i1',('proxy',))
output_proxies_assimilated = outputfile.createVariable('proxies_assimilated','i1',('age','proxy',))

output_proxyprior_mean[:]     = prior_proxy_means
output_proxyrecon_mean[:]     = recon_mean_proxies
output_proxyrecon_ens[:]      = recon_ens_proxies
output_ages[:]                = proxy_data['age_centers'] 
output_lat[:]                 = model_data['lat']
output_lon[:]                 = model_data['lon']
output_proxy_vals[:]          = np.transpose(proxy_data['values_binned'])
output_proxy_res[:]           = np.transpose(proxy_data['resolution_binned'])
output_proxy_uncer[:]         = proxy_data['uncertainty']
output_metadata[:]            = proxy_data['metadata']
output_options[:]             = np.array(options_list)
output_proxies_selected[:]    = proxy_ind_random_selected.astype(int)
output_proxies_assimilated[:] = proxies_to_assimilate_all.astype(int)

outputfile.title = 'Holocene climate reconstruction'
outputfile.close()

endtime_total = time.time()  # End timer
print('Total time: '+str('%1.2f' % ((endtime_total-starttime_total)/60))+' minutes')
print(' === Reconstruction complete ===')

