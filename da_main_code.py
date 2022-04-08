#==============================================================================
# This script contains the main code of the Holocene data assimilation.
# Options are set in the config yml file. See README.txt for a more complete
# explanation of the code and setup.
#    author: Michael P. Erb
#    date  : 3/29/2022
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


#%% SETTINGS

starttime_total = time.time() # Start timer

# Use a given config file.  If not given, use config_default.yml.
if len(sys.argv) > 1: config_file = sys.argv[1]
else:                 config_file = 'config_default.yml'

# Load the configuration options and print them to the screen.
print('Using configuration file: '+config_file)
with open(config_file,'r') as file: options = yaml.load(file,Loader=yaml.FullLoader)

print('=== SETTINGS ===')
for key in options.keys():
    print('%30s: %-15s' % (key,str(options[key])))
print('=== END SETTINGS ===')


#%% LOAD AND PROCESS DATA

# Load the chosen proxy data
proxy_ts,collection_all = da_load_proxies.load_proxies(options)
proxy_data = da_load_proxies.process_proxies(proxy_ts,collection_all,options)

# Load the chosen model data
model_data = da_load_models.load_model_data(options)

# Detrend the model data if selected
model_data = da_load_models.detrend_model_data(model_data,options)

# Get some dimensions
n_models_in_prior = len(options['models_for_prior'])
n_proxies         = proxy_data['values_binned'].shape[0]

# If the prior is allowed to change through time, remove the mean of the reference period from each model.
if options['reconstruction_type'] == 'relative':
    for i in range(n_models_in_prior):
        ind_for_model = (model_data['number'] == (i+1))
        ind_ref = (model_data['age'] >= options['reference_period'][0]) & (model_data['age'] < options['reference_period'][1]) & ind_for_model
        model_data['tas'][ind_for_model,:,:,:]      = model_data['tas'][ind_for_model,:,:,:]      - np.mean(model_data['tas'][ind_ref,:,:,:],axis=0)
        model_data['tas_annual'][ind_for_model,:,:] = model_data['tas_annual'][ind_for_model,:,:] - np.mean(model_data['tas_annual'][ind_ref,:,:],axis=0)

# If requested, alter the proxy uncertainty values.
if options['change_uncertainty']:
    if options['change_uncertainty'][0:5] == 'mult_':
        uncertainty_multiplier = float(options['change_uncertainty'][5:])
        proxy_data['uncertainty'] = proxy_data['uncertainty']*uncertainty_multiplier
        print(' --- Processing: All uncertainty values multiplied by '+str(uncertainty_multiplier)+' ---')
    elif options['change_uncertainty'][0:4] == 'all_':
        prescribed_uncertainty = float(options['change_uncertainty'][4:])
        proxy_data['uncertainty'][:] = prescribed_uncertainty
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
                proxy_data['uncertainty'][i] = proxy_uncertainties_from_file[index_uncertainty,1].astype(float)

# Use PSMs to get model-based proxy estimates
proxy_estimates_all,_ = da_psms.psm_main(model_data,proxy_data,options)


#%% SET THINGS UP

# Get more dimensions
n_vars       = len(options['vars_to_reconstruct'])
n_ages       = proxy_data['values_binned'].shape[1]
n_lat        = len(model_data['lat'])
n_lon        = len(model_data['lon'])
n_latlonvars = n_lat*n_lon*n_vars
n_state      = (n_latlonvars) + n_proxies

# Determine the total possible number of ensemble members
n_ens_possible = len(da_load_models.get_indices_for_prior(options,model_data,0))

# If using less than 100 percent for the ensemble members, randomly choose them here.
np.random.seed(seed=options['seed_for_prior'])
n_ens = int(round(n_ens_possible*(options['percent_of_prior']/100)))
ind_to_use = np.random.choice(n_ens_possible,n_ens,replace=False)
ind_to_use = np.sort(ind_to_use)
print(' --- Processing: Choosing '+str(options['percent_of_prior'])+'% of possible prior states, n_ens='+str(n_ens)+' ---')

# Randomly select the ensemble members to save (max=100) to reduce output filesizes
np.random.seed(seed=0)
n_ens_to_save = min([n_ens,100])
ind_to_save = np.random.choice(n_ens,n_ens_to_save,replace=False)
ind_to_save = np.sort(ind_to_save)

# Set up arrays for reconstruction values and more outputs
recon_ens         = np.zeros((n_state,n_ens_to_save,n_ages)); recon_ens[:]         = np.nan
recon_mean        = np.zeros((n_state,n_ages));               recon_mean[:]        = np.nan
recon_global_all  = np.zeros((n_ages,n_ens,n_vars));          recon_global_all[:]  = np.nan
recon_nh_all      = np.zeros((n_ages,n_ens,n_vars));          recon_nh_all[:]      = np.nan
recon_sh_all      = np.zeros((n_ages,n_ens,n_vars));          recon_sh_all[:]      = np.nan
prior_ens         = np.zeros((n_state,n_ens_to_save,n_ages)); prior_ens[:]         = np.nan
prior_mean        = np.zeros((n_state,n_ages));               prior_mean[:]        = np.nan
prior_global_all  = np.zeros((n_ages,n_ens,n_vars));          prior_global_all[:]  = np.nan
prior_proxy_means = np.zeros((n_ages,n_proxies));             prior_proxy_means[:] = np.nan
proxies_to_assimilate_all = np.zeros((n_ages,n_proxies));     proxies_to_assimilate_all[:] = np.nan


#%% FIND PROXIES TO ASSIMILATE AND MORE

print(' === FINDING PROXIES TO ASSIMILATE BASED ON CHOSEN OPTIONS ===')

# Find proxies with data in selected age range (and reference period, if doing a relative reconstruction)
proxy_ind_with_valid_values = np.isfinite(np.nanmean(proxy_data['values_binned'],axis=1))
print(' - Number of records with valid values for the chosen experiment: '+str(sum(proxy_ind_with_valid_values)))

# Find the proxies with uncertainty values
proxy_ind_with_uncertainty = np.isfinite(proxy_data['uncertainty'])
print(' - Number of records with uncertainty values: '+str(sum(proxy_ind_with_uncertainty)))

# If requested, select only proxies with certain seasonalities
if options['assimilate_selected_seasons']:
    proxy_ind_of_seasonality = np.full((n_proxies),False,dtype=bool)
    ind_seasons = [i for i, seasontype in enumerate(proxy_data['metadata'][:,5]) if seasontype in options['assimilate_selected_seasons']]
    proxy_ind_of_seasonality[ind_seasons] = True
    print(' - Number of records with seasonalities '+str(options['assimilate_selected_seasons'])+': '+str(sum(proxy_ind_of_seasonality)))
else:
    proxy_ind_of_seasonality = np.full((n_proxies),True,dtype=bool)

# If requested, select only certain archive types
if options['assimilate_selected_archives']:
    proxy_ind_of_archive_type = np.full((n_proxies),False,dtype=bool)
    ind_archives = [i for i, atype in enumerate(proxy_data['archivetype']) if atype in options['assimilate_selected_archives']]
    proxy_ind_of_archive_type[ind_archives] = True
    print(' - Number of records with archive types '+str(options['assimilate_selected_archives'])+': '+str(sum(proxy_ind_of_archive_type)))
else:
    proxy_ind_of_archive_type = np.full((n_proxies),True,dtype=bool)

# If requested, select the proxies within the specified region
if options['assimilate_selected_region']:
    region_lat_min,region_lat_max,region_lon_min,region_lon_max = options['assimilate_selected_region']
    proxy_ind_in_region = (proxy_data['lats'] >= region_lat_min) & (proxy_data['lats'] <= region_lat_max) & (proxy_data['lons'] >= region_lon_min) & (proxy_data['lons'] <= region_lon_max)
    print(' - Number of records in region '+str(options['assimilate_selected_region'])+': '+str(sum(proxy_ind_in_region)))
else:
    proxy_ind_in_region = np.full((n_proxies),True,dtype=bool)

# If requested, select the proxies with median resolution within a certain window
if options['assimilate_selected_resolution']: 
    proxy_med_res = proxy_data['metadata'][:,6].astype(float)
    proxy_ind_in_resolution_band = (proxy_med_res >= options['assimilate_selected_resolution'][0]) & (proxy_med_res < options['assimilate_selected_resolution'][1])
    print(' - Number of records with median resolution in the range '+str(options['assimilate_selected_resolution'])+': '+str(sum(proxy_ind_in_resolution_band)))
else:
    proxy_ind_in_resolution_band = np.full((n_proxies),True,dtype=bool)

# Count the selected records so far
proxy_ind_chosen_criteria = proxy_ind_with_valid_values &\
                            proxy_ind_with_uncertainty &\
                            proxy_ind_of_seasonality &\
                            proxy_ind_of_archive_type &\
                            proxy_ind_in_region &\
                            proxy_ind_in_resolution_band
ind_values_chosen_criteria = np.where(proxy_ind_chosen_criteria)[0]
n_proxies_meeting_criteria = sum(proxy_ind_chosen_criteria)
print(' --- Number of records meeting ALL of the above criteria: '+str(n_proxies_meeting_criteria))

# If requested, select the portion of the proxies which are to be assimilated
if options['percent_to_assimilate'] < 100:
    print(' - Processing: Choosing only '+str(options['percent_to_assimilate'])+'% of possible proxies')
    proxy_ind_selected = np.full((n_proxies),False,dtype=bool)
    np.random.seed(seed=options['seed_for_proxy_choice'])
    n_proxies_to_choose = int(round(n_proxies_meeting_criteria*(options['percent_to_assimilate']/100)))
    proxy_ind_random = np.random.choice(n_proxies_meeting_criteria,n_proxies_to_choose,replace=False)
    proxy_ind_random = np.sort(proxy_ind_random)
    proxy_ind_selected[ind_values_chosen_criteria[proxy_ind_random]] = True
else:
    proxy_ind_selected = proxy_ind_chosen_criteria

print(' --- Final number of selected records: '+str(sum(proxy_ind_selected)))

# Calculate the localization matrix (it may not be used)
if options['assimate_together'] == False:
    proxy_localization_all = da_utils.loc_matrix(options,model_data,proxy_data)


#%% DO DATA ASSIMILATION

# Loop through every age, doing the data assimilation with a time-varying prior
print(' === Starting data assimilation === ')
#age_counter = 0; age = proxy_data['age_centers'][age_counter]
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
    if len(indices_for_prior) != n_ens_possible: print(' !!! Warning: number of prior ages selected does not match n_ens.  Age='+str(age))
    #
    # Get the prior values for the variables to reconstruct
    for j,var_name in enumerate(options['vars_to_reconstruct']):
        var_annual_for_prior = model_data[var_name+'_annual'][indices_for_prior,:,:][:,:,:,None]
        if j == 0: vars_annual_for_prior_all = var_annual_for_prior
        else:      vars_annual_for_prior_all = np.concatenate((vars_annual_for_prior_all,var_annual_for_prior),axis=3)
    #
    # For each proxy, get the proxy estimates for the correct resolution
    model_estimates_for_age = np.zeros((n_ens_possible,n_proxies)); model_estimates_for_age[:] = np.nan
    for j in range(n_proxies):
        res = proxy_resolution_for_age[j]
        if np.isnan(proxy_values_for_age[j]): continue
        model_estimates_for_age[:,j] = proxy_estimates_all[j][int(res)][indices_for_prior]
    #
    # Use only the randomly selected climate states in the prior
    vars_annual_for_prior_all = vars_annual_for_prior_all[ind_to_use,:,:,:]
    model_estimates_for_age   = model_estimates_for_age[ind_to_use,:]
    model_number_for_prior    = model_number_for_prior[ind_to_use]
    #
    # For a relative reconstruction, remove the means of each model seperately
    if ((options['reconstruction_type'] == 'relative') and (options['prior_mean_always_0'] == True)):
        for i in range(n_models_in_prior):
            ind_for_model = np.where(model_number_for_prior == (i+1))[0]
            vars_annual_for_prior_all[ind_for_model,:,:,:] = vars_annual_for_prior_all[ind_for_model,:,:,:] - np.mean(vars_annual_for_prior_all[ind_for_model,:,:,:],axis=0)
            model_estimates_for_age[ind_for_model,:]       = model_estimates_for_age[ind_for_model,:]       - np.mean(model_estimates_for_age[ind_for_model,:],axis=0)
    #
    # Make the prior (Xb)
    prior = np.reshape(vars_annual_for_prior_all,(n_ens,n_latlonvars))
    #
    # Append the proxy estimate to the prior, so that proxy estimates are reconstructed too
    prior = np.append(prior,model_estimates_for_age,axis=1)
    Xb = np.transpose(prior)
    #
    # Get the mean and selected ensemble values
    prior_mean[:,age_counter]  = np.mean(Xb,axis=1)
    prior_ens[:,:,age_counter] = Xb[:,ind_to_save]
    #
    # Save the prior estimates of proxies, for analysis later
    prior_proxy_means[age_counter,:] = np.mean(model_estimates_for_age,axis=0)
    #
    # Select only the proxies which meet the criteria
    proxies_to_assimilate = proxy_ind_selected & np.isfinite(proxy_values_for_age)
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
    recon_ens[:,:,age_counter] = Xa[:,ind_to_save]
    #
    # Note progression of the reconstruction
    print('Time step '+str(age_counter)+'/'+str(len(proxy_data['age_centers'] ))+' complete.  Time: '+str('%1.2f' % (time.time()-starttime_loop))+' sec')

# Reshape the data arrays
recon_mean = np.transpose(recon_mean)
recon_ens  = np.swapaxes(recon_ens,0,2)
prior_mean = np.transpose(prior_mean)
prior_ens  = np.swapaxes(prior_ens,0,2)

# Reshape the gridded reconstruction to a lat-lon grid
recon_mean_grid = np.reshape(recon_mean[:,:n_latlonvars], (n_ages,              n_lat,n_lon,n_vars))
recon_ens_grid  = np.reshape(recon_ens[:,:,:n_latlonvars],(n_ages,n_ens_to_save,n_lat,n_lon,n_vars))
prior_mean_grid = np.reshape(prior_mean[:,:n_latlonvars], (n_ages,              n_lat,n_lon,n_vars))
prior_ens_grid  = np.reshape(prior_ens[:,:,:n_latlonvars],(n_ages,n_ens_to_save,n_lat,n_lon,n_vars))

# Put the proxy reconstructions into separate variables
recon_mean_proxies = recon_mean[:,n_latlonvars:]
recon_ens_proxies  = recon_ens[:,:,n_latlonvars:]

# Store the options into as list to save
n_options = len(options.keys())
options_list = []
for key,value in options.items():
    options_list.append(key+':'+str(value))


#%% SAVE THE OUTPUT

time_str = str(datetime.datetime.now()).replace(' ','_')
output_filename = 'holocene_recon_'+time_str+'_'+str(options['exp_name'])
print('Saving the reconstruction as '+output_filename)

# Save all data into a netCDF file
output_dir = options['data_dir']+'results/'
outputfile = netCDF4.Dataset(output_dir+output_filename+'.nc','w')
outputfile.createDimension('ages',        n_ages)
outputfile.createDimension('ens',         n_ens)
outputfile.createDimension('ens_selected',n_ens_to_save)
outputfile.createDimension('lat',         n_lat)
outputfile.createDimension('lon',         n_lon)
outputfile.createDimension('proxy',       n_proxies)
outputfile.createDimension('metadata',    proxy_data['metadata'].shape[1])
outputfile.createDimension('exp_options', n_options)

output_recon_mean,output_recon_ens,output_recon_global,output_recon_nh,output_recon_sh,output_prior_mean,output_prior_ens,output_prior_global = {},{},{},{},{},{},{},{}
for i,var_name in enumerate(options['vars_to_reconstruct']):
    output_recon_mean[var_name]   = outputfile.createVariable('recon_'+var_name+'_mean',       'f4',('ages','lat','lon',))
    output_recon_ens[var_name]    = outputfile.createVariable('recon_'+var_name+'_ens',        'f4',('ages','ens_selected','lat','lon',))
    output_recon_global[var_name] = outputfile.createVariable('recon_'+var_name+'_global_mean','f4',('ages','ens',))
    output_recon_nh[var_name]     = outputfile.createVariable('recon_'+var_name+'_nh_mean',    'f4',('ages','ens',))
    output_recon_sh[var_name]     = outputfile.createVariable('recon_'+var_name+'_sh_mean',    'f4',('ages','ens',))
    output_prior_mean[var_name]   = outputfile.createVariable('prior_'+var_name+'_mean',       'f4',('ages','lat','lon',))
    #output_prior_ens[var_name]    = outputfile.createVariable('prior_'+var_name+'_ens',        'f4',('ages','ens_selected','lat','lon',))
    output_prior_global[var_name] = outputfile.createVariable('prior_'+var_name+'_global_mean','f4',('ages','ens',))
    output_recon_mean[var_name][:]   = recon_mean_grid[:,:,:,i]
    output_recon_ens[var_name][:]    = recon_ens_grid[:,:,:,:,i]
    output_recon_global[var_name][:] = recon_global_all[:,:,i]
    output_recon_nh[var_name][:]     = recon_nh_all[:,:,i]
    output_recon_sh[var_name][:]     = recon_sh_all[:,:,i]
    output_prior_mean[var_name][:]   = prior_mean_grid[:,:,:,i]
    #output_prior_ens[var_name][:]    = prior_ens_grid[:,:,:,:,i]
    output_prior_global[var_name][:] = prior_global_all[:,:,i]

output_proxyprior_mean     = outputfile.createVariable('proxyprior_mean',    'f4',('ages','proxy',))
output_proxyrecon_mean     = outputfile.createVariable('proxyrecon_mean',    'f4',('ages','proxy',))
output_proxyrecon_ens      = outputfile.createVariable('proxyrecon_ens',     'f4',('ages','ens_selected','proxy',))
output_ages                = outputfile.createVariable('ages',               'f4',('ages',))
output_lat                 = outputfile.createVariable('lat',                'f4',('lat',))
output_lon                 = outputfile.createVariable('lon',                'f4',('lon',))
output_proxy_vals          = outputfile.createVariable('proxy_values',       'f4',('ages','proxy',))
output_proxy_res           = outputfile.createVariable('proxy_resolutions',  'f4',('ages','proxy',))
output_proxy_uncer         = outputfile.createVariable('proxy_uncertainty',  'f4',('proxy',))
output_metadata            = outputfile.createVariable('proxy_metadata',     'str',('proxy','metadata',))
output_options             = outputfile.createVariable('options',            'str',('exp_options',))
output_proxies_selected    = outputfile.createVariable('proxies_selected',   'i1',('proxy',))
output_proxies_assimilated = outputfile.createVariable('proxies_assimilated','i1',('ages','proxy',))

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
output_proxies_selected[:]    = proxy_ind_selected.astype(int)
output_proxies_assimilated[:] = proxies_to_assimilate_all.astype(int)

outputfile.title = 'Holocene climate reconstruction'
outputfile.close()

endtime_total = time.time()  # End timer
print('Total time: '+str('%1.2f' % ((endtime_total-starttime_total)/60))+' minutes')
print(' === Reconstruction complete ===')

