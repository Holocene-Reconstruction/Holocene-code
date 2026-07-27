#==============================================================================
# This script contains the main code of the Holocene data assimilation.
# Options are set in the config yml file. See README.txt for a more complete
# explanation of the code and setup.
# ---
# This version of the code has been updated to focus on North America for the
# North American ecological sensitivity project.
# ---
# author: Michael Erb
#==============================================================================

# Change working directory
import os
os.chdir('C:/Users/erbm/Documents/GitHub/Holocene-code/')

# Import libraries
import sys
import numpy as np
import xarray as xr
import yaml
import time
import datetime
import da_utils
import da_utils_lmr
import da_load_models
import da_load_proxies
import da_psms
import da_plot_results

# Make maps of proxy-by-proxy updates
age_ind_to_map = False
#age_ind_to_map = [0,100,200]


#%% SETTINGS

starttime_total = time.time() # Start timer

# Set the config file and seed to use.
config_file    = 'config.yml'
seed_overwrite = 'None'
if   len(sys.argv) > 1: config_file    = sys.argv[1]
elif len(sys.argv) > 2: seed_overwrite = sys.argv[2]

# Load the configuration options
print('Using configuration file: '+config_file)
with open(config_file,'r') as file: options = yaml.load(file,Loader=yaml.FullLoader)

# If the seed has been overwritten using sys.argv, update it here
if seed_overwrite != 'None':
    options['seed_for_prior']        = int(seed_overwrite)
    options['seed_for_proxy_choice'] = int(seed_overwrite)

# Get the root vars (e.g., "tas")
vars_to_reconstruct = options['vars_to_reconstruct']
vars_to_reconstruct_root = [var.split('_')[0] for var in vars_to_reconstruct]
options['vars_root'] = np.unique(vars_to_reconstruct_root)

# Set some text for the output filename
if seed_overwrite == 'None': extra_txt = ''
else:                        extra_txt = '_iter_'+str(int(seed_overwrite)).zfill(2)
exp_name_full = str(options['exp_name'])+extra_txt

print('=== SETTINGS ===')
for key in options.keys():
    print('%30s: %-15s' % (key,str(options[key])))
print('=== END SETTINGS ===')


#%% LOAD AND PROCESS PROXY DATA

# Load the chosen proxy data
proxy_ts,collection_all = da_load_proxies.load_proxies(options)

# Print summaries of proxy data
# (Use this to look for useful proxies and iterate the experimental design)
#da_utils.print_proxy_metadata(proxy_ts,'interpretation1_variable')
#da_utils.print_proxy_metadata(proxy_ts,'paleoData_variableName')
#da_utils.print_proxy_metadata(proxy_ts,'archiveType')
#da_utils.print_proxy_metadata(proxy_ts,'paleoData_proxy')
#da_utils.print_proxy_metadata(proxy_ts,'paleoData_units')
#da_utils.print_proxy_metadata(proxy_ts,'interpretation1_direction')

# Filter the proxies and set PSMs
proxy_ts_selected,psms_selected,collection_selected = da_load_proxies.filter_proxies_and_set_psms(proxy_ts,collection_all,options)

# Process the proxy data
proxy_data = da_load_proxies.process_proxies(proxy_ts_selected,psms_selected,collection_selected,options)

# Average nearby proxies, if selected
if options['average_nearby_proxies']:
    proxy_data = da_load_proxies.average_nearby_records(proxy_data,options)

n_proxies = proxy_data['values_binned'].shape[0]


#%% LOAD AND PROCESS MODEL DATA

# Load the chosen model data
model_data = da_load_models.load_model_data(options)
n_models_in_prior = len(options['models_for_prior'])

# If the prior is allowed to change through time, remove the mean of the reference period from each model.
if options['reconstruction_type'] == 'relative':
    print('Processing model: relative reconstruction')
    for i in range(n_models_in_prior):
        ind_for_model = (model_data['number'] == (i+1))
        ind_ref = (model_data['age'] >= options['reference_period'][0]) & (model_data['age'] < options['reference_period'][1]) & ind_for_model
        for var in options['vars_root']:
            model_data[var][ind_for_model,:,:,:] = model_data[var][ind_for_model,:,:,:] - np.mean(model_data[var][ind_ref,:,:,:],axis=0)
        for var in options['vars_to_reconstruct']:
            model_data[var][ind_for_model,:,:]   = model_data[var][ind_for_model,:,:]   - np.mean(model_data[var][ind_ref,:,:],axis=0)

# Do more prior processing, if requested
if options['model_processing'] != 'None':
    print('Processing model: '+str(options['model_processing']))
    print("WARNING MODEL PROCESSING STILL IN DEVELOPMENT")
    model_data = da_load_models.detrend_model_data(model_data,options)

#TODO: In the code above, is there a way to apply the filtering above only to covariances but not means?

# Use PSMs to get model-based proxy estimates
proxy_estimates_all,_,proxy_data = da_psms.psm_main(model_data,proxy_data,options)


#%% SET UNCERTAINTIES

# If requested, alter the proxy uncertainty values.
if options['change_uncertainty'] != False:
    
    if options['change_uncertainty'][0:5] == 'mult_':
        uncertainty_multiplier = float(options['change_uncertainty'][5:])
        proxy_data['uncertainty'] = proxy_data['uncertainty']*uncertainty_multiplier
        print(' --- Processing proxies: uncertainty values MULTIPLIED BY '+str(uncertainty_multiplier)+' ---')
    
    elif options['change_uncertainty'][0:4] == 'all_':
        prescribed_uncertainty = float(options['change_uncertainty'][4:])
        proxy_data['uncertainty'][:] = prescribed_uncertainty
        print(' --- Processing proxies: uncertainty values SET TO '+str(prescribed_uncertainty)+' ---')
    
    elif options['change_uncertainty'][0:6] == 'stdev_':
        uncertainty_multiplier = float(options['change_uncertainty'][6:])
        print(' --- Processing proxies: uncertainty values SET TO the standard deviation of each record MULTIPLIED BY '+str(uncertainty_multiplier)+' ---')
        for i in range(n_proxies):
            proxy_data['uncertainty'][i] = np.nanstd(proxy_data['values_binned'][i,:])*uncertainty_multiplier
    
    else:
        # If using this option, the text file below should contain TSids and MSE for every proxy record
        print(' --- Processing proxies: uncertainty values SET TO values from the following file ---')
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


#%% FIND PROXIES TO ASSIMILATE AND MORE

print(' === FINDING PROXIES TO ASSIMILATE BASED ON CHOSEN OPTIONS ===')

# Find proxies with data in selected age range(s)
proxy_ind_with_valid_values = np.isfinite(np.nanmean(proxy_data['values_binned'],axis=1))
print(' - Number of records with valid values for the chosen experiment: '+str(sum(proxy_ind_with_valid_values)))

# Find the proxies with uncertainty values
proxy_ind_with_uncertainty = np.isfinite(proxy_data['uncertainty'])
if len(proxy_ind_with_uncertainty.shape) == 2: proxy_ind_with_uncertainty = proxy_ind_with_uncertainty.any(axis=1)
print(' - Number of records with uncertainty values: '+str(sum(proxy_ind_with_uncertainty)))

# If requested, select only proxies with certain seasonalities
if options['assimilate_selected_seasons'] in ['jja_preferred','djf_preferred']:
    proxy_ind_of_seasonality = da_utils.find_seasons(proxy_data,options)
elif options['assimilate_selected_seasons'] != False:
    proxy_ind_of_seasonality = np.full((n_proxies),False,dtype=bool)
    ind_seasons = [i for i, seasontype in enumerate(proxy_data['metadata'][:,5]) if seasontype.lower() in options['assimilate_selected_seasons']]
    proxy_ind_of_seasonality[ind_seasons] = True
    print(' - Number of records with seasonalities '+str(options['assimilate_selected_seasons'])+': '+str(sum(proxy_ind_of_seasonality)))
else:
    proxy_ind_of_seasonality = np.full((n_proxies),True,dtype=bool)

# If requested, select only certain archive types
if options['assimilate_selected_archives'] != False:
    proxy_ind_of_archive_type = np.full((n_proxies),False,dtype=bool)
    ind_archives = [i for i, atype in enumerate(proxy_data['archivetype']) if atype in options['assimilate_selected_archives']]
    proxy_ind_of_archive_type[ind_archives] = True
    print(' - Number of records with archive types '+str(options['assimilate_selected_archives'])+': '+str(sum(proxy_ind_of_archive_type)))
else:
    proxy_ind_of_archive_type = np.full((n_proxies),True,dtype=bool)

# If requested, select the proxies within the specified region
if options['assimilate_selected_region'] != False:
    region_lat_min,region_lat_max,region_lon_min,region_lon_max = options['assimilate_selected_region']
    proxy_ind_in_region = (proxy_data['lats'] >= region_lat_min) & (proxy_data['lats'] <= region_lat_max) & (proxy_data['lons'] >= region_lon_min) & (proxy_data['lons'] <= region_lon_max)
    print(' - Number of records in region '+str(options['assimilate_selected_region'])+': '+str(sum(proxy_ind_in_region)))
else:
    proxy_ind_in_region = np.full((n_proxies),True,dtype=bool)

# If requested, select the proxies with median resolution within a certain window
if options['assimilate_selected_resolution'] != False: 
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


#%% SET THINGS UP

# Get more dimensions
n_vars       = len(options['vars_to_reconstruct'])
n_ages       = proxy_data['values_binned'].shape[1]
n_lat        = len(model_data['lat'])
n_lon        = len(model_data['lon'])
n_varslatlon = n_vars*n_lat*n_lon
n_state      = (n_varslatlon) + n_proxies

# Determine the total possible number of ensemble members
n_ens_possible = len(da_load_models.get_indices_for_prior(options,model_data,0))

# If using less than 100 percent for the ensemble members, randomly choose them here.
np.random.seed(seed=options['seed_for_prior'])
n_ens = int(round(n_ens_possible*(options['percent_of_prior']/100)))
ind_random_ens_to_use = np.random.choice(n_ens_possible,n_ens,replace=False)
ind_random_ens_to_use = np.sort(ind_random_ens_to_use)
print(' --- Processing: Choosing '+str(options['percent_of_prior'])+'% of possible prior states, n_ens='+str(n_ens)+' ---')

# Randomly select the ensemble members to save (max=100) to reduce output filesizes
np.random.seed(seed=0)
n_ens_to_save = min([n_ens,100])
ind_random_ens_to_save = np.random.choice(n_ens,n_ens_to_save,replace=False)
ind_random_ens_to_save = np.sort(ind_random_ens_to_save)

# Set up arrays for reconstruction values and more outputs
recon_ens         = np.zeros((n_state,n_ens_to_save,n_ages));   recon_ens[:]         = np.nan
recon_mean        = np.zeros((n_state,n_ages));                 recon_mean[:]        = np.nan
recon_global_all  = np.zeros((n_ages,n_ens,n_vars));            recon_global_all[:]  = np.nan
prior_ens         = np.zeros((n_state,n_ens_to_save,n_ages));   prior_ens[:]         = np.nan
prior_mean        = np.zeros((n_state,n_ages));                 prior_mean[:]        = np.nan
prior_global_all  = np.zeros((n_ages,n_ens,n_vars));            prior_global_all[:]  = np.nan
prior_proxy_means = np.zeros((n_ages,n_proxies));               prior_proxy_means[:] = np.nan
prior_proxy_ens   = np.zeros((n_ages,n_ens_to_save,n_proxies)); prior_proxy_ens[:]   = np.nan
proxies_to_assimilate_all = np.zeros((n_ages,n_proxies));       proxies_to_assimilate_all[:] = np.nan

# Calculate the localization matrix (it may not be used)
if options['assimate_together'] == False:
    proxy_localization_all = da_utils.loc_matrix(options,model_data,proxy_data)


#%% DO DATA ASSIMILATION

# Loop through every age, doing the data assimilation with a time-varying prior
print(' === Starting data assimilation === ')
age_counter = 0; age = proxy_data['age_centers'][age_counter]
for age_counter,age in enumerate(proxy_data['age_centers']):
    #
    starttime_loop = time.time()
    #
    # Get all proxy values, resolutions, and uncertainties for the current age
    proxy_values_for_age        = proxy_data['values_binned'][:,age_counter]
    proxy_resolution_for_age    = proxy_data['resolution_binned'][:,age_counter]
    proxy_uncertainties_for_age = proxy_data['uncertainty']
    #
    # Get the indices of the prior which will be used for this data assimilation step
    indices_for_prior = da_load_models.get_indices_for_prior(options,model_data,age)
    model_number_for_prior = model_data['number'][indices_for_prior]
    if len(indices_for_prior) != n_ens_possible: print(' !!! Warning: number of prior ages selected does not match n_ens.  Age='+str(age))
    #
    # Get the prior values for the variables to reconstruct
    for j,var_name in enumerate(options['vars_to_reconstruct']):
        model_values_for_prior = model_data[var_name][indices_for_prior,:,:][:,None,:,:]
        if j == 0: model_values_for_prior_all = model_values_for_prior
        else:      model_values_for_prior_all = np.concatenate((model_values_for_prior_all,model_values_for_prior),axis=1)
    #
    # For each proxy, get the proxy estimates for the correct resolution
    model_estimates_for_age = np.zeros((n_ens_possible,n_proxies)); model_estimates_for_age[:] = np.nan
    for j in range(n_proxies):
        res = proxy_resolution_for_age[j]
        if np.isnan(proxy_values_for_age[j]) or np.isnan(res): continue
        model_estimates_for_age[:,j] = proxy_estimates_all[j][int(res)][indices_for_prior]
    #
    # Use only the randomly selected climate states in the prior
    model_values_for_prior_all = model_values_for_prior_all[ind_random_ens_to_use,:,:,:]
    model_estimates_for_age    = model_estimates_for_age[ind_random_ens_to_use,:]
    model_number_for_prior     = model_number_for_prior[ind_random_ens_to_use]
    #
    # Make the prior (Xb)
    prior = np.reshape(model_values_for_prior_all,(n_ens,n_varslatlon))
    #
    # Append the proxy estimate to the prior, so that proxy estimates are reconstructed too
    prior = np.append(prior,model_estimates_for_age,axis=1)
    Xb = np.transpose(prior)
    Xb_original = Xb
    #
    # Save some values for later
    prior_mean[:,age_counter]        = np.mean(Xb,axis=1)                                 # Prior mean
    prior_ens[:,:,age_counter]       = Xb[:,ind_random_ens_to_save]                       # Prior ensembles
    prior_proxy_means[age_counter,:] = np.mean(model_estimates_for_age,axis=0)            # Prior proxy mean
    prior_proxy_ens[age_counter,:,:] = model_estimates_for_age[ind_random_ens_to_save,:]  # Prior proxy ensembles
    #
    # Select only the proxies which meet the criteria:
    #  - Proxies were selected for analysis
    #  - Proxies have values and uncertainties for the age of interest
    proxies_to_assimilate = proxy_ind_selected &\
                            np.isfinite(proxy_values_for_age) &\
                            np.isfinite(proxy_uncertainties_for_age)
                            #&\
                            #np.isfinite(proxy_resolution_for_age) &\
                            #np.isfinite(prior_proxy_means[age_counter,:])
    #
    # Keep a record of which proxies are assimilated
    proxies_to_assimilate_all[age_counter,:] = proxies_to_assimilate
    proxy_ind_to_assimilate = np.where(proxies_to_assimilate)[0]
    #
    # If valid proxies are present for this time step, do the data assimilation
    n_proxies_at_age = proxy_ind_to_assimilate.shape[0]
    if n_proxies_at_age > 0:
        #
        # Do the DA update, either together or one at a time.
        if options['assimate_together']:
            #
            proxy_values_selected       = proxy_values_for_age[proxy_ind_to_assimilate]
            proxy_uncertainty_selected  = proxy_uncertainties_for_age[proxy_ind_to_assimilate]
            model_estimates_selected    = model_estimates_for_age[:,proxy_ind_to_assimilate]
            R_diagonal = np.diag(proxy_uncertainty_selected)
            #
            Xa,_,_ = da_utils.damup(Xb,np.transpose(model_estimates_selected),R_diagonal,proxy_values_selected)
        else:
            ind_proxy = proxy_ind_to_assimilate[0]
            for ind_proxy in proxy_ind_to_assimilate:
                #print(ind_proxy)
                #
                # Get values for proxy
                proxy_value       = proxy_values_for_age[ind_proxy]
                proxy_uncertainty = proxy_uncertainties_for_age[ind_proxy]
                proxy_lat         = proxy_data['lats'][ind_proxy]
                proxy_lon         = proxy_data['lons'][ind_proxy]
                model_estimates   = model_estimates_for_age[:,ind_proxy]
                if options['localization_radius'] != 'None': loc = proxy_localization_all[ind_proxy,:]
                else: loc = None
                #
                # Do data assimilation
                Xb_updated = da_utils_lmr.enkf_update_array(Xb,proxy_value,model_estimates,proxy_uncertainty,options,loc=loc,inflate=None)
                if np.isnan(Xb).all(): print(' !!! ERROR.  ALL RECONSTRUCTION VALUES SET TO NAN.  Age='+str(age)+', proxy number='+str(ind_proxy)+' !!!')
                #
                # If desired, make a map of the current state of the reconstruction for this age
                if age_counter in age_ind_to_map:
                    # Make an updated map
                    var_toplot_start   = np.mean(np.reshape(Xb[:n_varslatlon,:],        (n_vars,n_lat,n_lon,n_ens))[0,:,:,:],axis=2)
                    var_toplot_updated = np.mean(np.reshape(Xb_updated[:n_varslatlon,:],(n_vars,n_lat,n_lon,n_ens))[0,:,:,:],axis=2)
                    var_toplot_change  = var_toplot_updated-var_toplot_start
                    da_plot_results.make_map(var_toplot_updated,model_data,proxy_value,proxy_lat,proxy_lon,proxy_uncertainty,ind_proxy,age,'after', exp_name_full,bounds=5,  save_instead_of_plot=True)
                    #da_plot_results.make_map(var_toplot_change, model_data,proxy_value,proxy_lat,proxy_lon,proxy_uncertainty,ind_proxy,age,'C_diff',exp_name_full,bounds=.25,save_instead_of_plot=True)
                #
                Xb = Xb_updated
            #
            # Set the final values
            Xa = Xb
        #
    else:
        # No proxies are assimilated
        Xa = Xb
    #
    # Compute the spatial-mean of the prior
    prior_global = da_utils.global_mean(model_values_for_prior_all,model_data['lat'],2,3)
    prior_global_all[age_counter,:,:] = prior_global
    #
    # Compute the mean of the reconstruction
    Xa_latlon = np.reshape(Xa[:n_varslatlon,:],(n_vars,n_lat,n_lon,n_ens))
    recon_global = da_utils.global_mean(Xa_latlon,model_data['lat'],1,2)
    recon_global_all[age_counter,:,:] = np.transpose(recon_global)
    #
    # Get the mean of all ensemble members and the values of some
    recon_mean[:,age_counter]  = np.mean(Xa,axis=1)
    recon_ens[:,:,age_counter] = Xa[:,ind_random_ens_to_save]
    #
    # Note progression of the reconstruction
    print('Time step '+str(age_counter)+'/'+str(len(proxy_data['age_centers'] ))+' complete.  Time: '+str('%1.2f' % (time.time()-starttime_loop))+' sec')

# Reshape the data arrays
recon_mean = np.transpose(recon_mean)
recon_ens  = np.swapaxes(recon_ens,0,2)
prior_mean = np.transpose(prior_mean)
prior_ens  = np.swapaxes(prior_ens,0,2)

# Reshape the gridded reconstruction to a lat-lon grid
recon_mean_grid = np.reshape(recon_mean[:,:n_varslatlon], (n_ages,              n_vars,n_lat,n_lon))
recon_ens_grid  = np.reshape(recon_ens[:,:,:n_varslatlon],(n_ages,n_ens_to_save,n_vars,n_lat,n_lon))
prior_mean_grid = np.reshape(prior_mean[:,:n_varslatlon], (n_ages,              n_vars,n_lat,n_lon))
prior_ens_grid  = np.reshape(prior_ens[:,:,:n_varslatlon],(n_ages,n_ens_to_save,n_vars,n_lat,n_lon))

# Put the proxy reconstructions into separate variables
recon_mean_proxies = recon_mean[:,n_varslatlon:]
recon_ens_proxies  = recon_ens[:,:,n_varslatlon:]

# Store the options into as list to save
n_options = len(options.keys())
options_list = []
for key,value in options.items():
    options_list.append(key+':'+str(value))


#%% SAVE THE OUTPUT

# Set variables for saving the data
time_str = str(datetime.datetime.now()).replace(' ','_')
output_filename = 'holocene_recon_'+time_str+'_'+exp_name_full
output_filename = output_filename.replace(':','_').replace('.','_')
output_dir = options['data_dir']+'results/'
print('Saving the reconstruction as:\n  '+output_dir+output_filename+'.nc')

# Create new dataset
data_xarray_output = xr.Dataset(
    {
        'proxyprior_mean':   (['ages','proxy'],                prior_proxy_means.astype("float32"),                             {'units':'proxy dependent'}),
        'proxyprior_ens':    (['ages','ens_selected','proxy'], prior_proxy_ens.astype("float32"),                               {'units':'proxy dependent'}),
        'proxyrecon_mean':   (['ages','proxy'],                recon_mean_proxies.astype("float32"),                            {'units':'proxy dependent'}),
        'proxyrecon_ens':    (['ages','ens_selected','proxy'], recon_ens_proxies.astype("float32"),                             {'units':'proxy dependent'}),
        'proxy_values':      (['ages','proxy'],                np.transpose(proxy_data['values_binned']).astype("float32"),     {'units':'proxy dependent'}),
        'proxy_resolutions': (['ages','proxy'],                np.transpose(proxy_data['resolution_binned']).astype("float32"), {'units':'multiple of timestep'}),
        'proxy_metadata':    (['proxy','metadata'],            proxy_data['metadata'].astype(str)),
        'options':           (['exp_options'],                 np.array(options_list).astype(str)),
        'proxies_selected':  (['proxy'],                       proxy_ind_selected.astype(int),        {'units':'Logical'}),
        'proxies_assimilated': (['ages','proxy'],              proxies_to_assimilate_all.astype(int), {'units':'Logical'}),
    },
    coords={
        'ages':        (['ages'],        proxy_data['age_centers'],{'units':'yr BP'}),
        'ens':         (['ens'],         np.arange(n_ens),         {'description':'ensemble members'}),
        'ens_selected':(['ens_selected'],ind_random_ens_to_save,   {'description':'selected ensemble members'}),
        'lat':         (['lat'],         model_data['lat'],        {'units':'degrees_north'}),
        'lon':         (['lon'],         model_data['lon'],        {'units':'degrees_east'}),
    },
    attrs={
        'dataset_name':'Holocene Reconstruction',
    },
)

# Add additional variables
for i,var_name in enumerate(options['vars_to_reconstruct']):
    data_xarray_output['recon_'+var_name+'_mean']        = (['ages','lat','lon'],               recon_mean_grid[:,i,:,:].astype("float32"), {'units':'degC or mm/day'})
    data_xarray_output['recon_'+var_name+'_ens']         = (['ages','ens_selected','lat','lon'],recon_ens_grid[:,:,i,:,:].astype("float32"),{'units':'degC or mm/day'})
    data_xarray_output['recon_'+var_name+'_global_mean'] = (['ages','ens'],                     recon_global_all[:,:,i].astype("float32"),  {'units':'degC or mm/day'})
    data_xarray_output['prior_'+var_name+'_mean']        = (['ages','lat','lon'],               prior_mean_grid[:,i,:,:].astype("float32"), {'units':'degC or mm/day'})
    data_xarray_output['prior_'+var_name+'_ens']         = (['ages','ens_selected','lat','lon'],prior_ens_grid[:,:,i,:,:].astype("float32"),{'units':'degC or mm/day'})
    data_xarray_output['prior_'+var_name+'_global_mean'] = (['ages','ens'],                     prior_global_all[:,:,i].astype("float32"),  {'units':'degC or mm/day'})

# Add uncertainty values, which are sometimes one or two dimensions
if   len(proxy_data['uncertainty'].shape) == 1: data_xarray_output['uncertainty'] = (['proxy'],       proxy_data['uncertainty'].astype("float32"),{'units':'proxy dependent'})
elif len(proxy_data['uncertainty'].shape) == 2: data_xarray_output['uncertainty'] = (['proxy','ages'],proxy_data['uncertainty'].astype("float32"),{'units':'proxy dependent'})

# Save the output
data_xarray_output.to_netcdf(output_dir+output_filename+'.nc')

# Wrap up
endtime_total = time.time()  # End timer
print('Total time: '+str('%1.2f' % ((endtime_total-starttime_total)/60))+' minutes')
print(' === Reconstruction complete ===')
