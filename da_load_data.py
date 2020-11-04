#==============================================================================
# Functions for loading code for the data assimilation project.
#    author: Michael P. Erb
#    date  : 9/3/2020
#==============================================================================

import sys
import da_utils_lmr
import da_process_models
import numpy as np
import pickle
import xarray as xr
from datetime import datetime
import netCDF4
import glob


# A function to load model data
def load_model_data(options):
    #
    model_dir          = options['data_dir']+'models/processed_model_data/'
    original_model_dir = options['data_dir']+'models/original_model_data/'
    age_range_model_txt = str(options['age_range_model'][1]-1)+'-'+str(options['age_range_model'][0])
    #
    # Load the model data
    n_models = len(options['models_for_prior'])
    #i=0;model=options['models_for_prior'][i]
    for i,model in enumerate(options['models_for_prior']):
        #
        print('Loading model '+str(i+1)+'/'+str(n_models)+': '+model)
        #
        # Get the model filename
        model_filename = model+'.'+age_range_model_txt+'BP.TREFHT.timeres_'+str(options['time_resolution'])+'.nc'
        #
        # Check to see if the file exists.  If not, create it.
        filenames_all = glob.glob(model_dir+'*.nc')
        filenames_all = [filename.split('/')[-1] for filename in filenames_all]
        if model_filename not in filenames_all:
            print('File '+model_dir+model_filename+' does not exist.  Creating it now.')
            da_process_models.process_models(model,options['time_resolution'],options['age_range_model'],model_dir,original_model_dir)
            print('File '+model_dir+model_filename+' created!')
        #
        # Load model surface air temperature
        handle_model = xr.open_dataset(model_dir+model_filename,decode_times=False)
        tas_model_individual        = handle_model['tas'].values
        lat_model                   = handle_model['lat'].values
        lon_model                   = handle_model['lon'].values
        age_model_individual        = handle_model['age'].values
        time_ndays_model_individual = handle_model['days_per_month_all'].values
        handle_model.close()
        #
        # Trim the data #TODO: I've commented this out.  What was it's original purpose?
        #if (model == 'hadcm3_regrid') or (model == 'trace_regrid'):
        #    indices_selected = np.where((age_model_individual >= 0) & (age_model_individual < 13000))[0]
        #    tas_model_individual        = tas_model_individual[indices_selected,:,:,:]
        #    time_ndays_model_individual = time_ndays_model_individual[indices_selected,:]
        #    age_model_individual        = age_model_individual[indices_selected]
        #
        # In each model, central values will not be selected within max_resolution/2 of the edges
        valid_model_inds_individual = np.full((tas_model_individual.shape[0]),True,dtype=bool)
        buffer = int(np.floor((options['maximum_resolution']/options['time_resolution'])/2))
        if buffer > 0:
            valid_model_inds_individual[:buffer]  = False
            valid_model_inds_individual[-buffer:] = False
        #
        # Set the model number for each data point
        model_num_individual = np.full((tas_model_individual.shape[0]),(i+1),dtype=int)
        #
        # Join the values together
        if i == 0:
            tas_model        = tas_model_individual
            age_model        = age_model_individual
            time_ndays_model = time_ndays_model_individual
            valid_model_inds = valid_model_inds_individual
            model_num        = model_num_individual
        else:
            tas_model        = np.concatenate((tas_model,       tas_model_individual),       axis=0)
            age_model        = np.concatenate((age_model,       age_model_individual),       axis=0)
            time_ndays_model = np.concatenate((time_ndays_model,time_ndays_model_individual),axis=0)
            valid_model_inds = np.concatenate((valid_model_inds,valid_model_inds_individual),axis=0)
            model_num        = np.concatenate((model_num,       model_num_individual),       axis=0)
    #
    return tas_model,age_model,time_ndays_model,valid_model_inds,model_num,lat_model,lon_model


# A function to load the chosen proxy datasets
def load_proxies(options):
    #
    # Set the necessary directories
    dir_proxies_temp12k = options['data_dir']+'proxies/temp12k/'
    dir_proxies_pages2k = options['data_dir']+'proxies/pages2k/'
    collection_all = []
    #
    append_proxies = False
    n_datasets = len(options['proxy_datasets_to_assimilate'])
    for i,proxy_dataset in enumerate(options['proxy_datasets_to_assimilate']):
        if proxy_dataset == 'temp12k':
            #
            print('Loading proxy dataset '+str(i+1)+'/'+str(n_datasets)+': Temp12k')
            #
            # Load lipd
            if 'lipd_dir' in options: sys.path.append(options['lipd_dir'])
            import lipd
            #
            # Load the Temp12k proxy metadata
            file_to_open = open(dir_proxies_temp12k+'Temp12k1_0_1.pkl','rb')
            proxies_all_12k = pickle.load(file_to_open)['D']
            file_to_open.close()
            #
            # Extract the time series and use only those which are in Temp12k and in units of degC
            all_ts_12k = lipd.extractTs(proxies_all_12k)
            filtered_ts_temp12k = lipd.filterTs(all_ts_12k,         'paleoData_inCompilation == Temp12k')
            filtered_ts         = lipd.filterTs(filtered_ts_temp12k,'paleoData_units == degC')
            #
            # Specify the collection
            collection_all = collection_all + ([proxy_dataset] * len(filtered_ts))
            append_proxies = True
            #
        elif proxy_dataset == 'pages2k':
            #
            print('Loading proxy dataset '+str(i+1)+'/'+str(n_datasets)+': PAGES2k')
            #
            # Load the PAGES2k proxies
            file_to_open = open(dir_proxies_pages2k+'proxies_pages2k_temp_with_psms.pkl','rb')
            filtered_ts_pages2k = pickle.load(file_to_open)
            file_to_open.close()
            #
            collection_all = collection_all + ([proxy_dataset] * len(filtered_ts_pages2k))
            if append_proxies == False: filtered_ts = filtered_ts_pages2k
            else: filtered_ts = filtered_ts + filtered_ts_pages2k
            #
        elif proxy_dataset == 'pages2k_screened':
            #
            if 'pages2k' in options['proxy_datasets_to_assimilate']:
                print('Warning: Do not load both "pages2k" and "pages2k_screened".')
                print('"pages2k_screened" is a subset of "pages2k".')
                print('Skipping "pages2k_screened".')
                continue
            #
            print('Loading proxy dataset '+str(i+1)+'/'+str(n_datasets)+': PAGES2k_screened')
            #
            # Load the screened PAGES2k proxies
            file_to_open = open(dir_proxies_pages2k+'proxies_pages2k_temp_with_psms_screened.pkl','rb')
            filtered_ts_pages2k_screened = pickle.load(file_to_open)
            file_to_open.close()
            #
            collection_all = collection_all + ([proxy_dataset] * len(filtered_ts_pages2k_screened))
            if append_proxies == False: filtered_ts = filtered_ts_pages2k_screened
            else: filtered_ts = filtered_ts + filtered_ts_pages2k_screened
    #
    return filtered_ts,collection_all


# A function to compute a localization matrix
def loc_matrix(options,lon_model,lat_model,proxy_metadata_all):
    #
    # Get dimensions
    n_proxies = proxy_metadata_all.shape[0]
    n_latlon  = len(lat_model) * len(lon_model)
    n_state = n_latlon + n_proxies
    #
    # Compute the localization values for every proxy
    if options['localization_radius'] == 'None':
        proxy_localization_all = np.ones((n_proxies,n_state))
    else:
        #
        # Get lat and lon values for the prior
        lon_model_2d,lat_model_2d = np.meshgrid(lon_model,lat_model)
        lat_prior = np.reshape(lat_model_2d,(n_latlon))
        lon_prior = np.reshape(lon_model_2d,(n_latlon))
        prior_coords = np.concatenate((lat_prior[:,None],lon_prior[:,None]),axis=1)
        #
        # Include the proxy coordinates with the model coordinates
        proxy_coords_all = proxy_metadata_all[:,2:4].astype(np.float)
        prior_coords = np.append(prior_coords,proxy_coords_all,axis=0)
        #
        for i in range(n_proxies):
            #
            # Get proxy metdata
            proxy_lat = proxy_metadata_all[i,2]
            proxy_lon = proxy_metadata_all[i,3]
            #
            # Compute the localization values and save it to a common variable
            proxy_localization = da_utils_lmr.cov_localization(options['localization_radius'],proxy_lat,proxy_lon,prior_coords)
            proxy_localization_all[i,:] = proxy_localization
    #
    return proxy_localization_all


# A function to output some data
def output_materials(options,tas_model_annual,lat_model,lon_model):
    #
    time_str = str(datetime.now()).replace(' ','_')
    output_dir = options['data_dir']+'results/'
    output_filename = 'materials_holocene_recon_timevarying_'+str(options['prior_window'])+'yr_prior_'+time_str
    #
    # Save all data into a netCDF file
    outputfile = netCDF4.Dataset(output_dir+output_filename+'.nc','w')
    outputfile.createDimension('model_age',tas_model_annual.shape[0])
    outputfile.createDimension('lat',      len(lat_model))
    outputfile.createDimension('lon',      len(lon_model))
    #
    output_tas_model = outputfile.createVariable('tas_model_annual','f8',('model_age','lat','lon',))
    output_lat       = outputfile.createVariable('model_lat',       'f8',('lat',))
    output_lon       = outputfile.createVariable('model_lon',       'f8',('lon',))
    #
    output_tas_model[:] = tas_model_annual
    output_lat[:]       = lat_model
    output_lon[:]       = lon_model
    #
    outputfile.close()


# A function to get the age indices for the prior
def get_age_indices_for_prior(options,age_model,age,valid_model_inds):
    #
    # Set age bounds for the prior
    edge_cushion = int(np.ceil(options['maximum_resolution']/2))
    prior_age_bound_recent = min(age_model) + edge_cushion - (options['time_resolution']/2)  # The prior will never select ages more recent than this
    prior_age_bound_old    = max(age_model) - edge_cushion + (options['time_resolution']/2)  # The prior will never select ages older than this
    #
    # Get the age intervals for the prior
    if options['prior_window'] == 'all':
        prior_age_window_recent = prior_age_bound_recent
        prior_age_window_old    = prior_age_bound_old
    else:
        prior_age_window_recent = age - (options['prior_window']/2)
        prior_age_window_old    = age + (options['prior_window']/2)
        if prior_age_window_recent < prior_age_bound_recent:
            prior_age_window_recent = prior_age_bound_recent
            prior_age_window_old    = prior_age_bound_recent + options['prior_window']
        elif prior_age_window_old > prior_age_bound_old:
            prior_age_window_old    = prior_age_bound_old
            prior_age_window_recent = prior_age_bound_old - options['prior_window']
    #
    age_indices_for_prior = np.where((age_model >= prior_age_window_recent) & (age_model < prior_age_window_old) & valid_model_inds)[0]
    #
    return age_indices_for_prior
