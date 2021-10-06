#==============================================================================
# Functions for loading code for the data assimilation project.
#    author: Michael P. Erb
#    date  : 9/3/2020
#==============================================================================

import da_utils
import da_pseudoproxies
import numpy as np
import pickle
import glob


# A function to load the chosen proxy datasets
def load_proxies(options):
    #
    # Set the necessary directories
    dir_proxies_temp12k = options['data_dir']+'proxies/temp12k/'
    dir_proxies_pages2k = options['data_dir']+'proxies/pages2k/'
    dir_proxies_pseudo  = options['data_dir']+'proxies/pseudoproxies/'
    collection_all = []
    #
    append_proxies = False
    n_datasets = len(options['proxy_datasets_to_assimilate'])
    for i,proxy_dataset in enumerate(options['proxy_datasets_to_assimilate']):
        print('Loading proxy dataset '+str(i+1)+'/'+str(n_datasets)+': '+proxy_dataset)
        if proxy_dataset == 'temp12k':
            #
            # Load lipd
            #if 'lipd_dir' in options: sys.path.append(options['lipd_dir'])
            import lipd
            #
            # Load the Temp12k proxy metadata
            file_to_open = open(dir_proxies_temp12k+'Temp12k1_0_1.pkl','rb')
            proxies_all_12k = pickle.load(file_to_open)['D']
            file_to_open.close()
            #
            # Extract the time series and use only those which are in Temp12k and in units of degC
            all_ts_12k = lipd.extractTs(proxies_all_12k)
            #
            # Fix the GISP2 ages - Note: this is a temporary fix, since lipd isn't loading the right ages.
            for i in range(len(all_ts_12k)):
                if (all_ts_12k[i]['dataSetName'] == 'Alley.GISP2.2000') and (all_ts_12k[i]['paleoData_variableName'] == 'age'): gisp2_ages = all_ts_12k[i]['paleoData_values']
            #
            for i in range(len(all_ts_12k)):
                if (all_ts_12k[i]['dataSetName'] == 'Alley.GISP2.2000') and (all_ts_12k[i]['paleoData_variableName'] == 'temperature') and (np.max(np.array(all_ts_12k[i]['age']).astype(np.float)) < 50):
                    print('Fixing GISP2 ages:',all_ts_12k[i]['paleoData_variableName'],', Index:',i)
                    all_ts_12k[i]['age'] = gisp2_ages
            #
            filtered_ts_temp12k = lipd.filterTs(all_ts_12k,         'paleoData_inCompilation == Temp12k')
            filtered_ts         = lipd.filterTs(filtered_ts_temp12k,'paleoData_units == degC')
            if options['reconstruction_type'] == 'absolute': filtered_ts = lipd.filterTs(filtered_ts,'paleoData_datum == abs')
            #
            # Specify the collection
            collection_all = collection_all + ([proxy_dataset] * len(filtered_ts))
            append_proxies = True
            #
        elif proxy_dataset == 'temp12k_screened':
            #
            # Load the screened Temp12k proxies (Note: this is for a custom-screened proxy file you make yourself)
            file_to_open = open(dir_proxies_temp12k+'Temp12k_screened.pkl','rb')
            filtered_ts_temp12k = pickle.load(file_to_open)
            file_to_open.close()
            #
            collection_all = collection_all + ([proxy_dataset] * len(filtered_ts_temp12k))
            if append_proxies == False: filtered_ts = filtered_ts_temp12k
            else: filtered_ts = filtered_ts + filtered_ts_temp12k
            #
        elif proxy_dataset == 'pages2k':
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
            # Load the screened PAGES2k proxies
            file_to_open = open(dir_proxies_pages2k+'proxies_pages2k_temp_with_psms_screened.pkl','rb')
            filtered_ts_pages2k_screened = pickle.load(file_to_open)
            file_to_open.close()
            #
            collection_all = collection_all + ([proxy_dataset] * len(filtered_ts_pages2k_screened))
            if append_proxies == False: filtered_ts = filtered_ts_pages2k_screened
            else: filtered_ts = filtered_ts + filtered_ts_pages2k_screened
            #
        elif proxy_dataset[0:7] == 'pseudo_':
            #
            # Check to see if the file exists.  If not, create it.
            proxy_filename = proxy_dataset+'.pkl'
            filenames_all = glob.glob(dir_proxies_pseudo+'*.pkl')
            filenames_all = [filename.split('/')[-1] for filename in filenames_all]
            if proxy_filename not in filenames_all:
                print('File '+dir_proxies_pseudo+proxy_filename+' does not exist.  Creating it now.')
                proxies_to_use = proxy_dataset.split('_')[1]
                model_to_use   = proxy_dataset.split('_')[3]
                noise_to_use   = proxy_dataset.split('_')[5]
                da_pseudoproxies.make_pseudoproxies(proxies_to_use,model_to_use,noise_to_use,options)
                print('File '+dir_proxies_pseudo+proxy_filename+' created!')
            #
            # Load the pseudoproxies
            file_to_open = open(dir_proxies_pseudo+proxy_filename,'rb')
            filtered_ts_pseudo = pickle.load(file_to_open)
            file_to_open.close()
            #
            collection_all = collection_all + ([proxy_dataset] * len(filtered_ts_pseudo))
            if append_proxies == False: filtered_ts = filtered_ts_pseudo
            else: filtered_ts = filtered_ts + filtered_ts_pseudo
            #
        else:
            print('ERROR: invalid proxy dataset: '+proxy_dataset)
    #
    # Process proxy data
    return filtered_ts,collection_all


# Process the proxy data
def process_proxies(filtered_ts,collection_all,options):
    #
    # Set age range to reconstruct, as well as the reference period
    age_bounds = np.arange(options['age_range_to_reconstruct'][0],options['age_range_to_reconstruct'][1]+1,options['time_resolution'])
    age_centers = (age_bounds[:-1]+age_bounds[1:])/2
    if options['reconstruction_type'] == 'relative':
        age_bounds_ref = np.arange(options['reference_period'][0],options['reference_period'][1]+1,options['time_resolution'])
        age_centers_ref = (age_bounds_ref[:-1]+age_bounds_ref[1:])/2
    #
    # Set the maximum proxy resolution
    max_res_value = int(options['maximum_resolution']/options['time_resolution'])
    #
    # Get dimensions
    n_ages    = len(age_centers)
    n_proxies = len(filtered_ts)
    #
    # Set up arrays (y, ya, HXb, and R)
    proxy_data = {}
    proxy_data['values_binned']     = np.zeros((n_proxies,n_ages));         proxy_data['values_binned'][:]     = np.nan  # y
    proxy_data['resolution_binned'] = np.zeros((n_proxies,n_ages));         proxy_data['resolution_binned'][:] = np.nan  # ya
    proxy_data['uncertainty']       = np.zeros((n_proxies));                proxy_data['uncertainty'][:]       = np.nan  # R
    proxy_data['metadata']          = np.zeros((n_proxies,8),dtype=object); proxy_data['metadata'][:]          = np.nan
    proxy_data['lats']              = np.zeros((n_proxies));                proxy_data['lats'][:]              = np.nan
    proxy_data['lons']              = np.zeros((n_proxies));                proxy_data['lons'][:]              = np.nan
    proxy_data['archivetype']       = []
    proxy_data['proxytype']         = []
    proxy_data['units']             = []
    proxy_data['seasonality_array'] = {}
    #
    # Loop through proxies, saving the necessary values to common variables.
    no_ref_data = 0; missing_uncertainty = 0
    i=0
    for i in range(n_proxies):
        #
        if options['verbose_level'] > 0: print(' - Calculating estimates for proxy '+str(i)+'/'+str(n_proxies))
        #
        # Get proxy data
        proxy_ages = np.array(filtered_ts[i]['age']).astype(np.float)
        proxy_values = np.array(filtered_ts[i]['paleoData_values']).astype(np.float)
        #
        # If any NaNs exist in the ages, remove those values
        proxy_values = proxy_values[np.isfinite(proxy_ages)]
        proxy_ages = proxy_ages[np.isfinite(proxy_ages)]
        #
        # Make sure the ages go from newest to oldest
        proxy_median_res = np.median(proxy_ages[1:] - proxy_ages[:-1])
        if proxy_median_res < 0:
            proxy_values = np.flip(proxy_values)
            proxy_ages = np.flip(proxy_ages)
        #
        # Compute age bounds of the proxy observations as the midpoints between data
        proxy_age_bounds = (proxy_ages[1:]+proxy_ages[:-1])/2
        end_newest = proxy_ages[0]  - (proxy_ages[1]-proxy_ages[0])/2
        end_oldest = proxy_ages[-1] + (proxy_ages[-1]-proxy_ages[-2])/2
        proxy_age_bounds = np.insert(proxy_age_bounds,0,end_newest)
        proxy_age_bounds = np.append(proxy_age_bounds,end_oldest)
        #
        # Interpolate proxy data to the base resolution, using nearest-neighbor interpolation
        proxy_values_12ka = np.zeros((n_ages)); proxy_values_12ka[:] = np.nan
        proxy_res_12ka    = np.zeros((n_ages)); proxy_res_12ka[:]    = np.nan
        for j in range(len(proxy_age_bounds)-1):
            indices_selected = np.where((age_centers >= proxy_age_bounds[j]) & (age_centers < proxy_age_bounds[j+1]))[0]
            proxy_values_12ka[indices_selected] = proxy_values[j]
            proxy_res_12ka[indices_selected] = int(round((proxy_age_bounds[j+1] - proxy_age_bounds[j]) / options['time_resolution']))
        #
        # If the reconstruction type is "relative," remove the mean of the reference period
        # Note: This is done using a somewhat-complicated way in case the reference period isn't contained within the reconstruction period.  #TODO: Check this.
        if options['reconstruction_type'] == 'relative':
            proxy_values_12ka_ref = np.zeros((len(age_centers_ref))); proxy_values_12ka_ref[:] = np.nan
            for j in range(len(proxy_age_bounds)-1):
                indices_selected_ref = np.where((age_centers_ref >= proxy_age_bounds[j]) & (age_centers_ref < proxy_age_bounds[j+1]))[0]
                proxy_values_12ka_ref[indices_selected_ref] = proxy_values[j]
            #
            proxy_values_12ka = proxy_values_12ka - np.nanmean(proxy_values_12ka_ref)
            if np.isnan(proxy_values_12ka_ref).all(): print('No data in reference period, index: '+str(i)); no_ref_data += 1
        #
        #plt.plot(age_centers,proxy_values_12ka)
        #plt.show()
        #
        # Set resolutions to a minimum of 1 and a maximum of max_res_value
        proxy_res_12ka[proxy_res_12ka == 0] = 1
        proxy_res_12ka[proxy_res_12ka > max_res_value] = max_res_value
        #
        # Save to common variables (y and ya)
        proxy_data['values_binned'][i,:]     = proxy_values_12ka
        proxy_data['resolution_binned'][i,:] = proxy_res_12ka
        #
        # Get proxy metdata
        missing_uncertainty_value = np.nan  #TODO eventually: Should I be using nan or a number (2.1?) for unknown uncertainties?
        proxy_lat                 = filtered_ts[i]['geo_meanLat']
        proxy_lon                 = filtered_ts[i]['geo_meanLon']
        proxy_seasonality_txt     = filtered_ts[i]['paleoData_interpretation'][0]['seasonality']
        proxy_seasonality_general = filtered_ts[i]['paleoData_interpretation'][0]['seasonalityGeneral']
        try:    proxy_uncertainty = filtered_ts[i]['paleoData_temperature12kUncertainty']
        except: proxy_uncertainty = missing_uncertainty_value; missing_uncertainty += 1
        if proxy_uncertainty == 'NA': proxy_uncertainty = missing_uncertainty_value; missing_uncertainty += 1
        proxy_data['archivetype'].append(filtered_ts[i]['archiveType'])
        proxy_data['proxytype'].append(filtered_ts[i]['paleoData_proxy'])
        proxy_data['units'].append(filtered_ts[i]['paleoData_units'])
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
        elif options['assign_seasonality'] == 'jja':
            proxy_seasonality_array = np.array([6,7,8])
        elif options['assign_seasonality'] == 'djf':
            proxy_seasonality_array = np.array([-12,1,2])
        #
        proxy_data['seasonality_array'][i] = proxy_seasonality_array
        #
        # Save some metadata to common variables
        if proxy_lon < 0: proxy_lon = proxy_lon+360
        proxy_data['uncertainty'][i] = np.square(proxy_uncertainty)  # Proxy uncertainty was give as RMSE, but the code uses MSE
        proxy_data['metadata'][i,0] = filtered_ts[i]['dataSetName']
        proxy_data['metadata'][i,1] = filtered_ts[i]['paleoData_TSid']
        proxy_data['metadata'][i,2] = str(proxy_lat)
        proxy_data['metadata'][i,3] = str(proxy_lon)
        proxy_data['metadata'][i,4] = str(proxy_seasonality_array)
        proxy_data['metadata'][i,5] = proxy_seasonality_general
        proxy_data['metadata'][i,6] = str(np.median(proxy_ages[1:]-proxy_ages[:-1]))
        proxy_data['metadata'][i,7] = collection_all[i]
        proxy_data['lats'][i] = proxy_lat
        proxy_data['lons'][i] = proxy_lon
        #
    proxy_data['age_centers'] = age_centers
    proxy_data['archivetype'] = np.array(proxy_data['archivetype'])
    proxy_data['proxytype']   = np.array(proxy_data['proxytype'])
    proxy_data['units']       = np.array(proxy_data['units'])
    #
    print('\n=== PROXY DATA LOADED ===')
    print('Proxy datasets loaded (n='+str(len(options['proxy_datasets_to_assimilate']))+'):'+str(options['proxy_datasets_to_assimilate']))
    print('Proxies loaded        (n='+str(len(filtered_ts))+')')
    print('---')
    print('Proxies without data in reference period (n='+str(no_ref_data)+')')
    print('Proxies without uncertainty value        (n='+str(missing_uncertainty)+'). Set to '+str(missing_uncertainty_value))
    print('---')
    print('Data stored in dictionary "proxy_data", with keys and dimensions:')
    for key in list(proxy_data.keys()):
        try:    print('%20s %-15s' % (key,str(proxy_data[key].shape)))
        except: print('%20s %-15s' % (key,str(len(proxy_data[key]))))
    print('=========================\n')
    #
    return proxy_data

