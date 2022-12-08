#==============================================================================
# Functions for loading proxy data for the data assimilation project.
#    author: Michael P. Erb
#    date  : 3/16/2022
#==============================================================================

import da_utils
import da_pseudoproxies
import numpy as np
import pickle
import lipd
import glob
from scipy import interpolate

# A function to load the chosen proxy datasets
def load_proxies(options):
    #
    # Set the necessary directories
    dir_proxies_temp12k = options['data_dir']+'proxies/temp12k/'
    dir_proxies_pages2k = options['data_dir']+'proxies/pages2k/'
    dir_proxies_pseudo  = options['data_dir']+'proxies/pseudoproxies/'
    collection_all = []
    proxy_ts = []
    #
    n_datasets = len(options['proxy_datasets_to_assimilate'])
    for i,proxy_dataset in enumerate(options['proxy_datasets_to_assimilate']):
        print('Loading proxy dataset '+str(i+1)+'/'+str(n_datasets)+': '+proxy_dataset)
        if proxy_dataset == 'temp12k':
            #
            # Load the Temp12k proxy metadata
            file_to_open = open(dir_proxies_temp12k+'Temp12k1_0_2.pkl','rb')
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
                if (all_ts_12k[i]['dataSetName'] == 'Alley.GISP2.2000') and (all_ts_12k[i]['paleoData_variableName'] == 'temperature') and (np.max(np.array(all_ts_12k[i]['age']).astype(float)) < 50):
                    print('Fixing GISP2 ages:',all_ts_12k[i]['paleoData_variableName'],', Index:',i)
                    all_ts_12k[i]['age'] = gisp2_ages
            #
            proxy_ts_temp12k = lipd.filterTs(all_ts_12k,      'paleoData_inCompilation == Temp12k')
            proxy_ts_temp12k = lipd.filterTs(proxy_ts_temp12k,'paleoData_units == degC')
            if options['reconstruction_type'] == 'absolute': proxy_ts_temp12k = lipd.filterTs(proxy_ts_temp12k,'paleoData_datum == abs')
            #
            proxy_ts = proxy_ts + proxy_ts_temp12k
            collection_all = collection_all + ([proxy_dataset] * len(proxy_ts_temp12k))
            #
            # Some proxies have problems in the metadata.  Fix them here.
            for i in range(len(proxy_ts)):
                if proxy_ts[i]['paleoData_TSid'] == 'RXEc3JaUSUk': proxy_ts[i]['paleoData_temperature12kUncertainty'] = 2.1  # This record has an uncertainty value of "3; 2", but it should be 2.1.
                if proxy_ts[i]['paleoData_TSid'] == 'RWzl4NCma8r': proxy_ts[i]['paleoData_interpretation'][0]['seasonality'] = 'summer'  # This record is lacking a seasonality field.
            #
        elif proxy_dataset == 'pages2k':
            #
            # Load the PAGES2k proxies
            file_to_open = open(dir_proxies_pages2k+'proxies_pages2k_temp_with_psms.pkl','rb')
            proxy_ts_pages2k = pickle.load(file_to_open)
            file_to_open.close()
            #
            proxy_ts = proxy_ts + proxy_ts_pages2k
            collection_all = collection_all + ([proxy_dataset] * len(proxy_ts_pages2k))
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
            proxy_ts_pseudo = pickle.load(file_to_open)
            file_to_open.close()
            #
            proxy_ts = proxy_ts + proxy_ts_pseudo
            collection_all = collection_all + ([proxy_dataset] * len(proxy_ts_pseudo))
            #
        else:
            print('ERROR: invalid proxy dataset: '+proxy_dataset)
    #
    # Process proxy data
    return proxy_ts,collection_all


# Process the proxy data
def process_proxies(proxy_ts,collection_all,options):
    #
    print('\n=== Processing proxy data. This can take a few minutes. Please wait. ===')
    #
    # Set age range to reconstruct, as well as the reference period (The -0.5 accounts for the fact that age years are represented as whole numbers)
    age_bounds = np.arange(options['age_range_to_reconstruct'][0],options['age_range_to_reconstruct'][1]+1,options['time_resolution']) - 0.5
    age_centers = (age_bounds[:-1]+age_bounds[1:])/2
    #
    # Set the maximum proxy resolution
    max_res_value = int(options['maximum_resolution']/options['time_resolution'])
    #
    # Get dimensions
    n_ages    = len(age_centers)
    n_proxies = len(proxy_ts)
    #
    # Set up arrays for the processed proxy data to be stored in
    proxy_data = {}
    proxy_data['values_binned']     = np.zeros((n_proxies,n_ages));         proxy_data['values_binned'][:]     = np.nan
    proxy_data['resolution_binned'] = np.zeros((n_proxies,n_ages));         proxy_data['resolution_binned'][:] = np.nan
    proxy_data['uncertainty']       = np.zeros((n_proxies));                proxy_data['uncertainty'][:]       = np.nan
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
    for i in range(n_proxies):
        #
        # Get proxy data
        proxy_ages   = np.array(proxy_ts[i]['age']).astype(float)
        proxy_values = np.array(proxy_ts[i]['paleoData_values']).astype(float)
        #
        # If any NaNs exist in the ages, remove those values
        proxy_values = proxy_values[np.isfinite(proxy_ages)]
        proxy_ages   = proxy_ages[np.isfinite(proxy_ages)]
        #
        # Sort the data so that ages go from newest to oldest
        ind_sorted = np.argsort(proxy_ages)
        proxy_values = proxy_values[ind_sorted]
        proxy_ages   = proxy_ages[ind_sorted]
        #
        # INTERPOLATION
        # To interpolate the proxy data to the base resolution (by default: decadal):
        #   1. Values in the same year are averaged
        #   2. Records are then interpolated to annual using nearest neighbor interpolation
        #   3. Records are binned to the base resolution
        # To get the mean resolution of the proxy data at each time interval, it is treated in a similar way.
        #
        # Average values in the same year
        if min(proxy_ages[1:]-proxy_ages[:-1]) < 1:
            proxy_values_ann = []
            proxy_ages_ann   = []
            for int_age in np.arange(int(np.floor(proxy_ages[0])),int(np.ceil(proxy_ages[-1])+1)):
                ind_in_year = np.where((proxy_ages > int_age) & (proxy_ages <= (int_age+1)))[0]
                if len(ind_in_year) > 0:
                    proxy_values_ann.append(np.nanmean(proxy_values[ind_in_year]))
                    proxy_ages_ann.append(np.nanmean(proxy_ages[ind_in_year]))
            #
            proxy_values_ann = np.array(proxy_values_ann)
            proxy_ages_ann   = np.array(proxy_ages_ann)
        else:
            proxy_values_ann = proxy_values
            proxy_ages_ann   = proxy_ages
        #
        # Represent annual proxy ages as integers, where e.g., age 100 represent values <=100 and >99
        proxy_ages = np.ceil(proxy_ages)
        #
        # Compute age bounds of the proxy observations as the midpoints between data
        proxy_age_bounds = (proxy_ages_ann[1:]+proxy_ages_ann[:-1])/2
        end_newest = proxy_ages_ann[0]  - (proxy_ages_ann[1]-proxy_ages_ann[0])/2
        end_oldest = proxy_ages_ann[-1] + (proxy_ages_ann[-1]-proxy_ages_ann[-2])/2
        proxy_age_bounds = np.insert(proxy_age_bounds,0,end_newest)
        proxy_age_bounds = np.append(proxy_age_bounds,end_oldest)
        proxy_res_ann = proxy_age_bounds[1:] - proxy_age_bounds[:-1]
        #
        """
        # Checking calculations
        import matplotlib.pyplot as plt
        plt.plot(proxy_ages,proxy_values,'ko')
        plt.plot(proxy_ages_ann,proxy_values_ann,'b-')
        plt.show()
        """
        #
        # Use nearest neighbor interpolation to get data to the interpolation resolution
        interp_res = 1
        interp_function = interpolate.interp1d(proxy_ages_ann,proxy_values_ann,kind='nearest',bounds_error=False,fill_value='extrapolate')
        proxy_ages_interp = np.arange(int(np.ceil(end_newest)),int(np.floor(end_oldest))+interp_res,interp_res)
        proxy_values_interp = interp_function(proxy_ages_interp)
        #
        # Use nearest neighbor interpolation to get the resolution to annual
        interp_function_res = interpolate.interp1d(proxy_ages_ann,proxy_res_ann,kind='nearest',bounds_error=False,fill_value='extrapolate')
        proxy_res_interp = interp_function_res(proxy_ages_interp)
        #
        # Bin the annual data to the base resolution
        proxy_values_12ka = np.zeros((n_ages)); proxy_values_12ka[:] = np.nan
        proxy_res_12ka    = np.zeros((n_ages)); proxy_res_12ka[:]    = np.nan
        for j in range(n_ages):
            ind_selected = np.where((proxy_ages_interp >= age_bounds[j]) & (proxy_ages_interp < age_bounds[j+1]))[0]
            proxy_values_12ka[j] = np.nanmean(proxy_values_interp[ind_selected])
            res_avg              = np.nanmean(proxy_res_interp[ind_selected])
            if np.isnan(res_avg): proxy_res_12ka[j] = np.nan
            else:                 proxy_res_12ka[j] = int(round(res_avg / options['time_resolution']))
        #
        """
        # Checking calculations
        import matplotlib.pyplot as plt
        plt.plot(proxy_ages,proxy_values,'ko')
        plt.plot(proxy_ages_interp,proxy_values_interp,'b-')
        plt.plot(age_centers,proxy_values_12ka,'g-')
        plt.show()
        """
        #
        # If the reconstruction type is "relative," remove the mean of the reference period
        if options['reconstruction_type'] == 'relative':
            ind_ref = np.where((proxy_ages_interp >= options['reference_period'][0]) & (proxy_ages_interp < options['reference_period'][1]))[0]
            proxy_values_12ka = proxy_values_12ka - np.nanmean(proxy_values_interp[ind_ref])
            if np.isnan(proxy_values_interp[ind_ref]).all(): print('No data in reference period, index: '+str(i)); no_ref_data += 1
        #
        # Set resolutions to a minimum of 1 and a maximum of max_res_value
        proxy_res_12ka[proxy_res_12ka < 1] = 1
        proxy_res_12ka[proxy_res_12ka > max_res_value] = max_res_value
        #
        # Save to common variables (y and ya)
        proxy_data['values_binned'][i,:]     = proxy_values_12ka
        proxy_data['resolution_binned'][i,:] = proxy_res_12ka
        #
        # Get proxy metdata
        missing_uncertainty_value = np.nan
        proxy_lat                 = proxy_ts[i]['geo_meanLat']
        proxy_lon                 = proxy_ts[i]['geo_meanLon']
        proxy_seasonality_txt     = proxy_ts[i]['paleoData_interpretation'][0]['seasonality']
        proxy_seasonality_general = proxy_ts[i]['paleoData_interpretation'][0]['seasonalityGeneral']
        try:    proxy_uncertainty = proxy_ts[i]['paleoData_temperature12kUncertainty']
        except: proxy_uncertainty = missing_uncertainty_value; missing_uncertainty += 1
        proxy_uncertainty = float(proxy_uncertainty)
        proxy_data['archivetype'].append(proxy_ts[i]['archiveType'])
        proxy_data['proxytype'].append(proxy_ts[i]['paleoData_proxy'])
        proxy_data['units'].append(proxy_ts[i]['paleoData_units'])
        #
        # Convert seasonality to a list of months, with negative values corresponding to the previous year.
        if 'seasonality_array' in list(proxy_ts[i].keys()):
            proxy_seasonality_array = proxy_ts[i]['seasonality_array']
        else:
            proxy_seasonality = da_utils.interpret_seasonality(proxy_seasonality_txt,proxy_lat,unknown_option='annual')
            proxy_seasonality_array = np.array(proxy_seasonality.split()).astype(int)
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
        proxy_data['metadata'][i,0] = proxy_ts[i]['dataSetName']
        proxy_data['metadata'][i,1] = proxy_ts[i]['paleoData_TSid']
        proxy_data['metadata'][i,2] = str(proxy_lat)
        proxy_data['metadata'][i,3] = str(proxy_lon)
        proxy_data['metadata'][i,4] = str(proxy_seasonality_array)
        proxy_data['metadata'][i,5] = proxy_seasonality_general
        proxy_data['metadata'][i,6] = str(np.median(proxy_ages[1:]-proxy_ages[:-1]))  #TODO: Consider calculating this a different way.
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
    print('Proxies loaded        (n='+str(len(proxy_ts))+')')
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

