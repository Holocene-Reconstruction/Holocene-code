#==============================================================================
# Functions for loading proxy data for the data assimilation project.
#    author: Michael P. Erb
#==============================================================================

import da_utils
import numpy as np
import lipd
from scipy import interpolate
import os

# A function to load the chosen proxy datasets
def load_proxies(options):
    #
    # Create lists to the store the proxy data in
    collection_all = []
    proxy_ts       = []
    #
    #i = 0; proxy_dataset = options['proxy_datasets_to_assimilate'][i]
    for i,proxy_dataset in enumerate(options['proxy_datasets_to_assimilate']):
        #
        # Get the proxy dataset name
        print('Loading proxy dataset: '+proxy_dataset)
        proxy_dataset_name,proxy_version = proxy_dataset.split('/')
        dir_proxies = options['data_dir']+'proxies/'+proxy_dataset_name+proxy_version+'/'
        file_to_load = dir_proxies+proxy_dataset_name+proxy_version+'.npy'
        #
        # Load the EcoClimate proxy data
        if os.path.isfile(file_to_load):
            all_ts = np.load(file_to_load,allow_pickle=True)
        else:
            D = lipd.readLipd(dir_proxies)
            all_ts = lipd.extractTs(D)
            np.save(file_to_load,all_ts,allow_pickle=True)
        #
        # If there's an uncertainty file, use TSids to select data to add to compilation (allows for flexibility for metadata issues)
        try:
            proxy_uncertainties_from_file = np.genfromtxt(options['change_uncertainty'],delimiter=',',dtype='str')
            proxy_tsids_from_file = [x[0] for x in proxy_uncertainties_from_file]
            print(' --- Uncertainty file found. Using uncertainties from the following file ---')
            print(options['change_uncertainty'])
        except:
            proxy_tsids_from_file = False
        #
        # Add data based on inCompilation metadata
        proxy_ts_ds = []
        for i in range(len(all_ts)):
            # Exclude based on geo
            lon = all_ts[i]['geo_meanLon']
            if lon < 0: lon = 360 + lon
            if all_ts[i]['geo_meanLat'] < options['model_region'][0]: continue
            if all_ts[i]['geo_meanLat'] > options['model_region'][1]: continue
            if lon < options['model_region'][2]: continue
            if lon > options['model_region'][3]: continue
            # Exclude relative data for absolute reconstruction
            if options['reconstruction_type'] == 'absolute':
                if ('paleoData_datum' not in all_ts[i].keys()): continue
                if (all_ts[i]['paleoData_datum'] != 'abs'): continue  
            # Include if within provided list, and update uncertainty
            # The text file below should contain TSids and MSE for every proxy record we want to use
            if type(proxy_tsids_from_file) == list:
                if all_ts[i]['paleoData_TSid'] in proxy_tsids_from_file:
                    index_uncertainty = np.where(all_ts[i]['paleoData_TSid'] == proxy_uncertainties_from_file[:,0])[0][0]
                    uncertainty_selected = proxy_uncertainties_from_file[index_uncertainty,1]
                    if uncertainty_selected == "NA": uncertainty_selected = np.nan
                    else: uncertainty_selected = uncertainty_selected.astype(float)
                    all_ts[i]['uncertainty'] = uncertainty_selected
                    proxy_ts_ds.append(all_ts[i])
            else: 
                if 'paleoData_inCompilation' in all_ts[i].keys():
                    # Include data based on compilation
                    for comp in all_ts[i]['paleoData_inCompilation']:
                        if (proxy_version in comp['compilationVersion']) & (proxy_dataset_name == comp['compilationName']):
                            proxy_ts_ds.append(all_ts[i])

        #print(f'Number of {proxy_dataset_name} records selected:',len(proxy_ts_ds))
        #
        # Some proxies have problems in the metadata.  Fix them here.
        for i in range(len(proxy_ts_ds)):
            proxy_ts_ds[i]['paleoData_values'] = [float(x) for x in proxy_ts_ds[i]['paleoData_values']]
            if 'paleoData_interpretation' in proxy_ts_ds[i].keys():
                if 'interpretation1_variable' not in proxy_ts_ds[i].keys():
                    try:    proxy_ts_ds[i]['interpretation1_variable'] = proxy_ts_ds[i]['paleoData_interpretation'][0]['variable']
                    except: proxy_ts_ds[i]['interpretation1_variable'] = 'Not Given'
                if 'interpretation1_seasonality' not in proxy_ts_ds[i].keys():
                    try:    proxy_ts_ds[i]['interpretation1_seasonality'] = proxy_ts_ds[i]['paleoData_interpretation'][0]['seasonality']
                    except: proxy_ts_ds[i]['interpretation1_seasonality'] = 'Not Given'
                if 'interpretation1_direction' not in proxy_ts_ds[i].keys():
                    try:    proxy_ts_ds[i]['interpretation1_direction'] = proxy_ts_ds[i]['paleoData_interpretation'][0]['direction']
                    except: proxy_ts_ds[i]['interpretation1_direction'] = 'Not Given'
        
        proxy_ts = proxy_ts + proxy_ts_ds
        collection_all = collection_all + ([proxy_dataset] * len(proxy_ts_ds))
        #
    return proxy_ts,collection_all


#%% A function for getting metadata
def get_var(data_to_use,var_to_find):
    #
    if var_to_find in data_to_use.keys():
        var_value = data_to_use[var_to_find]
        if type(var_value) == list:
            var_value = var_value[0]
    else:
        var_value = 'Not given'
        print("Variable not found:",var_to_find)
    #
    return var_value


#%% Filter the proxy data
def filter_proxies_and_set_psms(proxy_ts,collection_all,options):
    #
    # Set up variables
    print('Filtering proxies and assigning PSMs')
    n_proxies = len(proxy_ts)
    print('Proxies, all loaded:',n_proxies)
    boolean_selected_all = np.array([False]*n_proxies)
    proxy_psms = ["None"]*n_proxies
    #
    # Get some metadata fields from proxies
    proxy_variable = []
    proxy_interp   = []
    proxy_archive  = []
    proxy_proxy    = []
    proxy_unit     = []
    for i in range(n_proxies): #paleoData_interpretation
        proxy_interp.append(get_var(proxy_ts[i],  'interpretation1_variable'))
        proxy_variable.append(get_var(proxy_ts[i],'paleoData_variableName'))
        proxy_archive.append(get_var(proxy_ts[i], 'archiveType'))
        proxy_proxy.append(get_var(proxy_ts[i],   'paleoData_proxy'))
        proxy_unit.append(get_var(proxy_ts[i],    'paleoData_units'))
    #
    # Find the proxies to keep
    n_combinations = len(options['proxies_to_use'])
    for i in range(n_combinations):
        #
        combination_selected = options['proxies_to_use'][i]
        chosen_interp   = combination_selected.split('|')[0].split(',')
        chosen_variable = combination_selected.split('|')[1].split(',')
        chosen_archive  = combination_selected.split('|')[2].split(',')
        chosen_proxy    = combination_selected.split('|')[3].split(',')
        chosen_unit     = combination_selected.split('|')[4].split(',')
        chosen_psm      = combination_selected.split('|')[5]
        boolean_interp   = np.array([value in chosen_interp   for value in proxy_interp])
        boolean_variable = np.array([value in chosen_variable for value in proxy_variable])
        boolean_archive  = np.array([value in chosen_archive  for value in proxy_archive])
        boolean_proxy    = np.array([value in chosen_proxy    for value in proxy_proxy])
        boolean_unit     = np.array([value in chosen_unit     for value in proxy_unit])
        if chosen_interp[0]   == 'any': boolean_interp[:]   = True
        if chosen_variable[0] == 'any': boolean_variable[:] = True
        if chosen_archive[0]  == 'any': boolean_archive[:]  = True
        if chosen_proxy[0]    == 'any': boolean_proxy[:]    = True
        if chosen_unit[0]     == 'any': boolean_unit[:]     = True
        boolean_selected = boolean_interp & boolean_variable & boolean_archive & boolean_proxy & boolean_unit
        ind_selected = np.where(boolean_selected)[0]
        print('Proxies in group ('+combination_selected+'):',len(ind_selected))
        #
        # Set the PSM
        for index in ind_selected:
            proxy_psms[index] = chosen_psm
        #
        # Combine each boolean combination
        boolean_selected_all = boolean_selected_all | boolean_selected
    #
    ind_selected_all = np.where(boolean_selected_all)[0]
    print('Proxies in all groups:',len(ind_selected_all))
    #
    # Get proxy data to keep
    proxy_ts_selected   = [proxy_ts[index]       for index in ind_selected_all]
    psms_selected       = [proxy_psms[index]     for index in ind_selected_all]
    collection_selected = [collection_all[index] for index in ind_selected_all]
    #
    return proxy_ts_selected,psms_selected,collection_selected


#%% Process the proxy data
def process_proxies(proxy_ts_selected,psms_selected,collection_selected,options):
    #
    print('\n=== Processing proxy data. This can take a few minutes. Please wait. ===')
    #
    # Set age range to reconstruct, as well as the reference period (The -0.5 accounts for the fact that age years are represented as whole numbers)
    age_bounds = np.arange(options['age_range_to_reconstruct'][0],options['age_range_to_reconstruct'][1]+1,options['time_resolution']) - 0.5
    age_centers = (age_bounds[:-1]+age_bounds[1:])/2
    #
    # Set the maximum proxy resolution relative to the base resolution
    max_res_value = int(options['maximum_resolution']/options['time_resolution'])
    #
    # Get dimensions
    n_ages    = len(age_centers)
    n_proxies = len(proxy_ts_selected)
    #
    # Set up arrays for the processed proxy data to be stored in
    proxy_data = {}
    proxy_data['values_binned']     = np.zeros((n_proxies,n_ages));          proxy_data['values_binned'][:]     = np.nan
    proxy_data['resolution_binned'] = np.zeros((n_proxies,n_ages));          proxy_data['resolution_binned'][:] = np.nan
    proxy_data['metadata']          = np.zeros((n_proxies,13),dtype=object); proxy_data['metadata'][:]          = np.nan
    proxy_data['lats']              = np.zeros((n_proxies));                 proxy_data['lats'][:]              = np.nan
    proxy_data['lons']              = np.zeros((n_proxies));                 proxy_data['lons'][:]              = np.nan
    proxy_data['uncertainty']       = []
    proxy_data['archivetype']       = []
    proxy_data['proxytype']         = []
    proxy_data['units']             = []
    proxy_data['interp']            = []
    proxy_data['direction']         = []
    proxy_data['seasonality_array'] = {}
    proxy_data['psm']               = []
    #
    # Loop through proxies, saving the necessary values to common variables.
    no_ref_data = 0; missing_uncertainty = 0
    i = 16
    for i in range(n_proxies):
        #
        # Get proxy data
        #print('Processing proxies:',i)
        proxy_values = np.array(proxy_ts_selected[i]['paleoData_values']).astype(float)
        proxy_ages   = np.array(proxy_ts_selected[i]['age']).astype(float)
        if np.ma.isMaskedArray(proxy_ages): proxy_ages = proxy_ages.filled(np.nan)
        else:                               proxy_ages = np.array(proxy_ages)
        #
        # If any NaNs exist in the ages, remove those values
        proxy_values = proxy_values[np.isfinite(proxy_ages)]
        proxy_ages   = proxy_ages[np.isfinite(proxy_ages)]
        #
        # If any NaNs exist in the paleoData_values, remove those values if options['remove_nans'] == True
        if options['remove_nans'] == True:
            proxy_ages   = proxy_ages[np.isfinite(proxy_values)]
            proxy_values = proxy_values[np.isfinite(proxy_values)]
        #
        # Sort the data so that ages go from newest to oldest
        ind_sorted = np.argsort(proxy_ages)
        proxy_values = proxy_values[ind_sorted]
        proxy_ages   = proxy_ages[ind_sorted]
        #
        # Get uncertainty metadata
        missing_uncertainty_value = np.nan
        try:    proxy_uncertainty = proxy_ts_selected[i]['uncertainty']
        except: proxy_uncertainty = missing_uncertainty_value; missing_uncertainty += 1
        if ((proxy_uncertainty == 'NA') or (type(proxy_uncertainty) == str)):
            proxy_uncertainty = missing_uncertainty_value; missing_uncertainty += 1
        #
        # Update the units  #TODO: Work on this (and consider using a different value than 365.25)
        #  - Precipitation: mm/yr -> mm/day
        data_units = get_var(proxy_ts_selected[i],'paleoData_units')
        if psms_selected[i] == 'calibrated_precip':
            if data_units != "mm/day":
                print('WARNING: Check units for proxy '+str(i))
        #
        # INTERPOLATION
        # To interpolate the proxy data to the base resolution (by default: centennial):
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
                ind_in_year = np.where((proxy_ages >= int_age) & (proxy_ages < (int_age+1)))[0]
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
        # Only retain interpolated values within specified number years of an actual observation
        if options['spread_limit'] != False:
            nearest_dist = np.min(np.abs(proxy_ages_interp[:, None] - proxy_ages_ann[None, :]), axis=1)
            proxy_values_interp[nearest_dist > options['spread_limit']] = np.nan
            proxy_res_interp[nearest_dist > options['spread_limit']] = np.nan
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
            # Note: The "proxy_res_12ka" variable rounded to be a factor of the base resolution
        #
        """
        # Checking calculations
        import matplotlib.pyplot as plt
        plt.plot(proxy_ages,proxy_values,'ko')
        plt.plot(proxy_ages_interp,proxy_values_interp,'b-')
        plt.plot(age_centers,proxy_values_12ka,'g-')
        plt.xlim(21000,0)
        plt.show()
        """
        #
        # Set resolutions to a minimum of 1 and a maximum of max_res_value
        #TODO: Consider moving back to a higher max_res_value
        proxy_res_12ka[proxy_res_12ka < 1] = 1
        proxy_res_12ka[proxy_res_12ka > max_res_value] = max_res_value
        #
        # If the reconstruction type is "relative," remove the mean of the reference period
        if options['reconstruction_type'] == 'relative':
            # The reference period is calculated using annualized data in this case.
            ind_ref = np.where((proxy_ages_interp >= options['reference_period'][0]) & (proxy_ages_interp < options['reference_period'][1]))[0]  
            proxy_values_12ka = proxy_values_12ka - np.nanmean(proxy_values_interp[ind_ref])
            if np.isnan(proxy_values_interp[ind_ref]).all(): print('Proxy '+str(i)+': No data in reference period.'); no_ref_data += 1
        #
        # Get some metadata
        archive_type          = get_var(proxy_ts_selected[i],'archiveType')
        proxy_type            = get_var(proxy_ts_selected[i],'paleoData_proxy')
        interp_variable       = get_var(proxy_ts_selected[i],'interpretation1_variable')
        interp_direction      = get_var(proxy_ts_selected[i],'interpretation1_direction')
        proxy_seasonality_txt = get_var(proxy_ts_selected[i],'interpretation1_seasonality')
        #
        # Convert seasonality to a list of months, with negative values corresponding to the previous year.
        proxy_lat = proxy_ts_selected[i]['geo_meanLat']
        proxy_lon = proxy_ts_selected[i]['geo_meanLon']
        if proxy_lon < 0: proxy_lon = proxy_lon+360
        proxy_seasonality,proxy_seasonality_general = da_utils.interpret_seasonality(proxy_seasonality_txt,proxy_lat,unknown_option='annual')
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
        # In the interpretation direction is negative, flip the proxy data
        if interp_direction.lower() == "negative":
            print('Proxy '+str(i)+': Interpretation direction is negative. Flipping proxy.')
            proxy_values_12ka = -1 * proxy_values_12ka
        #
        # Save proxy data (y and ya)
        proxy_data['values_binned'][i,:]     = proxy_values_12ka
        proxy_data['resolution_binned'][i,:] = proxy_res_12ka
        proxy_data['uncertainty'].append(proxy_uncertainty)
        #
        # Save metadata
        proxy_data['archivetype'].append(archive_type)
        proxy_data['proxytype'].append(proxy_type)
        proxy_data['units'].append(data_units)
        proxy_data['interp'].append(interp_variable)
        proxy_data['direction'].append(interp_direction)
        proxy_data['psm'].append(psms_selected[i])
        proxy_data['seasonality_array'][i] = proxy_seasonality_array
        #
        # Save more metadata
        proxy_data['metadata'][i,0] = get_var(proxy_ts_selected[i],'dataSetName')
        proxy_data['metadata'][i,1] = get_var(proxy_ts_selected[i],'paleoData_TSid')
        proxy_data['metadata'][i,2] = str(proxy_lat)
        proxy_data['metadata'][i,3] = str(proxy_lon)
        proxy_data['metadata'][i,4] = str(proxy_seasonality_array)
        proxy_data['metadata'][i,5] = proxy_seasonality_general
        proxy_data['metadata'][i,6] = str(np.median(proxy_ages[1:]-proxy_ages[:-1]))
        proxy_data['metadata'][i,7] = collection_selected[i]
        proxy_data['metadata'][i,8] = data_units
        proxy_data['metadata'][i,9] = interp_variable
        proxy_data['metadata'][i,10] = interp_direction
        proxy_data['metadata'][i,11] = psms_selected[i]
        proxy_data['metadata'][i,12] = archive_type
        proxy_data['lats'][i] = proxy_lat
        proxy_data['lons'][i] = proxy_lon
        proxy_data['lons'][i] = proxy_lon
    #
    # Convert data types
    proxy_data['archivetype'] = np.array(proxy_data['archivetype'])
    proxy_data['proxytype']   = np.array(proxy_data['proxytype'])
    proxy_data['units']       = np.array(proxy_data['units'])
    proxy_data['uncertainty'] = np.array(proxy_data['uncertainty'])
    #
    # Save the ages for the reconstruction
    proxy_data['age_centers'] = age_centers
    #
    print('\n=== PROXY DATA LOADED ===')
    print('Proxy datasets loaded (n='+str(len(options['proxy_datasets_to_assimilate']))+'):'+str(options['proxy_datasets_to_assimilate']))
    print('Proxies loaded        (n='+str(len(proxy_ts_selected))+')')
    print('---')
    print('Proxies without data in reference period (n='+str(no_ref_data)+')')
    print('---')
    print('Data stored in dictionary "proxy_data", with keys and dimensions:')
    for key in list(proxy_data.keys()):
        try:    print('%20s %-15s' % (key,str(proxy_data[key].shape)))
        except: print('%20s %-15s' % (key,str(len(proxy_data[key]))))
    print('=========================\n')
    #
    return proxy_data
