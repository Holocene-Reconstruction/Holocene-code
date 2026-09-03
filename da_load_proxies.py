#==============================================================================
# Functions for loading proxy data for the data assimilation project.
#    author: Michael P. Erb
#==============================================================================

import da_utils
import numpy as np
import lipd
from scipy import interpolate
import rdata
import geopy.distance

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
        # Load the EcoClimate proxy metadata
        print('Loading proxy dataset: '+proxy_dataset)
        try: proxy_ts_ecoclimate = rdata.read_rds(options['data_dir']+'proxies/ecoclimate/'+proxy_dataset+'.rds')
        except: print('ERROR: invalid proxy dataset: '+proxy_dataset)
        if options['reconstruction_type'] == 'absolute': proxy_ts_ecoclimate = lipd.filterTs(proxy_ts_ecoclimate,'paleoData_datum == abs')
        #
        # Add these proxies to the full proxy dataset
        proxy_ts = proxy_ts + proxy_ts_ecoclimate
        proxy_dataset_short = proxy_dataset.split('_')[0]
        collection_all = collection_all + ([proxy_dataset_short] * len(proxy_ts_ecoclimate))
    #
    return proxy_ts,collection_all


# Filter the proxy data
def filter_proxies_and_set_psms(proxy_ts,collection_all,options):
    #
    # Set up variables
    print('Filtering proxies and assigning PSMs')
    n_proxies = len(proxy_ts)
    print('Proxies, all loaded:',n_proxies)
    logical_selected_all = np.array([False]*n_proxies)
    proxy_psms = ["None"]*n_proxies
    #
    # Get some metadata fields from proxies
    proxy_variable = []
    proxy_interp   = []
    proxy_archive  = []
    proxy_proxy    = []
    proxy_unit     = []
    for i in range(n_proxies):
        try:    proxy_interp.append(proxy_ts[i]['interpretation1_variable'][0])
        except: proxy_interp.append("Not given")
        try:    proxy_variable.append(proxy_ts[i]['paleoData_variableName'][0])
        except: proxy_variable.append("Not given")
        try:    proxy_archive.append(proxy_ts[i]['archiveType'][0])
        except: proxy_archive.append("Not given")
        try:    proxy_proxy.append(proxy_ts[i]['paleoData_proxy'][0])
        except: proxy_proxy.append("Not given")
        try:    proxy_unit.append(proxy_ts[i]['paleoData_units'][0])
        except: proxy_unit.append("Not given")
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
        logical_interp   = np.array([value in chosen_interp   for value in proxy_interp])
        logical_variable = np.array([value in chosen_variable for value in proxy_variable])
        logical_archive  = np.array([value in chosen_archive  for value in proxy_archive])
        logical_proxy    = np.array([value in chosen_proxy    for value in proxy_proxy])
        logical_unit     = np.array([value in chosen_unit     for value in proxy_unit])
        if chosen_interp[0]   == 'any': logical_interp[:]   = True
        if chosen_variable[0] == 'any': logical_variable[:] = True
        if chosen_archive[0]  == 'any': logical_archive[:]  = True
        if chosen_proxy[0]    == 'any': logical_proxy[:]    = True
        if chosen_unit[0]     == 'any': logical_unit[:]     = True
        logical_selected = logical_interp & logical_variable & logical_archive & logical_proxy & logical_unit
        ind_selected = np.where(logical_selected)[0]
        print('Proxies in group ('+combination_selected+'):',len(ind_selected))
        #
        # Set the PSM
        for index in ind_selected:
            proxy_psms[index] = chosen_psm
        #
        # Combine each logical combination
        logical_selected_all = logical_selected_all | logical_selected
    #
    ind_selected_all = np.where(logical_selected_all)[0]
    print('Proxies in all groups:',len(ind_selected_all))
    #
    # Get proxy data to keep
    proxy_ts_selected   = [proxy_ts[index]       for index in ind_selected_all]
    psms_selected       = [proxy_psms[index]     for index in ind_selected_all]
    collection_selected = [collection_all[index] for index in ind_selected_all]
    #
    return proxy_ts_selected,psms_selected,collection_selected


# Process the proxy data
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
    proxy_data['metadata']          = np.zeros((n_proxies,11),dtype=object); proxy_data['metadata'][:]          = np.nan
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
    i = 0
    for i in range(n_proxies):
        #
        # Get proxy data
        print('Processing proxies:',i)
        proxy_values = np.array(proxy_ts_selected[i]['paleoData_values']).astype(float)
        proxy_ages = proxy_ts_selected[i]['age'].astype(float)
        if np.ma.isMaskedArray(proxy_ages): proxy_ages = proxy_ages.filled(np.nan)
        else:                               proxy_ages = np.array(proxy_ages)
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
        # Get uncertainty metadata
        missing_uncertainty_value = np.nan
        try:    proxy_uncertainty = proxy_ts_selected[i]['paleoData_temperature12kUncertainty'][0]
        except: proxy_uncertainty = missing_uncertainty_value; missing_uncertainty += 1
        if proxy_uncertainty == 'NA': proxy_uncertainty = missing_uncertainty_value; missing_uncertainty += 1
        proxy_uncertainty = np.square(float(proxy_uncertainty))  # Proxy uncertainty was give as RMSE, but the code uses MSE
        #
        # Update the units  #TODO: Work on this (and consider using a different value than 365.25)
        #  - Precipitation: mm/yr -> mm/day
        try:    data_units = proxy_ts_selected[i]['paleoData_units'][0]
        except: data_units = 'Not given'
        if data_units == 'mm/yr':
            proxy_values      = proxy_values/365.25
            proxy_uncertainty = proxy_uncertainty/365.25
            data_units = 'mm/day'
            print('Proxy '+str(i)+': Updating units from mm/yr to mm/day')
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
        #plt.xlim(21000,0)
        plt.show()
        """
        #
        # Set resolutions to a minimum of 1 and a maximum of max_res_value
        proxy_res_12ka[proxy_res_12ka < 1] = 1
        proxy_res_12ka[proxy_res_12ka > max_res_value] = max_res_value
        #
        # If the reconstruction type is "relative," remove the mean of the reference period
        if options['reconstruction_type'] == 'relative':
            # The reference period is calculated using annualized data in this case.
            ind_ref = np.where((proxy_ages_interp >= options['reference_period'][0]) & (proxy_ages_interp < options['reference_period'][1]))[0]  
            proxy_values_12ka = proxy_values_12ka - np.nanmean(proxy_values_interp[ind_ref])
            if np.isnan(proxy_values_interp[ind_ref]).all(): print('No data in reference period, index: '+str(i)); no_ref_data += 1
        #
        # Convert seasonality to a list of months, with negative values corresponding to the previous year.
        proxy_lat = proxy_ts_selected[i]['geo_latitude'][0]
        proxy_lon = proxy_ts_selected[i]['geo_longitude'][0]
        try:    proxy_seasonality_txt = proxy_ts_selected[i]['interpretation1_seasonality'][0]
        except: proxy_seasonality_txt = 'Not given'
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
        try:    interp_direction = proxy_ts_selected[i]['interpretation1_direction'][0]
        except: interp_direction = 'Not given'
        if interp_direction.lower() == "negative":
            print("Interpretation direction is negative. Flipping proxy")
            proxy_values_12ka = -1 * proxy_values_12ka
        #
        # Save proxy data (y and ya)
        proxy_data['values_binned'][i,:]     = proxy_values_12ka
        proxy_data['resolution_binned'][i,:] = proxy_res_12ka
        proxy_data['uncertainty'].append(proxy_uncertainty)
        #
        # Save proxy metdata
        try:    proxy_type = proxy_ts_selected[i]['paleoData_proxy'][0]
        except: proxy_type = 'Not given'
        proxy_data['archivetype'].append(proxy_ts_selected[i]['archiveType'][0])
        proxy_data['proxytype'].append(proxy_type)
        proxy_data['units'].append(data_units)
        proxy_data['interp'].append(proxy_ts_selected[i]['interpretation1_variable'][0])
        proxy_data['direction'].append(interp_direction)
        proxy_data['psm'].append(psms_selected[i])
        proxy_data['seasonality_array'][i] = proxy_seasonality_array
        #
        # Save more metadata
        if proxy_lon < 0: proxy_lon = proxy_lon+360
        proxy_data['metadata'][i,0] = proxy_ts_selected[i]['dataSetName'][0]
        proxy_data['metadata'][i,1] = proxy_ts_selected[i]['paleoData_TSid'][0]
        proxy_data['metadata'][i,2] = str(proxy_lat)
        proxy_data['metadata'][i,3] = str(proxy_lon)
        proxy_data['metadata'][i,4] = str(proxy_seasonality_array)
        proxy_data['metadata'][i,5] = proxy_seasonality_general
        proxy_data['metadata'][i,6] = str(np.median(proxy_ages[1:]-proxy_ages[:-1]))
        proxy_data['metadata'][i,7] = collection_selected[i]
        proxy_data['metadata'][i,8] = data_units
        proxy_data['metadata'][i,9] = proxy_ts_selected[i]['interpretation1_variable'][0]
        proxy_data['metadata'][i,10] = interp_direction
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
    print('Proxies without uncertainty value        (n='+str(missing_uncertainty)+'). Set to '+str(missing_uncertainty_value))
    print('---')
    print('Data stored in dictionary "proxy_data", with keys and dimensions:')
    for key in list(proxy_data.keys()):
        try:    print('%20s %-15s' % (key,str(proxy_data[key].shape)))
        except: print('%20s %-15s' % (key,str(len(proxy_data[key]))))
    print('=========================\n')
    #
    return proxy_data


# Average records within 1 km of each other
def average_nearby_records(proxy_data,options):
    #print('NOTE: Proxy averaging not yet implmented')
    #return proxy_data
    #
    # Settings
    distance_threshhold = 1  # Distance threshhold in km
    #
    n_proxies = proxy_data['values_binned'].shape[0]
    n_ages    = proxy_data['values_binned'].shape[1]
    #
    proxy_data.keys()
    # Compute distances
    distances_km = np.zeros((n_proxies,n_proxies)); distances_km[:] = np.nan
    i=0; j=1
    for i in range(n_proxies):
        proxy_lat_1 = proxy_data['lats'][i]
        proxy_lon_1 = proxy_data['lons'][i]
        distances_km[i,i] = 0
        for j in range(i+1,n_proxies):
            proxy_lat_2 = proxy_data['lats'][j]
            proxy_lon_2 = proxy_data['lons'][j]
            #
            # Calculate distance
            distances_km[i,j] = geopy.distance.great_circle((proxy_lat_1,proxy_lon_1),(proxy_lat_2,proxy_lon_2)).km
    #
    # Find groups of nearby_proxies
    groups = {}
    group_key = 0
    for i in range(n_proxies):
        # Find other proxies close to the selected proxy
        distances_to_selected_proxy = distances_km[i,:]
        ind_within_distance = np.where((distances_to_selected_proxy <= distance_threshhold))[0]
        # If at least one record is found, create a group
        if len(ind_within_distance) > 0:
            groups[group_key] = ind_within_distance
            group_key += 1
        # Set selected distances to nan, so selected proxies don't appear in any more groups
        distances_km[:,ind_within_distance] = np.nan
    #
    # Set up new variables
    n_groups = group_key
    proxy_data_new = {}
    proxy_data_new['values_binned']     = np.zeros((n_groups,n_ages));          proxy_data_new['values_binned'][:]     = np.nan
    proxy_data_new['resolution_binned'] = np.zeros((n_groups,n_ages));          proxy_data_new['resolution_binned'][:] = np.nan
    proxy_data_new['metadata']          = np.zeros((n_groups,11),dtype=object); proxy_data_new['metadata'][:]          = np.nan
    proxy_data_new['lats']              = np.zeros((n_groups));                 proxy_data_new['lats'][:]              = np.nan
    proxy_data_new['lons']              = np.zeros((n_groups));                 proxy_data_new['lons'][:]              = np.nan
    proxy_data_new['uncertainty']       = np.zeros((n_groups));                 proxy_data_new['uncertainty'][:]       = np.nan
    proxy_data_new['archivetype']       = []
    proxy_data_new['proxytype']         = []
    proxy_data_new['units']             = []
    proxy_data_new['interp']            = []
    proxy_data_new['direction']         = []
    proxy_data_new['seasonality_array'] = {}
    proxy_data_new['psm']               = []
    #
    # For each group, compute the mean
    i = 0
    for i in range(n_groups):
        ind_to_mean = groups[i]
        # Compute mean of some fields
        proxy_data_new['values_binned'][i,:]     = np.nanmean(proxy_data['values_binned'][ind_to_mean,:],axis=0)
        proxy_data_new['resolution_binned'][i,:] = np.nanmean(proxy_data['resolution_binned'][ind_to_mean,:],axis=0)
        proxy_data_new['lats'][i]                = np.nanmean(proxy_data['lats'][ind_to_mean],axis=0)
        proxy_data_new['lons'][i]                = np.nanmean(proxy_data['lons'][ind_to_mean],axis=0)
        proxy_data_new['uncertainty'][i]         = np.nanmean(proxy_data['uncertainty'][ind_to_mean],axis=0)
        # Concatenate some fields
        proxy_data_new['archivetype'].append("_".join(proxy_data['archivetype'][ind_to_mean]))
        proxy_data_new['proxytype'].append("_".join(proxy_data['proxytype'][ind_to_mean]))
        proxy_data_new['units'].append("_".join(proxy_data['units'][ind_to_mean]))
        proxy_data_new['interp'].append("_".join(np.array(proxy_data['interp'])[ind_to_mean]))
        proxy_data_new['direction'].append("_".join(np.array(proxy_data['direction'])[ind_to_mean]))
        # Use the first value for PSM
        proxy_data_new['psm'].append(proxy_data['psm'][ind_to_mean[0]])
        # Use a different approach for some fields
        if len(ind_to_mean) == 0:
            proxy_data_new['metadata'][i,:]        = proxy_data['metadata'][ind_to_mean[0],:]
            proxy_data_new['seasonality_array'][i] = proxy_data['seasonality_array'][ind_to_mean[0]]
            proxy_data_new['metadata'][i,2]        = proxy_data['metadata'][ind_to_mean[0],2]
            proxy_data_new['metadata'][i,3]        = proxy_data['metadata'][ind_to_mean[0],3]
            proxy_data_new['metadata'][i,5]        = proxy_data['metadata'][ind_to_mean[0],5]
        else:
            proxy_data_new['seasonality_array'][i] = np.arange(1,13)
            proxy_data_new['metadata'][i,2]        = np.nanmean(proxy_data['metadata'][ind_to_mean,2].astype(float),axis=0)
            proxy_data_new['metadata'][i,3]        = np.nanmean(proxy_data['metadata'][ind_to_mean,3].astype(float),axis=0)
            proxy_data_new['metadata'][i,5]        = "annual"
    #
    # Copy the age_centers values
    proxy_data_new['age_centers'] = proxy_data['age_centers']
    #
    return proxy_data_new
            

"""
# A wrapper for average_nearby_records function, so that only records with the same PSM are averaged
def average_by_psm(proxy_data,options):
    #
    #proxy_data.keys()
    # Treat each PSM seperately
    psms_all = np.unique(proxy_data['psm'])
    psm_selected = psms_all[0]
    for psm_selected in psms_all:
        #
        ind_selected = np.where(np.array(proxy_data['psm']) == psm_selected)[0]
"""
