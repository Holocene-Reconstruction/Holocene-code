#==============================================================================
# Different PSMs for use in the Holocene DA project.
#    author: Michael P. Erb
#    date  : 1/14/2025
#==============================================================================

import numpy as np
from scipy.stats import rankdata

# Use PSMs to get model-based proxy estimates
def psm_main(model_data,proxy_data,options):
    #
    n_proxies = proxy_data['values_binned'].shape[0]
    proxy_estimates_all = np.array([dict() for k in range(n_proxies)])  # HXb
    i = 210
    for i in range(n_proxies):
        #
        # Set PSMs requirements
        psm_requirements = {}
        psm_requirements['calibrated_tas']    = {'interp':['temperature'],'units':['degc']}
        psm_requirements['calibrated_precip'] = {'interp':['p','precipitation'],'units':['mm/day']}  # Other possible precip units: 'mm/a','mm/yr'
        psm_requirements['rank_based_tas']    = {'interp':['temperature']}
        #psm_requirements['get_p_e']           = {'units':'mm/a','interp':'P-E'}  #TODO: Update this.
        #
        # Set the PSMs to use
        #psms_to_use = ['calibrated_tas','calibrated_precip','rank_based_tas']
        #psms_to_use = ['calibrated_tas','calibrated_precip']
        psms_to_use = options['psms_to_use']
        psm_if_no_match = 'use_nans'
        #
        # The code will use the first PSM in the list above that meets the requirements
        psm_selected = None
        for psm_type in psms_to_use:
            psm_keys = list(psm_requirements[psm_type].keys())
            psm_check = np.full(len(psm_keys),False,dtype=bool)
            for counter,psm_key in enumerate(psm_keys):
                #psm_check[counter] = (proxy_data[psm_key][i].lower() == psm_requirements[psm_type][psm_key])
                psm_check[counter] = (proxy_data[psm_key][i].lower() in psm_requirements[psm_type][psm_key])
            #
            if psm_check.all() == True: psm_selected = psm_type; break
        #
        if psm_selected == None:
            #print('WARNING: No PSM found. Using PSM:',psm_if_no_match)
            psm_selected = psm_if_no_match
        #
        print('Proxy',i,'PSM selected:',psm_selected,'|',proxy_data['archivetype'][i],proxy_data['proxytype'][i],proxy_data['interp'][i],proxy_data['units'][i])
        #
        # Calculate the model-based proxy estimate depending on the PSM (or variable to compare, it the proxy is already calibrated)
        # Model values are in units of degree C (for tas) and mm/day (for precip)
        if   psm_selected == 'calibrated_tas':    proxy_estimate = get_model_values(model_data,proxy_data,'tas',i)
        elif psm_selected == 'calibrated_precip': proxy_estimate = get_model_values(model_data,proxy_data,'precip',i)
        elif psm_selected == 'rank_based_tas':
            proxy_estimate,proxy_update = rank_based(model_data,proxy_data,'tas',i,options)
            proxy_data['values_binned'][i,:] = proxy_update
            proxy_data['metadata'][i,8] = proxy_data['metadata'][i,8]+'_percentile'
            proxy_data['units'][i] = proxy_data['units'][i]+'_percentile'
        elif psm_selected == 'use_nans':       proxy_estimate = use_nans(model_data)
        else:                                  proxy_estimate = use_nans(model_data)
        #
        # If the proxy units are mm/a, convert the model-based estimates from mm/day to mm/year
        if proxy_data['units'][i] == 'mm/a': proxy_estimate = proxy_estimate*365.25  #TODO: Is there a better way to account for leap years in these decadal means?
        #
        # Find all time resolutions in the record
        proxy_res_12ka_unique = np.unique(proxy_data['resolution_binned'][i,:])
        proxy_res_12ka_unique_sorted = np.sort(proxy_res_12ka_unique[np.isfinite(proxy_res_12ka_unique)]).astype(int)
        #
        # Loop through each time resolution, computing a running mean of the selected duration and save the values to a common variable
        # Note: While convolve may average across different models, those values won't be used (because of the model_data['valid_inds'] variable).
        for res in proxy_res_12ka_unique_sorted:
            proxy_estimate_nyear_mean = np.convolve(proxy_estimate,np.ones((res,))/res,mode='same')
            proxy_estimates_all[i][res] = proxy_estimate_nyear_mean
    #
    print('Finished preprocessing proxies and making model-based proxy estimates.')
    return proxy_estimates_all,proxy_estimate,proxy_data


# A function to get the model values at the same location and seasonality as the proxy
def get_model_values(model_data,proxy_data,var_name,i,verbose=False):
    #
    var_model    = model_data[var_name]
    lat_model    = model_data['lat']
    lon_model    = model_data['lon']
    ndays_model  = model_data['time_ndays']
    proxy_lat    = proxy_data['lats'][i]
    proxy_lon    = proxy_data['lons'][i]
    proxy_season = proxy_data['seasonality_array'][i]
    #
    # Find the model gridpoint closest to the proxy location
    if proxy_lon < 0: proxy_lon = proxy_lon+360
    lon_model_wrapped = np.append(lon_model,lon_model[0]+360)
    j_selected = np.argmin(np.abs(lat_model-proxy_lat))
    i_selected = np.argmin(np.abs(lon_model_wrapped-proxy_lon))
    if np.abs(proxy_lat-lat_model[j_selected])         > 2: print('WARNING: Too large of a lat difference. Proxy lat: '+str(proxy_lat)+', model lat: '+str(lat_model[j_selected]))
    if np.abs(proxy_lon-lon_model_wrapped[i_selected]) > 2: print('WARNING: Too large of a lon difference. Proxy lon: '+str(proxy_lon)+', model lon: '+str(lon_model_wrapped[i_selected]))
    if i_selected == len(lon_model_wrapped)-1: i_selected = 0
    if verbose: print('Proxy location vs. nearest model gridpoint.  Lat: '+str(proxy_lat)+', '+str(lat_model[j_selected])+'.  Lon: '+str(proxy_lon)+', '+str(lon_model[i_selected]))
    var_model_location = var_model[:,:,j_selected,i_selected]
    #
    # Compute an average over months according to the proxy seasonality
    # Note: months are always taken from the current year, not from the previous year
    proxy_seasonality_indices = np.abs(proxy_season)-1
    proxy_seasonality_indices[proxy_seasonality_indices > 11] = proxy_seasonality_indices[proxy_seasonality_indices > 11] - 12
    var_model_location_season = np.average(var_model_location[:,proxy_seasonality_indices],weights=ndays_model[:,proxy_seasonality_indices],axis=1)
    #
    return var_model_location_season


# A function to do rank-based comparison with temperature data
#var_name,verbose = 'tas',False
def rank_based(model_data,proxy_data,var_name,i,options,verbose=False):
    #
    var_model    = model_data[var_name]
    lat_model    = model_data['lat']
    lon_model    = model_data['lon']
    ndays_model  = model_data['time_ndays']
    age_model    = model_data['age']
    proxy_lat    = proxy_data['lats'][i]
    proxy_lon    = proxy_data['lons'][i]
    proxy_ages   = proxy_data['age_centers']
    proxy_season = proxy_data['seasonality_array'][i]
    proxy_values = proxy_data['values_binned'][i]
    proxy_direction = proxy_data['direction'][i]
    #
    # Find the model gridpoint closest to the proxy location
    if proxy_lon < 0: proxy_lon = proxy_lon+360
    lon_model_wrapped = np.append(lon_model,lon_model[0]+360)
    j_selected = np.argmin(np.abs(lat_model-proxy_lat))
    i_selected = np.argmin(np.abs(lon_model_wrapped-proxy_lon))
    if np.abs(proxy_lat-lat_model[j_selected])         > 2: print('WARNING: Too large of a lat difference. Proxy lat: '+str(proxy_lat)+', model lat: '+str(lat_model[j_selected]))
    if np.abs(proxy_lon-lon_model_wrapped[i_selected]) > 2: print('WARNING: Too large of a lon difference. Proxy lon: '+str(proxy_lon)+', model lon: '+str(lon_model_wrapped[i_selected]))
    if i_selected == len(lon_model_wrapped)-1: i_selected = 0
    if verbose: print('Proxy location vs. nearest model gridpoint.  Lat: '+str(proxy_lat)+', '+str(lat_model[j_selected])+'.  Lon: '+str(proxy_lon)+', '+str(lon_model[i_selected]))
    var_model_location = var_model[:,:,j_selected,i_selected]
    #
    # Compute an average over months according to the proxy seasonality
    # Note: months are always taken from the current year, not from the previous year
    proxy_seasonality_indices = np.abs(proxy_season)-1
    proxy_seasonality_indices[proxy_seasonality_indices > 11] = proxy_seasonality_indices[proxy_seasonality_indices > 11] - 12
    var_model_location_season = np.average(var_model_location[:,proxy_seasonality_indices],weights=ndays_model[:,proxy_seasonality_indices],axis=1)
    if proxy_direction.lower() == 'negative': var_model_location_season = -1*var_model_location_season  # Note: If interpretation direction is not given, it is assumed to be positive.
    #
    #TODO: Conceptually, does this work if the proxy and prior resolutions are different? 
    #
    # Get the model values corresponding to the valid data of the proxy
    ind_proxy_valid = np.isfinite(proxy_values)
    proxy_values_valid = proxy_values[ind_proxy_valid]
    proxy_ages_valid   = proxy_ages[ind_proxy_valid]
    proxy_ages_valid_start = proxy_ages_valid[0]  - (options['time_resolution']/2)
    proxy_ages_valid_end   = proxy_ages_valid[-1] + (options['time_resolution']/2)
    ind_model_selected = np.where((age_model >= proxy_ages_valid_start) & (age_model <= proxy_ages_valid_end))[0]
    #var_model_location_season_selected = var_model_location_season[ind_model_selected]
    #
    # Calculate ranks for the proxy and prior data
    ranks_proxy_values = rankdata(proxy_values_valid) - 1
    ranks_prior_values = rankdata(var_model_location_season) - 1
    percentile_proxy_values = (ranks_proxy_values / max(ranks_proxy_values)) * 100
    percentile_prior_values = ((ranks_prior_values-min(ranks_prior_values[ind_model_selected])) / (max(ranks_prior_values[ind_model_selected])-min(ranks_prior_values[ind_model_selected]))) * 100
    #
    # Put the proxy data into the same length array as it was originally
    percentile_proxy_values_all = np.zeros((len(proxy_values))); percentile_proxy_values_all[:] = np.nan
    percentile_proxy_values_all[ind_proxy_valid] = percentile_proxy_values
    #
    # Remove the reference period from the proxy and prior values  #TODO: Check that this is a good solution.
    ind_ref_proxy = np.where((proxy_ages >= options['reference_period'][0]) & (proxy_ages < options['reference_period'][1]))[0]
    ind_ref_prior = np.where((age_model  >= options['reference_period'][0]) & (age_model  < options['reference_period'][1]))[0]
    percentile_proxy_values_all = percentile_proxy_values_all - np.nanmean(percentile_proxy_values_all[ind_ref_proxy])
    percentile_prior_values     = percentile_prior_values     - np.mean(percentile_prior_values[ind_ref_prior])
    if np.isnan(percentile_proxy_values_all[ind_ref_proxy]).all(): print('No data in reference period, index: '+str(i))
    #
    """
    import matplotlib.pyplot as plt
    plt.plot(proxy_ages,proxy_values,'k-')
    plt.plot(age_model,var_model_location_season,'b-')
    plt.show()
    #
    plt.plot(proxy_ages,percentile_proxy_values_all,'k-')
    plt.plot(age_model,percentile_prior_values,'b-')
    plt.show()
    #
    plt.scatter(proxy_values_valid,ranks_proxy_values)
    """
    #
    return percentile_prior_values,percentile_proxy_values_all


# A function to get the NaNs with the same length as the model data
def use_nans(model_data):
    #
    n_time = model_data['tas'].shape[0]
    nan_array = np.zeros((n_time)); nan_array[:] = np.nan
    #
    return nan_array
