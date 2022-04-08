#==============================================================================
# Different PSMs for use in the Holocene DA project.
#    author: Michael P. Erb
#    date  : 3/16/2022
#==============================================================================

import numpy as np

# Use PSMs to get model-based proxy estimates
def psm_main(model_data,proxy_data,options):
    #
    n_proxies = proxy_data['values_binned'].shape[0]
    proxy_estimates_all = np.array([dict() for k in range(n_proxies)])  # HXb
    for i in range(n_proxies):
        #
        # Set PSMs requirements
        psm_requirements = {}
        psm_requirements['get_tas'] = {'units':'degC'}
        #
        # Set the PSMs to use
        psm_types_to_use = ['get_tas']
        #
        # The code will use the first PSM  in the list above that meets the requirements
        psm_selected = None
        for psm_type in psm_types_to_use:
            psm_keys = list(psm_requirements[psm_type].keys())
            psm_check = np.full(len(psm_keys),False,dtype=bool)
            for counter,psm_key in enumerate(psm_keys):
                psm_check[counter] = (proxy_data[psm_key][i] == psm_requirements[psm_type][psm_key])
            #
            if psm_check.all() == True: psm_selected = psm_type; break
        #
        if psm_selected == None:
            print('WARNING: No PSM found. Using NaNs.')
            psm_selected = 'use_nans'
        #
        print('Proxy',i,'PSM selected:',psm_selected,'|',proxy_data['archivetype'][i],proxy_data['proxytype'][i],proxy_data['units'][i])
        #
        # Calculate the model-based proxy estimate depending on the PSM (or variable to compare, it the proxy is already calibrated)
        if   psm_selected == 'get_tas':  proxy_estimate = get_model_values(model_data,proxy_data,'tas',i)
        elif psm_selected == 'use_nans': proxy_estimate = use_nans(model_data)
        else:                            proxy_estimate = use_nans(model_data)
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
    return proxy_estimates_all,proxy_estimate


# A function to get the model values at the same location and seasonality as the proxy
def get_model_values(model_data,proxy_data,var_name,i,verbose=False):
    #
    tas_model    = model_data[var_name]
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
    tas_model_location = tas_model[:,:,j_selected,i_selected]
    #
    # Compute an average over months according to the proxy seasonality
    # Note: months are always taken from the current year, not from the previous year
    proxy_seasonality_indices = np.abs(proxy_season)-1
    proxy_seasonality_indices[proxy_seasonality_indices > 11] = proxy_seasonality_indices[proxy_seasonality_indices > 11] - 12
    tas_model_location_season = np.average(tas_model_location[:,proxy_seasonality_indices],weights=ndays_model[:,proxy_seasonality_indices],axis=1)
    #
    return tas_model_location_season


# A function to get the NaNs with the same length as the model data
def use_nans(model_data):
    #
    n_time = model_data['tas'].shape[0]
    nan_array = np.zeros((n_time)); nan_array[:] = np.nan
    #
    return nan_array
