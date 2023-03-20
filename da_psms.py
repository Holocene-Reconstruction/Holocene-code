#==============================================================================
# Different PSMs for use in the Holocene DA project.
#    author: Michael P. Erb
#    date  : 3/16/2022
#==============================================================================

import scipy
import numpy as np

# Use PSMs to get model-based proxy estimates
def psm_main(model_data,proxy_data,options):
    #
    n_proxies = proxy_data['values_binned'].shape[0]
    proxy_estimates_all = np.array([dict() for k in range(n_proxies)])  # HXb
    for i in range(n_proxies):
        #
        print(i)
        psm_selected = select_PSM(proxy_data,i)
        #model_data['LakeStatus'][:,:,j_selected,i_selected]
        # Calculate the model-based proxy estimate depending on the PSM (or variable to compare, it the proxy is already calibrated)
        # Model values are in units of degree C (for tas) and mm/day (for precip)
        if   psm_selected == 'get_tas':    proxy_estimate = get_model_values(model_data,proxy_data,'tas',i)
        elif psm_selected == 'get_precip': proxy_estimate = get_model_values(model_data,proxy_data,'precip',i)
        elif psm_selected == 'get_LakeStatus': 
            proxy_estimate = get_model_values(model_data,proxy_data,'LakeStatus',i)*1  ##DAMP12k- add lake option
            proxyvals = proxy_data['values_binned'][i]
            idx = np.where(np.isfinite(proxyvals))[0]
        #     if (np.sum(np.isfinite(proxy_estimate))>0) and (np.sum(np.isfinite(proxyvals))>0):
        #          if idx[0] >0:                       proxy_estimate[:idx[0]]      *= np.NaN
        #          if idx[-1] < len(proxy_estimate)-1: proxy_estimate[(idx[-1]+1):] *= np.NaN
        #          proxy_estimate = vals2percentile(proxy_estimate)
        #          if options['reconstruction_type'] == 'relative':
        #              ind_ref = (model_data['age'] >= options['reference_period'][0]) & (model_data['age'] < options['reference_period'][1])
        #              proxy_estimate  -= np.nanmean(proxy_estimate[ind_ref])
        # elif psm_selected == 'use_nans':   proxy_estimate = use_nans(model_data,options['vars_to_reconstruct'][0])
        else:                              proxy_estimate = use_nans(model_data,options['vars_to_reconstruct'][0])
        if sum(np.isfinite(proxy_estimate))==0:proxy_estimate = use_nans(model_data,options['vars_to_reconstruct'][0])
        # #Revisions to above must be duplicated in DA_load_proxies #TODO
        # # If the proxy units are mm/a, convert the model-based estimates from mm/day to mm/year
        if proxy_data['units'][i] == 'mm/a': proxy_estimate = proxy_estimate*365.25  #TODO: Is there a better way to account for leap years in these decadal means?
        #
        # Find all time resolutions in the record
        proxy_res_12ka_unique = np.unique(proxy_data['resolution_binned'][i,:])
        proxy_res_12ka_unique_sorted = np.sort(proxy_res_12ka_unique[np.isfinite(proxy_res_12ka_unique)]).astype(int)
        #
        # Loop through each time resolution, computing a running mean of the selected duration and save the values to a common variable
        # Note: While convolve may average across different models, those values won't be used (because of the model_data['valid_inds'] variable).
        #
        for res in proxy_res_12ka_unique_sorted:
            proxy_estimate_nyear_mean = np.convolve(proxy_estimate,np.ones((res,))/res,mode='same')
            proxy_estimates_all[i][res] = proxy_estimate_nyear_mean
    #
    print('Finished preprocessing proxies and making model-based proxy estimates.')
    return proxy_estimates_all,proxy_estimate


# A function to get the model values at the same location and seasonality as the proxy
def get_model_values(model_data,proxy_data,var_name,i,verbose=False):
    #
    var_model    = model_data[var_name]
    lat_model    = model_data['lat']
    lon_model    = model_data['lon']
    if 'time_ndays' in model_data.keys(): ndays_model = model_data['time_ndays'] #DAMP12k- no monthly data in prio
    if 'season'     in model_data.keys(): season      = model_data['season'] #DAMP12k- no monthly data in prior
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
    if 'time_ndays' in model_data.keys():
        var_model_location_season = np.average(var_model_location[:,proxy_seasonality_indices],weights=ndays_model[:,proxy_seasonality_indices],axis=1) #DAMP12k- no monthly data in prior
    elif 'season' in model_data.keys():
        if len(proxy_seasonality_indices)>=6: proxy_seasonality_indices = 'ANN'
        else: proxy_seasonality_indices = ['DJF','MAM','JJA','SON'][np.argmin(abs([1,4,7,10]-np.median(proxy_seasonality_indices)))]
        proxy_seasonality_indices = [x[:3].upper() for x in season].index(proxy_seasonality_indices)
        var_model_location_season = var_model_location[:,proxy_seasonality_indices]
        #var_model_location_season = var_model_location*np.NaN
    #
    return var_model_location_season


# A function to get the NaNs with the same length as the model data
def use_nans(model_data,var_name):
    #
    n_time = model_data[var_name].shape[0]
    nan_array = np.zeros((n_time)); nan_array[:] = np.nan
    #
    return nan_array

#A function to slect which PSM to use
#DAMP12- Moved to seperate function so can be also be added into da_load_proxies to save info for future plotting
def select_PSM(proxy_data,i): 
    # Set PSMs requirements
    psm_requirements = {}
    psm_requirements['get_tas']    = {'units':['degC']} #DAMP12k- change to make multiple options easier
    psm_requirements['get_precip'] = {'units':['mm/a'],'interp':['P']} #DAMP12k- change to make multiple options easier
    psm_requirements['get_LakeStatus'] = {'archivetype':['Shoreline','LakeDeposits']} #DAMP12k- change to make multiple options easier
    #psm_requirements['get_p_e']    = {'units':'mm/a','interp':'P-E'}  #TODO: Update this.
    #
    # Set the PSMs to use
    psm_types_to_use = ['get_tas','get_precip','get_LakeStatus'] ##DAMP12k- add lake option
    #
    # The code will use the first PSM  in the list above that meets the requirements
    psm_selected = None
    for psm_type in psm_types_to_use:
        psm_keys = list(psm_requirements[psm_type].keys())
        psm_check = np.full(len(psm_keys),False,dtype=bool)
        for counter,psm_key in enumerate(psm_keys):
            psm_check[counter] = (proxy_data[psm_key][i] in psm_requirements[psm_type][psm_key])  #DAMP12k- change to make multiple options easier
        #
        if psm_check.all() == True: psm_selected = psm_type; break
    #
    if psm_selected == None:
        print('WARNING: No PSM found. Using NaNs.')
        psm_selected = 'use_nans'
    #
    print('Proxy',i,'PSM selected:',psm_selected,'|',proxy_data['archivetype'][i],proxy_data['proxytype'][i],proxy_data['interp'][i],proxy_data['units'][i])
    return(psm_selected)

def vals2percentile(invec):
    if np.sum(np.isfinite(invec)) > 0:
        ranks = scipy.stats.mstats.rankdata(np.ma.masked_invalid(invec))                                        #Additional step to mask nan values
        ranks[ranks == 0] = np.nan
        if np.sum(np.isfinite(ranks)) > 1:
            ranks-=1
            ranks/= (np.sum(np.isfinite(ranks))-1)
        else: ranks*=np.NaN
    else: ranks=invec
    return(ranks)
