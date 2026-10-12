#==============================================================================
# Different PSMs for use in the Holocene DA project.
#    author: Michael P. Erb
#==============================================================================

import numpy as np
from scipy.stats import rankdata
import xarray as xr
import da_utils_lmr
import copy
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
plt.style.use('ggplot')

# Use PSMs to get model-based proxy estimates
def psm_main(model_data,proxy_data,options):
    #
    # If SSTs are needed, load the SST model data
    if "calibrated_sst" in proxy_data['psm']:
        #
        model_filename = "P:/data_paleoclimate/data_assimilation/models/processed_model_data/trace2.21999-0BP.sst.timeres_100.nc"
        handle_model_sst = xr.open_dataset(model_filename,decode_times=False)
        model_data['sst']       = handle_model_sst['sst'].values
        model_data['lat_sst']   = handle_model_sst['lat'].values
        model_data['lon_sst']   = handle_model_sst['lon'].values
        model_data['ndays_sst'] = handle_model_sst['days_per_month_all'].values
        model_data['age_sst']   = handle_model_sst['age'].values
        handle_model_sst.close()
        #
        if (model_data['age_sst'] == model_data['age']).all() == False:
            print('WARNING: SST ages are different from other model ages. This will cause the code to crash.')
    #
    # Loop through proxies, using PSMs
    n_proxies = proxy_data['values_binned'].shape[0]
    proxy_estimates_all = np.array([dict() for k in range(n_proxies)])  # HXb
    i = 341
    for i in range(n_proxies):
        #
        psm_selected = proxy_data['psm'][i]
        proxy_description = 'Proxy '+str(i)+', PSM selected: '+str(psm_selected)+' | '+str(proxy_data['archivetype'][i])+', '+str(proxy_data['proxytype'][i])+', '+str(proxy_data['interp'][i])+', '+str(proxy_data['units'][i])
        print(proxy_description)
        #
        psm_name = psm_selected[:10]
        var_name = psm_selected[11:]
        proxy_orig = copy.deepcopy(proxy_data['values_binned'][i,:])
        #
        # Calculate the model-based proxy estimate depending on the PSM (or variable to compare, if the proxy is already calibrated)
        # Model values are in units of degree C (for tas) and mm/day (for precip)
        if psm_selected == "calibrated_sst": proxy_estimate = get_model_values_nearest_sst(model_data,proxy_data,var_name,i,options)
        elif psm_name == 'calibrated': proxy_estimate = get_model_values_bilinear(model_data,proxy_data,var_name,i)
        elif psm_name == 'rank_based':
            #proxy_estimate,proxy_update = rank_based(model_data,proxy_data,var_name,i,options)
            proxy_estimate,proxy_update = rank_based_older(model_data,proxy_data,var_name,i,options)
            proxy_data['values_binned'][i,:] = proxy_update
            units_new = proxy_data['units'][i]+'_percentile'
            proxy_data['units'][i]      = units_new
            proxy_data['metadata'][i,8] = units_new
        elif psm_selected == 'use_nans': proxy_estimate = use_nans(model_data,var_name)
        else:                            proxy_estimate = use_nans(model_data,var_name)
        #
        # If the proxy units are mm/a, convert the model-based estimates from mm/day to mm/year
        #TODO: Work more on converting units
        if proxy_data['units'][i] == 'mm/a':
            print("Converting units from mm/a to mm/day")
            proxy_estimate = proxy_estimate*365.25
        #
        # Find all time resolutions in the record
        proxy_res_12ka_unique = np.unique(proxy_data['resolution_binned'][i,:])
        proxy_res_12ka_unique_sorted = np.sort(proxy_res_12ka_unique[np.isfinite(proxy_res_12ka_unique)]).astype(int)
        #
        # Allow for proxy estimate and estimate to have same reference window if data does not cover entire window (allows for more flexibility with larger windows)
        if options['reconstruction_type'] == 'relative':
            age_model = model_data['age']
            proxy_ages_valid = proxy_data['age_centers'][np.isfinite(proxy_data['values_binned'][i,:])]
            if len(proxy_ages_valid) > 0:
                ind_model_proxy     = ((age_model >= proxy_ages_valid[0] - (options['time_resolution']/2)) & (age_model <= proxy_ages_valid[-1] + (options['time_resolution']/2)))
                ind_model_refwindow = ((age_model >= options['reference_period'][0]) & (age_model  < options['reference_period'][1]))
                proxy_estimate = proxy_estimate - np.mean(proxy_estimate[ind_model_proxy & ind_model_refwindow])
        #
        if options['plot_proxies']:

            # Specify subplots
            plt.figure(figsize=(16,3))
            region_to_plot = [-160,-40,0,80]; region_center = (region_to_plot[0]+region_to_plot[1])/2
            ax1 = plt.subplot2grid((1,3),(0,0),projection=ccrs.LambertConformal(central_longitude=region_center)); ax1.set_extent(region_to_plot,ccrs.PlateCarree())
            ax2 = plt.subplot2grid((1,3),(0,1),colspan=2)
            #
            # Make a map of the proxy location
            ax1.scatter(proxy_data['lons'][i],proxy_data['lats'][i],100,marker='o',facecolor='y',edgecolor='k',linewidth=3,alpha=1,zorder=1,transform=ccrs.PlateCarree())
            ax1.set_title('Proxy location',fontsize=9)
            ax1.coastlines(zorder=2)
            ax1.add_feature(cfeature.LAKES,facecolor='none',edgecolor='k')
            ax1.gridlines(zorder=3,color='k',linewidth=1,linestyle=(0,(1,5)))

            ax2.plot(proxy_data['age_centers'],proxy_orig,c='tab:blue',label='Proxy, original')
            if psm_name == 'rank_based': ax2.plot(proxy_data['age_centers'],proxy_update,c='tab:green',label='Proxy, updated')
            ax2.plot(model_data['age'],proxy_estimate,c='tab:red',label='Model at proxy location')
            ax2.legend(fontsize=7)  # loc="upper right"
            ax2.set_title(proxy_description+'\nLat = '+str("{:.2f}".format(proxy_data['lats'][i]))+'$^\circ$N, lon='+str("{:.2f}".format(proxy_data['lons'][i]))+'$^\circ$E, uncertainty='+str(proxy_data['uncertainty'][i]),fontsize=9)
            ax2.axvspan(options['reference_period'][0],options['reference_period'][1],alpha=.25)
            ax2.set_xlim(options["age_range_to_reconstruct"][1],options["age_range_to_reconstruct"][0])
            ax2.set_xlabel("Age (yr BP)")
            ax2.set_ylabel(proxy_data['units'][i])
            #plt.show()
            plt.savefig('figures/psm_'+str(psm_selected)+'_proxy_'+str(i).zfill(3)+'.png',dpi=200,format='png',bbox_inches='tight')
            plt.close()
        #
        # Loop through each time resolution, computing a running mean of the selected duration and save the values to a common variable
        # Note: While convolve may average across different models, those values won't be used (because of the model_data['valid_inds'] variable).
        for res in proxy_res_12ka_unique_sorted:
            if res == 1:
                proxy_estimates_all[i][res] = proxy_estimate
            else:
                proxy_estimate_nyear_mean = np.convolve(proxy_estimate,np.ones((res,))/res,mode='same')
                proxy_estimates_all[i][res] = proxy_estimate_nyear_mean
    #
    print('Finished preprocessing proxies and making model-based proxy estimates.')
    return proxy_estimates_all,proxy_estimate,proxy_data


# A function to get the model values at the same location and seasonality as the proxy, using bilinear interpolation
#var_name = 'tas'; verbose = False
def get_model_values_bilinear(model_data,proxy_data,var_name,i,verbose=False):
    #
    proxy_lat    = proxy_data['lats'][i]
    proxy_lon    = proxy_data['lons'][i]
    proxy_season = proxy_data['seasonality_array'][i]
    ndays_model  = model_data['time_ndays']
    #proxy_direction = proxy_data['direction'][i]
    #
    if (proxy_lat > np.max(model_data['lat'])) | (proxy_lat < np.min(model_data['lat'])): print("WARNING: Proxy lat outside of model bounds:",proxy_lat)
    if (proxy_lon > np.max(model_data['lon'])) | (proxy_lon < np.min(model_data['lon'])): print("WARNING: Proxy lon outside of model bounds:",proxy_lon)
    #
    # Put the model data in a dataframe
    model_data_df = xr.Dataset(
        {
            "var":(["age","month","lat","lon"],model_data[var_name]),
        },
        coords = {
            "age":   (["age"],  model_data['age'],{"units":"yr BP (ref 1950)"}),
            'month': (["month"],np.arange(1,13),  {"units":"month_number"}),
            "lat":   (["lat"],  model_data['lat'],{"units":"degrees_north"}),
            "lon":   (["lon"],  model_data['lon'],{"units":"degrees_east"}),
        }
    )
    #
    # Use bilinear interpolation to get proxy values at the proxy location
    model_data_df_location = model_data_df.interp(lat=proxy_lat,lon=proxy_lon,method="linear")
    #model_data_df_location2 = model_data_df.interp(lat=proxy_lat,lon=proxy_lon,method="nearest")
    var_model_location = model_data_df_location["var"].values
    #
    """
    plt.contourf(model_data['lon'],model_data['lat'],model_data_df["var"][0,0,:,:])
    plt.scatter(proxy_lon,proxy_lat)
    """
    #
    # Compute an average over months according to the proxy seasonality
    # Note: months are always taken from the current year, not from the previous year
    proxy_seasonality_indices = np.abs(proxy_season)-1
    proxy_seasonality_indices[proxy_seasonality_indices > 11] = proxy_seasonality_indices[proxy_seasonality_indices > 11] - 12
    var_model_location_season = np.average(var_model_location[:,proxy_seasonality_indices],weights=ndays_model[:,proxy_seasonality_indices],axis=1)
    #if proxy_direction.lower() == 'negative': var_model_location_season = -1*var_model_location_season  # Note: If interpretation direction is not given, it is assumed to be positive.
    #
    return var_model_location_season


# A function to get the model values at the same location and seasonality as the proxy
def get_model_values_nearest(model_data,proxy_data,var_name,i,verbose=False):
    #
    var_model    = model_data[var_name]
    lat_model    = model_data['lat']
    lon_model    = model_data['lon']
    ndays_model  = model_data['time_ndays']
    proxy_lat    = proxy_data['lats'][i]
    proxy_lon    = proxy_data['lons'][i]
    proxy_season = proxy_data['seasonality_array'][i]
    #proxy_direction = proxy_data['direction'][i]
    #
    # Find the model gridpoint closest to the proxy location
    if proxy_lon < 0: proxy_lon = proxy_lon+360
    lon_model_wrapped = np.append(lon_model,lon_model[0]+360)
    j_selected = np.argmin(np.abs(lat_model-proxy_lat))
    i_selected = np.argmin(np.abs(lon_model_wrapped-proxy_lon))
    if np.abs(proxy_lat-lat_model[j_selected])         > 2: print('WARNING: Too large of a lat difference. Proxy lat: '+str(proxy_lat)+', model lat: '+str(lat_model[j_selected]))
    if np.abs(proxy_lon-lon_model_wrapped[i_selected]) > 2: print('WARNING: Too large of a lon difference. Proxy lon: '+str(proxy_lon)+', model lon: '+str(lon_model_wrapped[i_selected]))
    if i_selected == len(lon_model_wrapped)-1: i_selected = 0
    if verbose: print('Proxy location vs. nearest model gridpoint. Lat: '+str(proxy_lat)+', '+str(lat_model[j_selected])+'. Lon: '+str(proxy_lon)+', '+str(lon_model[i_selected]))
    var_model_location = var_model[:,:,j_selected,i_selected]
    #
    # Compute an average over months according to the proxy seasonality
    # Note: months are always taken from the current year, not from the previous year
    proxy_seasonality_indices = np.abs(proxy_season)-1
    proxy_seasonality_indices[proxy_seasonality_indices > 11] = proxy_seasonality_indices[proxy_seasonality_indices > 11] - 12
    var_model_location_season = np.average(var_model_location[:,proxy_seasonality_indices],weights=ndays_model[:,proxy_seasonality_indices],axis=1)
    #if proxy_direction.lower() == 'negative': var_model_location_season = -1*var_model_location_season  # Note: If interpretation direction is not given, it is assumed to be positive.
    #
    return var_model_location_season


# A function to get the model values at the same location and seasonality as the proxy
def get_model_values_nearest_sst(model_data,proxy_data,var_name,i,options,verbose=False):
    #
    #print("Using the SST psm")
    #
    # Load selected variables
    proxy_lat    = proxy_data['lats'][i]
    proxy_lon    = proxy_data['lons'][i]
    proxy_season = proxy_data['seasonality_array'][i]
    #
    # Find the model gridpoint closest to the proxy location
    if proxy_lon < 0: proxy_lon = proxy_lon+360
    #
    # CALCULATE DISTANCES
    # Calculate mean SST, for identiying nans
    sst_mean = np.mean(np.mean(model_data['sst'],axis=1),axis=0)
    #plt.scatter(model_data['lon_sst'],model_data['lat_sst'],c=sst_mean)
    #
    # For each proxy location, calculate distances to each model location
    n_lat = model_data['lat_sst'].shape[0]
    n_lon = model_data['lat_sst'].shape[1]
    distances = np.zeros((n_lat,n_lon)); distances[:] = np.nan
    for i in range(n_lon):
        #print('Calculating distance to each gridcell:',i,'/',n_lon)
        for j in range(n_lat):
            if np.isfinite(sst_mean[j,i]):
                distances[j,i] = da_utils_lmr.haversine(proxy_lon,proxy_lat,model_data['lon_sst'][j,i],model_data['lat_sst'][j,i])
            else:
                distances[j,i] = 10000
    #
    # Get indices of minimum value
    ind_min = np.unravel_index(distances.argmin(), distances.shape)
    """
    plt.scatter(model_data['lon_sst'],model_data['lat_sst'],c=distances)
    plt.scatter(model_data['lon_sst'][ind_min],model_data['lat_sst'][ind_min],200,'y')
    plt.scatter(proxy_lon,proxy_lat,100,'k')
    """
    #
    if np.abs(proxy_lat-model_data['lat_sst'][ind_min]) > 2: print('WARNING: Too large of a lat difference. Proxy lat: '+str(proxy_lat)+', model lat: '+str(model_data['lat_sst'][ind_min]))
    if np.abs(proxy_lon-model_data['lon_sst'][ind_min]) > 2: print('WARNING: Too large of a lon difference. Proxy lon: '+str(proxy_lon)+', model lon: '+str(model_data['lon_sst'][ind_min]))
    if verbose: print('Proxy location vs. nearest model gridpoint. Lat: '+str(proxy_lat)+', '+str(model_data['lat_sst'][ind_min])+'. Lon: '+str(proxy_lon)+', '+str(model_data['lon_sst'][ind_min]))
    var_model_location = model_data['sst'][:,:,ind_min[0],ind_min[1]]
    #
    # Compute an average over months according to the proxy seasonality
    # Note: months are always taken from the current year, not from the previous year
    proxy_seasonality_indices = np.abs(proxy_season)-1
    proxy_seasonality_indices[proxy_seasonality_indices > 11] = proxy_seasonality_indices[proxy_seasonality_indices > 11] - 12
    var_model_location_season = np.average(var_model_location[:,proxy_seasonality_indices],weights=model_data['ndays_sst'][:,proxy_seasonality_indices],axis=1)
    #if proxy_direction.lower() == 'negative': var_model_location_season = -1*var_model_location_season  # Note: If interpretation direction is not given, it is assumed to be positive.
    #
    return var_model_location_season


# A function to do rank-based comparison with model data - method 1: assign values from the model to the proxy based on rank
#var_name,verbose = 'LakeStatus',False
def rank_based(model_data,proxy_data,var_name,i,options,verbose=False):
    #
    age_model    = model_data['age']
    proxy_ages   = proxy_data['age_centers']
    proxy_values = proxy_data['values_binned'][i]
    #plt.plot(proxy_ages,proxy_values)
    var_model_location_season = get_model_values_nearest(model_data,proxy_data,var_name,i)
    # Get the model values corresponding to the valid data of the proxy
    logical_proxy_valid = (np.isfinite(proxy_values) & [np.isfinite(var_model_location_season[age_model==age])[0] for age in proxy_ages])
    if sum(logical_proxy_valid) == 0: return use_nans(model_data,var_name),proxy_values
    proxy_values_valid = proxy_values[logical_proxy_valid]
    proxy_ages_valid   = proxy_ages[logical_proxy_valid]
    ind_model_selected = [ind for ind in range(len(age_model)) if age_model[ind] in proxy_ages_valid]
    var_model_selected = var_model_location_season[ind_model_selected]
    #
    if sum(np.isfinite(proxy_values_valid)) != sum(np.isfinite(var_model_selected)): print("WARNING: Rank-based PSM: Number of indices don't match")
    #
    # Calculate ranks for the proxy and prior data
    ranks_proxy_values = rankdata(proxy_values_valid)
    ranks_prior_values = rankdata(var_model_selected, method="ordinal")
    #
    # Assign ranked values from the model to the proxy
    proxy_values_valid_new = np.zeros((len(proxy_values_valid))); proxy_values_valid_new[:] = np.nan
    for counter in range(len(proxy_values_valid_new)):
        rank_selected = ranks_proxy_values[counter]
        if rank_selected.is_integer():
            ind_of_ranked_prior = np.where(ranks_prior_values == rank_selected)[0]
            proxy_values_valid_new[counter] = var_model_selected[ind_of_ranked_prior]
        else:
            rank_lower  = np.floor(rank_selected)
            rank_higher = np.ceil(rank_selected)
            ind_rank_lower  = np.where(ranks_prior_values == rank_lower)[0]
            ind_rank_higher = np.where(ranks_prior_values == rank_higher)[0]
            ind_combined = np.concatenate((ind_rank_lower,ind_rank_higher),axis=0)
            proxy_values_valid_new[counter] = np.mean(var_model_selected[ind_combined])
    #
    # Put the proxy data into the same length array as it was originally
    proxy_values_new = np.zeros((len(proxy_values))); proxy_values_new[:] = np.nan
    proxy_values_new[logical_proxy_valid] = proxy_values_valid_new
    #
    # If the proxy has too little varibility (e.g., all initial values were the same), set values to na
    if np.nanstd(proxy_values_new) < 1e-5: proxy_values_new[:] = np.nan
    #
    """
    import matplotlib.pyplot as plt
    plt.plot(proxy_ages,proxy_values,'k-')
    plt.plot(proxy_ages,proxy_values_new,'b-')
    plt.title("Original")
    plt.show()
    """
    #
    return var_model_location_season,proxy_values_new


# A function to do rank-based comparison with model data - method 2: compute ranks of both the proxy and model
#var_name,verbose = 'tas',False
#var_name,verbose = 'LakeStatus',False
def rank_based_older(model_data,proxy_data,var_name,i,options,verbose=False):
    #
    age_model    = model_data['age']
    proxy_ages   = proxy_data['age_centers']
    proxy_values = proxy_data['values_binned'][i]
    var_model_location_season = get_model_values_nearest(model_data,proxy_data,var_name,i)
    #
    # Get the model values corresponding to the valid data of the proxy
    logical_proxy_valid = np.isfinite(proxy_values)
    if sum(logical_proxy_valid) == 0: return use_nans(model_data,var_name),proxy_values
    proxy_values_valid = proxy_values[logical_proxy_valid]
    proxy_ages_valid   = proxy_ages[logical_proxy_valid]
    proxy_ages_valid_start = proxy_ages_valid[0]  - (options['time_resolution']/2)
    proxy_ages_valid_end   = proxy_ages_valid[-1] + (options['time_resolution']/2)
    ind_model_selected = ((age_model >= proxy_ages_valid_start) & (age_model <= proxy_ages_valid_end))
    #
    # Calculate ranks for the proxy and prior data and scale them between 0 and 100 - CH change to allow 0-100 or smaller range given repeat values (important for Lake status)
    ranks_proxy_values = rankdata(proxy_values_valid, nan_policy='omit') - 1
    ranks_prior_values = rankdata(np.where(ind_model_selected,var_model_location_season,np.nan), nan_policy='omit') - 1
    percentile_proxy_values = (ranks_proxy_values / (np.sum(np.isfinite(ranks_proxy_values))-1)) * 100
    percentile_prior_values = []
    for model_n in np.unique(model_data['number']):
        ranks_prior_values_by_model = rankdata(ranks_prior_values[model_data['number']==model_n], nan_policy='omit') - 1
        percentile_prior_values_by_model = (ranks_prior_values_by_model / (np.sum(np.isfinite(ranks_prior_values_by_model))-1)) * 100
        proxy_min = np.nanmin(percentile_proxy_values)
        proxy_max = np.nanmax(percentile_proxy_values)
        prior_min = np.nanmin(percentile_prior_values_by_model)
        prior_max = np.nanmax(percentile_prior_values_by_model)
        #print(proxy_min,proxy_max,prior_min,prior_max)
        percentile_prior_values_by_model = ((percentile_prior_values_by_model-prior_min) / (prior_max-prior_min))
        percentile_prior_values_by_model = (percentile_prior_values_by_model * (proxy_max-proxy_min)) + proxy_min
        percentile_prior_values.append(percentile_prior_values_by_model)
    percentile_prior_values = np.array(percentile_prior_values).flatten()
    #percentile_proxy_values2 = ((ranks_proxy_values-min(ranks_proxy_values)) / (max(ranks_proxy_values)-min(ranks_proxy_values))) * 100
    #percentile_prior_values = ((ranks_prior_values-min(ranks_prior_values[ind_model_selected])) / (max(ranks_prior_values[ind_model_selected])-min(ranks_prior_values[ind_model_selected]))) * 100
    #
    # Put the proxy data into the same length array as it was originally
    percentile_proxy_values_all = np.zeros((len(proxy_values))); percentile_proxy_values_all[:] = np.nan
    percentile_proxy_values_all[logical_proxy_valid] = percentile_proxy_values
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
    plt.title("Original")
    plt.show()
    #
    plt.plot(proxy_ages,percentile_proxy_values_all,'k-')
    plt.plot(age_model,percentile_prior_values,'b-')
    plt.title("Ranks")
    plt.show()
    """
    #
    return percentile_prior_values,percentile_proxy_values_all


# A function to get the NaNs with the same length as the model data
def use_nans(model_data,var_name):
    #
    n_time = model_data[var_name].shape[0]
    nan_array = np.zeros((n_time)); nan_array[:] = np.nan
    #
    return nan_array
