#==============================================================================
# Different PSMs for use in the DA project.
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
        if ~(model_data['age_sst'] == model_data['age']).all():
            print('WARNING: SST ages are different from other model ages. This will cause the code to crash.')
    #
    # Loop through proxies, using PSMs
    n_proxies = proxy_data['values_binned'].shape[0]
    proxy_estimates_all = np.array([dict() for k in range(n_proxies)])  # HXb
    i = 0  # calibrated_sst
    i = 4  # calibrated_tas
    i = 5  # rank_based_LakeStatus
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
        if psm_selected == "calibrated_sst": proxy_estimate = get_model_values_sst(model_data,proxy_data,i,options)
        elif psm_name == 'calibrated': proxy_estimate = get_model_values(model_data,proxy_data,var_name,i,interp_method="bilinear")
        elif psm_name == 'rank_based':
            proxy_estimate,proxy_update = rank_based(model_data,proxy_data,var_name,i,options)
            proxy_data['values_binned'][i,:] = proxy_update
            units_new = proxy_data['units'][i]+'_percentile'
            proxy_data['units'][i]      = units_new
            proxy_data['metadata'][i,8] = units_new
        elif psm_selected == 'use_nans': proxy_estimate = use_nans(model_data,var_name)
        else:                            proxy_estimate = use_nans(model_data,var_name)
        #
        # Find all time resolutions in the record
        proxy_res_12ka_unique = np.unique(proxy_data['resolution_binned'][i,:])
        proxy_res_12ka_unique_sorted = np.sort(proxy_res_12ka_unique[np.isfinite(proxy_res_12ka_unique)]).astype(int)
        #
        # Allow for proxy estimate and estimate to have same reference window if data does not cover entire window (allows for more flexibility with larger windows)
        if options['reconstruction_type'] == 'relative':
            age_model = model_data['age']
            ind_model_refwindow = []
            for a,age in enumerate(age_model):
                if (age >= options['reference_period'][0]) & (age < options['reference_period'][1]):
                    if np.isfinite(proxy_data['values_binned'][i,proxy_data['age_centers']==age]):
                        ind_model_refwindow.append(a)                
            if len(ind_model_refwindow) == 0:
                proxy_estimate[:] = np.nan
            else: 
                proxy_estimate = proxy_estimate - np.nanmean(proxy_estimate[ind_model_refwindow])
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
def get_model_values(model_data,proxy_data,var_name,i,interp_method="linear",verbose=False):
    #
    proxy_lat    = proxy_data['lats'][i]
    proxy_lon    = proxy_data['lons'][i]
    proxy_season = proxy_data['seasonality_array'][i]
    ndays_model  = model_data['time_ndays']
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
    #model_data_df_location = model_data_df.interp(lat=proxy_lat,lon=proxy_lon,method="linear")
    model_data_df_location = model_data_df.interp(lat=proxy_lat,lon=proxy_lon,method="nearest")
    var_model_location = model_data_df_location["var"].values
    #
    """
    plt.contourf(model_data['lon'],model_data['lat'],model_data_df["var"][0,0,:,:])
    plt.scatter(proxy_lon,proxy_lat)
    """
    #
    # Compute an average over months according to the proxy seasonality
    # Note: months are always taken from the current year, not from the previous year. Since we're working with multi-year means, this makes little difference
    proxy_seasonality_indices = np.abs(proxy_season)-1
    proxy_seasonality_indices[proxy_seasonality_indices > 11] = proxy_seasonality_indices[proxy_seasonality_indices > 11] - 12
    var_model_location_season = np.average(var_model_location[:,proxy_seasonality_indices],weights=ndays_model[:,proxy_seasonality_indices],axis=1)
    #
    return var_model_location_season


# A function to get the model values at the same location and seasonality as the proxy
def get_model_values_sst(model_data,proxy_data,i,options,verbose=False):
    #
    # Load selected variables
    proxy_lat    = proxy_data['lats'][i]
    proxy_lon    = proxy_data['lons'][i]
    proxy_season = proxy_data['seasonality_array'][i]
    #
    # CALCULATE DISTANCES
    # Calculate mean SST, for identiying nans
    sst_mean = np.mean(np.mean(model_data['sst'],axis=1),axis=0)
    #plt.scatter(model_data['lon_sst'],model_data['lat_sst'],c=sst_mean)
    #
    # Calculate distances to each model location
    n_lat = model_data['lat_sst'].shape[0]
    n_lon = model_data['lat_sst'].shape[1]
    distances = np.zeros((n_lat,n_lon)); distances[:] = np.nan
    for ind_lon in range(n_lon):
        for ind_lat in range(n_lat):
            if np.isfinite(sst_mean[ind_lat,ind_lon]):
                distances[ind_lat,ind_lon] = da_utils_lmr.haversine(proxy_lon,proxy_lat,model_data['lon_sst'][ind_lat,ind_lon],model_data['lat_sst'][ind_lat,ind_lon])
            else:
                distances[ind_lat,ind_lon] = 10000  # This is a high number to make sure the PSM doesn't select grid cells with nans
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
    if np.isnan(var_model_location).all():
        print("Warning: All values are nan in SST PSM for proxy",i)
    #
    # Compute an average over months according to the proxy seasonality
    # Note: months are always taken from the current year, not from the previous year. Since we're working with multi-year means, this makes little difference
    proxy_seasonality_indices = np.abs(proxy_season)-1
    proxy_seasonality_indices[proxy_seasonality_indices > 11] = proxy_seasonality_indices[proxy_seasonality_indices > 11] - 12
    var_model_location_season = np.average(var_model_location[:,proxy_seasonality_indices],weights=model_data['ndays_sst'][:,proxy_seasonality_indices],axis=1)
    #
    return var_model_location_season


# A function to do rank-based comparison with model data - method 1: compute ranks for both the proxy and model prior
#var_name,verbose = 'LakeStatus',False
def rank_based(model_data,proxy_data,var_name,i,options,verbose=False):
    #
    age_model    = model_data['age']
    proxy_ages   = proxy_data['age_centers']
    proxy_values = proxy_data['values_binned'][i]
    var_model_location_season = get_model_values(model_data,proxy_data,var_name,i,interp_method="nearest")
    #
    # Get the model values corresponding to the valid data of the proxy
    boolean_proxy_valid = (np.isfinite(proxy_values) & [np.isfinite(var_model_location_season[age_model==age])[0] for age in proxy_ages])
    if sum(boolean_proxy_valid) == 0: return use_nans(model_data,var_name),proxy_values[:]*np.nan
    proxy_values_valid = np.round(proxy_values[boolean_proxy_valid],9)
    proxy_ages_valid   = proxy_ages[boolean_proxy_valid]
    boolean_model_selected = np.array([age in proxy_ages_valid for age in age_model])
    """
    plt.plot(proxy_ages[boolean_proxy_valid],proxy_values_valid,'bo-')
    plt.plot(age_model[boolean_model_selected],var_model_location_season[boolean_model_selected],'ro-')
    """
    #
    # Calculate ranks for the proxy data and scale it between 0 and 100. (Ranges can be lower than 0-100 with repeat values; important for Lake status)
    ranks_proxy_values = rankdata(proxy_values_valid, nan_policy='omit') - 1  # Put values in order, with the theoretical lowest being 0. For ties, ranks are averaged.
    percentile_proxy_values = (ranks_proxy_values / (np.sum(np.isfinite(ranks_proxy_values))-1)) * 100  # Scale values, so the range is 0-100 in theory. For ties, values are averaged.
    #
    # Put the proxy data into the same length array as it was originally
    percentile_proxy_values_all = np.zeros((len(proxy_values))); percentile_proxy_values_all[:] = np.nan
    percentile_proxy_values_all[boolean_proxy_valid] = percentile_proxy_values
    #
    # Rank the prior data
    #ranks_prior_values = rankdata(var_model_location_season, nan_policy='omit') - 1
    percentile_prior_values = []
    for model_n in np.unique(model_data['number']):
        ranks_prior_values_by_model = rankdata(var_model_location_season[model_data['number']==model_n], nan_policy='omit') - 1
        ranks_overlapping_with_proxy = ranks_prior_values_by_model[boolean_model_selected[model_data['number']==model_n]]
        ranks_prior_values_by_model = (ranks_prior_values_by_model - np.nanmin(ranks_overlapping_with_proxy)) * (100/np.nanmax(ranks_overlapping_with_proxy))
        percentile_prior_values.extend(ranks_prior_values_by_model)
    percentile_prior_values = np.array(percentile_prior_values)
    #
    # Remove the reference period from the proxy and prior values
    #TODO: Check that this is a good solution. If I want to use this, consider doing it seperately for each model
    ind_ref_proxy = np.where((proxy_ages >= options['reference_period'][0]) & (proxy_ages < options['reference_period'][1]))[0]
    ind_ref_prior = np.where((age_model  >= options['reference_period'][0]) & (age_model  < options['reference_period'][1]))[0]
    percentile_proxy_values_all = percentile_proxy_values_all - np.nanmean(percentile_proxy_values_all[ind_ref_proxy])
    percentile_prior_values     = percentile_prior_values     - np.nanmean(percentile_prior_values[ind_ref_prior])
    if np.isnan(percentile_proxy_values_all[ind_ref_proxy]).all(): print('No data in reference period, index: '+str(i))
    #
    """
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


# A function to do rank-based comparison with model data - method 2: assign values from the model to the proxy based on rank
#var_name,verbose = 'LakeStatus',False
def rank_based_quantile_mapping(model_data,proxy_data,var_name,i,options,verbose=False):
    #
    age_model    = model_data['age']
    proxy_ages   = proxy_data['age_centers']
    proxy_values = proxy_data['values_binned'][i]
    #plt.plot(proxy_ages,proxy_values)
    var_model_location_season = get_model_values(model_data,proxy_data,var_name,i,interp_method="nearest")
    # Get the model values corresponding to the valid data of the proxy
    boolean_proxy_valid = (np.isfinite(proxy_values) & [np.isfinite(var_model_location_season[age_model==age])[0] for age in proxy_ages])
    if sum(boolean_proxy_valid) == 0: return use_nans(model_data,var_name),proxy_values
    proxy_values_valid = proxy_values[boolean_proxy_valid]
    proxy_ages_valid   = proxy_ages[boolean_proxy_valid]
    boolean_model_selected = [ind for ind in range(len(age_model)) if age_model[ind] in proxy_ages_valid]
    var_model_selected = var_model_location_season[boolean_model_selected]
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
    proxy_values_new[boolean_proxy_valid] = proxy_values_valid_new
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


# A function to get the NaNs with the same length as the model data
def use_nans(model_data,var_name):
    #
    n_time = model_data[var_name].shape[0]
    nan_array = np.zeros((n_time)); nan_array[:] = np.nan
    #
    return nan_array
