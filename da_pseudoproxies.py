#==============================================================================
# This script creates pseudoproxy files based on selected model data with the
# same characteristics as the selected proxy network, then saves those
# pseudoproxies in the same format as the actual proxies.
#    author: Michael P. Erb
#    date  : 11/17/2020
#==============================================================================

import da_utils
import da_load_models
import da_load_proxies
import numpy as np
import pickle

"""
proxies_to_use = 'temp12k'
model_to_use = 'hadcm3'
#model_to_use = 'trace'
#model_to_use = 'famous'

options = {}
options['data_dir'] = '/projects/pd_lab/data/data_assimilation/'
options['lipd_dir'] = '/projects/pd_lab/mpe32/LiPD-utilities/Python/'
"""

#%%
# A funtion to make pseudoproxies
def make_pseudoproxies(proxies_to_use,model_to_use,noise_to_use,options):
    #
    #%% LOAD DATA
    print(' === Generating pseudoproxies. Settings: Model: '+model_to_use+', Adding noise: none ===')
    #
    if proxies_to_use[0:9] == 'basicgrid':
        #
        # This option creates a very basic pseudoproxy network
        timestep_specified = 10
        gridstep_specified = int(proxies_to_use[9:])
        gridstep_half      = gridstep_specified/2
        lats_specified      = np.arange((90-gridstep_half),-90,-gridstep_specified)
        lons_specified      = np.arange(gridstep_half,360,gridstep_specified)
        ages_specified      = np.arange(5,12000,timestep_specified)
        values_specified    = ages_specified*0  # The values here don't matter. They will be overwritten later.
        season_specified    = 'annual'
        uncertain_specified = 2.1
        #
        # Loop through the specfied grid, creating the grid of basic proxies
        filtered_ts = []
        counter = 1
        for lat_chosen in lats_specified:
            for lon_chosen in lons_specified:
                proxy_new = {}
                proxy_new['geo_meanLat']      = lat_chosen
                proxy_new['geo_meanLon']      = lon_chosen
                proxy_new['age']              = ages_specified
                proxy_new['paleoData_values'] = values_specified
                proxy_new['paleoData_interpretation'] = []
                proxy_new['paleoData_interpretation'].append({'seasonality':season_specified,'seasonalityGeneral':season_specified})
                proxy_new['paleoData_temperature12kUncertainty'] = uncertain_specified
                proxy_new['archiveType']    = 'pseudoproxy'
                proxy_new['dataSetName']    = 'pseudoproxy_'+str(counter)+'_lat_'+str(int(lat_chosen))+'+lon_'+str(int(lon_chosen))
                proxy_new['paleoData_TSid'] = 'pseudoproxy_'+str(counter)+'_lat_'+str(int(lat_chosen))+'+lon_'+str(int(lon_chosen))
                filtered_ts.append(proxy_new)
                counter += 1
        #
        print('=== Creating "basicgrid" pseudoproxy network ===')
        print('Number of pseudoproxies:',len(filtered_ts))
        print('Lats:        ',lats_specified)
        print('Lons:        ',lons_specified)
        print('Age timestep:',timestep_specified)
        print('Seasonality: ',season_specified)
        print('Uncertainty: ',uncertain_specified)
        print('================================================')
        #
    else:
        # Set the right options for loading data
        options_new = {}
        options_new['data_dir'] = options['data_dir']
        options_new['lipd_dir'] = options['lipd_dir']
        options_new['proxy_datasets_to_assimilate'] = [proxies_to_use]
        #
        # Load the proxy data
        filtered_ts,_ = da_load_proxies.load_proxies(options_new)  #TODO: Fix this (Fix what?)
    #
    # Load the model data
    original_model_dir = options['data_dir']+'models/original_model_data/'
    tas_model,ages_model,lat_model,lon_model,time_ndays_model = da_load_models.process_models(model_to_use,'tas',None,None,None,original_model_dir,return_variables=True)
    #
    #
    #%% CALCULATIONS
    #
    lat_res = np.abs(np.median(lat_model[1:] - lat_model[:-1]))
    lon_res = np.abs(np.median(lon_model[1:] - lon_model[:-1]))
    #
    # Loop through the proxies to generate pseudoproxies
    n_proxies = len(filtered_ts)
    missing_uncertainty_count = 0
    i=2
    for i in range(n_proxies):
        #
        #print(' -- Making pseudoproxy '+str(i)+'/'+str(n_proxies)+' --')
        #
        # Get proxy metadata
        proxy_lat                 = filtered_ts[i]['geo_meanLat']
        proxy_lon                 = filtered_ts[i]['geo_meanLon']
        proxy_seasonality_txt     = filtered_ts[i]['paleoData_interpretation'][0]['seasonality']
        try:    proxy_uncertainty = filtered_ts[i]['paleoData_temperature12kUncertainty']
        except: proxy_uncertainty = np.nan; missing_uncertainty_count += 1
        if proxy_uncertainty  == 'NA': proxy_uncertainty = np.nan
        #if proxy_uncertainty  == 'NA': proxy_uncertainty = 2.1 #TODO: Should I be using nan or a number for unknown uncertainties?
        #
        # Convert seasonality to a list of months, with negative values corresponding to the previous year.
        try:
            proxy_seasonality_array = filtered_ts[i]['seasonality_array']
        except:
            proxy_seasonality = da_utils.interpret_seasonality(proxy_seasonality_txt,proxy_lat,'annual')
            proxy_seasonality_array = np.array(proxy_seasonality.split()).astype(np.int)
        #
        # Find the model gridpoint closest to the proxy location
        if proxy_lon < 0: proxy_lon = proxy_lon+360
        lon_model_wrapped = np.append(lon_model,lon_model[0]+360)
        j_selected = np.argmin(np.abs(lat_model-proxy_lat))
        i_selected = np.argmin(np.abs(lon_model_wrapped-proxy_lon))
        if np.abs(proxy_lat-lat_model[j_selected])         > (lat_res/2): print('WARNING: Too large of a lat difference. Proxy lat: '+str(proxy_lat)+', model lat: '+str(lat_model[j_selected]))
        if np.abs(proxy_lon-lon_model_wrapped[i_selected]) > (lon_res/2): print('WARNING: Too large of a lon difference. Proxy lon: '+str(proxy_lon)+', model lon: '+str(lon_model_wrapped[i_selected]))
        if i_selected == len(lon_model_wrapped)-1: i_selected = 0
        #print('Proxy location vs. nearest model gridpoint.  Lat: '+str(proxy_lat)+', '+str(lat_model[j_selected])+'.  Lon: '+str(proxy_lon)+', '+str(lon_model[i_selected]))
        tas_model_location = tas_model[:,:,j_selected,i_selected]
        #
        # Compute an average over months according to the proxy seasonality
        proxy_seasonality_indices = np.abs(proxy_seasonality_array)-1
        tas_model_season = np.average(tas_model_location[:,proxy_seasonality_indices],weights=time_ndays_model[:,proxy_seasonality_indices],axis=1)
        #
        # Get proxy ages
        proxy_ages   = np.array(filtered_ts[i]['age']).astype(np.float)
        proxy_values = np.array(filtered_ts[i]['paleoData_values']).astype(np.float)
        #
        # Find age bounds of proxy data
        proxy_age_bounds = (proxy_ages[1:]+proxy_ages[:-1])/2
        end_newest = proxy_ages[0]  - (proxy_ages[1]-proxy_ages[0])/2
        end_oldest = proxy_ages[-1] + (proxy_ages[-1]-proxy_ages[-2])/2
        proxy_age_bounds = np.insert(proxy_age_bounds,0,end_newest)
        proxy_age_bounds = np.append(proxy_age_bounds,end_oldest)
        #
        # Compute means of the intervals spanned by the proxy data
        n_ages = len(proxy_ages)
        tas_model_season_averaged = np.zeros((n_ages)); tas_model_season_averaged[:] = np.nan
        for j in range(n_ages):
            if np.isnan(proxy_age_bounds[j]) or np.isnan(proxy_age_bounds[j+1]): continue  # This is added for proxies which have nans in their ages.  Should something else be done instead?
            indices_selected = np.where((ages_model >= proxy_age_bounds[j]) & (ages_model < proxy_age_bounds[j+1]))[0]
            tas_model_season_averaged[j] = np.mean(tas_model_season[indices_selected])
        #
        # If there are NaNs in the original data, set them in the pseudoproxies
        tas_model_season_averaged[np.isnan(proxy_values)] = np.nan
        #
        # Add proxy uncertainty (and specify it in the uncertainty variable?), if wanted.
        #TODO: How to handle the uncertainty variable?
        ind_valid = np.isfinite(tas_model_season_averaged)
        if noise_to_use == 'whitesnr05':
            print('Adding noise to pseudoproxies: White noise with a SNR of 0.5')
            #
            # Generate noise for the proxy with a signal-to-noise ratio of 0.5
            # This is the simple version of adding noise that Nathan mentions in Steiger and Hakim, Clim. Past, 2016
            signal_to_noise = 0.5
            target_noise_variance = np.nanvar(tas_model_season_averaged) / np.square(signal_to_noise)
            white_noise = np.random.normal(0,1,sum(ind_valid))
            white_noise_scaled = (white_noise/np.std(white_noise)) * np.sqrt(target_noise_variance)
            white_noise_scaled = white_noise_scaled - np.mean(white_noise_scaled)
            #
            # Add noise to the pseudodata
            tas_model_season_averaged[ind_valid] = tas_model_season_averaged[ind_valid] + white_noise_scaled
            #
        elif noise_to_use == 'whiteproxyrmse':
            #
            # Generate noise with a mse equal to the proxy mse, then add it to the pseudodata
            stdev_of_noise = proxy_uncertainty
            if np.isfinite(stdev_of_noise):
                white_noise_scaled = np.random.normal(0,stdev_of_noise,sum(ind_valid))
                tas_model_season_averaged[ind_valid] = tas_model_season_averaged[ind_valid] + white_noise_scaled
            else:
                tas_model_season_averaged[:] = np.nan
            #
        else:
            print('Not adding noise to pseudoproxies. Keywork is "none" or unknown: '+noise_to_use)
        #
        # Save pseudoproxy data
        filtered_ts[i]['paleoData_values'] = tas_model_season_averaged
    #
    #%% OUTPUT
    #
    # Save the data into a format I can read in python2
    output_dir = options['data_dir']+'proxies/pseudoproxies/'
    output_filename = 'pseudo_'+proxies_to_use+'_using_'+model_to_use+'_noise_'+str(noise_to_use).lower()+'.pkl'
    file_to_save = open(output_dir+output_filename,'wb')
    pickle.dump(filtered_ts,file_to_save,protocol=2)
    file_to_save.close()
    #
    print(' === COMPLETE ==')

