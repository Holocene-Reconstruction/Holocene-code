#==============================================================================
# This script creates pseudoproxy files based on selected model data with the
# same characteristics as the selected proxy network, then saves those
# pseudoproxies in the same format as the actual proxies.
#    author: Michael P. Erb
#    date  : 3/16/2022
#==============================================================================

import da_utils
import da_load_models
import da_load_proxies
import da_psms
import numpy as np
import pickle

#%%
# A function to make pseudoproxies
def make_pseudoproxies(proxies_to_use,model_to_use,noise_to_use,options):
    #
    # LOAD DATA
    print(' === Generating pseudoproxies. Settings: Proxies: '+proxies_to_use+', Model: '+model_to_use+', Noise: '+noise_to_use+' ===')
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
        uncertainty_specified = 2.1
        #
        # Loop through the specfied grid, creating the grid of basic proxies
        pseudoproxy_data = []
        counter = 1
        for lat_chosen in lats_specified:
            for lon_chosen in lons_specified:
                pseudoproxy_new = {}
                pseudoproxy_new['geo_meanLat']      = lat_chosen
                pseudoproxy_new['geo_meanLon']      = lon_chosen
                pseudoproxy_new['age']              = ages_specified
                pseudoproxy_new['paleoData_values'] = values_specified
                pseudoproxy_new['paleoData_interpretation'] = [{'seasonality':season_specified,'seasonalityGeneral':season_specified}]
                pseudoproxy_new['paleoData_temperature12kUncertainty'] = uncertainty_specified
                pseudoproxy_new['archiveType']     = 'pseudoproxy'
                pseudoproxy_new['paleoData_proxy'] = 'pseudoproxy'
                pseudoproxy_new['paleoData_units'] = 'degC'  #Note: Consider changing this line in the future, for other sorts of pseudoproxies.
                pseudoproxy_new['dataSetName']     = 'pseudoproxy_'+str(counter)+'_lat_'+str(int(lat_chosen))+'+lon_'+str(int(lon_chosen))
                pseudoproxy_new['paleoData_TSid']  = 'pseudoproxy_'+str(counter)+'_lat_'+str(int(lat_chosen))+'+lon_'+str(int(lon_chosen))
                pseudoproxy_data.append(pseudoproxy_new)
                counter += 1
        #
        print('=== Creating "basicgrid" pseudoproxy network ===')
        print('Number of pseudoproxies:',len(pseudoproxy_data))
        print('Lats:        ',lats_specified)
        print('Lons:        ',lons_specified)
        print('Age timestep:',timestep_specified)
        print('Seasonality: ',season_specified)
        print('Uncertainty: ',uncertainty_specified)
        print('================================================')
        #
    else:
        # Set the right options for loading data
        options_new = {}
        options_new['data_dir']                     = options['data_dir']
        options_new['reconstruction_type']          = options['reconstruction_type']
        options_new['proxy_datasets_to_assimilate'] = [proxies_to_use]
        #
        # Load the proxy data
        pseudoproxy_data,_ = da_load_proxies.load_proxies(options_new)
    #
    # Load the model data
    original_model_dir = options['data_dir']+'models/original_model_data/'
    tas_model,ages_model,lat_model,lon_model,time_ndays_model = da_load_models.process_models(model_to_use,'tas',None,None,None,original_model_dir,return_variables=True)
    #
    #
    #%% CALCULATIONS
    #
    # Loop through the proxies to generate pseudoproxies
    n_proxies = len(pseudoproxy_data)
    missing_uncertainty_count = 0
    for i in range(n_proxies):
        #
        # Get proxy metadata
        missing_uncertainty_value = np.nan
        try:    proxy_uncertainty = pseudoproxy_data[i]['paleoData_temperature12kUncertainty']
        except: proxy_uncertainty = missing_uncertainty_value; missing_uncertainty_count += 1
        proxy_uncertainty = float(proxy_uncertainty)
        #
        # Convert seasonality to a list of months, with negative values corresponding to the previous year.
        try:
            proxy_seasonality_array = pseudoproxy_data[i]['seasonality_array']
        except:
            proxy_lat             = pseudoproxy_data[i]['geo_meanLat']
            proxy_seasonality_txt = pseudoproxy_data[i]['paleoData_interpretation'][0]['seasonality']
            proxy_seasonality = da_utils.interpret_seasonality(proxy_seasonality_txt,proxy_lat,'annual')
            proxy_seasonality_array = np.array(proxy_seasonality.split()).astype(int)
        #
        # Find the model gridpoint closest to the proxy location
        model_data_for_pseudo = {'tas':tas_model, 'lat':lat_model, 'lon':lon_model, 'time_ndays':time_ndays_model}
        proxy_data_for_pseudo = {'lats':[pseudoproxy_data[i]['geo_meanLat']], 'lons':[pseudoproxy_data[i]['geo_meanLon']], 'seasonality_array':[proxy_seasonality_array]}
        tas_model_season = da_psms.get_model_values(model_data_for_pseudo,proxy_data_for_pseudo,'tas',0)
        #
        # Get proxy ages
        proxy_ages   = np.array(pseudoproxy_data[i]['age']).astype(float)
        proxy_values = np.array(pseudoproxy_data[i]['paleoData_values']).astype(float)
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
            if np.isnan(proxy_age_bounds[j]) or np.isnan(proxy_age_bounds[j+1]): continue
            indices_selected = np.where((ages_model > proxy_age_bounds[j]) & (ages_model <= proxy_age_bounds[j+1]))[0]
            tas_model_season_averaged[j] = np.mean(tas_model_season[indices_selected])
        #
        # If there are NaNs in the original data, set them in the pseudoproxies
        tas_model_season_averaged[np.isnan(proxy_values)] = np.nan
        #
        # Add proxy uncertainty, if specified.
        # Note: In the future, consider whether there are better ways of generating noise
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
            # Update the uncertainty value for the pseudoproxy
            pseudoproxy_data[i]['paleoData_temperature12kUncertainty'] = np.stdev(white_noise_scaled)
            #
        elif noise_to_use == 'whiteproxyrmse':
            #
            # Generate noise with a mse equal to the proxy mse, then add it to the pseudodata
            stdev_of_noise = proxy_uncertainty
            if np.isfinite(stdev_of_noise):
                white_noise_scaled = np.random.normal(0,stdev_of_noise,sum(ind_valid))
                white_noise_scaled = white_noise_scaled - np.mean(white_noise_scaled)
                tas_model_season_averaged[ind_valid] = tas_model_season_averaged[ind_valid] + white_noise_scaled
            else:
                tas_model_season_averaged[:] = np.nan
            #
        else:
            print('Not adding noise to pseudoproxies. Keywork is "none" or unknown: '+noise_to_use)
        #
        # Save pseudoproxy data
        pseudoproxy_data[i]['paleoData_values'] = tas_model_season_averaged
        pseudoproxy_data[i]['age']              = proxy_ages
    #
    #%% OUTPUT
    #
    # Save the pseudoproxy data
    output_dir = options['data_dir']+'proxies/pseudoproxies/'
    output_filename = 'pseudo_'+proxies_to_use+'_using_'+model_to_use+'_noise_'+str(noise_to_use).lower()+'.pkl'
    file_to_save = open(output_dir+output_filename,'wb')
    pickle.dump(pseudoproxy_data,file_to_save,protocol=2)
    file_to_save.close()
    #
    print(' === COMPLETE ==')

