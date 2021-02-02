#==============================================================================
# This script creates pseudoproxy files based on selected model data with the
# same characteristics as the selected proxy network, then saves those
# pseudoproxies in the same format as the actual proxies.
#    author: Michael P. Erb
#    date  : 11/17/2020
#==============================================================================

import da_utils
import da_process_models
import da_load_data
import numpy as np
import pickle

"""
proxies_to_use = 'temp12k'
#model_to_use = 'hadcm3'
model_to_use = 'trace'

options = {}
options['data_dir'] = '/projects/pd_lab/data/data_assimilation/'
options['lipd_dir'] = '/projects/pd_lab/mpe32/LiPD-utilities/Python/'
"""

# A funtion to make pseudoproxies
def make_pseudoproxies(proxies_to_use,model_to_use,options):
    #
    #%% LOAD DATA
    print(' === Generating pseudoproxies. Settings: Model: '+model_to_use+', Adding noise: none ===')
    #
    # Set the right options for loading data
    options_new = {}
    options_new['data_dir'] = options['data_dir']
    options_new['lipd_dir'] = options['lipd_dir']
    options_new['proxy_datasets_to_assimilate'] = [proxies_to_use]
    #
    # Load the proxy data
    filtered_ts,_ = da_load_data.load_proxies(options_new)
    #
    # Load the model data
    model_dir = options_new['data_dir']+'models/original_model_data/'
    tas_model,ages_model,lat_model,lon_model,time_ndays_model = da_process_models.process_models(model_to_use,None,None,None,model_dir,return_variables=True)
    #
    #
    #%% CALCULATIONS
    #    
    # Loop through the proxies to generate pseudoproxies
    n_proxies = len(filtered_ts)
    missing_uncertainty_count = 0
    i=0
    for i in range(n_proxies):
        #
        #print(' -- Making pseudoproxy '+str(i)+'/'+str(n_proxies)+' --')
        #
        # Get proxy metdata
        proxy_lat                 = filtered_ts[i]['geo_meanLat']
        proxy_lon                 = filtered_ts[i]['geo_meanLon']
        proxy_seasonality_txt     = filtered_ts[i]['paleoData_interpretation'][0]['seasonality']
        try:    proxy_uncertainty = filtered_ts[i]['paleoData_temperature12kUncertainty']
        except: proxy_uncertainty = np.nan; missing_uncertainty_count += 1
        if proxy_uncertainty  == 'NA': proxy_uncertainty = np.nan
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
        if np.abs(proxy_lat-lat_model[j_selected])         > 2: print('WARNING: Too large of a lat difference. Proxy lat: '+str(proxy_lat)+', model lat: '+str(lat_model[j_selected]))
        if np.abs(proxy_lon-lon_model_wrapped[i_selected]) > 2: print('WARNING: Too large of a lon difference. Proxy lon: '+str(proxy_lon)+', model lon: '+str(lon_model_wrapped[i_selected]))
        if i_selected == len(lon_model_wrapped)-1: i_selected = 0
        print('Proxy location vs. nearest model gridpoint.  Lat: '+str(proxy_lat)+', '+str(lat_model[j_selected])+'.  Lon: '+str(proxy_lon)+', '+str(lon_model[i_selected]))
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
        #TODO: Add proxy uncertainty and specify it in the uncertainty variable, if wanted
        #
        # Save pseudoproxy data
        filtered_ts[i]['paleoData_values'] = tas_model_season_averaged
    #
    #%% OUTPUT
    proxy_dataset_names = '_'.join(options_new['proxy_datasets_to_assimilate'])
    #
    # Save the data into a format I can read in python2
    output_dir = options_new['data_dir']+'proxies/pseudoproxies/'
    output_filename = 'pseudoproxies_'+proxy_dataset_names+'_generated_from_'+model_to_use+'.pkl'
    file_to_save = open(output_dir+output_filename,'wb')
    pickle.dump(filtered_ts,file_to_save,protocol=2)
    file_to_save.close()
    #
    print(' === COMPLETE ==')
