#==============================================================================
# Functions for loading proxy data for the data assimilation project.
#    author: Michael P. Erb
#    date  : 10/12/2022
#==============================================================================

import da_utils
import da_pseudoproxies
import da_psms
import numpy as np
import pickle
import lipd
import glob
from scipy import interpolate
from scipy import stats

# A function to load the chosen proxy datasets
def load_proxies(options):
    #
    # Set the necessary directories
    dir_proxies_temp12k  = options['data_dir']+'proxies/temp12k/'
    dir_proxies_hydro12k = options['data_dir']+'proxies/hydro12k/'
    dir_proxies_pages2k  = options['data_dir']+'proxies/pages2k/'
    dir_proxies_pseudo   = options['data_dir']+'proxies/pseudoproxies/'
    collection_all = []
    proxy_ts = []
    #
    n_datasets = len(options['proxy_datasets_to_assimilate'])
    for i,proxy_dataset in enumerate(options['proxy_datasets_to_assimilate']):
        print('Loading proxy dataset '+str(i+1)+'/'+str(n_datasets)+': '+proxy_dataset)
        if proxy_dataset == 'temp12k':
            #
            # Load the Temp12k proxy metadata
            file_to_open = open(dir_proxies_temp12k+'Temp12k'+options['version_temp12k']+'.pkl','rb')
            proxies_all_12k = pickle.load(file_to_open)['D']
            file_to_open.close()
            #
            # Extract the time series and use only those which are in Temp12k and in units of degC
            all_ts_12k = lipd.extractTs(proxies_all_12k)
            #
            # Fix the GISP2 ages - Note: this is a temporary fix, since lipd isn't loading the right ages.
            for i in range(len(all_ts_12k)):
                if (all_ts_12k[i]['dataSetName'] == 'Alley.GISP2.2000') and (all_ts_12k[i]['paleoData_variableName'] == 'age'): gisp2_ages = all_ts_12k[i]['paleoData_values']
            #
            for i in range(len(all_ts_12k)):
                if (all_ts_12k[i]['dataSetName'] == 'Alley.GISP2.2000') and (all_ts_12k[i]['paleoData_variableName'] == 'temperature') and (np.max(np.array(all_ts_12k[i]['age']).astype(float)) < 50):
                    print('Fixing GISP2 ages:',all_ts_12k[i]['paleoData_variableName'],', Index:',i)
                    all_ts_12k[i]['age'] = gisp2_ages
            #
            proxy_ts_temp12k = lipd.filterTs(all_ts_12k,      'paleoData_inCompilation == Temp12k')
            proxy_ts_temp12k = lipd.filterTs(proxy_ts_temp12k,'paleoData_units == degC')
            if options['reconstruction_type'] == 'absolute': proxy_ts_temp12k = lipd.filterTs(proxy_ts_temp12k,'paleoData_datum == abs')
            #
            proxy_ts = proxy_ts + proxy_ts_temp12k
            collection_all = collection_all + ([proxy_dataset] * len(proxy_ts_temp12k))
            #
            # Some proxies have problems in the metadata.  Fix them here.
            for i in range(len(proxy_ts)):
                if proxy_ts[i]['paleoData_TSid'] == 'RXEc3JaUSUk': proxy_ts[i]['paleoData_temperature12kUncertainty'] = 2.1  # This record has an uncertainty value of "3; 2", but it should be 2.1.
                if proxy_ts[i]['paleoData_TSid'] == 'RWzl4NCma8r': proxy_ts[i]['paleoData_interpretation'][0]['seasonality'] = 'summer'  # This record is lacking a seasonality field.
                if ('seasonality'        in proxy_ts[i]['paleoData_interpretation'][0].keys())==False: proxy_ts[i]['paleoData_interpretation'][0]['seasonality']        = ''
                if ('seasonalityGeneral' in proxy_ts[i]['paleoData_interpretation'][0].keys())==False: proxy_ts[i]['paleoData_interpretation'][0]['seasonalityGeneral'] = ''
            #
        elif proxy_dataset == 'hydro12k':
            #
            # Load the uncertainty file #TODO: Incorporate this into the code better
            uncertainty_file = dir_proxies_hydro12k+'LegacyClimateTSidPassQC.csv'
            proxy_info_from_file = np.genfromtxt(uncertainty_file,delimiter=',',dtype='str')
            tsid_from_file = proxy_info_from_file[1:,2]
            rmse_from_file = proxy_info_from_file[1:,9].astype(float)
            tsid_from_file = np.array([entry[1:-1] for entry in tsid_from_file])
            rmse_selected_median = 114
            #
            # Load the hydro12k proxy metadata #DAMP12k - change to open 0_7_0 which doesn't have a pickle file. once this is fixed, only try code will be needed 
            try: 
                file_to_open = open(dir_proxies_hydro12k+options['version_hydro12k']+'.pkl','rb') #DAMP12k- make more flexible
                proxies_all_hydro12k = pickle.load(file_to_open)['D']
                file_to_open.close()
            #
            # Extract the time series
                all_ts_hydro12k = lipd.extractTs(proxies_all_hydro12k)
            #
            except: #pickle not available for 0_7_0. load npy file saved by except code if running the script for the first time #TODO
                try:
                    all_ts_hydro12k = np.load(dir_proxies_hydro12k+'HoloceneHydroclimate'+options['version_hydro12k']+'.npy',allow_pickle=True)
                except:
                    D = lipd.readLipd(dir_proxies_hydro12k+dir_proxies_hydro12k+'HoloceneHydroclimate'+options['version_hydro12k']+'/')
                    all_ts_hydro12k = lipd.extractTs(D)
                    np.save(dir_proxies_hydro12k+dir_proxies_hydro12k+'HoloceneHydroclimate'+options['version_hydro12k']+'.npy',all_ts_hydro12k,allow_pickle=True)
            # Get all proxies in the compilation
            ind_hydro = []
            for i in range(len(all_ts_hydro12k)):
                keys = list(all_ts_hydro12k[i].keys())
                if ('paleoData_inCompilationBeta' in keys) and ('age' in keys) and ('archiveType' in keys):
                    compilations,versions = [],[]
                    n_values = len(all_ts_hydro12k[i]['paleoData_inCompilationBeta'])
                    for j in range(n_values):
                        compilations.append(all_ts_hydro12k[i]['paleoData_inCompilationBeta'][j]['compilationName'])
                        versions.append(all_ts_hydro12k[i]['paleoData_inCompilationBeta'][j]['compilationVersion'])
                    #
                    if ('HoloceneHydroclimate' in compilations) and (options['version_hydro12k'] in str(versions)):
                        try:    _ = all_ts_hydro12k[i]['paleoData_interpretation'][0]['seasonality']  #TODO: Update this later
                        except: print('"seasonality" key not found for index',i); continue
                        dataunits = all_ts_hydro12k[i]['paleoData_units']
                        interp    = all_ts_hydro12k[i]['paleoData_interpretation'][0]['variable']
                        archive   = all_ts_hydro12k[i]['archiveType']     #DAMP12k- used for selecting lakestatus PSM
                        proxy     = all_ts_hydro12k[i]['paleoData_proxy'] #DAMP12k- used for selecting lakestatus PSM
                        #
                        # Hydro12k data currently lack uncertainty values. Set them here. #TODO: Update this later.
                        tsid = all_ts_hydro12k[i]['paleoData_TSid']
                        ind_rmse = np.where(tsid_from_file == tsid)[0]
                        if len(ind_rmse) == 1: all_ts_hydro12k[i]['paleoData_temperature12kUncertainty'] = rmse_from_file[ind_rmse[0]]
                        else:                  all_ts_hydro12k[i]['paleoData_temperature12kUncertainty'] = rmse_selected_median
                        #
                        all_ts_hydro12k[i]['paleoData_temperature12kUncertainty'] = rmse_selected_median
                        if 'CalibratedPollen' in options['assimilate_selected_HCproxies']: #Get Calibrated Pollen Data
                            if (dataunits in ['mm/a']) and (interp in ['P']) and (proxy.lower()=='pollen'): ind_hydro.append(i) 
                        if 'CalibratedOther' in options['assimilate_selected_HCproxies']: #Get Other Calibrated Data
                            if (dataunits in ['mm/a']) and (interp in ['P']) and (proxy.lower()!='pollen'): ind_hydro.append(i) 
                        if 'LakeStatus' in options['assimilate_selected_HCproxies']: #Get Lake Status Data
                            if archive in ['LakeDeposits','Shoreline']:#'Shoreline',
                                ind_hydro.append(i) #Get Shoreline Data
                                #Convert paleoData_values to percentile within age range of model
                                ages = np.array(all_ts_hydro12k[i]['age'])
                                idx = np.where((ages <= options['age_range_model'][1]) & (ages >= options['age_range_model'][0]))#Get index for values which cover model prior
                                vals =  np.array([float(x) for x in all_ts_hydro12k[i]['paleoData_values']])[idx]               #Some values stored as string
                                if all_ts_hydro12k[i]['paleoData_interpretation'][0]['direction'] == 'negative': vals*=-1        #Orient values
                                vals = da_psms.vals2percentile(vals)
                                #Rescale option test (0-1)
                                vals -= np.nanmin(vals) 
                                vals /= np.nanmax(vals) 
                                #
                                all_ts_hydro12k[i]['age']=list(ages[idx])
                                all_ts_hydro12k[i]['paleoData_values']=list(vals)
                                all_ts_hydro12k[i]['paleoData_temperature12kUncertainty'] = 0.3#round(np.nanmax(np.diff(np.unique(np.append(vals,[0,1])))),3) #Median difference between percentile ranks as unc. value
            #
            proxy_ts_hydro12k = [all_ts_hydro12k[i] for i in ind_hydro]
            if options['reconstruction_type'] == 'absolute': proxy_ts_hydro12k = lipd.filterTs(proxy_ts_hydro12k,'paleoData_datum == abs')
            print('Number of hydro12k records selected:',len(proxy_ts_hydro12k))
            #
            proxy_ts = proxy_ts + proxy_ts_hydro12k
            collection_all = collection_all + ([proxy_dataset] * len(proxy_ts_hydro12k))
            #
        elif proxy_dataset == 'pages2k':
            #
            # Load the PAGES2k proxies
            file_to_open = open(dir_proxies_pages2k+'proxies_pages2k_temp_with_psms.pkl','rb')
            proxy_ts_pages2k = pickle.load(file_to_open)
            file_to_open.close()
            #
            proxy_ts = proxy_ts + proxy_ts_pages2k
            collection_all = collection_all + ([proxy_dataset] * len(proxy_ts_pages2k))
            #
        elif proxy_dataset[0:7] == 'pseudo_':
            #
            # Check to see if the file exists.  If not, create it.
            proxy_filename = proxy_dataset+'.pkl'
            filenames_all = glob.glob(dir_proxies_pseudo+'*.pkl')
            filenames_all = [filename.split('/')[-1] for filename in filenames_all]
            if proxy_filename not in filenames_all:
                print('File '+dir_proxies_pseudo+proxy_filename+' does not exist.  Creating it now.')
                proxies_to_use = proxy_dataset.split('_')[1]
                model_to_use   = proxy_dataset.split('_')[3]
                noise_to_use   = proxy_dataset.split('_')[5]
                var_to_use     = proxy_dataset.split('_')[6]
                da_pseudoproxies.make_pseudoproxies(proxies_to_use,model_to_use,noise_to_use,options)
                print('File '+dir_proxies_pseudo+proxy_filename+' created!')
            #
            # Load the pseudoproxies
            file_to_open = open(dir_proxies_pseudo+proxy_filename,'rb')
            proxy_ts_pseudo = pickle.load(file_to_open)
            file_to_open.close()
            #
            proxy_ts = proxy_ts + proxy_ts_pseudo
            collection_all = collection_all + ([proxy_dataset] * len(proxy_ts_pseudo))
            #
        else:
            print('ERROR: invalid proxy dataset: '+proxy_dataset)
    #
    # Process proxy data
    return proxy_ts,collection_all


# Process the proxy data
def process_proxies(proxy_ts,collection_all,options):
    #
    print('\n=== Processing proxy data. This can take a few minutes. Please wait. ===')
    #
    # Set age range to reconstruct, as well as the reference period (The -0.5 accounts for the fact that age years are represented as whole numbers)
    age_bounds = np.arange(options['age_range_to_reconstruct'][0],options['age_range_to_reconstruct'][1]+1,options['time_resolution']) - 0.5
    age_centers = (age_bounds[:-1]+age_bounds[1:])/2
    #
    # Set the maximum proxy resolution
    max_res_value = int(options['maximum_resolution']/options['time_resolution'])
    #
    # Get dimensions
    n_ages    = len(age_centers)
    n_proxies = len(proxy_ts)
    #
    # Set up arrays for the processed proxy data to be stored in
    proxy_data = {}
    proxy_data['values_binned']     = np.zeros((n_proxies,n_ages));          proxy_data['values_binned'][:]     = np.nan
    proxy_data['resolution_binned'] = np.zeros((n_proxies,n_ages));          proxy_data['resolution_binned'][:] = np.nan
    proxy_data['metadata']          = np.zeros((n_proxies,13),dtype=object); proxy_data['metadata'][:]          = np.nan #DAMP12k- increase info about proxies saved
    proxy_data['lats']              = np.zeros((n_proxies));                 proxy_data['lats'][:]              = np.nan
    proxy_data['lons']              = np.zeros((n_proxies));                 proxy_data['lons'][:]              = np.nan
    #proxy_data['uncertainty']       = np.zeros((n_proxies));                 proxy_data['uncertainty'][:]       = np.nan
    proxy_data['uncertainty']       = []
    proxy_data['archivetype']       = []
    proxy_data['proxytype']         = []
    proxy_data['units']             = []
    proxy_data['interp']            = []
    proxy_data['seasonality_array'] = {}
    #
    # If using data from age-uncertainty ensembles, load the data here
    if options['age_uncertain_method']:
        file_ageuncertain_median = options['data_dir']+'proxies/temp12k/ageuncertain_files/ageuncertainty_testing_medians_Temp12k'+options['version_temp12k']+'.csv'
        file_ageuncertain_mse    = options['data_dir']+'proxies/temp12k/ageuncertain_files/ageuncertainty_testing_mse_Temp12k'+options['version_temp12k']+'.csv'
        #
        ageuncertain_ages          = np.genfromtxt(file_ageuncertain_median,delimiter=',',dtype='str')[1:,0].astype(float)
        ageuncertain_tsids         = np.genfromtxt(file_ageuncertain_median,delimiter=',',dtype='str')[0,1:]
        ageuncertain_values        = np.genfromtxt(file_ageuncertain_median,delimiter=',',dtype='str')[1:,1:].astype(float)
        ageuncertain_ages2         = np.genfromtxt(file_ageuncertain_mse,   delimiter=',',dtype='str')[1:,0].astype(float)
        ageuncertain_tsids2        = np.genfromtxt(file_ageuncertain_mse,   delimiter=',',dtype='str')[0,1:]
        ageuncertain_uncertainties = np.genfromtxt(file_ageuncertain_mse,   delimiter=',',dtype='str')[1:,1:].astype(float)
        #
        if not np.array_equal(ageuncertain_ages, ageuncertain_ages2):  print('WARNING: COMPARING THE TWO AGE UNCERTAINTY FILES: DIFFERENT AGES')
        if not np.array_equal(ageuncertain_tsids,ageuncertain_tsids2): print('WARNING: COMPARING THE TWO AGE UNCERTAINTY FILES: DIFFERENT TSIDS')
        if not np.array_equal(ageuncertain_ages, age_centers):         print('WARNING: COMPARING AGE_CENTERS TO UNCERTAINTY AGES: DIFFERENT AGES')
    #
    # Loop through proxies, saving the necessary values to common variables.
    no_ref_data = 0; missing_uncertainty = 0
    for i in range(n_proxies):
        #
        # Get proxy data
        proxy_ages   = np.array(proxy_ts[i]['age']).astype(float)
        proxy_values = np.array(proxy_ts[i]['paleoData_values']).astype(float)
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
        # INTERPOLATION
        # To interpolate the proxy data to the base resolution (by default: decadal):
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
        plt.show()
        """
        #
        # Get uncertainty metadata
        missing_uncertainty_value = np.nan
        try:    proxy_uncertainty = proxy_ts[i]['paleoData_temperature12kUncertainty']
        except: proxy_uncertainty = missing_uncertainty_value; missing_uncertainty += 1
        proxy_uncertainty = np.square(float(proxy_uncertainty))  # Proxy uncertainty was give as RMSE, but the code uses MSE
        #
        # If selected, get the age-uncertain proxy data
        if options['age_uncertain_method']:
            #
            # Find the index of the proxy record in the age-uncertainty data.
            if proxy_ts[i]['paleoData_TSid'] not in ageuncertain_tsids: print('WARNING: TSID NOT IN AGE-UNCERTAINTY FILE, INDEX='+str(i))
            ind_tsid = np.where(ageuncertain_tsids==proxy_ts[i]['paleoData_TSid'])[0][0]
            #
            # Assign the values and uncertainty data for the given proxy, in 1D arrays of length n_time.
            proxy_values_12ka = ageuncertain_values[:,ind_tsid]
            proxy_uncertainty = ageuncertain_uncertainties[:,ind_tsid]
            #proxy_res_12ka    = #TODO: Should proxy resolution be changed when this method is used?
        #
        # Set resolutions to a minimum of 1 and a maximum of max_res_value
        proxy_res_12ka[proxy_res_12ka < 1] = 1
        proxy_res_12ka[proxy_res_12ka > max_res_value] = max_res_value
        #
        # If the reconstruction type is "relative," remove the mean of the reference period
        if options['reconstruction_type'] == 'relative':
            if options['age_uncertain_method']:
                ind_ref = np.where((age_centers >= options['reference_period'][0]) & (age_centers < options['reference_period'][1]))[0]
                proxy_values_12ka = proxy_values_12ka - np.nanmean(proxy_values_12ka[ind_ref])
                if np.isnan(proxy_values_12ka[ind_ref]).all(): print('No data in reference period, index: '+str(i)); no_ref_data += 1
            else:
                # The reference period is calculated using annualized data in this case.
                ind_ref = np.where((proxy_ages_interp >= options['reference_period'][0]) & (proxy_ages_interp < options['reference_period'][1]))[0]  
                proxy_values_12ka = proxy_values_12ka - np.nanmean(proxy_values_interp[ind_ref])
                if np.isnan(proxy_values_interp[ind_ref]).all(): print('No data in reference period, index: '+str(i)); no_ref_data += 1
        #
        # Get proxy metdata
        proxy_lat                 = proxy_ts[i]['geo_meanLat']
        proxy_lon                 = proxy_ts[i]['geo_meanLon']
        proxy_seasonality_txt     = proxy_ts[i]['paleoData_interpretation'][0]['seasonality']
        proxy_seasonality_general = proxy_ts[i]['paleoData_interpretation'][0]['seasonalityGeneral']
        proxy_data['archivetype'].append(proxy_ts[i]['archiveType'])
        proxy_data['proxytype'].append(proxy_ts[i]['paleoData_proxy'])
        proxy_data['units'].append(proxy_ts[i]['paleoData_units'])
        proxy_data['interp'].append(proxy_ts[i]['paleoData_interpretation'][0]['variable'])
        #
        # Convert seasonality to a list of months, with negative values corresponding to the previous year.
        if 'seasonality_array' in list(proxy_ts[i].keys()):
            proxy_seasonality_array = proxy_ts[i]['seasonality_array']
        else:
            proxy_seasonality = da_utils.interpret_seasonality(proxy_seasonality_txt,proxy_lat,unknown_option='annual')
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
        proxy_data['seasonality_array'][i] = proxy_seasonality_array
        #
        # Save to common variables (y and ya)
        # Adjust pseodoproxy data
        if ((proxy_ts[i]['archiveType'][:4].lower() == 'lake') & (options['proxy_datasets_to_assimilate'][0][0:7] == 'pseudo_')):
            if np.sum(np.isfinite(proxy_values_12ka))>0:
                proxy_values_12ka = da_psms.vals2percentile(proxy_values_12ka)
                proxy_values_12ka[np.where(proxy_values_12ka>=0.66)] = 3
                proxy_values_12ka[np.where((proxy_values_12ka>0.33) & (proxy_values_12ka<0.66))] = 2
                proxy_values_12ka[np.where(proxy_values_12ka<0.33)] = 1   
                proxy_values_12ka = da_psms.vals2percentile(proxy_values_12ka) 
                if options['reconstruction_type'] == 'relative':
                    proxy_values_12ka = proxy_values_12ka - np.nanmean(proxy_values_12ka[np.where((age_centers >= options['reference_period'][0]) & (age_centers < options['reference_period'][1]))[0]  ])
            else: proxy_values_12ka*=np.NaN
        proxy_data['values_binned'][i,:]     = proxy_values_12ka
        proxy_data['resolution_binned'][i,:] = proxy_res_12ka
        proxy_data['uncertainty'].append(proxy_uncertainty)
        #
        # Save some more metadata to a common variables
        if proxy_lon < 0: proxy_lon = proxy_lon+360
        proxy_data['metadata'][i,0] = proxy_ts[i]['dataSetName']
        proxy_data['metadata'][i,1] = proxy_ts[i]['paleoData_TSid']
        proxy_data['metadata'][i,2] = str(proxy_lat)
        proxy_data['metadata'][i,3] = str(proxy_lon)
        proxy_data['metadata'][i,4] = str(proxy_seasonality_array)
        proxy_data['metadata'][i,5] = proxy_seasonality_general
        proxy_data['metadata'][i,6] = str(np.median(proxy_ages[1:]-proxy_ages[:-1]))
        proxy_data['metadata'][i,7] = collection_all[i]
        proxy_data['metadata'][i,8] = proxy_ts[i]['paleoData_units']
        proxy_data['metadata'][i,9] = proxy_ts[i]['paleoData_interpretation'][0]['variable']
        proxy_data['metadata'][i,10] = proxy_ts[i]['archiveType']                                #DAMP12k- increase info about proxies saved
        proxy_data['metadata'][i,11]= proxy_ts[i]['paleoData_proxy']                            #DAMP12k- increase info about proxies saved
        proxy_data['metadata'][i,12]= da_psms.select_PSM(proxy_data,i)                          #DAMP12k- increase info about proxies saved
        proxy_data['lats'][i] = proxy_lat
        proxy_data['lons'][i] = proxy_lon
        #
    proxy_data['age_centers'] = age_centers
    proxy_data['archivetype'] = np.array(proxy_data['archivetype'])
    proxy_data['proxytype']   = np.array(proxy_data['proxytype'])
    proxy_data['units']       = np.array(proxy_data['units'])
    proxy_data['uncertainty'] = np.array(proxy_data['uncertainty'])
    #
    print('\n=== PROXY DATA LOADED ===')
    print('Proxy datasets loaded (n='+str(len(options['proxy_datasets_to_assimilate']))+'):'+str(options['proxy_datasets_to_assimilate']))
    print('Proxies loaded        (n='+str(len(proxy_ts))+')')
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