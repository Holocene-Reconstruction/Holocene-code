#==============================================================================
# Functions for loading and processing model data for the data assimilation
# project.
#    author: Michael P. Erb
#    date  : 3/16/2022
#==============================================================================

import numpy as np
import xarray as xr
import glob
import da_utils
import netCDF4
from scipy import stats

# A function to load model data
def load_model_data(options):
    #
    model_dir          = options['data_dir']+'models/processed_model_data/'
    original_model_dir = options['data_dir']+'models/original_model_data/'
    age_range_model_txt = str(options['age_range_model'][1]-1)+'-'+str(options['age_range_model'][0])
    #
    # Load the model data
    n_models = len(options['models_for_prior'])
    model_data = {}
    for j,var_name in enumerate(options['vars_to_reconstruct']):
        for i,model in enumerate(options['models_for_prior']):
            #
            print('Loading variable '+var_name+' for model '+str(i+1)+'/'+str(n_models)+': '+model)
            #
            # Get the model filename
            model_filename = model+'.'+age_range_model_txt+'BP.'+var_name+'.timeres_'+str(options['time_resolution'])+'.nc'
            #
            # Check to see if the file exists.  If not, create it.
            filenames_all = glob.glob(model_dir+'*.nc')
            filenames_all = [filename.split('/')[-1] for filename in filenames_all]
            if model_filename not in filenames_all:
                print('File '+model_dir+model_filename+' does not exist.  Creating it now.')
                process_models(model,var_name,options['time_resolution'],options['age_range_model'],model_dir,original_model_dir)
                print('File '+model_dir+model_filename+' created!')
            #
            # Load selected variables
            model_individual = {}
            handle_model = xr.open_dataset(model_dir+model_filename,decode_times=False)
            model_data['lat']              = handle_model['lat'].values
            model_data['lon']              = handle_model['lon'].values
            model_individual['age']        = handle_model['age'].values
            model_individual['time_ndays'] = handle_model['days_per_month_all'].values
            model_individual[var_name]     = handle_model[var_name].values
            handle_model.close()
            #
            # Compute annual means of the model data
            n_lat = len(model_data['lat'])
            n_lon = len(model_data['lon'])
            time_ndays_model_latlon = np.repeat(np.repeat(model_individual['time_ndays'][:,:,None,None],n_lat,axis=2),n_lon,axis=3)
            model_individual[var_name+'_annual'] = np.average(model_individual[var_name],axis=1,weights=time_ndays_model_latlon)
            #
            # In each model, central values will not be selected within max_resolution/2 of the edges
            n_time = len(model_individual['age'])
            model_individual['valid_inds'] = np.full((n_time),True,dtype=bool)
            buffer = int(np.floor((options['maximum_resolution']/options['time_resolution'])/2))
            if buffer > 0:
                model_individual['valid_inds'][:buffer]  = False
                model_individual['valid_inds'][-buffer:] = False
            #
            # Set the model number for each data point
            model_individual['number'] = np.full((n_time),(i+1),dtype=int)
            #
            # Join the values together
            for key in list(model_individual.keys()):
                if i == 0: model_data[key] = model_individual[key]
                else:      model_data[key] = np.concatenate((model_data[key],model_individual[key]),axis=0)
    #
    print('\n=== MODEL DATA LOADED ===')
    print('Models loaded    (n='+str(n_models)+'):'+str(options['models_for_prior']))
    print('Variables loaded (n='+str(len(options['vars_to_reconstruct']))+'):'+str(options['vars_to_reconstruct']))
    print('---')
    print('Data stored in dictionary "model_data", with keys and dimensions:')
    for key in list(model_data.keys()): print('%20s %-20s' % (key,str(model_data[key].shape)))
    print('=========================\n')
    #
    return model_data


# A function to get the indices of the prior for selected ages
def get_indices_for_prior(options,model_data,age):
    #
    age_model        = model_data['age']
    valid_model_inds = model_data['valid_inds']
    #
    # Set age bounds for the prior
    prior_age_bound_recent = min(age_model[valid_model_inds]) - (options['time_resolution']/2)  # The prior will never select ages more recent than this
    prior_age_bound_old    = max(age_model[valid_model_inds]) + (options['time_resolution']/2)  # The prior will never select ages older than this
    #
    # Get the age intervals for the prior
    if options['prior_window'] == 'all':
        prior_age_window_recent = prior_age_bound_recent
        prior_age_window_old    = prior_age_bound_old
    else:
        prior_age_window_recent = age - (options['prior_window']/2)
        prior_age_window_old    = age + (options['prior_window']/2)
        if prior_age_window_recent < prior_age_bound_recent:
            prior_age_window_recent = prior_age_bound_recent
            prior_age_window_old    = prior_age_bound_recent + options['prior_window']
        elif prior_age_window_old > prior_age_bound_old:
            prior_age_window_old    = prior_age_bound_old
            prior_age_window_recent = prior_age_bound_old - options['prior_window']
    #
    indices_for_prior = np.where((age_model > prior_age_window_recent) & (age_model <= prior_age_window_old) & valid_model_inds)[0]
    #
    return indices_for_prior


# A function to detrend the model data
def detrend_model_data(model_data,options):
    #
    # Get dimensions
    n_lat = len(model_data['lat'])
    n_lon = len(model_data['lon'])
    #
    # If desired, do a highpass filter on every location
    if options['model_processing'] == 'linear_global':
        #
        print("Model processing: '"+options['model_processing']+"' - Removing the global mean trend from each grid point")
        for var_name in options['vars_to_reconstruct']:
            var_global = da_utils.global_mean(model_data[var_name+'_annual'],model_data['lat'],1,2)
            slope,intercept,_,_,_ = stats.linregress(model_data['age'],var_global)
            var_global_linear = (model_data['age']*slope)+intercept
            model_data[var_name+'_annual'] = model_data[var_name+'_annual'] - var_global_linear[:,None,None]
            model_data[var_name]           = model_data[var_name]           - var_global_linear[:,None,None,None]
        #
    elif options['model_processing'] == 'linear_spatial':
        #
        print("Model processing: '"+options['model_processing']+"' - Removing the local trend from each grid point")
        for var_name in options['vars_to_reconstruct']:
            for j in range(n_lat):
                for i in range(n_lon):
                    slope,intercept,_,_,_ = stats.linregress(model_data['age'],model_data[var_name+'_annual'][:,j,i])
                    var_linear = (model_data['age']*slope)+intercept
                    model_data[var_name+'_annual'][:,j,i] = model_data[var_name+'_annual'][:,j,i] - var_linear
                    model_data[var_name][:,:,j,i]         = model_data[var_name][:,:,j,i]         - var_linear[:,None]
        #
    elif options['model_processing'] in [None,'None','none']:
        print("Model processing: '"+options['model_processing']+"' - Returning data unchanged.")
        pass
    #
    return model_data


# Load the data from the TraCE simulation
def load_trace(var_txt,data_dir_model):
    #
    # Get the names of all files for the given variable
    filenames_model = sorted(glob.glob(data_dir_model+'trace*'+var_txt+'*.nc'))
    #
    # Load the model data
    handle_model = xr.open_mfdataset(filenames_model,decode_times=False,join='override')
    var_model = handle_model[var_txt].values
    lat_model = handle_model['lat'].values
    lon_model = handle_model['lon'].values
    age_model_monthly = handle_model['time'].values
    handle_model.close()
    #
    # Reshape the variable to 2D and calculate the ages
    nyears = int(var_model.shape[0]/12)
    var_model_yearsmonths = np.reshape(var_model,(nyears,12,len(lat_model),len(lon_model)))
    age_model = -1*np.floor(np.mean(np.reshape(age_model_monthly*1000,(int(len(age_model_monthly)/12),12)),axis=1))
    #
    return var_model_yearsmonths,lat_model,lon_model,age_model


# A function to process model data
def process_models(model_name,var_name,time_resolution,age_range,output_dir,original_model_dir,return_variables=False):
    #
    """
    # Variables for testing the code
    model_name         = 'trace_regrid'
    var_name           = 'tas'
    time_resolution    = 10
    age_range          = [0,12500]
    output_dir         = '/projects/pd_lab/data/data_assimilation/models/processed_model_data/'
    original_model_dir = '/projects/pd_lab/data/data_assimilation/models/original_model_data/'
    """
    #
    # If the model name ends in "_regrid", remove that part of the model name.
    if model_name[-7:] == '_regrid': model_name = model_name[:-7]
    #
    # Set directories
    data_dir = {}
    data_dir['hadcm3'] = original_model_dir+'HadCM3B_transient21k/'
    data_dir['trace']  = original_model_dir+'TraCE_21ka/'
    data_dir['famous'] = original_model_dir+'FAMOUS_glacial_cycle/'
    #
    # Set the names of the variables
    var_names = {}
    var_names['hadcm3'] = {'tas':'temp_mm_1_5m',   'precip':'precip_mm_srf'}
    var_names['trace']  = {'tas':'TREFHT',         'precip':'special'}
    var_names['famous'] = {'tas':'air_temperature','precip':'precipitation_flux'}
    #
    var_txt = var_names[model_name][var_name]
    print(' === Processing model data for '+model_name+', variable: '+var_name+', directory: '+data_dir[model_name]+' ===')
    #
    #%% LOAD DATA AND DO MODEL-SPECIFIC CALCULATIONS
    #
    # Get the following variables
    #   - var_model_yearsmonths [n_years,12,n_lat,n_lon]
    #   - lat_model [n_lat]
    #   - lon_model [n_lon]
    #   - age_model [n_years]
    #   - time_ndays_model_yearsmonths [n_years,12]
    #
    if model_name == 'hadcm3':
        #
        # Load model data
        handle_model = xr.open_dataset(data_dir[model_name]+'deglh.vn1_0.'+var_txt+'.monthly.MON.001_s.nc',decode_times=False)
        var_model = np.squeeze(handle_model[var_txt].values)
        lat_model = handle_model['latitude'].values
        lon_model = handle_model['longitude'].values
        age_model_monthly = handle_model['t'].values
        handle_model.close()
        age_model = -1*np.floor(np.mean(np.reshape(age_model_monthly,(int(len(age_model_monthly)/12),12)),axis=1))
        #
        # Set the number of days per month in every year
        time_ndays_model = np.array([30,30,30,30,30,30,30,30,30,30,30,30])
        time_ndays_model_yearsmonths = np.repeat(time_ndays_model[None,:],len(age_model),axis=0)
        #
        # Reshape the HadMC3 array to have months and years on different axes.
        var_model_yearsmonths = np.reshape(var_model,(int(len(age_model)),12,len(lat_model),len(lon_model)))
        #
        # Convert the model units to tas=C, precip=mm/day
        if   var_name == 'tas':    var_model_yearsmonths = var_model_yearsmonths - 273.15
        elif var_name == 'precip': var_model_yearsmonths = var_model_yearsmonths*60*60*24
        #
    elif model_name == 'trace':
        #
        # Load model data
        if var_name == 'precip':
            var_precc,lat_model,lon_model,age_model = load_trace('PRECC',data_dir[model_name])
            var_precl,_,_,_                         = load_trace('PRECL',data_dir[model_name])
            var_model_yearsmonths = var_precc + var_precl
        else:
            var_model_yearsmonths,lat_model,lon_model,age_model = load_trace(var_txt,data_dir[model_name])
        #
        # Set the number of days per month in every year
        time_ndays_model = np.array([31,28,31,30,31,30,31,31,30,31,30,31])
        time_ndays_model_yearsmonths = np.repeat(time_ndays_model[None,:],len(age_model),axis=0)
        #
        # Convert the model units to tas=C, precip=mm/day
        if   var_name == 'tas':    var_model_yearsmonths = var_model_yearsmonths - 273.15
        elif var_name == 'precip': var_model_yearsmonths = var_model_yearsmonths*60*60*24*1000
        #
    elif model_name == 'famous':
        #
        if   var_name == 'tas':    filename_txt = 'ALL-5G-MON_3236.cdf'
        elif var_name == 'precip': filename_txt = 'ALL-5G-MON_5216.cdf'
        #        
        # Load model surface air temperature
        handle_model = xr.open_dataset(data_dir[model_name]+filename_txt,decode_times=False)
        var_model = handle_model[var_txt].values[1:-1,:,:]
        lat_model = handle_model['latitude'].values
        lon_model = handle_model['longitude'].values
        age_model_monthly = handle_model['time'].values[1:-1]
        handle_model.close()
        #
        age_model = -1*np.floor(np.mean(np.reshape(age_model_monthly,(int(len(age_model_monthly)/12),12)),axis=1))
        age_model = age_model*10   # The model was 10x acceleration
        age_model = age_model+1950 # Make the time relative to 1950 CE
        #
        # Set the number of days per month in every year
        time_ndays_model = np.array([30,30,30,30,30,30,30,30,30,30,30,30])
        time_ndays_model_yearsmonths = np.repeat(time_ndays_model[None,:],len(age_model),axis=0)
        #
        # Reshape the FAMOUS array to have months and years on different axes.
        var_model_yearsmonths = np.reshape(var_model,(int(len(age_model)),12,len(lat_model),len(lon_model)))
        #
        # Convert the model units to tas=C, precip=mm/day
        if   var_name == 'tas':    var_model_yearsmonths = var_model_yearsmonths - 273.15
        elif var_name == 'precip': var_model_yearsmonths = var_model_yearsmonths*60*60*24
    #
    #
    #%% CALCULATIONS
    #
    # Print some model details
    print('Variable shape:',var_model_yearsmonths.shape)
    print('Model resolution:',np.mean(lat_model[:-1] - lat_model[1:]),np.mean(lon_model[1:] - lon_model[:-1]))
    print('Age bounds:',age_model[0],age_model[-1])
    j0 = np.argmin(np.abs(lat_model-0))
    i0 = np.argmin(np.abs(lon_model-0))
    print('Mean value at '+str(lat_model[j0])+'N, '+str(lon_model[i0])+'E:',np.mean(var_model_yearsmonths[:,:,j0,i0].flatten()))
    #
    # Check the number of days per month
    years_same = np.all(np.all(time_ndays_model_yearsmonths == time_ndays_model_yearsmonths[0,:],axis=0),axis=0)
    if years_same == False: print('Note: Months in different years have different numbers of days.')
    #
    # Return variables if requested
    if return_variables: return var_model_yearsmonths,age_model,lat_model,lon_model,time_ndays_model_yearsmonths
    #
    # Get indices for the selected age range
    age_indices_for_model_means = np.where((age_model >= age_range[0]) & (age_model < age_range[1]))[0]
    #
    # The famous model is already decadal (it's an accelerated-forcing) simulation, so I define a new variable to account for this.
    if model_name == 'famous': effective_time_resolution = int(time_resolution/10)
    else:                      effective_time_resolution = time_resolution
    #
    # Check to see if the time-averaging will work
    if (len(age_indices_for_model_means)%effective_time_resolution != 0): print('!!! WARNING: The selected data length is not a multiple of the time resolution. data length='+str(len(age_indices_for_model_means))+', time_resolution='+str(time_resolution))
    #
    # Average the model data into the chosen time resolution
    if effective_time_resolution == 1:
        var_model_yearsmonths_nyearmean = var_model_yearsmonths[age_indices_for_model_means,:,:,:]
        age_model_nyearmean             = age_model[age_indices_for_model_means]
        time_ndays_model_nyearmean      = time_ndays_model_yearsmonths[age_indices_for_model_means,:]
    else:
        n_means = int(len(age_indices_for_model_means)/effective_time_resolution)
        var_model_yearsmonths_nyearmean = np.mean(np.reshape(var_model_yearsmonths[age_indices_for_model_means,:,:,:],   (n_means,effective_time_resolution,12,len(lat_model),len(lon_model))),axis=1)
        age_model_nyearmean             = np.mean(np.reshape(age_model[age_indices_for_model_means],                     (n_means,effective_time_resolution)),   axis=1)
        time_ndays_model_nyearmean      = np.mean(np.reshape(time_ndays_model_yearsmonths[age_indices_for_model_means,:],(n_means,effective_time_resolution,12)),axis=1)
    #
    # Regrid the models
    var_model_regrid,lat_model_regrid,lon_model_regrid = da_utils.regrid_model(var_model_yearsmonths_nyearmean,lat_model,lon_model,age_model_nyearmean)
    #
    #
    #%% SAVE DATA
    #
    # Save model output
    age_range_txt = str(age_range[1]-1)+'-'+str(age_range[0])
    outputfile = netCDF4.Dataset(output_dir+model_name+'.'+age_range_txt+'BP.'+var_name+'.timeres_'+str(time_resolution)+'.nc','w')
    outputfile.createDimension('age',  age_model_nyearmean.shape[0])
    outputfile.createDimension('month',12)
    outputfile.createDimension('lat',  len(lat_model))
    outputfile.createDimension('lon',  len(lon_model))
    #
    output_var       = outputfile.createVariable(var_name,'f8',('age','month','lat','lon',))
    output_age       = outputfile.createVariable('age','f8',('age',))
    output_lat       = outputfile.createVariable('lat','f8',('lat',))
    output_lon       = outputfile.createVariable('lon','f8',('lon',))
    output_ndays     = outputfile.createVariable('days_per_month','f8',('month',))
    output_ndays_all = outputfile.createVariable('days_per_month_all','f8',('age','month',))
    #
    output_var[:]       = var_model_yearsmonths_nyearmean
    output_age[:]       = age_model_nyearmean
    output_lat[:]       = lat_model
    output_lon[:]       = lon_model
    output_ndays[:]     = time_ndays_model
    output_ndays_all[:] = time_ndays_model_nyearmean
    #
    outputfile.close()
    #
    #
    # Save regridded HadCM3 output
    outputfile = netCDF4.Dataset(output_dir+model_name+'_regrid.'+age_range_txt+'BP.'+var_name+'.timeres_'+str(time_resolution)+'.nc','w')
    outputfile.createDimension('age',  age_model_nyearmean.shape[0])
    outputfile.createDimension('month',12)
    outputfile.createDimension('lat',  len(lat_model_regrid))
    outputfile.createDimension('lon',  len(lon_model_regrid))
    #
    output_var       = outputfile.createVariable(var_name,'f8',('age','month','lat','lon',))
    output_age       = outputfile.createVariable('age','f8',('age',))
    output_lat       = outputfile.createVariable('lat','f8',('lat',))
    output_lon       = outputfile.createVariable('lon','f8',('lon',))
    output_ndays     = outputfile.createVariable('days_per_month','f8',('month',))
    output_ndays_all = outputfile.createVariable('days_per_month_all','f8',('age','month',))
    #
    output_var[:]       = var_model_regrid
    output_age[:]       = age_model_nyearmean
    output_lat[:]       = lat_model_regrid
    output_lon[:]       = lon_model_regrid
    output_ndays[:]     = time_ndays_model
    output_ndays_all[:] = time_ndays_model_nyearmean
    #
    outputfile.close()
