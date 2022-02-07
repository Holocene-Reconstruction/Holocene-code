#==============================================================================
# Functions for loading and processing model data for the data assimilation
# project.
#    author: Michael P. Erb
#    date  : 4/26/2021
#==============================================================================

import numpy as np
import xarray as xr
import glob
import da_utils
import copy
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
            #model_filename = model+'.'+age_range_model_txt+'BP.'+var_name+'.timeres_'+str(options['time_resolution'])+'.nc'
            model_filename = model+'.'+age_range_model_txt+'BP.'+var_name+'.timeres_'+str(options['time_resolution_adjusted'])+'.nc'
            #
            # Check to see if the file exists.  If not, create it.
            filenames_all = glob.glob(model_dir+'*.nc')
            filenames_all = [filename.split('/')[-1] for filename in filenames_all]
            if model_filename not in filenames_all:
                print('File '+model_dir+model_filename+' does not exist.  Creating it now.')
                #process_models(model,var_name,options['time_resolution'],options['age_range_model'],model_dir,original_model_dir)
                process_models(model,var_name,options['time_resolution_adjusted'],options['age_range_model'],model_dir,original_model_dir)
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
            # In each model, central values will not be selected within max_resolution/2 of the edges
            n_time = len(model_individual['age'])
            model_individual['valid_inds'] = np.full((n_time),True,dtype=bool)
            buffer = int(np.floor((options['maximum_resolution']/options['time_resolution'])/2))
            #buffer = int(np.floor((options['maximum_resolution']/options['time_resolution_adjusted'])/2))
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
    # Compute annual means of the model data
    n_lat = len(model_data['lat'])
    n_lon = len(model_data['lon'])
    time_ndays_model_latlon = np.repeat(np.repeat(model_data['time_ndays'][:,:,None,None],n_lat,axis=2),n_lon,axis=3)
    for variable in options['vars_to_reconstruct']:
        model_data[variable+'_annual'] = np.average(model_data[variable],axis=1,weights=time_ndays_model_latlon)
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
    prior_age_bound_recent = min(age_model[valid_model_inds]) - (options['time_resolution_adjusted']/2)  # The prior will never select ages more recent than this
    prior_age_bound_old    = max(age_model[valid_model_inds]) + (options['time_resolution_adjusted']/2)  # The prior will never select ages older than this
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
    indices_for_prior = np.where((age_model >= prior_age_window_recent) & (age_model < prior_age_window_old) & valid_model_inds)[0]
    #
    return indices_for_prior


# A function to detrend the model data
def detrend_model_data(model_data,options):
    #
    # Get dimensions
    n_time = len(model_data['age'])
    n_lat  = len(model_data['lat'])
    n_lon  = len(model_data['lon'])
    #
    # If desired, do a highpass filter on every location
    if options['model_processing'] == 'linear_global':
        #
        #TODO eventually: Should I handle individual months seperately?  Ask Luke what they did about this.
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
    elif options['model_processing'] == 'highpass_global':
        print("Model processing: '"+options['model_processing']+"' - Not yet available. Returning data unchanged.")
        pass
    elif options['model_processing'] == 'highpass_spatial':
        print("Model processing: '"+options['model_processing']+"' - Not yet available. Returning data unchanged.")
        pass
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
    # Get lat/lon data
    handle_model = xr.open_dataset(filenames_model[-1],decode_times=False)
    lat_model = handle_model['lat'].values
    lon_model = handle_model['lon'].values
    handle_model.close()
    #
    # Load the TraCE-21ka data
    for filenum in range(len(filenames_model)):
        print("Loading data: file "+str(filenum+1)+"/"+str(len(filenames_model)))
        handle_model = xr.open_dataset(filenames_model[filenum],decode_times=False)
        var_model_section = handle_model[var_txt].values
        handle_model.close()
        #
        # Reshape the variable to 2D
        nyears = int(var_model_section.shape[0]/12)
        var_model_section_2d = np.reshape(var_model_section,(nyears,12,len(lat_model),len(lon_model)))
        #
        if filenum == 0: var_model_yearsmonths = copy.deepcopy(var_model_section_2d)
        else:            var_model_yearsmonths = np.concatenate((var_model_yearsmonths,var_model_section_2d),axis=0)
    #
    return var_model_yearsmonths,lat_model,lon_model


# A function to process model data
def process_models(model_name,var_name,time_resolution,age_range,output_dir,original_model_dir,return_variables=False):
    #
    """
    # Variables for testing the code
    model_name         = 'speedyier_regrid'
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
    data_dir['hadcm3']            = original_model_dir+'HadCM3B_transient21k/'
    data_dir['trace']             = original_model_dir+'TraCE_21ka/'
    data_dir['famous']            = original_model_dir+'FAMOUS_glacial_cycle/'
    data_dir['speedyier']         = original_model_dir+'SPEEDYIER/'
    data_dir['cesm_lme']          = original_model_dir+'CESM_LME/'
    data_dir['gfdl_cm21_control'] = original_model_dir+'GFDL_CM21_control/'
    data_dir['ipsl_control']      = original_model_dir+'IPSL_CM6A_LR_control/'
    #
    # Set the names of the variables
    #TODO eventually: Finish implementing this below.
    var_names = {}
    var_names['hadcm3']    = {'tas':'temp_mm_1_5m',   'precip':'precip_mm_srf',     'hght_500hpa':''}
    var_names['trace']     = {'tas':'TREFHT',         'precip':'special',           'hght_500hpa':''}
    var_names['famous']    = {'tas':'air_temperature','precip':'precipitation_flux','hght_500hpa':''}
    var_names['speedyier'] = {'tas':'TAS',            'precip':'PRECIP',            'hght_500hpa':''}
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
        handle_model = xr.open_dataset(data_dir[model_name]+'deglh.vn1_0.'+var_txt+'.monthly.MON.001.nc',decode_times=False)
        var_model = np.squeeze(handle_model[var_txt].values)
        lat_model = handle_model['latitude'].values
        lon_model = handle_model['longitude'].values
        #age_model_monthly = handle_model['t'].values
        handle_model.close()
        age_model = -1*(np.arange(-22999.5,2050,1) - 0.5)
        #
        # Set the number of days per month in every year
        time_ndays_model = np.array([30,30,30,30,30,30,30,30,30,30,30,30])
        time_ndays_model_yearsmonths = np.repeat(time_ndays_model[None,:],len(age_model),axis=0)
        #
        # Reshape the HadMC3 array to have months and years on different axes.
        var_model_yearsmonths = np.reshape(var_model,(int(len(age_model)),12,len(lat_model),len(lon_model)))
        #
        # Convert the model units to tas=C, precip=mm/day
        #TODO eventually: Implement this change for all of the models, and check them everywhere
        if   var_name == 'tas':    var_model_yearsmonths = var_model_yearsmonths - 273.15
        elif var_name == 'precip': var_model_yearsmonths = var_model_yearsmonths*60*60*24
        #
    elif model_name == 'trace':
        #
        # Set the ages and number of days per month in every year
        age_model = -1*np.arange(-22000,40)  # Ages are shown as "before present," where present is 1950 CE.
        time_ndays_model = np.array([31,28,31,30,31,30,31,31,30,31,30,31])
        time_ndays_model_yearsmonths = np.repeat(time_ndays_model[None,:],len(age_model),axis=0)
        #
        # Load model data
        if var_name == 'precip':
            var_precc,lat_model, lon_model  = load_trace('PRECC',data_dir[model_name])
            var_precl,lat_model2,lon_model2 = load_trace('PRECL',data_dir[model_name])
            if (lat_model == lat_model2).all() == False: print('TraCE differs for lat_model')
            if (lon_model == lon_model2).all() == False: print('TraCE differs for lon_model')
            var_model_yearsmonths = var_precc + var_precl
        else:
            var_model_yearsmonths,lat_model,lon_model = load_trace(var_txt,data_dir[model_name])
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
        age_model_monthly = -1*handle_model['time'].values[1:-1]
        handle_model.close()
        #
        age_model = np.mean(np.reshape(age_model_monthly,(int(len(age_model_monthly)/12),12)),axis=1)
        age_model = age_model*10   # The model was 10x acceleration
        age_model = age_model-1950 # Make the time relative to 1950 CE
        #TODO eventually: Are the ages above right?  See email from 4/8/2019
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
    elif model_name == 'speedyier':
        #
        # Load model data
        handle_model = xr.open_dataset(data_dir[model_name]+'SPEEDY_r1_0850-2005.nc',decode_times=False)
        var_model = handle_model[var_txt].values
        lat_model = handle_model['lat'].values
        lon_model = handle_model['lon'].values
        handle_model.close()
        age_model = 1850 - (np.arange(850,2006,1) + 0.5)
        #
        # Trim the data, since the first 150 years don't appear to have any data. #TODO eventually: Revisit this.
        var_model = var_model[150:,:,:]
        age_model = age_model[150:]
        #
        # Set the number of days per month in every year
        time_ndays_model = np.array([1,1,1,1,1,1,1,1,1,1,1,1])
        time_ndays_model_yearsmonths = np.repeat(time_ndays_model[None,:],len(age_model),axis=0)
        #
        # Copy annual values into "months".  #TODO eventually: Fix this when I get the actual monthly data.
        var_model_yearsmonths = np.repeat(var_model[:,None,:,:],12,axis=1)
        #
        # Convert the model units to tas=C, precip=mm/day
        if   var_name == 'tas':    var_model_yearsmonths = var_model_yearsmonths - 273.15
        elif var_name == 'precip': var_model_yearsmonths = var_model_yearsmonths
        #
    elif model_name == 'ecbiltclio':
        #
        print('!!! Model '+model_name+' not available !!!')
        """
        # NOTE: This model does not have monthly data.  Probably don't use it.
        #
        if   var_name == 'tas':    print('!!! tas for model '+model_name+' does not have monthly data.  Everything is annual !!!')
        elif var_name == 'precip': print('!!! precip not available for '+model_name+' !!!')
        #
        # Load the ECBilt-Clio data
        data_url = 'http://apdrc.soest.hawaii.edu:80/dods/public_data/Paleoclimate_modeling/ECBilt-CLIO/SIM2bl/atm-annual/single_level'
        data_loveclim = netCDF4.Dataset(data_url)
        var_model_annual = np.array(data_loveclim.variables['t2m'])
        lat_model        = np.array(data_loveclim.variables['lat'])
        lon_model        = np.array(data_loveclim.variables['lon'])
        age_model        = -1*np.arange(-20999,0.1,1) # Ages span from 0 to 20999 years B.P
        #time_days_model = np.array(data_loveclim.variables['time'])  # Age in days?
        #
        # Set the number of days per month in every year
        time_ndays_model = np.array([30,30,30,30,30,30,30,30,30,30,30,30])
        time_ndays_model_yearsmonths = np.repeat(time_ndays_model[None,:],len(age_model),axis=0)
        #
        # Reshape the array to have months and years on different axes.
        var_model_yearsmonths = np.reshape(var_model,(int(len(age_model)),12,len(lat_model),len(lon_model)))
        #
        # Convert the model units to tas=C, precip=mm/day
        #if var_name == 'tas': var_model_yearsmonths = var_model_yearsmonths - 273.15
        """
        #
    elif model_name == 'cesm_lme':
        #
        # Load model surface air temperature
        handle_model = xr.open_dataset(data_dir[model_name]+'b.e11.BLMTRC5CN.f19_g16.001.cam.h0.TREFHT.085001-184912.nc',decode_times=False)
        var_model  = handle_model['TREFHT'].values
        lat_model  = handle_model['lat'].values
        lon_model  = handle_model['lon'].values
        time_model = handle_model['time'].values
        handle_model.close()
        age_model = 1950 - (np.arange(850,1850,1) + 0.5)
        #
        # Set the number of days per month in every year
        time_model = np.insert(time_model,0,0,axis=0)
        time_ndays_model_all = time_model[1:] - time_model[:-1]
        time_ndays_model_yearsmonths = np.reshape(time_ndays_model_all,(len(age_model),12))
        time_ndays_model = np.mean(time_ndays_model_yearsmonths,axis=0)
        #
        # Reshape the HadMC3 array to have months and years on different axes.
        var_model_yearsmonths = np.reshape(var_model,(int(len(age_model)),12,len(lat_model),len(lon_model)))
        #
    elif model_name == 'gfdl_cm21_control':
        #
        handle_model = xr.open_dataset(data_dir[model_name]+'atmos.000101-400012.t_ref.nc',decode_times=False)
        var_model         = handle_model['t_ref'].values        # Temperature [months,lat,lon]
        lat_model         = handle_model['lat'].values          # Latitude [lat]
        lon_model         = handle_model['lon'].values          # Longitude [lon]
        time_bounds_model = handle_model['time_bounds'].values  # Days per month [months,2]
        handle_model.close()
        age_model = np.arange(3999,-1,-1)  # This is an arbitrary age model, since this is a control simlation that doesn't correspond to specific years
        #
        # Calculate the number of days in each month
        time_ndays_model_all = time_bounds_model[:,1] - time_bounds_model[:,0]
        time_ndays_model_yearsmonths = np.reshape(time_ndays_model_all,(len(age_model),12))
        time_ndays_model = np.mean(time_ndays_model_yearsmonths,axis=0)  #TODO eventually: Is there a better solution here?
        #
        # Reshape the HadMC3 array to have months and years on different axes.
        var_model_yearsmonths = np.reshape(var_model,(len(age_model),12,len(lat_model),len(lon_model)))
        #
    elif model_name == 'ipsl_control':
        #
        handle_model = xr.open_dataset(data_dir[model_name]+'tas_Amon_IPSL-CM6A-LR_piControl_r1i1p1f1_gr_concat_185001-384912.nc',decode_times=False)
        var_model         = handle_model['tas'].values
        lat_model         = handle_model['lat'].values
        lon_model         = handle_model['lon'].values
        time_bounds_model = handle_model['time_bounds'].values
        handle_model.close()
        years_model = np.arange(1850,3850)
        age_model = np.arange(1999,-1,-1)  # This is an arbitrary age model, since this is a control simlation that doesn't correspond to specific years
        #
        # Calculate the number of days in each month
        time_ndays_model_all = time_bounds_model[:,1] - time_bounds_model[:,0]
        time_ndays_model_yearsmonths = np.reshape(time_ndays_model_all,(len(years_model),12))
        time_ndays_model = np.mean(time_ndays_model_yearsmonths,axis=0)  #TODO eventually: Is there a better solution here?
        #
        # Reshape the HadMC3 array to have months and years on different axes.
        var_model_yearsmonths = np.reshape(var_model,(len(years_model),12,len(lat_model),len(lon_model)))
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
    # Remove the modern end of the simulation, since we want to minimize the anthropogenic signal as much as possible.
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
    #ges_selected = age_model[age_indices_for_model_means]
    #age_range_txt = str(int(ages_selected[0]))+'-'+str(int(ages_selected[-1]))
    age_range_txt = str(age_range[1]-1)+'-'+str(age_range[0])
    #
    # Save model output
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

