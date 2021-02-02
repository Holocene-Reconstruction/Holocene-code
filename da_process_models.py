#==============================================================================
# This script processes model data and saves new files.
#    author: Michael P. Erb
#    date  : 9/18/2020
#==============================================================================

import da_utils
import numpy as np
import xarray as xr
import glob
import copy
import netCDF4

"""
model_name         = 'cesm_lme'
time_resolution    = 1
age_range          = [101,1099]
model_dir          = '/projects/pd_lab/data/data_assimilation/models/processed_model_data/'
original_model_dir = '/projects/pd_lab/data/data_assimilation/models/original_model_data/'
"""

# A function to process model data
def process_models(model_name,time_resolution,age_range,output_dir,original_model_dir,return_variables=False):
    #
    # If the model name ends in "_regrid", remove that part of the model name.
    if model_name[-7:] == '_regrid': model_name = model_name[:-7]
    #
    # Set directories
    data_dir = {}
    data_dir['hadcm3']            = original_model_dir+'HadCM3B_transient21k/'
    data_dir['trace']             = original_model_dir+'TraCE_21ka/'
    data_dir['famous']            = original_model_dir+'FAMOUS_glacial_cycle/'
    data_dir['cesm_lme']          = original_model_dir+'CESM_LME/'
    data_dir['gfdl_cm21_control'] = original_model_dir+'GFDL_CM21_control/'
    data_dir['ipsl_control']      = original_model_dir+'IPSL_CM6A_LR_control/'
    #
    print(' === Processing model data for '+model_name+', directory: '+data_dir[model_name]+' ===')
    #
    #%% LOAD DATA AND DO MODEL-SPECIFIC CALCULATIONS
    #
    # Get the following variables
    #   - tas_model_yearsmonths [n_years,12,n_lat,n_lon]
    #   - lat_model [n_lat]
    #   - lon_model [n_lon]
    #   - age_model [n_years]
    #   - time_ndays_model_yearsmonths [n_years,12]
    #
    if model_name == 'hadcm3':
        #
        # Load model surface air temperature
        handle_model = xr.open_dataset(data_dir[model_name]+'deglh.vn1_0.temp_mm_1_5m.monthly.MON.001.nc',decode_times=False)
        tas_model = np.squeeze(handle_model['temp_mm_1_5m'].values)
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
        tas_model_yearsmonths = np.reshape(tas_model,(int(len(age_model)),12,len(lat_model),len(lon_model)))
        #
    elif model_name == 'trace':
        #
        filenames_model = sorted(glob.glob(data_dir[model_name]+'trace*TREFHT*.nc'))
        age_model = -1*np.arange(-22000,40)  # Ages are shown as "before present," where present is 1950 CE.
        #
        # Set the number of days per month in every year
        time_ndays_model = np.array([31,28,31,30,31,30,31,31,30,31,30,31])
        time_ndays_model_yearsmonths = np.repeat(time_ndays_model[None,:],len(age_model),axis=0)
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
            tas_model_section = handle_model['TREFHT'].values
            handle_model.close()
            #
            # Reshape the variable to 2D
            nyears = int(tas_model_section.shape[0]/12)
            tas_model_section_2d = np.reshape(tas_model_section,(nyears,12,len(lat_model),len(lon_model)))
            #
            if filenum == 0: tas_model_yearsmonths = copy.deepcopy(tas_model_section_2d)
            else:            tas_model_yearsmonths = np.concatenate((tas_model_yearsmonths,tas_model_section_2d),axis=0)
        #
    elif model_name == 'famous':
        #
        # Load model surface air temperature
        handle_model = xr.open_dataset(data_dir[model_name]+'ALL-5G-MON_3236.cdf',decode_times=False)
        tas_model = handle_model['air_temperature'].values[1:-1,:,:]
        lat_model = handle_model['latitude'].values
        lon_model = handle_model['longitude'].values
        age_model_monthly = -1*handle_model['time'].values[1:-1]
        handle_model.close()
        #
        age_model = np.mean(np.reshape(age_model_monthly,(int(len(age_model_monthly)/12),12)),axis=1)
        age_model = age_model*10   # The model was 10x accelleration
        age_model = age_model-1950 # Make the time relative to 1950 CE
        #TODO: Are the ages above right?  See email from 4/8/2019
        #
        # Set the number of days per month in every year
        time_ndays_model = np.array([30,30,30,30,30,30,30,30,30,30,30,30])
        time_ndays_model_yearsmonths = np.repeat(time_ndays_model[None,:],len(age_model),axis=0)
        #
        # Reshape the HadMC3 array to have months and years on different axes.
        tas_model_yearsmonths = np.reshape(tas_model,(int(len(age_model)),12,len(lat_model),len(lon_model)))
        #
    elif model_name == 'ecbilt_clio':
        #
        print('!!!',model_name,'not available !!!')
        #
    elif model_name == 'cesm_lme':
        #
        # Load model surface air temperature
        handle_model = xr.open_dataset(data_dir[model_name]+'b.e11.BLMTRC5CN.f19_g16.001.cam.h0.TREFHT.085001-184912.nc',decode_times=False)
        tas_model  = handle_model['TREFHT'].values
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
        tas_model_yearsmonths = np.reshape(tas_model,(int(len(age_model)),12,len(lat_model),len(lon_model)))
        #
    elif model_name == 'gfdl_cm21_control':
        #
        handle_model = xr.open_dataset(data_dir[model_name]+'atmos.000101-400012.t_ref.nc',decode_times=False)
        tas_model         = handle_model['t_ref'].values        # Temperature [months,lat,lon]
        lat_model         = handle_model['lat'].values          # Latitude [lat]
        lon_model         = handle_model['lon'].values          # Longitude [lon]
        time_bounds_model = handle_model['time_bounds'].values  # Days per month [months,2]
        handle_model.close()
        #years_model = np.arange(1,4001)
        age_model = np.arange(3999,-1,-1)  # This is an arbitrary age model, since this is a control simlation that doesn't correspond to specific years
        #
        # Calculate the number of days in each month
        time_ndays_model_all = time_bounds_model[:,1] - time_bounds_model[:,0]
        time_ndays_model_yearsmonths = np.reshape(time_ndays_model_all,(len(age_model),12))
        time_ndays_model = np.mean(time_ndays_model_yearsmonths,axis=0)  #TODO: Is there a better solution here?
        #
        # Reshape the HadMC3 array to have months and years on different axes.
        tas_model_yearsmonths = np.reshape(tas_model,(len(age_model),12,len(lat_model),len(lon_model)))
        #
    elif model_name == 'ipsl_control':
        #
        handle_model = xr.open_dataset(data_dir[model_name]+'tas_Amon_IPSL-CM6A-LR_piControl_r1i1p1f1_gr_concat_185001-384912.nc',decode_times=False)
        tas_model         = handle_model['tas'].values
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
        time_ndays_model = np.mean(time_ndays_model_yearsmonths,axis=0)  #TODO: Is there a better solution here?
        #
        # Reshape the HadMC3 array to have months and years on different axes.
        tas_model_yearsmonths = np.reshape(tas_model,(len(years_model),12,len(lat_model),len(lon_model)))
    #
    #
    #%% CALCULATIONS
    #
    # Print some model details
    print('tas variable shape:',tas_model_yearsmonths.shape)
    print('Model resolution:',np.mean(lat_model[:-1] - lat_model[1:]),np.mean(lon_model[1:] - lon_model[:-1]))
    print('Age bounds:',age_model[0],age_model[-1])
    j0 = np.argmin(np.abs(lat_model-0))
    i0 = np.argmin(np.abs(lon_model-0))
    print('Mean temp at '+str(lat_model[j0])+'N, '+str(lon_model[i0])+'E:',np.mean(tas_model_yearsmonths[:,:,j0,i0].flatten()))
    #
    # Check the number of days per month
    years_same = np.all(np.all(time_ndays_model_yearsmonths == time_ndays_model_yearsmonths[0,:],axis=0),axis=0)
    if years_same == False: print('Note: Months in different years have different numbers of days.')
    #
    # Convert the model output from K to C
    tas_model_yearsmonths = tas_model_yearsmonths - 273.15
    #
    # Return variables if requested
    if return_variables: return tas_model_yearsmonths,age_model,lat_model,lon_model,time_ndays_model_yearsmonths
    #
    # Make the prior years run from newest to oldest  #TODO: Why do I do this?
    #age_model             = np.flip(age_model,            axis=0)
    #tas_model_yearsmonths = np.flip(tas_model_yearsmonths,axis=0)
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
        tas_model_yearsmonths_nyearmean = tas_model_yearsmonths[age_indices_for_model_means,:,:,:]
        age_model_nyearmean             = age_model[age_indices_for_model_means]
        time_ndays_model_nyearmean      = time_ndays_model_yearsmonths[age_indices_for_model_means,:]
    else:
        n_means = int(len(age_indices_for_model_means)/effective_time_resolution)
        tas_model_yearsmonths_nyearmean = np.mean(np.reshape(tas_model_yearsmonths[age_indices_for_model_means,:,:,:],   (n_means,effective_time_resolution,12,len(lat_model),len(lon_model))),axis=1)
        age_model_nyearmean             = np.mean(np.reshape(age_model[age_indices_for_model_means],                     (n_means,effective_time_resolution)),   axis=1)
        time_ndays_model_nyearmean      = np.mean(np.reshape(time_ndays_model_yearsmonths[age_indices_for_model_means,:],(n_means,effective_time_resolution,12)),axis=1)
    #
    # Regrid the models
    tas_model_regrid,lat_model_regrid,lon_model_regrid = da_utils.regrid_model(tas_model_yearsmonths_nyearmean,lat_model,lon_model,age_model_nyearmean)
    #
    #
    #%% SAVE DATA
    #
    #ges_selected = age_model[age_indices_for_model_means]
    #age_range_txt = str(int(ages_selected[0]))+'-'+str(int(ages_selected[-1]))
    age_range_txt = str(age_range[1]-1)+'-'+str(age_range[0])
    #
    # Save model output
    outputfile = netCDF4.Dataset(output_dir+model_name+'.'+age_range_txt+'BP.TREFHT.timeres_'+str(time_resolution)+'.nc','w')
    outputfile.createDimension('age',  age_model_nyearmean.shape[0])
    outputfile.createDimension('month',12)
    outputfile.createDimension('lat',  len(lat_model))
    outputfile.createDimension('lon',  len(lon_model))
    #
    output_tas       = outputfile.createVariable('tas','f8',('age','month','lat','lon',))
    output_age       = outputfile.createVariable('age','f8',('age',))
    output_lat       = outputfile.createVariable('lat','f8',('lat',))
    output_lon       = outputfile.createVariable('lon','f8',('lon',))
    output_ndays     = outputfile.createVariable('days_per_month','f8',('month',))
    output_ndays_all = outputfile.createVariable('days_per_month_all','f8',('age','month',))
    #
    output_tas[:]       = tas_model_yearsmonths_nyearmean
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
    outputfile = netCDF4.Dataset(output_dir+model_name+'_regrid.'+age_range_txt+'BP.TREFHT.timeres_'+str(time_resolution)+'.nc','w')
    outputfile.createDimension('age',  age_model_nyearmean.shape[0])
    outputfile.createDimension('month',12)
    outputfile.createDimension('lat',  len(lat_model_regrid))
    outputfile.createDimension('lon',  len(lon_model_regrid))
    #
    output_tas       = outputfile.createVariable('tas','f8',('age','month','lat','lon',))
    output_age       = outputfile.createVariable('age','f8',('age',))
    output_lat       = outputfile.createVariable('lat','f8',('lat',))
    output_lon       = outputfile.createVariable('lon','f8',('lon',))
    output_ndays     = outputfile.createVariable('days_per_month','f8',('month',))
    output_ndays_all = outputfile.createVariable('days_per_month_all','f8',('age','month',))
    #
    output_tas[:]       = tas_model_regrid
    output_age[:]       = age_model_nyearmean
    output_lat[:]       = lat_model_regrid
    output_lon[:]       = lon_model_regrid
    output_ndays[:]     = time_ndays_model
    output_ndays_all[:] = time_ndays_model_nyearmean
    #
    outputfile.close()


# Run the code for each of the four models
#output_dir = '/projects/pd_lab/data/data_assimilation/models/processed_model_data/'
#process_models('hadcm3',           10,[0,22000],output_dir)
#process_models('trace',            10,[0,22000],output_dir)
#process_models('gfdl_cm21_control',10,[0,22000],output_dir)
#process_models('ipsl_control',     10,[0,22000],output_dir)
