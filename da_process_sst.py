#==============================================================================
# Functions for loading and processing model data.
#    author: Michael Erb
#==============================================================================

import numpy as np
import xarray as xr
import netCDF4

# SETTINGS
age_range       = [0,22000]
time_resolution = 100
model_name      = 'trace2'
var_name        = 'sst'
output_dir = 'P:/data_paleoclimate/data_assimilation/models/processed_model_data/'


#%% LOAD MODEL

# Load SSTs
data_dir = 'P:/data_paleoclimate/models/trace21k2/'
handle_model = xr.open_dataset(data_dir+'TraCE-21K-II.monthly.SST.nc',decode_times=False)
var_model         = np.squeeze(handle_model['TEMP'].values)
lat_model         = handle_model['TLAT'].values
lon_model         = handle_model['TLONG'].values
age_model_monthly = handle_model['time'].values
handle_model.close()
age_model = -1*np.floor(np.mean(np.reshape(age_model_monthly*1000,(int(len(age_model_monthly)/12),12)),axis=1))

# Set the number of days per month in every year
time_ndays_model = np.array([31,28,31,30,31,30,31,31,30,31,30,31])
time_ndays_model_yearsmonths = np.repeat(time_ndays_model[None,:],len(age_model),axis=0)

# Reshape the TraCE-21ka-II array to have months and years on different axes.
n_lat = lat_model.shape[0]
n_lon = lat_model.shape[1]
var_model_yearsmonths = np.reshape(var_model,(int(len(age_model)),12,n_lat,n_lon))


#%% CALCULATIONS

# Print some model details
print('Variable shape:',var_model_yearsmonths.shape)
print('Age bounds:',min(age_model.flatten()),max(age_model.flatten()))

# Get indices for the selected age range
age_indices_for_model_means = np.where((age_model >= age_range[0]) & (age_model < age_range[1]))[0]

# Check to see if the time-averaging will work
effective_time_resolution = time_resolution
if (len(age_indices_for_model_means)%effective_time_resolution != 0): print('!!! WARNING: The selected data length is not a multiple of the time resolution. data length='+str(len(age_indices_for_model_means))+', time_resolution='+str(time_resolution))

# Average the model data into the chosen time resolution
if effective_time_resolution == 1:
    var_model_yearsmonths_nyearmean = var_model_yearsmonths[age_indices_for_model_means,:,:,:]
    age_model_nyearmean             = age_model[age_indices_for_model_means]
    time_ndays_model_nyearmean      = time_ndays_model_yearsmonths[age_indices_for_model_means,:]
else:
    n_means = int(len(age_indices_for_model_means)/effective_time_resolution)
    var_model_yearsmonths_nyearmean = np.nanmean(np.reshape(var_model_yearsmonths[age_indices_for_model_means,:,:,:],(n_means,effective_time_resolution,12,n_lat,n_lon)),axis=1)
    age_model_nyearmean             = np.mean(np.reshape(age_model[age_indices_for_model_means],                     (n_means,effective_time_resolution)),   axis=1)
    time_ndays_model_nyearmean      = np.mean(np.reshape(time_ndays_model_yearsmonths[age_indices_for_model_means,:],(n_means,effective_time_resolution,12)),axis=1)


#%% SAVE DATA

# Save model output
age_range_txt = str(age_range[1]-1)+'-'+str(age_range[0])
outputfile = netCDF4.Dataset(output_dir+model_name+'.'+age_range_txt+'BP.'+var_name+'.timeres_'+str(time_resolution)+'.nc','w')
outputfile.createDimension('age',  age_model_nyearmean.shape[0])
outputfile.createDimension('month',12)
outputfile.createDimension('lat',  n_lat)
outputfile.createDimension('lon',  n_lon)

output_var       = outputfile.createVariable(var_name,'f8',('age','month','lat','lon',))
output_age       = outputfile.createVariable('age','f8',('age',))
output_lat       = outputfile.createVariable('lat','f8',('lat','lon',))
output_lon       = outputfile.createVariable('lon','f8',('lat','lon',))
output_ndays     = outputfile.createVariable('days_per_month','f8',('month',))
output_ndays_all = outputfile.createVariable('days_per_month_all','f8',('age','month',))

output_var[:]       = var_model_yearsmonths_nyearmean
output_age[:]       = age_model_nyearmean
output_lat[:]       = lat_model
output_lon[:]       = lon_model
output_ndays[:]     = time_ndays_model
output_ndays_all[:] = time_ndays_model_nyearmean

outputfile.close()
