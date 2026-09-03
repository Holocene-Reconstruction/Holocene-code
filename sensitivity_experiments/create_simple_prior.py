#==============================================================================
# Load the model data, then replace the actual data for very simple patterns
# for testing.
#    author: Michael P. Erb
#==============================================================================

import numpy as np
import xarray as xr
import netCDF4
import math

#%% LOAD MODEL

model_dir = "P:/data_paleoclimate/data_assimilation/models/processed_model_data/"
model_file_ending = ".21999-0BP.tas.timeres_100.nc"

# Load selected variables
handle_model = xr.open_dataset(model_dir+"trace_regrid"+model_file_ending,decode_times=False)
tas        = handle_model['tas'].values
lat_model  = handle_model['lat'].values
lon_model  = handle_model['lon'].values
age_model  = handle_model['age'].values
days_per_month     = handle_model['days_per_month'].values
days_per_month_all = handle_model['days_per_month_all'].values
handle_model.close()


#%% CREATE SIMPLE PATTERNS

# Get dimensions
print(tas.shape)
n_time   = tas.shape[0]
n_months = tas.shape[1]
n_lat    = tas.shape[2]
n_lon    = tas.shape[3]
tas_new = np.zeros((n_time,n_months,n_lat,n_lon)); tas_new[:] = np.nan
spatial_pattern = np.zeros((n_lat,n_lon)); spatial_pattern[:] = np.nan

# Create a sample spatial pattern
counter = 50
for ind_lat in range(n_lat):
    counter -= 1
    for ind_lon in range(n_lon):
        spatial_pattern[ind_lat,ind_lon] = counter

# Create a sample 4D pattern
for ind_time in range(n_time):
    time_jitter = ind_time % 5
    for ind_month in range(n_months):
        month_curve = -1 * np.cos((ind_month / 12) * (2 * math.pi))
        #tas_new[ind_time,ind_month,:,:] = spatial_pattern + time_jitter + month_curve
        tas_new[ind_time,ind_month,:,:] = spatial_pattern + time_jitter  # For now, let's not deal with monthly variability

"""
import matplotlib.pyplot as plt
months = np.arange(0,12)
months_scaled = months / 12 * (2 * math.pi)
curve = -1*np.cos(months_scaled)
plt.plot(curve)
"""

#%% SAVE OUTPUT

# Save model output
outputfile = netCDF4.Dataset(model_dir+"sample_regrid"+model_file_ending,'w')
outputfile.createDimension('age',  len(age_model))
outputfile.createDimension('month',12)
outputfile.createDimension('lat',  len(lat_model))
outputfile.createDimension('lon',  len(lon_model))

output_var       = outputfile.createVariable('tas','f8',('age','month','lat','lon',))
output_age       = outputfile.createVariable('age','f8',('age',))
output_lat       = outputfile.createVariable('lat','f8',('lat',))
output_lon       = outputfile.createVariable('lon','f8',('lon',))
output_ndays     = outputfile.createVariable('days_per_month','f8',('month',))
output_ndays_all = outputfile.createVariable('days_per_month_all','f8',('age','month',))

output_var[:]       = tas_new
output_age[:]       = age_model
output_lat[:]       = lat_model
output_lon[:]       = lon_model
output_ndays[:]     = days_per_month
output_ndays_all[:] = days_per_month_all

outputfile.close()
