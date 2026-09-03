#==============================================================================
# Downscaling the modeled precipitation, method 2. This is a multiplicative
# linear scaling method.
#
# Method:
#  - Load model data and ERA5 reference data.
#  - Put both datasets in the same format, with years and months on different
#    axes.
#  - Remove a reference period from the modeled output, to get anomalies.
#  - Compute 100-year means, since this is the final resolution I want, and
#    otherwise the files are too large.
#  - Use bilinear interpolation to put the model on the same resolution as the
#    reference dataset (ERA5)
#  - Multiply the modeled anomalies by the reference dataset, which has been
#    averaged over the reference period.
#  - Save the output.
#
#    author: Michael Erb
#==============================================================================

import numpy as np
import xarray as xr

# Choose the model to downscale
model_to_downscale = "trace"
#model_to_downscale = "trace2"
#model_to_downscale = "itrace"
#model_to_downscale = "hadcm3"

model_dir = "P:/data_paleoclimate/data_assimilation/models/processed_model_data/"
era5_dir  = "P:/data_reanalyses/data_ERA5/"

filename_input  = model_to_downscale+".21999--100BP.precip.timeres_1.nc"
filename_output = model_to_downscale+"_downscaled_multiplicative.21999-0BP.precip.timeres_100.nc"

print(" === Processing model data. Downscaling, debiasing, and summing to 100 year resolution ===")
#print(filename_input)


#%% LOAD DATA

# Load the model data
xarray_model = xr.open_dataset(model_dir+filename_input)
xarray_era5  = xr.open_dataset(era5_dir+'era5_monthly_tp_NorthAmerica.nc')

# Trim the region, to save memory later
if max(xarray_model['lon'].values) > 270: xarray_model['lon'] = xarray_model['lon'] - 360
xarray_model = xarray_model.sel(lat=slice(-5,90),lon=slice(-180,0))

# Compute years
xarray_model['year'] = 1950 - xarray_model['age']

# Convert units
xarray_era5['tp'] = xarray_era5['tp'] * 1000  # m/day to mm/day


#%% STEP 1: COMPUTE MEANS OVER THE YEARS OF OVERLAP

# Find the common period in the model and ERA5
years_model = xarray_model['year'].values.astype(int)
time_era5  = xarray_era5['valid_time'].values
years_era5 = time_era5.astype('datetime64[Y]').astype(int) + 1970
year_min = max([min(years_model),min(years_era5)])
year_max = min([max(years_model),max(years_era5)])
age_min = 1950 - year_min
age_max = 1950 - year_max

# Compute the model mean over the common period
model_mean_common_period = xarray_model.sel(age=slice(age_min,age_max)).mean('age')

# Compute the ERA5 mean over the common period
era5_mean_common_period = xarray_era5.sel(valid_time=slice(str(year_min)+'-01-01',str(year_max)+'-12-31')).groupby('valid_time.month').mean('valid_time')

#print(model_mean_common_period)
#print(era5_mean_common_period)


#%% STEP 2: REMOVE MODEL MEAN

# Divide by the model mean
xarray_model['precip'] = xarray_model['precip'] / model_mean_common_period['precip']


#%% STEP 3: COMPUTE 100-YEAR MEANS (TO MAKE A SMALLER FILE)

# Get some values
age_model = xarray_model['age'].values
lat_model = xarray_model['lat'].values
lon_model = xarray_model['lon'].values

# Select years to keep and compute 100-year means
ind_selected = np.where((age_model < 22000) & (age_model >= 0))[0]
precip_model_fraction_100yr = np.nanmean(np.reshape(xarray_model['precip'][ind_selected,:,:,:].values,(220,100,12,len(lat_model),len(lon_model))),axis=1)
age_100yr = np.nanmean(np.reshape(age_model[ind_selected],(220,100)),axis=1)
days_per_month_all = np.nanmean(np.reshape(xarray_model['days_per_month_all'][ind_selected,:].values,(220,100,12)),axis=1)

# Construct the variable to save
xarray_model_100year = xr.Dataset(
    {
        "precip":(["age","month","lat","lon"],precip_model_fraction_100yr),
        "days_per_month":    (["month"],      xarray_model['days_per_month'].values,{"units":"days"}),
        "days_per_month_all":(["age","month"],days_per_month_all,                   {"units":"days"}),
    },
    coords = {
        "age":   (["age"],  age_100yr,      {"units":"yr BP (ref 1950)"}),
        'month': (["month"],np.arange(1,13),{"units":"month_number"}),
        "lat":   (["lat"],  lat_model,      {"units":"degrees_north"}),
        "lon":   (["lon"],  lon_model,      {"units":"degrees_east"}),
    }
)


#%% STEP 4: DOWNSCALE AND BIAS CORRECT

# Interpolate model to ERA5
xarray_model_100year_downscaled = xarray_model_100year.interp(lat=xarray_era5["latitude"].values,lon=xarray_era5["longitude"].values)
#xarray_model_100year_downscaled.precip[0,0,:,:].plot()

# Multiply by the ERA5 mean to bias correct
xarray_model_100year_downscaled['precip'] = xarray_model_100year_downscaled['precip'] * era5_mean_common_period['tp'].values[np.newaxis,:,:,:]


#%% MAKE MAPS TO CHECK
"""
import matplotlib.pyplot as plt
# Plot time series
f, ax = plt.subplots(2,2,figsize=(16,10))
ax = ax.ravel()
model_mean_common_period.precip[0,:,:].plot(ax=ax[0])
era5_mean_common_period.tp[0,:,:].plot(ax=ax[1])
xarray_model_100year.precip[-1,0,:,:].plot(ax=ax[2])
xarray_model_100year_downscaled.precip[-1,0,:,:].plot(ax=ax[3])
"""

#%% STEP 5: SAVE THE OUTPUT

# Save the data
xarray_model_100year_downscaled.to_netcdf(model_dir+filename_output)
